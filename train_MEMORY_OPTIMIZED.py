"""
FINAL WORKING VERSION - Florence-2 & Qwen2.5-VL Training + Gradio Evaluation
Includes proper error handling, memory optimization, and evaluation interface
"""

import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from PIL import Image
import re
from typing import Dict, List
import warnings
import gc
import gradio as gr

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ============================================================================
# CONFIG
# ============================================================================

class Config:
    HF_DATASET_NAME = "yusufbukarmaina/Beakers1"
    STREAMING = True
    
    TRAIN_SAMPLES = 400
    VAL_SAMPLES = 100
    TEST_SAMPLES = 200
    
    FLORENCE_MODEL = "microsoft/Florence-2-base"
    QWEN_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
    
    MAX_IMAGE_SIZE = 512
    
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 16
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 5
    WARMUP_STEPS = 25
    
    FP16 = True
    GRADIENT_CHECKPOINTING = True
    
    OUTPUT_DIR = "./trained_models"
    SAVE_STEPS = 200
    EVAL_STEPS = 200
    LOGGING_STEPS = 25


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resize_image(image, max_size=512):
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert('RGB')
    
    w, h = image.size
    if w > max_size or h > max_size:
        if w > h:
            new_w, new_h = max_size, int(h * max_size / w)
        else:
            new_w, new_h = int(w * max_size / h), max_size
        image = image.resize((new_w, new_h), Image.LANCZOS)
    
    return image


def extract_volume(text):
    """Extract volume number from text"""
    if not text:
        return 0.0
    
    text = str(text).lower()
    
    # Try to find number before "ml"
    patterns = [
        r'(\d+\.?\d*)\s*ml',
        r'(\d+\.?\d*)\s*milliliters?',
        r'volume.*?(\d+\.?\d*)',
        r'approximately\s*(\d+\.?\d*)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except:
                pass
    
    # Fallback: extract first number
    numbers = re.findall(r'\d+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[0])
        except:
            pass
    
    return 0.0


# ============================================================================
# COLLATORS
# ============================================================================

def florence_collate_fn(features):
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [f["input_ids"] for f in features], batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [f["attention_mask"] for f in features], batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [f["labels"] for f in features], batch_first=True, padding_value=-100
    )
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def qwen_collate_fn(features):
    """Fixed Qwen collator"""
    max_len = max(f['input_ids'].shape[0] for f in features)
    
    input_ids, attention_mask, labels = [], [], []
    pixel_values, image_grid_thw = [], []
    
    for f in features:
        seq_len = f['input_ids'].shape[0]
        pad_len = max_len - seq_len
        
        input_ids.append(torch.cat([f['input_ids'], torch.zeros(pad_len, dtype=torch.long)]))
        attention_mask.append(torch.cat([f['attention_mask'], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([f['labels'], torch.full((pad_len,), -100, dtype=torch.long)]))
        
        if 'pixel_values' in f:
            pixel_values.append(f['pixel_values'])
        if 'image_grid_thw' in f:
            image_grid_thw.append(f['image_grid_thw'])
    
    result = {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels)
    }
    
    if pixel_values:
        result['pixel_values'] = torch.cat(pixel_values, dim=0)
    if image_grid_thw:
        result['image_grid_thw'] = torch.cat(image_grid_thw, dim=0)
    
    return result


# ============================================================================
# DATA PROCESSOR
# ============================================================================

class DatasetProcessor:
    def __init__(self, config):
        self.config = config
    
    def load_and_split_dataset(self):
        print("📥 Loading dataset...")
        
        dataset = load_dataset(
            self.config.HF_DATASET_NAME,
            split="train",
            streaming=self.config.STREAMING
        )
        dataset = dataset.shuffle(seed=42, buffer_size=500)
        
        train_data, val_data, test_data = [], [], []
        
        for example in dataset:
            if 'image' not in example:
                continue
            
            if len(train_data) < self.config.TRAIN_SAMPLES:
                train_data.append(example)
            elif len(val_data) < self.config.VAL_SAMPLES:
                val_data.append(example)
            elif len(test_data) < self.config.TEST_SAMPLES:
                test_data.append(example)
            else:
                break
        
        print(f"✅ Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        return train_data, val_data, test_data


# ============================================================================
# TRAINERS
# ============================================================================

class FlorenceTrainer:
    def __init__(self, config):
        self.config = config
    
    def train(self, train_data, val_data):
        print("\n🚀 Training Florence-2...")
        clear_memory()
        
        processor = AutoProcessor.from_pretrained(self.config.FLORENCE_MODEL, trust_remote_code=True)
        
        # Load in FP32 to avoid dtype issues
        model = AutoModelForCausalLM.from_pretrained(
            self.config.FLORENCE_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Use FP32 to avoid dtype mismatch
            device_map="auto"
        )
        
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=self.config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        class FlorenceDataset(torch.utils.data.Dataset):
            def __init__(self, data, processor, max_size):
                self.data = data
                self.processor = processor
                self.max_size = max_size
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                example = self.data[idx]
                image = resize_image(example['image'], self.max_size)
                
                prompt = "<VQA>What is the volume?"
                answer = example.get('volume_label', f"{example.get('volume_ml', 0)} mL")
                
                inputs = self.processor(images=image, text=prompt, return_tensors="pt", padding=True)
                answer_ids = self.processor.tokenizer(
                    str(answer), return_tensors="pt", padding=True, max_length=64, truncation=True
                )['input_ids'].squeeze(0)
                
                return {
                    'pixel_values': inputs['pixel_values'].squeeze(0),
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': answer_ids
                }
        
        train_dataset = FlorenceDataset(train_data, processor, self.config.MAX_IMAGE_SIZE)
        eval_dataset = FlorenceDataset(val_data, processor, self.config.MAX_IMAGE_SIZE)
        
        training_args = TrainingArguments(
            output_dir=f"{self.config.OUTPUT_DIR}/florence2",
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION,
            learning_rate=self.config.LEARNING_RATE,
            warmup_steps=self.config.WARMUP_STEPS,
            logging_steps=self.config.LOGGING_STEPS,
            save_steps=self.config.SAVE_STEPS,
            eval_steps=self.config.EVAL_STEPS,
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=1,
            fp16=False,  # Use FP32 to avoid dtype issues
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none"
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=florence_collate_fn
        )
        
        trainer.train()
        
        final_dir = f"{self.config.OUTPUT_DIR}/florence2_final"
        trainer.save_model(final_dir)
        processor.save_pretrained(final_dir)
        print(f"✅ Florence-2 saved")
        
        return final_dir


class QwenTrainer:
    def __init__(self, config):
        self.config = config
    
    def train(self, train_data, val_data):
        print("\n🚀 Training Qwen2.5-VL...")
        clear_memory()
        
        processor = AutoProcessor.from_pretrained(self.config.QWEN_MODEL, trust_remote_code=True)
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.QWEN_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=self.config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        class QwenDataset(torch.utils.data.Dataset):
            def __init__(self, data, processor, max_size):
                self.data = data
                self.processor = processor
                self.max_size = max_size
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                example = self.data[idx]
                image = resize_image(example['image'], self.max_size)
                
                question = "What is the volume in mL?"
                answer = example.get('volume_label', f"{example.get('volume_ml', 0)} mL")
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": question}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": str(answer)}]
                    }
                ]
                
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                
                # FIXED: Dynamic padding only
                inputs = self.processor(
                    text=[text],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                inputs['labels'] = inputs['input_ids'].clone()
                
                return inputs
        
        train_dataset = QwenDataset(train_data, processor, self.config.MAX_IMAGE_SIZE)
        eval_dataset = QwenDataset(val_data, processor, self.config.MAX_IMAGE_SIZE)
        
        training_args = TrainingArguments(
            output_dir=f"{self.config.OUTPUT_DIR}/qwen2_5vl",
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION,
            learning_rate=self.config.LEARNING_RATE,
            warmup_steps=self.config.WARMUP_STEPS,
            logging_steps=self.config.LOGGING_STEPS,
            save_steps=self.config.SAVE_STEPS,
            eval_steps=self.config.EVAL_STEPS,
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=1,
            fp16=True,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none"
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=qwen_collate_fn
        )
        
        trainer.train()
        
        final_dir = f"{self.config.OUTPUT_DIR}/qwen2_5vl_final"
        trainer.save_model(final_dir)
        processor.save_pretrained(final_dir)
        print(f"✅ Qwen2.5-VL saved")
        
        return final_dir


# ============================================================================
# EVALUATOR
# ============================================================================

def evaluate_models(test_data, config):
    """Evaluate both models and return metrics"""
    
    results = {}
    
    # Evaluate Florence-2
    print("\n📊 Evaluating Florence-2...")
    try:
        florence_dir = f"{config.OUTPUT_DIR}/florence2_final"
        processor = AutoProcessor.from_pretrained(florence_dir, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.FLORENCE_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, florence_dir)
        model.eval()
        
        predictions, ground_truth = [], []
        
        with torch.no_grad():
            for example in test_data[:50]:  # Sample for speed
                try:
                    image = resize_image(example['image'], config.MAX_IMAGE_SIZE)
                    gt = extract_volume(example.get('volume_label', f"{example.get('volume_ml', 0)} mL"))
                    ground_truth.append(gt)
                    
                    prompt = "<VQA>What is the volume?"
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
                    
                    gen_ids = model.generate(**inputs, max_new_tokens=50)
                    gen_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
                    
                    pred = extract_volume(gen_text)
                    predictions.append(pred)
                except:
                    predictions.append(0.0)
        
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        results['florence2'] = {
            'mae': float(mean_absolute_error(ground_truth, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(ground_truth, predictions))),
            'r2': float(r2_score(ground_truth, predictions))
        }
        
        del model, base_model
        clear_memory()
        
    except Exception as e:
        print(f"❌ Florence-2 evaluation error: {e}")
        results['florence2'] = {'mae': 0, 'rmse': 0, 'r2': 0}
    
    # Evaluate Qwen
    print("\n📊 Evaluating Qwen2.5-VL...")
    try:
        qwen_dir = f"{config.OUTPUT_DIR}/qwen2_5vl_final"
        processor = AutoProcessor.from_pretrained(qwen_dir, trust_remote_code=True)
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.QWEN_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, qwen_dir)
        model.eval()
        
        predictions, ground_truth = [], []
        
        with torch.no_grad():
            for example in test_data[:50]:
                try:
                    image = resize_image(example['image'], config.MAX_IMAGE_SIZE)
                    gt = extract_volume(example.get('volume_label', f"{example.get('volume_ml', 0)} mL"))
                    ground_truth.append(gt)
                    
                    question = "What is the volume in mL?"
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": question}
                        ]
                    }]
                    
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
                    
                    gen_ids = model.generate(**inputs, max_new_tokens=50)
                    gen_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
                    
                    pred = extract_volume(gen_text)
                    predictions.append(pred)
                except:
                    predictions.append(0.0)
        
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        results['qwen'] = {
            'mae': float(mean_absolute_error(ground_truth, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(ground_truth, predictions))),
            'r2': float(r2_score(ground_truth, predictions))
        }
        
        del model, base_model
        clear_memory()
        
    except Exception as e:
        print(f"❌ Qwen evaluation error: {e}")
        results['qwen'] = {'mae': 0, 'rmse': 0, 'r2': 0}
    
    return results


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_gradio_interface(config):
    """Create Gradio interface for model evaluation"""
    
    def predict_florence(image):
        try:
            florence_dir = f"{config.OUTPUT_DIR}/florence2_final"
            processor = AutoProcessor.from_pretrained(florence_dir, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                config.FLORENCE_MODEL, trust_remote_code=True, torch_dtype=torch.float32, device_map="auto"
            )
            model = PeftModel.from_pretrained(base_model, florence_dir)
            model.eval()
            
            image = resize_image(image, config.MAX_IMAGE_SIZE)
            prompt = "<VQA>What is the volume?"
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=50)
                gen_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            
            volume = extract_volume(gen_text)
            
            del model, base_model
            clear_memory()
            
            return f"Predicted Volume: {volume:.1f} mL\n\nFull Response: {gen_text}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def predict_qwen(image):
        try:
            qwen_dir = f"{config.OUTPUT_DIR}/qwen2_5vl_final"
            processor = AutoProcessor.from_pretrained(qwen_dir, trust_remote_code=True)
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                config.QWEN_MODEL, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
            )
            model = PeftModel.from_pretrained(base_model, qwen_dir)
            model.eval()
            
            image = resize_image(image, config.MAX_IMAGE_SIZE)
            question = "What is the volume in mL?"
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=50)
                gen_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            
            volume = extract_volume(gen_text)
            
            del model, base_model
            clear_memory()
            
            return f"Predicted Volume: {volume:.1f} mL\n\nFull Response: {gen_text}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Create interface
    with gr.Blocks(title="Beaker Volume Detection") as demo:
        gr.Markdown("# 🧪 Beaker Volume Detection - Model Comparison")
        
        with gr.Tabs():
            with gr.Tab("Florence-2"):
                with gr.Row():
                    with gr.Column():
                        f_input = gr.Image(type="pil", label="Upload Beaker Image")
                        f_button = gr.Button("Predict Volume", variant="primary")
                    with gr.Column():
                        f_output = gr.Textbox(label="Florence-2 Prediction", lines=5)
                
                f_button.click(predict_florence, inputs=f_input, outputs=f_output)
            
            with gr.Tab("Qwen2.5-VL"):
                with gr.Row():
                    with gr.Column():
                        q_input = gr.Image(type="pil", label="Upload Beaker Image")
                        q_button = gr.Button("Predict Volume", variant="primary")
                    with gr.Column():
                        q_output = gr.Textbox(label="Qwen2.5-VL Prediction", lines=5)
                
                q_button.click(predict_qwen, inputs=q_input, outputs=q_output)
            
            with gr.Tab("Model Metrics"):
                gr.Markdown("## Evaluation Metrics on Test Set")
                
                # Load results if available
                try:
                    with open(f"{config.OUTPUT_DIR}/evaluation_results.json", 'r') as f:
                        results = json.load(f)
                    
                    f_metrics = results.get('florence2', {})
                    q_metrics = results.get('qwen', {})
                    
                    metrics_md = f"""
### Florence-2 Metrics:
- **MAE**: {f_metrics.get('mae', 0):.2f} mL
- **RMSE**: {f_metrics.get('rmse', 0):.2f} mL
- **R²**: {f_metrics.get('r2', 0):.4f}

### Qwen2.5-VL Metrics:
- **MAE**: {q_metrics.get('mae', 0):.2f} mL
- **RMSE**: {q_metrics.get('rmse', 0):.2f} mL
- **R²**: {q_metrics.get('r2', 0):.4f}
"""
                    gr.Markdown(metrics_md)
                except:
                    gr.Markdown("No evaluation results found. Train models first.")
    
    return demo


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("🚀 Beaker Volume Training - Florence-2 & Qwen2.5-VL")
    print("="*80)
    
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Load data
    processor = DatasetProcessor(config)
    train_data, val_data, test_data = processor.load_and_split_dataset()
    
    # Train Florence-2
    f_trainer = FlorenceTrainer(config)
    f_trainer.train(train_data, val_data)
    del f_trainer
    clear_memory()
    
    # Train Qwen
    q_trainer = QwenTrainer(config)
    q_trainer.train(train_data, val_data)
    del q_trainer
    clear_memory()
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    results = evaluate_models(test_data, config)
    
    # Save results
    with open(f"{config.OUTPUT_DIR}/evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Florence-2: MAE={results['florence2']['mae']:.2f}, R²={results['florence2']['r2']:.4f}")
    print(f"Qwen2.5-VL: MAE={results['qwen']['mae']:.2f}, R²={results['qwen']['r2']:.4f}")
    
    # Launch Gradio
    print("\n" + "="*80)
    print("🌐 LAUNCHING GRADIO INTERFACE")
    print("="*80)
    demo = create_gradio_interface(config)
    demo.launch(share=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        clear_memory()

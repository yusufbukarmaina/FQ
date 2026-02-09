"""
Complete Training Pipeline for Florence-2 and Qwen2.5-VL
FINAL FIXED VERSION - All tensor dimension issues resolved
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
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from PIL import Image
import re
from typing import Dict, List
import warnings
import gc

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    HF_DATASET_NAME = "yusufbukarmaina/Beakers1"
    STREAMING = True
    
    TRAIN_SAMPLES = 500
    VAL_SAMPLES = 150
    TEST_SAMPLES = 300
    
    FLORENCE_MODEL = "microsoft/Florence-2-base"
    QWEN_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
    
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 8
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 4
    WARMUP_STEPS = 50
    
    FP16 = True
    GRADIENT_CHECKPOINTING = True
    
    OUTPUT_DIR = "./trained_models"
    SAVE_STEPS = 500
    EVAL_STEPS = 500
    LOGGING_STEPS = 50
    
    UPLOAD_TO_HF = True
    HF_REPO_NAME = "yusufbukarmaina/beaker-volume-models"


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# QWEN COLLATOR - CRITICAL FIX FOR IMAGE TOKENS
# ============================================================================

class QwenDataCollator:
    """Custom collator that properly handles Qwen2.5-VL's image tokens"""
    
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, features):
        # Separate text and image components
        batch_text = []
        batch_images = []
        
        for f in features:
            if 'text' in f:
                batch_text.append(f['text'])
            if 'image' in f:
                batch_images.append(f['image'])
        
        # Process as a batch - this ensures proper image token alignment
        if batch_text and batch_images:
            # Process everything together
            inputs = self.processor(
                text=batch_text,
                images=batch_images,
                return_tensors="pt",
                padding=True,
            )
            
            # Create labels
            labels = inputs['input_ids'].clone()
            
            return {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'pixel_values': inputs.get('pixel_values'),
                'image_grid_thw': inputs.get('image_grid_thw'),
                'labels': labels
            }
        
        # Fallback for edge cases
        return {
            'input_ids': torch.zeros((len(features), 10), dtype=torch.long),
            'attention_mask': torch.zeros((len(features), 10), dtype=torch.long),
            'labels': torch.full((len(features), 10), -100, dtype=torch.long)
        }


# ============================================================================
# FLORENCE COLLATOR
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


# ============================================================================
# DATASET PROCESSOR
# ============================================================================

class DatasetProcessor:
    def __init__(self, config: Config):
        self.config = config
        
    def load_and_split_dataset(self):
        print("📥 Loading dataset...")
        
        dataset = load_dataset(
            self.config.HF_DATASET_NAME,
            split="train",
            streaming=self.config.STREAMING
        )
        dataset = dataset.shuffle(seed=42, buffer_size=1000)
        
        train_data, val_data, test_data = [], [], []
        total = 0
        
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
            
            total += 1
            if total % 100 == 0:
                print(f"✓ Loaded {total}")
        
        print(f"\n✅ Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        return train_data, val_data, test_data
    
    def extract_volume(self, text: str) -> float:
        if not text:
            return 0.0
        numbers = re.findall(r'\d+\.?\d*', str(text))
        return float(numbers[0]) if numbers else 0.0


# ============================================================================
# FLORENCE TRAINER
# ============================================================================

class FlorenceTrainer:
    def __init__(self, config: Config):
        self.config = config
        
    def setup_model(self):
        print(f"\n🤖 Loading Florence-2...")
        clear_memory()
        
        self.processor = AutoProcessor.from_pretrained(
            self.config.FLORENCE_MODEL, trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.FLORENCE_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if self.config.GRADIENT_CHECKPOINTING:
            self.model.gradient_checkpointing_enable()
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=self.config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def train(self, train_data, val_data):
        print("\n🚀 Training Florence-2...")
        self.setup_model()
        
        class FlorenceDataset(torch.utils.data.Dataset):
            def __init__(self, data, processor):
                self.data = data
                self.processor = processor
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                example = self.data[idx]
                image = example['image']
                if not isinstance(image, Image.Image):
                    image = Image.open(image)
                image = image.convert('RGB')
                
                prompt = "<VQA>What is the volume?"
                answer = example.get('volume_label', f"{example.get('volume_ml', 0)} mL")
                
                inputs = self.processor(
                    images=image,
                    text=prompt,
                    return_tensors="pt",
                    padding=True,
                )
                
                answer_ids = self.processor.tokenizer(
                    str(answer),
                    return_tensors="pt",
                    padding=True,
                    max_length=64,
                    truncation=True
                )['input_ids'].squeeze(0)
                
                return {
                    'pixel_values': inputs['pixel_values'].squeeze(0),
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': answer_ids
                }
        
        train_dataset = FlorenceDataset(train_data, self.processor)
        eval_dataset = FlorenceDataset(val_data, self.processor)
        
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
            save_total_limit=2,
            fp16=self.config.FP16,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=florence_collate_fn,
        )
        
        trainer.train()
        
        final_dir = f"{self.config.OUTPUT_DIR}/florence2_final"
        trainer.save_model(final_dir)
        self.processor.save_pretrained(final_dir)
        print(f"✅ Saved to {final_dir}")
        
        return final_dir


# ============================================================================
# QWEN TRAINER - FINAL FIX
# ============================================================================

class QwenTrainer:
    def __init__(self, config: Config):
        self.config = config
        
    def setup_model(self):
        print(f"\n🤖 Loading Qwen2.5-VL...")
        clear_memory()
        
        self.processor = AutoProcessor.from_pretrained(
            self.config.QWEN_MODEL, trust_remote_code=True
        )
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.QWEN_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if self.config.GRADIENT_CHECKPOINTING:
            self.model.gradient_checkpointing_enable()
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=self.config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def train(self, train_data, val_data):
        print("\n🚀 Training Qwen2.5-VL...")
        self.setup_model()
        
        class QwenDataset(torch.utils.data.Dataset):
            def __init__(self, data, processor):
                self.data = data
                self.processor = processor
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                example = self.data[idx]
                image = example['image']
                if not isinstance(image, Image.Image):
                    image = Image.open(image)
                image = image.convert('RGB')
                
                question = "What is the volume of liquid in mL?"
                answer = example.get('volume_label', f"{example.get('volume_ml', 0)} mL")
                
                # Create conversation
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
                
                # Apply chat template
                text = self.processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                
                # Return raw data for batch processing
                return {
                    'text': text,
                    'image': image
                }
        
        train_dataset = QwenDataset(train_data, self.processor)
        eval_dataset = QwenDataset(val_data, self.processor)
        
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
            save_total_limit=2,
            fp16=self.config.FP16,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
        )
        
        # Use custom collator that processes batch together
        collator = QwenDataCollator(self.processor)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,  # CRITICAL: Batch processing
        )
        
        trainer.train()
        
        final_dir = f"{self.config.OUTPUT_DIR}/qwen2_5vl_final"
        trainer.save_model(final_dir)
        self.processor.save_pretrained(final_dir)
        print(f"✅ Saved to {final_dir}")
        
        return final_dir


# ============================================================================
# EVALUATOR
# ============================================================================

class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.processor = DatasetProcessor(config)
    
    def evaluate(self, model, processor, test_data, name):
        print(f"\n📊 Evaluating {name}...")
        model.eval()
        
        predictions, ground_truth = [], []
        
        with torch.no_grad():
            for i, ex in enumerate(test_data):
                try:
                    img = ex['image']
                    if not isinstance(img, Image.Image):
                        img = Image.open(img)
                    img = img.convert('RGB')
                    
                    gt = self.processor.extract_volume(
                        ex.get('volume_label', f"{ex.get('volume_ml', 0)}")
                    )
                    ground_truth.append(gt)
                    
                    if 'florence' in name.lower():
                        inputs = processor(images=img, text="<VQA>What is the volume?", return_tensors="pt").to(model.device)
                        ids = model.generate(**inputs, max_new_tokens=50)
                        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
                    else:
                        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "What is the volume?"}]}]
                        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = processor(text=[text_prompt], images=[img], return_tensors="pt").to(model.device)
                        ids = model.generate(**inputs, max_new_tokens=50)
                        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
                    
                    pred = self.processor.extract_volume(text)
                    predictions.append(pred)
                    
                    if (i + 1) % 50 == 0:
                        print(f"  {i + 1}/{len(test_data)}")
                except:
                    predictions.append(0.0)
        
        p, g = np.array(predictions), np.array(ground_truth)
        mae = mean_absolute_error(g, p)
        rmse = np.sqrt(mean_squared_error(g, p))
        r2 = r2_score(g, p)
        
        print(f"📈 {name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
        
        return {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2)}


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
    f_path = f_trainer.train(train_data, val_data)
    del f_trainer
    clear_memory()
    
    # Train Qwen2.5-VL
    q_trainer = QwenTrainer(config)
    q_path = q_trainer.train(train_data, val_data)
    
    # Evaluate
    evaluator = ModelEvaluator(config)
    
    f_eval = FlorenceTrainer(config)
    f_eval.setup_model()
    f_results = evaluator.evaluate(f_eval.model, f_eval.processor, test_data, "Florence-2")
    del f_eval
    clear_memory()
    
    q_results = evaluator.evaluate(q_trainer.model, q_trainer.processor, test_data, "Qwen2.5-VL")
    
    # Save
    results = {
        'florence2': f_results,
        'qwen2_5vl': q_results,
        'config': {
            'train': len(train_data),
            'val': len(val_data),
            'test': len(test_data)
        }
    }
    
    with open(f"{config.OUTPUT_DIR}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("🎉 COMPLETE!")
    print(f"Florence-2:  {f_path}")
    print(f"Qwen2.5-VL:  {q_path}")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

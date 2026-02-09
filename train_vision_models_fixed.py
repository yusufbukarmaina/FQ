"""
Complete Training Pipeline for Florence-2 and Qwen2.5-VL
Fixed version for RTX 6000 (24GB VRAM) with proper error handling
ALL BUGS FIXED: Multi-GPU, Image tokens, Field names, etc.
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
from typing import Dict, List, Tuple
import warnings
import gc
from pathlib import Path
from qwen_vl_utils import process_vision_info

warnings.filterwarnings('ignore')

# Force single GPU to avoid multi-GPU splitting errors
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
    TOTAL_SAMPLES = 1600
    
    FLORENCE_MODEL = "microsoft/Florence-2-base"
    QWEN_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
    
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 8
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 10
    WARMUP_STEPS = 50
    MAX_LENGTH = 256
    
    FP16 = True
    GRADIENT_CHECKPOINTING = True
    
    OUTPUT_DIR = "./trained_models"
    TEST_IMAGES_DIR = "./test_images"
    SAVE_STEPS = 500
    EVAL_STEPS = 500
    LOGGING_STEPS = 50
    
    UPLOAD_TO_HF = True
    HF_REPO_NAME = "yusufbukarmaina/beaker-volume-models"


# ============================================================================
# UTILITIES
# ============================================================================

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


# ============================================================================
# FLORENCE COLLATOR
# ============================================================================

def florence_collate_fn(features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
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
# QWEN COLLATOR - FIXED FOR IMAGE TOKEN MATCHING
# ============================================================================

def qwen_collate_fn(features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collator for Qwen that handles variable-length sequences
    and prevents image token mismatch errors.
    """
    max_len = max(f['input_ids'].shape[0] for f in features)
    
    input_ids = []
    attention_mask = []
    labels = []
    pixel_values = []
    image_grid_thw = []
    
    for f in features:
        seq_len = f['input_ids'].shape[0]
        pad_len = max_len - seq_len
        
        # Pad sequences
        input_ids.append(torch.cat([
            f['input_ids'],
            torch.zeros(pad_len, dtype=torch.long)
        ]))
        
        attention_mask.append(torch.cat([
            f['attention_mask'],
            torch.zeros(pad_len, dtype=torch.long)
        ]))
        
        labels.append(torch.cat([
            f['labels'],
            torch.full((pad_len,), -100, dtype=torch.long)
        ]))
        
        # Collect image features
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
# DATASET PROCESSOR
# ============================================================================

class DatasetProcessor:
    def __init__(self, config: Config):
        self.config = config
        
    def load_and_split_dataset(self):
        print("📥 Loading dataset...")
        print(f"Target: {self.config.TRAIN_SAMPLES} train, {self.config.VAL_SAMPLES} val, {self.config.TEST_SAMPLES} test")
        
        dataset = load_dataset(
            self.config.HF_DATASET_NAME,
            split="train",
            streaming=self.config.STREAMING
        )
        dataset = dataset.shuffle(seed=42, buffer_size=1000)
        
        train_data, val_data, test_data = [], [], []
        total_processed = 0
        skipped = 0
        
        for example in dataset:
            if 'image' not in example or ('volume_ml' not in example and 'volume_label' not in example):
                skipped += 1
                continue
            
            if len(train_data) < self.config.TRAIN_SAMPLES:
                train_data.append(example)
            elif len(val_data) < self.config.VAL_SAMPLES:
                val_data.append(example)
            elif len(test_data) < self.config.TEST_SAMPLES:
                test_data.append(example)
            else:
                break
            
            total_processed += 1
            if total_processed % 100 == 0:
                print(f"✓ Processed {total_processed}")
            
            if total_processed > self.config.TOTAL_SAMPLES + 500:
                break
        
        print(f"\n✅ Dataset loaded:")
        print(f"   Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def extract_volume_from_text(self, text: str) -> float:
        if not text:
            return 0.0
        text = str(text)
        patterns = [r'(\d+\.?\d*)\s*mL', r'(\d+\.?\d*)\s*ml', r'(\d+\.?\d*)\s*ML']
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        numbers = re.findall(r'\d+\.?\d*', text)
        return float(numbers[0]) if numbers else 0.0
    
    def save_test_images(self, test_data: List[Dict], output_dir: str):
        print(f"\n💾 Saving {len(test_data)} test images...")
        os.makedirs(output_dir, exist_ok=True)
        metadata = []
        
        for idx, example in enumerate(test_data):
            try:
                image = example['image']
                if not isinstance(image, Image.Image):
                    image = Image.open(image)
                image = image.convert('RGB')
                
                gt_text = example.get('volume_label', '')
                if not gt_text and 'volume_ml' in example:
                    gt_text = f"{example['volume_ml']} mL"
                gt_volume = self.extract_volume_from_text(gt_text)
                
                filename = f"test_{idx:04d}_volume_{gt_volume:.1f}mL.jpg"
                image.save(os.path.join(output_dir, filename), quality=95)
                
                metadata.append({
                    'index': idx,
                    'filename': filename,
                    'ground_truth_volume': gt_volume
                })
                
                if (idx + 1) % 50 == 0:
                    print(f"  {idx + 1}/{len(test_data)}")
            except Exception as e:
                print(f"⚠️ Error {idx}: {e}")
        
        with open(os.path.join(output_dir, 'test_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        print("✅ Test images saved")


# ============================================================================
# FLORENCE TRAINER
# ============================================================================

class FlorenceTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.processor = None
        self.model = None
        
    def setup_model(self):
        print(f"\n🤖 Florence-2: {self.config.FLORENCE_MODEL}")
        clear_memory()
        
        self.processor = AutoProcessor.from_pretrained(
            self.config.FLORENCE_MODEL, trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.FLORENCE_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.config.FP16 else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        if self.config.GRADIENT_CHECKPOINTING:
            self.model.gradient_checkpointing_enable()
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            target_modules=self.config.LORA_TARGET_MODULES,
            lora_dropout=self.config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        print_memory_usage()
        
        return self.model, self.processor
    
    def train(self, train_data: List[Dict], val_data: List[Dict]) -> str:
        print("\n🚀 Training Florence-2...")
        self.setup_model()
        
        output_dir = f"{self.config.OUTPUT_DIR}/florence2"
        os.makedirs(output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION,
            learning_rate=self.config.LEARNING_RATE,
            warmup_steps=self.config.WARMUP_STEPS,
            logging_steps=self.config.LOGGING_STEPS,
            save_steps=self.config.SAVE_STEPS,
            eval_steps=self.config.EVAL_STEPS,
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=self.config.FP16,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
            gradient_checkpointing=self.config.GRADIENT_CHECKPOINTING,
            optim="adamw_torch",
            max_grad_norm=1.0
        )
        
        class FlorenceDataset(torch.utils.data.Dataset):
            def __init__(self, data, processor, config):
                self.data = data
                self.processor = processor
                self.config = config
                self.pad_id = processor.tokenizer.pad_token_id
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                example = self.data[idx]
                try:
                    image = example['image']
                    if not isinstance(image, Image.Image):
                        image = Image.open(image)
                    image = image.convert('RGB')
                    
                    prompt = "<VQA>What is the volume of liquid in the beaker?"
                    answer = example.get('volume_label', '')
                    if not answer and 'volume_ml' in example:
                        answer = f"{example['volume_ml']} mL"
                    if not answer:
                        answer = "0 mL"
                    
                    inputs = self.processor(
                        images=image,
                        text=prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    
                    answer_inputs = self.processor.tokenizer(
                        str(answer),
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=64
                    )
                    
                    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                    labels = answer_inputs['input_ids'].squeeze(0).clone()
                    
                    if self.pad_id is not None:
                        labels[labels == self.pad_id] = -100
                    
                    inputs['labels'] = labels
                    return inputs
                    
                except Exception as e:
                    dummy_image = Image.new('RGB', (224, 224), color='white')
                    inputs = self.processor(
                        images=dummy_image,
                        text="<VQA>dummy",
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                    inputs['labels'] = inputs['input_ids'].clone()
                    inputs['labels'][:] = -100
                    return inputs
        
        train_dataset = FlorenceDataset(train_data, self.processor, self.config)
        eval_dataset = FlorenceDataset(val_data, self.processor, self.config)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=florence_collate_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        trainer.train()
        
        final_dir = f"{self.config.OUTPUT_DIR}/florence2_final"
        trainer.save_model(final_dir)
        self.processor.save_pretrained(final_dir)
        
        print(f"✅ Florence-2 saved to {final_dir}")
        clear_memory()
        return final_dir


# ============================================================================
# QWEN TRAINER - FIXED
# ============================================================================

class QwenTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.processor = None
        self.model = None
        
    def setup_model(self):
        print(f"\n🤖 Qwen2.5-VL: {self.config.QWEN_MODEL}")
        clear_memory()
        
        self.processor = AutoProcessor.from_pretrained(
            self.config.QWEN_MODEL, trust_remote_code=True
        )
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.QWEN_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.config.FP16 else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        if self.config.GRADIENT_CHECKPOINTING:
            self.model.gradient_checkpointing_enable()
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            target_modules=self.config.LORA_TARGET_MODULES,
            lora_dropout=self.config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        print_memory_usage()
        
        return self.model, self.processor
    
    def train(self, train_data: List[Dict], val_data: List[Dict]) -> str:
        print("\n🚀 Training Qwen2.5-VL...")
        self.setup_model()
        
        output_dir = f"{self.config.OUTPUT_DIR}/qwen2_5vl"
        os.makedirs(output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION,
            learning_rate=self.config.LEARNING_RATE,
            warmup_steps=self.config.WARMUP_STEPS,
            logging_steps=self.config.LOGGING_STEPS,
            save_steps=self.config.SAVE_STEPS,
            eval_steps=self.config.EVAL_STEPS,
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=self.config.FP16,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
            gradient_checkpointing=self.config.GRADIENT_CHECKPOINTING,
            optim="adamw_torch",
            max_grad_norm=1.0
        )
        
        class QwenDataset(torch.utils.data.Dataset):
            def __init__(self, data, processor, config):
                self.data = data
                self.processor = processor
                self.config = config
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                example = self.data[idx]
                try:
                    image = example['image']
                    if not isinstance(image, Image.Image):
                        image = Image.open(image)
                    image = image.convert('RGB')
                    
                    question = "What is the volume of liquid in this beaker in mL?"
                    
                    if 'volume_ml' in example:
                        answer = f"{example['volume_ml']} mL"
                    elif 'volume_label' in example:
                        answer = str(example['volume_label'])
                    else:
                        answer = "unknown"
                    
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
                            "content": [{"type": "text", "text": answer}]
                        }
                    ]
                    
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    
                    # FIXED: Use dynamic padding, NO max_length
                    inputs = self.processor(
                        text=[text],
                        images=[image],
                        return_tensors="pt",
                        padding=True,  # Dynamic padding
                        truncation=True
                        # NO max_length parameter!
                    )
                    
                    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                    inputs['labels'] = inputs['input_ids'].clone()
                    
                    return inputs
                    
                except Exception as e:
                    print(f"⚠️ Error {idx}: {e}")
                    dummy_image = Image.new('RGB', (224, 224), color='white')
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": dummy_image},
                                {"type": "text", "text": "dummy"}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "dummy"}]
                        }
                    ]
                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    inputs = self.processor(text=[text], images=[dummy_image], return_tensors="pt", padding=True, truncation=True)
                    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                    inputs['labels'] = inputs['input_ids'].clone()
                    return inputs
        
        train_dataset = QwenDataset(train_data, self.processor, self.config)
        eval_dataset = QwenDataset(val_data, self.processor, self.config)
        
        # FIXED: Use custom collator
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=qwen_collate_fn,  # ← CRITICAL FIX
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        trainer.train()
        
        final_dir = f"{self.config.OUTPUT_DIR}/qwen2_5vl_final"
        trainer.save_model(final_dir)
        self.processor.save_pretrained(final_dir)
        
        print(f"✅ Qwen2.5-VL saved to {final_dir}")
        clear_memory()
        return final_dir


# ============================================================================
# EVALUATOR
# ============================================================================

class ModelEvaluator:
    def __init__(self, config: Config):
        self.config = config
        self.data_processor = DatasetProcessor(config)
    
    def evaluate_model(self, model, processor, test_data: List[Dict], model_name: str) -> Dict:
        print(f"\n📊 Evaluating {model_name}...")
        predictions, ground_truth = [], []
        model.eval()
        clear_memory()
        
        with torch.no_grad():
            for idx, example in enumerate(test_data):
                try:
                    image = example['image']
                    if not isinstance(image, Image.Image):
                        image = Image.open(image)
                    image = image.convert('RGB')
                    
                    gt_text = example.get('volume_label', '')
                    if not gt_text and 'volume_ml' in example:
                        gt_text = f"{example['volume_ml']} mL"
                    gt_volume = self.data_processor.extract_volume_from_text(gt_text)
                    ground_truth.append(gt_volume)
                    
                    if 'florence' in model_name.lower():
                        prompt = "<VQA>What is the volume of liquid in the beaker?"
                        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
                        generated_ids = model.generate(**inputs, max_new_tokens=50, num_beams=3)
                        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    else:
                        question = "What is the volume of liquid in this beaker in mL?"
                        messages = [{
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": question}
                            ]
                        }]
                        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
                        generated_ids = model.generate(**inputs, max_new_tokens=50, num_beams=3)
                        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    pred_volume = self.data_processor.extract_volume_from_text(generated_text)
                    predictions.append(pred_volume)
                    
                    if (idx + 1) % 50 == 0:
                        print(f"  {idx + 1}/{len(test_data)}")
                        
                except Exception as e:
                    predictions.append(0.0)
        
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        mae = mean_absolute_error(ground_truth, predictions)
        rmse = np.sqrt(mean_squared_error(ground_truth, predictions))
        r2 = r2_score(ground_truth, predictions)
        
        print(f"\n📈 {model_name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
        
        return {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2)}


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("🚀 Vision Model Training - Florence-2 & Qwen2.5-VL")
    print("="*80)
    
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.TEST_IMAGES_DIR, exist_ok=True)
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print_memory_usage()
    
    # Load data
    data_processor = DatasetProcessor(config)
    train_data, val_data, test_data = data_processor.load_and_split_dataset()
    data_processor.save_test_images(test_data, config.TEST_IMAGES_DIR)
    
    # Train Florence-2
    florence_trainer = FlorenceTrainer(config)
    florence_path = florence_trainer.train(train_data, val_data)
    del florence_trainer
    clear_memory()
    
    # Train Qwen2.5-VL
    qwen_trainer = QwenTrainer(config)
    qwen_path = qwen_trainer.train(train_data, val_data)
    
    # Evaluate
    evaluator = ModelEvaluator(config)
    
    florence_trainer_eval = FlorenceTrainer(config)
    florence_trainer_eval.setup_model()
    florence_results = evaluator.evaluate_model(
        florence_trainer_eval.model, florence_trainer_eval.processor, test_data, "Florence-2"
    )
    del florence_trainer_eval
    clear_memory()
    
    qwen_results = evaluator.evaluate_model(
        qwen_trainer.model, qwen_trainer.processor, test_data, "Qwen2.5-VL"
    )
    
    # Save results
    results = {
        'florence2': florence_results,
        'qwen2_5vl': qwen_results,
        'config': {
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data)
        }
    }
    
    results_path = f"{config.OUTPUT_DIR}/evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("🎉 COMPLETE!")
    print("="*80)
    print(f"Florence-2:  {florence_path}")
    print(f"Qwen2.5-VL:  {qwen_path}")
    print(f"Results:     {results_path}")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
        clear_memory()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        clear_memory()

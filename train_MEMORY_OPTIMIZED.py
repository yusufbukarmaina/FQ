"""
Memory-Optimized Training Pipeline for Florence-2 and Qwen2.5-VL
Handles large images (3468x4624) by resizing to manageable dimensions
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
from PIL import Image
import re
from typing import Dict, List
import warnings
import gc

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    HF_DATASET_NAME = "yusufbukarmaina/Beakers1"
    STREAMING = True
    
    # Reduced samples for memory
    TRAIN_SAMPLES = 400
    VAL_SAMPLES = 100
    TEST_SAMPLES = 200
    
    FLORENCE_MODEL = "microsoft/Florence-2-base"
    QWEN_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
    
    # Image resizing to reduce memory
    MAX_IMAGE_SIZE = 512  # Resize large images to this
    
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    
    # Memory-optimized training settings
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 16  # Increased to maintain effective batch size
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 5  # Reduced for faster training
    WARMUP_STEPS = 25
    
    FP16 = True
    GRADIENT_CHECKPOINTING = True
    
    OUTPUT_DIR = "./trained_models"
    SAVE_STEPS = 200
    EVAL_STEPS = 200
    LOGGING_STEPS = 25
    
    UPLOAD_TO_HF = False
    HF_REPO_NAME = "yusufbukarmaina/beaker-volume-models"


def clear_memory():
    """Aggressively clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def resize_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    """Resize large images while maintaining aspect ratio"""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    
    image = image.convert('RGB')
    
    # Get current size
    width, height = image.size
    
    # Only resize if larger than max_size
    if width > max_size or height > max_size:
        # Calculate new size maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(f"  Resized from {width}x{height} to {new_width}x{new_height}")
    
    return image


# ============================================================================
# COLLATORS
# ============================================================================

def florence_collate_fn(features):
    """Florence-2 collator"""
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


class QwenDataCollator:
    """Memory-optimized Qwen collator with image resizing"""
    
    def __init__(self, processor, max_image_size=512):
        self.processor = processor
        self.max_image_size = max_image_size
    
    def __call__(self, features):
        batch_text = []
        batch_images = []
        
        for f in features:
            if 'text' in f and 'image' in f:
                batch_text.append(f['text'])
                # Resize image before processing
                resized_img = resize_image(f['image'], self.max_image_size)
                batch_images.append(resized_img)
        
        if batch_text and batch_images:
            try:
                inputs = self.processor(
                    text=batch_text,
                    images=batch_images,
                    return_tensors="pt",
                    padding=True,
                )
                
                labels = inputs['input_ids'].clone()
                
                return {
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask'],
                    'pixel_values': inputs.get('pixel_values'),
                    'image_grid_thw': inputs.get('image_grid_thw'),
                    'labels': labels
                }
            except Exception as e:
                print(f"⚠️ Collator error: {e}")
        
        # Fallback
        return {
            'input_ids': torch.zeros((len(features), 10), dtype=torch.long),
            'attention_mask': torch.zeros((len(features), 10), dtype=torch.long),
            'labels': torch.full((len(features), 10), -100, dtype=torch.long)
        }


# ============================================================================
# DATASET PROCESSOR
# ============================================================================

class DatasetProcessor:
    def __init__(self, config: Config):
        self.config = config
        
    def load_and_split_dataset(self):
        print("📥 Loading dataset (with image size checking)...")
        
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
            
            if (len(train_data) + len(val_data) + len(test_data)) % 50 == 0:
                print(f"✓ Loaded {len(train_data) + len(val_data) + len(test_data)}")
        
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
            device_map="auto",
            low_cpu_mem_usage=True
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
            def __init__(self, data, processor, max_size):
                self.data = data
                self.processor = processor
                self.max_size = max_size
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                example = self.data[idx]
                
                # Resize image
                image = resize_image(example['image'], self.max_size)
                
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
        
        train_dataset = FlorenceDataset(train_data, self.processor, self.config.MAX_IMAGE_SIZE)
        eval_dataset = FlorenceDataset(val_data, self.processor, self.config.MAX_IMAGE_SIZE)
        
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
            gradient_checkpointing=self.config.GRADIENT_CHECKPOINTING,
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
        print(f"✅ Florence-2 saved to {final_dir}")
        
        return final_dir


# ============================================================================
# QWEN TRAINER - MEMORY OPTIMIZED
# ============================================================================

class QwenTrainer:
    def __init__(self, config: Config):
        self.config = config
        
    def setup_model(self):
        print(f"\n🤖 Loading Qwen2.5-VL (memory-optimized)...")
        clear_memory()
        
        self.processor = AutoProcessor.from_pretrained(
            self.config.QWEN_MODEL, trust_remote_code=True
        )
        
        # Load with memory optimization
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.QWEN_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={0: "20GB"}  # Limit memory usage
        )
        
        if self.config.GRADIENT_CHECKPOINTING:
            self.model.gradient_checkpointing_enable()
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            target_modules=["q_proj", "v_proj"],  # Minimal targets
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
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                example = self.data[idx]
                
                # Don't resize here - done in collator
                image = example['image']
                if not isinstance(image, Image.Image):
                    image = Image.open(image)
                image = image.convert('RGB')
                
                question = "What is the volume in mL?"
                answer = example.get('volume_label', f"{example.get('volume_ml', 0)} mL")
                
                # Format conversation
                from transformers import Qwen2VLProcessor
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
                
                # Create text template
                text = self.processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                
                return {'text': text, 'image': image}
        
        # Attach processor to dataset class
        QwenDataset.processor = self.processor
        
        train_dataset = QwenDataset(train_data)
        eval_dataset = QwenDataset(val_data)
        
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
            gradient_checkpointing=self.config.GRADIENT_CHECKPOINTING,
            max_grad_norm=0.5,  # Prevent gradient explosion
        )
        
        # Use memory-optimized collator
        collator = QwenDataCollator(self.processor, self.config.MAX_IMAGE_SIZE)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
        )
        
        print("🔥 Starting training (images will be resized to save memory)...")
        trainer.train()
        
        final_dir = f"{self.config.OUTPUT_DIR}/qwen2_5vl_final"
        trainer.save_model(final_dir)
        self.processor.save_pretrained(final_dir)
        print(f"✅ Qwen2.5-VL saved to {final_dir}")
        
        return final_dir


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("🚀 Memory-Optimized Beaker Training - Florence-2 & Qwen2.5-VL")
    print("="*80)
    print("⚙️  Memory optimizations:")
    print("   - Images resized to 512px max")
    print("   - Reduced batch processing")
    print("   - Aggressive memory clearing")
    print("="*80)
    
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Load data
    processor = DatasetProcessor(config)
    train_data, val_data, test_data = processor.load_and_split_dataset()
    
    # Train Florence-2
    print("\n" + "="*80)
    print("FLORENCE-2 TRAINING")
    print("="*80)
    f_trainer = FlorenceTrainer(config)
    f_path = f_trainer.train(train_data, val_data)
    del f_trainer
    clear_memory()
    
    # Train Qwen2.5-VL
    print("\n" + "="*80)
    print("QWEN2.5-VL TRAINING")
    print("="*80)
    q_trainer = QwenTrainer(config)
    q_path = q_trainer.train(train_data, val_data)
    
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE!")
    print("="*80)
    print(f"Florence-2:  {f_path}")
    print(f"Qwen2.5-VL:  {q_path}")
    print("\n💡 Models trained on resized images (512px max)")
    print("   Original images: 3468x4624 → 512x683 (memory optimized)")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        clear_memory()

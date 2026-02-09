"""
Complete Training Pipeline for Florence-2 and Qwen2.5-VL
Fixed version for RTX 6000 (24GB VRAM) with proper error handling
Optimized configuration: 1000 train, 300 val, 300 test samples
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
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - OPTIMIZED FOR RTX 6000 (24GB VRAM)
# ============================================================================

class Config:
    # Dataset settings
    HF_DATASET_NAME = "yusufbukarmaina/Beakers1"
    STREAMING = True  # Essential for memory management
    
    # Dataset sizes - YOUR EXACT REQUIREMENTS
    TRAIN_SAMPLES = 1000
    VAL_SAMPLES = 300
    TEST_SAMPLES = 300
    TOTAL_SAMPLES = 1600
    
    # Model settings
    FLORENCE_MODEL = "microsoft/Florence-2-base"
    QWEN_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
    
    # LoRA settings - Optimized for 24GB VRAM
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    # Training settings - REDUCED FOR RTX 6000
    BATCH_SIZE = 2  # Reduced from 4 for 24GB VRAM
    GRADIENT_ACCUMULATION = 8  # Increased to maintain effective batch size of 16
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 10
    WARMUP_STEPS = 50
    MAX_LENGTH = 512
    
    # Memory optimization
    FP16 = True
    GRADIENT_CHECKPOINTING = True
    
    # Output settings
    OUTPUT_DIR = "./trained_models"
    TEST_IMAGES_DIR = "./test_images"  # NEW: Export test images here
    SAVE_STEPS = 500
    EVAL_STEPS = 500
    LOGGING_STEPS = 50
    
    # HuggingFace upload
    UPLOAD_TO_HF = False
    HF_REPO_NAME = "yusufbukarmaina/beaker-volume-models"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clear_memory():
    """Clear CUDA cache and run garbage collection"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_memory_usage():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


# ============================================================================
# DATA PROCESSING
# ============================================================================

class DatasetProcessor:
    """Process and split dataset with streaming support"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def load_and_split_dataset(self):
        """Load dataset with streaming and create splits"""
        print("📥 Loading dataset with streaming...")
        print(f"Dataset: {self.config.HF_DATASET_NAME}")
        print(f"Target: {self.config.TRAIN_SAMPLES} train, {self.config.VAL_SAMPLES} val, {self.config.TEST_SAMPLES} test")
        
        try:
            # Load full dataset in streaming mode
            dataset = load_dataset(
                self.config.HF_DATASET_NAME,
                split="train",
                streaming=self.config.STREAMING,
                trust_remote_code=True
            )
            
            # Shuffle the dataset
            dataset = dataset.shuffle(seed=42, buffer_size=1000)
            
            print("📊 Creating splits with streaming...")
            
            # Collect splits
            train_data = []
            val_data = []
            test_data = []
            
            total_processed = 0
            skipped = 0
            
            for example in dataset:
                # Validate example has required fields
                if 'image' not in example:
                    skipped += 1
                    continue
                
                # ✅ FIX: Check for correct Beakers1 dataset fields
                # Dataset has: volume_ml (float) and volume_label (text)
                if 'volume_ml' not in example and 'volume_label' not in example:
                    skipped += 1
                    continue
                
                # Route to appropriate split
                if len(train_data) < self.config.TRAIN_SAMPLES:
                    train_data.append(example)
                elif len(val_data) < self.config.VAL_SAMPLES:
                    val_data.append(example)
                elif len(test_data) < self.config.TEST_SAMPLES:
                    test_data.append(example)
                else:
                    # We have enough data
                    break
                
                total_processed += 1
                
                # Print progress every 100 examples
                if total_processed % 100 == 0:
                    print(f"✓ Processed {total_processed} - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
                
                # Safety limit to avoid infinite loop
                if total_processed > self.config.TOTAL_SAMPLES + 500:
                    print(f"⚠️ Reached safety limit, stopping data collection")
                    break
            
            print(f"\n✅ Dataset split complete:")
            print(f"   Train: {len(train_data)} examples")
            print(f"   Val: {len(val_data)} examples")
            print(f"   Test: {len(test_data)} examples")
            print(f"   Skipped: {skipped} examples (missing fields)")
            
            if len(train_data) < self.config.TRAIN_SAMPLES:
                print(f"⚠️ Warning: Only got {len(train_data)} training samples, expected {self.config.TRAIN_SAMPLES}")
            
            if len(test_data) < self.config.TEST_SAMPLES:
                print(f"⚠️ Warning: Only got {len(test_data)} test samples, expected {self.config.TEST_SAMPLES}")
            
            return train_data, val_data, test_data
        
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def extract_volume_from_text(self, text: str) -> float:
        """Extract volume value from text answer"""
        if not text:
            return 0.0
        
        text = str(text)
        
        # Look for patterns like "250 mL", "250mL", "250.5 mL", etc.
        patterns = [
            r'(\d+\.?\d*)\s*mL',
            r'(\d+\.?\d*)\s*ml',
            r'(\d+\.?\d*)\s*milliliters?',
            r'(\d+\.?\d*)\s*ML',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        # If no pattern found, try to extract any number
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            return float(numbers[0])
        
        return 0.0
    
    def save_test_images(self, test_data: List[Dict], output_dir: str):
        """Save test images to a separate folder with metadata"""
        print(f"\n💾 Saving {len(test_data)} test images to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Also save metadata
        metadata = []
        
        for idx, example in enumerate(test_data):
            try:
                # Load image
                if isinstance(example['image'], str):
                    image = Image.open(example['image']).convert('RGB')
                else:
                    image = example['image'].convert('RGB')
                
                # Extract volume for filename
                # ✅ FIX: Use Beakers1 dataset fields (volume_label or volume_ml)
                gt_text = example.get('volume_label', '')
                if not gt_text and 'volume_ml' in example:
                    gt_text = f"{example['volume_ml']} mL"
                gt_volume = self.extract_volume_from_text(gt_text)
                
                # Save with descriptive filename
                filename = f"test_{idx:04d}_volume_{gt_volume:.1f}mL.jpg"
                save_path = os.path.join(output_dir, filename)
                image.save(save_path, quality=95)
                
                # Store metadata
                metadata.append({
                    'index': idx,
                    'filename': filename,
                    'ground_truth_volume': gt_volume,
                    'ground_truth_text': gt_text
                })
                
                if (idx + 1) % 50 == 0:
                    print(f"  Saved {idx + 1}/{len(test_data)} images...")
            
            except Exception as e:
                print(f"⚠️ Error saving test image {idx}: {e}")
                continue
        
        # Save metadata JSON
        metadata_path = os.path.join(output_dir, 'test_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved {len(test_data)} test images to {output_dir}")
        print(f"✅ Saved metadata to {metadata_path}")


# ============================================================================
# FLORENCE-2 TRAINING
# ============================================================================

class FlorenceTrainer:
    """Florence-2 model trainer with LoRA"""
    
    def __init__(self, config: Config):
        self.config = config
        self.processor = None
        self.model = None
        
    def setup_model(self):
        """Initialize Florence-2 model with LoRA"""
        print(f"\n🤖 Setting up Florence-2 model: {self.config.FLORENCE_MODEL}")
        print("⚙️ Memory optimization: FP16 enabled")
        
        clear_memory()
        
        try:
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.config.FLORENCE_MODEL,
                trust_remote_code=True
            )
            
            # Load model with FP16
            # ✅ FIX: Don't use device_map="auto" for Florence-2, manually move to GPU
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.FLORENCE_MODEL,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.config.FP16 else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Manually move to GPU
            self.model = self.model.to('cuda')
            
            print_memory_usage()
            
            # Enable gradient checkpointing for memory efficiency
            if self.config.GRADIENT_CHECKPOINTING:
                self.model.gradient_checkpointing_enable()
                print("✓ Gradient checkpointing enabled")
            
            # Prepare for LoRA
            self.model = prepare_model_for_kbit_training(self.model)
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=self.config.LORA_R,
                lora_alpha=self.config.LORA_ALPHA,
                target_modules=self.config.LORA_TARGET_MODULES,
                lora_dropout=self.config.LORA_DROPOUT,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Apply LoRA
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            print_memory_usage()
            
            return self.model, self.processor
        
        except Exception as e:
            print(f"❌ Error setting up Florence-2 model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def train(self, train_data: List[Dict], val_data: List[Dict]) -> str:
        """Train Florence-2 model"""
        print("\n🚀 Starting Florence-2 training...")
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        
        # Setup model
        self.setup_model()
        
        # Create output directory
        output_dir = f"{self.config.OUTPUT_DIR}/florence2"
        os.makedirs(output_dir, exist_ok=True)
        
        # Training arguments - Optimized for RTX 6000
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
            eval_strategy="steps",  # ✅ FIX: Changed from evaluation_strategy
            save_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=self.config.FP16,
            dataloader_num_workers=2,  # Reduced for stability
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="tensorboard",
            logging_dir=f"{output_dir}/runs",
            gradient_checkpointing=self.config.GRADIENT_CHECKPOINTING,
            optim="adamw_torch",
            max_grad_norm=1.0
        )
        
        # Create custom dataset class
        class FlorenceDataset(torch.utils.data.Dataset):
            def __init__(self, data, processor, config):
                self.data = data
                self.processor = processor
                self.config = config
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                example = self.data[idx]
                
                try:
                    # Load image
                    if isinstance(example['image'], str):
                        image = Image.open(example['image']).convert('RGB')
                    else:
                        image = example['image'].convert('RGB')
                    
                    # Create prompt
                    prompt = "<VQA>What is the volume of liquid in the beaker?"
                    
                    # Get answer - ✅ FIX: Use Beakers1 dataset fields
                    answer = example.get('volume_label', '')
                    if not answer and 'volume_ml' in example:
                        answer = f"{example['volume_ml']} mL"
                    
                    # Process
                    inputs = self.processor(
                        images=image,
                        text=prompt,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=self.config.MAX_LENGTH
                    )
                    
                    # Tokenize answer
                    answer_inputs = self.processor.tokenizer(
                        answer,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=self.config.MAX_LENGTH
                    )
                    
                    # Squeeze batch dimension
                    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                    inputs['labels'] = answer_inputs['input_ids'].squeeze(0)
                    
                    return inputs
                
                except Exception as e:
                    print(f"⚠️ Error processing example {idx}: {e}")
                    # Return a dummy sample to avoid crashes
                    dummy_image = Image.new('RGB', (224, 224), color='white')
                    inputs = self.processor(
                        images=dummy_image,
                        text="<VQA>dummy",
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=self.config.MAX_LENGTH
                    )
                    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                    inputs['labels'] = inputs['input_ids'].clone()
                    return inputs
        
        # Create datasets
        train_dataset = FlorenceDataset(train_data, self.processor, self.config)
        eval_dataset = FlorenceDataset(val_data, self.processor, self.config)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        try:
            print("Starting training loop...")
            print_memory_usage()
            trainer.train()
            print("Training completed successfully!")
        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Save final model
        final_dir = f"{self.config.OUTPUT_DIR}/florence2_final"
        print(f"\n💾 Saving final model to {final_dir}")
        trainer.save_model(final_dir)
        self.processor.save_pretrained(final_dir)
        
        print(f"✅ Florence-2 training complete! Model saved to {final_dir}")
        
        # Clear memory
        clear_memory()
        
        return final_dir


# ============================================================================
# QWEN2.5-VL TRAINING
# ============================================================================

class QwenTrainer:
    """Qwen2.5-VL model trainer with LoRA"""
    
    def __init__(self, config: Config):
        self.config = config
        self.processor = None
        self.model = None
        
    def setup_model(self):
        """Initialize Qwen2.5-VL model with LoRA"""
        print(f"\n🤖 Setting up Qwen2.5-VL model: {self.config.QWEN_MODEL}")
        print("⚙️ Memory optimization: FP16 enabled")
        
        clear_memory()
        
        try:
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.config.QWEN_MODEL,
                trust_remote_code=True
            )
            
            # Load model with FP16
            # ✅ TRY device_map="auto" first, fallback to manual GPU if it fails
            try:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.config.QWEN_MODEL,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.config.FP16 else torch.float32,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            except ValueError as e:
                if "doesn't have any device set" in str(e):
                    print("⚠️ device_map='auto' failed, loading to GPU directly...")
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        self.config.QWEN_MODEL,
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if self.config.FP16 else torch.float32,
                        low_cpu_mem_usage=True
                    )
                    self.model = self.model.to('cuda')
                else:
                    raise
            
            print_memory_usage()
            
            # Enable gradient checkpointing
            if self.config.GRADIENT_CHECKPOINTING:
                self.model.gradient_checkpointing_enable()
                print("✓ Gradient checkpointing enabled")
            
            # Prepare for LoRA
            self.model = prepare_model_for_kbit_training(self.model)
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=self.config.LORA_R,
                lora_alpha=self.config.LORA_ALPHA,
                target_modules=self.config.LORA_TARGET_MODULES,
                lora_dropout=self.config.LORA_DROPOUT,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Apply LoRA
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            print_memory_usage()
            
            return self.model, self.processor
        
        except Exception as e:
            print(f"❌ Error setting up Qwen2.5-VL model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def train(self, train_data: List[Dict], val_data: List[Dict]) -> str:
        """Train Qwen2.5-VL model"""
        print("\n🚀 Starting Qwen2.5-VL training...")
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        
        # Setup model
        self.setup_model()
        
        # Create output directory
        output_dir = f"{self.config.OUTPUT_DIR}/qwen2_5vl"
        os.makedirs(output_dir, exist_ok=True)
        
        # Training arguments - Optimized for RTX 6000
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
            eval_strategy="steps",  # ✅ FIX: Changed from evaluation_strategy
            save_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=self.config.FP16,
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="tensorboard",
            logging_dir=f"{output_dir}/runs",
            gradient_checkpointing=self.config.GRADIENT_CHECKPOINTING,
            optim="adamw_torch",
            max_grad_norm=1.0
        )
        
        # Create custom dataset class
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
                    # Load image
                    if isinstance(example['image'], str):
                        image = Image.open(example['image']).convert('RGB')
                    else:
                        image = example['image'].convert('RGB')
                    
                    # Create messages
                    question = "What is the volume of liquid in this beaker in mL?"
                    # ✅ FIXED: Get answer using correct Beakers1 field names
                    if 'volume_ml' in example:
                        answer = f"{example['volume_ml']} mL"
                    elif 'volume_label' in example:
                        answer = example['volume_label']
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
                            "content": [
                                {"type": "text", "text": answer}
                            ]
                        }
                    ]
                    
                    # Apply chat template
                    text = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    
                    # Process
                    inputs = self.processor(
                        text=[text],
                        images=[image],
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=self.config.MAX_LENGTH
                    )
                    
                    # Squeeze batch dimension
                    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                    inputs['labels'] = inputs['input_ids'].clone()
                    
                    return inputs
                
                except Exception as e:
                    print(f"⚠️ Error processing example {idx}: {e}")
                    # Return dummy sample
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
                            "content": [
                                {"type": "text", "text": "dummy"}
                            ]
                        }
                    ]
                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    inputs = self.processor(text=[text], images=[dummy_image], return_tensors="pt", padding="max_length", truncation=True, max_length=self.config.MAX_LENGTH)
                    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                    inputs['labels'] = inputs['input_ids'].clone()
                    return inputs
        
        # Create datasets
        train_dataset = QwenDataset(train_data, self.processor, self.config)
        eval_dataset = QwenDataset(val_data, self.processor, self.config)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        try:
            print("Starting training loop...")
            print_memory_usage()
            trainer.train()
            print("Training completed successfully!")
        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Save final model
        final_dir = f"{self.config.OUTPUT_DIR}/qwen2_5vl_final"
        print(f"\n💾 Saving final model to {final_dir}")
        trainer.save_model(final_dir)
        self.processor.save_pretrained(final_dir)
        
        print(f"✅ Qwen2.5-VL training complete! Model saved to {final_dir}")
        
        # Clear memory
        clear_memory()
        
        return final_dir


# ============================================================================
# EVALUATION
# ============================================================================

class ModelEvaluator:
    """Evaluate trained models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_processor = DatasetProcessor(config)
    
    def evaluate_model(self, model, processor, test_data: List[Dict], model_name: str) -> Dict:
        """Evaluate model on test set"""
        print(f"\n📊 Evaluating {model_name} on {len(test_data)} test samples...")
        
        predictions = []
        ground_truth = []
        
        model.eval()
        clear_memory()
        
        with torch.no_grad():
            for idx, example in enumerate(test_data):
                try:
                    # Load image
                    if isinstance(example['image'], str):
                        image = Image.open(example['image']).convert('RGB')
                    else:
                        image = example['image'].convert('RGB')
                    
                    # Get ground truth - ✅ FIX: Use Beakers1 dataset fields
                    gt_text = example.get('volume_label', '')
                    if not gt_text and 'volume_ml' in example:
                        gt_text = f"{example['volume_ml']} mL"
                    gt_volume = self.data_processor.extract_volume_from_text(gt_text)
                    ground_truth.append(gt_volume)
                    
                    # Generate prediction
                    if 'florence' in model_name.lower():
                        prompt = "<VQA>What is the volume of liquid in the beaker?"
                        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
                        
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=50,
                            num_beams=3,
                            early_stopping=True
                        )
                        
                        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    else:  # Qwen
                        question = "What is the volume of liquid in this beaker in mL?"
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": image},
                                    {"type": "text", "text": question}
                                ]
                            }
                        ]
                        
                        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
                        
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=50,
                            num_beams=3,
                            early_stopping=True
                        )
                        
                        generated_text = processor.batch_decode(
                            generated_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        )[0]
                    
                    # Extract predicted volume
                    pred_volume = self.data_processor.extract_volume_from_text(generated_text)
                    predictions.append(pred_volume)
                    
                    if (idx + 1) % 50 == 0:
                        print(f"  ✓ Evaluated {idx + 1}/{len(test_data)} samples")
                
                except Exception as e:
                    print(f"⚠️ Error evaluating example {idx}: {e}")
                    predictions.append(0.0)
                    continue
        
        # Calculate metrics
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        mae = mean_absolute_error(ground_truth, predictions)
        rmse = np.sqrt(mean_squared_error(ground_truth, predictions))
        r2 = r2_score(ground_truth, predictions)
        
        print(f"\n📈 {model_name} Results:")
        print(f"   MAE:  {mae:.2f} mL")
        print(f"   RMSE: {rmse:.2f} mL")
        print(f"   R²:   {r2:.4f}")
        
        # Create visualization
        self.plot_predictions(ground_truth, predictions, model_name)
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'predictions': predictions.tolist(),
            'ground_truth': ground_truth.tolist()
        }
    
    def plot_predictions(self, ground_truth, predictions, model_name):
        """Create prediction plots"""
        plt.figure(figsize=(12, 5))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(ground_truth, predictions, alpha=0.5, s=30)
        plt.plot([ground_truth.min(), ground_truth.max()], 
                 [ground_truth.min(), ground_truth.max()], 
                 'r--', lw=2, label='Perfect prediction')
        plt.xlabel('Ground Truth (mL)', fontsize=12)
        plt.ylabel('Predictions (mL)', fontsize=12)
        plt.title(f'{model_name} - Predictions vs Ground Truth', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Error distribution
        plt.subplot(1, 2, 2)
        errors = predictions - ground_truth
        plt.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        plt.xlabel('Prediction Error (mL)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'{model_name} - Error Distribution', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero error')
        plt.axvline(x=errors.mean(), color='g', linestyle='--', lw=2, label=f'Mean error: {errors.mean():.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = f"{self.config.OUTPUT_DIR}/{model_name.replace(' ', '_')}_evaluation.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Plot saved to: {plot_path}")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("="*80)
    print("🚀 Vision Model Training Pipeline - Florence-2 & Qwen2.5-VL")
    print("="*80)
    print(f"GPU: RTX 6000 (24GB VRAM)")
    print(f"Configuration:")
    print(f"  Dataset: {Config.HF_DATASET_NAME}")
    print(f"  Train samples: {Config.TRAIN_SAMPLES}")
    print(f"  Val samples: {Config.VAL_SAMPLES}")
    print(f"  Test samples: {Config.TEST_SAMPLES}")
    print(f"  Batch size: {Config.BATCH_SIZE}")
    print(f"  Gradient accumulation: {Config.GRADIENT_ACCUMULATION}")
    print(f"  Effective batch size: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION}")
    print(f"  FP16: {Config.FP16}")
    print(f"  Gradient checkpointing: {Config.GRADIENT_CHECKPOINTING}")
    print("="*80)
    
    # Initialize config
    config = Config()
    
    # Create output directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.TEST_IMAGES_DIR, exist_ok=True)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print_memory_usage()
    else:
        print("\n❌ WARNING: No GPU detected! Training will be very slow.")
    
    # Load and split dataset
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    data_processor = DatasetProcessor(config)
    train_data, val_data, test_data = data_processor.load_and_split_dataset()
    
    # Save test images to separate folder
    print("\n" + "="*80)
    print("EXPORTING TEST IMAGES")
    print("="*80)
    data_processor.save_test_images(test_data, config.TEST_IMAGES_DIR)
    
    # Train Florence-2
    print("\n" + "="*80)
    print("FLORENCE-2 TRAINING")
    print("="*80)
    florence_trainer = FlorenceTrainer(config)
    florence_path = florence_trainer.train(train_data, val_data)
    
    # Clear memory before next model
    del florence_trainer
    clear_memory()
    print("\n🧹 Cleared memory after Florence-2 training")
    print_memory_usage()
    
    # Train Qwen2.5-VL
    print("\n" + "="*80)
    print("QWEN2.5-VL TRAINING")
    print("="*80)
    qwen_trainer = QwenTrainer(config)
    qwen_path = qwen_trainer.train(train_data, val_data)
    
    # Reload models for evaluation
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    evaluator = ModelEvaluator(config)
    
    # Evaluate Florence-2
    print("\n--- Florence-2 Evaluation ---")
    florence_trainer_eval = FlorenceTrainer(config)
    florence_trainer_eval.setup_model()
    
    florence_results = evaluator.evaluate_model(
        florence_trainer_eval.model,
        florence_trainer_eval.processor,
        test_data,
        "Florence-2"
    )
    
    del florence_trainer_eval
    clear_memory()
    
    # Evaluate Qwen2.5-VL
    print("\n--- Qwen2.5-VL Evaluation ---")
    qwen_results = evaluator.evaluate_model(
        qwen_trainer.model,
        qwen_trainer.processor,
        test_data,
        "Qwen2.5-VL"
    )
    
    # Save results
    results = {
        'florence2': florence_results,
        'qwen2_5vl': qwen_results,
        'config': {
            'dataset': config.HF_DATASET_NAME,
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data),
            'epochs': config.NUM_EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'gradient_accumulation': config.GRADIENT_ACCUMULATION,
            'effective_batch_size': config.BATCH_SIZE * config.GRADIENT_ACCUMULATION,
            'learning_rate': config.LEARNING_RATE,
            'fp16': config.FP16,
            'gradient_checkpointing': config.GRADIENT_CHECKPOINTING
        }
    }
    
    results_path = f"{config.OUTPUT_DIR}/evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")
    
    # Upload to HuggingFace (optional)
    if config.UPLOAD_TO_HF:
        print("\n" + "="*80)
        print("UPLOADING TO HUGGINGFACE")
        print("="*80)
        
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            # Upload Florence-2
            print("📤 Uploading Florence-2...")
            api.upload_folder(
                folder_path=florence_path,
                repo_id=f"{config.HF_REPO_NAME}-florence2",
                repo_type="model"
            )
            print("✅ Florence-2 uploaded")
            
            # Upload Qwen2.5-VL
            print("📤 Uploading Qwen2.5-VL...")
            api.upload_folder(
                folder_path=qwen_path,
                repo_id=f"{config.HF_REPO_NAME}-qwen2-5vl",
                repo_type="model"
            )
            print("✅ Qwen2.5-VL uploaded")
            
            print("✅ Models uploaded to HuggingFace!")
        
        except Exception as e:
            print(f"⚠️ Error uploading to HuggingFace: {e}")
    
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModels saved to:")
    print(f"  Florence-2:   {florence_path}")
    print(f"  Qwen2.5-VL:   {qwen_path}")
    print(f"\nResults:        {results_path}")
    print(f"Test images:    {config.TEST_IMAGES_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Review evaluation plots in {config.OUTPUT_DIR}/")
    print(f"  2. Test models: python gradio_demo.py --share")
    print(f"  3. Check test images in {config.TEST_IMAGES_DIR}/")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        clear_memory()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        clear_memory()

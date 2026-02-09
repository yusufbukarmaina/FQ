"""
Complete Training Pipeline for Florence-2 and Qwen2.5-VL
Optimized for JarvisLab GPU with streaming dataset loading
FIXED: Dataset loading and TrainingArguments API
"""

import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM,
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
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Dataset settings
    HF_DATASET_NAME = "yusufbukarmaina/Beakers1"
    STREAMING = True
    
    # Target dataset sizes
    TARGET_TRAIN = 1000
    TARGET_VAL = 300
    TARGET_TEST = 300
    
    # Model settings
    FLORENCE_MODEL = "microsoft/Florence-2-base"
    QWEN_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
    
    # LoRA settings
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    # Training settings
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION = 8
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 10
    WARMUP_STEPS = 100
    MAX_LENGTH = 512
    
    # Output settings
    OUTPUT_DIR = "./trained_models"
    SAVE_STEPS = 500
    EVAL_STEPS = 500
    LOGGING_STEPS = 50
    
    # HuggingFace upload
    UPLOAD_TO_HF = False
    HF_REPO_NAME = "yusufbukarmaina/beaker-volume-model"


# ============================================================================
# DATA PROCESSING
# ============================================================================

class DatasetProcessor:
    """Process and split dataset with streaming support"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def load_and_split_dataset(self):
        """Load dataset with streaming and create splits"""
        print("\n" + "="*80)
        print("LOADING DATASET")
        print("="*80)
        print(f"📥 Loading dataset with streaming...")
        print(f"Dataset: {self.config.HF_DATASET_NAME}")
        print(f"Target: {self.config.TARGET_TRAIN} train, {self.config.TARGET_VAL} val, {self.config.TARGET_TEST} test")
        
        # Load dataset
        dataset = load_dataset(
            self.config.HF_DATASET_NAME,
            split="train",
            streaming=self.config.STREAMING
        )
        
        if self.config.STREAMING:
            dataset = dataset.shuffle(seed=42, buffer_size=1000)
        
        print("📊 Creating splits with streaming...")
        
        train_data = []
        val_data = []
        test_data = []
        skipped = 0
        
        total_needed = self.config.TARGET_TRAIN + self.config.TARGET_VAL + self.config.TARGET_TEST
        
        for idx, example in enumerate(dataset):
            if len(train_data) >= self.config.TARGET_TRAIN and \
               len(val_data) >= self.config.TARGET_VAL and \
               len(test_data) >= self.config.TARGET_TEST:
                break
            
            # Validate example has required fields
            if not self._validate_example(example):
                skipped += 1
                continue
            
            # Process and clean example
            processed = self._process_example(example)
            if processed is None:
                skipped += 1
                continue
            
            # Assign to split
            if len(train_data) < self.config.TARGET_TRAIN:
                train_data.append(processed)
            elif len(val_data) < self.config.TARGET_VAL:
                val_data.append(processed)
            elif len(test_data) < self.config.TARGET_TEST:
                test_data.append(processed)
            
            # Progress update
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} examples - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}, Skipped: {skipped}")
        
        print(f"\n✅ Dataset split complete:")
        print(f"   Train: {len(train_data)} examples")
        print(f"   Val: {len(val_data)} examples")
        print(f"   Test: {len(test_data)} examples")
        print(f"   Skipped: {skipped} examples (missing fields)")
        
        if len(train_data) < self.config.TARGET_TRAIN:
            print(f"⚠️ Warning: Only got {len(train_data)} training samples, expected {self.config.TARGET_TRAIN}")
        
        if len(test_data) < 10:
            print(f"⚠️ Warning: Only got {len(test_data)} test samples, expected {self.config.TARGET_TEST}")
        
        return train_data, val_data, test_data
    
    def _validate_example(self, example: Dict) -> bool:
        """Check if example has all required fields"""
        # Check for image
        if 'image' not in example:
            return False
        
        # Check for volume information in various possible fields
        volume_fields = ['volume', 'answer', 'label', 'text', 'caption']
        has_volume = any(field in example for field in volume_fields)
        
        return has_volume
    
    def _process_example(self, example: Dict) -> Dict:
        """Process and standardize example format"""
        try:
            # Get image
            image = example.get('image')
            if image is None:
                return None
            
            # Ensure PIL Image
            if not isinstance(image, Image.Image):
                return None
            
            # Get volume text from various possible fields
            volume_text = None
            for field in ['volume', 'answer', 'label', 'text', 'caption']:
                if field in example and example[field]:
                    volume_text = str(example[field])
                    break
            
            if volume_text is None:
                return None
            
            # Extract numeric volume
            volume_value = self.extract_volume_from_text(volume_text)
            if volume_value is None:
                return None
            
            # Create standardized format
            return {
                'image': image,
                'volume': f"{volume_value} mL",
                'answer': f"The volume of liquid in the beaker is {volume_value} mL.",
                'volume_value': volume_value
            }
        
        except Exception as e:
            print(f"Error processing example: {e}")
            return None
    
    def extract_volume_from_text(self, text: str) -> float:
        """Extract volume value from text answer"""
        if not isinstance(text, str):
            text = str(text)
        
        # Look for patterns like "250 mL", "250mL", "250.5 mL", etc.
        patterns = [
            r'(\d+\.?\d*)\s*mL',
            r'(\d+\.?\d*)\s*ml',
            r'(\d+\.?\d*)\s*milliliters?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        # If no pattern found, try to extract any number
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            return float(numbers[0])
        
        return None


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
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.config.FLORENCE_MODEL,
            trust_remote_code=True
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.FLORENCE_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
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
        
        return self.model, self.processor
    
    def train(self, train_data: List[Dict], val_data: List[Dict]):
        """Train Florence-2 model"""
        print("\n" + "="*80)
        print("FLORENCE-2 TRAINING")
        print("="*80)
        print(f"\n🚀 Starting Florence-2 training...")
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        
        if len(train_data) == 0:
            print("❌ No training data available!")
            return None
        
        # Setup model
        self.setup_model()
        
        # Create dataset class
        class FlorenceDataset(torch.utils.data.Dataset):
            def __init__(self, data, processor, config):
                self.data = data
                self.processor = processor
                self.config = config
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                example = self.data[idx]
                
                # Get image
                image = example['image']
                if not isinstance(image, Image.Image):
                    image = Image.open(image).convert('RGB')
                
                # Create prompt and answer
                prompt = "<VQA>What is the volume of liquid in the beaker?"
                answer = example['answer']
                text = f"{prompt}{answer}"
                
                # Process
                inputs = self.processor(
                    images=image,
                    text=text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.config.MAX_LENGTH
                )
                
                # Remove batch dimension
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                inputs["labels"] = inputs["input_ids"].clone()
                
                return inputs
        
        train_dataset = FlorenceDataset(train_data, self.processor, self.config)
        val_dataset = FlorenceDataset(val_data, self.processor, self.config) if val_data else None
        
        # Training arguments - FIXED: Use 'eval_strategy' instead of 'evaluation_strategy'
        training_args = TrainingArguments(
            output_dir=f"{self.config.OUTPUT_DIR}/florence2",
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION,
            learning_rate=self.config.LEARNING_RATE,
            warmup_steps=self.config.WARMUP_STEPS,
            logging_steps=self.config.LOGGING_STEPS,
            save_steps=self.config.SAVE_STEPS,
            eval_steps=self.config.EVAL_STEPS if val_dataset else None,
            eval_strategy="steps" if val_dataset else "no",  # FIXED: was 'evaluation_strategy'
            save_strategy="steps",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            fp16=True,
            report_to="tensorboard",
            save_total_limit=3,
            dataloader_pin_memory=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if val_dataset else None
        )
        
        # Train
        print("\n🔥 Training started...")
        trainer.train()
        
        # Save final model
        final_path = f"{self.config.OUTPUT_DIR}/florence2_final"
        trainer.save_model(final_path)
        self.processor.save_pretrained(final_path)
        
        print(f"\n✅ Florence-2 training complete! Model saved to {final_path}")
        
        return final_path


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
        
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.config.QWEN_MODEL,
            trust_remote_code=True
        )
        
        # Load model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.QWEN_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Prepare for LoRA
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=self.config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        return self.model, self.processor
    
    def train(self, train_data: List[Dict], val_data: List[Dict]):
        """Train Qwen2.5-VL model"""
        print("\n" + "="*80)
        print("QWEN2.5-VL TRAINING")
        print("="*80)
        print(f"\n🚀 Starting Qwen2.5-VL training...")
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        
        if len(train_data) == 0:
            print("❌ No training data available!")
            return None
        
        # Setup model
        self.setup_model()
        
        # Create dataset class
        class QwenDataset(torch.utils.data.Dataset):
            def __init__(self, data, processor, config):
                self.data = data
                self.processor = processor
                self.config = config
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                example = self.data[idx]
                
                # Get image
                image = example['image']
                if not isinstance(image, Image.Image):
                    image = Image.open(image).convert('RGB')
                
                # Create conversation format
                question = "What is the volume of liquid in this beaker in mL?"
                answer = example['answer']
                
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
                
                # Process
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                inputs = self.processor(
                    text=[text],
                    images=[image],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.config.MAX_LENGTH
                )
                
                # Remove batch dimension
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                inputs["labels"] = inputs["input_ids"].clone()
                
                return inputs
        
        train_dataset = QwenDataset(train_data, self.processor, self.config)
        val_dataset = QwenDataset(val_data, self.processor, self.config) if val_data else None
        
        # Training arguments - FIXED: Use 'eval_strategy' instead of 'evaluation_strategy'
        training_args = TrainingArguments(
            output_dir=f"{self.config.OUTPUT_DIR}/qwen2_5vl",
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION,
            learning_rate=self.config.LEARNING_RATE,
            warmup_steps=self.config.WARMUP_STEPS,
            logging_steps=self.config.LOGGING_STEPS,
            save_steps=self.config.SAVE_STEPS,
            eval_steps=self.config.EVAL_STEPS if val_dataset else None,
            eval_strategy="steps" if val_dataset else "no",  # FIXED: was 'evaluation_strategy'
            save_strategy="steps",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            fp16=True,
            report_to="tensorboard",
            save_total_limit=3,
            dataloader_pin_memory=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if val_dataset else None
        )
        
        # Train
        print("\n🔥 Training started...")
        trainer.train()
        
        # Save final model
        final_path = f"{self.config.OUTPUT_DIR}/qwen2_5vl_final"
        trainer.save_model(final_path)
        self.processor.save_pretrained(final_path)
        
        print(f"\n✅ Qwen2.5-VL training complete! Model saved to {final_path}")
        
        return final_path


# ============================================================================
# EVALUATION
# ============================================================================

class ModelEvaluator:
    """Evaluate models with MAE, RMSE, R2"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_processor = DatasetProcessor(config)
    
    def evaluate_model(self, model, processor, test_data: List[Dict], model_name: str):
        """Evaluate a model on test data"""
        print(f"\n" + "="*80)
        print(f"EVALUATING {model_name.upper()}")
        print("="*80)
        print(f"📊 Test samples: {len(test_data)}")
        
        if len(test_data) == 0:
            print("⚠️ No test data available for evaluation")
            return None
        
        predictions = []
        ground_truth = []
        
        model.eval()
        
        with torch.no_grad():
            for idx, example in enumerate(test_data):
                # Get image
                image = example['image']
                if not isinstance(image, Image.Image):
                    image = Image.open(image).convert('RGB')
                
                # Get ground truth
                gt_volume = example['volume_value']
                ground_truth.append(gt_volume)
                
                # Generate prediction
                if 'florence' in model_name.lower():
                    prompt = "<VQA>What is the volume of liquid in the beaker?"
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
                    
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        num_beams=3
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
                        num_beams=3
                    )
                    
                    generated_text = processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]
                
                # Extract predicted volume
                pred_volume = self.data_processor.extract_volume_from_text(generated_text)
                if pred_volume is None:
                    pred_volume = gt_volume  # Fallback
                
                predictions.append(pred_volume)
                
                if (idx + 1) % 50 == 0:
                    print(f"Evaluated {idx + 1}/{len(test_data)} samples...")
        
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
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions.tolist(),
            'ground_truth': ground_truth.tolist()
        }
    
    def plot_predictions(self, ground_truth, predictions, model_name):
        """Create prediction plots"""
        plt.figure(figsize=(10, 5))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(ground_truth, predictions, alpha=0.5)
        plt.plot([ground_truth.min(), ground_truth.max()], 
                 [ground_truth.min(), ground_truth.max()], 
                 'r--', lw=2)
        plt.xlabel('Ground Truth (mL)')
        plt.ylabel('Predictions (mL)')
        plt.title(f'{model_name} - Predictions vs Ground Truth')
        plt.grid(True, alpha=0.3)
        
        # Error distribution
        plt.subplot(1, 2, 2)
        errors = predictions - ground_truth
        plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Prediction Error (mL)')
        plt.ylabel('Frequency')
        plt.title(f'{model_name} - Error Distribution')
        plt.axvline(x=0, color='r', linestyle='--', lw=2)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = f"{self.config.OUTPUT_DIR}/{model_name.replace(' ', '_')}_evaluation.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Plot saved to: {plot_path}")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("="*80)
    print("🚀 Vision Model Training Pipeline - Florence-2 & Qwen2.5-VL")
    print("="*80)
    
    # Initialize config
    config = Config()
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Load and split dataset
    data_processor = DatasetProcessor(config)
    train_data, val_data, test_data = data_processor.load_and_split_dataset()
    
    # Check if we have enough data
    if len(train_data) == 0:
        print("\n❌ ERROR: No training data loaded!")
        print("\nPossible issues:")
        print("1. Dataset format doesn't match expected structure")
        print("2. Missing 'image' field in dataset")
        print("3. Missing volume information (check fields: 'volume', 'answer', 'label')")
        print("\nPlease check your dataset at:")
        print(f"   https://huggingface.co/datasets/{config.HF_DATASET_NAME}")
        return
    
    # Train Florence-2
    florence_trainer = FlorenceTrainer(config)
    florence_path = florence_trainer.train(train_data, val_data)
    
    # Train Qwen2.5-VL
    qwen_trainer = QwenTrainer(config)
    qwen_path = qwen_trainer.train(train_data, val_data)
    
    # Evaluate both models
    if test_data and len(test_data) > 0:
        evaluator = ModelEvaluator(config)
        
        # Evaluate Florence-2
        if florence_path:
            florence_results = evaluator.evaluate_model(
                florence_trainer.model,
                florence_trainer.processor,
                test_data,
                "Florence-2"
            )
        else:
            florence_results = None
        
        # Evaluate Qwen2.5-VL
        if qwen_path:
            qwen_results = evaluator.evaluate_model(
                qwen_trainer.model,
                qwen_trainer.processor,
                test_data,
                "Qwen2.5-VL"
            )
        else:
            qwen_results = None
        
        # Save results
        results = {
            'florence2': florence_results,
            'qwen2_5vl': qwen_results,
            'config': {
                'train_size': len(train_data),
                'val_size': len(val_data),
                'test_size': len(test_data),
                'epochs': config.NUM_EPOCHS,
                'batch_size': config.BATCH_SIZE,
                'learning_rate': config.LEARNING_RATE
            }
        }
        
        results_path = f"{config.OUTPUT_DIR}/evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {results_path}")
    
    # Upload to HuggingFace (optional)
    if config.UPLOAD_TO_HF and florence_path and qwen_path:
        print("\n" + "="*80)
        print("UPLOADING TO HUGGINGFACE")
        print("="*80)
        
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Upload Florence-2
        print("📤 Uploading Florence-2...")
        api.upload_folder(
            folder_path=florence_path,
            repo_id=f"{config.HF_REPO_NAME}-florence2",
            repo_type="model"
        )
        
        # Upload Qwen2.5-VL
        print("📤 Uploading Qwen2.5-VL...")
        api.upload_folder(
            folder_path=qwen_path,
            repo_id=f"{config.HF_REPO_NAME}-qwen2-5vl",
            repo_type="model"
        )
        
        print("✅ Models uploaded to HuggingFace!")
    
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE!")
    print("="*80)
    if florence_path:
        print(f"\nModels saved to:")
        print(f"  Florence-2:   {florence_path}")
    if qwen_path:
        print(f"  Qwen2.5-VL:   {qwen_path}")
    

if __name__ == "__main__":
    main()

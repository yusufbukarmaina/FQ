"""
Complete Training Pipeline with Comprehensive Metrics Tracking
For Thesis Documentation - Tracks training time, GPU usage, and all evaluation metrics
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
    EarlyStoppingCallback,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import re
from typing import Dict, List
import warnings
import gc
import time
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ============================================================================
# METRICS TRACKER CLASS
# ============================================================================

class MetricsTracker:
    """Track all metrics for thesis documentation"""
    
    def __init__(self):
        self.metrics = {
            'experiment_info': {},
            'dataset_info': {},
            'training_metrics': {},
            'gpu_metrics': {},
            'evaluation_metrics': {},
            'timing_metrics': {}
        }
        self.gpu_history = []
        self.start_time = None
    
    def start_experiment(self, config):
        """Record experiment start"""
        self.start_time = time.time()
        self.metrics['experiment_info'] = {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'gpu_total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A'
        }
    
    def record_dataset_info(self, train_size, val_size, test_size):
        """Record dataset statistics"""
        self.metrics['dataset_info'] = {
            'total_samples': train_size + val_size + test_size,
            'train_samples': train_size,
            'val_samples': val_size,
            'test_samples': test_size,
            'train_ratio': train_size / (train_size + val_size + test_size),
            'val_ratio': val_size / (train_size + val_size + test_size),
            'test_ratio': test_size / (train_size + val_size + test_size)
        }
    
    def record_gpu_usage(self, phase='training'):
        """Record current GPU usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9
            
            self.gpu_history.append({
                'timestamp': time.time() - self.start_time,
                'phase': phase,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated
            })
            
            return allocated, reserved, max_allocated
        return 0, 0, 0
    
    def record_training_metrics(self, model_name, train_time, num_epochs, num_parameters, trainable_parameters):
        """Record training metrics"""
        if model_name not in self.metrics['training_metrics']:
            self.metrics['training_metrics'][model_name] = {}
        
        self.metrics['training_metrics'][model_name].update({
            'training_time_seconds': train_time,
            'training_time_minutes': train_time / 60,
            'training_time_hours': train_time / 3600,
            'num_epochs': num_epochs,
            'total_parameters': num_parameters,
            'trainable_parameters': trainable_parameters,
            'trainable_percentage': (trainable_parameters / num_parameters) * 100
        })
    
    def record_evaluation_metrics(self, model_name, metrics_dict):
        """Record evaluation metrics"""
        self.metrics['evaluation_metrics'][model_name] = metrics_dict
    
    def generate_summary_table(self):
        """Generate comprehensive summary table"""
        summary = {
            'Model': [],
            'Train Time (min)': [],
            'Parameters': [],
            'Trainable %': [],
            'MAE (mL)': [],
            'RMSE (mL)': [],
            'R²': [],
            'MAPE (%)': [],
            'Peak GPU (GB)': []
        }
        
        for model_name in self.metrics['training_metrics'].keys():
            summary['Model'].append(model_name)
            summary['Train Time (min)'].append(
                f"{self.metrics['training_metrics'][model_name]['training_time_minutes']:.2f}"
            )
            summary['Parameters'].append(
                f"{self.metrics['training_metrics'][model_name]['total_parameters']:,}"
            )
            summary['Trainable %'].append(
                f"{self.metrics['training_metrics'][model_name]['trainable_percentage']:.2f}"
            )
            
            if model_name in self.metrics['evaluation_metrics']:
                eval_metrics = self.metrics['evaluation_metrics'][model_name]
                summary['MAE (mL)'].append(f"{eval_metrics['mae']:.2f}")
                summary['RMSE (mL)'].append(f"{eval_metrics['rmse']:.2f}")
                summary['R²'].append(f"{eval_metrics['r2']:.4f}")
                summary['MAPE (%)'].append(f"{eval_metrics.get('mape', 0):.2f}")
            else:
                summary['MAE (mL)'].append('N/A')
                summary['RMSE (mL)'].append('N/A')
                summary['R²'].append('N/A')
                summary['MAPE (%)'].append('N/A')
            
            # Get peak GPU for this model
            model_gpu = [g for g in self.gpu_history if model_name.lower() in g['phase'].lower()]
            peak_gpu = max([g['max_allocated_gb'] for g in model_gpu]) if model_gpu else 0
            summary['Peak GPU (GB)'].append(f"{peak_gpu:.2f}")
        
        return pd.DataFrame(summary)
    
    def save_all_metrics(self, output_dir):
        """Save all metrics to various formats"""
        # Save complete JSON
        json_path = os.path.join(output_dir, 'complete_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✅ Complete metrics saved to: {json_path}")
        
        # Save GPU history CSV
        if self.gpu_history:
            gpu_df = pd.DataFrame(self.gpu_history)
            gpu_csv = os.path.join(output_dir, 'gpu_usage_history.csv')
            gpu_df.to_csv(gpu_csv, index=False)
            print(f"✅ GPU usage history saved to: {gpu_csv}")
        
        # Save summary table
        summary_df = self.generate_summary_table()
        summary_csv = os.path.join(output_dir, 'summary_table.csv')
        summary_df.to_csv(summary_csv, index=False)
        print(f"✅ Summary table saved to: {summary_csv}")
        
        # Save LaTeX table for thesis
        latex_path = os.path.join(output_dir, 'summary_table.tex')
        with open(latex_path, 'w') as f:
            f.write(summary_df.to_latex(index=False))
        print(f"✅ LaTeX table saved to: {latex_path}")
        
        return summary_df


# ============================================================================
# CUSTOM TRAINER CALLBACK FOR METRICS
# ============================================================================

class MetricsCallback(TrainerCallback):
    """Callback to track training metrics"""
    
    def __init__(self, tracker, model_name):
        self.tracker = tracker
        self.model_name = model_name
        self.epoch_start_time = None
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        self.tracker.record_gpu_usage(phase=f'{self.model_name}_epoch_{state.epoch}')
    
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        print(f"  Epoch {state.epoch} completed in {epoch_time/60:.2f} minutes")
        self.tracker.record_gpu_usage(phase=f'{self.model_name}_epoch_{state.epoch}_end')
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:
            self.tracker.record_gpu_usage(phase=f'{self.model_name}_step_{state.global_step}')


# ============================================================================
# [INCLUDE ALL PREVIOUS CODE HERE - Config, DatasetProcessor, etc.]
# ============================================================================

class Config:
    HF_DATASET_NAME = "yusufbukarmaina/Beakers1"
    STREAMING = True
    
    TRAIN_SAMPLES = 1000
    VAL_SAMPLES = 300
    TEST_SAMPLES = 300
    
    FLORENCE_MODEL = "microsoft/Florence-2-base"
    QWEN_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
    
    MAX_IMAGE_SIZE = 512  # Resize images to save memory
    
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj"]
    
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 16
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 5
    WARMUP_STEPS = 50
    MAX_LENGTH = 512
    
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
        torch.cuda.synchronize()


def resize_image(image, max_size=512):
    """Resize image to reduce memory"""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert('RGB')
    
    width, height = image.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image


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
            
            if (len(train_data) + len(val_data) + len(test_data)) % 100 == 0:
                print(f"✓ Loaded {len(train_data) + len(val_data) + len(test_data)}")
        
        print(f"\n✅ Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        return train_data, val_data, test_data
    
    def extract_volume(self, text):
        if not text:
            return 0.0
        numbers = re.findall(r'\d+\.?\d*', str(text))
        return float(numbers[0]) if numbers else 0.0


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


class FlorenceTrainer:
    def __init__(self, config, tracker):
        self.config = config
        self.tracker = tracker
        
    def setup_model(self):
        print("\n🤖 Loading Florence-2...")
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
            target_modules=self.config.LORA_TARGET_MODULES,
            lora_dropout=self.config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Record model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return total_params, trainable_params
    
    def train(self, train_data, val_data):
        print("\n🚀 Training Florence-2...")
        
        train_start = time.time()
        total_params, trainable_params = self.setup_model()
        self.tracker.record_gpu_usage('florence2_model_loaded')
        
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
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=florence_collate_fn,
            callbacks=[MetricsCallback(self.tracker, "Florence-2")]
        )
        
        trainer.train()
        
        train_time = time.time() - train_start
        self.tracker.record_training_metrics(
            "Florence-2",
            train_time,
            self.config.NUM_EPOCHS,
            total_params,
            trainable_params
        )
        
        final_dir = f"{self.config.OUTPUT_DIR}/florence2_final"
        trainer.save_model(final_dir)
        self.processor.save_pretrained(final_dir)
        
        print(f"✅ Florence-2 trained in {train_time/60:.2f} minutes")
        return final_dir


class QwenTrainer:
    def __init__(self, config, tracker):
        self.config = config
        self.tracker = tracker
        
    def setup_model(self):
        print("\n🤖 Loading Qwen2.5-VL...")
        clear_memory()
        
        self.processor = AutoProcessor.from_pretrained(
            self.config.QWEN_MODEL, trust_remote_code=True
        )
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.QWEN_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={0: "35GB"}  # Limit for A100 40GB
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
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return total_params, trainable_params
    
    def train(self, train_data, val_data):
        print("\n🚀 Training Qwen2.5-VL...")
        
        train_start = time.time()
        total_params, trainable_params = self.setup_model()
        self.tracker.record_gpu_usage('qwen2.5vl_model_loaded')
        
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
                
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                
                inputs = self.processor(
                    text=[text],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                )
                
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                inputs['labels'] = inputs['input_ids'].clone()
                
                return inputs
        
        train_dataset = QwenDataset(train_data, self.processor, self.config.MAX_IMAGE_SIZE)
        eval_dataset = QwenDataset(val_data, self.processor, self.config.MAX_IMAGE_SIZE)
        
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
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[MetricsCallback(self.tracker, "Qwen2.5-VL")]
        )
        
        trainer.train()
        
        train_time = time.time() - train_start
        self.tracker.record_training_metrics(
            "Qwen2.5-VL",
            train_time,
            self.config.NUM_EPOCHS,
            total_params,
            trainable_params
        )
        
        final_dir = f"{self.config.OUTPUT_DIR}/qwen2_5vl_final"
        trainer.save_model(final_dir)
        self.processor.save_pretrained(final_dir)
        
        print(f"✅ Qwen2.5-VL trained in {train_time/60:.2f} minutes")
        return final_dir


class ModelEvaluator:
    def __init__(self, config, tracker):
        self.config = config
        self.tracker = tracker
        self.data_processor = DatasetProcessor(config)
    
    def evaluate(self, model, processor, test_data, model_name):
        print(f"\n📊 Evaluating {model_name}...")
        
        eval_start = time.time()
        model.eval()
        
        predictions, ground_truth = [], []
        
        with torch.no_grad():
            for i, ex in enumerate(test_data):
                try:
                    img = resize_image(ex['image'], self.config.MAX_IMAGE_SIZE)
                    gt = self.data_processor.extract_volume(
                        ex.get('volume_label', f"{ex.get('volume_ml', 0)}")
                    )
                    ground_truth.append(gt)
                    
                    if 'florence' in model_name.lower():
                        inputs = processor(images=img, text="<VQA>What is the volume?", return_tensors="pt").to(model.device)
                        ids = model.generate(**inputs, max_new_tokens=50)
                        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
                    else:
                        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "What is the volume?"}]}]
                        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = processor(text=[text_prompt], images=[img], return_tensors="pt").to(model.device)
                        ids = model.generate(**inputs, max_new_tokens=50)
                        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
                    
                    pred = self.data_processor.extract_volume(text)
                    predictions.append(pred)
                    
                    if (i + 1) % 50 == 0:
                        print(f"  {i + 1}/{len(test_data)}")
                except:
                    predictions.append(0.0)
        
        eval_time = time.time() - eval_start
        
        p, g = np.array(predictions), np.array(ground_truth)
        mae = mean_absolute_error(g, p)
        rmse = np.sqrt(mean_squared_error(g, p))
        r2 = r2_score(g, p)
        
        # Additional metrics for thesis
        mape = mean_absolute_percentage_error(g[g != 0], p[g != 0]) * 100 if any(g != 0) else 0
        max_error = np.max(np.abs(p - g))
        min_error = np.min(np.abs(p - g))
        std_error = np.std(p - g)
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'max_error': float(max_error),
            'min_error': float(min_error),
            'std_error': float(std_error),
            'eval_time_seconds': eval_time,
            'predictions': p.tolist(),
            'ground_truth': g.tolist()
        }
        
        self.tracker.record_evaluation_metrics(model_name, metrics)
        
        print(f"📈 {model_name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}, MAPE={mape:.2f}%")
        
        return metrics


def main():
    print("="*80)
    print("🚀 Complete Training Pipeline with Comprehensive Metrics")
    print("="*80)
    
    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Initialize metrics tracker
    tracker = MetricsTracker()
    tracker.start_experiment(config)
    tracker.record_gpu_usage('experiment_start')
    
    # Load data
    processor = DatasetProcessor(config)
    train_data, val_data, test_data = processor.load_and_split_dataset()
    tracker.record_dataset_info(len(train_data), len(val_data), len(test_data))
    
    # Train Florence-2
    f_trainer = FlorenceTrainer(config, tracker)
    f_path = f_trainer.train(train_data, val_data)
    del f_trainer
    clear_memory()
    
    # Train Qwen2.5-VL
    q_trainer = QwenTrainer(config, tracker)
    q_path = q_trainer.train(train_data, val_data)
    
    # Evaluate
    evaluator = ModelEvaluator(config, tracker)
    
    f_eval = FlorenceTrainer(config, tracker)
    f_eval.setup_model()
    f_results = evaluator.evaluate(f_eval.model, f_eval.processor, test_data, "Florence-2")
    del f_eval
    clear_memory()
    
    q_results = evaluator.evaluate(q_trainer.model, q_trainer.processor, test_data, "Qwen2.5-VL")
    
    # Save all metrics
    print("\n" + "="*80)
    print("SAVING COMPREHENSIVE METRICS")
    print("="*80)
    summary_df = tracker.save_all_metrics(config.OUTPUT_DIR)
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE FOR THESIS")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("🎉 COMPLETE!")
    print("="*80)
    print(f"\nAll metrics saved to: {config.OUTPUT_DIR}/")
    print("  - complete_metrics.json")
    print("  - summary_table.csv")
    print("  - summary_table.tex (for LaTeX)")
    print("  - gpu_usage_history.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

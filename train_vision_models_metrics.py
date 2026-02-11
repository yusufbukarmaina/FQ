"""
COMPLETE MEMORY-EFFICIENT Training Pipeline
✅ Lazy loading (handles 1000+ samples)
✅ Test image export
✅ Full evaluation with plots
✅ Metrics tracking (MAE, RMSE, R²)
✅ HuggingFace push
"""

import os, json, gc, time, re
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from datasets import load_dataset, Dataset
from transformers import (
    AutoProcessor, AutoModelForCausalLM,
    Qwen2VLForConditionalGeneration,
    TrainingArguments, Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from PIL import Image
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    HF_DATASET_NAME = "yusufbukarmaina/Beakers1"
    
    # Now you can use 1000+ samples without OOM!
    TRAIN_SAMPLES = 1000
    VAL_SAMPLES = 300
    TEST_SAMPLES = 300
    
    FLORENCE_MODEL = "microsoft/Florence-2-base"
    QWEN_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
    
    MAX_IMAGE_SIZE = 512
    
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10
    WARMUP_STEPS = 100
    
    FP16 = True
    GRADIENT_CHECKPOINTING = True
    
    OUTPUT_DIR = "./trained_models"
    TEST_IMAGES_DIR = "./test_images"
    SAVE_STEPS = 500
    EVAL_STEPS = 500
    LOGGING_STEPS = 25
    
    # HuggingFace Hub
    HF_USERNAME = "yusufbukarmaina"
    HF_FLORENCE_REPO = f"{HF_USERNAME}/beaker-florence2"
    HF_QWEN_REPO = f"{HF_USERNAME}/beaker-qwen2-5vl"
    UPLOAD_TO_HF = False  # Set to True to upload


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def resize_image(img, max_size=512):
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    img = img.convert("RGB")
    w, h = img.size
    if w > max_size or h > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return img

def extract_volume(text):
    """Smart volume extraction"""
    if not text:
        return 0.0
    text = str(text).lower()
    
    # Look for number before "ml"
    for pat in [r"(\d+\.?\d*)\s*ml", r"(\d+\.?\d*)\s*milliliter"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    
    # Fallback: last valid number
    nums = re.findall(r"\d+\.?\d*", text)
    if nums:
        valid = [float(n) for n in nums if float(n) <= 5000]
        if valid:
            return valid[-1]
    return 0.0


# ============================================================================
# MEMORY-EFFICIENT DATA PROCESSOR
# ============================================================================

class LazyDatasetProcessor:
    """
    Stores only indices and labels, NOT full images.
    Images loaded on-the-fly during training.
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = None
    
    def load_indices(self):
        """Load indices only (not images) to save RAM"""
        print(f"📥 Loading dataset indices (lazy mode)...")
        print(f"   Target: {self.cfg.TRAIN_SAMPLES} train, {self.cfg.VAL_SAMPLES} val, {self.cfg.TEST_SAMPLES} test")
        
        ds = load_dataset(
            self.cfg.HF_DATASET_NAME,
            split="train",
            streaming=True
        ).shuffle(seed=42, buffer_size=500)
        
        # Store ONLY indices and labels
        train_idx, val_idx, test_idx = [], [], []
        train_lbl, val_lbl, test_lbl = [], [], []
        
        idx = 0
        for example in ds:
            if 'image' not in example:
                continue
            
            # Get label
            label = example.get('volume_label', '')
            if not label and 'volume_ml' in example:
                label = f"{example['volume_ml']} mL"
            
            # Store index + label (NOT the image!)
            if len(train_idx) < self.cfg.TRAIN_SAMPLES:
                train_idx.append(idx)
                train_lbl.append(label)
            elif len(val_idx) < self.cfg.VAL_SAMPLES:
                val_idx.append(idx)
                val_lbl.append(label)
            elif len(test_idx) < self.cfg.TEST_SAMPLES:
                test_idx.append(idx)
                test_lbl.append(label)
            else:
                break
            
            idx += 1
            
            if idx % 200 == 0:
                print(f"  ✓ {idx} samples (train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)})")
        
        print(f"✅ Loaded: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
        
        # Create lightweight datasets
        train_data = Dataset.from_dict({'index': train_idx, 'label': train_lbl})
        val_data = Dataset.from_dict({'index': val_idx, 'label': val_lbl})
        test_data = Dataset.from_dict({'index': test_idx, 'label': test_lbl})
        
        # Load full dataset for image access
        print("📥 Loading full dataset for image access...")
        self.dataset = load_dataset(self.cfg.HF_DATASET_NAME, split="train")
        print(f"✅ Dataset ready ({len(self.dataset)} total samples)")
        
        return train_data, val_data, test_data
    
    def get_image(self, idx):
        """Load single image on-demand"""
        return self.dataset[idx]['image']
    
    def save_test_images(self, test_data):
        """Export test images with metadata"""
        print(f"\n💾 Saving {len(test_data)} test images to {self.cfg.TEST_IMAGES_DIR}...")
        os.makedirs(self.cfg.TEST_IMAGES_DIR, exist_ok=True)
        
        metadata = []
        for i in range(len(test_data)):
            try:
                row = test_data[i]
                img = self.get_image(row['index'])
                img = resize_image(img, self.cfg.MAX_IMAGE_SIZE)
                
                vol = extract_volume(row['label'])
                filename = f"test_{i:04d}_vol_{vol:.1f}mL.jpg"
                
                img.save(os.path.join(self.cfg.TEST_IMAGES_DIR, filename), quality=95)
                
                metadata.append({
                    'index': i,
                    'filename': filename,
                    'volume_ml': vol,
                    'gt_text': row['label']
                })
                
                if (i + 1) % 50 == 0:
                    print(f"  Saved {i + 1}/{len(test_data)}")
            except Exception as e:
                print(f"  ⚠️ Error {i}: {e}")
        
        # Save metadata
        with open(os.path.join(self.cfg.TEST_IMAGES_DIR, 'test_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Test images saved to {self.cfg.TEST_IMAGES_DIR}")


# ============================================================================
# COLLATORS
# ============================================================================

def florence_collate(features):
    return {
        "pixel_values": torch.stack([f["pixel_values"] for f in features]),
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [f["input_ids"] for f in features], batch_first=True, padding_value=0
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [f["attention_mask"] for f in features], batch_first=True, padding_value=0
        ),
        "labels": torch.nn.utils.rnn.pad_sequence(
            [f["labels"] for f in features], batch_first=True, padding_value=-100
        ),
    }


# ============================================================================
# TRAINERS
# ============================================================================

class FlorenceTrainer:
    def __init__(self, cfg, data_processor):
        self.cfg = cfg
        self.dp = data_processor
        self.model = None
        self.processor = None
    
    def setup(self):
        print(f"\n🤖 Loading Florence-2: {self.cfg.FLORENCE_MODEL}")
        clear_memory()
        
        self.processor = AutoProcessor.from_pretrained(
            self.cfg.FLORENCE_MODEL, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.FLORENCE_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=self.cfg.LORA_R,
            lora_alpha=self.cfg.LORA_ALPHA,
            target_modules=self.cfg.LORA_TARGETS,
            lora_dropout=self.cfg.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def train(self, train_data, val_data):
        print("\n🚀 Training Florence-2...")
        self.setup()
        
        class LazyDataset(torch.utils.data.Dataset):
            def __init__(self, data, dp, processor, max_size):
                self.data = data
                self.dp = dp
                self.processor = processor
                self.max_size = max_size
                self.pad_id = processor.tokenizer.pad_token_id
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, i):
                row = self.data[i]
                img = self.dp.get_image(row['index'])  # Load on-demand
                img = resize_image(img, self.max_size)
                label = row['label'] or "0 mL"
                
                inputs = self.processor(
                    images=img,
                    text="<VQA>What is the volume?",
                    return_tensors="pt",
                    padding=True
                )
                
                answer_ids = self.processor.tokenizer(
                    str(label),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=64
                )['input_ids'].squeeze(0).clone()
                
                if self.pad_id:
                    answer_ids[answer_ids == self.pad_id] = -100
                
                return {
                    'pixel_values': inputs['pixel_values'].squeeze(0),
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': answer_ids
                }
        
        train_dataset = LazyDataset(train_data, self.dp, self.processor, self.cfg.MAX_IMAGE_SIZE)
        val_dataset = LazyDataset(val_data, self.dp, self.processor, self.cfg.MAX_IMAGE_SIZE)
        
        training_args = TrainingArguments(
            output_dir=f"{self.cfg.OUTPUT_DIR}/florence2",
            num_train_epochs=self.cfg.NUM_EPOCHS,
            per_device_train_batch_size=self.cfg.BATCH_SIZE,
            gradient_accumulation_steps=self.cfg.GRADIENT_ACCUMULATION,
            learning_rate=self.cfg.LEARNING_RATE,
            warmup_steps=self.cfg.WARMUP_STEPS,
            logging_steps=self.cfg.LOGGING_STEPS,
            save_steps=self.cfg.SAVE_STEPS,
            eval_steps=self.cfg.EVAL_STEPS,
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=2,
            fp16=self.cfg.FP16,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
            gradient_checkpointing=self.cfg.GRADIENT_CHECKPOINTING,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=florence_collate
        )
        
        trainer.train()
        
        final_dir = f"{self.cfg.OUTPUT_DIR}/florence2_final"
        trainer.save_model(final_dir)
        self.processor.save_pretrained(final_dir)
        print(f"✅ Florence-2 saved to {final_dir}")
        
        return final_dir


class QwenTrainer:
    def __init__(self, cfg, data_processor):
        self.cfg = cfg
        self.dp = data_processor
        self.model = None
        self.processor = None
    
    def setup(self):
        print(f"\n🤖 Loading Qwen2.5-VL: {self.cfg.QWEN_MODEL}")
        clear_memory()
        
        self.processor = AutoProcessor.from_pretrained(
            self.cfg.QWEN_MODEL, trust_remote_code=True
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.cfg.QWEN_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=self.cfg.LORA_R,
            lora_alpha=self.cfg.LORA_ALPHA,
            target_modules=self.cfg.LORA_TARGETS,
            lora_dropout=self.cfg.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def train(self, train_data, val_data):
        print("\n🚀 Training Qwen2.5-VL...")
        self.setup()
        
        class LazyDataset(torch.utils.data.Dataset):
            def __init__(self, data, dp, processor, max_size):
                self.data = data
                self.dp = dp
                self.processor = processor
                self.max_size = max_size
            
            def __len__(self):
                return len(self.data)

            class LazyDataset(torch.utils.data.Dataset):
    def __init__(self, data, dp, processor, max_size):
        self.data = data
        self.dp = dp
        self.processor = processor
        self.max_size = max_size
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        row = self.data[i]
        img = self.dp.get_image(row['index'])
        img = resize_image(img, self.max_size)
        label = row['label'] or "0 mL"
        
        messages = [
            {
                "role": "system",
                "content": "Reply with ONLY the volume and unit, e.g. '150 mL'."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "What is the volume in mL?"}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(label)}]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        # ✅ FIXED: Dynamic padding only
        inputs = self.processor(
            text=[text],
            images=[img],
            return_tensors="pt",
            padding=True,  # Dynamic padding
        )
        
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = inputs['input_ids'].clone()
        
        return inputs
        
        train_dataset = LazyDataset(train_data, self.dp, self.processor, self.cfg.MAX_IMAGE_SIZE)
        val_dataset = LazyDataset(val_data, self.dp, self.processor, self.cfg.MAX_IMAGE_SIZE)
        
        training_args = TrainingArguments(
            output_dir=f"{self.cfg.OUTPUT_DIR}/qwen2_5vl",
            num_train_epochs=self.cfg.NUM_EPOCHS,
            per_device_train_batch_size=self.cfg.BATCH_SIZE,
            gradient_accumulation_steps=self.cfg.GRADIENT_ACCUMULATION,
            learning_rate=self.cfg.LEARNING_RATE,
            warmup_steps=self.cfg.WARMUP_STEPS,
            logging_steps=self.cfg.LOGGING_STEPS,
            save_steps=self.cfg.SAVE_STEPS,
            eval_steps=self.cfg.EVAL_STEPS,
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=2,
            fp16=self.cfg.FP16,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
            gradient_checkpointing=self.cfg.GRADIENT_CHECKPOINTING,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        trainer.train()
        
        final_dir = f"{self.cfg.OUTPUT_DIR}/qwen2_5vl_final"
        trainer.save_model(final_dir)
        self.processor.save_pretrained(final_dir)
        print(f"✅ Qwen2.5-VL saved to {final_dir}")
        
        return final_dir


# ============================================================================
# EVALUATOR
# ============================================================================

class Evaluator:
    def __init__(self, cfg, dp):
        self.cfg = cfg
        self.dp = dp
    
    def evaluate(self, model, processor, test_data, name):
        print(f"\n📊 Evaluating {name} on {len(test_data)} samples...")
        model.eval()
        clear_memory()
        
        predictions, ground_truth = [], []
        
        with torch.no_grad():
            for i in range(len(test_data)):
                try:
                    row = test_data[i]
                    img = self.dp.get_image(row['index'])
                    img = resize_image(img, self.cfg.MAX_IMAGE_SIZE)
                    
                    gt = extract_volume(row['label'])
                    ground_truth.append(gt)
                    
                    if 'florence' in name.lower():
                        # Florence-2 inference
                        inputs = processor(
                            images=img,
                            text="<VQA>What is the volume?",
                            return_tensors="pt"
                        )
                        
                        # Fix dtype
                        dtype = next(model.parameters()).dtype
                        inputs = {
                            k: v.to(model.device).to(dtype) if v.dtype.is_floating_point 
                               else v.to(model.device)
                            for k, v in inputs.items()
                        }
                        
                        gen_ids = model.generate(**inputs, max_new_tokens=30)
                        text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
                    
                    else:
                        # Qwen inference
                        messages = [
                            {
                                "role": "system",
                                "content": "Reply with ONLY the volume and unit, e.g. '150 mL'."
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": img},
                                    {"type": "text", "text": "What is the volume in mL?"}
                                ]
                            }
                        ]
                        
                        text_prompt = processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        inputs = processor(
                            text=[text_prompt],
                            images=[img],
                            return_tensors="pt"
                        ).to(model.device)
                        
                        gen_ids = model.generate(**inputs, max_new_tokens=20)
                        text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
                    
                    pred = extract_volume(text)
                    predictions.append(pred)
                    
                    if (i + 1) % 50 == 0:
                        print(f"  {i + 1}/{len(test_data)}")
                
                except Exception as e:
                    print(f"  ⚠️ Error {i}: {e}")
                    predictions.append(0.0)
        
        # Calculate metrics
        p = np.array(predictions)
        g = np.array(ground_truth)
        
        mae = mean_absolute_error(g, p)
        rmse = np.sqrt(mean_squared_error(g, p))
        r2 = r2_score(g, p)
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'predictions': p.tolist(),
            'ground_truth': g.tolist()
        }
        
        print(f"📈 {name}: MAE={mae:.2f} mL, RMSE={rmse:.2f} mL, R²={r2:.4f}")
        
        # Plot
        self._plot(g, p, name)
        
        return metrics
    
    def _plot(self, g, p, name):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Scatter
        axes[0].scatter(g, p, alpha=0.4, s=20)
        lo, hi = min(g.min(), p.min()), max(g.max(), p.max())
        axes[0].plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect')
        axes[0].set_xlabel('Ground Truth (mL)')
        axes[0].set_ylabel('Prediction (mL)')
        axes[0].set_title(f'{name} - Pred vs GT', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Error histogram
        err = p - g
        axes[1].hist(err, bins=30, edgecolor='black', alpha=0.75)
        axes[1].axvline(0, color='red', lw=2, linestyle='--', label='Zero')
        axes[1].axvline(err.mean(), color='lime', lw=2, linestyle='--',
                       label=f'Mean: {err.mean():.1f}')
        axes[1].set_xlabel('Error (mL)')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'{name} - Error Dist', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Residuals
        axes[2].scatter(g, err, alpha=0.4, s=20)
        axes[2].axhline(0, color='red', lw=2, linestyle='--')
        axes[2].set_xlabel('Ground Truth (mL)')
        axes[2].set_ylabel('Residual (mL)')
        axes[2].set_title(f'{name} - Residuals', fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = f"{self.cfg.OUTPUT_DIR}/{name.replace(' ', '_')}_eval.png"
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  📊 Plot saved: {path}")


# ============================================================================
# HUGGINGFACE PUSH
# ============================================================================

def push_to_hub(cfg, florence_dir, qwen_dir):
    """Upload models to HuggingFace Hub"""
    print("\n" + "="*70)
    print("📤 PUSHING TO HUGGINGFACE HUB")
    print("="*70)
    
    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi()
        
        # Florence-2
        print(f"\n→ Uploading Florence-2 to {cfg.HF_FLORENCE_REPO}")
        try:
            create_repo(cfg.HF_FLORENCE_REPO, repo_type="model", exist_ok=True)
        except:
            pass
        
        # Create README
        readme = f"""# Beaker Volume Estimator - Florence-2

Fine-tuned on {cfg.TRAIN_SAMPLES} beaker images for liquid volume estimation.

## Usage
```python
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

processor = AutoProcessor.from_pretrained("{cfg.HF_FLORENCE_REPO}", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("{cfg.HF_FLORENCE_REPO}", trust_remote_code=True)

from PIL import Image
image = Image.open("beaker.jpg")
inputs = processor(images=image, text="<VQA>What is the volume?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=30)
print(processor.batch_decode(outputs, skip_special_tokens=True)[0])
```
"""
        with open(os.path.join(florence_dir, "README.md"), 'w') as f:
            f.write(readme)
        
        api.upload_folder(
            folder_path=florence_dir,
            repo_id=cfg.HF_FLORENCE_REPO,
            repo_type="model"
        )
        print(f"✅ Florence-2: https://huggingface.co/{cfg.HF_FLORENCE_REPO}")
        
        # Qwen
        print(f"\n→ Uploading Qwen2.5-VL to {cfg.HF_QWEN_REPO}")
        try:
            create_repo(cfg.HF_QWEN_REPO, repo_type="model", exist_ok=True)
        except:
            pass
        
        readme_q = f"""# Beaker Volume Estimator - Qwen2.5-VL

Fine-tuned on {cfg.TRAIN_SAMPLES} beaker images for liquid volume estimation.

## Usage
```python
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

processor = AutoProcessor.from_pretrained("{cfg.HF_QWEN_REPO}", trust_remote_code=True)
model = Qwen2VLForConditionalGeneration.from_pretrained("{cfg.HF_QWEN_REPO}", trust_remote_code=True)

from PIL import Image
image = Image.open("beaker.jpg")
messages = [{{"role": "user", "content": [
    {{"type": "image", "image": image}},
    {{"type": "text", "text": "What is the volume in mL?"}}
]}}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(processor.batch_decode(outputs, skip_special_tokens=True)[0])
```
"""
        with open(os.path.join(qwen_dir, "README.md"), 'w') as f:
            f.write(readme_q)
        
        api.upload_folder(
            folder_path=qwen_dir,
            repo_id=cfg.HF_QWEN_REPO,
            repo_type="model"
        )
        print(f"✅ Qwen2.5-VL: https://huggingface.co/{cfg.HF_QWEN_REPO}")
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🚀 COMPLETE Memory-Efficient Training Pipeline")
    print("="*70)
    print("✅ Lazy loading (1000+ samples)")
    print("✅ Test image export")
    print("✅ Full evaluation with plots")
    print("✅ Metrics (MAE, RMSE, R²)")
    print("✅ HuggingFace push")
    print("="*70)
    
    cfg = Config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.TEST_IMAGES_DIR, exist_ok=True)
    
    # Load data (lazy mode)
    dp = LazyDatasetProcessor(cfg)
    train_data, val_data, test_data = dp.load_indices()
    
    # Save test images
    dp.save_test_images(test_data)
    
    # Train Florence-2
    print("\n" + "="*70)
    print("FLORENCE-2 TRAINING")
    print("="*70)
    ft = FlorenceTrainer(cfg, dp)
    florence_path = ft.train(train_data, val_data)
    clear_memory()
    
    # Train Qwen
    print("\n" + "="*70)
    print("QWEN2.5-VL TRAINING")
    print("="*70)
    qt = QwenTrainer(cfg, dp)
    qwen_path = qt.train(train_data, val_data)
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    evaluator = Evaluator(cfg, dp)
    
    # Reload Florence for eval
    ft_eval = FlorenceTrainer(cfg, dp)
    ft_eval.setup()
    florence_metrics = evaluator.evaluate(ft_eval.model, ft_eval.processor, test_data, "Florence-2")
    del ft_eval
    clear_memory()
    
    # Eval Qwen
    qwen_metrics = evaluator.evaluate(qt.model, qt.processor, test_data, "Qwen2.5-VL")
    
    # Save results
    results = {
        'florence2': florence_metrics,
        'qwen': qwen_metrics,
        'config': {
            'train_samples': cfg.TRAIN_SAMPLES,
            'val_samples': cfg.VAL_SAMPLES,
            'test_samples': cfg.TEST_SAMPLES,
            'epochs': cfg.NUM_EPOCHS,
            'learning_rate': cfg.LEARNING_RATE
        }
    }
    
    results_path = os.path.join(cfg.OUTPUT_DIR, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved: {results_path}")
    
    # Push to HuggingFace
    if cfg.UPLOAD_TO_HF:
        push_to_hub(cfg, florence_path, qwen_path)
    else:
        print("\n💡 To upload to HuggingFace:")
        print("   1. Set UPLOAD_TO_HF = True in Config")
        print("   2. Run: huggingface-cli login")
        print("   3. Re-run this script")
    
    # Summary
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"Florence-2:  {florence_path}")
    print(f"Qwen2.5-VL:  {qwen_path}")
    print(f"Results:     {results_path}")
    print(f"Test images: {cfg.TEST_IMAGES_DIR}")
    print(f"\nFlorence-2: MAE={florence_metrics['mae']:.2f}, R²={florence_metrics['r2']:.4f}")
    print(f"Qwen2.5-VL: MAE={qwen_metrics['mae']:.2f}, R²={qwen_metrics['r2']:.4f}")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        clear_memory()

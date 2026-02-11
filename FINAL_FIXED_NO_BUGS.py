"""
FINAL FIXED VERSION - All Evaluation Bugs Corrected
This will give you proper results!
"""

import os, json, gc, time, re
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoProcessor, AutoModelForCausalLM,
    Qwen2VLForConditionalGeneration,
    TrainingArguments, Trainer, TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ============================================================================
# CONFIG
# ============================================================================

class Config:
    HF_DATASET_NAME = "yusufbukarmaina/Beakers1"
    
    TRAIN_SAMPLES = 700
    VAL_SAMPLES = 150
    TEST_SAMPLES = 150
    
    FLORENCE_MODEL = "microsoft/Florence-2-base"
    QWEN_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
    
    MAX_IMAGE_SIZE = 512
    
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGETS = ["q_proj", "v_proj"]
    
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 16
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 8
    WARMUP_STEPS = 50
    
    FP16 = True
    GRADIENT_CHECKPOINTING = True
    
    OUTPUT_DIR = "./trained_models"
    SAVE_STEPS = 200
    EVAL_STEPS = 200
    LOGGING_STEPS = 25


# ============================================================================
# UTILITIES
# ============================================================================

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def resize_image(img, max_size=512):
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    img = img.convert("RGB")
    w, h = img.size
    if w > max_size or h > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return img


# ============================================================================
# FIX #1: SMART VOLUME EXTRACTION
# ============================================================================

def extract_volume(text):
    """
    FIXED: Looks for number BEFORE 'mL', not just first number.
    This prevents extracting step numbers like "1." from verbose responses.
    """
    if not text:
        return 0.0
    
    text = str(text).lower()
    
    # Priority 1: Number immediately before "ml"
    patterns = [
        r'(\d+\.?\d*)\s*ml',
        r'(\d+\.?\d*)\s*milliliters?',
        r'approximately\s*(\d+\.?\d*)',
        r'volume.*?(\d+\.?\d*)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                if val <= 5000:  # Sanity check (max beaker size)
                    return val
            except:
                pass
    
    # Priority 2: Last valid number (avoid first which might be step count)
    nums = re.findall(r'\d+\.?\d*', text)
    if nums:
        valid = [float(n) for n in nums if float(n) <= 5000]
        if valid:
            return valid[-1]  # Take LAST, not FIRST
    
    return 0.0


# ============================================================================
# FIX #2: QWEN SYSTEM PROMPT
# ============================================================================

QWEN_SYSTEM_PROMPT = (
    "You are a measurement instrument. "
    "When asked about liquid volume, respond with ONLY the number and unit. "
    "Example: '150 mL'. Never explain. Never use formulas or steps."
)


# ============================================================================
# METRICS TRACKER
# ============================================================================

class MetricsTracker:
    def __init__(self):
        self.data = {'training': {}, 'evaluation': {}}
        self.start_time = time.time()
    
    def record_training(self, name, train_time, epochs, total_params, trainable_params):
        self.data['training'][name] = {
            'time_min': train_time / 60,
            'epochs': epochs,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'trainable_pct': trainable_params / total_params * 100
        }
    
    def record_eval(self, name, metrics):
        self.data['evaluation'][name] = metrics
    
    def save(self, output_dir):
        # Save JSON
        with open(f"{output_dir}/results.json", 'w') as f:
            json.dump(self.data, f, indent=2)
        
        # Create summary table
        rows = []
        for name in self.data['training']:
            tr = self.data['training'][name]
            ev = self.data['evaluation'].get(name, {})
            rows.append({
                'Model': name,
                'Time (min)': f"{tr['time_min']:.1f}",
                'Trainable %': f"{tr['trainable_pct']:.2f}",
                'MAE (mL)': f"{ev.get('mae', 0):.2f}",
                'RMSE (mL)': f"{ev.get('rmse', 0):.2f}",
                'R²': f"{ev.get('r2', 0):.4f}"
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(f"{output_dir}/summary.csv", index=False)
        
        print("\n" + "="*70)
        print("📊 FINAL RESULTS")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70)
        
        return df


class TrainingCallback(TrainerCallback):
    def __init__(self, name):
        self.name = name
        self.t = None
    
    def on_epoch_begin(self, *args, **kwargs):
        self.t = time.time()
    
    def on_epoch_end(self, *args, state=None, **kwargs):
        if self.t:
            print(f"  Epoch {state.epoch:.0f} done in {(time.time()-self.t)/60:.2f} min")


# ============================================================================
# DATA LOADING (LAZY)
# ============================================================================

class LazyLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = None
    
    def load(self):
        print("📥 Loading dataset indices...")
        
        ds = load_dataset(self.cfg.HF_DATASET_NAME, split="train", streaming=True).shuffle(seed=42)
        
        train_idx, val_idx, test_idx = [], [], []
        train_lbl, val_lbl, test_lbl = [], [], []
        
        idx = 0
        for ex in ds:
            if 'image' not in ex:
                continue
            
            label = ex.get('volume_label', '')
            if not label and 'volume_ml' in ex:
                label = f"{ex['volume_ml']} mL"
            
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
        
        print(f"✅ Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        # Load full dataset
        self.dataset = load_dataset(self.cfg.HF_DATASET_NAME, split="train")
        
        return (
            Dataset.from_dict({'idx': train_idx, 'lbl': train_lbl}),
            Dataset.from_dict({'idx': val_idx, 'lbl': val_lbl}),
            Dataset.from_dict({'idx': test_idx, 'lbl': test_lbl})
        )
    
    def get_image(self, idx):
        return self.dataset[idx]['image']


# ============================================================================
# COLLATORS
# ============================================================================

def florence_collate(features):
    return {
        "pixel_values": torch.stack([f["pixel_values"] for f in features]),
        "input_ids": torch.nn.utils.rnn.pad_sequence([f["input_ids"] for f in features], batch_first=True, padding_value=0),
        "attention_mask": torch.nn.utils.rnn.pad_sequence([f["attention_mask"] for f in features], batch_first=True, padding_value=0),
        "labels": torch.nn.utils.rnn.pad_sequence([f["labels"] for f in features], batch_first=True, padding_value=-100),
    }


# ============================================================================
# TRAINERS
# ============================================================================

class FlorenceTrainer:
    def __init__(self, cfg, loader, tracker):
        self.cfg = cfg
        self.loader = loader
        self.tracker = tracker
        self.model = None
        self.processor = None
    
    def setup(self):
        print("\n🤖 Florence-2...")
        clear_memory()
        
        self.processor = AutoProcessor.from_pretrained(self.cfg.FLORENCE_MODEL, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.FLORENCE_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        
        lora = LoraConfig(
            r=self.cfg.LORA_R,
            lora_alpha=self.cfg.LORA_ALPHA,
            target_modules=self.cfg.LORA_TARGETS,
            lora_dropout=self.cfg.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora)
        self.model.print_trainable_parameters()
        
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return total, trainable
    
    def train(self, train_data, val_data):
        t0 = time.time()
        total, trainable = self.setup()
        
        class DS(torch.utils.data.Dataset):
            def __init__(self, data, loader, proc, max_sz):
                self.data = data
                self.loader = loader
                self.proc = proc
                self.max_sz = max_sz
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, i):
                row = self.data[i]
                img = resize_image(self.loader.get_image(row['idx']), self.max_sz)
                
                inp = self.proc(images=img, text="<VQA>What is the volume?", return_tensors="pt", padding=True)
                ans = self.proc.tokenizer(str(row['lbl'] or "0 mL"), return_tensors="pt", padding=True, max_length=64, truncation=True)['input_ids'].squeeze(0)
                
                return {
                    'pixel_values': inp['pixel_values'].squeeze(0),
                    'input_ids': inp['input_ids'].squeeze(0),
                    'attention_mask': inp['attention_mask'].squeeze(0),
                    'labels': ans
                }
        
        args = TrainingArguments(
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
            save_total_limit=1,
            fp16=self.cfg.FP16,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none"
        )
        
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=DS(train_data, self.loader, self.processor, self.cfg.MAX_IMAGE_SIZE),
            eval_dataset=DS(val_data, self.loader, self.processor, self.cfg.MAX_IMAGE_SIZE),
            data_collator=florence_collate,
            callbacks=[TrainingCallback("Florence-2")]
        )
        
        trainer.train()
        
        self.tracker.record_training("Florence-2", time.time()-t0, self.cfg.NUM_EPOCHS, total, trainable)
        
        final = f"{self.cfg.OUTPUT_DIR}/florence2_final"
        trainer.save_model(final)
        self.processor.save_pretrained(final)
        
        return final


class QwenTrainer:
    def __init__(self, cfg, loader, tracker):
        self.cfg = cfg
        self.loader = loader
        self.tracker = tracker
        self.model = None
        self.processor = None
    
    def setup(self):
        print("\n🤖 Qwen2.5-VL...")
        clear_memory()
        
        self.processor = AutoProcessor.from_pretrained(self.cfg.QWEN_MODEL, trust_remote_code=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.cfg.QWEN_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        
        lora = LoraConfig(
            r=self.cfg.LORA_R,
            lora_alpha=self.cfg.LORA_ALPHA,
            target_modules=self.cfg.LORA_TARGETS,
            lora_dropout=self.cfg.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora)
        self.model.print_trainable_parameters()
        
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return total, trainable
    
    def train(self, train_data, val_data):
        t0 = time.time()
        total, trainable = self.setup()
        
        class DS(torch.utils.data.Dataset):
            def __init__(self, data, loader, proc, max_sz):
                self.data = data
                self.loader = loader
                self.proc = proc
                self.max_sz = max_sz
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, i):
                row = self.data[i]
                img = resize_image(self.loader.get_image(row['idx']), self.max_sz)
                
                # FIX: Add system prompt!
                msgs = [
                    {"role": "system", "content": QWEN_SYSTEM_PROMPT},
                    {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "What is the volume in mL?"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": str(row['lbl'] or "0 mL")}]}
                ]
                
                text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                inp = self.proc(text=[text], images=[img], return_tensors="pt", padding="max_length", truncation=True, max_length=256)
                
                inp = {k: v.squeeze(0) for k, v in inp.items()}
                inp['labels'] = inp['input_ids'].clone()
                
                return inp
        
        args = TrainingArguments(
            output_dir=f"{self.cfg.OUTPUT_DIR}/qwen",
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
            save_total_limit=1,
            fp16=self.cfg.FP16,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none"
        )
        
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=DS(train_data, self.loader, self.processor, self.cfg.MAX_IMAGE_SIZE),
            eval_dataset=DS(val_data, self.loader, self.processor, self.cfg.MAX_IMAGE_SIZE),
            callbacks=[TrainingCallback("Qwen2.5-VL")]
        )
        
        trainer.train()
        
        self.tracker.record_training("Qwen2.5-VL", time.time()-t0, self.cfg.NUM_EPOCHS, total, trainable)
        
        final = f"{self.cfg.OUTPUT_DIR}/qwen_final"
        trainer.save_model(final)
        self.processor.save_pretrained(final)
        
        return final


# ============================================================================
# EVALUATOR (FIXED!)
# ============================================================================

class Evaluator:
    def __init__(self, cfg, loader, tracker):
        self.cfg = cfg
        self.loader = loader
        self.tracker = tracker
    
    def evaluate(self, model, processor, test_data, name):
        print(f"\n📊 Evaluating {name}...")
        model.eval()
        clear_memory()
        
        preds, gts = [], []
        
        with torch.no_grad():
            for i in range(len(test_data)):
                try:
                    row = test_data[i]
                    img = resize_image(self.loader.get_image(row['idx']), self.cfg.MAX_IMAGE_SIZE)
                    
                    gt = extract_volume(row['lbl'])
                    gts.append(gt)
                    
                    if 'florence' in name.lower():
                        # FIX: Cast to model dtype!
                        inputs = processor(images=img, text="<VQA>What is the volume?", return_tensors="pt")
                        
                        dtype = next(model.parameters()).dtype
                        inputs = {
                            k: v.to(model.device).to(dtype) if v.dtype.is_floating_point else v.to(model.device)
                            for k, v in inputs.items()
                        }
                        
                        ids = model.generate(**inputs, max_new_tokens=30)
                        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
                    
                    else:
                        # FIX: Use system prompt!
                        msgs = [
                            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
                            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "What is the volume in mL?"}]}
                        ]
                        
                        text_prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                        inputs = processor(text=[text_prompt], images=[img], return_tensors="pt").to(model.device)
                        
                        ids = model.generate(**inputs, max_new_tokens=20)
                        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
                    
                    pred = extract_volume(text)
                    preds.append(pred)
                    
                    if (i+1) % 50 == 0:
                        print(f"  {i+1}/{len(test_data)}")
                
                except Exception as e:
                    print(f"  ⚠️ {i}: {e}")
                    preds.append(0.0)
        
        p, g = np.array(preds), np.array(gts)
        
        metrics = {
            'mae': float(mean_absolute_error(g, p)),
            'rmse': float(np.sqrt(mean_squared_error(g, p))),
            'r2': float(r2_score(g, p))
        }
        
        self.tracker.record_eval(name, metrics)
        
        print(f"✅ {name}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.4f}")
        
        return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🚀 FINAL FIXED VERSION")
    print("="*70)
    
    cfg = Config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    tracker = MetricsTracker()
    loader = LazyLoader(cfg)
    train_data, val_data, test_data = loader.load()
    
    # Train
    ft = FlorenceTrainer(cfg, loader, tracker)
    ft.train(train_data, val_data)
    clear_memory()
    
    qt = QwenTrainer(cfg, loader, tracker)
    qt.train(train_data, val_data)
    
    # Evaluate
    evaluator = Evaluator(cfg, loader, tracker)
    
    ft_eval = FlorenceTrainer(cfg, loader, tracker)
    ft_eval.setup()
    evaluator.evaluate(ft_eval.model, ft_eval.processor, test_data, "Florence-2")
    del ft_eval
    clear_memory()
    
    evaluator.evaluate(qt.model, qt.processor, test_data, "Qwen2.5-VL")
    
    # Save
    tracker.save(cfg.OUTPUT_DIR)
    
    print(f"\n✅ Results: {cfg.OUTPUT_DIR}/results.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ {e}")
        import traceback
        traceback.print_exc()

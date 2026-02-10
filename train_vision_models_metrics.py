"""
Complete Training Pipeline — Fixed Metrics Tracker
Fixes all evaluation bugs, adds HuggingFace push, and correct overfitting analysis.

ROOT CAUSES of catastrophic metrics (MAE=487471, R²=−21 billion):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BUG 1 — Florence-2 dtype: processor returns float32 pixel_values,
         model weights are float16 → silent NaN/garbage output.
         Fix: cast pixel_values to model dtype before generate().

BUG 2 — Qwen no system prompt: default "helpful assistant" persona
         outputs long formulas. extract_volume finds first number
         e.g. from "1. Volume (mL) = ..." → extracts "1" not the answer.
         Fix: system prompt forces "X mL" only output.

BUG 3 — extract_volume too greedy: r"\\d+\\.?\\d*" picks the FIRST
         digit string in the full response, not the answer number.
         Fix: look for a number followed by mL / at end of string.

BUG 4 — NOT overfitting (yet): 5 epochs on 500 samples is severe
         underfitting for vision-language models. Both metrics show
         the models haven't converged, not that they've memorised.
         Fix: 10+ epochs, 1000 samples minimum, lower LR, more LoRA.
"""

import os, json, gc, time, re
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoProcessor, AutoModelForCausalLM,
    Qwen2VLForConditionalGeneration,
    TrainingArguments, Trainer,
    EarlyStoppingCallback, TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score, mean_absolute_percentage_error,
)
from PIL import Image
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    HF_DATASET_NAME = "yusufbukarmaina/Beakers1"
    STREAMING = True

    # ── Sample sizes (increase from 500 to reduce underfitting) ────────────
    TRAIN_SAMPLES = 1000   # was 500 — more data = less underfitting
    VAL_SAMPLES   = 300
    TEST_SAMPLES  = 300

    FLORENCE_MODEL = "microsoft/Florence-2-base"
    QWEN_MODEL     = "Qwen/Qwen2-VL-2B-Instruct"

    MAX_IMAGE_SIZE = 512

    # ── LoRA ────────────────────────────────────────────────────────────────
    LORA_R       = 16       # was 8 — larger rank captures more detail
    LORA_ALPHA   = 32       # keep 2× LORA_R
    LORA_DROPOUT = 0.05
    LORA_TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj"]  # more targets

    # ── Training ────────────────────────────────────────────────────────────
    BATCH_SIZE            = 1
    GRADIENT_ACCUMULATION = 16   # effective batch = 16
    LEARNING_RATE         = 1e-4  # was 2e-4 — lower to avoid overshooting
    NUM_EPOCHS            = 10    # was 5 — more epochs for convergence
    WARMUP_STEPS          = 100   # was 50
    MAX_LENGTH            = 256

    FP16                  = True
    GRADIENT_CHECKPOINTING = True

    OUTPUT_DIR      = "./trained_models"
    TEST_IMAGES_DIR = "/FQ/test_images"
    SAVE_STEPS      = 500
    EVAL_STEPS      = 500
    LOGGING_STEPS   = 25

    # ── HuggingFace Hub ─────────────────────────────────────────────────────
    HF_USERNAME   = "yusufbukarmaina"   # your HF username
    HF_FLORENCE_REPO = f"{HF_USERNAME}/beaker-florence2"
    HF_QWEN_REPO     = f"{HF_USERNAME}/beaker-qwen2-5vl"


# ============================================================================
# HELPERS
# ============================================================================

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()

def print_gpu(prefix=""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1e9
        r = torch.cuda.memory_reserved()  / 1e9
        print(f"  {prefix}GPU: {a:.2f}GB alloc / {r:.2f}GB reserved")

def resize_image(img, max_size=512):
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    img = img.convert("RGB")
    w, h = img.size
    if w > max_size or h > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
    return img


# ── FIX 3: smart extract_volume ─────────────────────────────────────────────
def extract_volume(text: str) -> float:
    """
    Robustly parse a mL value from model output.
    Prioritises: number immediately before 'mL', then last number in text.
    Avoids picking up step counts / loss values / token IDs.
    """
    if not text:
        return 0.0
    text = str(text)

    # 1. Explicit unit: "250 mL" / "250mL" / "250 ml" / "250 milliliters"
    for pat in [r"(\d+\.?\d*)\s*mL", r"(\d+\.?\d*)\s*ml",
                r"(\d+\.?\d*)\s*milliliter"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))

    # 2. Last numeric token in the response (avoid first token = step count)
    nums = re.findall(r"\d+\.?\d*", text)
    if nums:
        # Filter obvious non-volumes (>5000 mL is unrealistic for a beaker)
        valid = [float(n) for n in nums if float(n) <= 5000]
        if valid:
            return valid[-1]   # last valid number, not first
    return 0.0

def get_gt(example):
    label = example.get("volume_label", "")
    if label: return str(label)
    ml = example.get("volume_ml")
    if ml is not None: return f"{ml} mL"
    return ""


# ============================================================================
# METRICS TRACKER
# ============================================================================

class MetricsTracker:
    def __init__(self):
        self.data    = {}
        self.gpu_log = []
        self.t0      = None

    def start(self, cfg):
        self.t0 = time.time()
        self.data["experiment"] = {
            "date":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "gpu":        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "total_vram": torch.cuda.get_device_properties(0).total_memory/1e9 if torch.cuda.is_available() else 0,
            "pytorch":    torch.__version__,
        }

    def gpu(self, phase):
        if not torch.cuda.is_available(): return 0,0
        a = torch.cuda.memory_allocated()/1e9
        r = torch.cuda.max_memory_allocated()/1e9
        self.gpu_log.append({"t": time.time()-self.t0, "phase": phase, "alloc": a, "peak": r})
        return a, r

    def record_train(self, name, secs, epochs, total, trainable):
        self.data.setdefault("training", {})[name] = {
            "time_s": secs, "time_min": secs/60, "epochs": epochs,
            "total_params": total, "trainable_params": trainable,
            "trainable_pct": trainable/total*100,
        }

    def record_eval(self, name, metrics):
        self.data.setdefault("evaluation", {})[name] = metrics

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        # JSON
        j = os.path.join(out_dir, "complete_metrics.json")
        with open(j,"w") as f: json.dump(self.data, f, indent=2)
        # GPU CSV
        if self.gpu_log:
            pd.DataFrame(self.gpu_log).to_csv(
                os.path.join(out_dir, "gpu_history.csv"), index=False)
        # Summary table
        rows = []
        for name in self.data.get("training",{}):
            tr = self.data["training"][name]
            ev = self.data.get("evaluation",{}).get(name,{})
            gps = [g["peak"] for g in self.gpu_log if name.lower() in g["phase"].lower()]
            rows.append({
                "Model":            name,
                "Train Time (min)": f"{tr['time_min']:.1f}",
                "Parameters":       f"{tr['total_params']:,}",
                "Trainable %":      f"{tr['trainable_pct']:.2f}",
                "MAE (mL)":         f"{ev.get('mae',float('nan')):.2f}",
                "RMSE (mL)":        f"{ev.get('rmse',float('nan')):.2f}",
                "R²":               f"{ev.get('r2',float('nan')):.4f}",
                "MAPE (%)":         f"{ev.get('mape',float('nan')):.2f}",
                "Peak GPU (GB)":    f"{max(gps):.2f}" if gps else "N/A",
            })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, "summary_table.csv"), index=False)
        with open(os.path.join(out_dir, "summary_table.tex"),"w") as f:
            f.write(df.to_latex(index=False))
        print(f"\n📁 Metrics saved to {out_dir}/")
        return df


# ============================================================================
# CALLBACK
# ============================================================================

class MetricsCallback(TrainerCallback):
    def __init__(self, tracker, name):
        self.tracker, self.name, self.t = tracker, name, None
    def on_epoch_begin(self, *a, state=None, **kw):
        self.t = time.time()
        self.tracker.gpu(f"{self.name}_epoch_{state.epoch}_begin")
    def on_epoch_end(self, *a, state=None, **kw):
        if self.t:
            print(f"  ↳ Epoch {state.epoch:.0f} done in {(time.time()-self.t)/60:.1f} min")
        self.tracker.gpu(f"{self.name}_epoch_{state.epoch}_end")


# ============================================================================
# DATA
# ============================================================================

class DatasetProcessor:
    def __init__(self, cfg): self.cfg = cfg

    def load(self):
        print("📥  Streaming", self.cfg.HF_DATASET_NAME)
        ds = load_dataset(self.cfg.HF_DATASET_NAME,
                          split="train", streaming=True).shuffle(seed=42)
        train, val, test = [], [], []
        for ex in ds:
            if "image" not in ex: continue
            if len(train) < self.cfg.TRAIN_SAMPLES: train.append(ex)
            elif len(val) < self.cfg.VAL_SAMPLES:   val.append(ex)
            elif len(test) < self.cfg.TEST_SAMPLES: test.append(ex)
            else: break
            n = len(train)+len(val)+len(test)
            if n % 200 == 0:
                print(f"  ✓ {n} — {len(train)}/{len(val)}/{len(test)}")
        print(f"✅  {len(train)} train / {len(val)} val / {len(test)} test")
        return train, val, test

    def save_test_images(self, test, out):
        os.makedirs(out, exist_ok=True)
        meta = []
        print(f"\n💾  Saving {len(test)} test images → {out}")
        for i, ex in enumerate(test):
            try:
                img = resize_image(ex["image"], self.cfg.MAX_IMAGE_SIZE)
                gt  = get_gt(ex);  vol = extract_volume(gt)
                fn  = f"test_{i:04d}_vol_{vol:.1f}mL.jpg"
                img.save(os.path.join(out, fn), quality=95)
                meta.append({"index":i,"filename":fn,"volume_ml":vol,"gt_text":gt})
                if (i+1) % 50 == 0: print(f"  {i+1}/{len(test)}")
            except Exception as e:
                print(f"  ⚠️  {i}: {e}")
        with open(os.path.join(out,"test_metadata.json"),"w") as f:
            json.dump(meta, f, indent=2)
        print(f"✅  Test images saved")


# ============================================================================
# COLLATORS
# ============================================================================

def florence_collate(features):
    return {
        "pixel_values":   torch.stack([f["pixel_values"]   for f in features]),
        "input_ids":      torch.nn.utils.rnn.pad_sequence([f["input_ids"]      for f in features], batch_first=True, padding_value=0),
        "attention_mask": torch.nn.utils.rnn.pad_sequence([f["attention_mask"] for f in features], batch_first=True, padding_value=0),
        "labels":         torch.nn.utils.rnn.pad_sequence([f["labels"]         for f in features], batch_first=True, padding_value=-100),
    }

QWEN_SYS = (
    "You are a lab measurement instrument. "
    "When asked about liquid volume, respond with ONLY the number and unit. "
    "Example: '150 mL'. Never explain. Never use formulas."
)


# ============================================================================
# FLORENCE-2 TRAINER
# ============================================================================

class FlorenceTrainer:
    def __init__(self, cfg, tracker):
        self.cfg, self.tracker = cfg, tracker
        self.model = self.processor = None

    def setup(self):
        print(f"\n🤖  Loading {self.cfg.FLORENCE_MODEL}")
        clear_memory()
        self.processor = AutoProcessor.from_pretrained(
            self.cfg.FLORENCE_MODEL, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.FLORENCE_MODEL, trust_remote_code=True,
            torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,
        )
        if self.cfg.GRADIENT_CHECKPOINTING:
            self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        lora = LoraConfig(r=self.cfg.LORA_R, lora_alpha=self.cfg.LORA_ALPHA,
                          target_modules=self.cfg.LORA_TARGETS,
                          lora_dropout=self.cfg.LORA_DROPOUT,
                          bias="none", task_type="CAUSAL_LM")
        self.model = get_peft_model(self.model, lora)
        self.model.print_trainable_parameters(); print_gpu()
        total     = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable

    def train(self, train_data, val_data):
        t0 = time.time()
        total, trainable = self.setup()
        self.tracker.gpu("florence_loaded")

        pad_id = self.processor.tokenizer.pad_token_id
        max_sz  = self.cfg.MAX_IMAGE_SIZE

        class DS(torch.utils.data.Dataset):
            def __init__(self, data, proc):
                self.data, self.proc = data, proc
            def __len__(self): return len(self.data)
            def __getitem__(self, i):
                ex  = self.data[i]
                img = resize_image(ex["image"], max_sz)
                ans = get_gt(ex) or "0 mL"
                inp = self.proc(images=img, text="<VQA>What is the volume?",
                                return_tensors="pt", padding=True)
                lbl = self.proc.tokenizer(str(ans), return_tensors="pt",
                                          padding=True, truncation=True,
                                          max_length=64)["input_ids"].squeeze(0).clone()
                if pad_id: lbl[lbl == pad_id] = -100
                return {
                    "pixel_values":   inp["pixel_values"].squeeze(0),
                    "input_ids":      inp["input_ids"].squeeze(0),
                    "attention_mask": inp["attention_mask"].squeeze(0),
                    "labels":         lbl,
                }

        out = f"{self.cfg.OUTPUT_DIR}/florence2"
        os.makedirs(out, exist_ok=True)
        args = TrainingArguments(
            output_dir=out,
            num_train_epochs=self.cfg.NUM_EPOCHS,
            per_device_train_batch_size=self.cfg.BATCH_SIZE,
            per_device_eval_batch_size=self.cfg.BATCH_SIZE,
            gradient_accumulation_steps=self.cfg.GRADIENT_ACCUMULATION,
            learning_rate=self.cfg.LEARNING_RATE,
            warmup_steps=self.cfg.WARMUP_STEPS,
            logging_steps=self.cfg.LOGGING_STEPS,
            save_steps=self.cfg.SAVE_STEPS,
            eval_steps=self.cfg.EVAL_STEPS,
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=self.cfg.FP16,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="tensorboard",
            logging_dir=f"{out}/runs",
            gradient_checkpointing=self.cfg.GRADIENT_CHECKPOINTING,
            optim="adamw_torch",
            max_grad_norm=1.0,
        )
        trainer = Trainer(
            model=self.model, args=args,
            train_dataset=DS(train_data, self.processor),
            eval_dataset =DS(val_data,   self.processor),
            data_collator=florence_collate,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
                MetricsCallback(self.tracker, "Florence-2"),
            ],
        )
        trainer.train()
        self.tracker.record_train("Florence-2", time.time()-t0,
                                  self.cfg.NUM_EPOCHS, total, trainable)
        final = f"{self.cfg.OUTPUT_DIR}/florence2_final"
        trainer.save_model(final); self.processor.save_pretrained(final)
        print(f"✅  Florence-2 saved → {final}")
        clear_memory()
        return final


# ============================================================================
# QWEN2.5-VL TRAINER
# ============================================================================

class QwenTrainer:
    def __init__(self, cfg, tracker):
        self.cfg, self.tracker = cfg, tracker
        self.model = self.processor = None

    def setup(self):
        print(f"\n🤖  Loading {self.cfg.QWEN_MODEL}")
        clear_memory()
        self.processor = AutoProcessor.from_pretrained(
            self.cfg.QWEN_MODEL, trust_remote_code=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.cfg.QWEN_MODEL, trust_remote_code=True,
            torch_dtype=torch.float16, device_map="auto",
            low_cpu_mem_usage=True, max_memory={0: "22GB"},
        )
        if self.cfg.GRADIENT_CHECKPOINTING:
            self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        lora = LoraConfig(r=self.cfg.LORA_R, lora_alpha=self.cfg.LORA_ALPHA,
                          target_modules=self.cfg.LORA_TARGETS,
                          lora_dropout=self.cfg.LORA_DROPOUT,
                          bias="none", task_type="CAUSAL_LM")
        self.model = get_peft_model(self.model, lora)
        self.model.print_trainable_parameters(); print_gpu()
        total     = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable

    def train(self, train_data, val_data):
        t0 = time.time()
        total, trainable = self.setup()
        self.tracker.gpu("qwen_loaded")
        proc   = self.processor
        max_sz = self.cfg.MAX_IMAGE_SIZE

        class DS(torch.utils.data.Dataset):
            def __init__(self, data): self.data = data
            def __len__(self): return len(self.data)
            def __getitem__(self, i):
                ex  = self.data[i]
                img = resize_image(ex["image"], max_sz)
                ans = get_gt(ex) or "0 mL"
                msgs = [
                    {"role": "system", "content": QWEN_SYS},
                    {"role": "user", "content": [
                        {"type": "image", "image": img},
                        {"type": "text",  "text":  "What is the volume in mL?"},
                    ]},
                    {"role": "assistant", "content": [{"type": "text", "text": str(ans)}]},
                ]
                text = proc.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=False)
                inp  = proc(text=[text], images=[img], return_tensors="pt",
                            padding="max_length", truncation=True,
                            max_length=256)
                inp  = {k: v.squeeze(0) for k, v in inp.items()}
                inp["labels"] = inp["input_ids"].clone()
                return inp

        out = f"{self.cfg.OUTPUT_DIR}/qwen2_5vl"
        os.makedirs(out, exist_ok=True)
        args = TrainingArguments(
            output_dir=out,
            num_train_epochs=self.cfg.NUM_EPOCHS,
            per_device_train_batch_size=self.cfg.BATCH_SIZE,
            per_device_eval_batch_size=self.cfg.BATCH_SIZE,
            gradient_accumulation_steps=self.cfg.GRADIENT_ACCUMULATION,
            learning_rate=self.cfg.LEARNING_RATE,
            warmup_steps=self.cfg.WARMUP_STEPS,
            logging_steps=self.cfg.LOGGING_STEPS,
            save_steps=self.cfg.SAVE_STEPS,
            eval_steps=self.cfg.EVAL_STEPS,
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=self.cfg.FP16,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="tensorboard",
            logging_dir=f"{out}/runs",
            gradient_checkpointing=self.cfg.GRADIENT_CHECKPOINTING,
            optim="adamw_torch",
            max_grad_norm=0.5,
        )
        trainer = Trainer(
            model=self.model, args=args,
            train_dataset=DS(train_data),
            eval_dataset =DS(val_data),
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
                MetricsCallback(self.tracker, "Qwen2.5-VL"),
            ],
        )
        trainer.train()
        self.tracker.record_train("Qwen2.5-VL", time.time()-t0,
                                  self.cfg.NUM_EPOCHS, total, trainable)
        final = f"{self.cfg.OUTPUT_DIR}/qwen2_5vl_final"
        trainer.save_model(final); self.processor.save_pretrained(final)
        print(f"✅  Qwen2.5-VL saved → {final}")
        clear_memory()
        return final


# ============================================================================
# EVALUATOR  (all three bugs fixed here)
# ============================================================================

class Evaluator:
    def __init__(self, cfg, tracker):
        self.cfg, self.tracker = cfg, tracker

    def run(self, model, processor, test_data, name):
        print(f"\n📊  Evaluating {name} on {len(test_data)} samples…")
        model.eval(); clear_memory()
        preds, gts = [], []

        with torch.no_grad():
            for i, ex in enumerate(test_data):
                try:
                    img = resize_image(ex["image"], self.cfg.MAX_IMAGE_SIZE)
                    gts.append(extract_volume(get_gt(ex)))

                    if "florence" in name.lower():
                        # ── FIX 1: cast pixel_values to model dtype ──────────
                        inputs = processor(
                            images=img, text="<VQA>What is the volume?",
                            return_tensors="pt")
                        dtype = next(model.parameters()).dtype
                        inputs = {
                            k: v.to(model.device).to(dtype)
                               if v.dtype.is_floating_point else v.to(model.device)
                            for k, v in inputs.items()
                        }
                        ids  = model.generate(**inputs, max_new_tokens=30)
                        text = processor.batch_decode(
                            ids, skip_special_tokens=True)[0]

                    else:
                        # ── FIX 2: system prompt for direct answer ───────────
                        msgs = [
                            {"role": "system", "content": QWEN_SYS},
                            {"role": "user", "content": [
                                {"type": "image", "image": img},
                                {"type": "text",  "text": "What is the volume in mL?"},
                            ]},
                        ]
                        txt = processor.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True)
                        inp = processor(text=[txt], images=[img],
                                        return_tensors="pt").to(model.device)
                        ids  = model.generate(**inp, max_new_tokens=20, do_sample=False)
                        text = processor.batch_decode(
                            ids, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)[0]

                    # ── FIX 3: smart volume extraction ───────────────────────
                    preds.append(extract_volume(text))

                    if (i+1) % 50 == 0:
                        print(f"  {i+1}/{len(test_data)}  last: '{text[:60]}'")

                except Exception as e:
                    print(f"  ⚠️  sample {i}: {e}")
                    preds.append(0.0)

        p, g = np.array(preds), np.array(gts)
        mask = g != 0  # avoid division by zero in MAPE

        mae  = mean_absolute_error(g, p)
        rmse = np.sqrt(mean_squared_error(g, p))
        r2   = r2_score(g, p)
        mape = mean_absolute_percentage_error(g[mask], p[mask]) * 100 if mask.any() else 0
        err  = p - g

        metrics = {
            "mae":        float(mae),
            "rmse":       float(rmse),
            "r2":         float(r2),
            "mape":       float(mape),
            "max_error":  float(np.max(np.abs(err))),
            "std_error":  float(np.std(err)),
            "predictions":  p.tolist(),
            "ground_truth": g.tolist(),
        }
        self.tracker.record_eval(name, metrics)
        print(f"📈  {name}: MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f} | MAPE={mape:.1f}%")

        self._plot(g, p, name)
        return metrics

    def _plot(self, g, p, name):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Scatter
        axes[0].scatter(g, p, alpha=0.4, s=20, color="#38bdf8")
        lo, hi = min(g.min(), p.min()), max(g.max(), p.max())
        axes[0].plot([lo,hi],[lo,hi],"r--",lw=2,label="Perfect")
        axes[0].set_xlabel("Ground Truth (mL)"); axes[0].set_ylabel("Prediction (mL)")
        axes[0].set_title(f"{name} — Pred vs GT", fontweight="bold")
        axes[0].legend(); axes[0].grid(True,alpha=0.3)

        # Error histogram
        err = p - g
        axes[1].hist(err, bins=30, edgecolor="black", alpha=0.75, color="#818cf8")
        axes[1].axvline(0, color="red", lw=2, linestyle="--", label="Zero error")
        axes[1].axvline(err.mean(), color="lime", lw=2, linestyle="--",
                        label=f"Mean: {err.mean():.1f}")
        axes[1].set_xlabel("Error (mL)"); axes[1].set_ylabel("Count")
        axes[1].set_title(f"{name} — Error Dist", fontweight="bold")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        # Residuals vs GT
        axes[2].scatter(g, err, alpha=0.4, s=20, color="#34d399")
        axes[2].axhline(0, color="red", lw=2, linestyle="--")
        axes[2].set_xlabel("Ground Truth (mL)"); axes[2].set_ylabel("Residual (mL)")
        axes[2].set_title(f"{name} — Residuals", fontweight="bold")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        path = f"{self.cfg.OUTPUT_DIR}/{name.replace(' ','_')}_eval.png"
        plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
        print(f"   📊  → {path}")


# ============================================================================
# HUGGINGFACE PUSH
# ============================================================================

def push_to_hub(cfg: Config, florence_dir: str, qwen_dir: str):
    """
    Push fine-tuned models to HuggingFace Hub.
    Requires: huggingface-cli login  (or HF_TOKEN env var).
    """
    print("\n" + "="*60)
    print("📤  PUSHING MODELS TO HUGGINGFACE HUB")
    print("="*60)

    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi()

        # ── Florence-2 ────────────────────────────────────────────────────
        print(f"\n→ Uploading Florence-2 → {cfg.HF_FLORENCE_REPO}")
        try:
            create_repo(cfg.HF_FLORENCE_REPO, repo_type="model",
                        private=False, exist_ok=True)
        except Exception as e:
            print(f"  ℹ️  Repo create: {e}")

        # Write a model card
        card_f = os.path.join(florence_dir, "README.md")
        with open(card_f, "w") as f:
            f.write(f"""---
language: en
tags:
- vision
- florence-2
- beaker
- volume-estimation
- peft
- lora
license: mit
datasets:
- yusufbukarmaina/Beakers1
---

# Beaker Volume Estimator — Florence-2

Fine-tuned **microsoft/Florence-2-base** on the Beakers1 dataset using LoRA (r={cfg.LORA_R}).

## Task
Given an image of a lab beaker, predict the volume of liquid in mL.

## Training
- Dataset: yusufbukarmaina/Beakers1 ({cfg.TRAIN_SAMPLES} train / {cfg.VAL_SAMPLES} val)
- LoRA rank: {cfg.LORA_R}, alpha: {cfg.LORA_ALPHA}
- Epochs: {cfg.NUM_EPOCHS}, LR: {cfg.LEARNING_RATE}
- Batch size (effective): {cfg.BATCH_SIZE * cfg.GRADIENT_ACCUMULATION}

## Usage
```python
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

model_id = "{cfg.HF_FLORENCE_REPO}"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype=torch.float16
).eval().cuda()

from PIL import Image
image = Image.open("beaker.jpg").convert("RGB")
dtype = next(model.parameters()).dtype
inputs = processor(images=image, text="<VQA>What is the volume?",
                   return_tensors="pt")
inputs = {{k: v.cuda().to(dtype) if v.dtype.is_floating_point else v.cuda()
           for k, v in inputs.items()}}
with torch.no_grad():
    ids = model.generate(**inputs, max_new_tokens=30)
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```
""")

        api.upload_folder(
            folder_path=florence_dir,
            repo_id=cfg.HF_FLORENCE_REPO,
            repo_type="model",
            commit_message="Upload fine-tuned Florence-2 beaker volume model",
        )
        print(f"✅  Florence-2 uploaded → https://huggingface.co/{cfg.HF_FLORENCE_REPO}")

        # ── Qwen2.5-VL ────────────────────────────────────────────────────
        print(f"\n→ Uploading Qwen2.5-VL → {cfg.HF_QWEN_REPO}")
        try:
            create_repo(cfg.HF_QWEN_REPO, repo_type="model",
                        private=False, exist_ok=True)
        except Exception as e:
            print(f"  ℹ️  Repo create: {e}")

        card_q = os.path.join(qwen_dir, "README.md")
        with open(card_q, "w") as f:
            f.write(f"""---
language: en
tags:
- vision
- qwen2-vl
- beaker
- volume-estimation
- peft
- lora
license: apache-2.0
datasets:
- yusufbukarmaina/Beakers1
---

# Beaker Volume Estimator — Qwen2.5-VL

Fine-tuned **Qwen/Qwen2-VL-2B-Instruct** on the Beakers1 dataset using LoRA (r={cfg.LORA_R}).

## Task
Given an image of a lab beaker, predict the volume of liquid in mL.

## Training
- Dataset: yusufbukarmaina/Beakers1 ({cfg.TRAIN_SAMPLES} train)
- LoRA rank: {cfg.LORA_R}, alpha: {cfg.LORA_ALPHA}
- Epochs: {cfg.NUM_EPOCHS}, LR: {cfg.LEARNING_RATE}

## Usage
```python
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import torch

model_id = "{cfg.HF_QWEN_REPO}"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype=torch.float16
).eval().cuda()

from PIL import Image
image = Image.open("beaker.jpg").convert("RGB")
messages = [
    {{"role": "system", "content": "Reply with ONLY the volume, e.g. '150 mL'."}},
    {{"role": "user", "content": [
        {{"type": "image", "image": image}},
        {{"type": "text",  "text":  "What is the volume in mL?"}},
    ]}},
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").cuda()
with torch.no_grad():
    ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```
""")

        api.upload_folder(
            folder_path=qwen_dir,
            repo_id=cfg.HF_QWEN_REPO,
            repo_type="model",
            commit_message="Upload fine-tuned Qwen2.5-VL beaker volume model",
        )
        print(f"✅  Qwen2.5-VL uploaded → https://huggingface.co/{cfg.HF_QWEN_REPO}")

    except ImportError:
        print("❌  huggingface_hub not installed: pip install huggingface_hub")
    except Exception as e:
        print(f"❌  Upload failed: {e}")
        import traceback; traceback.print_exc()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🚀  Beaker Volume Pipeline — Fixed Metrics + HF Push")
    print("="*70)

    cfg = Config()
    os.makedirs(cfg.OUTPUT_DIR,      exist_ok=True)
    os.makedirs(cfg.TEST_IMAGES_DIR, exist_ok=True)

    tracker = MetricsTracker()
    tracker.start(cfg)
    if torch.cuda.is_available():
        print(f"✅  {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
        tracker.gpu("start")

    # 1. Data
    dp = DatasetProcessor(cfg)
    train, val, test = dp.load()
    tracker.data["dataset"] = {"train":len(train), "val":len(val), "test":len(test)}

    # 2. Save test images BEFORE training
    dp.save_test_images(test, cfg.TEST_IMAGES_DIR)

    # 3. Florence-2
    print("\n" + "="*70); print("FLORENCE-2 TRAINING"); print("="*70)
    ft = FlorenceTrainer(cfg, tracker)
    f_path = ft.train(train, val)
    del ft; clear_memory()

    # 4. Qwen2.5-VL
    print("\n" + "="*70); print("QWEN2.5-VL TRAINING"); print("="*70)
    qt = QwenTrainer(cfg, tracker)
    q_path = qt.train(train, val)

    # 5. Evaluate
    print("\n" + "="*70); print("EVALUATION"); print("="*70)
    ev = Evaluator(cfg, tracker)

    fe = FlorenceTrainer(cfg, tracker); fe.setup()
    ev.run(fe.model, fe.processor, test, "Florence-2")
    del fe; clear_memory()

    ev.run(qt.model, qt.processor, test, "Qwen2.5-VL")
    del qt; clear_memory()

    # 6. Save metrics
    print("\n" + "="*70); print("METRICS SUMMARY"); print("="*70)
    df = tracker.save(cfg.OUTPUT_DIR)
    print("\n" + df.to_string(index=False))

    # 7. Push to HuggingFace (set UPLOAD_TO_HF = True or call manually)
    UPLOAD_TO_HF = os.getenv("UPLOAD_TO_HF", "0") == "1"
    if UPLOAD_TO_HF:
        push_to_hub(cfg, f_path, q_path)
    else:
        print("\n💡  To push to HuggingFace:")
        print("      huggingface-cli login")
        print("      UPLOAD_TO_HF=1 python train_vision_models_metrics.py")
        print("  or call push_to_hub(cfg, f_path, q_path) directly.")

    print("\n" + "="*70)
    print("🎉  COMPLETE")
    print("="*70)
    print(f"  Florence-2 : {f_path}")
    print(f"  Qwen2.5-VL : {q_path}")
    print(f"  Metrics    : {cfg.OUTPUT_DIR}/complete_metrics.json")
    print(f"  LaTeX table: {cfg.OUTPUT_DIR}/summary_table.tex")
    print(f"  Test images: {cfg.TEST_IMAGES_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted"); clear_memory()
    except Exception as e:
        print(f"\n❌  Fatal: {e}")
        import traceback; traceback.print_exc()
        clear_memory()

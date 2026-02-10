"""
✅ Kaggle 2×T4 (16GB each) Training Pipeline
Florence-2 + Qwen2-VL-2B-Instruct with LoRA + 4-bit (recommended)

Run (2 GPUs):
  torchrun --nproc_per_node=2 train_t4x2.py

Notes:
- T4 supports FP16, NOT BF16.
- 4-bit (bitsandbytes) is strongly recommended for Qwen2-VL on 16GB.
- Uses DDP automatically when launched via torchrun.
"""

import os
import json
import gc
import re
import warnings
from typing import Dict, List

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

warnings.filterwarnings("ignore")

# ---------------------------
# bitsandbytes / 4-bit
# ---------------------------
try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None
    HAS_BNB = False


# ============================================================================
# CONFIG (T4×2)
# ============================================================================

class Config:
    # Dataset
    HF_DATASET_NAME = "yusufbukarmaina/Beakers1"
    STREAMING = True

    TRAIN_SAMPLES = 500
    VAL_SAMPLES   = 150
    TEST_SAMPLES  = 300
    TOTAL_SAMPLES = 950

    # Models
    FLORENCE_MODEL = "microsoft/Florence-2-base"
    QWEN_MODEL     = "Qwen/Qwen2-VL-2B-Instruct"

    # LoRA
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

    # Training (2×T4)
    # With 2 GPUs, per_device_train_batch_size=2 often works in 4-bit + checkpointing.
    # If OOM, set BATCH_SIZE=1 and increase GRAD_ACCUM.
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 8  # effective batch = 2(gpu)*2(batch)*8 = 32
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 8
    WARMUP_STEPS = 50

    # Sequence length (reduce if OOM)
    MAX_LENGTH = 384

    # Memory
    FP16 = True
    GRADIENT_CHECKPOINTING = True
    USE_4BIT = True

    # I/O
    OUTPUT_DIR = "./trained_models_t4x2"
    TEST_IMAGES_DIR = "./test_images_export_t4x2"
    SAVE_STEPS = 600
    EVAL_STEPS = 600
    LOGGING_STEPS = 50

    # HF upload
    UPLOAD_TO_HF = True
    HF_REPO_NAME = "yusufbukarmaina/beaker-volume-models"


# ============================================================================
# DDP helpers
# ============================================================================

def is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def is_main_process() -> bool:
    return (not is_distributed()) or torch.distributed.get_rank() == 0

def rank0_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)

def barrier():
    if is_distributed():
        torch.distributed.barrier()

# ============================================================================
# Utils
# ============================================================================

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_memory_usage():
    if torch.cuda.is_available() and is_main_process():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def get_bnb_config():
    if not (HAS_BNB and Config.USE_4BIT):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # T4: fp16
    )

def pick_optim():
    # if bnb 4-bit available, use paged adamw 8bit (memory-friendly)
    if HAS_BNB and Config.USE_4BIT:
        return "paged_adamw_8bit"
    return "adamw_torch"


# ============================================================================
# Collators (variable length padding)
# ============================================================================

def pad_1d(seqs: List[torch.Tensor], pad_value: int):
    return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_value)

def florence_collate_fn(features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    input_ids = pad_1d([f["input_ids"] for f in features], pad_value=0)
    attention_mask = pad_1d([f["attention_mask"] for f in features], pad_value=0)
    labels = pad_1d([f["labels"] for f in features], pad_value=-100)
    return dict(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels)

def qwen_collate_fn(features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch = {}
    # fixed-ish tensors
    for k in ["pixel_values", "image_grid_thw"]:
        if k in features[0]:
            batch[k] = torch.stack([f[k] for f in features])

    # variable length
    for k, padv in [("input_ids", 0), ("attention_mask", 0), ("labels", -100)]:
        if k in features[0]:
            batch[k] = pad_1d([f[k] for f in features], pad_value=padv)

    # passthrough others if stackable
    for k in features[0].keys():
        if k in batch:
            continue
        try:
            batch[k] = torch.stack([f[k] for f in features])
        except Exception:
            pass

    return batch


# ============================================================================
# Data
# ============================================================================

class DatasetProcessor:
    def __init__(self, config: Config):
        self.config = config

    def load_and_split_dataset(self):
        rank0_print("📥 Loading dataset with streaming...")
        rank0_print(f"Dataset: {self.config.HF_DATASET_NAME}")
        rank0_print(f"Target: {self.config.TRAIN_SAMPLES} train, {self.config.VAL_SAMPLES} val, {self.config.TEST_SAMPLES} test")

        ds = load_dataset(
            self.config.HF_DATASET_NAME,
            split="train",
            streaming=self.config.STREAMING
        ).shuffle(seed=42, buffer_size=1000)

        train_data, val_data, test_data = [], [], []
        total_processed, skipped = 0, 0

        for ex in ds:
            if "image" not in ex:
                skipped += 1
                continue
            if "volume_ml" not in ex and "volume_label" not in ex:
                skipped += 1
                continue

            if len(train_data) < self.config.TRAIN_SAMPLES:
                train_data.append(ex)
            elif len(val_data) < self.config.VAL_SAMPLES:
                val_data.append(ex)
            elif len(test_data) < self.config.TEST_SAMPLES:
                test_data.append(ex)
            else:
                break

            total_processed += 1
            if total_processed % 100 == 0:
                rank0_print(f"✓ Processed {total_processed} - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

            if total_processed > self.config.TOTAL_SAMPLES + 800:
                rank0_print("⚠️ Safety limit reached, stopping.")
                break

        rank0_print("\n✅ Dataset split complete:")
        rank0_print(f"   Train: {len(train_data)} examples")
        rank0_print(f"   Val:   {len(val_data)} examples")
        rank0_print(f"   Test:  {len(test_data)} examples")
        rank0_print(f"   Skipped: {skipped} examples (missing fields)")

        return train_data, val_data, test_data

    def extract_volume_from_text(self, text: str) -> float:
        if not text:
            return 0.0
        text = str(text)
        patterns = [
            r"(\d+\.?\d*)\s*mL",
            r"(\d+\.?\d*)\s*ml",
            r"(\d+\.?\d*)\s*ML",
            r"(\d+\.?\d*)\s*milliliters?",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return float(m.group(1))
        nums = re.findall(r"\d+\.?\d*", text)
        return float(nums[0]) if nums else 0.0

    def save_test_images(self, test_data: List[Dict], output_dir: str):
        if not is_main_process():
            return
        rank0_print(f"\n💾 Saving {len(test_data)} test images to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        meta = []

        for i, ex in enumerate(test_data):
            try:
                img = Image.open(ex["image"]).convert("RGB") if isinstance(ex["image"], str) else ex["image"].convert("RGB")
                gt_text = ex.get("volume_label", "")
                if not gt_text and "volume_ml" in ex:
                    gt_text = f"{ex['volume_ml']} mL"
                gt_v = self.extract_volume_from_text(gt_text)

                fn = f"test_{i:04d}_volume_{gt_v:.1f}mL.jpg"
                img.save(os.path.join(output_dir, fn), quality=95)
                meta.append({"index": i, "filename": fn, "ground_truth_volume": gt_v, "ground_truth_text": gt_text})

                if (i + 1) % 50 == 0:
                    rank0_print(f"  Saved {i+1}/{len(test_data)}")
            except Exception as e:
                rank0_print(f"⚠️ Save error {i}: {e}")

        with open(os.path.join(output_dir, "test_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        rank0_print("✅ Test images + metadata saved.")


# ============================================================================
# Florence-2 Trainer
# ============================================================================

class FlorenceTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.processor = None
        self.model = None

    def setup_model(self):
        rank0_print(f"\n🤖 Setting up Florence-2: {self.config.FLORENCE_MODEL}")
        clear_memory()
        bnb = get_bnb_config()

        self.processor = AutoProcessor.from_pretrained(self.config.FLORENCE_MODEL, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.FLORENCE_MODEL,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16 if self.config.FP16 else torch.float32,
            low_cpu_mem_usage=True,
            quantization_config=bnb if bnb is not None else None,
        )

        if self.config.GRADIENT_CHECKPOINTING:
            self.model.gradient_checkpointing_enable()
            rank0_print("✓ Gradient checkpointing enabled")

        self.model = prepare_model_for_kbit_training(self.model)

        lora_cfg = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            target_modules=self.config.LORA_TARGET_MODULES,
            lora_dropout=self.config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        if is_main_process():
            self.model.print_trainable_parameters()
        print_memory_usage()
        return self.model, self.processor

    def train(self, train_data: List[Dict], val_data: List[Dict]) -> str:
        self.setup_model()
        out_dir = os.path.join(self.config.OUTPUT_DIR, "florence2_t4x2")
        os.makedirs(out_dir, exist_ok=True)

        args = TrainingArguments(
            output_dir=out_dir,
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
            bf16=False,  # T4 no bf16
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="tensorboard",
            logging_dir=os.path.join(out_dir, "runs"),
            gradient_checkpointing=self.config.GRADIENT_CHECKPOINTING,
            optim=pick_optim(),
            max_grad_norm=1.0,
            ddp_find_unused_parameters=False,
        )

        class FlorenceDataset(torch.utils.data.Dataset):
            def __init__(self, data, processor):
                self.data = data
                self.processor = processor
                self.pad_id = processor.tokenizer.pad_token_id

            def __len__(self): return len(self.data)

            def __getitem__(self, idx):
                ex = self.data[idx]
                img = Image.open(ex["image"]).convert("RGB") if isinstance(ex["image"], str) else ex["image"].convert("RGB")

                prompt = "<VQA>What is the volume of liquid in the beaker?"
                ans = ex.get("volume_label", "")
                if not ans and "volume_ml" in ex:
                    ans = f"{ex['volume_ml']} mL"
                if not ans:
                    ans = "0 mL"

                inputs = self.processor(images=img, text=prompt, return_tensors="pt", padding=True, truncation=True)
                ans_ids = self.processor.tokenizer(str(ans), return_tensors="pt", padding=True, truncation=True, max_length=64)

                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                labels = ans_ids["input_ids"].squeeze(0).clone()
                if self.pad_id is not None:
                    labels[labels == self.pad_id] = -100
                inputs["labels"] = labels
                return inputs

        train_ds = FlorenceDataset(train_data, self.processor)
        val_ds   = FlorenceDataset(val_data, self.processor)

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=florence_collate_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        rank0_print("\n🚀 Training Florence-2 (T4×2)...")
        print_memory_usage()
        trainer.train()

        final_dir = os.path.join(self.config.OUTPUT_DIR, "florence2_final_t4x2")
        if is_main_process():
            trainer.save_model(final_dir)
            self.processor.save_pretrained(final_dir)
        barrier()
        clear_memory()
        return final_dir


# ============================================================================
# Qwen2-VL Trainer
# ============================================================================

class QwenTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.processor = None
        self.model = None

    def setup_model(self):
        rank0_print(f"\n🤖 Setting up Qwen2-VL: {self.config.QWEN_MODEL}")
        clear_memory()
        bnb = get_bnb_config()

        self.processor = AutoProcessor.from_pretrained(self.config.QWEN_MODEL, trust_remote_code=True)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.QWEN_MODEL,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16 if self.config.FP16 else torch.float32,
            low_cpu_mem_usage=True,
            quantization_config=bnb if bnb is not None else None,
        )

        if self.config.GRADIENT_CHECKPOINTING:
            self.model.gradient_checkpointing_enable()
            rank0_print("✓ Gradient checkpointing enabled")

        self.model = prepare_model_for_kbit_training(self.model)

        lora_cfg = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            target_modules=self.config.LORA_TARGET_MODULES,
            lora_dropout=self.config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        if is_main_process():
            self.model.print_trainable_parameters()
        print_memory_usage()
        return self.model, self.processor

    def train(self, train_data: List[Dict], val_data: List[Dict]) -> str:
        self.setup_model()
        out_dir = os.path.join(self.config.OUTPUT_DIR, "qwen2vl_t4x2")
        os.makedirs(out_dir, exist_ok=True)

        args = TrainingArguments(
            output_dir=out_dir,
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
            bf16=False,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="tensorboard",
            logging_dir=os.path.join(out_dir, "runs"),
            gradient_checkpointing=self.config.GRADIENT_CHECKPOINTING,
            optim=pick_optim(),
            max_grad_norm=1.0,
            ddp_find_unused_parameters=False,
        )

        class QwenDataset(torch.utils.data.Dataset):
            def __init__(self, data, processor, max_length):
                self.data = data
                self.processor = processor
                self.max_length = max_length
                self.pad_id = getattr(processor.tokenizer, "pad_token_id", None)

            def __len__(self): return len(self.data)

            def __getitem__(self, idx):
                ex = self.data[idx]
                img = Image.open(ex["image"]).convert("RGB") if isinstance(ex["image"], str) else ex["image"].convert("RGB")

                q = "What is the volume of liquid in this beaker in mL?"
                if "volume_ml" in ex:
                    a = f"{ex['volume_ml']} mL"
                elif "volume_label" in ex:
                    a = str(ex["volume_label"])
                else:
                    a = "0 mL"

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a lab measurement tool. "
                            "Respond with ONLY the volume as a number and unit, e.g. '150 mL'. "
                            "Never explain or use formulas."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": q},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": a}],
                    },
                ]

                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

                inputs = self.processor(
                    text=[text],
                    images=[img],
                    return_tensors="pt",
                    padding=True,         # dynamic
                    truncation=True,
                    max_length=self.max_length,
                )

                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                labels = inputs["input_ids"].clone()
                if self.pad_id is not None:
                    labels[labels == self.pad_id] = -100
                inputs["labels"] = labels
                return inputs

        train_ds = QwenDataset(train_data, self.processor, self.config.MAX_LENGTH)
        val_ds   = QwenDataset(val_data, self.processor, self.config.MAX_LENGTH)

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=qwen_collate_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        rank0_print("\n🚀 Training Qwen2-VL (T4×2)...")
        print_memory_usage()
        trainer.train()

        final_dir = os.path.join(self.config.OUTPUT_DIR, "qwen2vl_final_t4x2")
        if is_main_process():
            trainer.save_model(final_dir)
            self.processor.save_pretrained(final_dir)
        barrier()
        clear_memory()
        return final_dir


# ============================================================================
# Evaluation (rank0 only)
# ============================================================================

class ModelEvaluator:
    def __init__(self, config: Config):
        self.config = config
        self.dp = DatasetProcessor(config)

    def evaluate(self, model, processor, test_data: List[Dict], name: str) -> Dict:
        if not is_main_process():
            return {}
        rank0_print(f"\n📊 Evaluating {name} on {len(test_data)} samples...")
        model.eval()

        preds, gts = [], []
        clear_memory()

        with torch.no_grad():
            for i, ex in enumerate(test_data):
                try:
                    img = Image.open(ex["image"]).convert("RGB") if isinstance(ex["image"], str) else ex["image"].convert("RGB")

                    gt_text = ex.get("volume_label", "")
                    if not gt_text and "volume_ml" in ex:
                        gt_text = f"{ex['volume_ml']} mL"
                    gt_v = self.dp.extract_volume_from_text(gt_text)
                    gts.append(gt_v)

                    if "florence" in name.lower():
                        prompt = "<VQA>What is the volume of liquid in the beaker?"
                        inp = processor(images=img, text=prompt, return_tensors="pt")
                        model_dtype = next(model.parameters()).dtype
                        inp = {
                            k: (v.to(model.device).to(model_dtype) if v.dtype.is_floating_point else v.to(model.device))
                            for k, v in inp.items()
                        }
                        out = model.generate(**inp, max_new_tokens=30)
                        txt = processor.batch_decode(out, skip_special_tokens=True)[0]
                    else:
                        messages = [
                            {
                                "role": "system",
                                "content": (
                                    "You are a lab measurement tool. "
                                    "Respond with ONLY the volume as a number and unit, e.g. '150 mL'. "
                                    "Never explain or use formulas."
                                ),
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": img},
                                    {"type": "text", "text": "What is the volume of liquid in this beaker in mL?"},
                                ],
                            },
                        ]
                        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inp = processor(text=[text], images=[img], return_tensors="pt")
                        inp = {k: v.to(model.device) for k, v in inp.items()}
                        out = model.generate(**inp, max_new_tokens=20, do_sample=False)
                        txt = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

                    pred_v = self.dp.extract_volume_from_text(txt)
                    preds.append(pred_v)

                    if (i + 1) % 50 == 0:
                        rank0_print(f"  ✓ {i+1}/{len(test_data)}")

                except Exception as e:
                    rank0_print(f"⚠️ Eval error {i}: {e}")
                    preds.append(0.0)

        preds = np.array(preds)
        gts = np.array(gts)

        mae = mean_absolute_error(gts, preds)
        rmse = np.sqrt(mean_squared_error(gts, preds))
        r2 = r2_score(gts, preds)

        rank0_print(f"\n{name} Metrics:")
        rank0_print(f"  MAE:  {mae:.2f} mL")
        rank0_print(f"  RMSE: {rmse:.2f} mL")
        rank0_print(f"  R²:   {r2:.4f}")

        return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


# ============================================================================
# Main
# ============================================================================

def main():
    cfg = Config()

    if is_main_process():
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cfg.TEST_IMAGES_DIR, exist_ok=True)

    barrier()

    rank0_print("=" * 80)
    rank0_print("🚀 Kaggle Training Pipeline: Florence-2 + Qwen2-VL (2×T4)")
    rank0_print("=" * 80)

    if torch.cuda.is_available() and is_main_process():
        rank0_print(f"GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            rank0_print(f"  - {i}: {torch.cuda.get_device_name(i)}")
        print_memory_usage()
    elif is_main_process():
        rank0_print("❌ No GPU detected.")

    dp = DatasetProcessor(cfg)
    train_data, val_data, test_data = dp.load_and_split_dataset()

    dp.save_test_images(test_data, cfg.TEST_IMAGES_DIR)
    barrier()

    # Florence
    flor_tr = FlorenceTrainer(cfg)
    flor_path = flor_tr.train(train_data, val_data)
    del flor_tr
    clear_memory()
    barrier()

    # Qwen
    qwen_tr = QwenTrainer(cfg)
    qwen_path = qwen_tr.train(train_data, val_data)
    barrier()

    # Evaluate (rank0)
    ev = ModelEvaluator(cfg)

    flor_eval = FlorenceTrainer(cfg)
    flor_eval.setup_model()
    flor_res = ev.evaluate(flor_eval.model, flor_eval.processor, test_data, "Florence-2")
    del flor_eval
    clear_memory()

    qwen_res = ev.evaluate(qwen_tr.model, qwen_tr.processor, test_data, "Qwen2-VL")

    if is_main_process():
        results = {
            "florence2": flor_res,
            "qwen2vl": qwen_res,
            "paths": {"florence2": flor_path, "qwen2vl": qwen_path},
            "config": {
                "dataset": cfg.HF_DATASET_NAME,
                "train": len(train_data),
                "val": len(val_data),
                "test": len(test_data),
                "batch_size": cfg.BATCH_SIZE,
                "grad_accum": cfg.GRADIENT_ACCUMULATION,
                "effective_batch": cfg.BATCH_SIZE * cfg.GRADIENT_ACCUMULATION * max(1, torch.cuda.device_count()),
                "max_length": cfg.MAX_LENGTH,
                "use_4bit": bool(cfg.USE_4BIT and HAS_BNB),
            },
        }
        out_json = os.path.join(cfg.OUTPUT_DIR, "evaluation_results_t4x2.json")
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)

        rank0_print("\n" + "=" * 80)
        rank0_print("🎉 DONE (T4×2)")
        rank0_print("=" * 80)
        rank0_print(f"Florence-2 saved: {flor_path}")
        rank0_print(f"Qwen2-VL saved:   {qwen_path}")
        rank0_print(f"Results saved:    {out_json}")
        rank0_print(f"Test images:      {cfg.TEST_IMAGES_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        rank0_print("\n⚠️ Interrupted.")
        clear_memory()
    except Exception as e:
        rank0_print(f"\n❌ Fatal: {e}")
        import traceback
        traceback.print_exc()
        clear_memory()


"""
Complete Training Pipeline for Florence-2 and Qwen2.5-VL
✅ Converted/optimized for NVIDIA Tesla P100 (16GB VRAM)

Key P100 changes:
- Smaller micro-batch (BATCH_SIZE=1) + higher gradient accumulation
- 4-bit quantization (bitsandbytes) to fit models in 16GB
- More aggressive max_length and image/token memory control
- dataloader workers = 0 (stability), pin_memory off
- fp16 enabled (P100 supports FP16), bf16 disabled
- safer dynamic padding + custom collators (avoids stacking errors)
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
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from PIL import Image
import re
from typing import Dict, List
import warnings
import gc
warnings.filterwarnings("ignore")

# Optional: bitsandbytes 4-bit
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None
    _HAS_BNB = False


# ============================================================================
# CONFIGURATION - OPTIMIZED FOR TESLA P100 (16GB)
# ============================================================================

class Config:
    # Dataset
    HF_DATASET_NAME = "yusufbukarmaina/Beakers1"
    STREAMING = True

    TRAIN_SAMPLES = 1000
    VAL_SAMPLES   = 300
    TEST_SAMPLES  = 300
    TOTAL_SAMPLES = 1600

    # Models
    FLORENCE_MODEL = "microsoft/Florence-2-base"
    QWEN_MODEL     = "Qwen/Qwen2-VL-2B-Instruct"

    # LoRA (keep modest on P100)
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

    # Training (P100 memory-friendly)
    BATCH_SIZE = 1                     # ↓ micro-batch for 16GB
    GRADIENT_ACCUMULATION = 16         # ↑ keep effective batch = 16
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 8                     # often enough; reduce if slow
    WARMUP_STEPS = 50

    # Token length (reduce to save VRAM)
    MAX_LENGTH = 384                   # was 512; reduce for P100

    # Memory
    FP16 = True
    GRADIENT_CHECKPOINTING = True
    USE_4BIT = True                    # IMPORTANT for P100
    # If bitsandbytes missing, it will fallback automatically.

    # Output
    OUTPUT_DIR = "./trained_models"
    TEST_IMAGES_DIR = "./test_images_export"   # safer than "/FQ/..."
    SAVE_STEPS = 600
    EVAL_STEPS = 600
    LOGGING_STEPS = 50

    # HF upload
    UPLOAD_TO_HF = False
    HF_REPO_NAME = "yusufbukarmaina/beaker-volume-models"


# ============================================================================
# UTILS
# ============================================================================

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def get_bnb_config():
    """
    4-bit config that works well on 16GB GPUs.
    Note: compute dtype float16 (P100 supports fp16).
    """
    if not (_HAS_BNB and Config.USE_4BIT):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

def pick_optim():
    """
    If bitsandbytes is present, paged_adamw_8bit is usually more memory-friendly.
    Otherwise use adamw_torch.
    """
    if _HAS_BNB and Config.USE_4BIT:
        return "paged_adamw_8bit"
    return "adamw_torch"


# ============================================================================
# COLLATORS (PADDING FIXES)
# ============================================================================

def pad_1d(seqs: List[torch.Tensor], pad_value: int):
    return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_value)

def florence_collate_fn(features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    input_ids = pad_1d([f["input_ids"] for f in features], pad_value=0)
    attention_mask = pad_1d([f["attention_mask"] for f in features], pad_value=0)
    labels = pad_1d([f["labels"] for f in features], pad_value=-100)
    return dict(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

def qwen_collate_fn(features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Qwen2-VL processor returns multiple tensors; some are variable-length (input_ids/attention_mask/labels),
    others are fixed-ish (pixel_values, image_grid_thw). We pad the variable-length ones.
    """
    batch = {}
    # stack fixed keys if present
    fixed_keys = ["pixel_values", "image_grid_thw"]
    for k in fixed_keys:
        if k in features[0]:
            batch[k] = torch.stack([f[k] for f in features])

    # pad variable length keys
    for k, padv in [("input_ids", 0), ("attention_mask", 0), ("labels", -100)]:
        if k in features[0]:
            batch[k] = pad_1d([f[k] for f in features], pad_value=padv)

    # pass through any other keys (rare)
    for k in features[0].keys():
        if k not in batch:
            try:
                batch[k] = torch.stack([f[k] for f in features])
            except Exception:
                # ignore if cannot stack (better to fail early, but we keep it safe)
                pass
    return batch


# ============================================================================
# DATA PROCESSING
# ============================================================================

class DatasetProcessor:
    def __init__(self, config: Config):
        self.config = config

    def load_and_split_dataset(self):
        print("📥 Loading dataset with streaming...")
        print(f"Dataset: {self.config.HF_DATASET_NAME}")
        print(f"Target: {self.config.TRAIN_SAMPLES} train, {self.config.VAL_SAMPLES} val, {self.config.TEST_SAMPLES} test")

        dataset = load_dataset(
            self.config.HF_DATASET_NAME,
            split="train",
            streaming=self.config.STREAMING
        ).shuffle(seed=42, buffer_size=1000)

        train_data, val_data, test_data = [], [], []
        total_processed, skipped = 0, 0

        for example in dataset:
            if "image" not in example:
                skipped += 1
                continue
            if "volume_ml" not in example and "volume_label" not in example:
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
                print(f"✓ Processed {total_processed} - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

            if total_processed > self.config.TOTAL_SAMPLES + 500:
                print("⚠️ Safety limit reached, stopping.")
                break

        print("\n✅ Dataset split complete:")
        print(f"   Train: {len(train_data)} examples")
        print(f"   Val:   {len(val_data)} examples")
        print(f"   Test:  {len(test_data)} examples")
        print(f"   Skipped: {skipped} examples (missing fields)")

        return train_data, val_data, test_data

    def extract_volume_from_text(self, text: str) -> float:
        if not text:
            return 0.0
        text = str(text)
        patterns = [
            r"(\d+\.?\d*)\s*mL",
            r"(\d+\.?\d*)\s*ml",
            r"(\d+\.?\d*)\s*milliliters?",
            r"(\d+\.?\d*)\s*ML",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return float(m.group(1))
        nums = re.findall(r"\d+\.?\d*", text)
        return float(nums[0]) if nums else 0.0

    def save_test_images(self, test_data: List[Dict], output_dir: str):
        print(f"\n💾 Saving {len(test_data)} test images to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        metadata = []

        for idx, example in enumerate(test_data):
            try:
                image = Image.open(example["image"]).convert("RGB") if isinstance(example["image"], str) else example["image"].convert("RGB")
                gt_text = example.get("volume_label", "")
                if not gt_text and "volume_ml" in example:
                    gt_text = f"{example['volume_ml']} mL"
                gt_volume = self.extract_volume_from_text(gt_text)

                filename = f"test_{idx:04d}_volume_{gt_volume:.1f}mL.jpg"
                image.save(os.path.join(output_dir, filename), quality=95)

                metadata.append({
                    "index": idx,
                    "filename": filename,
                    "ground_truth_volume": gt_volume,
                    "ground_truth_text": gt_text,
                })

                if (idx + 1) % 50 == 0:
                    print(f"  Saved {idx + 1}/{len(test_data)} images...")

            except Exception as e:
                print(f"⚠️ Error saving test image {idx}: {e}")

        with open(os.path.join(output_dir, "test_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print("✅ Test images + metadata saved.")


# ============================================================================
# FLORENCE-2 TRAINING (P100)
# ============================================================================

class FlorenceTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.processor = None
        self.model = None

    def setup_model(self):
        print(f"\n🤖 Setting up Florence-2: {self.config.FLORENCE_MODEL}")
        clear_memory()

        bnb_config = get_bnb_config()

        self.processor = AutoProcessor.from_pretrained(
            self.config.FLORENCE_MODEL,
            trust_remote_code=True
        )

        # For P100: prefer 4-bit if available
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.FLORENCE_MODEL,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16 if self.config.FP16 else torch.float32,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config if bnb_config is not None else None,
        )

        if self.config.GRADIENT_CHECKPOINTING:
            self.model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")

        # Required for LoRA + k-bit
        self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            target_modules=self.config.LORA_TARGET_MODULES,
            lora_dropout=self.config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        print_memory_usage()
        return self.model, self.processor

    def train(self, train_data: List[Dict], val_data: List[Dict]) -> str:
        self.setup_model()
        out_dir = f"{self.config.OUTPUT_DIR}/florence2_p100"
        os.makedirs(out_dir, exist_ok=True)

        training_args = TrainingArguments(
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
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="tensorboard",
            logging_dir=f"{out_dir}/runs",
            gradient_checkpointing=self.config.GRADIENT_CHECKPOINTING,
            optim=pick_optim(),
            max_grad_norm=1.0,
        )

        class FlorenceDataset(torch.utils.data.Dataset):
            def __init__(self, data, processor):
                self.data = data
                self.processor = processor
                self.pad_id = processor.tokenizer.pad_token_id

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                ex = self.data[idx]
                image = Image.open(ex["image"]).convert("RGB") if isinstance(ex["image"], str) else ex["image"].convert("RGB")

                prompt = "<VQA>What is the volume of liquid in the beaker?"
                answer = ex.get("volume_label", "")
                if not answer and "volume_ml" in ex:
                    answer = f"{ex['volume_ml']} mL"
                if not answer:
                    answer = "0 mL"

                inputs = self.processor(
                    images=image,
                    text=prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                ans = self.processor.tokenizer(
                    str(answer),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=64,

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
                )

                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                labels = ans["input_ids"].squeeze(0).clone()
                if self.pad_id is not None:
                    labels[labels == self.pad_id] = -100
                inputs["labels"] = labels
                return inputs

        train_ds = FlorenceDataset(train_data, self.processor)
        val_ds   = FlorenceDataset(val_data, self.processor)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=florence_collate_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        print("\n🚀 Training Florence-2 on P100...")
        print_memory_usage()
        trainer.train()

        final_dir = f"{self.config.OUTPUT_DIR}/florence2_final_p100"
        trainer.save_model(final_dir)
        self.processor.save_pretrained(final_dir)

        clear_memory()
        return final_dir


# ============================================================================
# QWEN2.5-VL TRAINING (P100)
# ============================================================================

class QwenTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.processor = None
        self.model = None

    def setup_model(self):
        print(f"\n🤖 Setting up Qwen2.5-VL: {self.config.QWEN_MODEL}")
        clear_memory()

        bnb_config = get_bnb_config()

        self.processor = AutoProcessor.from_pretrained(
            self.config.QWEN_MODEL,
            trust_remote_code=True
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.QWEN_MODEL,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16 if self.config.FP16 else torch.float32,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config if bnb_config is not None else None,
        )

        if self.config.GRADIENT_CHECKPOINTING:
            self.model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")

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
        out_dir = f"{self.config.OUTPUT_DIR}/qwen2_5vl_p100"
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
            dataloader_num_workers=0,     # stability on P100 machines
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="tensorboard",
            logging_dir=f"{out_dir}/runs",
            gradient_checkpointing=self.config.GRADIENT_CHECKPOINTING,
            optim=pick_optim(),
            max_grad_norm=1.0,
        )

        class QwenDataset(torch.utils.data.Dataset):
            def __init__(self, data, processor, max_length):
                self.data = data
                self.processor = processor
                self.max_length = max_length

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                ex = self.data[idx]
                image = Image.open(ex["image"]).convert("RGB") if isinstance(ex["image"], str) else ex["image"].convert("RGB")

                question = "What is the volume of liquid in this beaker in mL?"
                if "volume_ml" in ex:
                    answer = f"{ex['volume_ml']} mL"
                elif "volume_label" in ex:
                    answer = str(ex["volume_label"])
                else:
                    answer = "0 mL"

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
                            {"type": "image", "image": image},
                            {"type": "text", "text": question},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": answer}],
                    },
                ]

                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )

                # IMPORTANT for P100: use dynamic padding here, and keep max_length smaller
                inputs = self.processor(
                    text=[text],
                    images=[image],
                    return_tensors="pt",
                    padding=True,           # dynamic
                    truncation=True,
                    max_length=self.max_length,
                )

                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                labels = inputs["input_ids"].clone()
                # mask pads if tokenizer pad exists
                pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
                if pad_id is not None:
                    labels[labels == pad_id] = -100
                inputs["labels"] = labels
                return inputs

        train_ds = QwenDataset(train_data, self.processor, self.config.MAX_LENGTH)
        val_ds   = QwenDataset(val_data, self.processor, self.config.MAX_LENGTH)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=qwen_collate_fn,   # critical for variable-length batches
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        print("\n🚀 Training Qwen2.5-VL on P100...")
        print_memory_usage()
        trainer.train()

        final_dir = f"{self.config.OUTPUT_DIR}/qwen2_5vl_final_p100"
        trainer.save_model(final_dir)
        self.processor.save_pretrained(final_dir)

        clear_memory()
        return final_dir


# ============================================================================
# EVALUATION (unchanged, but safe casts)
# ============================================================================

class ModelEvaluator:
    def __init__(self, config: Config):
        self.config = config
        self.data_processor = DatasetProcessor(config)

    def evaluate_model(self, model, processor, test_data: List[Dict], model_name: str) -> Dict:
        print(f"\n📊 Evaluating {model_name} on {len(test_data)} test samples...")
        predictions, ground_truth = [], []

        model.eval()
        clear_memory()

        with torch.no_grad():
            for idx, ex in enumerate(test_data):
                try:
                    image = Image.open(ex["image"]).convert("RGB") if isinstance(ex["image"], str) else ex["image"].convert("RGB")

                    gt_text = ex.get("volume_label", "")
                    if not gt_text and "volume_ml" in ex:
                        gt_text = f"{ex['volume_ml']} mL"
                    gt_volume = self.data_processor.extract_volume_from_text(gt_text)
                    ground_truth.append(gt_volume)

                    if "florence" in model_name.lower():
                        prompt = "<VQA>What is the volume of liquid in the beaker?"
                        inputs = processor(images=image, text=prompt, return_tensors="pt")
                        model_dtype = next(model.parameters()).dtype
                        inputs = {
                            k: (v.to(model.device).to(model_dtype) if v.dtype.is_floating_point else v.to(model.device))
                            for k, v in inputs.items()
                        }
                        gen = model.generate(**inputs, max_new_tokens=30)
                        txt = processor.batch_decode(gen, skip_special_tokens=True)[0]
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
                                    {"type": "image", "image": image},
                                    {"type": "text", "text": "What is the volume of liquid in this beaker in mL?"},
                                ],
                            },
                        ]
                        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = processor(text=[text], images=[image], return_tensors="pt")
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}
                        gen = model.generate(**inputs, max_new_tokens=20, do_sample=False)
                        txt = processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

                    pred = self.data_processor.extract_volume_from_text(txt)
                    predictions.append(pred)

                    if (idx + 1) % 50 == 0:
                        print(f"  ✓ Evaluated {idx + 1}/{len(test_data)}")

                except Exception as e:
                    print(f"⚠️ Eval error {idx}: {e}")
                    predictions.append(0.0)

        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)

        mae = mean_absolute_error(ground_truth, predictions)
        rmse = np.sqrt(mean_squared_error(ground_truth, predictions))
        r2 = r2_score(ground_truth, predictions)

        print(f"\n📈 {model_name} Results:")
        print(f"   MAE:  {mae:.2f} mL")
        print(f"   RMSE: {rmse:.2f} mL")
        print(f"   R²:   {r2:.4f}")

        return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("🚀 Vision Model Training Pipeline - Florence-2 & Qwen2.5-VL (P100)")
    print("=" * 80)

    config = Config()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.TEST_IMAGES_DIR, exist_ok=True)

    if torch.cuda.is_available():
        print(f"\n✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print_memory_usage()
    else:
        print("\n❌ No GPU detected. This script is intended for P100 GPU.")

    data_processor = DatasetProcessor(config)
    train_data, val_data, test_data = data_processor.load_and_split_dataset()
    data_processor.save_test_images(test_data, config.TEST_IMAGES_DIR)

    # Train Florence
    florence_trainer = FlorenceTrainer(config)
    florence_path = florence_trainer.train(train_data, val_data)
    del florence_trainer
    clear_memory()
    print("🧹 Cleared memory after Florence training.")
    print_memory_usage()

    # Train Qwen
    qwen_trainer = QwenTrainer(config)
    qwen_path = qwen_trainer.train(train_data, val_data)

    # Evaluate
    evaluator = ModelEvaluator(config)

    florence_eval = FlorenceTrainer(config)
    florence_eval.setup_model()
    florence_results = evaluator.evaluate_model(florence_eval.model, florence_eval.processor, test_data, "Florence-2")
    del florence_eval
    clear_memory()

    qwen_results = evaluator.evaluate_model(qwen_trainer.model, qwen_trainer.processor, test_data, "Qwen2.5-VL")

    results = {
        "florence2": florence_results,
        "qwen2_5vl": qwen_results,
        "config": {
            "gpu_target": "Tesla P100 (16GB)",
            "use_4bit": bool(Config.USE_4BIT and _HAS_BNB),
            "max_length": Config.MAX_LENGTH,
            "batch_size": Config.BATCH_SIZE,
            "gradient_accumulation": Config.GRADIENT_ACCUMULATION,
            "effective_batch": Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION,
        },
        "model_paths": {"florence2": florence_path, "qwen2_5vl": qwen_path},
    }

    results_path = f"{config.OUTPUT_DIR}/evaluation_results_p100.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE (P100)")
    print("=" * 80)
    print(f"Florence-2 saved: {florence_path}")
    print(f"Qwen2.5-VL saved: {qwen_path}")
    print(f"Results saved: {results_path}")
    print(f"Test images: {config.TEST_IMAGES_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        clear_memory()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        clear_memory()

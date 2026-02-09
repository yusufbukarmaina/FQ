"""
Complete Training Pipeline for Florence-2 and Qwen2.5-VL
Updated version: fixes Florence-2 OverflowError (negative int to unsigned) by:
  - avoiding padding="max_length" in Florence processor/tokenizer calls
  - using short label max_length=64
  - masking pad tokens in labels to -100
  - setting dataloader_num_workers=0 for stability
Also removes trust_remote_code from load_dataset (per HF warning).
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

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Dataset settings
    HF_DATASET_NAME = "yusufbukarmaina/Beakers1"
    STREAMING = True  # Essential for memory management

    # Dataset sizes
    TRAIN_SAMPLES = 500
    VAL_SAMPLES = 150
    TEST_SAMPLES = 300
    TOTAL_SAMPLES = 1600

    # Model settings
    FLORENCE_MODEL = "microsoft/Florence-2-base"
    QWEN_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

    # LoRA settings
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

    # Training settings
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 8
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 10
    WARMUP_STEPS = 50

    # NOTE: keep MAX_LENGTH for Qwen; Florence will use dynamic padding
    MAX_LENGTH = 256

    # Memory optimization
    FP16 = True
    GRADIENT_CHECKPOINTING = True

    # Output settings
    OUTPUT_DIR = "./trained_models"
    TEST_IMAGES_DIR = "./test_images"
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
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


# ============================================================================
# DATA PROCESSING
# ============================================================================

class DatasetProcessor:
    def __init__(self, config: Config):
        self.config = config

    def load_and_split_dataset(self):
        print("📥 Loading dataset with streaming...")
        print(f"Dataset: {self.config.HF_DATASET_NAME}")
        print(
            f"Target: {self.config.TRAIN_SAMPLES} train, "
            f"{self.config.VAL_SAMPLES} val, {self.config.TEST_SAMPLES} test"
        )

        # ✅ FIX: remove trust_remote_code (HF warning)
        dataset = load_dataset(
            self.config.HF_DATASET_NAME,
            split="train",
            streaming=self.config.STREAMING,
        )

        dataset = dataset.shuffle(seed=42, buffer_size=1000)
        print("📊 Creating splits with streaming...")

        train_data, val_data, test_data = [], [], []
        total_processed = 0
        skipped = 0

        for example in dataset:
            if "image" not in example:
                skipped += 1
                continue
            if ("volume_ml" not in example) and ("volume_label" not in example):
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
                print(
                    f"✓ Processed {total_processed} - "
                    f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
                )

            if total_processed > self.config.TOTAL_SAMPLES + 500:
                print("⚠️ Reached safety limit, stopping data collection")
                break

        print("\n✅ Dataset split complete:")
        print(f"   Train: {len(train_data)} examples")
        print(f"   Val: {len(val_data)} examples")
        print(f"   Test: {len(test_data)} examples")
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
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))

        numbers = re.findall(r"\d+\.?\d*", text)
        if numbers:
            return float(numbers[0])
        return 0.0

    def save_test_images(self, test_data: List[Dict], output_dir: str):
        print(f"\n💾 Saving {len(test_data)} test images to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)

        metadata = []
        for idx, example in enumerate(test_data):
            try:
                image = example["image"]
                if isinstance(image, str):
                    image = Image.open(image).convert("RGB")
                else:
                    image = image.convert("RGB")

                gt_text = example.get("volume_label", "")
                if (not gt_text) and ("volume_ml" in example):
                    gt_text = f"{example['volume_ml']} mL"
                gt_volume = self.extract_volume_from_text(gt_text)

                filename = f"test_{idx:04d}_volume_{gt_volume:.1f}mL.jpg"
                save_path = os.path.join(output_dir, filename)
                image.save(save_path, quality=95)

                metadata.append(
                    {
                        "index": idx,
                        "filename": filename,
                        "ground_truth_volume": gt_volume,
                        "ground_truth_text": gt_text,
                    }
                )

                if (idx + 1) % 50 == 0:
                    print(f"  Saved {idx + 1}/{len(test_data)} images...")
            except Exception as e:
                print(f"⚠️ Error saving test image {idx}: {e}")
                continue

        metadata_path = os.path.join(output_dir, "test_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Saved {len(test_data)} test images to {output_dir}")
        print(f"✅ Saved metadata to {metadata_path}")


# ============================================================================
# FLORENCE-2 TRAINING
# ============================================================================

class FlorenceTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.processor = None
        self.model = None

    def setup_model(self):
        print(f"\n🤖 Setting up Florence-2 model: {self.config.FLORENCE_MODEL}")
        print("⚙️ Memory optimization: FP16 enabled")

        clear_memory()

        self.processor = AutoProcessor.from_pretrained(
            self.config.FLORENCE_MODEL,
            trust_remote_code=True,  # Florence still needs this
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.FLORENCE_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.config.FP16 else torch.float32,
            low_cpu_mem_usage=True,
        ).to("cuda")

        print_memory_usage()

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
        print("\n🚀 Starting Florence-2 training...")
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")

        self.setup_model()

        output_dir = f"{self.config.OUTPUT_DIR}/florence2"
        os.makedirs(output_dir, exist_ok=True)

        # ✅ FIX: dataloader_num_workers=0 for stability; pin_memory off
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
            push_to_hub=False,
            report_to="tensorboard",
            logging_dir=f"{output_dir}/runs",
            gradient_checkpointing=self.config.GRADIENT_CHECKPOINTING,
            optim="adamw_torch",
            max_grad_norm=1.0,
        )

        class FlorenceDataset(torch.utils.data.Dataset):
            """
            ✅ FIX: Avoid padding="max_length" for Florence to prevent
            OverflowError: can't convert negative int to unsigned
            """
            def __init__(self, data, processor, config):
                self.data = data
                self.processor = processor
                self.config = config
                self.pad_id = processor.tokenizer.pad_token_id

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                ex = self.data[idx]

                image = ex["image"]
                if isinstance(image, str):
                    image = Image.open(image).convert("RGB")
                else:
                    image = image.convert("RGB")

                prompt = "<VQA>What is the volume of liquid in the beaker?"
                answer = ex.get("volume_label", "")
                if (not answer) and ("volume_ml" in ex):
                    answer = f"{ex['volume_ml']} mL"
                answer = str(answer).strip()
                if not answer:
                    answer = "0 mL"

                # ✅ FIX: dynamic padding (no max_length) for inputs
                inputs = self.processor(
                    images=image,
                    text=prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )

                # ✅ FIX: keep labels short and stable
                ans = self.processor.tokenizer(
                    answer,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=64,
                )

                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                labels = ans["input_ids"].squeeze(0).clone()

                # ✅ FIX: ignore pad tokens in loss
                labels[labels == self.pad_id] = -100
                inputs["labels"] = labels

                return inputs

        train_dataset = FlorenceDataset(train_data, self.processor, self.config)
        eval_dataset = FlorenceDataset(val_data, self.processor, self.config)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        print("Starting training loop...")
        print_memory_usage()
        trainer.train()
        print("Training completed successfully!")

        final_dir = f"{self.config.OUTPUT_DIR}/florence2_final"
        print(f"\n💾 Saving final model to {final_dir}")
        trainer.save_model(final_dir)
        self.processor.save_pretrained(final_dir)

        print(f"✅ Florence-2 training complete! Model saved to {final_dir}")
        clear_memory()
        return final_dir


# ============================================================================
# QWEN2.5-VL TRAINING (unchanged, but you can also switch to dynamic padding)
# ============================================================================

class QwenTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.processor = None
        self.model = None

    def setup_model(self):
        print(f"\n🤖 Setting up Qwen2.5-VL model: {self.config.QWEN_MODEL}")
        print("⚙️ Memory optimization: FP16 enabled")

        clear_memory()

        self.processor = AutoProcessor.from_pretrained(
            self.config.QWEN_MODEL,
            trust_remote_code=True
        )

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
                ).to("cuda")
            else:
                raise

        print_memory_usage()

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
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        print_memory_usage()
        return self.model, self.processor

    def train(self, train_data: List[Dict], val_data: List[Dict]) -> str:
        print("\n🚀 Starting Qwen2.5-VL training...")
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")

        self.setup_model()

        output_dir = f"{self.config.OUTPUT_DIR}/qwen2_5vl"
        os.makedirs(output_dir, exist_ok=True)

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
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=self.config.FP16,
            dataloader_num_workers=0,      # ✅ more stable
            dataloader_pin_memory=False,   # ✅ more stable
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="tensorboard",
            logging_dir=f"{output_dir}/runs",
            gradient_checkpointing=self.config.GRADIENT_CHECKPOINTING,
            optim="adamw_torch",
            max_grad_norm=1.0
        )

        class QwenDataset(torch.utils.data.Dataset):
            def __init__(self, data, processor, config):
                self.data = data
                self.processor = processor
                self.config = config

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                example = self.data[idx]

                image = example["image"]
                if isinstance(image, str):
                    image = Image.open(image).convert("RGB")
                else:
                    image = image.convert("RGB")

                question = "What is the volume of liquid in this beaker in mL?"
                if "volume_ml" in example:
                    answer = f"{example['volume_ml']} mL"
                elif "volume_label" in example:
                    answer = str(example["volume_label"])
                else:
                    answer = "unknown"

                messages = [
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
                    add_generation_prompt=False
                )

                inputs = self.processor(
                    text=[text],
                    images=[image],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.config.MAX_LENGTH
                )

                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                inputs["labels"] = inputs["input_ids"].clone()
                return inputs

        train_dataset = QwenDataset(train_data, self.processor, self.config)
        eval_dataset = QwenDataset(val_data, self.processor, self.config)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        print("Starting training loop...")
        print_memory_usage()
        trainer.train()
        print("Training completed successfully!")

        final_dir = f"{self.config.OUTPUT_DIR}/qwen2_5vl_final"
        print(f"\n💾 Saving final model to {final_dir}")
        trainer.save_model(final_dir)
        self.processor.save_pretrained(final_dir)

        print(f"✅ Qwen2.5-VL training complete! Model saved to {final_dir}")
        clear_memory()
        return final_dir


# ============================================================================
# EVALUATION
# ============================================================================

class ModelEvaluator:
    def __init__(self, config: Config):
        self.config = config
        self.data_processor = DatasetProcessor(config)

    def evaluate_model(self, model, processor, test_data: List[Dict], model_name: str) -> Dict:
        print(f"\n📊 Evaluating {model_name} on {len(test_data)} test samples...")

        predictions = []
        ground_truth = []

        model.eval()
        clear_memory()

        with torch.no_grad():
            for idx, example in enumerate(test_data):
                try:
                    image = example["image"]
                    if isinstance(image, str):
                        image = Image.open(image).convert("RGB")
                    else:
                        image = image.convert("RGB")

                    gt_text = example.get("volume_label", "")
                    if (not gt_text) and ("volume_ml" in example):
                        gt_text = f"{example['volume_ml']} mL"
                    gt_volume = self.data_processor.extract_volume_from_text(gt_text)
                    ground_truth.append(gt_volume)

                    if "florence" in model_name.lower():
                        prompt = "<VQA>What is the volume of liquid in the beaker?"
                        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
                        generated_ids = model.generate(
                            **inputs, max_new_tokens=50, num_beams=3, early_stopping=True
                        )
                        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    else:
                        question = "What is the volume of liquid in this beaker in mL?"
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": image},
                                    {"type": "text", "text": question},
                                ],
                            }
                        ]
                        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
                        generated_ids = model.generate(
                            **inputs, max_new_tokens=50, num_beams=3, early_stopping=True
                        )
                        generated_text = processor.batch_decode(
                            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]

                    pred_volume = self.data_processor.extract_volume_from_text(generated_text)
                    predictions.append(pred_volume)

                    if (idx + 1) % 50 == 0:
                        print(f"  ✓ Evaluated {idx + 1}/{len(test_data)} samples")

                except Exception as e:
                    print(f"⚠️ Error evaluating example {idx}: {e}")
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

        # NOTE: Your plot function used subplots; keep as-is if you like.
        return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("🚀 Vision Model Training Pipeline - Florence-2 & Qwen2.5-VL")
    print("=" * 80)

    config = Config()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.TEST_IMAGES_DIR, exist_ok=True)

    if torch.cuda.is_available():
        print(f"\n✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print_memory_usage()
    else:
        print("\n❌ WARNING: No GPU detected! Training will be very slow.")

    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    data_processor = DatasetProcessor(config)
    train_data, val_data, test_data = data_processor.load_and_split_dataset()

    print("\n" + "=" * 80)
    print("EXPORTING TEST IMAGES")
    print("=" * 80)
    data_processor.save_test_images(test_data, config.TEST_IMAGES_DIR)

    print("\n" + "=" * 80)
    print("FLORENCE-2 TRAINING")
    print("=" * 80)
    florence_trainer = FlorenceTrainer(config)
    florence_path = florence_trainer.train(train_data, val_data)

    del florence_trainer
    clear_memory()
    print("\n🧹 Cleared memory after Florence-2 training")
    print_memory_usage()

    print("\n" + "=" * 80)
    print("QWEN2.5-VL TRAINING")
    print("=" * 80)
    qwen_trainer = QwenTrainer(config)
    qwen_path = qwen_trainer.train(train_data, val_data)

    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    evaluator = ModelEvaluator(config)

    print("\n--- Florence-2 Evaluation ---")
    flor_eval = FlorenceTrainer(config)
    flor_eval.setup_model()
    florence_results = evaluator.evaluate_model(flor_eval.model, flor_eval.processor, test_data, "Florence-2")
    del flor_eval
    clear_memory()

    print("\n--- Qwen2.5-VL Evaluation ---")
    qwen_results = evaluator.evaluate_model(qwen_trainer.model, qwen_trainer.processor, test_data, "Qwen2.5-VL")

    results = {
        "florence2": florence_results,
        "qwen2_5vl": qwen_results,
        "config": {
            "dataset": config.HF_DATASET_NAME,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "epochs": config.NUM_EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "gradient_accumulation": config.GRADIENT_ACCUMULATION,
            "effective_batch_size": config.BATCH_SIZE * config.GRADIENT_ACCUMULATION,
            "learning_rate": config.LEARNING_RATE,
            "fp16": config.FP16,
            "gradient_checkpointing": config.GRADIENT_CHECKPOINTING,
        },
    }

    results_path = f"{config.OUTPUT_DIR}/evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModels saved to:")
    print(f"  Florence-2:   {florence_path}")
    print(f"  Qwen2.5-VL:   {qwen_path}")
    print(f"\nResults:        {results_path}")
    print(f"Test images:    {config.TEST_IMAGES_DIR}")


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

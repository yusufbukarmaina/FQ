# ⚡ Quick Start - RTX 6000 Training (yusufbukarmaina/Beakers1)

## 🎯 Your Exact Setup

- **GPU**: RTX 6000 (24GB VRAM)
- **Dataset**: yusufbukarmaina/Beakers1 (already on HuggingFace)
- **Training**: 1000 samples, Validation: 300 samples, Test: 300 samples
- **Output**: Test images saved to `/test_images/` folder

---

## 🚀 Step-by-Step Instructions

### 1. Install Dependencies (5 minutes)

```bash
# Essential packages
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.44.0 datasets==2.18.0 accelerate==0.33.0
pip install peft==0.11.0 bitsandbytes==0.43.0 scikit-learn scipy
pip install matplotlib seaborn gradio Pillow numpy pandas tqdm
pip install huggingface-hub tensorboard
```

### 2. Login to Hugging Face

```bash
huggingface-cli login
# Paste your token from: https://huggingface.co/settings/tokens
```

### 3. Verify Setup

```bash
# Check GPU
nvidia-smi

# Test CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test dataset access
python -c "from datasets import load_dataset; ds = load_dataset('yusufbukarmaina/Beakers1', split='train', streaming=True); print(next(iter(ds)))"
```

### 4. Start Training (3-5 hours)

```bash
# Use screen to keep session alive
screen -S training

# Run training
python train_vision_models.py

# Detach from screen: Ctrl+A then D
# Reattach later: screen -r training
```

---

## 📊 What to Expect

### Training Timeline (RTX 6000):
- **Dataset Loading**: 2-3 minutes
- **Florence-2 Training**: 1.5-2 hours (10 epochs, 1000 samples)
- **Qwen2.5-VL Training**: 2-3 hours (10 epochs, 1000 samples)
- **Evaluation**: 10-15 minutes
- **Total**: ~4-5 hours

### Memory Usage:
- **Florence-2**: ~12GB VRAM during training
- **Qwen2.5-VL**: ~18GB VRAM during training
- **Peak**: ~20GB (safe for 24GB RTX 6000)

### Output Structure:
```
./
├── trained_models/
│   ├── florence2_final/          # Trained Florence-2 model
│   ├── qwen2_5vl_final/          # Trained Qwen2.5-VL model
│   ├── evaluation_results.json   # Metrics (MAE, RMSE, R²)
│   ├── Florence-2_evaluation.png # Prediction plot
│   └── Qwen2.5-VL_evaluation.png # Prediction plot
└── test_images/                  # 300 test images exported here
    ├── test_0000_volume_250.0mL.jpg
    ├── test_0001_volume_125.5mL.jpg
    ├── ...
    └── test_metadata.json        # Ground truth labels
```

---

## 🔥 Key Features in Fixed Script

### 1. Memory Optimizations for RTX 6000:
- ✅ Batch size: 2 (optimized for 24GB)
- ✅ Gradient accumulation: 8 (effective batch size = 16)
- ✅ FP16 precision (halves memory usage)
- ✅ Gradient checkpointing enabled

### 2. Crash Prevention:
- ✅ Robust error handling in data loading
- ✅ Automatic memory cleanup between models
- ✅ Dummy sample fallback for corrupted images
- ✅ Progress tracking every 100 samples

### 3. Data Management:
- ✅ Streaming mode (prevents OOM on large datasets)
- ✅ Exact 1000/300/300 split
- ✅ Test images automatically exported
- ✅ Metadata JSON with ground truth volumes

---

## 📈 Monitoring Training

### Terminal 1 - Training:
```bash
screen -S training
python train_vision_models.py
```

### Terminal 2 - GPU Monitoring:
```bash
watch -n 1 nvidia-smi
```

### Terminal 3 - TensorBoard (Optional):
```bash
tensorboard --logdir=./trained_models --port=6006
# Access at: http://localhost:6006
```

---

## 🎯 Expected Output

```
================================================================================
🚀 Vision Model Training Pipeline - Florence-2 & Qwen2.5-VL
================================================================================
GPU: RTX 6000 (24GB VRAM)
Configuration:
  Dataset: yusufbukarmaina/Beakers1
  Train samples: 1000
  Val samples: 300
  Test samples: 300
  Batch size: 2
  Gradient accumulation: 8
  Effective batch size: 16
  FP16: True
  Gradient checkpointing: True
================================================================================

📥 Loading dataset with streaming...
Dataset: yusufbukarmaina/Beakers1
Target: 1000 train, 300 val, 300 test

📊 Creating splits with streaming...
✓ Processed 100 - Train: 62, Val: 19, Test: 19
✓ Processed 200 - Train: 125, Val: 37, Test: 38
...
✓ Processed 1600 - Train: 1000, Val: 300, Test: 300

✅ Dataset split complete:
   Train: 1000 examples
   Val: 300 examples
   Test: 300 examples

💾 Saving 300 test images to ./test_images/...
  Saved 50/300 images...
  Saved 100/300 images...
  ...
✅ Saved 300 test images to ./test_images/
✅ Saved metadata to ./test_images/test_metadata.json

🤖 Setting up Florence-2 model: microsoft/Florence-2-base
GPU Memory: 2.35GB allocated, 2.50GB reserved
✓ Gradient checkpointing enabled
trainable params: 2,359,296 || all params: 232,359,296 || trainable%: 1.0156

🚀 Starting Florence-2 training...
Starting training loop...
GPU Memory: 11.87GB allocated, 12.20GB reserved
Epoch 1/10: 100%|███████| 125/125 [08:45<00:00, loss=0.234]
Epoch 2/10: 100%|███████| 125/125 [08:42<00:00, loss=0.156]
...

✅ Florence-2 training complete!

[Same process for Qwen2.5-VL]

📊 Evaluating Florence-2 on 300 test samples...
  ✓ Evaluated 50/300 samples
  ...

📈 Florence-2 Results:
   MAE:  12.34 mL
   RMSE: 18.67 mL
   R²:   0.9234
   📊 Plot saved to: ./trained_models/Florence-2_evaluation.png

🎉 TRAINING COMPLETE!
Models saved to:
  Florence-2:   ./trained_models/florence2_final
  Qwen2.5-VL:   ./trained_models/qwen2_5vl_final
Results:        ./trained_models/evaluation_results.json
Test images:    ./test_images
```

---

## ⚠️ Common Issues & Solutions

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size
```python
# Edit train_vision_models.py line 50:
BATCH_SIZE = 1  # Changed from 2
GRADIENT_ACCUMULATION = 16  # Changed from 8
```

### Issue: Training hangs at dataset loading
**Solution**: Check Hugging Face login
```bash
huggingface-cli whoami
# If error: huggingface-cli login
```

### Issue: Can't find test images
**Solution**: Check the folder
```bash
ls -lh ./test_images/
cat ./test_images/test_metadata.json
```

### More issues?
See `TROUBLESHOOTING.md` for comprehensive guide.

---

## 🎨 After Training

### 1. View Results:
```bash
# Check metrics
cat ./trained_models/evaluation_results.json

# View evaluation plots
ls ./trained_models/*.png
```

### 2. Test Models (Gradio Demo):
```bash
python gradio_demo.py --share
# Click the public URL to test your models
```

### 3. Examine Test Images:
```bash
cd test_images
ls -lh
# All 300 test images are here with ground truth in filenames
```

---

## 📌 Configuration Summary

| Setting | Value | Reason |
|---------|-------|--------|
| Batch size | 2 | Fits in 24GB VRAM |
| Gradient accumulation | 8 | Effective batch = 16 |
| FP16 | True | Halves memory usage |
| Gradient checkpointing | True | Saves ~30% memory |
| Max length | 512 | Standard for vision tasks |
| Learning rate | 2e-4 | Optimal for LoRA |
| Epochs | 10 | Good balance |
| Train/Val/Test | 1000/300/300 | Your specification |

---

## 🏁 Success Checklist

- [ ] Dependencies installed
- [ ] GPU detected (nvidia-smi shows RTX 6000)
- [ ] Hugging Face login successful
- [ ] Dataset accessible (test script works)
- [ ] Training started (see "Starting training loop...")
- [ ] No CUDA OOM errors
- [ ] Models saving checkpoints
- [ ] Test images exported to ./test_images/
- [ ] Training completed successfully
- [ ] Evaluation results generated

---

## 💡 Pro Tips

1. **Use Screen**: Always run in screen/tmux for long training
   ```bash
   screen -S training
   # Detach: Ctrl+A then D
   # Reattach: screen -r training
   ```

2. **Monitor GPU**: Keep nvidia-smi running in another terminal
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Save Logs**: Capture all output
   ```bash
   python train_vision_models.py 2>&1 | tee training.log
   ```

4. **Check Progress**: Training prints progress every 50 steps
   Look for: "Epoch X/10: Y%|███"

---

**Ready to train! Your dataset is already set up correctly.** 🚀

Just run:
```bash
screen -S training
python train_vision_models.py
```

Good luck! 🎉

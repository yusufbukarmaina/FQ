# 🔧 TROUBLESHOOTING GUIDE - RTX 6000 Training

## 🚨 Common Crash Issues and Fixes

### Issue 1: CUDA Out of Memory (OOM)

**Symptoms:**
- Training crashes with `CUDA out of memory` error
- Process killed without error message
- GPU memory usage at 100%

**Solutions:**

1. **Reduce batch size** (already optimized to 2 in fixed script):
```python
BATCH_SIZE = 1  # Try 1 if still crashing
GRADIENT_ACCUMULATION = 16  # Increase to maintain effective batch size
```

2. **Disable gradient checkpointing temporarily** (use more memory but might be more stable):
```python
GRADIENT_CHECKPOINTING = False
```

3. **Reduce sequence length**:
```python
MAX_LENGTH = 256  # Reduced from 512
```

4. **Monitor memory before crash**:
```bash
watch -n 0.5 nvidia-smi
```

---

### Issue 2: Dataset Loading Hangs/Freezes

**Symptoms:**
- Script hangs at "Loading dataset with streaming..."
- No progress after "Creating splits with streaming..."

**Solutions:**

1. **Check dataset accessibility**:
```python
from datasets import load_dataset
ds = load_dataset("yusufbukarmaina/Beakers1", split="train", streaming=True)
print(next(iter(ds)))
```

2. **Verify Hugging Face login**:
```bash
huggingface-cli whoami
# If not logged in:
huggingface-cli login
```

3. **Disable streaming if dataset is small enough**:
```python
STREAMING = False  # Loads entire dataset into RAM
```

4. **Check internet connection**:
```bash
ping huggingface.co
```

---

### Issue 3: Training Starts but Crashes Mid-Training

**Symptoms:**
- Training begins successfully
- Crashes after a few steps/epochs
- Loss becomes NaN

**Solutions:**

1. **Enable mixed precision safely**:
```python
FP16 = True
MAX_GRAD_NORM = 1.0  # Clip gradients
```

2. **Reduce learning rate**:
```python
LEARNING_RATE = 1e-4  # Reduced from 2e-4
```

3. **Add more warmup steps**:
```python
WARMUP_STEPS = 100  # Increased from 50
```

4. **Check for corrupted data**:
- Review the data loading loop
- Add more robust error handling

---

### Issue 4: Model Loading Fails

**Symptoms:**
- "Error loading model" messages
- Import errors for `Qwen2VLForConditionalGeneration`
- Trust remote code errors

**Solutions:**

1. **Update transformers**:
```bash
pip install --upgrade transformers==4.44.0
```

2. **Explicitly allow remote code**:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,  # Essential!
    revision="main"
)
```

3. **Clear Hugging Face cache**:
```bash
rm -rf ~/.cache/huggingface/hub/
```

---

### Issue 5: Dataloader Workers Crash

**Symptoms:**
- "DataLoader worker exited unexpectedly"
- Multiprocessing errors
- Segmentation faults

**Solutions:**

1. **Reduce or disable workers**:
```python
dataloader_num_workers=0  # Changed from 2
```

2. **Disable pin memory**:
```python
dataloader_pin_memory=False
```

3. **Set proper multiprocessing start method**:
```python
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
```

---

## 🎯 Optimization for RTX 6000 (24GB)

### Current Optimized Settings:
```python
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
FP16 = True
GRADIENT_CHECKPOINTING = True
MAX_LENGTH = 512
```

### Memory Usage Breakdown:
- **Florence-2**: ~10-12GB during training
- **Qwen2.5-VL**: ~16-18GB during training
- **Safe for 24GB** with current settings

### If You Have More VRAM:
```python
# For 32GB+ VRAM:
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4

# For 48GB VRAM (A6000):
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
GRADIENT_CHECKPOINTING = False
```

---

## 📊 Monitoring Training

### Essential Commands:

1. **GPU monitoring**:
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Temperature and power
nvidia-smi dmon

# Detailed memory usage
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv -l 1
```

2. **Training logs**:
```bash
# Save all output
python train_vision_models_fixed.py 2>&1 | tee training.log

# Monitor logs
tail -f training.log
```

3. **TensorBoard**:
```bash
tensorboard --logdir=./trained_models --port=6006
```

---

## 🔍 Debugging Steps

### Step 1: Minimal Test
```python
# Test basic model loading
python -c "
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    'microsoft/Florence-2-base',
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map='auto'
)
print('Model loaded successfully!')
print(f'Memory used: {torch.cuda.memory_allocated() / 1e9:.2f}GB')
"
```

### Step 2: Test Dataset
```python
# Test dataset loading
python -c "
from datasets import load_dataset

ds = load_dataset('yusufbukarmaina/Beakers1', split='train', streaming=True)
example = next(iter(ds))
print('Dataset loaded successfully!')
print(f'Keys: {example.keys()}')
"
```

### Step 3: Test Single Batch
```python
# Test single training step
# Add this to your script temporarily:
print("Testing single batch...")
for batch in train_dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    print(f"Loss: {loss.item()}")
    break
print("Single batch test passed!")
```

---

## 🛠️ Quick Fixes

### Fix 1: Reset Everything
```bash
# Clear all caches and start fresh
rm -rf ~/.cache/huggingface/
rm -rf ./trained_models/
pip install --upgrade transformers datasets peft accelerate
```

### Fix 2: Use Screen for Long Sessions
```bash
# Start screen session
screen -S training

# Run training
python train_vision_models_fixed.py

# Detach: Ctrl+A then D
# Reattach: screen -r training
```

### Fix 3: Emergency Stop and Resume
```bash
# Stop gracefully: Ctrl+C
# Training will save checkpoint

# Resume from checkpoint
# (automatic if checkpoints exist)
python train_vision_models_fixed.py
```

---

## 📝 Pre-Flight Checklist

Before starting training:

- [ ] GPU detected: `nvidia-smi`
- [ ] PyTorch with CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Logged into Hugging Face: `huggingface-cli whoami`
- [ ] Dataset accessible: Test with minimal script
- [ ] Enough disk space: `df -h` (need ~40GB)
- [ ] Models can load: Test Florence-2 and Qwen2 separately
- [ ] Using screen/tmux: `screen -S training`

---

## 🚀 Complete Setup Script

```bash
#!/bin/bash
# Run this if starting fresh

# 1. Update system
sudo apt-get update
sudo apt-get install -y tmux htop

# 2. Install PyTorch with CUDA
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 3. Install dependencies
pip install transformers==4.44.0 datasets==2.18.0 accelerate==0.33.0
pip install peft==0.11.0 bitsandbytes==0.43.0
pip install scikit-learn scipy matplotlib seaborn
pip install gradio Pillow numpy pandas tqdm huggingface-hub tensorboard

# 4. Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 5. Login to Hugging Face
huggingface-cli login

# 6. Test dataset
python -c "from datasets import load_dataset; ds = load_dataset('yusufbukarmaina/Beakers1', split='train', streaming=True); print(next(iter(ds)))"

echo "Setup complete!"
```

---

## 💡 Performance Tips

### 1. Faster Dataset Loading:
```python
# Enable Hugging Face transfer
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
# Then: pip install hf-transfer
```

### 2. Better Checkpointing:
```python
SAVE_STEPS = 250  # More frequent saves
SAVE_TOTAL_LIMIT = 3  # Keep more checkpoints
```

### 3. Learning Rate Scheduling:
```python
# Add to TrainingArguments:
lr_scheduler_type="cosine"
warmup_ratio=0.1
```

---

## 📞 Getting Help

If issues persist:

1. **Check training logs** for specific error messages
2. **Run diagnostics**:
   ```bash
   python -m torch.utils.collect_env
   ```
3. **Test with minimal config** (1 epoch, 100 samples)
4. **Check GitHub issues** for transformers/peft/datasets

---

## 🎉 Success Indicators

Your training is going well if you see:

✅ "✓ Processed X - Train: Y, Val: Z, Test: W"  
✅ "GPU Memory: X.XXG allocated"  
✅ "trainable params: X || all params: Y || trainable%: Z"  
✅ "Starting training loop..."  
✅ Loss decreasing each epoch  
✅ Evaluation running successfully  
✅ Models saved to ./trained_models/  

---

**Good luck with your training! 🚀**

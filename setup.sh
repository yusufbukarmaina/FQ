#!/bin/bash
# ============================================================================
# Complete Setup Script for JarvisLab A6000 GPU
# Run this script step-by-step to train Florence-2 and Qwen2.5-VL models
# ============================================================================

set -e  # Exit on error

echo "=================================="
echo "🚀 Vision Model Training Setup"
echo "=================================="
echo ""

# ============================================================================
# STEP 1: System Setup
# ============================================================================

echo "📦 STEP 1/6: Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y git wget curl vim htop

# ============================================================================
# STEP 2: Python Environment
# ============================================================================

echo ""
echo "🐍 STEP 2/6: Setting up Python environment..."

# Check Python version
python --version

# Upgrade pip
pip install --upgrade pip setuptools wheel

# ============================================================================
# STEP 3: Install Dependencies
# ============================================================================

echo ""
echo "📚 STEP 3/6: Installing Python packages..."

# Install PyTorch with CUDA support (for A6000)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Install flash attention for faster training (optional)
# pip install flash-attn --no-build-isolation

echo "✅ All dependencies installed!"

# ============================================================================
# STEP 4: Verify GPU
# ============================================================================

echo ""
echo "🖥️  STEP 4/6: Verifying GPU setup..."

python << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
EOF

# ============================================================================
# STEP 5/6: HuggingFace Login (non-interactive)
# ============================================================================

echo ""
echo "🔐 STEP 5/6: HuggingFace Authentication..."
echo ""

# Check if already logged in
if hf auth whoami >/dev/null 2>&1; then
  echo "✅ Already logged in to Hugging Face."
else
  if [ -n "$HF_TOKEN" ]; then
    echo "Logging in using HF_TOKEN env var..."
    hf auth login --token "$HF_TOKEN" --add-to-git-credential
    echo "✅ Logged in."
  else
    echo "⚠️ HF_TOKEN not set."
    echo "Run: export HF_TOKEN='hf_...'"
    echo "Or you will be prompted now."
    read -s -p "Enter your Hugging Face token (hidden): " HF_TOKEN
    echo ""
    hf auth login --token "$HF_TOKEN" --add-to-git-credential
    echo "✅ Logged in."
  fi
fi

# ============================================================================
# ALL DONE!
# ============================================================================

echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Start training:"
echo "   python train_vision_models.py"
echo ""
echo "2. Monitor training (in another terminal):"
echo "   tensorboard --logdir=./trained_models --port=6006"
echo ""
echo "3. After training, launch demo:"
echo "   python gradio_demo.py --share"
echo ""
echo "=================================="

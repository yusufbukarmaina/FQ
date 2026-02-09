#!/usr/bin/env python3
"""
All-in-One Runner Script
Executes the complete pipeline: setup check -> training -> evaluation -> demo
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import argparse

class Colors:
    """Terminal colors for pretty output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")

def run_command(command, description, check=True):
    """Run a shell command with nice output"""
    print_info(f"{description}...")
    print(f"{Colors.OKBLUE}$ {command}{Colors.ENDC}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print_success(f"{description} completed")
            return True
        else:
            print_error(f"{description} failed")
            return False
    
    except subprocess.CalledProcessError as e:
        print_error(f"{description} failed with error: {e}")
        return False

def check_environment():
    """Check if environment is properly set up"""
    print_header("ENVIRONMENT CHECK")
    
    checks = {
        "Python": "python --version",
        "pip": "pip --version",
        "CUDA": "nvidia-smi",
        "Git": "git --version"
    }
    
    all_passed = True
    
    for name, command in checks.items():
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            print_success(f"{name}: {result.stdout.strip()}")
        except:
            print_error(f"{name}: Not found or not working")
            all_passed = False
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_success(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print_warning("CUDA available but no GPU detected")
    except:
        print_error("PyTorch not installed or CUDA not available")
        all_passed = False
    
    return all_passed

def check_dependencies():
    """Check if required packages are installed"""
    print_header("CHECKING DEPENDENCIES")
    
    required_packages = [
        "torch",
        "transformers",
        "datasets",
        "peft",
        "accelerate",
        "gradio",
        "huggingface_hub"
    ]
    
    all_installed = True
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print_success(f"{package} installed")
        except ImportError:
            print_error(f"{package} NOT installed")
            all_installed = False
    
    if not all_installed:
        print_warning("Some packages are missing. Run: pip install -r requirements.txt")
    
    return all_installed

def check_huggingface_login():
    """Check if user is logged into HuggingFace"""
    print_header("HUGGINGFACE AUTHENTICATION")
    
    try:
        result = subprocess.run(
            "huggingface-cli whoami",
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print_success(f"Logged in as: {result.stdout.strip()}")
        return True
    except:
        print_warning("Not logged in to HuggingFace")
        print_info("Run: huggingface-cli login")
        return False

def check_dataset_config():
    """Check if dataset configuration is set"""
    print_header("CONFIGURATION CHECK")
    
    config_file = Path("train_vision_models.py")
    
    if not config_file.exists():
        print_error("train_vision_models.py not found!")
        return False
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Check if placeholder values are still present
    issues = []
    
    if 'YOUR_USERNAME/beaker-volume-dataset' in content:
        issues.append("Dataset name still has placeholder (YOUR_USERNAME)")
    
    if 'YOUR_USERNAME/beaker-volume-model' in content:
        issues.append("Model repo name still has placeholder (YOUR_USERNAME)")
    
    if issues:
        print_warning("Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print_info("Please edit train_vision_models.py and update the configuration")
        return False
    else:
        print_success("Configuration looks good!")
        return True

def train_models():
    """Run the training script"""
    print_header("STARTING MODEL TRAINING")
    
    print_info("This will take approximately 3-5 hours on A6000 GPU")
    print_info("You can monitor progress with TensorBoard in another terminal:")
    print(f"{Colors.OKBLUE}  tensorboard --logdir=./trained_models{Colors.ENDC}\n")
    
    time.sleep(2)
    
    success = run_command(
        "python train_vision_models.py",
        "Model training"
    )
    
    return success

def launch_demo(share=False):
    """Launch Gradio demo"""
    print_header("LAUNCHING DEMO")
    
    share_flag = "--share" if share else ""
    
    success = run_command(
        f"python gradio_demo.py {share_flag}",
        "Gradio demo"
    )
    
    return success

def main():
    parser = argparse.ArgumentParser(
        description="All-in-one runner for beaker volume training pipeline"
    )
    
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip environment and dependency checks"
    )
    
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only run training, skip demo"
    )
    
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help="Only run demo, skip training"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link for demo"
    )
    
    args = parser.parse_args()
    
    # Print welcome
    print_header("BEAKER VOLUME TRAINING PIPELINE")
    print(f"{Colors.BOLD}Florence-2 & Qwen2.5-VL Fine-tuning{Colors.ENDC}")
    print(f"{Colors.BOLD}Optimized for JarvisLab A6000 GPU{Colors.ENDC}\n")
    
    # Run checks unless skipped
    if not args.skip_checks and not args.demo_only:
        if not check_environment():
            print_error("Environment check failed!")
            sys.exit(1)
        
        if not check_dependencies():
            print_error("Dependency check failed!")
            print_info("Install dependencies with: pip install -r requirements.txt")
            sys.exit(1)
        
        if not check_huggingface_login():
            print_warning("HuggingFace login recommended")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
        
        if not check_dataset_config():
            print_error("Configuration check failed!")
            sys.exit(1)
    
    # Run training unless demo-only
    if not args.demo_only:
        print_header("STARTING TRAINING")
        
        if not args.train_only:
            response = input("Start training now? (y/n): ")
            if response.lower() != 'y':
                print_info("Training cancelled")
                sys.exit(0)
        
        start_time = time.time()
        
        if not train_models():
            print_error("Training failed!")
            sys.exit(1)
        
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        
        print_success(f"Training completed in {hours}h {minutes}m!")
    
    # Run demo unless train-only
    if not args.train_only:
        print_header("LAUNCHING DEMO")
        
        # Check if models exist
        florence_path = Path("./trained_models/florence2_final")
        qwen_path = Path("./trained_models/qwen2_5vl_final")
        
        if not florence_path.exists() and not qwen_path.exists():
            print_error("No trained models found!")
            print_info("Train models first or specify model paths")
            sys.exit(1)
        
        if not args.demo_only:
            response = input("Launch Gradio demo now? (y/n): ")
            if response.lower() != 'y':
                print_info("Demo cancelled")
                sys.exit(0)
        
        launch_demo(share=args.share)
    
    # Final summary
    print_header("PIPELINE COMPLETE")
    
    print_success("All tasks completed successfully!")
    
    print(f"\n{Colors.BOLD}Next steps:{Colors.ENDC}")
    print("  1. Review evaluation results:")
    print("     cat ./trained_models/evaluation_results.json")
    print("  2. View plots:")
    print("     ls ./trained_models/*.png")
    print("  3. Upload models to HuggingFace:")
    print("     Edit train_vision_models.py and set UPLOAD_TO_HF = True")
    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Pipeline interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

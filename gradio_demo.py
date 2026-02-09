"""
Gradio Demo for Beaker Volume Prediction
Supports both Florence-2 and Qwen2.5-VL models
"""

import gradio as gr
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
import re
import os

# ============================================================================
# MODEL LOADER
# ============================================================================

class ModelLoader:
    """Load and manage trained models"""
    
    def __init__(self, florence_path=None, qwen_path=None):
        self.florence_model = None
        self.florence_processor = None
        self.qwen_model = None
        self.qwen_processor = None
        
        if florence_path and os.path.exists(florence_path):
            self.load_florence(florence_path)
        
        if qwen_path and os.path.exists(qwen_path):
            self.load_qwen(qwen_path)
    
    def load_florence(self, model_path):
        """Load Florence-2 model"""
        print(f"Loading Florence-2 from {model_path}...")
        
        self.florence_processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.florence_model.eval()
        print("✅ Florence-2 loaded successfully!")
    
    def load_qwen(self, model_path):
        """Load Qwen2.5-VL model"""
        print(f"Loading Qwen2.5-VL from {model_path}...")
        
        from transformers import Qwen2VLForConditionalGeneration
        
        self.qwen_processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.qwen_model.eval()
        print("✅ Qwen2.5-VL loaded successfully!")
    
    def extract_volume(self, text):
        """Extract volume value from text"""
        patterns = [
            r'(\d+\.?\d*)\s*mL',
            r'(\d+\.?\d*)\s*ml',
            r'(\d+\.?\d*)\s*milliliters?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            return float(numbers[0])
        
        return None
    
    def predict_florence(self, image, prompt=None):
        """Generate prediction with Florence-2"""
        if self.florence_model is None:
            return "❌ Florence-2 model not loaded", None
        
        if prompt is None:
            prompt = "<VQA>What is the volume of liquid in the beaker?"
        
        try:
            inputs = self.florence_processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.florence_model.device)
            
            with torch.no_grad():
                generated_ids = self.florence_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=3,
                    early_stopping=True
                )
            
            generated_text = self.florence_processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            # Extract volume
            volume = self.extract_volume(generated_text)
            
            return generated_text, volume
        
        except Exception as e:
            return f"❌ Error: {str(e)}", None
    
    def predict_qwen(self, image, question=None):
        """Generate prediction with Qwen2.5-VL"""
        if self.qwen_model is None:
            return "❌ Qwen2.5-VL model not loaded", None
        
        if question is None:
            question = "What is the volume of liquid in this beaker in mL?"
        
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            text = self.qwen_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.qwen_processor(
                text=[text],
                images=[image],
                return_tensors="pt"
            ).to(self.qwen_model.device)
            
            with torch.no_grad():
                generated_ids = self.qwen_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=3,
                    early_stopping=True
                )
            
            generated_text = self.qwen_processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Extract volume
            volume = self.extract_volume(generated_text)
            
            return generated_text, volume
        
        except Exception as e:
            return f"❌ Error: {str(e)}", None


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_demo(florence_path=None, qwen_path=None):
    """Create Gradio demo interface"""
    
    # Load models
    model_loader = ModelLoader(florence_path, qwen_path)
    
    # Define prediction functions
    def predict_with_florence(image, prompt):
        """Predict with Florence-2"""
        if image is None:
            return "⚠️ Please upload an image", "N/A"
        
        text, volume = model_loader.predict_florence(image, prompt)
        
        if volume is not None:
            volume_str = f"{volume:.1f} mL"
        else:
            volume_str = "Could not extract volume"
        
        return text, volume_str
    
    def predict_with_qwen(image, question):
        """Predict with Qwen2.5-VL"""
        if image is None:
            return "⚠️ Please upload an image", "N/A"
        
        text, volume = model_loader.predict_qwen(image, question)
        
        if volume is not None:
            volume_str = f"{volume:.1f} mL"
        else:
            volume_str = "Could not extract volume"
        
        return text, volume_str
    
    def predict_both(image, prompt, question):
        """Predict with both models"""
        if image is None:
            return "⚠️ Please upload an image", "N/A", "⚠️ Please upload an image", "N/A"
        
        # Florence-2
        florence_text, florence_volume = model_loader.predict_florence(image, prompt)
        florence_volume_str = f"{florence_volume:.1f} mL" if florence_volume else "Could not extract"
        
        # Qwen2.5-VL
        qwen_text, qwen_volume = model_loader.predict_qwen(image, question)
        qwen_volume_str = f"{qwen_volume:.1f} mL" if qwen_volume else "Could not extract"
        
        return florence_text, florence_volume_str, qwen_text, qwen_volume_str
    
    # Create interface
    with gr.Blocks(title="Beaker Volume Prediction", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown(
            """
            # 🧪 Beaker Liquid Volume Prediction
            
            Upload an image of a beaker and get volume predictions from Florence-2 and Qwen2.5-VL models.
            Both models have been fine-tuned using LoRA on 2000 beaker images.
            """
        )
        
        with gr.Tabs():
            
            # Tab 1: Florence-2
            with gr.Tab("Florence-2"):
                gr.Markdown("### Florence-2 Model Prediction")
                
                with gr.Row():
                    with gr.Column():
                        florence_image = gr.Image(
                            type="pil",
                            label="Upload Beaker Image"
                        )
                        florence_prompt = gr.Textbox(
                            value="<VQA>What is the volume of liquid in the beaker?",
                            label="Prompt",
                            lines=2
                        )
                        florence_btn = gr.Button("🔍 Predict Volume", variant="primary")
                    
                    with gr.Column():
                        florence_output = gr.Textbox(
                            label="Model Response",
                            lines=4
                        )
                        florence_volume = gr.Textbox(
                            label="Extracted Volume",
                            lines=1
                        )
                
                florence_btn.click(
                    fn=predict_with_florence,
                    inputs=[florence_image, florence_prompt],
                    outputs=[florence_output, florence_volume]
                )
            
            # Tab 2: Qwen2.5-VL
            with gr.Tab("Qwen2.5-VL"):
                gr.Markdown("### Qwen2.5-VL Model Prediction")
                
                with gr.Row():
                    with gr.Column():
                        qwen_image = gr.Image(
                            type="pil",
                            label="Upload Beaker Image"
                        )
                        qwen_question = gr.Textbox(
                            value="What is the volume of liquid in this beaker in mL?",
                            label="Question",
                            lines=2
                        )
                        qwen_btn = gr.Button("🔍 Predict Volume", variant="primary")
                    
                    with gr.Column():
                        qwen_output = gr.Textbox(
                            label="Model Response",
                            lines=4
                        )
                        qwen_volume = gr.Textbox(
                            label="Extracted Volume",
                            lines=1
                        )
                
                qwen_btn.click(
                    fn=predict_with_qwen,
                    inputs=[qwen_image, qwen_question],
                    outputs=[qwen_output, qwen_volume]
                )
            
            # Tab 3: Compare Both
            with gr.Tab("Compare Models"):
                gr.Markdown("### Compare Florence-2 vs Qwen2.5-VL")
                
                with gr.Row():
                    compare_image = gr.Image(
                        type="pil",
                        label="Upload Beaker Image"
                    )
                
                with gr.Row():
                    with gr.Column():
                        compare_prompt = gr.Textbox(
                            value="<VQA>What is the volume of liquid in the beaker?",
                            label="Florence-2 Prompt",
                            lines=2
                        )
                    
                    with gr.Column():
                        compare_question = gr.Textbox(
                            value="What is the volume of liquid in this beaker in mL?",
                            label="Qwen Question",
                            lines=2
                        )
                
                compare_btn = gr.Button("🔍 Compare Both Models", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Florence-2 Results")
                        compare_florence_output = gr.Textbox(
                            label="Response",
                            lines=3
                        )
                        compare_florence_volume = gr.Textbox(
                            label="Volume",
                            lines=1
                        )
                    
                    with gr.Column():
                        gr.Markdown("#### Qwen2.5-VL Results")
                        compare_qwen_output = gr.Textbox(
                            label="Response",
                            lines=3
                        )
                        compare_qwen_volume = gr.Textbox(
                            label="Volume",
                            lines=1
                        )
                
                compare_btn.click(
                    fn=predict_both,
                    inputs=[compare_image, compare_prompt, compare_question],
                    outputs=[
                        compare_florence_output,
                        compare_florence_volume,
                        compare_qwen_output,
                        compare_qwen_volume
                    ]
                )
        
        gr.Markdown(
            """
            ---
            ### 📊 Model Information
            
            - **Florence-2**: Base model fine-tuned with LoRA (r=8, alpha=16)
            - **Qwen2.5-VL**: 2B model fine-tuned with LoRA (r=8, alpha=16)
            - **Training Data**: 1400 images (70%)
            - **Validation Data**: 300 images (15%)
            - **Test Data**: 300 images (15%)
            
            ### 📝 Usage Tips
            
            1. Upload a clear image of a beaker with liquid
            2. The prompt/question can be customized
            3. The model will extract the volume in mL from the response
            4. Use "Compare Models" to see both predictions side-by-side
            """
        )
    
    return demo


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Beaker Volume Prediction Demo")
    parser.add_argument(
        "--florence-path",
        type=str,
        default="./trained_models/florence2_final",
        help="Path to trained Florence-2 model"
    )
    parser.add_argument(
        "--qwen-path",
        type=str,
        default="./trained_models/qwen2_5vl_final",
        help="Path to trained Qwen2.5-VL model"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run demo on"
    )
    
    args = parser.parse_args()
    
    # Create and launch demo
    demo = create_demo(
        florence_path=args.florence_path,
        qwen_path=args.qwen_path
    )
    
    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0"
    )

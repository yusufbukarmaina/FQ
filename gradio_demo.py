"""
Beaker Volume Estimator — Gradio Demo
Tabs:
  1. Live Inference   — upload image → get volume from both models
  2. Florence-2 Metrics — MAE / RMSE / R² + plots
  3. Qwen2.5-VL Metrics — same for Qwen
  4. About
"""

import os, json, re, torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr
from transformers import AutoProcessor, AutoModelForCausalLM, Qwen2VLForConditionalGeneration

FLORENCE_MODEL_DIR = "./trained_models/florence2_final"
QWEN_MODEL_DIR     = "./trained_models/qwen2_5vl_final"
RESULTS_JSON       = "./trained_models/evaluation_results.json"
MAX_IMAGE_SIZE     = 512

# ── helpers ──────────────────────────────────────────────────────────────────

def resize_image(img, max_size=MAX_IMAGE_SIZE):
    img = img.convert("RGB")
    w, h = img.size
    if w > max_size or h > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
    return img

def extract_volume(text):
    if not text:
        return 0.0
    for pat in [r"(\d+\.?\d*)\s*mL", r"(\d+\.?\d*)\s*ml", r"(\d+\.?\d*)\s*milliliter"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    nums = re.findall(r"\d+\.?\d*", text)
    return float(nums[0]) if nums else 0.0

# ── lazy model cache ──────────────────────────────────────────────────────────

_fm = _fp = _qm = _qp = None

def load_florence():
    global _fm, _fp
    if _fm: return _fm, _fp
    if not os.path.isdir(FLORENCE_MODEL_DIR):
        raise FileNotFoundError(f"Not found: {FLORENCE_MODEL_DIR}")
    print("Loading Florence-2...")
    _fp = AutoProcessor.from_pretrained(FLORENCE_MODEL_DIR, trust_remote_code=True)
    _fm = AutoModelForCausalLM.from_pretrained(
        FLORENCE_MODEL_DIR, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    ).eval()
    return _fm, _fp

def load_qwen():
    global _qm, _qp
    if _qm: return _qm, _qp
    if not os.path.isdir(QWEN_MODEL_DIR):
        raise FileNotFoundError(f"Not found: {QWEN_MODEL_DIR}")
    print("Loading Qwen2.5-VL...")
    _qp = AutoProcessor.from_pretrained(QWEN_MODEL_DIR, trust_remote_code=True)
    _qm = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_DIR, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    ).eval()
    return _qm, _qp

# ── inference ─────────────────────────────────────────────────────────────────

QWEN_SYSTEM = (
    "You are a lab measurement tool. "
    "Respond with ONLY the volume as a number and unit, e.g. \"150 mL\". "
    "Never explain or use formulas."
)

def predict_florence(img):
    model, proc = load_florence()
    img = resize_image(img)
    dtype = next(model.parameters()).dtype
    inputs = proc(images=img, text="<VQA>What is the volume of liquid in the beaker?",
                  return_tensors="pt")
    # FIX: cast pixel_values to fp16 to match model weights
    inputs = {
        k: v.to(model.device).to(dtype) if v.dtype.is_floating_point else v.to(model.device)
        for k, v in inputs.items()
    }
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=30)
    text = proc.batch_decode(ids, skip_special_tokens=True)[0]
    return text, extract_volume(text)

def predict_qwen(img):
    model, proc = load_qwen()
    img = resize_image(img)
    messages = [
        {"role": "system", "content": QWEN_SYSTEM},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text": "What is the volume of liquid in this beaker in mL?"},
        ]},
    ]
    text_in = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs  = proc(text=[text_in], images=[img], return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    text = proc.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text, extract_volume(text)

def run_inference(pil_image):
    if pil_image is None:
        return "—", "—", "—", "—"
    try:
        f_raw, f_vol = predict_florence(pil_image)
        f_disp = f"{f_vol:.1f} mL"
    except Exception as e:
        f_raw, f_disp = f"❌ Error: {e}", "—"
    try:
        q_raw, q_vol = predict_qwen(pil_image)
        q_disp = f"{q_vol:.1f} mL"
    except Exception as e:
        q_raw, q_disp = f"❌ Error: {e}", "—"
    return f_raw, f_disp, q_raw, q_disp

# ── metrics helpers ───────────────────────────────────────────────────────────

def load_results():
    if not os.path.isfile(RESULTS_JSON):
        return None
    with open(RESULTS_JSON) as f:
        return json.load(f)

def metric_cards_html(m):
    mae  = m.get("mae",  float("nan"))
    rmse = m.get("rmse", float("nan"))
    r2   = m.get("r2",   float("nan"))
    card = lambda title, val, unit: (
        f"<div style='background:#1e293b;padding:18px 28px;border-radius:12px;"
        f"text-align:center;min-width:120px;box-shadow:0 2px 8px #0004'>"
        f"<div style='color:#94a3b8;font-size:13px;margin-bottom:4px'>{title}</div>"
        f"<div style='color:#f8fafc;font-size:30px;font-weight:700'>{val}</div>"
        f"<div style='color:#64748b;font-size:11px;margin-top:2px'>{unit}</div>"
        f"</div>"
    )
    return (
        "<div style='display:flex;gap:20px;flex-wrap:wrap;padding:12px 0'>"
        + card("MAE",  f"{mae:.2f}",  "mL — lower is better")
        + card("RMSE", f"{rmse:.2f}", "mL — lower is better")
        + card("R²",   f"{r2:.4f}",  "higher is better")
        + "</div>"
    )

def make_plots(metrics, model_name):
    preds = np.array(metrics.get("predictions", []))
    gt    = np.array(metrics.get("ground_truth", []))
    if len(preds) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data yet — run training first",
                ha="center", va="center", fontsize=13, color="#64748b")
        return fig, fig

    # scatter
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.scatter(gt, preds, alpha=0.45, s=18, color="#38bdf8", label="Samples")
    lo, hi = min(gt.min(), preds.min()), max(gt.max(), preds.max())
    ax1.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect")
    ax1.set_xlabel("Ground Truth (mL)", fontsize=11)
    ax1.set_ylabel("Prediction (mL)", fontsize=11)
    ax1.set_title(f"{model_name} — Predicted vs Ground Truth", fontsize=12, fontweight="bold")
    ax1.legend(); ax1.grid(True, alpha=0.3); plt.tight_layout()

    # error histogram
    err = preds - gt
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.hist(err, bins=30, edgecolor="black", alpha=0.75, color="#818cf8")
    ax2.axvline(0, color="red", lw=2, linestyle="--", label="Zero error")
    ax2.axvline(err.mean(), color="lime", lw=2, linestyle="--",
                label=f"Mean: {err.mean():.1f} mL")
    ax2.set_xlabel("Prediction Error (mL)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title(f"{model_name} — Error Distribution", fontsize=12, fontweight="bold")
    ax2.legend(); ax2.grid(True, alpha=0.3); plt.tight_layout()
    return fig1, fig2

def refresh_metrics(model_key, model_name):
    res = load_results()
    if res is None:
        return "<p style='color:#f87171'>evaluation_results.json not found — run training first.</p>", None, None
    m = res.get(model_key, {})
    return metric_cards_html(m), *make_plots(m, model_name)

# ── UI ────────────────────────────────────────────────────────────────────────

css = """
body        { background:#0f172a; color:#e2e8f0; font-family:'Inter',sans-serif; }
.gr-block   { background:#1e293b !important; border-color:#334155 !important; }
.gr-button  { background:#3b82f6 !important; color:white !important; border:none !important; }
.gr-button:hover { background:#2563eb !important; }
"""

HEADER = """
<div style='text-align:center;padding:28px 0 12px'>
  <h1 style='color:#f1f5f9;font-size:2.1rem;font-weight:800;margin:0'>
    🧪 Beaker Volume Estimator
  </h1>
  <p style='color:#64748b;margin:8px 0 0;font-size:15px'>
    Florence-2 &amp; Qwen2.5-VL fine-tuned on lab beaker images
  </p>
</div>
"""

with gr.Blocks(css=css, title="Beaker Volume Estimator") as demo:
    gr.HTML(HEADER)

    with gr.Tabs():

        # ── Tab 1: Live inference ──────────────────────────────────────────
        with gr.Tab("🔬 Live Inference"):
            gr.Markdown("Upload a beaker photo — both models will predict the liquid volume.")
            with gr.Row():
                with gr.Column(scale=1):
                    img_in  = gr.Image(type="pil", label="Beaker Image")
                    pred_btn = gr.Button("🚀 Predict", variant="primary")
                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown("#### Florence-2")
                        f_raw = gr.Textbox(label="Raw model output", lines=2, interactive=False)
                        f_vol = gr.Textbox(label="Estimated volume",        interactive=False)
                    with gr.Group():
                        gr.Markdown("#### Qwen2.5-VL")
                        q_raw = gr.Textbox(label="Raw model output", lines=2, interactive=False)
                        q_vol = gr.Textbox(label="Estimated volume",        interactive=False)
            pred_btn.click(run_inference, inputs=[img_in],
                           outputs=[f_raw, f_vol, q_raw, q_vol])

        # ── Tab 2: Florence-2 metrics ──────────────────────────────────────
        with gr.Tab("📊 Florence-2 Metrics"):
            gr.Markdown("## Florence-2 — Test-set Evaluation (n=300)")
            f2_btn  = gr.Button("🔄 Load / Refresh", variant="secondary")
            f2_html = gr.HTML("<p>Click 'Load / Refresh' to show results.</p>")
            with gr.Row():
                f2_scatter = gr.Plot(label="Predicted vs Ground Truth")
                f2_error   = gr.Plot(label="Error Distribution")
            f2_btn.click(
                fn=lambda: refresh_metrics("florence2", "Florence-2"),
                outputs=[f2_html, f2_scatter, f2_error]
            )

        # ── Tab 3: Qwen metrics ────────────────────────────────────────────
        with gr.Tab("📊 Qwen2.5-VL Metrics"):
            gr.Markdown("## Qwen2.5-VL — Test-set Evaluation (n=300)")
            q2_btn  = gr.Button("🔄 Load / Refresh", variant="secondary")
            q2_html = gr.HTML("<p>Click 'Load / Refresh' to show results.</p>")
            with gr.Row():
                q2_scatter = gr.Plot(label="Predicted vs Ground Truth")
                q2_error   = gr.Plot(label="Error Distribution")
            q2_btn.click(
                fn=lambda: refresh_metrics("qwen2_5vl", "Qwen2.5-VL"),
                outputs=[q2_html, q2_scatter, q2_error]
            )

        # ── Tab 4: About ───────────────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
## About

Two vision-language models fine-tuned to estimate liquid volume from beaker photos.

### Models
| Model | Base | Trainable params |
|-------|------|-----------------|
| Florence-2 | microsoft/Florence-2-base (232M) | ~0.29% via LoRA |
| Qwen2.5-VL | Qwen/Qwen2-VL-2B-Instruct (2B)  | LoRA r=8 |

### Training details
- **Dataset**: yusufbukarmaina/Beakers1
- **Split**: 1000 train / 300 val / 300 test
- **LoRA**: r=8, α=16 on q_proj + v_proj
- **Hardware**: RTX A5000 25 GB

### Key fixes applied
- **Florence-2**: `pixel_values` cast to `fp16` before inference to match model weights
- **Qwen2.5-VL**: System prompt forces `"X mL"` output — no formula explanations
- **Test images**: Saved to `/FQ/test_images/` *before* training begins

### Metrics
- **MAE** — mean absolute error in mL (lower = better)
- **RMSE** — root mean squared error (penalises outliers; lower = better)
- **R²** — coefficient of determination (1.0 = perfect fit)
""")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    a = p.parse_args()
    demo.launch(share=a.share, server_port=a.port, server_name="0.0.0.0")

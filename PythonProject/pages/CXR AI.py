# Wmed | CXR AI — LLM Teaching Explanation (Fixed .env + secrets)
# ------------------------------------------------------------------
# - Loads .env properly from this file's directory
# - Falls back to Streamlit secrets or inline key
# - Uses consistent OPENAI_* variables
# ------------------------------------------------------------------

import os
import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt  # (ok to keep even if unused)
import skimage.io  # (ok to keep even if unused)
import torchxrayvision as xrv
import dotenv
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms import Resize  # (ok to keep even if unused)
import torch.nn.functional as F
import requests
from pathlib import Path
from dotenv import load_dotenv

# ================== Secrets / ENV Loading ==================
ENV_PATH =  "/Users/waleedal-tahafi/PycharmProjects/PythonProject/pages/.env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

OPENAI_API_KEY_INLINE = ""  # leave empty in prod

OPENAI_API_KEY = (  os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY_INLINE
)

OPENAI_BASE_URL = ("https://api.openai.com/v1"
)
OPENAI_MODEL = ("gpt-4o-mini"
)

# Early check
st.set_page_config(page_title="Wmed | CXR AI", layout="wide")
if not OPENAI_API_KEY:
    st.error("No API key found. Add OPENAI_API_KEY to .env (same folder), or to Streamlit Secrets, or paste inline.")
    st.stop()

# ================== UI ==================
st.title("🧠 Wmed AI - Chest X-Ray Analysis")
st.markdown("Level up your diagnostics with **AI-powered precision**. Upload a chest X-ray to begin.")

st.sidebar.title("📤 Upload Chest X-Ray")
uploaded_file = st.sidebar.file_uploader("Accepted formats: PNG, JPG, JPEG", type=["png", "jpg", "jpeg"])

# ================== Diagnosis Heuristics ==================
def map_findings_to_diagnosis(results, threshold=0.59):
    if results.get("Tuberculosis", 0) > 0.6:
        return "Tuberculosis"
    elif results.get("Mass", 0) > 0.51 or results.get("Nodule", 0) > 0.65:
        return "Lung tumor"
    elif results.get("Emphysema", 0) > 0.51 or results.get("Fibrosis", 0) > 0.6:
        return "COPD"
    elif results.get("Cardiomegaly", 0) > 0.52 and results.get("Edema", 0) > 0.55:
        return "Heart failure"
    elif results.get("Consolidation", 0) > 0.58 or results.get("Edema", 0) > 0.63:
        return "Pneumonia"
    elif results.get("Pleural Effusion", 0) > 0.61 or results.get("Effusion", 0) > 0.55:
        return "Pleural effusion"
    elif results.get("Pneumothorax", 0) > 0.54:
        return "Pneumothorax"
    elif results.get("Fracture", 0) > 0.65 or results.get("Rib Fracture", 0) > 0.65:
        return "Fracture"
    elif results.get("Interstitial lung disease", 0) > 0.6 or results.get("Fibrosis", 0) > 0.6:
        return "Interstitial lung disease"
    elif results.get("Edema", 0) > 0.58 and results.get("Consolidation", 0) > 0.58:
        return "ARDS"
    elif all(score < 0.51 for score in results.values()):
        return "No finding"
    else:
        return "Other diseases"

explanations = {
    "Effusion": "Fluid accumulation in the pleural space, often linked to infection or heart failure.",
    "Lung Opacity": "An area of increased density, may indicate pneumonia, tumour, or fibrosis.",
    "Enlarged Cardiomediastinum": "Could suggest cardiomegaly or mediastinal masses.",
    "Mass": "Localized lesion that may suggest malignancy.",
    "Nodule": "Smaller lesion that may be benign or malignant depending on characteristics.",
    "Consolidation": "Lung filled with fluid or pus, typically seen in pneumonia.",
    "Edema": "Fluid accumulation usually due to cardiac causes like heart failure.",
    "Cardiomegaly": "Enlargement of the heart, commonly associated with chronic hypertension or heart failure.",
    "Pneumothorax": "Air trapped in pleural space causing lung collapse.",
    "Fibrosis": "Scarring of lung tissue, often from chronic inflammation or exposure.",
}

# ================== Image Processing ==================
def process_image(uploaded_file):
    image = Image.open(uploaded_file).convert("L")
    orig_size = image.size
    img = np.array(image).astype(np.float32)
    img = (img - np.mean(img)) / (np.std(img) + 1e-8)
    img = np.clip(img, -3, 3) * 300
    img = img[None, ...]
    img = xrv.datasets.XRayCenterCrop()(img)
    img = xrv.datasets.XRayResizer(224)(img)
    img_tensor = torch.from_numpy(img)[None, ...]
    return img_tensor, img, orig_size, np.array(image)

# ================== LLM Teaching Explanation ==================
def llm_teaching_explanation(diagnosis: str, top_findings: dict, notes: str = "") -> str:
    url = OPENAI_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    sys = (
        "You are a radiology tutor for medical students. Educational use only. "
        "Do not claim diagnostic certainty. Use plain language. Keep it under 120 words. "
        "Structure your answer with: Findings → Why they suggest the possible diagnosis → What else to consider / next steps."
    )
    user = f"Possible diagnosis: {diagnosis}. Top findings: {top_findings}. Notes: {notes or 'none'}. Write a concise explanation."

    payload = {"model": OPENAI_MODEL, "temperature": 0.4, "messages": [
        {"role": "system", "content": sys},
        {"role": "user", "content": user}
    ]}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(LLM explanation unavailable: {e})"

# ================== Main Inference ==================
if uploaded_file:
    img_tensor, processed_img, orig_size, raw_img = process_image(uploaded_file)

    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()

    with torch.no_grad():
        output = model(img_tensor)
        probs = output[0].detach().numpy()
        results = dict(zip(model.pathologies, probs))
        filtered = {k: v for k, v in results.items() if v > 0.05}
        top_3 = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:3]
        pred_idx = output[0].argmax().item()
        diagnosis = map_findings_to_diagnosis(results)

    target_layers = [model.features.denseblock4]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]

    grayscale_cam_tensor = torch.from_numpy(grayscale_cam).unsqueeze(0).unsqueeze(0)
    grayscale_cam_resized = F.interpolate(
        grayscale_cam_tensor,
        size=(raw_img.shape[0], raw_img.shape[1]),
        mode="bilinear",
        align_corners=False,
    )
    grayscale_cam_resized = grayscale_cam_resized.squeeze().numpy()

    rgb_img = np.repeat(raw_img[None, :, :], 3, axis=0).transpose(1, 2, 0)
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
    heatmap_img = show_cam_on_image(rgb_img.astype(np.float32), grayscale_cam_resized, use_rgb=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original X-Ray")
        st.image(raw_img, clamp=True, caption="Uploaded Chest X-Ray", use_container_width=True)
    with col2:
        st.subheader("🔥 AI Focus Heatmap")
        st.image(heatmap_img, caption="Where the AI looked", use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Prediction Results")
    for cond, score in top_3:
        st.write(f"- **{cond}**: {score:.2f}")
        st.caption(explanations.get(cond, "No explanation available."))

    st.success(f"🏁 **AI Diagnosis**: {diagnosis}")

    st.markdown("### 🎓 Teaching Explanation (LLM)")
    tf_dict = {k: float(v) for k, v in top_3}
    notes = "Heatmap highlights regions driving the model's top class; treat as pedagogical only."
    expl = llm_teaching_explanation(diagnosis, tf_dict, notes)
    st.write(expl)
    st.caption("Education only · Not for diagnosis or patient care")

else:
    st.info("📂 Upload a chest X-ray to begin diagnosis.")

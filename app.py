import os
import re
import datetime
import json
from io import BytesIO
import streamlit as st
from PIL import Image

import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account

# ---------------- CONFIG ----------------
PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]

# Create credentials from secrets
credentials = service_account.Credentials.from_service_account_info(
    dict(st.secrets["gcp_service_account"])
)

# Init Vertex AI for both regions
vertexai.init(project=PROJECT_ID, location="us-central1", credentials=credentials)

# Imagen 4 for generation
IMAGEN_MODEL = ImageGenerationModel.from_pretrained("imagen-4.0-generate-001")

# Nano Banana for editing
NANO_BANANA = GenerativeModel("gemini-2.5-flash-image")

# Gemini Flash for text prompt refinement
TEXT_MODEL = GenerativeModel("gemini-2.0-flash")

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="AI Image Generator + Editor", layout="wide")
st.title("🖼️ Imagen + Nano Banana: Generator + Editor")

# ---------------- STATE ----------------
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []
if "edited_images" not in st.session_state:
    st.session_state.edited_images = []

# ---------------- Helpers ----------------
def safe_get_enhanced_text(resp):
    if hasattr(resp, "text") and resp.text:
        return resp.text
    if hasattr(resp, "candidates") and resp.candidates:
        try:
            return resp.candidates[0].content.parts[0].text
        except Exception:
            pass
    return str(resp)

def get_image_bytes_from_genobj(gen_obj):
    if isinstance(gen_obj, (bytes, bytearray)):
        return bytes(gen_obj)
    for attr in ["image_bytes", "_image_bytes"]:
        if hasattr(gen_obj, attr):
            return getattr(gen_obj, attr)
    if hasattr(gen_obj, "image") and gen_obj.image:
        for attr in ["image_bytes", "_image_bytes"]:
            if hasattr(gen_obj.image, attr):
                return getattr(gen_obj.image, attr)
    return None

def run_edit_flow(edit_prompt, base_bytes):
    """Edit image using Nano Banana"""
    input_image = Part.from_data(mime_type="image/png", data=base_bytes)
    edit_instruction = f"Edit the image as follows: {edit_prompt}. Always return only the edited image as inline PNG."
    resp = NANO_BANANA.generate_content([edit_instruction, input_image])
    for part in resp.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data.data:
            return part.inline_data.data
    return None

# ---------------- PROMPTS ----------------
PROMPT_TEMPLATES = {
    "Marketing": """
You are a senior AI prompt engineer creating polished prompts for marketing and advertising visuals.
...
Refined marketing image prompt:
""",
    # (you can keep the rest of your templates here unchanged)
}

STYLE_DESCRIPTIONS = {
    "None": "No special styling — keep it natural.",
    "Cinematic": "Film-style composition with dramatic lighting.",
    "Vibrant": "High contrast, saturated colors, lively tone.",
    "Minimalist": "Simple, clean, elegant composition.",
}

# ---------------- TABS ----------------
tab_generate, tab_edit = st.tabs(["✨ Generate with Imagen", "🖌️ Edit with Nano Banana"])

# ---------------- GENERATE ----------------
with tab_generate:
    st.header("✨ Generate Images (Imagen 4)")

    dept = st.selectbox("🏢 Department", options=list(PROMPT_TEMPLATES.keys()), index=0, key="dept_gen")
    style = st.selectbox("🎨 Style", options=list(STYLE_DESCRIPTIONS.keys()), index=0, key="style_gen")
    user_prompt = st.text_area("Enter your prompt", height=120, key="prompt_gen")
    num_images = st.slider("🧾 Number of images", 1, 4, 1, key="num_gen")

    if st.button("🚀 Generate with Imagen"):
        if not user_prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Refining prompt with Gemini..."):
                refinement_prompt = PROMPT_TEMPLATES[dept].replace("{USER_PROMPT}", user_prompt)
                if style != "None":
                    refinement_prompt += f"\n\nApply style: {STYLE_DESCRIPTIONS[style]}"
                text_resp = TEXT_MODEL.generate_content(refinement_prompt)
                enhanced_prompt = safe_get_enhanced_text(text_resp).strip()
                st.info(f"🔮 Enhanced Prompt:\n\n{enhanced_prompt}")

            with st.spinner("Generating images with Imagen 4..."):
                try:
                    resp = IMAGEN_MODEL.generate_images(prompt=enhanced_prompt, number_of_images=num_images)
                except Exception as e:
                    st.error(f"⚠️ Imagen error: {e}")
                    st.stop()

                for i in range(num_images):
                    try:
                        gen_obj = resp.images[i]
                        img_bytes = get_image_bytes_from_genobj(gen_obj)
                        filename = f"{dept.lower()}_{style.lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.png"
                        st.session_state.generated_images.append({"filename": filename, "content": img_bytes})
                        st.image(Image.open(BytesIO(img_bytes)), caption=filename, use_column_width=True)
                        st.download_button("⬇️ Download", data=img_bytes, file_name=filename, mime="image/png", key=f"dl_{i}")
                    except Exception as e:
                        st.error(f"⚠️ Failed to display image {i}: {e}")

# ---------------- EDIT ----------------
with tab_edit:
    st.header("🖌️ Edit Images (Nano Banana)")

    uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg", "webp"])
    base_image = None
    if uploaded_file:
        image_bytes = uploaded_file.read()
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG")
        base_image = buf.getvalue()
        st.image(img, caption="Uploaded Image", use_column_width=True)

    edit_prompt = st.text_area("Enter your edit instruction", height=120, key="edit_prompt")
    num_edits = st.slider("🧾 Number of edited images", 1, 3, 1, key="num_edits")

    if st.button("🚀 Edit with Nano Banana"):
        if not base_image or not edit_prompt.strip():
            st.warning("Please upload an image and enter instructions.")
        else:
            with st.spinner("Editing with Nano Banana..."):
                edited_versions = []
                for _ in range(num_edits):
                    edited = run_edit_flow(edit_prompt, base_image)
                    if edited:
                        edited_versions.append(edited)

                if edited_versions:
                    for i, out_bytes in enumerate(edited_versions):
                        st.image(Image.open(BytesIO(out_bytes)), caption=f"Edited Version {i+1}", use_column_width=True)
                        st.download_button(
                            f"⬇️ Download Edited {i+1}",
                            data=out_bytes,
                            file_name=f"edited_{i}.png",
                            mime="image/png",
                            key=f"edit_dl_{i}"
                        )
                        st.session_state.edited_images.append({
                            "original": base_image,
                            "edited": out_bytes,
                            "prompt": edit_prompt
                        })
                else:
                    st.error("❌ No edited image returned by Nano Banana.")

# ---------------- HISTORY ----------------
st.subheader("📂 History")
if st.session_state.generated_images:
    st.markdown("### Generated Images")
    for i, img in enumerate(reversed(st.session_state.generated_images[-10:])):
        with st.expander(f"{i+1}. {img['filename']}"):
            st.image(Image.open(BytesIO(img["content"])), caption=img["filename"], use_container_width=True)
            st.download_button("⬇️ Download Again", data=img["content"], file_name=img["filename"], mime="image/png")

if st.session_state.edited_images:
    st.markdown("### Edited Images")
    for i, entry in enumerate(reversed(st.session_state.edited_images[-10:])):
        with st.expander(f"Edited {i+1}: {entry['prompt']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.image(Image.open(BytesIO(entry["original"])), caption="Original", use_column_width=True)
            with col2:
                st.image(Image.open(BytesIO(entry["edited"])), caption="Edited", use_column_width=True)

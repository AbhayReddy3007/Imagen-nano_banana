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

Task:
- Take the user’s raw input and turn it into a polished, professional, campaign-ready image prompt.
- Expand the idea with rich marketing-oriented details that make it visually persuasive.

When refining, include:
- Background & setting (modern, lifestyle, commercial, aspirational)
- Lighting & atmosphere (studio lights, golden hour, cinematic)
- Style (photorealistic, cinematic, product photography, lifestyle branding)
- Perspective & composition (wide shot, close-up, dramatic angles)
- Mood, tone & branding suitability (premium, sleek, aspirational)

Special Brand Rule:
- If the user asks for an image related to a specific brand, seamlessly add the brand’s tagline into the final image prompt.
- For **Dr. Reddy’s**, the correct tagline is: “Good Health Can’t Wait.”

Rules:
- Stay faithful to the user’s idea but elevate it for use in ads, social media, or presentations.
- Output **only** the final refined image prompt (no explanations, no extra text).

User raw input:
{USER_PROMPT}


Refined marketing image prompt:
""",

    "Design": """
You are a senior AI prompt engineer supporting a creative design team.

Your job:
- Expand raw input into a visually inspiring, design-oriented image prompt.
- Add imaginative details about:
  • Artistic styles (minimalist, abstract, futuristic, flat, 3D render, watercolor, digital illustration)
  • Color schemes, palettes, textures, and patterns
  • Composition and balance (symmetry, negative space, creative framing)
  • Lighting and atmosphere (soft glow, vibrant contrast, surreal shading)
  • Perspective (isometric, top-down, wide shot, close-up)

Rules:
- Keep fidelity to the idea but make it highly creative and visually unique.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined design image prompt:
""",

    "General": """
You are an expert AI prompt engineer specialized in creating vivid and descriptive image prompts.

Your job:
- Expand the user’s input into a detailed, clear prompt for an image generation model.
- Add missing details such as:
  • Background and setting
  • Lighting and mood
  • Style and realism level
  • Perspective and composition

Rules:
- Stay true to the user’s intent.
- Keep language concise, descriptive, and expressive.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined general image prompt:
""",

    "DPEX": """
You are a senior AI prompt engineer creating refined prompts for IT and technology-related visuals.

Your job:
- Transform the raw input into a detailed, professional, and technology-focused image prompt.
- Expand with contextual details about:
  • Technology environments (server rooms, data centers, cloud systems, coding workspaces)
  • Digital elements (network diagrams, futuristic UIs, holograms, cybersecurity visuals)
  • People in IT roles (developers, engineers, admins, tech support, collaboration)
  • Tone (innovative, technical, futuristic, professional)
  • Composition (screens, servers, code on monitors, abstract digital patterns)
  • Lighting and effects (LED glow, cyberpunk tones, neon highlights, modern tech ambiance)

Rules:
- Ensure images are suitable for IT presentations, product demos, training, technical documentation, and digital transformation campaigns.
- Stay true to the user’s intent but emphasize a technological and innovative look.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined DPEX image prompt:
""",

    "HR": """
You are a senior AI prompt engineer creating refined prompts for human resources and workplace-related visuals.

Your job:
- Transform the raw input into a detailed, professional, and HR-focused image prompt.
- Expand with contextual details about:
  • Workplace settings (modern office, meeting rooms, open workspaces, onboarding sessions)
  • People interactions (interviews, teamwork, training, collaboration, diversity and inclusion)
  • Themes (employee engagement, professional growth, recruitment, performance evaluation)
  • Composition (groups in discussion, managers mentoring, collaborative workshops)
  • Lighting and tone (bright, welcoming, professional, inclusive)

Rules:
- Ensure images are suitable for HR presentations, recruitment campaigns, internal training, or employee engagement material.
- Stay true to the user’s intent but emphasize people, culture, and workplace positivity.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined HR image prompt:
""",

    "Business": """
You are a senior AI prompt engineer creating refined prompts for business and corporate visuals.

Your job:
- Transform the raw input into a detailed, professional, and business-oriented image prompt.
- Expand with contextual details about:
  • Corporate settings (boardrooms, skyscrapers, modern offices, networking events)
  • Business activities (presentations, negotiations, brainstorming sessions, teamwork)
  • People (executives, entrepreneurs, consultants, diverse teams, global collaboration)
  • Tone (professional, ambitious, strategic, innovative)
  • Composition (formal meetings, handshake deals, conference tables, city skyline backgrounds)
  • Lighting and atmosphere (clean, modern, premium, professional)

Rules:
- Ensure images are suitable for corporate branding, investor decks, strategy sessions, or professional reports.
- Stay true to the user’s intent but emphasize professionalism, ambition, and success.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined business image prompt:
"""
}


STYLE_DESCRIPTIONS = {
    "None": "No special styling — keep the image natural, faithful to the user’s idea.",
    "Smart": "A clean, balanced, and polished look. Professional yet neutral, visually appealing without strong artistic bias.",
    "Cinematic": "Film-style composition with professional lighting. Wide dynamic range, dramatic highlights, storytelling feel.",
    "Creative": "Playful, imaginative, and experimental. Bold artistic choices, unexpected elements, and expressive color use.",
    "Bokeh": "Photography style with shallow depth of field. Subject in sharp focus with soft, dreamy, blurred backgrounds.",
    "Macro": "Extreme close-up photography. High detail, textures visible, shallow focus highlighting minute features.",
    "Illustration": "Hand-drawn or digitally illustrated style. Clear outlines, stylized shading, expressive and artistic.",
    "3D Render": "Photorealistic or stylized CGI. Crisp geometry, depth, shadows, and reflective surfaces with realistic rendering.",
    "Fashion": "High-end editorial photography. Stylish, glamorous poses, bold makeup, controlled lighting, and modern aesthetic.",
    "Minimalist": "Simple and uncluttered. Few elements, large negative space, flat or muted color palette, clean composition.",
    "Moody": "Dark, atmospheric, and emotional. Strong shadows, high contrast, deep tones, cinematic ambiance.",
    "Portrait": "Focus on the subject. Natural skin tones, shallow depth of field, close-up or waist-up framing, studio or natural lighting.",
    "Stock Photo": "Professional, commercial-quality photo. Neutral subject matter, polished composition, business-friendly aesthetic.",
    "Vibrant": "Bold, saturated colors. High contrast, energetic mood, eye-catching and lively presentation.",
    "Pop Art": "Comic-book and pop-art inspired. Bold outlines, halftone patterns, flat vivid colors, high contrast, playful tone.",
    "Vector": "Flat vector graphics. Smooth shapes, sharp edges, solid fills, and clean scalable style like logos or icons.",

    "Watercolor": "Soft, fluid strokes with delicate blending and washed-out textures. Artistic and dreamy.",
    "Oil Painting": "Rich, textured brushstrokes. Classic fine art look with deep color blending.",
    "Charcoal": "Rough, sketchy textures with dark shading. Artistic, raw, dramatic effect.",
    "Line Art": "Minimal monochrome outlines with clean, bold strokes. No shading, focus on form.",

    "Anime": "Japanese animation style with vibrant colors, clean outlines, expressive features, and stylized proportions.",
    "Cartoon": "Playful, exaggerated features, simplified shapes, bold outlines, and bright colors.",
    "Pixel Art": "Retro digital art style. Small, pixel-based visuals resembling old-school video games.",

    "Fantasy Art": "Epic fantasy scenes. Magical elements, mythical creatures, enchanted landscapes.",
    "Surreal": "Dreamlike, bizarre imagery. Juxtaposes unexpected elements, bending reality.",
    "Concept Art": "Imaginative, detailed artwork for games or films. Often moody and cinematic.",

    "Cyberpunk": "Futuristic neon city vibes. High contrast, glowing lights, dark tones, sci-fi feel.",
    "Steampunk": "Retro-futuristic style with gears, brass, Victorian aesthetics, and industrial design.",
    "Neon Glow": "Bright neon outlines and glowing highlights. Futuristic, nightlife aesthetic.",
    "Low Poly": "Simplified 3D style using flat geometric shapes and polygons.",
    "Isometric": "3D look with isometric perspective. Often used for architecture, games, and diagrams.",

    "Vintage": "Old-school, retro tones. Faded colors, film grain, sepia, or retro print feel.",
    "Graffiti": "Urban street art style with bold colors, spray paint textures, and rebellious tone."
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
# ---------------- HISTORY ----------------
st.subheader("📂 History")

# ===== Generated Images =====
if st.session_state.generated_images:
    st.markdown("### Generated Images")
    for i, img in enumerate(reversed(st.session_state.generated_images[-10:])):
        with st.expander(f"{i+1}. {img.get('filename', 'Unnamed Image')}"):
            content = img.get("content")

            if isinstance(content, (bytes, bytearray)) and len(content) > 0:
                try:
                    st.image(Image.open(BytesIO(content)), caption=img.get("filename", "Generated Image"), use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Unable to display image: {e}")
            else:
                st.warning(f"⚠️ Skipping invalid or empty image: {img.get('filename', 'unknown')}")

            if isinstance(content, (bytes, bytearray)):
                st.download_button(
                    "⬇️ Download Again",
                    data=content,
                    file_name=img.get("filename", "generated_image.png"),
                    mime="image/png",
                    key=f"gen_dl_{i}"
                )

# ===== Edited Images =====
if st.session_state.edited_images:
    st.markdown("### Edited Images")
    for i, entry in enumerate(reversed(st.session_state.edited_images[-10:])):
        with st.expander(f"Edited {i+1}: {entry.get('prompt', '')}"):
            col1, col2 = st.columns(2)

            # --- Original Image ---
            with col1:
                orig_bytes = entry.get("original")
                if isinstance(orig_bytes, (bytes, bytearray)) and len(orig_bytes) > 0:
                    try:
                        st.image(Image.open(BytesIO(orig_bytes)), caption="Original", use_column_width=True)
                    except Exception as e:
                        st.warning(f"⚠️ Failed to load original: {e}")
                else:
                    st.warning("⚠️ Original image missing or invalid.")

            # --- Edited Image ---
            with col2:
                edited_bytes = entry.get("edited")
                if isinstance(edited_bytes, (bytes, bytearray)) and len(edited_bytes) > 0:
                    try:
                        st.image(Image.open(BytesIO(edited_bytes)), caption="Edited", use_column_width=True)
                    except Exception as e:
                        st.warning(f"⚠️ Failed to load edited image: {e}")
                else:
                    st.warning("⚠️ Edited image missing or invalid.")

                if isinstance(edited_bytes, (bytes, bytearray)):
                    st.download_button(
                        "⬇️ Download Edited",
                        data=edited_bytes,
                        file_name=f"edited_{i}.png",
                        mime="image/png",
                        key=f"edit_dl_{i}"
                    )


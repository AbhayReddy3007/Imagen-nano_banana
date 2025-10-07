import os
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
credentials = service_account.Credentials.from_service_account_info(
    dict(st.secrets["gcp_service_account"])
)

vertexai.init(project=PROJECT_ID, location="us-central1", credentials=credentials)

# Models
IMAGEN_MODEL = ImageGenerationModel.from_pretrained("imagen-4.0-generate-001")
NANO_BANANA = GenerativeModel("gemini-2.5-flash-image")
TEXT_MODEL = GenerativeModel("gemini-2.0-flash")


# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="Imagen + Nano Banana", layout="wide")
st.title("🖼️ Imagen + Nano Banana | AI Image Generator & Editor")

# ---------------- STATE ----------------
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []
if "editing_image" not in st.session_state:
    st.session_state.editing_image = None


# ---------------- HELPERS ----------------
def safe_get_enhanced_text(resp):
    """Extract refined prompt text safely."""
    if hasattr(resp, "text") and resp.text:
        return resp.text
    if hasattr(resp, "candidates") and resp.candidates:
        try:
            return resp.candidates[0].content.parts[0].text
        except Exception:
            pass
    return str(resp)


def get_image_bytes_from_genobj(gen_obj):
    """Extract bytes from Imagen response object."""
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
    """Edit image using Nano Banana (Gemini 2.5 Flash Image)."""
    input_image = Part.from_data(mime_type="image/png", data=base_bytes)
    edit_instruction = (
        f"You are an expert AI image editor. "
        f"Apply the following edit: {edit_prompt}. "
        f"Return only the edited image as inline PNG — no text or explanation."
    )
    resp = NANO_BANANA.generate_content([edit_instruction, input_image])
    for candidate in getattr(resp, "candidates", []):
        for part in getattr(candidate.content, "parts", []):
            if hasattr(part, "inline_data") and part.inline_data.data:
                return part.inline_data.data
    return None


# ---------------- PROMPTS & STYLES ----------------
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


# ---------------- GENERATE SECTION ----------------
st.header("✨ Generate Images (Imagen 4)")

dept = st.selectbox("🏢 Department", list(PROMPT_TEMPLATES.keys()), index=0)
style = st.selectbox("🎨 Style", list(STYLE_DESCRIPTIONS.keys()), index=0)
user_prompt = st.text_area("Enter your prompt", height=120)
num_images = st.slider("🧾 Number of images", 1, 4, 1)

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
                gen_obj = None
                try:
                    gen_obj = resp.images[i]
                except Exception:
                    continue
                img_bytes = get_image_bytes_from_genobj(gen_obj)
                if not img_bytes:
                    continue

                filename = f"{dept.lower()}_{style.lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.png"
                st.session_state.generated_images.append({"filename": filename, "content": img_bytes})

                st.image(Image.open(BytesIO(img_bytes)), caption=filename, use_column_width=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.download_button("⬇️ Download", data=img_bytes, file_name=filename, mime="image/png", key=f"dl_{i}")
                with col_b:
                    if st.button("🪄 Edit with Nano Banana", key=f"edit_btn_{i}"):
                        st.session_state.editing_image = {"filename": filename, "content": img_bytes}
                        st.toast("✅ Image sent to Nano Banana Editor below!")


# ---------------- INLINE EDIT SECTION ----------------
if st.session_state.editing_image:
    st.divider()
    st.subheader("🖌️ Edit with Nano Banana")

    img_data = st.session_state.editing_image["content"]
    img_name = st.session_state.editing_image["filename"]

    # Display image to be edited
    st.image(Image.open(BytesIO(img_data)), caption=f"Editing: {img_name}", use_column_width=True)

    edit_prompt = st.text_area("Enter your edit instruction", height=100, key="inline_edit_prompt")
    num_edits = st.slider("🧾 Number of edited versions", 1, 3, 1, key="inline_num_edits")

    if st.button("🚀 Apply Edit"):
        if not edit_prompt.strip():
            st.warning("Please enter an edit instruction.")
        else:
            with st.spinner("Editing with Nano Banana..."):
                edited_versions = []
                for _ in range(num_edits):
                    edited = run_edit_flow(edit_prompt, img_data)
                    if edited:
                        edited_versions.append(edited)

                if edited_versions:
                    for i, out_bytes in enumerate(edited_versions):
                        st.image(Image.open(BytesIO(out_bytes)), caption=f"Edited Version {i+1}", use_column_width=True)
                        st.download_button(
                            f"⬇️ Download Edited {i+1}",
                            data=out_bytes,
                            file_name=f"edited_{i}_{img_name}",
                            mime="image/png",
                            key=f"edit_dl_{i}"
                        )
                else:
                    st.error("❌ No edited image returned by Nano Banana.")

    # Optional: Reset editor
    if st.button("❌ Clear Editor"):
        st.session_state.editing_image = None
        st.experimental_rerun()

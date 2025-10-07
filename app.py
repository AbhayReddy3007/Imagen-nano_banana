import os
import datetime
import json
from io import BytesIO
import streamlit as st
from PIL import Image
import base64

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
    """Extract refined text safely."""
    if hasattr(resp, "text") and resp.text:
        return resp.text
    if hasattr(resp, "candidates") and resp.candidates:
        try:
            return resp.candidates[0].content.parts[0].text
        except Exception:
            pass
    return str(resp)

def get_image_bytes_from_genobj(gen_obj):
    """Extract image bytes from Imagen output."""
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

def run_nano_banana_edit(edit_prompt, base_bytes):
    """Edit an image using Nano Banana (Gemini 2.5 Flash Image) with proper image editing."""
    try:
        # Create the image part
        input_image = Part.from_data(mime_type="image/png", data=base_bytes)
        
        # More specific editing instructions for better results
        edit_instruction = f"""
        <EDITING_TASK>
        You are an expert AI image editor. Modify the provided image according to these instructions: {edit_prompt}
        
        Important rules:
        1. Maintain the core composition and style when possible
        2. Apply the requested changes precisely
        3. Return ONLY the edited image as inline data
        4. Do not add any text, watermarks, or captions
        5. Keep the same aspect ratio and quality
        
        Expected output: Edited image only, no text response.
        </EDITING_TASK>
        """
        
        # Generate content with the image and edit instructions
        response = NANO_BANANA.generate_content([edit_instruction, input_image])
        
        # Extract the edited image from response
        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            return part.inline_data.data
        
        # Alternative extraction method
        if hasattr(response, 'parts'):
            for part in response.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    return part.inline_data.data
        
        st.error("❌ No edited image found in Nano Banana response")
        return None
        
    except Exception as e:
        st.error(f"❌ Nano Banana editing error: {str(e)}")
        return None

# ---------------- PROMPT TEMPLATES ----------------
PROMPT_TEMPLATES = {
    "Marketing": "You are a senior AI prompt engineer creating polished prompts for marketing and advertising visuals. Transform this into a professional marketing image prompt: {USER_PROMPT}",
    "Design": "You are a creative design expert. Transform this into a visually inspiring design image prompt: {USER_PROMPT}",
    "General": "You are an expert AI prompt engineer. Expand this into a detailed, descriptive image prompt: {USER_PROMPT}",
    "DPEX": "You are a technology expert. Transform this into a professional IT/tech-focused image prompt: {USER_PROMPT}",
    "HR": "You are an HR professional. Transform this into a workplace/HR-focused image prompt: {USER_PROMPT}",
    "Business": "You are a business expert. Transform this into a professional corporate/business image prompt: {USER_PROMPT}"
}

# ---------------- STYLE DESCRIPTIONS ----------------
STYLE_DESCRIPTIONS = {
    "None": "No special styling",
    "Smart": "Clean, balanced, professional look",
    "Cinematic": "Film-style composition with dramatic lighting",
    "Creative": "Playful, imaginative, experimental style",
    "Bokeh": "Shallow depth of field with blurred backgrounds",
    "Minimalist": "Simple, uncluttered with negative space",
    "Vibrant": "Bold, saturated colors, high contrast",
    "Watercolor": "Soft, fluid strokes with washed-out textures",
    "Oil Painting": "Rich, textured brushstrokes",
    "Anime": "Japanese animation style",
    "Cartoon": "Playful, exaggerated features",
    "Cyberpunk": "Futuristic neon city vibes",
    "Vintage": "Old-school, retro tones with film grain"
}

# ---------------- GENERATE SECTION ----------------
st.header("✨ Generate Images (Imagen 4)")

dept = st.selectbox("🏢 Department", list(PROMPT_TEMPLATES.keys()), index=0)
style = st.selectbox("🎨 Style", list(STYLE_DESCRIPTIONS.keys()), index=0)
user_prompt = st.text_area("Enter your prompt", height=120, placeholder="Describe the image you want to generate...")
num_images = st.slider("🧾 Number of images", 1, 4, 1)

# GENERATE BUTTON
if st.button("🚀 Generate with Imagen"):
    if not user_prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Refining prompt with Gemini..."):
            # Simple prompt refinement without complex templates
            refinement_prompt = PROMPT_TEMPLATES[dept].replace("{USER_PROMPT}", user_prompt)
            
            # Add style if selected
            if style != "None":
                refinement_prompt += f"\n\nApply this style: {STYLE_DESCRIPTIONS[style]}"
            
            st.write(f"**Refinement prompt sent to Gemini:** {refinement_prompt}")
            
            try:
                text_resp = TEXT_MODEL.generate_content(refinement_prompt)
                enhanced_prompt = safe_get_enhanced_text(text_resp).strip()
                st.success(f"**🔮 Enhanced Prompt:**\n\n{enhanced_prompt}")
            except Exception as e:
                st.error(f"Error refining prompt: {e}")
                # Use original prompt if refinement fails
                enhanced_prompt = user_prompt
                if style != "None":
                    enhanced_prompt += f", {STYLE_DESCRIPTIONS[style]}"
                st.info(f"Using fallback prompt: {enhanced_prompt}")

        with st.spinner("Generating with Imagen 4..."):
            try:
                resp = IMAGEN_MODEL.generate_images(
                    prompt=enhanced_prompt, 
                    number_of_images=num_images
                )
                
                if not resp or not resp.images:
                    st.error("No images generated. Please try a different prompt.")
                    st.stop()
                    
            except Exception as e:
                st.error(f"⚠️ Imagen error: {e}")
                st.stop()

            # Display generated images
            for i in range(min(num_images, len(resp.images))):
                try:
                    gen_obj = resp.images[i]
                    img_bytes = get_image_bytes_from_genobj(gen_obj)
                    if not img_bytes:
                        st.warning(f"Could not extract image {i+1}")
                        continue
                    
                    filename = f"{dept.lower()}_{style.lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.png"
                    
                    # Store in session state
                    st.session_state.generated_images.append({
                        "filename": filename, 
                        "content": img_bytes,
                        "prompt": enhanced_prompt
                    })

                    # Display image and controls
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.image(Image.open(BytesIO(img_bytes)), caption=filename, use_column_width=True)
                    with col2:
                        # Download button
                        st.download_button(
                            "⬇️ Download",
                            data=img_bytes,
                            file_name=filename,
                            mime="image/png",
                            key=f"dl_{i}_{datetime.datetime.now().timestamp()}",
                        )
                        
                        # Edit button
                        if st.button("🪄 Edit with Nano Banana", key=f"edit_btn_{i}"):
                            st.session_state.editing_image = {
                                "filename": filename, 
                                "content": img_bytes,
                                "original_prompt": enhanced_prompt
                            }
                            st.toast("✅ Image sent to Nano Banana Editor!")
                            st.rerun()
                            
                except Exception as e:
                    st.warning(f"⚠️ Unable to process image {i+1}: {e}")

# ---------------- NANO BANANA EDIT SECTION ----------------
if st.session_state.editing_image:
    st.divider()
    st.header("🖌️ Edit with Nano Banana")
    
    img_data = st.session_state.editing_image["content"]
    img_name = st.session_state.editing_image["filename"]
    original_prompt = st.session_state.editing_image.get("original_prompt", "")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(Image.open(BytesIO(img_data)), caption=f"Editing: {img_name}", use_column_width=True)
        if original_prompt:
            with st.expander("Original Prompt"):
                st.write(original_prompt)
    
    with col2:
        st.subheader("Edit Instructions")
        edit_prompt = st.text_area(
            "What changes would you like to make?",
            placeholder="Examples:\n• Change background to beach\n• Make it more colorful\n• Add a cat\n• Convert to black and white",
            height=100,
            key="nano_edit_prompt"
        )
        
        if st.button("🚀 Apply Edit", type="primary", key="nano_edit_btn"):
            if not edit_prompt.strip():
                st.warning("Please enter edit instructions.")
            else:
                with st.spinner("Editing image with Nano Banana..."):
                    edited_bytes = run_nano_banana_edit(edit_prompt, img_data)
                    
                    if edited_bytes:
                        st.success("✅ Edit completed!")
                        
                        # Display edited image
                        edited_filename = f"edited_{img_name}"
                        st.image(
                            Image.open(BytesIO(edited_bytes)), 
                            caption="Edited Version",
                            use_column_width=True
                        )
                        
                        # Download button for edited image
                        st.download_button(
                            "⬇️ Download Edited",
                            data=edited_bytes,
                            file_name=edited_filename,
                            mime="image/png",
                            key="nano_dl",
                        )
                    else:
                        st.error("❌ Editing failed. The Nano Banana model might not support direct image editing yet.")
        
        if st.button("❌ Clear Editor", type="secondary"):
            st.session_state.editing_image = None
            st.rerun()

# ---------------- DEBUG INFO ----------------
with st.sidebar:
    st.header("🔧 Debug Info")
    st.write(f"Generated images: {len(st.session_state.generated_images)}")
    st.write(f"Editing image: {st.session_state.editing_image is not None}")
    
    if st.button("Clear All Images"):
        st.session_state.generated_images = []
        st.session_state.editing_image = None
        st.rerun()

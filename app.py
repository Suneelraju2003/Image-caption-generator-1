import streamlit as st
from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    pipeline
)

st.set_page_config(page_title="Image Caption + AI Description")

# ---------- Load Models ----------
@st.cache_resource
def load_models():
    # Image Caption Model
    caption_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    caption_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    # Free Text Generation Model (LLM)
    text_generator = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=150
    )

    return caption_processor, caption_model, text_generator


processor, caption_model, text_generator = load_models()

# ---------- Functions ----------
def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    output = caption_model.generate(**inputs, max_length=30)
    return processor.decode(output[0], skip_special_tokens=True)


def generate_description(caption):
    prompt = (
        f"Expand the following image caption into a detailed, "
        f"clear description:\n\nCaption: {caption}\n\nDescription:"
    )
    result = text_generator(prompt)[0]["generated_text"]
    return result.replace(prompt, "").strip()


# ---------- UI ----------
st.title("🖼️ Image Caption + AI Description")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    if st.button("Generate"):
        with st.spinner("Analyzing image..."):
            caption = generate_caption(image)
            description = generate_description(caption)

        st.subheader("📌 Image Caption")
        st.write(caption)

        st.subheader("🤖 AI-Generated Description")
        st.write(description)

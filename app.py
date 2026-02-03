import streamlit as st
from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

st.set_page_config(page_title="Image Caption + AI Description")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    # Image Caption Model
    caption_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    caption_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    # Text Description Model (FLAN-T5 – SAFE LOADING)
    text_tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-base"
    )
    text_model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-base"
    )

    return caption_processor, caption_model, text_tokenizer, text_model


processor, caption_model, text_tokenizer, text_model = load_models()

# ---------------- FUNCTIONS ----------------
def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    output = caption_model.generate(
        **inputs,
        max_length=30
    )
    return processor.decode(
        output[0],
        skip_special_tokens=True
    )


def generate_description(caption):
    prompt = (
        "You are an AI that describes images.\n"
        "ONLY describe what is clearly visible in the image.\n"
        "DO NOT invent locations, history, names, or stories.\n\n"
        f"Image caption: {caption}\n\n"
        "Clear visual description:"
    )

    inputs = text_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    outputs = text_model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False
    )

    return text_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


# ---------------- UI ----------------
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

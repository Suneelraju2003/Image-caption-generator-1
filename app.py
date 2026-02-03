import streamlit as st
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    pipeline
)

st.set_page_config(page_title="Image Caption + AI Description")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    # Image Captioning Model
    caption_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    caption_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    # Text Model (YOUR CURRENT MODEL – USED CORRECTLY)
    text_generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

    return caption_processor, caption_model, text_generator


processor, caption_model, text_generator = load_models()

# ---------------- FUNCTIONS ----------------
def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    output = caption_model.generate(
        **inputs,
        max_length=30
    )
    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )
    return caption


def generate_description(caption):
    prompt = (
        "You are an AI that describes images.\n"
        "ONLY describe what is clearly visible in the image.\n"
        "DO NOT invent locations, history, names, or stories.\n"
        "DO NOT mention anything not visible.\n\n"
        f"Image caption: {caption}\n\n"
        "Clear visual description:"
    )

    result = text_generator(
        prompt,
        max_new_tokens=120,
        do_sample=False
    )[0]["generated_text"]

    return result.strip()


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

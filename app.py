import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

st.set_page_config(page_title="Image Caption Generator")

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model

processor, model = load_model()

st.title("🖼️ Image Caption Generator")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    if st.button("Generate Caption"):
        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(
            output[0],
            skip_special_tokens=True
        )
        st.success(f"📝 {caption}")

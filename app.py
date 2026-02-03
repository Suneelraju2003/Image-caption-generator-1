import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

st.set_page_config(page_title="Image Understanding AI")

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

st.title("🖼️ Image Caption, Description & Web Knowledge")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "png", "jpeg"]
)

def get_caption(image):
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs, max_length=30)
    return processor.decode(output[0], skip_special_tokens=True)

def get_detailed_description(image):
    prompt = "Describe this image in detail"
    inputs = processor(image, prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=120)
    return processor.decode(output[0], skip_special_tokens=True)

def search_wikipedia(query):
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_")
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "extract" in data:
            return data["extract"]
    return None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing image..."):
            caption = get_caption(image)
            description = get_detailed_description(image)

            wiki_result = search_wikipedia(caption)

        st.subheader("📌 Caption")
        st.write(caption)

        st.subheader("📝 Description")
        st.write(description)

        st.subheader("🌐 Web Availability")
        if wiki_result:
            st.success("Similar images / topic found on the web")
            st.subheader("🏛️ History / Geography")
            st.write(wiki_result)
        else:
            st.warning("No reliable web history found for this image")

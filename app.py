import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import pytesseract
import requests

st.set_page_config(page_title="Advanced Image Understanding AI")

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

def get_caption(image):
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs, max_length=30)
    return processor.decode(output[0], skip_special_tokens=True)

def get_detailed_description(image):
    prompt = "Describe this image in detail"
    inputs = processor(image, prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=120)
    return processor.decode(output[0], skip_special_tokens=True)

def extract_text(image):
    return pytesseract.image_to_string(image)

def search_wikipedia(query):
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json"
    }
    res = requests.get(search_url, params=params).json()
    if res["query"]["search"]:
        title = res["query"]["search"][0]["title"]
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        summary = requests.get(summary_url).json()
        return title, summary.get("extract")
    return None, None

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    if st.button("Analyze"):
        caption = get_caption(image)
        description = get_detailed_description(image)
        ocr_text = extract_text(image)

        title, wiki = search_wikipedia(ocr_text or caption)

        st.subheader("📌 Caption")
        st.write(caption)

        st.subheader("📝 Description")
        st.write(description)

        st.subheader("🔍 Extracted Text (OCR)")
        st.write(ocr_text.strip() if ocr_text else "No text detected")

        st.subheader("🌐 Web Availability")
        if wiki:
            st.success(f"Similar topic found: {title}")
            st.subheader("📚 History / Geography")
            st.write(wiki)
        else:
            st.warning("No reliable web history found")

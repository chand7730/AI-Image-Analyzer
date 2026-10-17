
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import pipeline

st.set_page_config(
    page_title="AI Image Analyzer",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ AI Image Analyzer")
st.write("Upload an image to explore image processing and AI.")

@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="google/vit-base-patch16-224"
    )

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

brightness = st.slider(
    "Adjust brightness",
    min_value=0,
    max_value=100,
    value=50,
    step=5
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(image)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    bright = cv2.convertScaleAbs(
        bgr, alpha=1.0, beta=brightness
    )
    bright_rgb = cv2.cvtColor(
        bright, cv2.COLOR_BGR2RGB
    )

    edges = cv2.Canny(gray, 100, 200)

    st.subheader("Image Information")
    st.write(f"Width: {rgb.shape[1]} pixels")
    st.write(f"Height: {rgb.shape[0]} pixels")
    st.write(f"Channels: {rgb.shape[2]}")

    c1, c2 = st.columns(2)

    with c1:
        st.image(rgb, caption="Original Image",
                 use_container_width=True)
        st.image(gray, caption="Grayscale",
                 use_container_width=True)

    with c2:
        st.image(bright_rgb, caption="Brightness Increased",
                 use_container_width=True)
        st.image(edges, caption="Edge Detection",
                 use_container_width=True)

    st.divider()
    st.subheader("🤖 AI Object Recognition")

    if st.button("Analyze Image with AI"):
        with st.spinner("AI model is loading and analyzing..."):
            try:
                classifier = load_model()
                results = classifier(image, top_k=5)

                st.success("AI analysis complete!")

                for result in results:
                    st.write(
                        f"**{result['label']}** — "
                        f"{result['score'] * 100:.2f}% confidence"
                    )

            except Exception as e:
                st.error("AI model could not run.")
                st.write(str(e))

    st.info(
        "Note: This model predicts image categories. "
        "It does not locate every object with bounding boxes."
    )

else:
    st.info("Upload an image to begin.")

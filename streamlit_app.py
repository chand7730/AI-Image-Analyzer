import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="AI Image Analyzer",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ AI Image Analyzer")
st.write("Upload an image to explore image processing with OpenCV.")

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

    st.subheader("Image information")
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

    st.success("Image analysis complete!")
else:
    st.info("Upload an image to begin.")


import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO

st.set_page_config(
    page_title="AI Image Analyzer",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ AI Image Analyzer")
st.write("Image Processing, AI Classification and Object Detection")

@st.cache_resource
def load_classifier():
    return pipeline(
        "image-classification",
        model="google/vit-base-patch16-224"
    )

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

uploaded_file = st.file_uploader(
    "Upload an image",
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
        st.image(
            rgb, caption="Original Image",
            use_container_width=True
        )
        st.image(
            gray, caption="Grayscale",
            use_container_width=True
        )

    with c2:
        st.image(
            bright_rgb, caption="Brightness Increased",
            use_container_width=True
        )
        st.image(
            edges, caption="Edge Detection",
            use_container_width=True
        )

    st.divider()
    st.subheader("🤖 AI Image Classification")

    if st.button("Analyze Image with AI"):
        with st.spinner("AI is analyzing the image..."):
            try:
                classifier = load_classifier()
                predictions = classifier(image, top_k=5)

                for item in predictions:
                    st.write(
                        f"**{item['label']}** — "
                        f"{item['score'] * 100:.2f}% confidence"
                    )
            except Exception as e:
                st.error(f"Classification error: {e}")

    st.divider()
    st.subheader("🎯 YOLO Object Detection")

    st.write(
        "Detect supported objects and draw boxes around them."
    )

    if st.button("Detect Objects with YOLO"):
        with st.spinner(
            "Loading YOLO model and detecting objects..."
        ):
            try:
                model = load_yolo()

                results = model.predict(
                    source=rgb,
                    conf=0.25,
                    verbose=False
                )

                result = results[0]

                # YOLO output is BGR; convert it to RGB
                detected_image = cv2.cvtColor(
                    result.plot(),
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    detected_image,
                    caption="YOLO Detection Results",
                    use_container_width=True
                )

                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    st.success(
                        f"Detected {len(boxes)} object(s)."
                    )

                    for box in boxes:
                        class_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        name = result.names[class_id]

                        st.write(
                            f"**{name}** — "
                            f"{confidence * 100:.2f}% confidence"
                        )
                else:
                    st.info(
                        "No supported objects were detected "
                        "in this image."
                    )

            except Exception as e:
                st.error("YOLO could not run.")
                st.write(str(e))

else:
    st.info("Upload an image to begin.")

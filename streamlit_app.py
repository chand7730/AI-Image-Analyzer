
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO

st.set_page_config(
    page_title="AI Image Analyzer",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Premium Green & White Design ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f3fbf6 0%, #ffffff 55%);
    color: #163b2a;
}
[data-testid="stHeader"] {
    background: rgba(255,255,255,0.85);
}
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    background: linear-gradient(135deg, #087f4f, #16a36a);
    padding: 32px;
    border-radius: 22px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(8,127,79,0.16);
}
.hero h1 {
    color: white;
    font-size: 2.35rem;
    margin-bottom: 8px;
}
.hero p {
    color: #e8fff1;
    font-size: 1.05rem;
}
.section-card {
    background: white;
    border: 1px solid #dcefe3;
    border-radius: 16px;
    padding: 20px;
    margin: 12px 0;
    box-shadow: 0 4px 16px rgba(20,80,45,0.05);
}
div.stButton > button {
    background: #087f4f;
    color: white;
    border: 1px solid #087f4f;
    border-radius: 10px;
    padding: 0.55rem 1rem;
    font-weight: 600;
}
div.stButton > button:hover {
    background: #06663f;
    color: white;
    border-color: #06663f;
}
[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #65b98b;
    border-radius: 14px;
    padding: 12px;
}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dcefe3;
    border-radius: 12px;
    padding: 14px;
}
h1, h2, h3 {
    color: #12613e;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1>🖼️ AI Image Analyzer</h1>
    <p>Smart Image Processing · AI Classification · Object Detection</p>
    <p>Upload an image and explore it with computer vision and AI.</p>
</div>
""", unsafe_allow_html=True)

# ---------- AI Models ----------
@st.cache_resource
def load_classifier():
    return pipeline(
        "image-classification",
        model="google/vit-base-patch16-224"
    )

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🌿 Control Panel")
    st.write("Upload an image to begin your analysis.")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"]
    )

    st.divider()
    brightness = st.slider(
        "☀️ Brightness adjustment",
        min_value=0,
        max_value=100,
        value=30,
        step=5
    )
    st.caption("Move the slider to adjust image brightness.")
  
# ---------- Analysis History ----------
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

with st.sidebar:
    st.divider()
    st.markdown("## 📜 Analysis History")

    if st.session_state.analysis_history:
        st.caption("Recent AI analysis results")

        for i, item in enumerate(
            reversed(st.session_state.analysis_history)
        ):
            st.markdown(
                f"**{len(st.session_state.analysis_history) - i}.** "
                f"{item['type']}"
            )
            st.caption(item["result"])

    else:
        st.info("Your analysis results will appear here.")

# ---------- Main App ----------
if uploaded_file is None:
    st.markdown("""
    <div class="section-card">
        <h3>👋 Welcome to your AI workspace</h3>
        <p>Upload a JPG or PNG image from the sidebar to get started.</p>
        <p>Explore image processing, AI classification and object detection.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ✨ Available Features")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="section-card">
            <h3>📷 OpenCV</h3>
            <p>Grayscale, brightness and edge detection.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="section-card">
            <h3>🧠 ViT AI</h3>
            <p>Classify an image and view the top predictions.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="section-card">
            <h3>🎯 YOLO</h3>
            <p>Detect supported objects and draw bounding boxes.</p>
        </div>
        """, unsafe_allow_html=True)

else:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        rgb = np.array(image)

        # OpenCV image processing
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        bright = cv2.convertScaleAbs(
            bgr, alpha=1.0, beta=brightness
        )
        bright_rgb = cv2.cvtColor(
            bright, cv2.COLOR_BGR2RGB
        )
        edges = cv2.Canny(gray, 100, 200)

        # Image information
        st.markdown("## 📊 Image Overview")
        m1, m2, m3 = st.columns(3)

        m1.metric("Width", f"{rgb.shape[1]} px")
        m2.metric("Height", f"{rgb.shape[0]} px")
        m3.metric("Color Channels", str(rgb.shape[2]))

        st.divider()

        # OpenCV results
        st.markdown("## 📷 Image Processing")
        download_col1, download_col2 = st.columns(2)

        with download_col1:
            st.download_button(
                label="⬇️ Download Original Image",
                data=uploaded_file.getvalue(),
                file_name="original_image.png",
                mime="image/png",
                key="download_original"
            )

        with download_col2:
            processed_image = Image.fromarray(bright_rgb)
            import io
            image_buffer = io.BytesIO()
            processed_image.save(image_buffer, format="PNG")

            st.download_button(
                label="⬇️ Download Brightness Image",
                data=image_buffer.getvalue(),
                file_name="brightness_image.png",
                mime="image/png",
                key="download_brightness"
            )

        gray_image = Image.fromarray(gray)
        gray_buffer = io.BytesIO()
        gray_image.save(gray_buffer, format="PNG")

        st.download_button(
            label="⬇️ Download Grayscale Image",
            data=gray_buffer.getvalue(),
            file_name="grayscale_image.png",
            mime="image/png",
            key="download_grayscale"
        )

        edge_image = Image.fromarray(edges)
        edge_buffer = io.BytesIO()
        edge_image.save(edge_buffer, format="PNG")

        st.download_button(
            label="⬇️ Download Edge Detection Image",
            data=edge_buffer.getvalue(),
            file_name="edge_detection.png",
            mime="image/png",
            key="download_edges"
        )

        tab1, tab2, tab3, tab4 = st.tabs([
            "Original", "Grayscale", "Brightness", "Edges"
        ])

        with tab1:
            st.image(
                rgb,
                caption="Original Image",
                use_container_width=True
            )

        with tab2:
            st.image(
                gray,
                caption="Grayscale Image",
                use_container_width=True
            )

        with tab3:
            st.image(
                bright_rgb,
                caption="Brightness Adjusted Image",
                use_container_width=True
            )

        with tab4:
            st.image(
                edges,
                caption="Canny Edge Detection",
                use_container_width=True
            )

        st.divider()

        # ViT classification
        st.markdown("## 🧠 AI Image Classification")
        st.write(
            "ViT predicts which categories best match the whole image."
        )

        if st.button("🔍 Analyze Image with ViT", key="vit_button"):
            with st.spinner("AI is analyzing your image..."):
                try:
                    classifier = load_classifier()
                    predictions = classifier(image, top_k=5)
                    
                    history_result = ", ".join(
                        [
                            f"{item['label']} ({item['score'] * 100:.1f}%)"
                            for item in predictions
                        ]
                    )

                    st.session_state.analysis_history.append({
                        "type": "🧠 ViT Classification",
                        "result": history_result
                    })


                    st.success("Image classification completed!")

                    for item in predictions:
                        score = item["score"] * 100
                        st.markdown(
                            f"**{item['label']}** — {score:.2f}% confidence"
                        )
                        st.progress(min(int(score), 100))

                except Exception as e:
                    st.error(f"Classification error: {e}")

        st.caption(
            "Confidence is the model's score for a category, "
            "not the percentage of image area."
        )

        st.divider()

        # YOLO object detection
        st.markdown("## 🎯 YOLO Object Detection")
        st.write(
            "Find supported objects and display boxes around them."
        )

        if st.button("🚀 Detect Objects with YOLO", key="yolo_button"):
            with st.spinner("YOLO is detecting objects..."):
                try:
                    model = load_yolo()
                    results = model.predict(
                        source=rgb,
                        conf=0.25,
                        verbose=False
                    )

                    result = results[0]
                    detected_image = cv2.cvtColor(
                        result.plot(),
                        cv2.COLOR_BGR2RGB
                    )

                    st.image(
                        detected_image,
                        caption="YOLO Detection Results",
                        use_container_width=True
                    )
                    
                    import io

                    yolo_buffer = io.BytesIO()
                    Image.fromarray(detected_image).save(
                        yolo_buffer, format="PNG"
                    )

                    st.download_button(
                        label="⬇️ Download YOLO Detection Image",
                        data=yolo_buffer.getvalue(),
                        file_name="yolo_detection_result.png",
                        mime="image/png",
                        key="download_yolo"
                    )


                    boxes = result.boxes

                    if boxes is not None and len(boxes) > 0:
                        st.success(
                            f"Detected {len(boxes)} object(s)."
                        )
                        
                        yolo_history_result = []

                        for box in boxes:
                            class_id = int(box.cls[0].item())
                            confidence = float(box.conf[0].item())
                            name = result.names[class_id]

                            yolo_history_result.append(
                                f"{name} ({confidence * 100:.1f}%)"
                            )

                        st.session_state.analysis_history.append({
                            "type": "🎯 YOLO Object Detection",
                            "result": ", ".join(yolo_history_result)
                        })


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

    except Exception as e:
        st.error(f"Could not open this image: {e}")

# ---------- Footer ----------
st.divider()

st.markdown("""
<div style="
    background: linear-gradient(135deg, #087f4f, #16a36a);
    padding: 24px;
    border-radius: 18px;
    text-align: center;
    color: white;
">
    <h2 style="color: white; margin-bottom: 8px;">
        🌿 AI Image Analyzer
    </h2>
    <p style="color: #e8fff1;">
        Powered by OpenCV, ViT &amp; YOLO
    </p>
    <hr style="border-color: #a6e5c0;">
    <p style="color: white; margin-bottom: 4px;">
        Developed by <strong>Chand Yousaf</strong>
    </p>
    <p style="color: #e8fff1; font-size: 13px;">
        AI-Powered Image Processing &amp; Object Detection
    </p>
</div>
""", unsafe_allow_html=True)

import gradio as gr
import cv2
import numpy as np


def analyze_image(image, brightness):
    if image is None:
        return None, None, None, None, "پہلے تصویر اپ لوڈ کریں۔"

    # Gradio سے آنے والی تصویر RGB فارمیٹ میں ہوتی ہے
    rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # تصویر کی معلومات
    height, width = rgb.shape[:2]
    info = (
        f"### تصویر کی معلومات\n"
        f"- چوڑائی: {width} pixels\n"
        f"- اونچائی: {height} pixels\n"
        f"- رنگوں کے چینلز: {rgb.shape[2]}"
    )

    # تصویر کو Grayscale میں تبدیل کرنا
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Brightness بڑھانا
    bright = cv2.convertScaleAbs(
        bgr, alpha=1.0, beta=int(brightness)
    )
    bright_rgb = cv2.cvtColor(bright, cv2.COLOR_BGR2RGB)

    # تصویر کے کنارے معلوم کرنا
    edges = cv2.Canny(gray, 100, 200)

    return rgb, gray, bright_rgb, edges, info


with gr.Blocks(
    theme=gr.themes.Soft(),
    title="AI Image Analyzer"
) as demo:

    gr.Markdown(
        """
        # 🖼️ AI Image Analyzer
        ### اپنی تصویر اپ لوڈ کریں اور اس پر مختلف تجربات کریں۔
        OpenCV کی مدد سے تصویر کے رنگ، روشنی اور کنارے دیکھیں۔
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="اپنی تصویر اپ لوڈ کریں"
            )

            brightness = gr.Slider(
                minimum=0,
                maximum=100,
                value=50,
                step=5,
                label="روشنی بڑھائیں"
            )

            analyze_button = gr.Button(
                "تصویر کا تجزیہ کریں",
                variant="primary"
            )

        with gr.Column(scale=2):
            info_output = gr.Markdown(
                "تصویر اپ لوڈ کرنے کے بعد تجزیہ کریں۔"
            )

            with gr.Tabs():
                with gr.Tab("اصل تصویر"):
                    original_output = gr.Image(
                        label="Original Image"
                    )

                with gr.Tab("بلیک اینڈ وائٹ"):
                    gray_output = gr.Image(
                        label="Grayscale"
                    )

                with gr.Tab("زیادہ روشنی"):
                    bright_output = gr.Image(
                        label="Brightness"
                    )

                with gr.Tab("کناروں کی شناخت"):
                    edges_output = gr.Image(
                        label="Edge Detection"
                    )

    analyze_button.click(
        fn=analyze_image,
        inputs=[image_input, brightness],
        outputs=[
            original_output,
            gray_output,
            bright_output,
            edges_output,
            info_output
        ]
    )

demo.launch(share=True)

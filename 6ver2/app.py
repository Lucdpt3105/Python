import cv2
import gradio as gr
import numpy as np
from PIL import Image


class ObjectRemovalApp:
    """Object removal with OpenCV inpainting (no deep learning)."""

    def image_size_text(self, input_data):
        if input_data is None:
            return "Image size: -"
        image = input_data.get("background")
        if image is None:
            return "Image size: -"
        if isinstance(image, Image.Image):
            image = np.array(image)
        if not isinstance(image, np.ndarray) or image.ndim < 2:
            return "Image size: -"
        h, w = image.shape[:2]
        return f"Image size: {w} x {h}px"

    def process(self, input_data, algorithm, radius):
        if input_data is None:
            return None, None

        image = input_data.get("background")
        layers = input_data.get("layers", [])

        if image is None:
            return None, None

        if isinstance(image, Image.Image):
            image = np.array(image)

        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        if not layers or layers[0] is None:
            return image, image

        layer = layers[0]
        if isinstance(layer, Image.Image):
            layer = np.array(layer)

        if layer.shape[:2] != image.shape[:2]:
            layer = cv2.resize(layer, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        if layer.ndim == 3 and layer.shape[2] == 4:
            mask = layer[:, :, 3]
        elif layer.ndim == 3:
            mask = np.max(layer[:, :, :3], axis=2)
        else:
            mask = layer

        mask = (mask > 10).astype(np.uint8)
        mask_8u = (mask * 255).astype(np.uint8)

        overlay = image.copy()
        red = np.zeros_like(image)
        red[mask == 1] = [255, 0, 0]
        overlay = cv2.addWeighted(overlay, 0.7, red, 0.3, 0)

        flag = cv2.INPAINT_TELEA if algorithm == "TELEA" else cv2.INPAINT_NS
        result = cv2.inpaint(image, mask_8u, float(radius), flag)

        return overlay, result


def create_demo():
    app = ObjectRemovalApp()

    with gr.Blocks(title="Object Removal - OpenCV") as demo:
        gr.Markdown("""
        # Object Removal (No Deep Learning)
        Upload image, paint over object to remove, then click Run.
        """)

        with gr.Row():
            with gr.Column():
                editor = gr.ImageEditor(
                    label="Input Image + Mask",
                    type="numpy",
                    brush=gr.Brush(colors=["#ff0000"], default_size=24),
                    height=450,
                )
                image_size = gr.Textbox(
                    label="Image Info",
                    value="Image size: -",
                    interactive=False,
                )
                algorithm = gr.Radio(
                    choices=["TELEA", "NS"],
                    value="TELEA",
                    label="Inpainting Algorithm",
                )
                radius = gr.Slider(
                    minimum=1,
                    maximum=15,
                    value=5,
                    step=1,
                    label="Inpaint Radius",
                )
                with gr.Row():
                    run_btn = gr.Button("Run", variant="primary")
                    clear_btn = gr.Button("Clear")

            with gr.Column():
                masked_preview = gr.Image(label="Mask Preview", height=220)
                output = gr.Image(label="Restored Image", height=220)

        editor.change(
            fn=app.image_size_text,
            inputs=[editor],
            outputs=[image_size],
        )

        run_btn.click(
            fn=app.process,
            inputs=[editor, algorithm, radius],
            outputs=[masked_preview, output],
        )

        clear_btn.click(
            fn=lambda: (None, "Image size: -", None, None),
            outputs=[editor, image_size, masked_preview, output],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False, show_error=True)

"""
Smart Content Filler - Gradio Interface
AI Inpainting với LaMa Model
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image
from mask_generator import MaskGenerator


class InpaintingApp:
    """Main application class - CV2 Inpainting"""

    def __init__(self):
        print("🎨 Using CV2 Inpainting (No deep learning model needed)")
        self.mask_generator = MaskGenerator()

    def process_image(self, input_dict, mask_type="Free Draw"):
        """
        Process image with mask
        Args:
            input_dict: Dictionary containing 'image' and 'mask' from ImageEditor
            mask_type: Type of automatic mask to apply if no manual mask
        Returns:
            Tuple of (masked_image, inpainted_image)
        """
        if input_dict is None:
            return None, None

        # Extract image and mask
        image = input_dict.get('background')
        mask_layers = input_dict.get('layers', [])

        if image is None:
            return None, None

        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert RGBA to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Resize to smaller size for processing (reduce memory)
        original_size = image.shape[:2]
        max_size = 512
        h, w = original_size

        # Scale down if too large
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))

        print(f"📏 Processing size: {image.shape[:2]}, Original: {original_size}")

        # Create or extract mask
        if len(mask_layers) > 0 and mask_layers[0] is not None:
            # Use user-drawn mask
            mask_image = mask_layers[0]
            if isinstance(mask_image, Image.Image):
                mask_image = np.array(mask_image)

            # Resize mask to match image size first
            if mask_image.shape[:2] != (512, 512):
                mask_image = cv2.resize(mask_image, (512, 512))

            # Convert RGBA to grayscale mask
            if len(mask_image.shape) == 3 and mask_image.shape[2] == 4:
                # Use alpha channel as mask
                mask = mask_image[:, :, 3]
            elif len(mask_image.shape) == 3:
                # Check for any non-zero pixels in RGB channels
                mask = np.max(mask_image[:, :, :3], axis=2)
            else:
                mask = mask_image

            # Threshold to binary
            mask = (mask > 10).astype(np.uint8)
        else:
            # Generate automatic mask based on type
            if mask_type == "Random Rectangle":
                mask = self.mask_generator.generate_rectangle_mask()
            elif mask_type == "Random Free Form":
                mask = self.mask_generator.generate_free_form_mask()
            elif mask_type == "Facial Features":
                mask = self.mask_generator.generate_facial_mask("mixed")
            else:
                # No mask provided
                return image, image

        # Ensure mask is same size as image
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # Create masked image for visualization
        masked_image = image.copy()
        mask_overlay = np.zeros_like(image)
        mask_overlay[mask == 1] = [255, 0, 0]  # Red overlay for masked region
        masked_image = cv2.addWeighted(masked_image, 0.7, mask_overlay, 0.3, 0)

        # Perform CV2 inpainting
        try:
            # Convert mask to 8-bit
            mask_8bit = (mask * 255).astype(np.uint8)

            # CV2 inpainting (TELEA algorithm - better for photos)
            inpainted = cv2.inpaint(image, mask_8bit, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

            print("✅ CV2 inpainting completed")
        except Exception as e:
            print(f"❌ Error during inpainting: {e}")
            return masked_image, image

        # Resize back to original size if needed
        if original_size != (512, 512):
            masked_image = cv2.resize(masked_image, (original_size[1], original_size[0]))
            inpainted = cv2.resize(inpainted, (original_size[1], original_size[0]))

        return masked_image, inpainted

    def generate_random_example(self, mask_type):
        """Generate a random mask example"""
        # Create a sample image (white canvas)
        image = np.ones((512, 512, 3), dtype=np.uint8) * 255

        # Generate mask based on type
        if mask_type == "Random Rectangle":
            mask = self.mask_generator.generate_rectangle_mask()
        elif mask_type == "Random Free Form":
            mask = self.mask_generator.generate_free_form_mask()
        elif mask_type == "Facial Features":
            mask = self.mask_generator.generate_facial_mask("mixed")
        else:
            mask = self.mask_generator.generate_random_mask("mixed")

        # Apply mask to image
        masked_image = self.mask_generator.apply_mask_to_image(image, mask, (200, 200, 200))

        return masked_image


def create_interface():
    """Create Gradio interface"""
    app = InpaintingApp()

    with gr.Blocks(title="Smart Content Filler - AI Inpainting", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎨 Smart Content Filler - CV2 Inpainting
        ### Xóa vật thể & Phục hồi ảnh thông minh

        **Hướng dẫn sử dụng:**
        1. Upload ảnh của bạn
        2. Dùng công cụ vẽ (brush) để đánh dấu vùng cần xóa/phục hồi (màu đỏ)
        3. Hoặc chọn "Auto Generate Mask" để tạo mask tự động
        4. Nhấn "🪄 Magic Inpaint" để xem kết quả
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📤 Input")
                input_image = gr.ImageEditor(
                    label="Upload & Draw Mask",
                    type="numpy",
                    brush=gr.Brush(colors=["#ff0000"], default_size=20),
                    height=400
                )

                mask_type = gr.Radio(
                    choices=["Free Draw", "Random Rectangle", "Random Free Form", "Facial Features"],
                    value="Free Draw",
                    label="Mask Type (if not drawing manually)"
                )

                with gr.Row():
                    inpaint_btn = gr.Button("🪄 Magic Inpaint", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear", size="lg")

            with gr.Column():
                gr.Markdown("### 🖼️ Preview & Result")
                masked_preview = gr.Image(label="Masked Preview", height=200)
                output_image = gr.Image(label="Inpainted Result", height=200)

        gr.Markdown("""
        ---
        ### 📊 Technical Details
        - **Method:** OpenCV Inpainting (TELEA Algorithm)
        - **Features:**
            - ✓ Fast processing (no GPU needed)
            - ✓ Good for small-medium masks
            - ✓ Traditional image processing
            - ✓ Lightweight & efficient
        """)

        with gr.Accordion("🎯 Examples & Demo", open=False):
            gr.Examples(
                examples=[
                    ["Free Draw"],
                    ["Random Rectangle"],
                    ["Random Free Form"],
                    ["Facial Features"]
                ],
                inputs=[mask_type],
                label="Try different mask types"
            )

        # Event handlers
        inpaint_btn.click(
            fn=app.process_image,
            inputs=[input_image, mask_type],
            outputs=[masked_preview, output_image]
        )

        clear_btn.click(
            fn=lambda: (None, None, None),
            outputs=[input_image, masked_preview, output_image]
        )

    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Launching Smart Content Filler...")
    print("=" * 60)

    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

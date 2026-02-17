"""
Smart Content Filler - Gradio Interface
AI Inpainting với LaMa Model
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image
import torch
from lama_model import LaMaInpainting
from mask_generator import MaskGenerator
import os


class InpaintingApp:
    """Main application class"""

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Initialize model
        self.model = LaMaInpainting(device=self.device)

        # Check if pretrained model exists
        checkpoint_path = "6/checkpoints/lama_model.pth"
        if os.path.exists(checkpoint_path):
            try:
                self.model.load_pretrained(checkpoint_path)
                self.model_loaded = True
                print("✓ Pretrained model loaded successfully!")
            except Exception as e:
                print(f"⚠ Could not load pretrained model: {e}")
                self.model_loaded = False
        else:
            print("⚠ No pretrained model found. Using random initialization.")
            print(f"   Expected path: {checkpoint_path}")
            self.model_loaded = False

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

        # Resize to standard size for processing
        original_size = image.shape[:2]
        image = cv2.resize(image, (512, 512))

        # Create or extract mask
        if len(mask_layers) > 0 and mask_layers[0] is not None:
            # Use user-drawn mask
            mask_image = mask_layers[0]
            if isinstance(mask_image, Image.Image):
                mask_image = np.array(mask_image)

            # Convert RGBA to grayscale mask
            if len(mask_image.shape) == 3 and mask_image.shape[2] == 4:
                # Use alpha channel as mask
                mask = mask_image[:, :, 3]
            elif len(mask_image.shape) == 3:
                mask = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY)
            else:
                mask = mask_image

            # Threshold to binary
            mask = (mask > 128).astype(np.uint8)
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
            mask = cv2.resize(mask, (512, 512))

        # Create masked image for visualization
        masked_image = image.copy()
        mask_overlay = np.zeros_like(image)
        mask_overlay[mask == 1] = [255, 0, 0]  # Red overlay for masked region
        masked_image = cv2.addWeighted(masked_image, 0.7, mask_overlay, 0.3, 0)

        # Perform inpainting
        try:
            if self.model_loaded:
                inpainted = self.model.inpaint(image, mask)
            else:
                # Fallback: simple cv2 inpainting if model not loaded
                print("⚠ Using CV2 fallback inpainting (model not loaded)")
                mask_8bit = (mask * 255).astype(np.uint8)
                inpainted = cv2.inpaint(image, mask_8bit, 3, cv2.INPAINT_TELEA)
        except Exception as e:
            print(f"Error during inpainting: {e}")
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
        # 🎨 Smart Content Filler - AI Inpainting
        ### Xóa vật thể & Phục hồi khuôn mặt thông minh

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
        - **Model:** LaMa (Resolution-robust Large Mask Inpainting)
        - **Architecture:** Fast Fourier Convolutions (FFC)
        - **Features:**
            - ✓ High-resolution inpainting
            - ✓ Large mask handling
            - ✓ Context-aware filling
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

"""
Random Mask Generator for Training/Testing Inpainting Models
Implements various mask generation strategies
"""

import numpy as np
import cv2
from typing import Tuple, List
import random


class MaskGenerator:
    """Generate random masks for inpainting tasks"""

    def __init__(self, height: int = 512, width: int = 512):
        self.height = height
        self.width = width

    def generate_random_mask(self, mask_type: str = "mixed") -> np.ndarray:
        """
        Generate random mask based on type
        Args:
            mask_type: Type of mask ('rectangle', 'free_form', 'mixed')
        Returns:
            Binary mask (H, W) with 1 indicating regions to inpaint
        """
        if mask_type == "rectangle":
            return self.generate_rectangle_mask()
        elif mask_type == "free_form":
            return self.generate_free_form_mask()
        elif mask_type == "mixed":
            # Randomly choose between different mask types
            choice = random.choice(["rectangle", "free_form", "multiple_rectangles"])
            if choice == "rectangle":
                return self.generate_rectangle_mask()
            elif choice == "free_form":
                return self.generate_free_form_mask()
            else:
                return self.generate_multiple_rectangles()
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")

    def generate_rectangle_mask(self, min_size: float = 0.1, max_size: float = 0.4) -> np.ndarray:
        """
        Generate random rectangular mask
        Args:
            min_size: Minimum size ratio (0-1)
            max_size: Maximum size ratio (0-1)
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # Random rectangle size
        h_ratio = random.uniform(min_size, max_size)
        w_ratio = random.uniform(min_size, max_size)

        h = int(self.height * h_ratio)
        w = int(self.width * w_ratio)

        # Random position
        y = random.randint(0, self.height - h)
        x = random.randint(0, self.width - w)

        mask[y:y+h, x:x+w] = 1

        return mask

    def generate_multiple_rectangles(self, num_rectangles: int = None) -> np.ndarray:
        """Generate multiple random rectangles"""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        if num_rectangles is None:
            num_rectangles = random.randint(2, 5)

        for _ in range(num_rectangles):
            h = random.randint(20, self.height // 3)
            w = random.randint(20, self.width // 3)
            y = random.randint(0, self.height - h)
            x = random.randint(0, self.width - w)
            mask[y:y+h, x:x+w] = 1

        return mask

    def generate_free_form_mask(self, num_strokes: int = None, max_vertex: int = None) -> np.ndarray:
        """
        Generate free-form mask with random brush strokes
        Args:
            num_strokes: Number of brush strokes
            max_vertex: Maximum number of vertices per stroke
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        if num_strokes is None:
            num_strokes = random.randint(1, 5)

        if max_vertex is None:
            max_vertex = random.randint(5, 15)

        for _ in range(num_strokes):
            # Random starting point
            start_x = random.randint(0, self.width)
            start_y = random.randint(0, self.height)

            # Random brush width
            brush_width = random.randint(10, 40)

            # Generate stroke
            points = [(start_x, start_y)]
            for _ in range(random.randint(3, max_vertex)):
                # Random direction and length
                angle = random.uniform(0, 2 * np.pi)
                length = random.randint(20, 100)

                new_x = int(points[-1][0] + length * np.cos(angle))
                new_y = int(points[-1][1] + length * np.sin(angle))

                # Clip to image boundaries
                new_x = np.clip(new_x, 0, self.width)
                new_y = np.clip(new_y, 0, self.height)

                points.append((new_x, new_y))

            # Draw stroke
            points = np.array(points, dtype=np.int32)
            cv2.polylines(mask, [points], False, 1, thickness=brush_width)

        return mask

    def generate_circular_mask(self, center: Tuple[int, int] = None, radius: int = None) -> np.ndarray:
        """
        Generate circular mask
        Args:
            center: Center position (x, y), random if None
            radius: Radius, random if None
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        if center is None:
            center = (random.randint(0, self.width), random.randint(0, self.height))

        if radius is None:
            radius = random.randint(min(self.height, self.width) // 10,
                                   min(self.height, self.width) // 4)

        cv2.circle(mask, center, radius, 1, -1)

        return mask

    def generate_facial_mask(self, mask_type: str = "random") -> np.ndarray:
        """
        Generate mask specifically for facial restoration
        Args:
            mask_type: 'eyes', 'nose', 'mouth', 'random', or 'mixed'
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # Approximate facial feature positions (normalized)
        # Assuming center face in image
        center_x, center_y = self.width // 2, self.height // 2

        if mask_type == "random" or mask_type not in ["eyes", "nose", "mouth", "mixed"]:
            mask_type = random.choice(["eyes", "nose", "mouth", "mixed"])

        if mask_type == "eyes" or mask_type == "mixed":
            # Left eye
            left_eye_x = int(center_x - self.width * 0.15)
            left_eye_y = int(center_y - self.height * 0.1)
            eye_w = int(self.width * 0.12)
            eye_h = int(self.height * 0.08)
            mask[left_eye_y:left_eye_y+eye_h, left_eye_x:left_eye_x+eye_w] = 1

            # Right eye
            right_eye_x = int(center_x + self.width * 0.03)
            right_eye_y = int(center_y - self.height * 0.1)
            mask[right_eye_y:right_eye_y+eye_h, right_eye_x:right_eye_x+eye_w] = 1

        if mask_type == "nose" or mask_type == "mixed":
            # Nose
            nose_x = int(center_x - self.width * 0.05)
            nose_y = int(center_y - self.height * 0.02)
            nose_w = int(self.width * 0.1)
            nose_h = int(self.height * 0.12)
            mask[nose_y:nose_y+nose_h, nose_x:nose_x+nose_w] = 1

        if mask_type == "mouth" or mask_type == "mixed":
            # Mouth
            mouth_x = int(center_x - self.width * 0.1)
            mouth_y = int(center_y + self.height * 0.12)
            mouth_w = int(self.width * 0.2)
            mouth_h = int(self.height * 0.08)
            mask[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w] = 1

        return mask

    def apply_mask_to_image(self, image: np.ndarray, mask: np.ndarray,
                           fill_value: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """
        Apply mask to image
        Args:
            image: Input image (H, W, 3)
            mask: Binary mask (H, W)
            fill_value: RGB color to fill masked region
        Returns:
            Masked image
        """
        masked_image = image.copy()
        mask_3d = np.stack([mask] * 3, axis=-1)
        masked_image[mask_3d == 1] = fill_value
        return masked_image


def visualize_masks(num_samples: int = 6):
    """Visualize different mask types"""
    import matplotlib.pyplot as plt

    generator = MaskGenerator(512, 512)
    mask_types = ["rectangle", "free_form", "multiple_rectangles",
                 "circular", "facial_eyes", "facial_mixed"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, mask_type in enumerate(mask_types):
        if mask_type == "circular":
            mask = generator.generate_circular_mask()
        elif mask_type.startswith("facial"):
            facial_type = mask_type.split("_")[1]
            mask = generator.generate_facial_mask(facial_type)
        elif mask_type == "multiple_rectangles":
            mask = generator.generate_multiple_rectangles()
        else:
            mask = generator.generate_random_mask(mask_type)

        axes[idx].imshow(mask, cmap='gray')
        axes[idx].set_title(mask_type.replace("_", " ").title())
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('6/sample_masks.png', dpi=150, bbox_inches='tight')
    print("Saved sample masks to '6/sample_masks.png'")


if __name__ == "__main__":
    # Demo: Generate and visualize masks
    print("Generating sample masks...")
    visualize_masks()
    print("Done!")

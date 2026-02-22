"""
LaMa (Resolution-robust Large Mask Inpainting with Fourier Convolutions)
Implementation for Smart Content Filler
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class FourierUnit(nn.Module):
    """Fast Fourier Convolution Unit - Core of LaMa"""

    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        self.groups = groups
        self.conv_layer = nn.Conv2d(
            in_channels * 2, out_channels * 2,
            kernel_size=1, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch = x.shape[0]

        # FFT
        ffted = torch.fft.rfftn(x, dim=(2, 3), norm='ortho')
        ffted = torch.stack([ffted.real, ffted.imag], dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view(batch, -1, ffted.shape[3], ffted.shape[4])

        # Convolution in frequency domain
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))

        # IFFT
        ffted = ffted.view(batch, -1, 2, ffted.shape[2], ffted.shape[3])
        ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        output = torch.fft.irfftn(ffted, s=x.shape[2:], dim=(2, 3), norm='ortho')

        return output


class FFCResNetBlock(nn.Module):
    """Residual block with Fast Fourier Convolution"""

    def __init__(self, dim, ratio_gin=0.5, ratio_gout=0.5):
        super().__init__()

        in_cg = int(dim * ratio_gin)
        in_cl = dim - in_cg
        out_cg = int(dim * ratio_gout)
        out_cl = dim - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        # Local branch (spatial)
        self.conv_l2l = nn.Conv2d(in_cl, out_cl, kernel_size=3, padding=1) if in_cl > 0 and out_cl > 0 else None
        self.conv_l2g = nn.Conv2d(in_cl, out_cg, kernel_size=3, padding=1) if in_cl > 0 and out_cg > 0 else None

        # Global branch (frequency)
        self.conv_g2l = nn.Conv2d(in_cg, out_cl, kernel_size=3, padding=1) if in_cg > 0 and out_cl > 0 else None
        self.ffc = FourierUnit(in_cg, out_cg) if in_cg > 0 and out_cg > 0 else None

        self.bn_l = nn.BatchNorm2d(out_cl) if out_cl > 0 else None
        self.bn_g = nn.BatchNorm2d(out_cg) if out_cg > 0 else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_l, x_g = x if isinstance(x, tuple) else (x, None)

        id_l, id_g = x_l, x_g

        if self.ratio_gin > 0:
            if x_g is None:
                x_g = torch.zeros_like(x_l[:, :int(x_l.shape[1] * self.ratio_gin)])

        out_xl, out_xg = 0, 0

        # Local to Local
        if self.conv_l2l is not None:
            out_xl += self.conv_l2l(x_l)

        # Global to Local
        if self.conv_g2l is not None and x_g is not None:
            out_xl += self.conv_g2l(x_g)

        # Local to Global
        if self.conv_l2g is not None:
            out_xg += self.conv_l2g(x_l)

        # Global to Global (FFC)
        if self.ffc is not None and x_g is not None:
            out_xg += self.ffc(x_g)

        if self.bn_l is not None and out_xl is not 0:
            out_xl = self.bn_l(out_xl)
        if self.bn_g is not None and out_xg is not 0:
            out_xg = self.bn_g(out_xg)

        out_xl = self.relu(out_xl + id_l if id_l is not None else out_xl)
        out_xg = self.relu(out_xg + id_g if id_g is not None and out_xg is not 0 else out_xg)

        return out_xl, out_xg


class LaMaGenerator(nn.Module):
    """LaMa Generator for Image Inpainting"""

    def __init__(self, input_channels=4, base_dim=64):
        super().__init__()

        # Encoder
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(input_channels, base_dim, kernel_size=7, padding=3),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True)
        )

        self.encoder_2 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_dim * 2),
            nn.ReLU(inplace=True)
        )

        self.encoder_3 = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_dim * 4),
            nn.ReLU(inplace=True)
        )

        # FFC Residual Blocks
        self.ffc_blocks = nn.ModuleList([
            FFCResNetBlock(base_dim * 4, ratio_gin=0.5, ratio_gout=0.5)
            for _ in range(9)
        ])

        # Decoder
        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(base_dim * 4, base_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_dim * 2),
            nn.ReLU(inplace=True)
        )

        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(base_dim * 2, base_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True)
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(base_dim, 3, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        # x shape: (batch, 4, H, W) - 3 channels RGB + 1 channel mask

        # Encode
        e1 = self.encoder_1(x)
        e2 = self.encoder_2(e1)
        e3 = self.encoder_3(e2)

        # FFC blocks
        x_l, x_g = e3, None
        for ffc_block in self.ffc_blocks:
            x_l, x_g = ffc_block((x_l, x_g))

        # Combine local and global
        if x_g is not None:
            out = x_l + x_g
        else:
            out = x_l

        # Decode
        d1 = self.decoder_1(out)
        d2 = self.decoder_2(d1)
        output = self.output_layer(d2)

        return output


class LaMaInpainting:
    """Wrapper class for LaMa Inpainting"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = LaMaGenerator().to(device)
        self.model.eval()

    def load_pretrained(self, checkpoint_path):
        """Load pretrained weights"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Loaded pretrained model from {checkpoint_path}")

    def preprocess(self, image: np.ndarray, mask: np.ndarray) -> torch.Tensor:
        """
        Preprocess image and mask
        Args:
            image: RGB image (H, W, 3) in range [0, 255]
            mask: Binary mask (H, W) where 1 indicates region to inpaint
        Returns:
            Tensor of shape (1, 4, H, W)
        """
        # Normalize image to [-1, 1]
        image = image.astype(np.float32) / 127.5 - 1.0

        # Normalize mask to [0, 1]
        mask = mask.astype(np.float32)
        if mask.max() > 1:
            mask = mask / 255.0

        # Convert to tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

        # Concatenate image and mask
        input_tensor = torch.cat([image_tensor, mask_tensor], dim=1)

        return input_tensor.to(self.device)

    def postprocess(self, output: torch.Tensor) -> np.ndarray:
        """
        Postprocess output tensor to image
        Args:
            output: Tensor of shape (1, 3, H, W) in range [-1, 1]
        Returns:
            RGB image (H, W, 3) in range [0, 255]
        """
        output = output.squeeze(0).cpu().detach().numpy()
        output = (output + 1.0) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        output = output.transpose(1, 2, 0)
        return output

    @torch.no_grad()
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Perform inpainting
        Args:
            image: RGB image (H, W, 3)
            mask: Binary mask (H, W)
        Returns:
            Inpainted image (H, W, 3)
        """
        input_tensor = self.preprocess(image, mask)
        output = self.model(input_tensor)
        result = self.postprocess(output)

        # Blend original and inpainted regions
        mask_3d = np.stack([mask] * 3, axis=-1)
        if mask_3d.max() > 1:
            mask_3d = mask_3d / 255.0

        result = (result * mask_3d + image * (1 - mask_3d)).astype(np.uint8)

        return result

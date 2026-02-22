"""
Training script for LaMa Inpainting Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2
from PIL import Image
import os
from tqdm import tqdm
import argparse
from pathlib import Path

from lama_model import LaMaGenerator
from mask_generator import MaskGenerator


class InpaintingDataset(Dataset):
    """Dataset for inpainting training"""

    def __init__(self, image_dir, transform=None, mask_generator=None):
        self.image_dir = Path(image_dir)
        self.image_paths = list(self.image_dir.glob('*.jpg')) + \
                          list(self.image_dir.glob('*.png')) + \
                          list(self.image_dir.glob('*.jpeg'))

        self.transform = transform
        self.mask_generator = mask_generator or MaskGenerator(256, 256)

        print(f"Found {len(self.image_paths)} images in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Generate random mask
        mask = self.mask_generator.generate_random_mask("mixed")
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        # Create masked image
        masked_image = image * (1 - mask)

        # Concatenate masked image and mask
        input_tensor = torch.cat([masked_image, mask], dim=0)

        return input_tensor, image, mask


class PerceptualLoss(nn.Module):
    """Perceptual Loss using VGG16 features"""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.layers = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16], # relu3_3
        ])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        loss = 0
        for layer in self.layers:
            x = layer(x)
            y = layer(y)
            loss += nn.functional.l1_loss(x, y)
        return loss


class InpaintingLoss(nn.Module):
    """Combined loss for inpainting"""

    def __init__(self, l1_weight=1.0, perceptual_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight

        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()

    def forward(self, pred, target, mask):
        # L1 loss (whole image)
        l1 = self.l1_loss(pred, target)

        # Perceptual loss
        perceptual = self.perceptual_loss(pred, target)

        # Total loss
        total_loss = self.l1_weight * l1 + self.perceptual_weight * perceptual

        return total_loss, {
            'l1': l1.item(),
            'perceptual': perceptual.item(),
            'total': total_loss.item()
        }


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, use_amp=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    loss_dict = {'l1': 0, 'perceptual': 0, 'total': 0}

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    scaler = torch.amp.GradScaler(device='cuda', enabled=(use_amp and device.type == 'cuda'))
    for batch_idx, (inputs, targets, masks) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        # Forward
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', enabled=(use_amp and device.type == 'cuda')):
            outputs = model(inputs)

            # Loss
            loss, losses = criterion(outputs, targets, masks)

        # Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Statistics
        total_loss += loss.item()
        for key in loss_dict:
            loss_dict[key] += losses[key]

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'l1': f"{losses['l1']:.4f}",
            'perc': f"{losses['perceptual']:.4f}"
        })

    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_loss_dict = {k: v / len(dataloader) for k, v in loss_dict.items()}

    return avg_loss, avg_loss_dict


def validate(model, dataloader, criterion, device, use_amp=False):
    """Validate the model"""
    model.eval()
    total_loss = 0
    loss_dict = {'l1': 0, 'perceptual': 0, 'total': 0}

    with torch.no_grad():
        for inputs, targets, masks in tqdm(dataloader, desc='Validation'):
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            with torch.amp.autocast(device_type='cuda', enabled=(use_amp and device.type == 'cuda')):
                outputs = model(inputs)
                loss, losses = criterion(outputs, targets, masks)

            total_loss += loss.item()
            for key in loss_dict:
                loss_dict[key] += losses[key]

    avg_loss = total_loss / len(dataloader)
    avg_loss_dict = {k: v / len(dataloader) for k, v in loss_dict.items()}

    return avg_loss, avg_loss_dict


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"✓ Saved checkpoint to {path}")


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Datasets
    print("Loading datasets...")
    mask_gen = MaskGenerator(256, 256)
    train_dataset = InpaintingDataset(args.train_dir, transform=transform, mask_generator=mask_gen)
    val_dataset = InpaintingDataset(args.val_dir, transform=transform, mask_generator=mask_gen)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0)
    )

    # Model
    print("Initializing model...")
    model = LaMaGenerator(input_channels=4, base_dim=64).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Loss and optimizer
    criterion = InpaintingLoss(l1_weight=1.0, perceptual_weight=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)

        # Train
        train_loss, train_losses = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, use_amp=args.amp
        )

        # Validate
        val_loss, val_losses = validate(model, val_loader, criterion, device, use_amp=args.amp)

        # Learning rate scheduling
        scheduler.step()

        # Print statistics
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} (L1: {train_losses['l1']:.4f}, Perc: {train_losses['perceptual']:.4f})")
        print(f"  Val Loss:   {val_loss:.4f} (L1: {val_losses['l1']:.4f}, Perc: {val_losses['perceptual']:.4f})")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / "best_model.pth"
            save_checkpoint(model, optimizer, epoch, val_loss, best_path)
            print(f"  ★ New best model! Val loss: {val_loss:.4f}")

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LaMa Inpainting Model')

    # Data
    parser.add_argument('--train_dir', type=str, default='6/data/train',
                       help='Training data directory')
    parser.add_argument('--val_dir', type=str, default='6/data/val',
                       help='Validation data directory')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    parser.add_argument('--amp', action='store_true',
                       help='Enable mixed precision on CUDA to reduce memory usage')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='6/checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')

    args = parser.parse_args()

    # Check if data directories exist
    if not Path(args.train_dir).exists():
        print(f"⚠ Warning: Training directory not found: {args.train_dir}")
        print("Please prepare your dataset first!")
        print("\nExample structure:")
        print("6/data/")
        print("  ├── train/")
        print("  │   ├── image1.jpg")
        print("  │   ├── image2.jpg")
        print("  │   └── ...")
        print("  └── val/")
        print("      ├── image1.jpg")
        print("      └── ...")
        exit(1)

    main(args)

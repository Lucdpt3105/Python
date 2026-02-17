# 🚀 Smart Content Filler - Setup Guide

## 📋 Yêu cầu hệ thống

- Python 3.8+
- CUDA (tùy chọn, để chạy trên GPU)
- RAM: >= 8GB
- Disk: >= 2GB trống

## 🔧 Cài đặt

### Bước 1: Cài đặt dependencies

```bash
cd 6
pip install -r requirements.txt
```

### Bước 2: Tải pretrained model (Tùy chọn)

Để có kết quả tốt nhất, bạn nên tải pretrained model của LaMa:

**Option 1: Tự download (Recommended)**

```bash
# Tạo thư mục checkpoints
mkdir -p checkpoints

# Download từ nguồn chính thức
# Link: https://github.com/advimman/lama/releases
# Hoặc từ Google Drive/Hugging Face

# Sau khi download, đặt file vào:
# 6/checkpoints/lama_model.pth
```

**Option 2: Sử dụng model có sẵn từ torch hub**

```python
import torch
# Code sẽ tự động tải nếu không tìm thấy local checkpoint
```

### Bước 3: Chạy ứng dụng

```bash
python app.py
```

Truy cập: http://localhost:7860

## 📚 Cấu trúc thư mục

```
6/
├── app.py                  # Gradio UI chính
├── lama_model.py          # LaMa model implementation
├── mask_generator.py      # Random mask generator
├── requirements.txt       # Dependencies
├── SETUP.md              # Hướng dẫn này
├── train.py              # Training script (nếu cần)
└── checkpoints/          # Pretrained models
    └── lama_model.pth
```

## 🎯 Sử dụng

### 1. Xóa vật thể (Object Removal)

1. Upload ảnh
2. Dùng brush tool vẽ lên vùng cần xóa (màu đỏ)
3. Click "🪄 Magic Inpaint"
4. Xem kết quả!

### 2. Phục hồi khuôn mặt (Facial Restoration)

1. Upload ảnh khuôn mặt
2. Chọn "Facial Features" trong Mask Type
3. Hoặc tự vẽ lên các vùng bị hỏng (mắt, mũi, miệng)
4. Click "🪄 Magic Inpaint"

### 3. Test với random masks

```bash
# Generate sample masks
python mask_generator.py
```

Sẽ tạo file `sample_masks.png` với các loại mask khác nhau.

## 🔬 Kiến trúc Model

### LaMa (Large Mask Inpainting)

**Đặc điểm:**
- ✅ Fast Fourier Convolutions (FFCs)
- ✅ Xử lý được large masks (>50% area)
- ✅ High resolution support
- ✅ Context-aware inpainting

**Cấu trúc:**
```
Input (RGB + Mask) → Encoder → FFC Blocks → Decoder → Output
    |                                              |
    └──────────────── Skip Connections ──────────┘
```

### Loss Functions

1. **L1 Loss**: Pixel-wise reconstruction
2. **Perceptual Loss**: Feature-level similarity (VGG16)
3. **Adversarial Loss**: GAN training (optional)

## 🎓 Training (Nâng cao)

Nếu bạn muốn train model từ đầu:

```bash
# Chuẩn bị dataset (CelebA-HQ hoặc Places2)
# Đặt vào thư mục: 6/data/

# Run training
python train.py --config configs/lama_config.yaml
```

## ⚠️ Troubleshooting

### Lỗi: CUDA out of memory
```python
# Trong lama_model.py, giảm batch size hoặc image size
# Hoặc chạy trên CPU:
device = 'cpu'
```

### Lỗi: Module not found
```bash
pip install --upgrade -r requirements.txt
```

### Model không load được
```bash
# Check path
ls checkpoints/lama_model.pth

# Nếu không có, app sẽ dùng CV2 fallback (kết quả kém hơn)
```

## 📊 Performance

| Method | Speed | Quality | Large Mask Support |
|--------|-------|---------|-------------------|
| CV2 Inpainting | ⚡⚡⚡ | ⭐⭐ | ❌ |
| LaMa (CPU) | ⚡⚡ | ⭐⭐⭐⭐⭐ | ✅ |
| LaMa (GPU) | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ✅ |

## 🔗 Resources

- [LaMa Paper](https://arxiv.org/abs/2109.07161)
- [Original Implementation](https://github.com/advimman/lama)
- [PapersWithCode](https://paperswithcode.com/paper/resolution-robust-large-mask-inpainting-with)

## 💡 Tips

1. **Cho kết quả tốt:**
   - Vẽ mask chính xác (không quá to, không quá nhỏ)
   - Sử dụng ảnh có độ phân giải tốt
   - Đảm bảo vùng xung quanh mask có đủ context

2. **Tối ưu tốc độ:**
   - Resize ảnh xuống 512x512 trước khi process
   - Sử dụng GPU nếu có
   - Batch processing cho nhiều ảnh

3. **Training custom model:**
   - Cần dataset lớn (>10K images)
   - Train ít nhất 100K iterations
   - Sử dụng data augmentation

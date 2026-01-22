# Bài Tập 2: Play Sound with OpenAL 3D

## 📋 Mô tả bài tập

Bài tập này gồm 2 yêu cầu chính về âm thanh 3D sử dụng OpenAL:

### Yêu cầu 1: Phát âm thanh ở nhiều vị trí
Play single sound at several positions (x, y, z), ví dụ:
- (0, 0, 0)
- (-10, 0, 0)
- (5, 0, 0)
- (0, 5, 0)
- (0, -10, 0)
- (0, 0, -10)
- (0, 0, 10)

### Yêu cầu 2: Mô phỏng không gian ảo 3D
Simulate 1 virtual space (ví dụ: phòng, khu vực rừng...) với nhiều nguồn âm thanh ở các vị trí khác nhau, mỗi nguồn phát âm thanh riêng và có thể di chuyển.

---

## 📁 Cấu trúc thư mục

```
BaiTap2/
├── yeu_cau_1_sound_positions.py    # Giải quyết yêu cầu 1
├── yeu_cau_2_virtual_space.py      # Giải quyết yêu cầu 2
└── README.md                        # File hướng dẫn này
```

---

## 🚀 Cách chạy

### Yêu cầu 1:
```bash
cd Examples/BaiTap2
python yeu_cau_1_sound_positions.py
```

### Yêu cầu 2:
```bash
cd Examples/BaiTap2
python yeu_cau_2_virtual_space.py
```

---

## ⚙️ Yêu cầu hệ thống

1. **Python 3.x**
2. **OpenAL32.dll** - Đảm bảo file này nằm trong thư mục gốc của project
3. **File âm thanh .wav** - Các script sử dụng file `tone5.wav` từ thư mục `Examples/3D_Audio/`

---

## 📝 Chi tiết từng file

### 1. yeu_cau_1_sound_positions.py

**Chức năng:**
- Phát cùng một âm thanh ở nhiều vị trí khác nhau trong không gian 3D
- Listener (người nghe) đứng ở vị trí (0, 0, 0)
- Âm thanh được phát lần lượt ở 9 vị trí khác nhau
- Tính toán và hiển thị khoảng cách từ listener đến từng vị trí

**Các tính năng:**
- Rolloff factor: 0.5 (độ suy giảm âm thanh theo khoảng cách)
- Thời gian phát: 2 giây mỗi vị trí
- Tự động dừng và dọn dẹp tài nguyên

### 2. yeu_cau_2_virtual_space.py

**Chức năng:**
- Mô phỏng không gian phòng 3D (20x20x10 mét)
- 5 nguồn âm thanh khác nhau:
  - **Bird 1**: Di chuyển từ trái sang phải
  - **Bird 2**: Di chuyển từ phải sang trái
  - **Wind**: Di chuyển từ trên xuống dưới
  - **Animal**: Di chuyển từ dưới lên trên
  - **Circular Sound**: Xoay tròn quanh listener

**Các tính năng:**
- Mỗi nguồn âm có pitch và gain khác nhau để tạo sự đa dạng
- Âm thanh tự động loop (phát lặp lại)
- Nguồn âm tự động đảo chiều khi chạm tường
- Hiển thị vị trí và khoảng cách theo thời gian thực
- Thời gian mô phỏng mặc định: 30 giây (có thể thay đổi)

---

## 🎮 Cách sử dụng

### Thay đổi file âm thanh:

Trong cả 2 file, tìm dòng:
```python
self.sound = LoadSound('../../Examples/3D_Audio/tone5.wav')
```

Thay đổi đường dẫn đến file .wav của bạn:
```python
self.sound = LoadSound('path/to/your/sound.wav')
```

### Điều chỉnh thời gian mô phỏng (Yêu cầu 2):

Trong file `yeu_cau_2_virtual_space.py`, tìm dòng:
```python
space.simulate_space(duration=30)
```

Thay đổi `duration` theo ý muốn (đơn vị: giây).

### Thêm nguồn âm thanh mới (Yêu cầu 2):

Trong hàm `create_sound_sources()`, thêm vào `source_configs`:
```python
{
    'name': 'Tên nguồn âm',
    'start_pos': (x, y, z),
    'direction': (dx, dy, dz),
    'speed': 0.3,
    'gain': 0.8,
    'pitch': 1.0
}
```

---

## 🎯 Kết quả mong đợi

### Yêu cầu 1:
- Nghe được âm thanh di chuyển qua các vị trí khác nhau
- Âm thanh to/nhỏ tùy theo khoảng cách
- Âm thanh từ trái/phải/trước/sau rõ ràng

### Yêu cầu 2:
- Nghe được nhiều nguồn âm thanh cùng lúc
- Mỗi nguồn âm có đặc điểm riêng (cao/thấp, to/nhỏ)
- Âm thanh thay đổi khi nguồn di chuyển
- Hiệu ứng 3D rõ ràng (trái/phải, gần/xa, trên/dưới)

---

## 🔧 Troubleshooting

### Lỗi: "Không tìm thấy file âm thanh"
- Kiểm tra đường dẫn đến file .wav
- Đảm bảo file tồn tại trong thư mục chỉ định

### Lỗi: "Cannot load OpenAL32.dll"
- Kiểm tra file OpenAL32.dll có trong thư mục gốc
- Đảm bảo file dll tương thích với kiến trúc hệ thống (x32/x64)

### Không nghe thấy hiệu ứng 3D
- Sử dụng tai nghe để nghe rõ hơn
- Kiểm tra cài đặt âm thanh hệ thống
- Tăng giá trị `rolloff` để tăng hiệu ứng khoảng cách

### Âm thanh bị giật/lag
- Giảm số lượng nguồn âm thanh
- Tăng thời gian sleep trong loop
- Kiểm tra tài nguyên CPU

---

## 📚 Tài liệu tham khảo

- [OpenAL Documentation](https://www.openal.org/documentation/)
- [OpenAL Soft](https://openal-soft.org/)
- [3D Audio Theory](https://en.wikipedia.org/wiki/3D_audio_effect)

---

## ✅ Checklist hoàn thành

- [x] Yêu cầu 1: Phát âm thanh ở nhiều vị trí
- [x] Yêu cầu 2: Mô phỏng không gian 3D với nhiều nguồn âm
- [x] Code có comment chi tiết bằng tiếng Việt
- [x] Hướng dẫn sử dụng đầy đủ
- [x] Xử lý lỗi và cleanup tài nguyên

---

**Chúc bạn thành công với bài tập! 🎵🎧**

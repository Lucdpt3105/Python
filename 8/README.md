# Camera Background Replacement (Project 8)

Project này dùng webcam hoặc video file để tách người ra khỏi nền và thay bằng ảnh nền khác.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy với webcam

```bash
python src/main.py --source camera --camera 0 --background backgrounds/default.jpg
```

## Chạy với video file (hữu ích khi WSL không truy cập được camera)

```bash
python src/main.py --source path/to/input.mp4 --background backgrounds/default.jpg
```

## Phím tắt

- `q`: thoát chương trình
- `b`: bật/tắt làm mờ nền (khi không dùng ảnh nền)

## Lưu ý WSL/Linux

- Nếu báo lỗi `Camera index out of range`, môi trường hiện tại không thấy camera (`/dev/video*`).
- Cách xử lý:
  - Chạy project bằng Python trên Windows (không qua WSL), hoặc
  - Dùng `--source <video-file>` để chạy bằng file video.

## Gợi ý

- Đặt thêm ảnh nền vào thư mục `backgrounds/`.
- Nếu camera lag, giảm độ phân giải trong code.

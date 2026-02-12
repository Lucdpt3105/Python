import cv2
import pytesseract
import numpy as np
import os


def remove_border(img):
    """Loại bỏ viền trắng/đen dư thừa xung quanh ảnh chữ ký"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Nhị phân hóa để tách nét mực khỏi nền (giảm threshold để bắt mực nhạt hơn)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Tìm tất cả các điểm ảnh có nét mực
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Thêm padding nhỏ để chữ ký không bị sát mép quá
        p = 10
        x_start = max(0, x - p)
        y_start = max(0, y - p)
        x_end = min(img.shape[1], x + w + p)
        y_end = min(img.shape[0], y + h + p)
        return img[y_start:y_end, x_start:x_end]
    return img


# 1. Cấu hình
input_file = 'Picture1.png'
output_dir = 'KetQua_ChuKy'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. Đọc ảnh
img = cv2.imread(input_file)
if img is None:
    print(f"Lỗi: Không tìm thấy file {input_file}")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h_img, w_img = gray.shape

# 3. Phát hiện đường kẻ (Giữ nguyên logic của bác vì nó đang chạy tốt)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w_img // 4, 1))
detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

horizontal_lines = []
for y in range(h_img):
    if np.sum(detect_horizontal[y, :]) > w_img * 0.5 * 255:
        horizontal_lines.append(y)

rows_y = []
if horizontal_lines:
    current_y = horizontal_lines[0]
    for y in horizontal_lines[1:]:
        if y - current_y > 10:  # Tăng khoảng cách gộp dòng
            rows_y.append(current_y)
            current_y = y
    rows_y.append(current_y)

# Tìm đường dọc giữa
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h_img // 4))
detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
vertical_lines = [x for x in range(w_img) if np.sum(detect_vertical[:, x]) > h_img * 0.5 * 255]
mid_x = min(vertical_lines, key=lambda x: abs(x - w_img // 2)) if vertical_lines else w_img // 2

print(f"Tìm thấy {len(rows_y) - 1} dòng. Bắt đầu xử lý...")

# 4. Vòng lặp xử lý
# Chỉnh i bắt đầu từ 2 để bỏ qua hàng "Họ Tên | Chữ ký"
for i in range(2, len(rows_y)):
    y_start = rows_y[i - 1]
    y_end = rows_y[i]

    # Thụt vào 10px để đảm bảo không dính đường kẻ bảng
    offset = 1

    # Cắt ô Tên (Trái)
    roi_name = gray[y_start + offset: y_end - offset, offset: mid_x - offset]
    # Cắt ô Chữ ký (Phải)
    roi_sig = img[y_start + offset: y_end - offset, mid_x + offset: w_img - offset]

    # 5. OCR đọc tên - Tiền xử lý tốt hơn
    # Resize lớn hơn để OCR chính xác hơn
    roi_name_large = cv2.resize(roi_name, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Tăng độ tương phản
    roi_name_large = cv2.threshold(roi_name_large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    name_raw = pytesseract.image_to_string(roi_name_large, lang='vie', config='--psm 7')
    name_clean = "".join([c for c in name_raw.strip() if c.isalnum() or c in (' ', '_')]).strip()

    # Kiểm tra nếu OCR ra tên "Chữ ký" hoặc "Họ Tên" thì bỏ qua
    if not name_clean or name_clean in ["Họ Tên", "Chữ ký"]:
        continue

    # 6. Hậu xử lý chữ ký (Gọt trắng)
    roi_sig_final = remove_border(roi_sig)

    # 7. Lưu file (Xử lý tên file tiếng Việt)
    save_path = os.path.join(output_dir, f"{name_clean}.png")
    is_success, im_buf_arr = cv2.imencode(".png", roi_sig_final)
    if is_success:
        im_buf_arr.tofile(save_path)
        print(f"✅ Đã lưu: {save_path}")

print(f"\n--- Hoàn tất! Kết quả tại: {output_dir} ---")
import cv2
import pytesseract
import numpy as np
import os


def add_border(img, vien1=None, vien2=None):
    """Thêm viền bóng (shadow) từ vien1 (dưới+phải) và vien2 (trên+trái)."""
    if vien1 is None or vien2 is None:
        return img

    h, w = img.shape[:2]
    is_gray = (len(img.shape) == 2)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if is_gray else img.copy()

    bh, bw = vien1.shape[:2]
    frame = np.full((h, w, 3), 255, dtype=np.uint8)

    mid = bh // 2
    strip_top = vien2[:bh, mid, :]       # gradient dọc cho cạnh trên
    strip_left = vien2[mid, :bw, :]      # gradient ngang cho cạnh trái
    strip_bot = vien1[:bh, mid, :]       # gradient dọc cho cạnh dưới
    strip_right = vien1[mid, :bw, :]     # gradient ngang cho cạnh phải

    # Cạnh trên + dưới
    if h >= bh:
        for r in range(bh):
            frame[r, :, :] = strip_top[r]
            frame[h - bh + r, :, :] = np.minimum(frame[h - bh + r, :, :], strip_bot[r])

    # Cạnh trái + phải
    if w >= bw:
        for c in range(bw):
            frame[:, c, :] = np.minimum(frame[:, c, :], strip_left[c])
            frame[:, w - bw + c, :] = np.minimum(frame[:, w - bw + c, :], strip_right[c])

    # 4 góc
    ch, cw = min(bh, h), min(bw, w)
    v2_flip, v1_flip = cv2.flip(vien2, 1), cv2.flip(vien1, 1)

    corners = [
        (slice(None, ch),  slice(None, cw),   vien2[:ch, :cw]),
        (slice(None, ch),  slice(w-cw, None), v2_flip[:ch, bw-cw:]),
        (slice(h-ch, None), slice(None, cw),  v1_flip[bh-ch:, :cw]),
        (slice(h-ch, None), slice(w-cw, None), vien1[bh-ch:, bw-cw:]),
    ]
    for rs, cs, patch in corners:
        frame[rs, cs] = np.minimum(frame[rs, cs], patch)

    result = np.minimum(img_color, frame)
    return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if is_gray else result


def remove_border(img):
    """Cắt viền thừa xung quanh nội dung."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=1)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)
    p = 2
    return img[max(0, y-p):min(img.shape[0], y+h+p),
               max(0, x-p):min(img.shape[1], x+w+p)]


def detect_lines(binary, kernel, img_len, threshold_ratio=0.5):
    """Phát hiện đường kẻ ngang/dọc, trả về danh sách tọa độ đã gom nhóm."""
    detected = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    raw = [i for i in range(detected.shape[0] if kernel.shape[1] > 1 else detected.shape[1])
           if np.sum(detected[i, :] if kernel.shape[1] > 1 else detected[:, i]) > img_len * threshold_ratio * 255]

    grouped = []
    if raw:
        current = raw[0]
        for val in raw[1:]:
            if val - current > 10:
                grouped.append(current)
            current = val
        grouped.append(current)
    return raw, grouped


def ocr_name(roi_gray):
    """OCR vùng tên, trả về tên đã clean hoặc None."""
    _, name_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Xóa đường dọc
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, roi_gray.shape[0] // 2))
    vlines = cv2.morphologyEx(name_thresh, cv2.MORPH_OPEN, vk, iterations=2)
    roi_gray[vlines > 0] = 255

    # Phóng to + threshold
    large = cv2.resize(roi_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    large = cv2.threshold(large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    raw = pytesseract.image_to_string(large, lang='vie', config='--psm 7')
    clean = "".join(c for c in raw.strip() if c.isalnum() or c in (' ', '_')).strip()

    if not clean or clean in ["Họ Tên", "Chữ ký"]:
        return None
    return clean


# === MAIN ===
input_file = 'Picture1.jpg'
output_dir = 'KetQua_ChuKy'
os.makedirs(output_dir, exist_ok=True)

img = cv2.imread(input_file)
if img is None:
    print("Không tìm thấy ảnh.")
    exit()

vien1 = cv2.imread('vien1.jpg')
vien2 = cv2.imread('vien2.jpg')
if vien1 is None or vien2 is None:
    print("Không tìm thấy vien1.jpg hoặc vien2.jpg. Sẽ không thêm viền.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h_img, w_img = gray.shape
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Phát hiện đường ngang
h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w_img // 4, 1))
_, rows_y = detect_lines(thresh, h_kernel, w_img)

# Phát hiện đường dọc
v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h_img // 4))
vertical_lines, _ = detect_lines(thresh, v_kernel, h_img)

mid_x = min(vertical_lines, key=lambda x: abs(x - w_img // 2)) if vertical_lines else w_img // 2
right_border = max((x for x in vertical_lines if x > mid_x), default=w_img)

print(f"Tìm thấy {len(rows_y)-1} dòng. Bắt đầu xử lý...")

off_v, off_l = 10, 10

for i in range(2, len(rows_y)):
    y_start, y_end = rows_y[i-1], rows_y[i]

    roi_name = gray[y_start + off_v : y_end - off_v, 10 : mid_x - 5]
    roi_sig = img[y_start + off_v : y_end - off_v, mid_x + off_l : right_border - 5]

    name = ocr_name(roi_name)
    if name is None:
        continue

    roi_sig_final = add_border(remove_border(roi_sig), vien1=vien1, vien2=vien2)

    save_path = os.path.join(output_dir, f"{name}.png")
    success, buf = cv2.imencode(".png", roi_sig_final)
    if success:
        buf.tofile(save_path)
        print(f"Đã lưu: {save_path}")

print("\nHoàn tất!")

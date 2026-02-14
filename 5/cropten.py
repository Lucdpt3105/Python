import cv2
import pytesseract
import numpy as np
import os


# ===============================
# AUTO CROP SIGNATURE (NO WHITE BORDER)
# ===============================
def remove_border(img):

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Blur nhẹ để bắt nét mực tốt hơn
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    # OTSU threshold (auto)
    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # nối nét chữ ký
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    coords = cv2.findNonZero(thresh)

    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)

    # padding rất nhỏ cho đẹp
    p = 2

    x_start = max(0, x - p)
    y_start = max(0, y - p)
    x_end = min(img.shape[1], x + w + p)
    y_end = min(img.shape[0], y + h + p)

    return img[y_start:y_end, x_start:x_end]


# ===============================
# CONFIG
# ===============================
input_file = 'Picture1.png'
output_dir = 'KetQua_ChuKy'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ===============================
# LOAD IMAGE
# ===============================
img = cv2.imread(input_file)

if img is None:
    print("Không tìm thấy ảnh.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h_img, w_img = gray.shape


# ===============================
# DETECT TABLE LINES
# ===============================
_, thresh = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Horizontal lines
horizontal_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT, (w_img // 4, 1))
detect_horizontal = cv2.morphologyEx(
    thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

horizontal_lines = []

for y in range(h_img):
    if np.sum(detect_horizontal[y, :]) > w_img * 0.5 * 255:
        horizontal_lines.append(y)

rows_y = []

if horizontal_lines:
    current_y = horizontal_lines[0]

    for y in horizontal_lines[1:]:
        if y - current_y > 10:
            rows_y.append(current_y)
            current_y = y

    rows_y.append(current_y)


# Vertical middle line
vertical_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT, (1, h_img // 4))

detect_vertical = cv2.morphologyEx(
    thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

vertical_lines = [x for x in range(w_img)
                  if np.sum(detect_vertical[:, x]) > h_img * 0.5 * 255]

mid_x = min(vertical_lines,
            key=lambda x: abs(x - w_img // 2)) if vertical_lines else w_img // 2

# tìm đường dọc bên phải gần mép ảnh nhất
right_border = w_img

if vertical_lines:
    right_candidates = [x for x in vertical_lines if x > mid_x]
    if right_candidates:
        right_border = max(right_candidates)


print(f"Tìm thấy {len(rows_y)-1} dòng. Bắt đầu xử lý...")


# ===============================
# MAIN LOOP
# ===============================
for i in range(2, len(rows_y)):

    y_start = rows_y[i-1]
    y_end = rows_y[i]

    offset_v = 10
    offset_left = 10

    # ROI NAME
    roi_name = gray[
        y_start + offset_v : y_end - offset_v,
        10 : mid_x - 5
    ]
    roi_sig = img[
    y_start + offset_v : y_end - offset_v,
    mid_x + offset_left : right_border - 5
    ]
    
    # ===============================
    # OCR NAME
    # ===============================
    _, name_thresh = cv2.threshold(
        roi_name, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    vert_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, roi_name.shape[0] // 2)
    )

    vert_lines = cv2.morphologyEx(
        name_thresh, cv2.MORPH_OPEN,
        vert_kernel, iterations=2)

    roi_name[vert_lines > 0] = 255

    roi_name_large = cv2.resize(
        roi_name, None,
        fx=2, fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    roi_name_large = cv2.threshold(
        roi_name_large,
        0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    name_raw = pytesseract.image_to_string(
        roi_name_large,
        lang='vie',
        config='--psm 7'
    )

    name_clean = "".join([
        c for c in name_raw.strip()
        if c.isalnum() or c in (' ', '_')
    ]).strip()

    if not name_clean or name_clean in ["Họ Tên", "Chữ ký"]:
        continue


    # ===============================
    # FINAL SIGNATURE CROP
    # ===============================
    roi_sig_final = remove_border(roi_sig)


    # ===============================
    # SAVE FILE
    # ===============================
    save_path = os.path.join(
        output_dir,
        f"{name_clean}.png"
    )

    success, buf = cv2.imencode(".png", roi_sig_final)

    if success:
        buf.tofile(save_path)
        print(f"Đã lưu: {save_path}")


print("\nHoàn tất!")

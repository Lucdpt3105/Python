import cv2
import numpy as np
import pytesseract
import os

# Tạo thư mục output
output_dir = 'KetQua_ChuKy'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

img = cv2.imread('Picture1.jpg')
temp1 = cv2.imread('kernell1.jpg')
temp2 = cv2.imread('kernell2.jpg')  

if img is None:
    print("Không tìm thấy ảnh Picture1.jpg")
    exit()
if temp1 is None or temp2 is None:
    print("Không tìm thấy template kernell1.jpg hoặc kernell2.jpg")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, ksize=3)
ret, thresh = cv2.threshold(blur, 15, 255, 0)

gray1 = cv2.cvtColor(temp1, cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(gray1, 10, 255, 0)

gray2 = cv2.cvtColor(temp2, cv2.COLOR_BGR2GRAY)
ret, thresh2 = cv2.threshold(gray2, 10, 255, 0)

h1, w1 = temp1.shape[:2]  # Kích thước kernel 1 (góc trên-trái)
h2, w2 = temp2.shape[:2]  # Kích thước kernel 2 (góc dưới-phải)
img_h, img_w = img.shape[:2]

def add_corner_borders(image, corner_tl, corner_br):
    """Thêm viền kernel1 vào góc trên-trái và kernel2 vào góc dưới-phải"""
    result = image.copy()
    h, w = result.shape[:2]
    h_tl, w_tl = corner_tl.shape[:2]
    h_br, w_br = corner_br.shape[:2]
    
    # Ghép kernel1 vào góc trên-trái
    if h >= h_tl and w >= w_tl:
        result[0:h_tl, 0:w_tl] = corner_tl
    
    # Ghép kernel2 vào góc dưới-phải
    if h >= h_br and w >= w_br:
        result[h-h_br:h, w-w_br:w] = corner_br
    
    return result

res1 = cv2.matchTemplate(thresh, thresh1, cv2.TM_CCOEFF_NORMED)
res2 = cv2.matchTemplate(thresh, thresh2, cv2.TM_CCOEFF_NORMED)
threshold1 = 0.8  # Hạ threshold để tìm được nhiều hơn
threshold2 = 0.8

loc1 = np.where(res1 >= threshold1)
loc2 = np.where(res2 >= threshold2)

# Loại bỏ các điểm trùng lặp (gần nhau < 10 pixel)
def remove_duplicates(points, min_dist=10):
    if len(points) == 0:
        return []
    result = [points[0]]
    for p in points[1:]:
        is_dup = False
        for r in result:
            if abs(p[0] - r[0]) < min_dist and abs(p[1] - r[1]) < min_dist:
                is_dup = True
                break
        if not is_dup:
            result.append(p)
    return result

points1 = sorted(zip(loc1[0], loc1[1]), key=lambda p: (p[0], p[1]))
points2 = sorted(zip(loc2[0], loc2[1]), key=lambda p: (p[0], p[1]))

points1 = remove_duplicates(points1)
points2 = remove_duplicates(points2)

print(f"Tìm thấy {len(points1)} vị trí kernel1, {len(points2)} vị trí kernel2")

def find_bottom_right(tl_y, tl_x, points_br, h_kernel, w_kernel):
    """Tìm điểm kernel2 gần nhất nằm bên phải-dưới của kernel1"""
    best = None
    best_dist = float('inf')
    for br_y, br_x in points_br:
        # kernel2 phải nằm dưới và bên phải của kernel1
        # Cho phép dung sai y (cùng hàng)
        if br_y >= tl_y - 5 and br_x > tl_x:
            dist = abs(br_y - tl_y) + (br_x - tl_x)
            if dist < best_dist:
                best_dist = dist
                best = (br_y + h_kernel, br_x + w_kernel)
    return best

# Ghép cặp kernel1 (góc trên-trái) với kernel2 (góc dưới-phải)
cells = []
used_br = []

for tl_y, tl_x in points1:
    br = find_bottom_right(tl_y, tl_x, points2, h2, w2)
    if br:
        # Kiểm tra xem br đã dùng chưa
        is_used = False
        for used in used_br:
            if abs(br[0] - used[0]) < 10 and abs(br[1] - used[1]) < 10:
                is_used = True
                break
        if not is_used:
            used_br.append(br)
            cells.append((tl_y, tl_x, br[0], br[1]))

# Sắp xếp cells theo y rồi x
cells = sorted(cells, key=lambda c: (c[0], c[1]))

print(f"Tìm thấy {len(cells)} ô hợp lệ")

fn = None
for i, (tly, tlx, bry, brx) in enumerate(cells):
    # Kiểm tra bounds
    tly = max(0, tly)
    tlx = max(0, tlx)
    bry = min(img_h, bry)
    brx = min(img_w, brx)
    
    if bry <= tly or brx <= tlx:
        continue
    
    # Crop bao gồm cả viền kernel1 và kernel2
    tmp = img[tly:bry, tlx:brx].copy()
    
    if tmp.size == 0:
        continue
    
    # Bỏ qua dòng tiêu đề (2 ô đầu tiên thường là "Họ Tên" và "Chữ ký")
    if i < 2:
        continue
    
    # Ô chẵn (0, 2, 4...) trong mỗi dòng là tên
    # Ô lẻ (1, 3, 5...) trong mỗi dòng là chữ ký
    if i % 2 == 0:
        # OCR để lấy tên
        name_raw = pytesseract.image_to_string(tmp, lang='vie', config='--psm 7')
        name_clean = "".join([
            c for c in name_raw.strip()
            if c.isalnum() or c in (' ', '_')
        ]).strip()
        
        if name_clean and name_clean not in ["Họ Tên", "Chữ ký", "Ho Tên", "Chu ký"]:
            fn = name_clean + ".png"
            print(f"Tên phát hiện: {name_clean}")
        else:
            fn = None
    else:
        # Lưu chữ ký với viền kernel1 và kernel2
        if fn:
            # Thêm viền góc từ kernel1 và kernel2
            tmp_with_border = add_corner_borders(tmp, temp1, temp2)
            save_path = os.path.join(output_dir, fn)
            success, buf = cv2.imencode(".png", tmp_with_border)
            if success:
                buf.tofile(save_path)
                print(f"Đã lưu: {save_path}")

print("\nHoàn tất!")
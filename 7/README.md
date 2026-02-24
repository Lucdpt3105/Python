# 4. Morphological Algorithms - ?ng d?ng kh? các d?i tu?ng d?c bi?t trong ?nh

## 1) Bài này dang nói v? gì?
`Morphological Algorithms` (toán t? hình thái h?c) là nhóm k? thu?t x? lý ?nh d?a trên **hình d?ng** c?a vùng sáng/t?i trong ?nh.

Ý tu?ng c?t lõi:
- Không nhìn ?nh nhu “màu d?p/x?u”, mà nhìn nhu các vùng d?i tu?ng (foreground) và n?n (background).
- Dùng m?t "khuôn" nh? g?i là `kernel` (hay `structuring element`) quét qua ?nh d? làm thay d?i hình d?ng vùng d?i tu?ng.

Ðây là k? thu?t r?t n?n t?ng trong x? lý ?nh truy?n th?ng (computer vision c? di?n), thu?ng dùng tru?c ho?c sau bu?c segmentation/detection.

## 2) Liên quan gì d?n x? lý ?nh?
Morphology giúp b?n làm nh?ng vi?c mà nhi?u bài x? lý ?nh c?n:
- Xóa nhi?u h?t nh? (noise).
- L?p l? nh? trong vùng d?i tu?ng.
- Làm m?n biên, n?i d?t gãy nét.
- Tách ho?c gom các vùng g?n nhau.
- Trích biên co b?n c?a d?i tu?ng.

Nói ng?n: nó là b? công c? d? "ch?nh hình" vùng pixel tru?c khi phân tích sâu hon.

## 3) Các phép toán quan tr?ng nh?t (nh?p môn)

### `Erosion` (co l?i)
- Làm vùng tr?ng nh? di.
- Tác d?ng: b? ch?m tr?ng nh?, làm m?ng nét.

### `Dilation` (n? ra)
- Làm vùng tr?ng to ra.
- Tác d?ng: n?i nét d?t, làm d?y vùng nh?.

### `Opening = Erosion -> Dilation`
- Xóa v?t th? tr?ng nh? nhung gi? hình l?n tuong d?i ?n.
- Dùng d? kh? nhi?u nh? l?.

### `Closing = Dilation -> Erosion`
- L?p l? den nh? trong vùng tr?ng.
- Dùng d? vá vùng b? th?ng, làm li?n kh?i.

### `Morphological Gradient`
- Nh?n m?nh biên (rìa) d?i tu?ng.

### `Top-hat` và `Black-hat`
- Top-hat: l?y chi ti?t sáng nh?.
- Black-hat: l?y chi ti?t t?i nh?.

## 4) V?i bài "kh? các d?i tu?ng d?c bi?t" thì x? lý th? nào?

Tùy “d?i tu?ng d?c bi?t” là gì (b?i nhi?u, ch?, logo, v?t d?m, du?ng m?nh), nhung pipeline th?c t? thu?ng là:

1. Ð?c ?nh, d?i sang grayscale.
2. T?o mask d?i tu?ng c?n xóa:
   - threshold thu?ng ho?c adaptive threshold.
   - k?t h?p edge n?u c?n.
3. Làm s?ch mask b?ng morphology:
   - `opening` d? b? nhi?u nh?.
   - `closing` d? l?p l?/ghép vùng.
4. Dùng mask dó d?:
   - ho?c gi?/xóa tr?c ti?p b?ng phép logic,
   - ho?c dua vào `inpaint` d? ph?c h?i n?n t? nhiên hon.
5. So sánh k?t qu? v?i nhi?u kích thu?c kernel (3x3, 5x5, 7x7).

## 5) Vì sao ph?i th? nhi?u kernel?
Kernel là tham s? quan tr?ng nh?t c?a morphology:
- Kernel nh?: gi? chi ti?t t?t nhung xóa y?u.
- Kernel l?n: xóa m?nh nhung d? m?t chi ti?t th?t.

Không có m?t kernel dúng cho m?i ?nh, nên luôn th? theo b? giá tr?.

## 6) Ví d? OpenCV ng?n (d? hi?u)

```python
import cv2
import numpy as np

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

# 1) T?o mask t?m b?ng threshold
_, mask = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)

# 2) Morphology d? làm s?ch mask
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# 3) Ph?c h?i ?nh b?ng inpaint (n?u mu?n xóa v?t th? và di?n n?n)
color = cv2.imread("input.jpg")
result = cv2.inpaint(color, mask, 5, cv2.INPAINT_TELEA)

cv2.imwrite("mask_clean.png", mask)
cv2.imwrite("result.png", result)
```

## 7) L?i ngu?i m?i hay g?p
- Dùng nh?m ?nh màu tr?c ti?p cho morphology mà không chu?n hóa mask.
- Threshold sai làm mask dính n?n quá nhi?u.
- Kernel quá to làm m?t luôn chi ti?t quan tr?ng.
- Ch? th? 1 c?u hình r?i k?t lu?n thu?t toán kém.

## 8) K?t lu?n ng?n
Morphological algorithms là nhóm k? thu?t x? lý hình d?ng pixel r?t co b?n nhung c?c h?u ích.
Trong bài toán "kh? d?i tu?ng d?c bi?t", morphology thu?ng là bu?c t?o/làm s?ch mask, sau dó k?t h?p inpainting d? có ?nh ph?c h?i t? nhiên.

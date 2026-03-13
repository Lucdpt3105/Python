import cv2

# Khởi tạo video
cap = cv2.VideoCapture('Jungle1.mp4')

# 1. Khởi tạo công cụ Trừ Nền (Background Subtractor)
fgbg = cv2.createBackgroundSubtractorMOG2()

while (cap.isOpened()):
    ret, frame = cap.read()

    # Bắt lỗi nếu video kết thúc (không đọc được frame nữa)
    if not ret:
        break

    # --- BẮT ĐẦU XỬ LÝ PHÁT HIỆN CHUYỂN ĐỘNG ---

    # 2. Áp dụng thuật toán trừ nền để tạo ra một "mặt nạ" (mask)
    # Những pixel chuyển động sẽ có màu trắng, nền đứng im sẽ có màu đen
    fgmask = fgbg.apply(frame)

    # 3. Tìm các đường viền (contours) của các đốm màu trắng (vật thể chuyển động)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Vẽ khung chữ nhật quanh các vùng có chuyển động
    for contour in contours:
        # Lọc bỏ bớt nhiễu (chỉ lấy những chuyển động có diện tích đủ lớn)
        if cv2.contourArea(contour) < 500:
            continue

        # Tính toán tọa độ và kích thước của khung chữ nhật
        x, y, w, h = cv2.boundingRect(contour)

        # Vẽ khung chữ nhật màu xanh lá lên khung hình gốc
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # --- KẾT THÚC XỬ LÝ ---

    # Hiển thị video gốc (đã được vẽ thêm khung chuyển động)
    cv2.imshow('view', frame)

    # Bạn có thể mở thêm dòng dưới đây để xem cách máy tính nhìn thấy chuyển động (trắng/đen)
    # cv2.imshow('Mask', fgmask)

    # Nhấn ESC để thoát
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from pathlib import Path

# Kết nối webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Khong mo duoc webcam")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Mở video nền
script_dir = Path(__file__).resolve().parent
bg_candidates = [
    script_dir / "cyberpunk.mp4",
    script_dir / "background.mp4",
]
bg_path = next((p for p in bg_candidates if p.exists()), None)

if bg_path is None:
    raise FileNotFoundError("Khong tim thay cyberpunk.mp4 hoac background.mp4 trong thu muc hien tai")

bg_video = cv2.VideoCapture(str(bg_path))
if not bg_video.isOpened():
    raise RuntimeError(f"Khong mo duoc video nen: {bg_path}")

# Tạo segmentor
segmentor = SelfiSegmentation()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Đọc frame từ video nền
    ret_bg, bg_frame = bg_video.read()

    # Nếu video nền đã hết → quay lại đầu (loop)
    if not ret_bg:
        bg_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_bg, bg_frame = bg_video.read()

    if not ret_bg or bg_frame is None:
        print("Khong doc duoc frame tu video nen")
        break

    # Resize video nền cho khớp với webcam
    bg_frame = cv2.resize(bg_frame, (width, height))

    # Tách nền và ghép với video nền (tuong thich nhieu phien ban cvzone)
    try:
        segmented_img = segmentor.removeBG(frame, bg_frame, 0.6)
    except TypeError:
        segmented_img = segmentor.removeBG(frame, bg_frame, cutThreshold=0.6)

    # Hiển thị song song: webcam gốc | đã thay nền
    concatenated_img = cv2.hconcat([frame, segmented_img])
    cv2.imshow("Camera Live | Background Replaced", concatenated_img)

    # Nhấn Q để thoát
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
bg_video.release()
cv2.destroyAllWindows()
import os
import re
from collections import Counter, deque

import cv2
import numpy as np
import pytesseract


def zoom(frame, zoom_rate=1.0):
    w = int(frame.shape[1] * zoom_rate)
    h = int(frame.shape[0] * zoom_rate)
    return cv2.resize(frame, (w, h))
## CODE NÀY LÀ CỦA THẦY

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    if max_width < 5 or max_height < 5:
        return None

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def normalize_text(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def to_digit_char(ch, aggressive=False):
    base_map = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
        "G": "6",
        "T": "7",
    }
    if aggressive:
        base_map.update({
            "R": "3",
            "E": "3",
            "A": "4",
            "Y": "7",
        })

    if ch.isdigit():
        return ch
    return base_map.get(ch, ch)


def to_letter_char(ch):
    letter_map = {
        "0": "D",
        "1": "I",
        "2": "Z",
        "4": "A",
        "5": "S",
        "6": "G",
        "8": "B",
    }
    if ch.isalpha():
        return ch
    return letter_map.get(ch, ch)


def canonicalize_vn_plate(text):
    # Expected common motorbike format after removing separators: NN L N NNNN/NNNNN
    # Example: 81B334303 -> 81-B3 34303
    text = normalize_text(text)
    if len(text) < 7:
        return ""

    # Prefer keeping a plausible length window.
    if len(text) > 10:
        text = text[:10]

    chars = list(text)
    if len(chars) < 4:
        return ""

    chars[0] = to_digit_char(chars[0])
    chars[1] = to_digit_char(chars[1])
    chars[2] = to_letter_char(chars[2])
    chars[3] = to_digit_char(chars[3])

    for i in range(4, len(chars)):
        chars[i] = to_digit_char(chars[i], aggressive=True)

    plate = "".join(chars)
    if not re.match(r"^\d{2}[A-Z]\d\d{4,5}$", plate):
        return ""
    return plate


def format_vn_plate(plate):
    if len(plate) < 5:
        return plate
    return plate[:2] + "-" + plate[2:4] + " " + plate[4:]


def pick_best_frame_result(frame_results, last_bbox, frame_shape):
    if len(frame_results) == 0:
        return None

    frame_h, frame_w = frame_shape[:2]
    frame_cx = frame_w / 2.0
    frame_cy = frame_h / 2.0

    best = None
    best_score = -1e9
    for txt, bbox, approx in frame_results:
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        area = w * h

        score = len(txt) * 50 + area * 0.01

        # Prefer candidates near frame center (works well for video1 framing)
        center_dist = abs(cx - frame_cx) + abs(cy - frame_cy)
        score -= center_dist * 0.03

        # Temporal tracking: prefer near previous accepted bbox.
        if last_bbox is not None:
            (lx, ly, lw, lh), _ = last_bbox
            lcx = lx + lw / 2.0
            lcy = ly + lh / 2.0
            track_dist = abs(cx - lcx) + abs(cy - lcy)
            score -= track_dist * 0.08

        if score > best_score:
            best_score = score
            best = (txt, bbox, approx)

    return best


def _contiguous_ranges(mask):
    ranges = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            ranges.append((start, i - 1))
            start = None
    if start is not None:
        ranges.append((start, len(mask) - 1))
    return ranges


def split_plate_lines(plate_bin):
    h, w = plate_bin.shape
    row_white = np.sum(plate_bin > 0, axis=1)
    mask = row_white > max(8, int(0.08 * w))
    ranges = _contiguous_ranges(mask)
    ranges = [r for r in ranges if (r[1] - r[0] + 1) >= max(6, int(0.08 * h))]

    if len(ranges) >= 2:
        top = ranges[0]
        bottom = ranges[-1]
        p = 3
        y1a = max(0, top[0] - p)
        y1b = min(h, top[1] + p + 1)
        y2a = max(0, bottom[0] - p)
        y2b = min(h, bottom[1] + p + 1)
        return plate_bin[y1a:y1b, :], plate_bin[y2a:y2b, :]

    mid = h // 2
    return plate_bin[:mid, :], plate_bin[mid:, :]


def ocr_line(line_bin, whitelist):
    line_for_ocr = cv2.bitwise_not(line_bin)
    line_for_ocr = cv2.copyMakeBorder(line_for_ocr, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
    configs = [
        f"--oem 3 --psm 7 -c tessedit_char_whitelist={whitelist}",
        f"--oem 3 --psm 8 -c tessedit_char_whitelist={whitelist}",
    ]
    best = ""
    for cfg in configs:
        raw = pytesseract.image_to_string(line_for_ocr, lang="eng", config=cfg)
        txt = normalize_text(raw)
        if len(txt) > len(best):
            best = txt
    return best


def score_text(text):
    canonical = canonicalize_vn_plate(text)
    if canonical == "":
        return -1
    digits = sum(ch.isdigit() for ch in canonical)
    letters = sum(ch.isalpha() for ch in canonical)
    return digits * 2 + letters


def ocr_plate(plate_gray, forced_prefix=""):
    plate_gray = cv2.bilateralFilter(plate_gray, 7, 75, 75)
    plate_gray = cv2.equalizeHist(plate_gray)
    plate_gray = cv2.resize(plate_gray, None, fx=3.2, fy=3.2, interpolation=cv2.INTER_CUBIC)

    _, plate_bin = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    plate_bin = cv2.morphologyEx(plate_bin, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    plate_bin = cv2.morphologyEx(plate_bin, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)

    top_line, bottom_line = split_plate_lines(plate_bin)

    top_raw = ocr_line(top_line, "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    bottom_raw = ocr_line(bottom_line, "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    top_norm = normalize_text(top_raw)
    bottom_norm = normalize_text(bottom_raw)

    candidates = []

    if len(top_norm) >= 4:
        top_chars = list(top_norm[:4])
        top_chars[0] = to_digit_char(top_chars[0])
        top_chars[1] = to_digit_char(top_chars[1])
        top_chars[2] = to_letter_char(top_chars[2])
        top_chars[3] = to_digit_char(top_chars[3])
        top_part = "".join(top_chars)

        bottom_digits = "".join(to_digit_char(ch, aggressive=True) for ch in bottom_norm)
        bottom_digits = re.sub(r"[^0-9]", "", bottom_digits)

        if len(bottom_digits) >= 5:
            candidates.append(top_part + bottom_digits[:5])
        if len(bottom_digits) >= 4:
            candidates.append(top_part + bottom_digits[:4])

        if len(forced_prefix) == 4:
            if len(bottom_digits) >= 5:
                candidates.append(forced_prefix + bottom_digits[:5])
            if len(bottom_digits) >= 4:
                candidates.append(forced_prefix + bottom_digits[:4])

    # Fallback: OCR all-in-one if line split is not good enough.
    fallback_configs = [
        "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    ]
    plate_for_ocr = cv2.bitwise_not(plate_bin)
    for cfg in fallback_configs:
        raw = pytesseract.image_to_string(plate_for_ocr, lang="eng", config=cfg)
        candidates.append(normalize_text(raw))

    scored_candidates = []
    for c in candidates:
        canonical = canonicalize_vn_plate(c)
        if len(forced_prefix) == 4 and canonical != "" and len(canonical) >= 8:
            canonical = forced_prefix + canonical[4:]
            canonical = canonicalize_vn_plate(canonical)
        sc = score_text(canonical)
        if sc >= 0:
            scored_candidates.append((canonical, sc))

    if len(scored_candidates) == 0:
        return "", plate_bin

    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return scored_candidates[0][0], plate_bin


def find_plate_candidates(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel, iterations=8)
    black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel, iterations=8)
    contrast = cv2.add(gray, top_hat)
    contrast = cv2.subtract(contrast, black_hat)

    blur = cv2.GaussianBlur(contrast, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        19,
        9,
    )
    edges = cv2.Canny(thresh, 120, 255)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1200:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if h == 0:
            continue
        ratio = w / float(h)
        if not (0.7 <= ratio <= 1.9 or 2.5 <= ratio <= 7.8):
            continue

        candidates.append((approx, (x, y, w, h)))

    return gray, thresh, edges, candidates


video_path = "data/video/video1.mp4"
if not os.path.exists(video_path):
    video_path = "video1.mp4"

zoom_rate = 0.8

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Cannot open video1.mp4. Check data/video/video1.mp4 path")

os.makedirs("result", exist_ok=True)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * zoom_rate)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * zoom_rate)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25.0

video_writer = cv2.VideoWriter(
    "result/video1_ocr_result.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height),
)

tesseract_cmd = os.environ.get("TESSERACT_CMD", "").strip()
if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
elif os.name == "nt":
    win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(win_path):
        pytesseract.pytesseract.tesseract_cmd = win_path

history = deque(maxlen=10)
stable_text = ""
stable_votes = 2
last_bbox = None
frames_since_last_stable = 0
last_tentative_text = ""
prefix_history = deque(maxlen=12)
stable_prefix = ""
stable_prefix_votes = 5
suffix_history = deque(maxlen=12)
stable_suffix = ""
stable_suffix_votes = 6
lock_after_frames = 10
locked_plate = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = zoom(frame, zoom_rate)

    gray, thresh, edges, candidates = find_plate_candidates(frame)
    frame_results = []
    debug_plate = None

    for approx, (x, y, w, h) in candidates:
        warped = four_point_transform(gray, approx.reshape(4, 2).astype("float32"))
        if warped is None:
            continue

        text, plate_bin = ocr_plate(warped, forced_prefix=stable_prefix)
        if text == "":
            continue

        frame_results.append((text, (x, y, w, h), approx))
        debug_plate = plate_bin

    if len(frame_results) > 0:
        best_result = pick_best_frame_result(frame_results, last_bbox, frame.shape)
        if best_result is not None:
            vote, best_bbox, best_approx = best_result
            last_bbox = (best_bbox, best_approx)

            if len(vote) >= 4:
                prefix_history.append(vote[:4])
            if len(vote) >= 9:
                suffix_history.append(vote[4:9])

            if len(prefix_history) > 0:
                candidate_prefix, prefix_votes = Counter(prefix_history).most_common(1)[0]
                if prefix_votes >= stable_prefix_votes:
                    stable_prefix = candidate_prefix

            if len(suffix_history) > 0:
                candidate_suffix, suffix_votes = Counter(suffix_history).most_common(1)[0]
                if suffix_votes >= stable_suffix_votes:
                    stable_suffix = candidate_suffix

            if len(stable_prefix) == 4 and len(vote) >= 8:
                vote = stable_prefix + vote[4:]
            if len(stable_prefix) == 4 and len(stable_suffix) == 5:
                vote = stable_prefix + stable_suffix

            # If already locked, do not let noisy frames overwrite unless same prefix.
            if locked_plate:
                if len(vote) >= 4 and vote[:4] == locked_plate[:4]:
                    last_tentative_text = vote
                else:
                    last_tentative_text = locked_plate
            else:
                last_tentative_text = vote

            history.append(last_tentative_text)
            majority_text, votes = Counter(history).most_common(1)[0]
            if votes >= stable_votes:
                stable_text = majority_text
                frames_since_last_stable = 0
                print(format_vn_plate(stable_text))

                # Lock final prediction for short stable video.
                if len(stable_text) >= 9 and len(history) >= lock_after_frames:
                    locked_plate = stable_text
                    stable_text = locked_plate
    else:
        last_tentative_text = ""
        frames_since_last_stable += 1

    if last_bbox is not None:
        (x, y, w, h), approx = last_bbox
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
        if stable_text:
            cv2.putText(frame, format_vn_plate(stable_text), (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if frames_since_last_stable > 80:
        if not locked_plate:
            stable_text = ""

    # HUD message to always show recognition status on screen
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (frame.shape[1] - 10, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    display_text = locked_plate if locked_plate else stable_text
    if display_text:
        hud_text = "Bien so: " + format_vn_plate(display_text)
        hud_color = (0, 255, 255)
    elif last_tentative_text:
        hud_text = "Tam nhan: " + format_vn_plate(last_tentative_text)
        hud_color = (0, 255, 0)
    else:
        hud_text = "Dang tim bien so..."
        hud_color = (0, 200, 255)
    cv2.putText(frame, hud_text, (25, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, hud_color, 2)

    cv2.imshow("thresh", thresh)
    cv2.imshow("edges", edges)
    if debug_plate is not None:
        cv2.imshow("plate_bin", debug_plate)
    cv2.imshow("Video", frame)

    video_writer.write(frame)

    k = cv2.waitKey(30)
    if (k & 0xFF) == 32:
        k = -1
        while k == -1:
            k = cv2.waitKey(1)

    if (k & 0xFF) == 27:
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

    


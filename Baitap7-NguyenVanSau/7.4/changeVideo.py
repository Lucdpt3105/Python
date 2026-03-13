from pathlib import Path

import cv2
import numpy as np


def build_reference_background(video_cap, sample_count=30):
	frames = []
	video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

	for _ in range(sample_count):
		ok, frame = video_cap.read()
		if not ok:
			break
		frames.append(frame.astype(np.float32))

	if not frames:
		raise RuntimeError("Khong the tao background tham chieu tu video nguon")

	background = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
	video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
	return background


def detect_objects_and_mask(frame, background_frame, min_area, diff_threshold=30):
	diff = cv2.absdiff(frame, background_frame)
	gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	_, binary = cv2.threshold(blur, diff_threshold, 255, cv2.THRESH_BINARY)

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	mask_hard = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
	mask_hard = cv2.dilate(mask_hard, kernel, iterations=2)

	contours, _ = cv2.findContours(mask_hard.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	refined_hard = np.zeros_like(mask_hard)
	boxes = []

	for contour in contours:
		area = cv2.contourArea(contour)
		if area < min_area:
			continue
		x, y, w, h = cv2.boundingRect(contour)
		boxes.append((x, y, w, h, int(area)))
		cv2.drawContours(refined_hard, [contour], -1, 255, thickness=cv2.FILLED)

	mask_soft = cv2.GaussianBlur(refined_hard, (9, 9), 0)
	return refined_hard, mask_soft, boxes, diff


def remove_original_background(frame, alpha_3ch):
	# Step 2: keep only detected objects from SampleVid.
	return (frame.astype(np.float32) * alpha_3ch).astype(np.uint8)


def replace_with_cyberpunk_bg(objects_only, cyberpunk_frame, alpha_3ch):
	# Step 3: place detected objects on cyberpunk background.
	return (
		objects_only.astype(np.float32)
		+ cyberpunk_frame.astype(np.float32) * (1.0 - alpha_3ch)
	).astype(np.uint8)


def main():
	script_dir = Path(__file__).resolve().parent
	sample_video_path = script_dir / "Jungle1.mp4"
	cyberpunk_video_path = script_dir / "cyberpunk.mp4"
	output_path = script_dir / "SampleVid_cyberpunk_output.mp4"

	sample_cap = cv2.VideoCapture(str(sample_video_path))
	bg_cap = cv2.VideoCapture(str(cyberpunk_video_path))

	if not sample_cap.isOpened():
		raise FileNotFoundError(f"Khong mo duoc SampleVid: {sample_video_path}")
	if not bg_cap.isOpened():
		raise FileNotFoundError(f"Khong mo duoc video cyberpunk: {cyberpunk_video_path}")

	width = int(sample_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(sample_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = sample_cap.get(cv2.CAP_PROP_FPS)
	if fps <= 0:
		fps = 25.0

	reference_background = build_reference_background(sample_cap, sample_count=40)

	writer = cv2.VideoWriter(
		str(output_path),
		cv2.VideoWriter_fourcc(*"mp4v"),
		fps,
		(width, height),
	)

	alpha_smooth = None
	min_area = width * height * 0.0015
	diff_threshold = 30

	while True:
		ok_sample, sample_frame = sample_cap.read()
		if not ok_sample:
			break

		ok_bg, bg_frame = bg_cap.read()
		if not ok_bg:
			bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
			ok_bg, bg_frame = bg_cap.read()
			if not ok_bg:
				break

		bg_frame = cv2.resize(bg_frame, (width, height))
		bg_ref = cv2.resize(reference_background, (width, height))

		# Step 1: detect objects in SampleVid and generate foreground mask.
		mask_hard, mask_soft, boxes, diff = detect_objects_and_mask(
			sample_frame,
			bg_ref,
			min_area,
			diff_threshold=diff_threshold,
		)
		alpha = mask_soft.astype(np.float32) / 255.0

		if alpha_smooth is None:
			alpha_smooth = alpha
		else:
			alpha_smooth = 0.6 * alpha_smooth + 0.4 * alpha

		alpha_3ch = np.expand_dims(alpha_smooth, axis=2)

		# Step 2: remove original SampleVid background.
		objects_only = remove_original_background(sample_frame, alpha_3ch)

		# Step 3: replace with cyberpunk background.
		composite_soft = replace_with_cyberpunk_bg(objects_only, bg_frame, alpha_3ch)

		# Hard mask keeps object edges clear, soft mask reduces jagged borders.
		mask_inv = cv2.bitwise_not(mask_hard)
		fg_hard = cv2.bitwise_and(sample_frame, sample_frame, mask=mask_hard)
		bg_hard = cv2.bitwise_and(bg_frame, bg_frame, mask=mask_inv)
		composite_hard = cv2.add(fg_hard, bg_hard)
		composite = cv2.addWeighted(composite_hard, 0.65, composite_soft, 0.35, 0)

		preview = sample_frame.copy()
		overlay = preview.copy()
		for x, y, w, h, area in boxes:
			cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.rectangle(overlay, (x, y), (x + w, y + h), (80, 255, 120), -1)
			cv2.putText(
				preview,
				f"Obj A={area}",
				(x, max(20, y - 6)),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5,
				(0, 255, 0),
				2,
			)

		# Transparent fill makes detected objects easier to see at a glance.
		preview = cv2.addWeighted(overlay, 0.18, preview, 0.82, 0)
		cv2.putText(preview, f"Objects: {len(boxes)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
		cv2.putText(preview, f"Diff threshold: {diff_threshold}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

		cv2.putText(composite, "Q: thoat", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
		
		writer.write(composite)

		cv2.imshow("SampleVid Objects", preview)
		cv2.imshow("Diff", diff)
		cv2.imshow("Foreground Mask", mask_hard)
		cv2.imshow("Objects Only", objects_only)
		cv2.imshow("Cyberpunk Replaced", composite)

		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
		if key == ord("["):
			diff_threshold = max(5, diff_threshold - 2)
		if key == ord("]"):
			diff_threshold = min(80, diff_threshold + 2)

	sample_cap.release()
	bg_cap.release()
	writer.release()
	cv2.destroyAllWindows()
	print(f"Da luu video ket qua tai: {output_path}")


if __name__ == "__main__":
	main()

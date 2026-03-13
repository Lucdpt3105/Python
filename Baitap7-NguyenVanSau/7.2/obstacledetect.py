import csv
from datetime import datetime
from pathlib import Path

import cv2


def intersects_zone(box, zone):
	x, y, w, h = box
	zx1, zy1, zx2, zy2 = zone

	overlap_w = max(0, min(x + w, zx2) - max(x, zx1))
	overlap_h = max(0, min(y + h, zy2) - max(y, zy1))
	return overlap_w > 0 and overlap_h > 0


def main():
	script_dir = Path(__file__).resolve().parent
	video_path = script_dir / "peoplenyc.mp4"
	output_csv = script_dir / "obstacle_times.csv"

	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise FileNotFoundError(f"Cannot open video: {video_path}")

	bg_subtractor = cv2.createBackgroundSubtractorMOG2(
		history=700,
		varThreshold=40,
		detectShadows=True,
	)

	status_list = [0, 0]
	times = []

	while True:
		ok, frame = cap.read()
		if not ok:
			if status_list[-1] == 1:
				times.append(datetime.now())
			break

		status = 0
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (7, 7), 0)

		fg_mask = bg_subtractor.apply(gray)

		# Remove shadows (value 127) and keep strong foreground only.
		thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, None, iterations=1)
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, None, iterations=2)
		thresh = cv2.dilate(thresh, None, iterations=1)

		contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		h, w = frame.shape[:2]
		min_area = h * w * 0.0025

		zone = (
			int(w * 0.30),
			int(h * 0.55),
			int(w * 0.70),
			int(h * 0.95),
		)
		zx1, zy1, zx2, zy2 = zone
		cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 180, 0), 2)
		cv2.putText(frame, "Obstacle zone", (zx1, max(20, zy1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

		for contour in contours:
			area = cv2.contourArea(contour)
			if area < min_area:
				continue

			x, y, bw, bh = cv2.boundingRect(contour)
			if bh < int(h * 0.06):
				continue

			box = (x, y, bw, bh)
			obstacle = intersects_zone(box, zone)

			color = (0, 255, 0)
			label = "Moving object"
			if obstacle:
				status = 1
				color = (0, 0, 255)
				label = "OBSTACLE"

			cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
			cv2.putText(frame, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

		if status == 1:
			cv2.putText(frame, "WARNING: obstacle detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

		status_list.append(status)
		status_list = status_list[-2:]

		if status_list[-1] == 1 and status_list[-2] == 0:
			times.append(datetime.now())
		if status_list[-1] == 0 and status_list[-2] == 1:
			times.append(datetime.now())

		cv2.imshow("Foreground", fg_mask)
		cv2.imshow("Threshold", thresh)
		cv2.imshow("Obstacle Detection", frame)

		key = cv2.waitKey(20) & 0xFF
		if key == ord("q"):
			if status == 1:
				times.append(datetime.now())
			break

	rows = []
	for i in range(0, len(times) - 1, 2):
		rows.append([times[i].isoformat(), times[i + 1].isoformat()])

	with output_csv.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["Start", "End"])
		writer.writerows(rows)

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()

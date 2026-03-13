import cv2


def main():
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		raise RuntimeError("Khong mo duoc webcam")

	threshold_value = 120
	min_contour_area = 1500

	while True:
		ok, frame = cap.read()
		if not ok:
			break

		# 1) Resize frame for stable processing speed.
		frame = cv2.resize(frame, (960, 540))

		# 2) Flip for mirror-like webcam view.
		frame = cv2.flip(frame, 1)

		# 3) Convert to grayscale.
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# 4) Blur to smooth small pixel noise.
		blur = cv2.GaussianBlur(gray, (7, 7), 0)

		# 5) Threshold to separate foreground-like areas.
		_, thresh = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)

		# 6) Denoise with morphology operations.
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
		denoised = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
		denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel, iterations=2)

		# 7) Find contours.
		contours, _ = cv2.findContours(denoised.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		large_count = 0

		# 8) BoundingRect + draw rectangle for large contours only.
		for contour in contours:
			area = cv2.contourArea(contour)
			if area < min_contour_area:
				continue

			x, y, w, h = cv2.boundingRect(contour)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.putText(
				frame,
				f"A={int(area)}",
				(x, max(20, y - 8)),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.55,
				(0, 255, 0),
				2,
			)
			large_count += 1

		cv2.putText(frame, f"Contours lon: {large_count}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 200, 255), 2)
		cv2.putText(frame, f"Thresh: {threshold_value} | MinArea: {min_contour_area}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
		cv2.putText(frame, "Q: thoat | I/K: tang/giam threshold | O/L: tang/giam min area", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

		cv2.imshow("Original + Rectangles", frame)
		cv2.imshow("Gray", gray)
		cv2.imshow("Blur", blur)
		cv2.imshow("Threshold", thresh)
		cv2.imshow("Denoised", denoised)

		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
		if key == ord("i"):
			threshold_value = min(255, threshold_value + 5)
		if key == ord("k"):
			threshold_value = max(0, threshold_value - 5)
		if key == ord("o"):
			min_contour_area += 100
		if key == ord("l"):
			min_contour_area = max(100, min_contour_area - 100)

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()

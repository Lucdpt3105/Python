import cv2, pandas
from datetime import datetime
from pathlib import Path

status_list = [None,None]
times = []

script_dir = Path(__file__).resolve().parent
video_candidates = [
    script_dir / "peoplenyc.mp4",
    script_dir.parent / "7.2" / "peoplenyc.mp4",
    script_dir / "Jungle1.mp4",
]

video_path = next((path for path in video_candidates if path.exists()), video_candidates[0])
video = cv2.VideoCapture(str(video_path))

if not video.isOpened():
    raise FileNotFoundError(f"Khong mo duoc video: {video_path}")

# MOG2 is more robust for crowded street scenes than fixed frame differencing.
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=45,
    detectShadows=True,
)

while True:
    check, frame = video.read()

    if not check:
        if status_list[-1] == 1:
            times.append(datetime.now())
        break

    status = 0
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)

    foreground = bg_subtractor.apply(gray)
    thresh_frame = cv2.threshold(foreground, 200, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.medianBlur(thresh_frame, 5)
    thresh_frame = cv2.morphologyEx(thresh_frame, cv2.MORPH_OPEN, None, iterations=1)
    thresh_frame = cv2.morphologyEx(thresh_frame, cv2.MORPH_CLOSE, None, iterations=2)

    (cnts,_)=cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_h, frame_w = gray.shape[:2]
    min_contour_area = frame_h * frame_w * 0.003

    # Define a danger zone near the lower-middle frame where obstacles matter most.
    zone_x1 = int(frame_w * 0.3)
    zone_x2 = int(frame_w * 0.7)
    zone_y1 = int(frame_h * 0.55)
    zone_y2 = int(frame_h * 0.95)
    cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 180, 0), 2)
    cv2.putText(frame, "Obstacle Zone", (zone_x1, zone_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

    for contour in cnts:
        if cv2.contourArea(contour) < min_contour_area:
            continue

        (x, y, w, h)=cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        overlap_x = max(0, min(x + w, zone_x2) - max(x, zone_x1))
        overlap_y = max(0, min(y + h, zone_y2) - max(y, zone_y1))
        is_obstacle = overlap_x > 0 and overlap_y > 0

        if is_obstacle:
            status = 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 3)
            cv2.putText(frame, "OBSTACLE", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    if status == 1:
        cv2.putText(frame, "CANH BAO: Co vat can", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

    status_list.append(status)

    status_list=status_list[-2:]


    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())


    cv2.imshow("Gray Frame",gray)
    cv2.imshow("Delta Frame",foreground)
    cv2.imshow("Threshold Frame",thresh_frame)
    cv2.imshow("Color Frame",frame)

    key=cv2.waitKey(1)

    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

print(status_list)
print(times)

records = [{"Start": times[i], "End": times[i + 1]} for i in range(0, len(times) - 1, 2)]
df = pandas.DataFrame(records, columns=["Start", "End"])
df.to_csv(Path(__file__).with_name("Times.csv"), index=False)

video.release()
cv2.destroyAllWindows()
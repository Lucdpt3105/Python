import cv2, pandas
from datetime import datetime
from pathlib import Path

status_list = [None,None]
times = []
background = None

script_dir = Path(__file__).resolve().parent
video_path = script_dir / "Jungle1.mp4"
video = cv2.VideoCapture(str(video_path))

if not video.isOpened():
    raise FileNotFoundError(f"Khong mo duoc video: {video_path}")

while True:
    check, frame = video.read()

    if not check:
        if status_list[-1] == 1:
            times.append(datetime.now())
        break

    status = 0
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)

    if background is None:
        background = gray.astype("float")
        continue

    cv2.accumulateWeighted(gray, background, 0.03)
    delta_frame = cv2.absdiff(gray, cv2.convertScaleAbs(background))
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.erode(thresh_frame, None, iterations=1)
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    (cnts,_)=cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_h, frame_w = gray.shape[:2]
    min_contour_area = frame_h * frame_w * 0.005

    for contour in cnts:
        if cv2.contourArea(contour) < min_contour_area:
            continue

        (x, y, w, h)=cv2.boundingRect(contour)
        status = 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)

    if status == 1:
        cv2.putText(frame, "MOTION DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

    status_list.append(status)

    status_list=status_list[-2:]


    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())


    cv2.imshow("Gray Frame",gray)
    cv2.imshow("Delta Frame",delta_frame)
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
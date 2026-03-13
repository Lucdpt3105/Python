import cv2


def main():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print('Cannot open camera')
        return

    bg_video = cv2.VideoCapture('Jungle1.mp4')
    if not bg_video.isOpened():
        print('Cannot open background video: Jungle1.mp4')
        cam.release()
        return

    background_frame = None
    show_mask = False

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        if background_frame is None:
            cv2.putText(
                frame,
                'Press B to capture background, ESC to quit',
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow('camera', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord('b'):
                background_frame = frame.copy()
            continue

        ret_bg, bg = bg_video.read()
        if not ret_bg:
            bg_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_bg, bg = bg_video.read()
        if not ret_bg:
            break

        bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))

        diff = cv2.absdiff(frame, background_frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask_inv = cv2.bitwise_not(mask)

        fg = cv2.bitwise_and(frame, frame, mask=mask)
        bg_part = cv2.bitwise_and(bg, bg, mask=mask_inv)
        combined = cv2.add(fg, bg_part)

        cv2.imshow('camera', combined)
        if show_mask:
            cv2.imshow('mask', mask)
        else:
            # Only destroy if it exists; some builds error on missing window.
            try:
                if cv2.getWindowProperty('mask', cv2.WND_PROP_VISIBLE) >= 0:
                    cv2.destroyWindow('mask')
            except cv2.error:
                pass

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        if key == ord('b'):
            background_frame = frame.copy()
        if key == ord('m'):
            show_mask = not show_mask

    cam.release()
    bg_video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

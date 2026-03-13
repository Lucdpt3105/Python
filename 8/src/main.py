import argparse
import cv2
import numpy as np
import mediapipe as mp


def parse_args():
    parser = argparse.ArgumentParser(description="Replace webcam background in real time")
    parser.add_argument("--background", type=str, default="", help="Path to background image")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--source",
        type=str,
        default="camera",
        help="Input source: 'camera' or path to a video file",
    )
    return parser.parse_args()


def load_background(path, frame_shape):
    h, w = frame_shape[:2]
    if not path:
        return None

    image = cv2.imread(path)
    if image is None:
        print(f"[WARN] Cannot read background image: {path}. Fallback to blur mode.")
        return None

    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


def try_open_camera(preferred_index, max_index=5):
    indices = [preferred_index] + [i for i in range(max_index + 1) if i != preferred_index]
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            return cap, idx
        cap.release()
    return None, None


def main():
    args = parse_args()

    using_camera = args.source.lower() == "camera"
    opened_index = None

    if using_camera:
        cap, opened_index = try_open_camera(args.camera)
        if cap is None:
            raise RuntimeError(
                "Cannot open any camera device. "
                "If you are using WSL, run from Windows Python instead, or pass --source <video-file>."
            )
    else:
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {args.source}")

    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    blur_mode = False
    cached_bg = None
    cached_shape = None

    if using_camera:
        print(f"[INFO] Camera opened at index: {opened_index}")
    else:
        print(f"[INFO] Video source opened: {args.source}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if using_camera:
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = segmenter.process(rgb)
            mask = result.segmentation_mask

            condition = mask > 0.6
            condition_3c = np.stack((condition,) * 3, axis=-1)

            if args.background and (cached_bg is None or cached_shape != frame.shape):
                cached_bg = load_background(args.background, frame.shape)
                cached_shape = frame.shape

            replacement = cached_bg
            if replacement is None:
                if blur_mode:
                    replacement = cv2.GaussianBlur(frame, (55, 55), 0)
                else:
                    replacement = np.full(frame.shape, (30, 30, 30), dtype=np.uint8)

            output = np.where(condition_3c, frame, replacement)

            cv2.putText(
                output,
                "q: quit | b: blur background",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Camera Background Replacement", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("b"):
                blur_mode = not blur_mode

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

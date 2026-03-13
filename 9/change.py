import argparse
import cv2
import numpy as np


def build_parser():
    p = argparse.ArgumentParser(description='Replace background of a video using background subtraction.')
    p.add_argument('--fg', default='Jungle1.mp4', help='Foreground video path (monkeys)')
    p.add_argument('--bg', default='futuristic-megalopolis-aerial-cyberpunk-neo-view-moewalls-com.mp4', help='Background video path')
    p.add_argument('--out', default='', help='Optional output video path (e.g. out.mp4)')
    p.add_argument('--scale-bg', type=float, default=1.0, help='Scale background before resize (e.g. 1.0)')
    p.add_argument('--show-mask', action='store_true', help='Show mask window')
    return p


def main():
    args = build_parser().parse_args()

    cap_fg = cv2.VideoCapture(args.fg)
    if not cap_fg.isOpened():
        print(f'Cannot open foreground video: {args.fg}')
        return

    cap_bg = cv2.VideoCapture(args.bg)
    if not cap_bg.isOpened():
        print(f'Cannot open background video: {args.bg}')
        cap_fg.release()
        return

    width = int(cap_fg.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_fg.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_fg.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f'Cannot open output writer: {args.out}')
            writer = None

    bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    while True:
        ret_fg, fg = cap_fg.read()
        if not ret_fg:
            break

        ret_bg, bg = cap_bg.read()
        if not ret_bg:
            cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_bg, bg = cap_bg.read()
        if not ret_bg:
            break

        if args.scale_bg != 1.0:
            bg = cv2.resize(bg, None, fx=args.scale_bg, fy=args.scale_bg)
        bg = cv2.resize(bg, (width, height))

        # Preprocess foreground to stabilize background subtraction.
        fg_blur = cv2.GaussianBlur(fg, (5, 5), 0)

        mask = bg_sub.apply(fg_blur)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Fill small holes then remove small noise.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Feather edges for a clearer subject boundary.
        mask_f = cv2.GaussianBlur(mask, (7, 7), 0)
        alpha = (mask_f.astype(np.float32) / 255.0)[..., None]

        combined = (fg.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

        cv2.imshow('result', combined)
        if args.show_mask:
            cv2.imshow('mask', mask)
        else:
            try:
                if cv2.getWindowProperty('mask', cv2.WND_PROP_VISIBLE) >= 0:
                    cv2.destroyWindow('mask')
            except cv2.error:
                pass

        if writer is not None:
            writer.write(combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap_fg.release()
    cap_bg.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

import io
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import streamlit as st


@dataclass
class AppConfig:
    rotate_angle: int
    resize_scale: float
    blur_kernel: int
    threshold_value: int
    canny_low: int
    canny_high: int
    noise_sigma: int
    mask_radius_ratio: float
    blob_min_area: int
    blob_max_area: int
    grabcut_iters: int


TECHNIQUES = [
    '1) Convert to Grayscale',
    '2) Rotate an Image',
    '3) Crop an Image',
    '4) Resize an Image',
    '5) Blur an Image',
    '6) Thresholding',
    '7) Boundary (Edge) Detection',
    '8) Noise Filtering',
    '9) Morphological Operations',
    '10) Contour Detection',
    '11) Image Masking and Bitwise Operations',
    '12) Histogram Equalization',
    '13) Perspective Transformation',
    '14) Blob Detection',
    '15) Remove Background',
]


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def decode_uploaded_image(uploaded_file) -> np.ndarray:
    bytes_data = uploaded_file.read()
    nparr = np.frombuffer(bytes_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot decode uploaded image.')
    return image


def technique_results(image: np.ndarray, cfg: AppConfig) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    results: Dict[str, List[Tuple[str, np.ndarray]]] = {}

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1) Grayscale
    results[TECHNIQUES[0]] = [('grayscale', gray)]

    # 2) Rotate
    center = (w // 2, h // 2)
    matrix_rotate = cv2.getRotationMatrix2D(center, cfg.rotate_angle, 1.0)
    rotated = cv2.warpAffine(image, matrix_rotate, (w, h))
    results[TECHNIQUES[1]] = [('rotated', rotated)]

    # 3) Crop
    crop_h = max(1, h // 2)
    crop_w = max(1, w // 2)
    y0 = max(0, h // 4)
    x0 = max(0, w // 4)
    crop = image[y0:y0 + crop_h, x0:x0 + crop_w]
    results[TECHNIQUES[2]] = [('center_crop', crop)]

    # 4) Resize
    resized = cv2.resize(image, None, fx=cfg.resize_scale, fy=cfg.resize_scale, interpolation=cv2.INTER_AREA)
    results[TECHNIQUES[3]] = [('resized', resized)]

    # 5) Blur
    blur_k = cfg.blur_kernel if cfg.blur_kernel % 2 == 1 else cfg.blur_kernel + 1
    blurred = cv2.GaussianBlur(image, (blur_k, blur_k), 0)
    results[TECHNIQUES[4]] = [('gaussian_blur', blurred)]

    # 6) Thresholding
    _, thresholded = cv2.threshold(gray, cfg.threshold_value, 255, cv2.THRESH_BINARY)
    results[TECHNIQUES[5]] = [('binary_threshold', thresholded)]

    # 7) Edge Detection
    edges = cv2.Canny(gray, cfg.canny_low, cfg.canny_high)
    results[TECHNIQUES[6]] = [('canny_edges', edges)]

    # 8) Noise Filtering
    noise = np.random.normal(0, cfg.noise_sigma, gray.shape).astype(np.int16)
    noisy = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    filtered = cv2.medianBlur(noisy, 5)
    results[TECHNIQUES[7]] = [('noisy', noisy), ('median_filtered', filtered)]

    # 9) Morphological Operations
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(thresholded, kernel, iterations=1)
    dilated = cv2.dilate(thresholded, kernel, iterations=1)
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    results[TECHNIQUES[8]] = [
        ('eroded', eroded),
        ('dilated', dilated),
        ('opened', opened),
        ('closed', closed),
    ]

    # 10) Contour Detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = image.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    results[TECHNIQUES[9]] = [('contours', contour_img)]

    # 11) Masking + Bitwise
    mask = np.zeros((h, w), dtype=np.uint8)
    radius = int(min(h, w) * cfg.mask_radius_ratio)
    cv2.circle(mask, (w // 2, h // 2), max(radius, 1), 255, -1)
    masked = cv2.bitwise_and(image, image, mask=mask)
    results[TECHNIQUES[10]] = [('mask', mask), ('masked_image', masked)]

    # 12) Histogram Equalization
    equalized = cv2.equalizeHist(gray)
    results[TECHNIQUES[11]] = [('equalized_gray', equalized)]

    # 13) Perspective Transform
    src = np.float32([
        [w * 0.10, h * 0.15],
        [w * 0.90, h * 0.10],
        [w * 0.10, h * 0.90],
        [w * 0.90, h * 0.85],
    ])
    dst = np.float32([
        [0, 0],
        [w - 1, 0],
        [0, h - 1],
        [w - 1, h - 1],
    ])
    matrix_persp = cv2.getPerspectiveTransform(src, dst)
    perspective = cv2.warpPerspective(image, matrix_persp, (w, h))
    results[TECHNIQUES[12]] = [('perspective', perspective)]

    # 14) Blob Detection
    blob_params = cv2.SimpleBlobDetector_Params()
    blob_params.filterByArea = True
    blob_params.minArea = float(cfg.blob_min_area)
    blob_params.maxArea = float(cfg.blob_max_area)
    blob_params.filterByCircularity = False
    blob_params.filterByConvexity = False
    blob_params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(blob_params)
    keypoints = detector.detect(gray)
    blob_view = cv2.drawKeypoints(
        image,
        keypoints,
        np.array([]),
        (0, 0, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    results[TECHNIQUES[13]] = [('blob_detection', blob_view)]

    # 15) Remove Background (GrabCut)
    mask_gc = np.zeros(image.shape[:2], np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    rect = (
        max(1, int(w * 0.05)),
        max(1, int(h * 0.05)),
        max(1, int(w * 0.90)),
        max(1, int(h * 0.90)),
    )
    cv2.grabCut(image, mask_gc, rect, bg_model, fg_model, cfg.grabcut_iters, cv2.GC_INIT_WITH_RECT)
    mask_fg = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    foreground = cv2.bitwise_and(image, image, mask=mask_fg)
    white_bg = np.full_like(image, 255)
    removed_bg = np.where(mask_fg[..., None] == 255, foreground, white_bg)
    results[TECHNIQUES[14]] = [('foreground_mask', mask_fg), ('background_removed', removed_bg)]

    return results


def image_to_png_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('Failed to encode image as PNG.')
    return encoded.tobytes()


def build_zip(results: Dict[str, List[Tuple[str, np.ndarray]]]) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for technique_name, images in results.items():
            safe_name = technique_name.split(')')[0].strip()
            for image_name, image in images:
                file_name = f'{safe_name}_{image_name}.png'
                zip_file.writestr(file_name, image_to_png_bytes(image))
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def main() -> None:
    st.set_page_config(page_title='Image Processing - 15 Techniques', layout='wide')
    st.title('Image Processing App (15 Techniques)')
    st.write('Upload an image, choose one or many techniques, and get results instantly.')

    with st.sidebar:
        st.header('Controls')
        selected = st.multiselect('Choose techniques', TECHNIQUES, default=TECHNIQUES)
        rotate_angle = st.slider('Rotate angle', -180, 180, 30, 1)
        resize_scale = st.slider('Resize scale', 0.1, 2.0, 0.6, 0.1)
        blur_kernel = st.slider('Blur kernel (odd)', 3, 21, 9, 2)
        threshold_value = st.slider('Threshold value', 0, 255, 120, 1)
        canny_low = st.slider('Canny low', 0, 255, 100, 1)
        canny_high = st.slider('Canny high', 0, 255, 200, 1)
        noise_sigma = st.slider('Noise sigma', 1, 60, 25, 1)
        mask_radius_ratio = st.slider('Mask radius ratio', 0.05, 0.45, 0.25, 0.01)
        blob_min_area = st.slider('Blob min area', 5, 5000, 40, 5)
        blob_max_area = st.slider('Blob max area', 1000, 200000, 100000, 1000)
        grabcut_iters = st.slider('GrabCut iterations', 1, 10, 5, 1)

    uploaded = st.file_uploader('Upload image', type=['jpg', 'jpeg', 'png', 'bmp', 'webp'])

    if uploaded is None:
        st.info('Please upload an image to start.')
        return

    image = decode_uploaded_image(uploaded)
    st.subheader('Original')
    st.image(bgr_to_rgb(image), use_container_width=True)

    config = AppConfig(
        rotate_angle=rotate_angle,
        resize_scale=resize_scale,
        blur_kernel=blur_kernel,
        threshold_value=threshold_value,
        canny_low=canny_low,
        canny_high=canny_high,
        noise_sigma=noise_sigma,
        mask_radius_ratio=mask_radius_ratio,
        blob_min_area=blob_min_area,
        blob_max_area=blob_max_area,
        grabcut_iters=grabcut_iters,
    )

    all_results = technique_results(image, config)
    selected_results = {name: all_results[name] for name in selected}

    if not selected_results:
        st.warning('Choose at least one technique in the sidebar.')
        return

    st.subheader('Results')
    for technique_name, images in selected_results.items():
        with st.expander(technique_name, expanded=False):
            cols = st.columns(2)
            for idx, (img_name, out_img) in enumerate(images):
                with cols[idx % 2]:
                    st.caption(img_name)
                    st.image(bgr_to_rgb(out_img), use_container_width=True)

    zip_bytes = build_zip(selected_results)
    st.download_button(
        label='Download selected results (.zip)',
        data=zip_bytes,
        file_name='image_processing_results.zip',
        mime='application/zip',
    )


if __name__ == '__main__':
    main()

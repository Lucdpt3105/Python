import cv2
import numpy as np


img = cv2.imread('car.jpg')

if img is not None:

    resized_small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Resized Image (OpenCV)', resized_small)

    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Không tìm thấy file ảnh.")

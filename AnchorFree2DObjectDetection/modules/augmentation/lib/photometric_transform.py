import cv2
import numpy as np

# ---------------------------------------------------------------------------------------------------------------
def augment_hsv(im, hgain=0.015, sgain=0.7, vgain=0.4):
    # https://github.com/ultralytics/yolov5/blob/master/utils/augmentations.py
    # HSV color-space augmentation
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_RGB2HSV))
    dtype = im.dtype  # uint8

    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    modified_img = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
    return modified_img

# ---------------------------------------------------------------------------------------------------------------s
def hist_equalize(im, clahe=True):
    # https://github.com/ultralytics/yolov5/blob/master/utils/augmentations.py
    # Equalize histogram on BGR image 'im' with im.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)  # convert YUV image to RGB
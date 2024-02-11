# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : geometric augmentation functions 
# ---------------------------------------------------------------------------------------------------------------
import numpy as np
import cv2
import math
import random
from typing import Tuple
from modules.augmentation.constants_bdd import _INTERPOLATION_MODE_, _BORDER_

# ---------------------------------------------------------------------------------------------------------------
def resize_with_padding(
    img: np.ndarray, 
    bboxes: np.ndarray, 
    img_h: int, 
    img_w: int,
    border: int = _BORDER_) -> Tuple[np.ndarray, np.ndarray]:
    ''' Resize the image while preserving the aspect ratio. 
        The bboxes are assumed to be un-normalized '''
    
    # handle the corner case
    if len(bboxes) == 0: bboxes = np.zeros([0, 4], dtype=np.float32)

    # normalize bbox
    bboxes = normalize_bbox(bboxes, img.shape[0], img.shape[1])

    # compute feasible scales
    scale_h = img_h / img.shape[0]
    scale_w = img_w / img.shape[1]
    
    # resize image
    new_w = int(scale_h * img.shape[1])
    new_h = int(scale_w * img.shape[0])

    if new_w <= img_w: new_h = img_h
    else: new_w = img_w
    img = cv2.resize(img, (new_w, new_h), _INTERPOLATION_MODE_)
    
    # compute where to put the image
    ow = ( img_w - new_w ) // 2 
    oh = ( img_h - new_h ) // 2 

    # new image
    new_img = np.full(shape=(img_h, img_w, 3), fill_value=border, dtype=img.dtype)
    new_img[oh:oh + new_h, ow:ow + new_w, :] = img
        
    # update the bbox
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * img.shape[0] + oh
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * img.shape[1] + ow
    return new_img, bboxes

# ---------------------------------------------------------------------------------------------------------------
def random_perspective(
    im: np.ndarray,
    degrees: float = 10,
    translate: float = .1,
    scale: float = .1,
    shear : float = 10,
    perspective : float = 0.0,
    border: int = _BORDER_) -> Tuple[np.ndarray, np.ndarray, int, int]:
    
    # https://github.com/ultralytics/yolov5/blob/master/utils/augmentations.py

    height = im.shape[0]  # shape(h,w,c)
    width = im.shape[1]

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if perspective:
        im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(border, border, border))
    else:  # affine
        im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(border, border, border))

    return im, M, height, width

# ---------------------------------------------------------------------------------------------------------------
def geometric_transform_bbox(
    bbox: np.ndarray, 
    T: np.ndarray,
    height: int, 
    width: int) -> np.ndarray:
    ''' Transform the bounding box. The bboxes are assumed to be un-normalized '''
    # handle the corner case
    if len(bbox) == 0: bbox = np.zeros([0, 4], dtype=np.float32)
    # shapes: (n x 2)
    bbox_x1y1 = bbox[:, [0,1]] @ T[:2, :2].transpose() + T[:2, -1]   # (x1, y1)
    bbox_x2y2 = bbox[:, [2,3]] @ T[:2, :2].transpose() + T[:2, -1]   # (x2, y2)
    bbox_x2y1 = bbox[:, [2,1]] @ T[:2, :2].transpose() + T[:2, -1]   # (x2, y1)
    bbox_x1y2 = bbox[:, [0,3]] @ T[:2, :2].transpose() + T[:2, -1]   # (x1, y2)
    bbox = np.stack([bbox_x1y1, bbox_x2y2, bbox_x2y1, bbox_x1y2], axis=-1)  # shape: (n x 2 x 4)

    x1 = np.clip(np.min(bbox[:, 0, :], axis=-1), 0, width)
    x2 = np.clip(np.max(bbox[:, 0, :], axis=-1), 0, width) 
    y1 = np.clip(np.min(bbox[:, 1, :], axis=-1), 0, height)
    y2 = np.clip(np.max(bbox[:, 1, :], axis=-1), 0, height) 
    bbox = np.stack([x1, y1, x2, y2], axis=-1)
    return bbox

# ---------------------------------------------------------------------------------------------------------------
def normalize_bbox(
    bboxes: np.ndarray, 
    img_h: int, 
    img_w: int) -> np.ndarray:
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / img_w
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / img_h
    return bboxes

def de_normalize_bbox(
    bboxes: np.ndarray, 
    img_h: int, 
    img_w: int) -> np.ndarray:
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * img_w
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * img_h
    return bboxes

# ---------------------------------------------------------------------------------------------------------------
def flip_image(
    image: np.ndarray, 
    bboxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    'Flip the image from left to right. The bboxes are assumed to be un-normalized'
    # handle the corner case
    if len(bboxes) == 0: bboxes = np.zeros([0, 4], dtype=np.float32)
    image = np.flip(image, axis=1).astype(image.dtype)
    bboxes = np.stack([image.shape[1] - bboxes[:, 2] , bboxes[:, 1], 
                       image.shape[1] - bboxes[:, 0] , bboxes[:, 3]], axis=-1)
    return image, bboxes

# ---------------------------------------------------------------------------------------------------------------
def resize_with_warping(
    img: np.ndarray, 
    bboxes: np.ndarray, 
    img_h: int, 
    img_w: int) -> Tuple[np.ndarray, np.ndarray]:
    'Resize/warp an image. The bboxes are assumed to be un-normalized'
    # handle the corner case
    if len(bboxes) == 0: bboxes = np.zeros([0, 4], dtype=np.float32)
    img_old_h, img_old_w = img.shape[0], img.shape[1]
    img = cv2.resize(img, (img_w, img_h), _INTERPOLATION_MODE_)
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * img_w / img_old_w
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * img_h / img_old_h
    return img, bboxes

# ---------------------------------------------------------------------------------------------------------------
def scaled_random_crop(
    img: np.ndarray, 
    bboxes: np.ndarray,
    scale: float) -> Tuple[np.ndarray, np.ndarray]:
    'randomly crop image while preserving the aspect ratio'
    # handle the corner case
    if len(bboxes) == 0: bboxes = np.zeros([0, 4], dtype=np.float32)
    img_old_h, img_old_w = img.shape[0], img.shape[1]
    img_new_h, img_new_w = int(scale*img_old_h), int(scale*img_old_w)
    xcoord_offset = np.random.randint(low=0, high=img_old_w-img_new_w)
    ycoord_offset = np.random.randint(low=0, high=img_old_h-img_new_h)
    new_img = np.full(shape=(img_new_h, img_new_w, 3), fill_value=0, dtype=img.dtype)
    new_img[:img_new_h, :img_new_w] = img[ycoord_offset:ycoord_offset+img_new_h, xcoord_offset:xcoord_offset+img_new_w]
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - xcoord_offset
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - ycoord_offset
    bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, img_new_w)
    bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, img_new_h)
    img, bboxes = resize_with_warping(new_img, bboxes, img_old_h, img_old_w)
    return img, bboxes
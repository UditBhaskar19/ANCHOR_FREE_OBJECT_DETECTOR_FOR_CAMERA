# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : mosaic augmentation functions : 1x2, 2x1, 2x2
# ---------------------------------------------------------------------------------------------------------------
import numpy as np
from typing import List, Tuple
from modules.augmentation.lib.geometric_transform import resize_with_padding, resize_with_warping
from modules.augmentation.constants_bdd import _BORDER_

# ---------------------------------------------------------------------------------------------------------------
def _create_mozaic(
    mozaic_obj, resize_function,
    image_list: List[np.ndarray], 
    bbox_list: List[np.ndarray], 
    classlabel_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    new_bbox = []
    for idx in range(len(image_list)):
        row_idx = idx // mozaic_obj.nc
        col_idx = idx % mozaic_obj.nc

        img, bboxs = resize_function(image_list[idx], bbox_list[idx].copy(), 
                                     mozaic_obj.heights[row_idx], mozaic_obj.widths[col_idx])
        mozaic_obj.new_img[mozaic_obj.y_offset[row_idx]:mozaic_obj.y_offset[row_idx] + mozaic_obj.heights[row_idx], \
                           mozaic_obj.x_offset[col_idx]:mozaic_obj.x_offset[col_idx] + mozaic_obj.widths[col_idx], :] = img

        bboxs[:, [0, 2]] = bboxs[:, [0, 2]] + mozaic_obj.x_offset[col_idx]
        bboxs[:, [1, 3]] = bboxs[:, [1, 3]] + mozaic_obj.y_offset[row_idx]
        new_bbox.append(bboxs)

    new_bbox = np.concatenate(new_bbox, axis=0)
    new_classlabel = np.concatenate(classlabel_list, axis=0)
    return mozaic_obj.new_img, new_bbox, new_classlabel

# ---------------------------------------------------------------------------------------------------------------
class mosaic():
    def __init__(self, img_h: int, img_w: int, nc: int, nr: int, border: int = _BORDER_):
        self.img_h = img_h
        self.img_w = img_w
        self.nc = nc
        self.nr = nr
        self.x_offset, self.y_offset, self.widths, self.heights = self.grid_properties()
        self.new_img = np.full(shape=(img_h, img_w, 3), fill_value=border, dtype=np.uint8)


    def grid_properties(self):
        # create x offset values for the mozaic
        intervals = self.img_w // self.nc
        x_offset = [0] + [i*intervals for i in range(1, self.nc)] + [self.img_w]
        x_offset = np.array(x_offset)

        # create y offset values for the mozaic
        intervals = self.img_h // self.nr
        y_offset = [0] + [i*intervals for i in range(1, self.nr)] + [self.img_h]
        y_offset = np.array(y_offset)

        # compute patch heights and widths
        widths = x_offset[1:] - x_offset[:-1]
        heights = y_offset[1:] - y_offset[:-1]
        return x_offset, y_offset, widths, heights

# ---------------------------------------------------------------------------------------------------------------
class mosaic_2x2(mosaic):
    def __init__(self, img_h: int, img_w: int, border: int = _BORDER_):
        super().__init__(img_h, img_w, 2, 2, border)

    def create_mozaic(
        self, 
        image_list: List[np.ndarray], 
        bbox_list: List[np.ndarray], 
        classlabel_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        new_img, new_bbox, new_classlabel = \
            _create_mozaic(self, resize_with_warping, image_list, bbox_list, classlabel_list)
        return new_img, new_bbox, new_classlabel

# ---------------------------------------------------------------------------------------------------------------
class mosaic_1x2(mosaic):
    def __init__(self, img_h: int, img_w: int, border: int = _BORDER_):
        super().__init__(img_h, img_w, 2, 1, border)

    def create_mozaic(
        self, 
        image_list: List[np.ndarray], 
        bbox_list: List[np.ndarray], 
        classlabel_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        new_img, new_bbox, new_classlabel = \
            _create_mozaic(self, resize_with_padding, image_list, bbox_list, classlabel_list)
        return new_img, new_bbox, new_classlabel
    
# ---------------------------------------------------------------------------------------------------------------
class mosaic_2x1(mosaic):
    def __init__(self, img_h: int, img_w: int, border: int = _BORDER_):
        super().__init__(img_h, img_w, 1, 2, border)

    def create_mozaic(
        self, 
        image_list: List[np.ndarray], 
        bbox_list: List[np.ndarray], 
        classlabel_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        new_img, new_bbox, new_classlabel = \
            _create_mozaic(self, resize_with_padding, image_list, bbox_list, classlabel_list)
        return new_img, new_bbox, new_classlabel
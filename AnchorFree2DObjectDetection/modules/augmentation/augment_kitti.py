# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Image Augmentation main functions
# ---------------------------------------------------------------------------------------------------------------
# import sys
import torch
from PIL import Image
import numpy as np
import cv2
from typing import Tuple, List

from modules.augmentation import constants_kitti
from modules.augmentation.lib.photometric_transform import augment_hsv, hist_equalize
from modules.augmentation.lib.geometric_transform import (
    random_perspective, geometric_transform_bbox, flip_image, scaled_random_crop )
from modules.augmentation.lib.mixup_transform import mixup
from modules.augmentation.lib.mosaic_transform import mosaic_2x2, mosaic_1x2, mosaic_2x1
from modules.augmentation.lib.dropout_transform import pixel_dropout, block_dropout, grid_dropout


# ---------------------------------------------------------------------------------------------------------------
class Augment():
    def __init__(self, img_w: int, img_h: int, prob_augment: float):
        self.img_w = img_w
        self.img_h = img_h
        self.prob_augment = prob_augment
        self.prob_geometric = constants_kitti._PROB_GEOMETRIC_ 
        self.prob_mosaic = constants_kitti._PROB_MOSAIC_ 
        self.prob_mixup = constants_kitti._PROB_MIXUP_
        self.prob_flip = constants_kitti._PROB_FLIP_
        self.prob_crop = constants_kitti._PROB_CROP_

        self.hgain = constants_kitti._H_GAIN_
        self.sgain = constants_kitti._S_GAIN_
        self.vgain = constants_kitti._V_GAIN_

        self.degree = constants_kitti._MAX_DEGREE_
        self.translate = constants_kitti._MAX_TRANSLATE_
        self.scale = constants_kitti._MAX_SCALE_
        self.shear = constants_kitti._MAX_SHEAR_
        self.perspective = constants_kitti._PERSPECTIVE_

        self.mixup = mixup(alpha=constants_kitti._MIXUP_ALPHA_, beta=constants_kitti._MIXUP_BETA_)
        self.mosaic_2x2 = mosaic_2x2(img_h, img_w)
        # self.mosaic_1x2 = mosaic_1x2(img_h, img_w)
        # self.mosaic_2x1 = mosaic_2x1(img_h, img_w)

        self.pixel_dropout = pixel_dropout(
            num_min=constants_kitti._MIN_PIXEL_DROPOUT_, num_max=constants_kitti._MAX_PIXEL_DROPOUT_)
        self.block_dropout = block_dropout(
            num_min=constants_kitti._MIN_BLOCK_DROPOUT_, num_max=constants_kitti._MAX_BLOCK_DROPOUT_, 
            h_min=constants_kitti._MIN_BLOCK_H_, w_min=constants_kitti._MIN_BLOCK_W_, 
            h_max=constants_kitti._MAX_BLOCK_H_, w_max=constants_kitti._MAX_BLOCK_W_)
        self.grid_dropout = grid_dropout(
            max_start_offset=constants_kitti._MAX_START_OFFSET_GRID_DROPOUT_, 
            h_min=constants_kitti._MIN_CELL_H_, w_min=constants_kitti._MIN_CELL_W_, 
            h_max=constants_kitti._MAX_CELL_H_, w_max=constants_kitti._MAX_CELL_W_,
            num_row_min=constants_kitti._MIN_GRID_ROWS_, num_row_max=constants_kitti._MAX_GRID_ROWS_, 
            num_col_min=constants_kitti._MIN_GRID_COLS_, num_col_max=constants_kitti._MAX_GRID_COLS_)
        

    def resize_image_bbox(
        self, 
        image: np.ndarray, 
        bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        img_height_old, img_width_old, _ = image.shape
        bbox[:, [0, 2]] *= ( self.img_w / img_width_old )
        bbox[:, [1, 3]] *= ( self.img_h / img_height_old )
        image = cv2.resize(image, (self.img_w, self.img_h), interpolation=constants_kitti._INTERPOLATION_MODE_)
        return image, bbox
        

    def extract_img_box_label(
        self, 
        dataset_gen: torch.utils.data.Dataset, 
        idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        img = np.array(Image.open(dataset_gen.selected_labels[idx]['image_path']).convert("RGB"))
        bbox = dataset_gen.selected_labels[idx]['bbox'].copy()
        if bbox.shape[0] == 0: bbox = np.zeros(shape=(0, 4), dtype=np.float32)
        img, bbox = self.resize_image_bbox(img, bbox)
        classlabels = dataset_gen.selected_labels[idx]['classid'].copy()
        return img, bbox, classlabels

        
    def create_data_list_mosaic(
        self, 
        dataset_gen: torch.utils.data.Dataset, 
        num_samples: int, 
        curr_img: np.ndarray , 
        curr_bbox: np.ndarray, 
        curr_labels: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

        sample_idx = np.random.randint(0, len(dataset_gen), num_samples)
        image_list = []
        bbox_list = []
        classlabel_list = []

        for idx in sample_idx:
            img, bbox, classlabels = self.extract_img_box_label(dataset_gen, idx)
            image_list.append(img)
            bbox_list.append(bbox)
            classlabel_list.append(classlabels)
        
        image_list.append(curr_img)
        bbox_list.append(curr_bbox)
        classlabel_list.append(curr_labels)
        return image_list, bbox_list, classlabel_list


    def perform_augmentation(
        self, 
        dataset_gen: torch.utils.data.Dataset, 
        curr_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:

        img, bbox, classlabels, is_augmented = augment_policy1(self, dataset_gen, curr_idx)
        return img, bbox, classlabels, is_augmented

# ---------------------------------------------------------------------------------------------------------------
def augment_policy1(
    aug_obj: Augment, 
    dataset_gen: torch.utils.data.Dataset, 
    curr_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:

    img, bbox, classlabels = aug_obj.extract_img_box_label(dataset_gen, curr_idx)
    is_augmented = False

    if np.random.rand() <= aug_obj.prob_augment:             # perform augmentation
        is_augmented = True

        if np.random.rand() <= aug_obj.prob_mosaic:          # perform 2 x 2 mosaic augmentation
            image_list, bbox_list, classlabel_list \
                = aug_obj.create_data_list_mosaic(dataset_gen, 3, img, bbox, classlabels)
            img, bbox, classlabels \
                = aug_obj.mosaic_2x2.create_mozaic(image_list, bbox_list, classlabel_list)

            # random crop
            if np.random.rand() <= aug_obj.prob_crop:
                scale = np.random.uniform(low=constants_kitti._SCALED_CROP_MIN_, high=constants_kitti._SCALED_CROP_MAX_)  
                img, bbox = scaled_random_crop(img, bbox, scale)   

        else:                                       
            if np.random.rand() <= aug_obj.prob_mixup:       # mixup
                idx = np.random.randint(0, len(dataset_gen))
                img2, bbox2, classlabels2 = aug_obj.extract_img_box_label(dataset_gen, idx)
                img, bbox, classlabels = aug_obj.mixup.create_mixup(img, img2, bbox, bbox2, classlabels, classlabels2)
                                                     
        img = augment_hsv(img, aug_obj.hgain, aug_obj.sgain, aug_obj.vgain)      # photometric

        if np.random.rand() <= aug_obj.prob_geometric:                   # geometric
            img, T, height, width = random_perspective(                      
                img, aug_obj.degree, aug_obj.translate, aug_obj.scale,
                aug_obj.shear, aug_obj.perspective, border = constants_kitti._BORDER_)
        
            bbox = geometric_transform_bbox(bbox, T, height, width)

        if np.random.rand() <= aug_obj.prob_flip:                        # lr flip
            img, bbox = flip_image(img, bbox)

        # if np.random.rand() <= aug_obj.prob_dropout:                    # dropout
        #     option = np.random.randint(low=1, high=4, dtype=int)
        #     if option == 1: img = aug_obj.pixel_dropout.perform_dropout(img)
        #     elif option == 2: img = aug_obj.block_dropout.perform_dropout(img)
        #     else: img = aug_obj.grid_dropout.perform_dropout(img)

    return img, bbox, classlabels, is_augmented
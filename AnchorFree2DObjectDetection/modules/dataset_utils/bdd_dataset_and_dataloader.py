# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : bdd dataset and dataloader functions for pytorch based training sample generation
# ---------------------------------------------------------------------------------------------------------------------
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Dict, Union, List
import torch
from torchvision import transforms
from modules.dataset_utils.bdd_dataset_utils.constants import (
    _BAD_BBOX_WIDTH_THR_, _BAD_BBOX_HEIGHT_THR_,
    _BAD_BBOX_ASPECT_RATIO_LOW_THR_, _BAD_BBOX_ASPECT_RATIO_HIGH_THR_,
    _INTERPOLATION_MODE_
)
from modules.augmentation.augment_bdd import Augment

# ---------------------------------------------------------------------------------------------------------------------
def extract_valid_bbox(
    gt_boxes: np.ndarray,
    gt_class: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    box_width = gt_boxes[:,2] - gt_boxes[:,0]
    box_height = gt_boxes[:,3] - gt_boxes[:,1]
    box_height = np.where(box_height > 0, box_height, 1e-10)
    box_aspect_ratio = box_width / box_height

    valid_condition = ( box_width >= _BAD_BBOX_WIDTH_THR_ ) & \
                      ( box_height >= _BAD_BBOX_HEIGHT_THR_ ) & \
                      ( box_aspect_ratio >= _BAD_BBOX_ASPECT_RATIO_LOW_THR_ ) & \
                      ( box_aspect_ratio <= _BAD_BBOX_ASPECT_RATIO_HIGH_THR_) 
                    
    return gt_boxes[valid_condition], gt_class[valid_condition]

# ---------------------------------------------------------------------------------------------------------------------
""" if the input image is in numpy array format (RGB with array shape (H, W, C) with each pixel value in the range [0, 255] of data type uint8),
    the data needs to be converted to pytorch tensor form with shape (C, H, W) with each pixel value in the range [0, 1] of data type float32.
    In order to ensure the input image data is compatible with the pretrained classification models, the image is scaled in the range [0, 1], and 
    then normalized with mean = [0.485, 0.456, 0.406] & std =  [0.229, 0.224, 0.225]
"""
preprocess = transforms.Compose([
    transforms.ToTensor(),                 # scale in [0, 1]
    transforms.Normalize(                  # normalize with mean = [0.485, 0.456, 0.406] & std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406], 
        std =  [0.229, 0.224, 0.225])
    ])

""" Define an "inverse" transform for the image that un-normalizes by ImageNet
    color. Without this, the images will NOT be visually understandable.
"""
inverse_norm = transforms.Compose([
    transforms.Normalize(
        mean=[0.0, 0.0, 0.0], 
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.Normalize(
        mean=[-0.485, -0.456, -0.406], 
        std=[1.0, 1.0, 1.0])])

# ---------------------------------------------------------------------------------------------------------------------
class BerkeleyDeepDriveDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        gt_labels,
        image_new_shape: Tuple[int, int, int],  # (num_channels, new_height, new_width)
        device: str,
        subset: int = -1,
        shuffle_dataset: bool = False,
        augment: bool = False,
        prob_augment: float = 0.0):
        
        self.device = device
        self.num_channels, self.img_h, self.img_w = image_new_shape
        self.selected_labels = gt_labels

        if shuffle_dataset:
            random_idx = np.arange(len(self.selected_labels))
            np.random.shuffle(random_idx)
            self.selected_labels = [self.selected_labels[idx] for idx in random_idx]
        if subset > 0:
            self.selected_labels = self.selected_labels[:subset]

        self.dataset_len = len(self.selected_labels)
        self.augment = augment
        self.Augment = Augment(self.img_w, self.img_h, prob_augment)

    
    def set_prob_augment(self, prob_augment):
        self.Augment.prob_augment = prob_augment


    def resize_image_bbox_np(
        self, image: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img_height_old, img_width_old, _ = image.shape
        bbox[:, [0, 2]] *= ( self.img_w / img_width_old )
        bbox[:, [1, 3]] *= ( self.img_h / img_height_old )
        image = cv2.resize(image, (self.img_w, self.img_h), interpolation=_INTERPOLATION_MODE_)
        return image, bbox
    

    def extract_img_box_label(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, int]:
        img_path = self.selected_labels[idx]['img_path']
        img = np.array(Image.open(img_path).convert("RGB"))
        bbox = self.selected_labels[idx]['boundingBox2D'].copy()
        if bbox.shape[0] == 0: bbox = np.zeros(shape=(0, 4), dtype=np.float32)
        img, bbox = self.resize_image_bbox_np(img, bbox)
        classlabels = self.selected_labels[idx]['objCategoryid'].copy()
        weatherid = self.selected_labels[idx]['weatherid']
        return img, bbox, classlabels, img_path, weatherid
    

    def __len__(self):
        return self.dataset_len
    

    def __getitem__(self, idx: int) \
        -> Tuple[torch.Tensor, Dict[str, Union[str, int, torch.Tensor]]]:

        if self.augment: 
            img, bbox, classlabels, is_augmented = self.Augment.perform_augmentation(self, idx)
            if is_augmented : 
                img_path = 'Augmented Image'
                weatherid = -1
            else: 
                img_path = self.selected_labels[idx]['img_path']
                weatherid = self.selected_labels[idx]['weatherid']
        else: 
            img, bbox, classlabels, \
            img_path, weatherid = self.extract_img_box_label(idx)
            
        img = preprocess(img)
        num_objects = classlabels.shape[0]
        labels = {}

        labels['img_path'] = img_path
        labels['weather_class_label'] = weatherid
        if num_objects > 0:
            bbox, gtcls = extract_valid_bbox(bbox, classlabels)
            labels['bbox'] = torch.Tensor(bbox)
            labels['obj_class_label'] = torch.Tensor(gtcls)
        else:
            labels['bbox'] = torch.zeros([0, 4], dtype=torch.float32)
            labels['obj_class_label'] = torch.zeros([0,], dtype=torch.float32)
        
        return img, labels

    
    def collate_fn(self, sample_batch):
        img_batch, labels_batch = bdd_collate_fn(self.device, sample_batch)
        return img_batch, labels_batch

# ---------------------------------------------------------------------------------------------------------------------
def bdd_collate_fn(
    device: str, 
    sample_batch: List[Tuple[torch.Tensor, Dict[str, Union[str, int, torch.Tensor]]]]) \
        -> Tuple[torch.Tensor, Dict[str, Union[ List[str], torch.Tensor, List[torch.Tensor] ]]]:
    """ Generate a batch of data """
    img_batch = []
    img_path_batch = []
    bbox_batch = []
    obj_class_label = []
    weather_class_label = []
    labels_batch = {}
    
    for i in range(len(sample_batch)):
        img, labels = sample_batch[i]
        img_batch.append(img.to(device))
        img_path_batch.append(labels['img_path'])
        bbox_batch.append(labels['bbox'].to(device))
        obj_class_label.append(labels['obj_class_label'].to(device))
        weather_class_label.append(labels['weather_class_label'])
        
    img_batch = torch.stack(img_batch, dim=0)
    labels_batch['img_path'] = img_path_batch
    labels_batch['weather_class_label'] =  torch.tensor(weather_class_label).to(device)
    labels_batch['bbox_batch'] =  bbox_batch
    labels_batch['obj_class_label'] =  obj_class_label
    return img_batch, labels_batch
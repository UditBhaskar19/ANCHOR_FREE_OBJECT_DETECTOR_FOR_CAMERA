# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : dataset summary utilities : functions to perform basic data analysis
# ---------------------------------------------------------------------------------------------------------------------
import numpy as np
from typing import List, Dict, Union, Tuple
from modules.dataset_utils.kitti_dataset_utils.constants import _NEW_OBJ_LABELS_ as obj_labels

# ---------------------------------------------------------------------------------------------------------------------
def class_labels_summary(aggregated_labels):
    # count the number of instances in each catagories
    obj_catagory_counter = {key: 0 for key in obj_labels}
    for label in aggregated_labels:
        for item in label['type']: obj_catagory_counter[item] += 1
    return {'obj_catagory_counter': obj_catagory_counter}

# ---------------------------------------------------------------------------------------------------------------------
def obj_box_summary(aggregated_labels):
    # bbox data analysis
    obj_category_bboxes = {key: [] for key in obj_labels}
    obj_category_bboxes_h = {key: [] for key in obj_labels}
    obj_category_bboxes_w = {key: [] for key in obj_labels}
    obj_category_bboxes_area = {key: [] for key in obj_labels}
    obj_category_bboxes_aspect_ratio = {key: [] for key in obj_labels}
    image_paths = { key: [] for key in obj_labels }
    obj_class = { key: [] for key in obj_labels }

    for label in aggregated_labels:
        for idx, item_type in enumerate(label['type']):
            if label['bbox'][idx].shape[0] > 0:
                obj_category_bboxes[item_type].append(label['bbox'][idx])
                image_paths[item_type].append(label['image_path'])
                obj_class[item_type].append(item_type)

    for key, item in obj_category_bboxes.items():
        obj_category_bboxes[key] = np.stack(item, axis=0)
        w = obj_category_bboxes[key][:,2] - obj_category_bboxes[key][:,0]
        h = obj_category_bboxes[key][:,3] - obj_category_bboxes[key][:,1]
        obj_category_bboxes_h[key] = h
        obj_category_bboxes_w[key] = w
        obj_category_bboxes_area[key] = w * h
        obj_category_bboxes_aspect_ratio[key] = w / h

    return {
        'obj_category_bboxes': obj_category_bboxes,
        'obj_category_bboxes_h': obj_category_bboxes_h,
        'obj_category_bboxes_w': obj_category_bboxes_w,
        'obj_category_bboxes_area': obj_category_bboxes_area,
        'obj_category_bboxes_aspect_ratio': obj_category_bboxes_aspect_ratio,
        'image_paths': image_paths,
        'obj_class': obj_class
    }

# ---------------------------------------------------------------------------------------------------------------------
def aggregated_bboxes(bbox_summary):
    boxes_all = []
    image_names_all = []
    obj_class_all = []
    for objecttype in obj_labels:
        boxes_all += [bbox_summary['obj_category_bboxes'][objecttype]]
        image_names_all += bbox_summary['image_paths'][objecttype]
        obj_class_all += bbox_summary['obj_class'][objecttype]
    boxes_all = np.concatenate(boxes_all, axis=0)

    w_all = boxes_all[:, 2] - boxes_all[:, 0]
    h_all = boxes_all[:, 3] - boxes_all[:, 1]
    area_all = w_all * h_all 
    aspect_ratio_all =  w_all / h_all 
    return boxes_all, obj_class_all, area_all, aspect_ratio_all, image_names_all

# ---------------------------------------------------------------------------------------------------------------------
def sort_according_to_box_criteria(
    boxes: List[np.ndarray], 
    obj_class: List[str], 
    box_area: List[float], 
    box_aspect_ratio: List[float], 
    image_names: List[str], 
    sorting_criteria: str='box_area', 
    order: str='ascending')\
        -> Tuple[List[np.ndarray], 
                 List[str], 
                 List[float], 
                 List[float],  
                 List[str]]: 

    if sorting_criteria == 'box_area': sorted_data = box_area
    elif sorting_criteria == 'box_aspect_ratio': sorted_data = box_aspect_ratio
    else: raise Exception("wrong sorting criteria selected")

    if order == 'ascending': sorted_idx = np.argsort(sorted_data)
    else: sorted_idx = np.argsort(sorted_data)[::-1]

    box_area = box_area[sorted_idx]
    boxes = boxes[sorted_idx]
    box_aspect_ratio = box_aspect_ratio[sorted_idx]
    image_names = [image_names[idx] for idx in sorted_idx]
    obj_class = [obj_class[idx] for idx in sorted_idx]
    return boxes, obj_class, box_area, box_aspect_ratio, image_names
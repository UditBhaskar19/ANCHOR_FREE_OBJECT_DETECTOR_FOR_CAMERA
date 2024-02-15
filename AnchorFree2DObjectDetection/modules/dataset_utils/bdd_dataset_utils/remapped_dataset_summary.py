# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : remapped bdd utils : functions to perform basic data analysis for remapped dataset
# ---------------------------------------------------------------------------------------------------------------------
import numpy as np
from typing import Dict, Union, List, Tuple
from modules.dataset_utils.bdd_dataset_utils.constants import (
    _ALL_WEATHER_, _NEW_OBJ_LABELS_,
    _BAD_BBOX_WIDTH_THR_, _BAD_BBOX_HEIGHT_THR_,
    _BAD_BBOX_ASPECT_RATIO_HIGH_THR_, _BAD_BBOX_ASPECT_RATIO_LOW_THR_)

# ---------------------------------------------------------------------------------------------------------------------
def class_labels_summary(
    aggregated_labels: List[Dict[str, Union[str, int, np.ndarray]]])\
        -> Dict[str, Dict[str, int]]:
    " count the number of instances in each catagories "
    obj_catagory_counter = {key: 0 for key in _NEW_OBJ_LABELS_}
    weather_label_counter = {key: 0 for key in _ALL_WEATHER_}
    for label in aggregated_labels:
        for idx, item in enumerate(label['objCategory']): 
            bbox = label['boundingBox2D'][idx]
            bbox_width  = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            bbox_aspect_ratio = bbox_width / bbox_height
            condition = ( bbox_width >= _BAD_BBOX_WIDTH_THR_ ) and \
                        ( bbox_height >= _BAD_BBOX_HEIGHT_THR_ ) and \
                        ( bbox_aspect_ratio >= _BAD_BBOX_ASPECT_RATIO_LOW_THR_ ) and \
                        ( bbox_aspect_ratio <= _BAD_BBOX_ASPECT_RATIO_HIGH_THR_ )
            if condition:
                obj_catagory_counter[item] += 1
        weather_label_counter[label['weather']] += 1
    return {
        'obj_catagory_counter': obj_catagory_counter,
        'weather_label_counter': weather_label_counter
    }

# ---------------------------------------------------------------------------------------------------------------------
def obj_box_summary(
    aggregated_labels: List[Dict[str, Union[str, int, np.ndarray]]])\
        -> Dict[str, Union[ Dict[str, List[np.ndarray]], Dict[str, List[float]], Dict[str, List[str]] ]]:
    " create a dictionary of aggregated bbox and other box attributes " 
    obj_category_bboxes = {key: [] for key in _NEW_OBJ_LABELS_}
    obj_category_bboxes_h = {key: [] for key in _NEW_OBJ_LABELS_}
    obj_category_bboxes_w = {key: [] for key in _NEW_OBJ_LABELS_}
    obj_category_bboxes_area = {key: [] for key in _NEW_OBJ_LABELS_}
    obj_category_bboxes_aspect_ratio = {key: [] for key in _NEW_OBJ_LABELS_}
    image_paths = { key: [] for key in _NEW_OBJ_LABELS_ }
    obj_class = { key: [] for key in _NEW_OBJ_LABELS_ }

    for label in aggregated_labels:
        for idx, item_type in enumerate(label['objCategory']):
            bbox_width  = label['boundingBox2D'][idx, 2] - label['boundingBox2D'][idx, 0]
            bbox_height = label['boundingBox2D'][idx, 3] - label['boundingBox2D'][idx, 1]
            bbox_aspect_ratio = bbox_width / bbox_height
            condition = ( bbox_width >= _BAD_BBOX_WIDTH_THR_ ) and \
                        ( bbox_height >= _BAD_BBOX_HEIGHT_THR_ ) and \
                        ( bbox_aspect_ratio >= _BAD_BBOX_ASPECT_RATIO_LOW_THR_ ) and \
                        ( bbox_aspect_ratio <= _BAD_BBOX_ASPECT_RATIO_HIGH_THR_ )
            if condition:
                obj_category_bboxes[item_type].append(label['boundingBox2D'][idx])
                image_paths[item_type].append(label['img_path'])
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
def aggregated_bboxes(
    bbox_summary: Dict[str, Union[ Dict[str, List[np.ndarray]], Dict[str, List[float]], Dict[str, List[str]] ]]) \
        -> Tuple[List[np.ndarray], List[str], List[float], List[float], List[str]]:
    " create a sorted list of aggregated bbox and other box attributes "

    boxes_all = []
    image_names_all = []
    obj_class_all = []
    for objecttype in bbox_summary['obj_category_bboxes'].keys():
        boxes_all += [bbox_summary['obj_category_bboxes'][objecttype]]
        image_names_all += bbox_summary['image_paths'][objecttype]
        obj_class_all += bbox_summary['obj_class'][objecttype]
    boxes_all = np.concatenate(boxes_all, axis=0)
    
    w_all = boxes_all[:, 2] - boxes_all[:, 0]
    h_all = boxes_all[:, 3] - boxes_all[:, 1]
    area_all = w_all * h_all 
    aspect_ratio_all =  w_all / h_all 
    return boxes_all, obj_class_all, area_all, aspect_ratio_all, image_names_all
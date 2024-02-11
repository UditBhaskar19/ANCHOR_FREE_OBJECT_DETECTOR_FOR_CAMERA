# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : dataset summary utilities : functions to perform basic data analysis
# ---------------------------------------------------------------------------------------------------------------------
import numpy as np
from typing import List, Dict, Union, Tuple
from modules.dataset_utils.bdd_dataset_utils.constants import (
    _ALL_OBJ_LABELS_,
    _ALL_TRAFFIC_LIGHT_LABELS_,
    _ALL_WEATHER_,
    _ALL_SCENE_,
    _ALL_TimeOfDay_,
    _ALL_LaneDirection_Labels_,
    _ALL_LaneStyle_Labels_,
    _ALL_LaneType_Labels_
)

# ---------------------------------------------------------------------------------------------------------------------
def class_labels_summary(
    aggregated_labels: List[Dict[str, Union[str, List[str], np.ndarray, List[np.ndarray]]]]) -> Dict[str, int]:

    # count the number of instances in each catagories
    obj_catagory_counter = {key: 0 for key in _ALL_OBJ_LABELS_}
    traffic_light_label_counter = {key: 0 for key in _ALL_TRAFFIC_LIGHT_LABELS_}

    weather_label_counter = {key: 0 for key in _ALL_WEATHER_}
    scene_label_counter = {key: 0 for key in _ALL_SCENE_}
    timeofday_label_counter = {key: 0 for key in _ALL_TimeOfDay_}

    laneDirection_label_counter = {key: 0 for key in _ALL_LaneDirection_Labels_}
    laneStyle_label_counter = {key: 0 for key in _ALL_LaneStyle_Labels_}
    laneType_label_counter = {key: 0 for key in _ALL_LaneType_Labels_}

    for label in aggregated_labels:

        for item in label['objCategory']: obj_catagory_counter[item] += 1
        for item in label['trafficLight']: 
            if item != 'none': traffic_light_label_counter[item] += 1
        for item in label['laneDirections']: laneDirection_label_counter[item] += 1
        for item in label['laneStyles']: laneStyle_label_counter[item] += 1
        for item in label['laneTypes']: laneType_label_counter[item] += 1

        weather_label_counter[label['weather']] += 1
        scene_label_counter[label['scene']] += 1
        timeofday_label_counter[label['timeofday']] += 1

    return {
        'obj_catagory_counter': obj_catagory_counter,
        'traffic_light_label_counter': traffic_light_label_counter,
        'laneDirection_label_counter': laneDirection_label_counter,
        'laneStyle_label_counter': laneStyle_label_counter,
        'laneType_label_counter': laneType_label_counter,
        'weather_label_counter': weather_label_counter,
        'scene_label_counter': scene_label_counter,
        'timeofday_label_counter': timeofday_label_counter
    }

# ---------------------------------------------------------------------------------------------------------------------
def obj_box_summary(
    aggregated_labels: List[Dict[str, Union[str, List[str], np.ndarray, List[np.ndarray]]]]) \
        -> Dict[str, Union[ Dict[str, List[str]], Dict[str, List[float]], Dict[str, List[np.ndarray]] ]]:

    obj_category_bboxes = {key: [] for key in _ALL_OBJ_LABELS_}
    obj_category_bboxes_h = {key: [] for key in _ALL_OBJ_LABELS_}
    obj_category_bboxes_w = {key: [] for key in _ALL_OBJ_LABELS_}
    obj_category_bboxes_area = {key: [] for key in _ALL_OBJ_LABELS_}
    obj_category_bboxes_aspect_ratio = {key: [] for key in _ALL_OBJ_LABELS_}
    image_paths = { key: [] for key in _ALL_OBJ_LABELS_ }
    obj_class = { key: [] for key in _ALL_OBJ_LABELS_ }

    for label in aggregated_labels:
        for idx, item_type in enumerate(label['objCategory']):
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
    bbox_summary: Dict[str, Union[ Dict[str, List[str]], Dict[str, List[float]], Dict[str, List[np.ndarray]] ]])\
        -> Tuple[List[np.ndarray], 
                 List[str], 
                 List[float], 
                 List[float], 
                 List[str]] :
    
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
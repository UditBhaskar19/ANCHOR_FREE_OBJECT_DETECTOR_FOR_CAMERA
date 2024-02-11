# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : remapped bdd utils : functions to rename/remap labels and aggregate the remapped ground truths
# ---------------------------------------------------------------------------------------------------------------------
import numpy as np
import json, os
from typing import List, Dict, Union
from modules.dataset_utils.bdd_dataset_utils.constants import (
    _OBJ_LABELS_CONSIDERED_, _LABEL_MAP_OLD_TO_NEW_, 
    _OBJ_CLASS_TO_IDX_, _WEATHER_CLASS_TO_IDX_)

# ---------------------------------------------------------------------------------------------------------------------
""" Save the ground-truth file """
def aggregate_and_save_ground_truths(
    label_rootdir: str,
    label_out_path : str,
    dataset_rootdir: str,
    label_file_path: str):

    print('Load JSON file .. please wait')
    label_file_path = os.path.join(dataset_rootdir, label_file_path)
    with open(label_file_path, 'r') as file: all_data = json.load(file)

    # selected in the sense that not all attributes are considered
    selected_labels = []   

    for _, data in enumerate(all_data):   # for each image
        objCategory = []         # object class
        boundingBox2D = []       # object box atrributes
        
        # extract the object category and bounding box info
        for label in data['labels']:
            if label['category'] in _OBJ_LABELS_CONSIDERED_:
                objCategory.append(label['category'])
                boundingBox2D.append([ label['box2d']['x1'], label['box2d']['y1'], 
                                       label['box2d']['x2'], label['box2d']['y2'] ])
                    
        objCategory = [_LABEL_MAP_OLD_TO_NEW_[label] for label in objCategory]  # set new class names to objects

        # class names to class ids
        objCategory_id = [_OBJ_CLASS_TO_IDX_[label] for label in objCategory]
        weather_id = _WEATHER_CLASS_TO_IDX_[data['attributes']['weather']]

        selected_labels.append( {
            'name': data['name'],
            'weather' : data['attributes']['weather'],
            'weather_id': weather_id,
            'objCategory': objCategory,
            'objCategory_id': objCategory_id, 
            'boundingBox2D': boundingBox2D
        } )

    # save the file
    label_out_path = os.path.join(label_rootdir, label_out_path)
    with open(label_out_path, 'w') as json_file: 
        json.dump(selected_labels, json_file, indent=4)
    print(f'Labels saved in : {label_out_path}')

# ---------------------------------------------------------------------------------------------------------------------
""" load the ground-truth file """
def load_ground_truths(
    label_rootdir: str,
    label_file_path: str, 
    dataset_rootdir: str,
    img_dir: str, 
    verbose: bool=True)\
        -> List[Dict[str, Union[str, int, np.ndarray]]]:
    """ The label file is a list of dict (JSON) and is structured as fllows:
        - name : image file name
        - weather : weather type (string)
        - weather_id : weather type id (numeric)
        - objCategory : object type (string)
        - objCategory_id : object type id (numeric)
        - boundingBox2D : a list of list of bounding box corner coordinates (x1,y1,x2,y2)
    """
    print('Load JSON file .. please wait')
    label_file_path = os.path.join(label_rootdir, label_file_path)
    with open(label_file_path, 'r') as file: all_data = json.load(file)
    selected_labels = []

    for i, data in enumerate(all_data):   # for each image
        selected_labels.append( { 
            'img_path' : os.path.join(dataset_rootdir, img_dir, data['name']),
            'weather' : data['weather'],
            'weatherid' : data['weather_id'],
            'objCategory': data['objCategory'],
            'objCategoryid': np.array(data['objCategory_id'], dtype=np.float32),
            'boundingBox2D' : np.array(data['boundingBox2D'], dtype=np.float32) })
        
        if verbose == True:
            if i % 2000 == 0 : print(f'annotations from {i+1}/{len(all_data)} aggregated')
    print(f'annotations from {len(all_data)}/{len(all_data)} aggregated : Aggregation COMPLETE')
    return selected_labels
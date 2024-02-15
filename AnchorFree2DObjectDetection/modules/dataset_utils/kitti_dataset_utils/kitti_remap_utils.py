# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : remapped bdd utils : functions to rename/remap labels and aggregate the remapped ground truths
# ---------------------------------------------------------------------------------------------------------------------
import numpy as np
import json, os
from typing import List, Dict, Union
from modules.dataset_utils.kitti_dataset_utils.kitti_file_utils import create_poses
from modules.dataset_utils.kitti_dataset_utils.constants import (
    _OBJ_LABELS_CONSIDERED_, _TRUNCATED_PROP_CONSIDERED_, _OCCLUDED_PROP_CONSIDERED_,
    _LABEL_MAP_OLD_TO_NEW_, _OBJ_CLASS_TO_IDX_)

# ---------------------------------------------------------------------------------------------------------------------
def selection_condition1(label):
    return label in _OBJ_LABELS_CONSIDERED_

def selection_condition2(label, truncated):
    return ( label in _OBJ_LABELS_CONSIDERED_ ) & \
           ( truncated in _TRUNCATED_PROP_CONSIDERED_ )

def selection_condition3(label, occluded):
    return ( label in _OBJ_LABELS_CONSIDERED_ ) & \
           ( occluded in _OCCLUDED_PROP_CONSIDERED_ )

def selection_condition4(label, truncated, occluded):
    return ( label in _OBJ_LABELS_CONSIDERED_ ) & \
           ( truncated in _TRUNCATED_PROP_CONSIDERED_ ) & \
           ( occluded in _OCCLUDED_PROP_CONSIDERED_ )

# ---------------------------------------------------------------------------------------------------------------------
def modify_annotations(seq_annotations, scene):
    # modify class label
    annotations_list = []
    for annotations in seq_annotations:
        annotations_dict = {}
        annotations_dict['scene'] = scene
        annotations_dict['image_path'] = annotations['image_path']
        annotations_dict['segmentation_image_path'] = annotations['segmentation_image_path']
        annotations_dict['classid'] = []
        annotations_dict['type'] = []
        annotations_dict['truncated'] = []
        annotations_dict['occluded'] = []
        annotations_dict['alpha'] = []
        annotations_dict['bbox'] = []
        annotations_dict['dimensions'] = []
        annotations_dict['location'] = []
        annotations_dict['rotation_y'] = []
        
        for idx, label in enumerate(annotations['type']):
            if selection_condition1(label):
            # if selection_condition2(label, annotations['truncated'][idx]):
            # if selection_condition3(label, annotations['occluded'][idx]):
            # if selection_condition4(label, annotations['truncated'][idx], annotations['occluded'][idx]):
                newlabel = _LABEL_MAP_OLD_TO_NEW_[label]
                annotations_dict['classid'].append(_OBJ_CLASS_TO_IDX_[newlabel])
                annotations_dict['type'].append(_LABEL_MAP_OLD_TO_NEW_[label])
                annotations_dict['truncated'].append(annotations['truncated'][idx])
                annotations_dict['occluded'].append(annotations['occluded'][idx])
                annotations_dict['alpha'].append(annotations['alpha'][idx])
                annotations_dict['bbox'].append(annotations['bbox'][idx])
                annotations_dict['dimensions'].append(annotations['dimensions'][idx])
                annotations_dict['location'].append(annotations['location'][idx])
                annotations_dict['rotation_y'].append(annotations['rotation_y'][idx])
        annotations_list.append(annotations_dict)
    return annotations_list

# ---------------------------------------------------------------------------------------------------------------------
def aggregate_and_save_remapped_groundtruths_json(
    label_rootdir: str,
    label_file_path: str, 
    label_out_path: str):

    # load file
    print('Loading JSON file .. please wait')
    label_file_path = os.path.join(label_rootdir, label_file_path)
    with open(label_file_path, 'r') as file: 
        all_data = json.load(file)

    # modify annotations
    for scene in all_data.keys():
        all_data[scene]['annotations'] = modify_annotations(all_data[scene]['annotations'], scene) 

    # save the file
    label_out_path = os.path.join(label_rootdir, label_out_path)
    with open(label_out_path, 'w') as json_file: 
        json.dump(all_data, json_file, indent=4)
    print(f'Labels saved in : {label_out_path}')

# ---------------------------------------------------------------------------------------------------------------------
def parse_sequence_groundtruth(seq_dataset, scene, dataset_rootdir):
    # get annotations, calibrations and ego pose data from the dictionary
    seq_annotations = seq_dataset['annotations']
    calibrations = seq_dataset['calibrations']
    oxts = seq_dataset['oxts']

    # ego vehicle poses
    poses_list = create_poses(oxts)

    # calibrations
    for key, value in calibrations.items():
        calibrations[key] = np.array(value)
    calibrations_list = [calibrations] * len(poses_list)

    # annotations as a list of dict
    annotations_list = []
    for annotations in seq_annotations:
        annotations_dict = {}
        annotations_dict['scene'] = scene
        annotations_dict['image_path'] = os.path.join(dataset_rootdir, annotations['image_path']) 
        annotations_dict['segmentation_image_path'] \
            = os.path.join(dataset_rootdir, annotations['segmentation_image_path'])
        annotations_dict['type'] = annotations['type']
        annotations_dict['classid'] = np.array(annotations['classid'], dtype=np.int16)
        annotations_dict['truncated'] = np.array(annotations['truncated'], dtype=np.int16)
        annotations_dict['occluded'] = np.array(annotations['occluded'], dtype=np.int16)
        annotations_dict['alpha'] = np.array(annotations['alpha'], dtype=np.float32)
        annotations_dict['bbox']= np.array(annotations['bbox'], dtype=np.float32)
        annotations_dict['dimensions'] = np.array(annotations['dimensions'], dtype=np.float32)
        annotations_dict['location'] = np.array(annotations['location'], dtype=np.float32)
        annotations_dict['rotation_y'] = np.array(annotations['rotation_y'], dtype=np.float32)
        annotations_list.append(annotations_dict)
    return annotations_list, calibrations_list, poses_list

# ---------------------------------------------------------------------------------------------------------------------
def load_specific_sequence_groundtruths_json(
    sequence, 
    aggregated_label_path, 
    label_rootdir, 
    dataset_rootdir):

    print('Loading JSON file .. please wait')
    aggregated_label_path = os.path.join(label_rootdir, aggregated_label_path)
    with open(aggregated_label_path, 'r') as file: all_data = json.load(file)
    annotations_list, calibrations_list, poses_list \
        = parse_sequence_groundtruth(all_data[sequence], sequence, dataset_rootdir)
    return annotations_list, calibrations_list, poses_list

# ---------------------------------------------------------------------------------------------------------------------
def load_all_sequence_groundtruths_json(
    sequences, 
    aggregated_label_path, 
    label_rootdir, 
    dataset_rootdir):

    print('Loading JSON file .. please wait')
    aggregated_label_path = os.path.join(label_rootdir, aggregated_label_path)
    with open(aggregated_label_path, 'r') as file: 
        all_data = json.load(file)
    
    AnnotationsList = []
    CalibrationsList = []
    PosesList = []
    for sequence in sequences:
        print(f'Sequence: {sequence}')
        annotations_list, calibrations_list, poses_list \
            = parse_sequence_groundtruth(all_data[sequence], sequence, dataset_rootdir)
        AnnotationsList += annotations_list
        CalibrationsList += calibrations_list
        PosesList += poses_list
    return AnnotationsList, CalibrationsList, PosesList
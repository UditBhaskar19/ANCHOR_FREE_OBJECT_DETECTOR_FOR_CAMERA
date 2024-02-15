# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : kitti object tracking dataset utilities
# ---------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import json
import config_dataset
from modules.dataset_utils.kitti_dataset_utils.kitti_math_utils import (
    latlonToMercator, latToScale, create_pose_matrix)

img_ext = '.png'
txt_ext = '.txt'

# ---------------------------------------------------------------------------------------------------------------------
def parse_labels_line(line):
    annotation = {}
    entries = line.strip().split()
    annotation['frame'] = int(entries[0])
    annotation['trackid'] = int(entries[1])
    annotation['type'] = entries[2]
    annotation['truncated'] = int(entries[3])
    annotation['occluded'] = int(entries[4])
    annotation['alpha'] = float(entries[5])
    annotation['bbox'] = [float(x) for x in entries[6:10]]
    annotation['dimensions'] = [float(x) for x in entries[10:13]]
    annotation['location'] = [float(x) for x in entries[13:16]]
    annotation['rotation_y'] = float(entries[16])
    annotation['file_name'] = "{:06d}".format(annotation['frame']) + img_ext
    return annotation

# ---------------------------------------------------------------------------------------------------------------------
def rearrange_annotation_struct(
    annotations, 
    image_path, 
    segmentation_image_path):
    frame = []
    trackid = []
    type = []
    truncated = []
    occluded = []
    alpha = []
    bbox = []
    dimensions = []
    location = []
    rotation_y = []
    file_name = []
    annotations_dict = {}
    
    for ann in annotations:
        frame.append(ann['frame'])
        trackid.append(ann['trackid'])
        type.append(ann['type'])
        truncated.append(ann['truncated'])
        occluded.append(ann['occluded'])
        alpha.append(ann['alpha'])
        bbox.append(ann['bbox'])
        dimensions.append(ann['dimensions'])
        location.append(ann['location'])
        rotation_y.append(ann['rotation_y'])
        file_name.append(ann['file_name'])

    annotations_dict['image_path'] = image_path
    annotations_dict['segmentation_image_path'] = segmentation_image_path
    annotations_dict['frame'] = frame
    annotations_dict['trackid'] = trackid
    annotations_dict['type'] = type
    annotations_dict['truncated'] = truncated
    annotations_dict['occluded'] = occluded
    annotations_dict['alpha'] = alpha
    annotations_dict['bbox']= bbox
    annotations_dict['dimensions'] = dimensions
    annotations_dict['location'] = location
    annotations_dict['rotation_y'] = rotation_y
    annotations_dict['file_name'] = file_name
    return annotations_dict

# ---------------------------------------------------------------------------------------------------------------------
def parse_labels_file(
    sequence, 
    annotation_dir, 
    image_dir, seg_dir, 
    rearrange_data, 
    dataset_rootdir):

    # compute the number of images in the sequence
    image_seq_dir = os.path.join(dataset_rootdir, image_dir, sequence)
    num_frames = len([fname for fname in os.listdir(image_seq_dir) if fname.endswith(img_ext)])

    # aggregate annotations from the txt file
    annotations = []
    annotation_file = os.path.join(dataset_rootdir, annotation_dir, sequence + txt_ext)
    with open(annotation_file, 'r') as file:
        for line in file:
            annotation = parse_labels_line(line)
            annotations.append(annotation)

    # create image-wise annotation in the form of a list
    annotation_list_temp = []
    for frameid in range(num_frames):
        file_name = "{:06d}".format(frameid) + img_ext
        annotation = {}
        annotation['data'] = []
        annotation['image_path'] = os.path.join(image_dir, sequence, file_name)
        annotation['segmentation_image_path'] = os.path.join(seg_dir, sequence, file_name)
        annotation_list_temp.append(annotation)

    for annotation in annotations:
        frame = annotation['frame']
        annotation_list_temp[frame]['data'] += [ annotation ]
    annotation_list = annotation_list_temp

    # rearrange the annotation structure if required
    rearranged_annotation_list = []
    if rearrange_data:
        for frameid in range(num_frames):
            annotations_dict = rearrange_annotation_struct(
                annotation_list_temp[frameid]['data'], 
                annotation_list_temp[frameid]['image_path'],
                annotation_list_temp[frameid]['segmentation_image_path'])
            rearranged_annotation_list.append(annotations_dict)
        annotation_list = rearranged_annotation_list
    return annotation_list

# ---------------------------------------------------------------------------------------------------------------------
def parse_calib_file(sequence, calibration_dir, dataset_rootdir):
    calib_dict = {}
    calib_file = os.path.join(dataset_rootdir, calibration_dir, sequence + txt_ext)
    with open(calib_file, 'r') as file:
        for line in file:
            entries = line.strip().split()
            calib_type = entries[0].replace(':', '')
            calib_data = entries[1:]
            if calib_type == 'R_rect': calib_data = [ calib_data[:3], calib_data[3:6], calib_data[6:] ]
            else: calib_data = [ calib_data[:3], calib_data[3:6], calib_data[6:9], calib_data[9:] ]
            calib_dict[calib_type] = calib_data
    return calib_dict

# ---------------------------------------------------------------------------------------------------------------------
def parse_oxts_line(line):
    oxts = {}
    entries = line.strip().split()
    oxts['lat'] = float(entries[0])
    oxts['lon'] = float(entries[1])
    oxts['alt'] = float(entries[2])

    oxts['roll'] = float(entries[3])
    oxts['pitch'] = float(entries[4])
    oxts['yaw'] = float(entries[5])

    oxts['vn'] = float(entries[6])
    oxts['ve'] = float(entries[7])
    oxts['vf'] = float(entries[8])
    oxts['vl'] = float(entries[9])
    oxts['vu'] = float(entries[10])

    oxts['ax'] = float(entries[11])
    oxts['ay'] = float(entries[12])
    oxts['az'] = float(entries[13])
    oxts['af'] = float(entries[14])
    oxts['al'] = float(entries[15])
    oxts['au'] = float(entries[16])

    oxts['wx'] = float(entries[17])
    oxts['wy'] = float(entries[18])
    oxts['wz'] = float(entries[19])
    oxts['wf'] = float(entries[20])
    oxts['wl'] = float(entries[21])
    oxts['wu'] = float(entries[22])

    oxts['posacc'] = float(entries[23])
    oxts['velacc'] = float(entries[24])

    oxts['navstat'] = float(entries[25])
    oxts['numsats'] = float(entries[26])
    oxts['posmode'] = float(entries[27])
    oxts['velmode'] = float(entries[28])
    oxts['orimode'] = float(entries[29])
    return oxts

# ---------------------------------------------------------------------------------------------------------------------
def parse_oxts_file(sequence, oxts_dir, dataset_rootdir):
    oxts_list = []
    oxts_file = os.path.join(dataset_rootdir, oxts_dir, sequence + txt_ext)
    with open(oxts_file, 'r') as file:
        for line in file:
            oxts = parse_oxts_line(line)
            scale = latToScale(oxts['lat'])
            oxts['tx'], oxts['ty'] = latlonToMercator(oxts['lat'], oxts['lon'], scale)
            oxts['tz'] = oxts['alt']
            oxts_list.append(oxts)
    return oxts_list

# ---------------------------------------------------------------------------------------------------------------------
def create_poses(oxts):
    tx0, ty0, tz0 = oxts[0]['tx'], oxts[0]['ty'], oxts[0]['tz']
    rx0, ry0, rz0 = oxts[0]['roll'], oxts[0]['pitch'], oxts[0]['yaw']
    SE3_t0, _, _ = create_pose_matrix(tx0, ty0, tz0, rx0, ry0, rz0)
    SE3_inv_t0 = np.linalg.inv(SE3_t0)

    poses = []
    for oxt in oxts:
        tx, ty, tz = oxt['tx'], oxt['ty'], oxt['tz']
        rx, ry, rz = oxt['roll'], oxt['pitch'], oxt['yaw']
        SE3, _, _ = create_pose_matrix(tx, ty, tz, rx, ry, rz)
        pose_wrt_ego_t0 = SE3_inv_t0 * SE3
        poses.append(pose_wrt_ego_t0)
    return poses

# ---------------------------------------------------------------------------------------------------------------------
def aggregate_all_json(
    dataset_rootdir, 
    image_dir, 
    seg_dir, 
    annotation_dir, 
    calibration_dir, 
    oxts_dir):

    rearrange_data = True
    all_data = {}
    for seq in config_dataset.kitti_all_sequences_folders:
        data = {}
        data['annotations'] = parse_labels_file(seq, annotation_dir, image_dir, seg_dir, rearrange_data, dataset_rootdir)
        data['calibrations'] = parse_calib_file(seq, calibration_dir, dataset_rootdir)
        data['oxts'] = parse_oxts_file(seq, oxts_dir, dataset_rootdir)
        print(f'sequence : {seq} aggreagted !!!..')
        all_data[seq] = data
    return all_data

# ---------------------------------------------------------------------------------------------------------------------
def aggregate_and_save_groundtruths_json(
    dataset_rootdir,
    image_dir, 
    seg_dir,
    annotation_dir, 
    calibration_dir, 
    oxts_dir,
    label_rootdir,
    aggregated_label_path):

    all_data = aggregate_all_json(
        dataset_rootdir, 
        image_dir, 
        seg_dir, 
        annotation_dir, 
        calibration_dir, 
        oxts_dir)
    
    aggregated_label_path = os.path.join(label_rootdir, aggregated_label_path)
    with open(aggregated_label_path, 'w') as json_file: 
        json.dump(all_data, json_file, indent=4)
    print(f'Labels saved in : {aggregated_label_path}')

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
        annotations_dict['truncated'] = np.array(annotations['truncated'])
        annotations_dict['occluded'] = np.array(annotations['occluded'])
        annotations_dict['alpha'] = np.array(annotations['alpha'])
        annotations_dict['bbox']= np.array(annotations['bbox'])
        annotations_dict['dimensions'] = np.array(annotations['dimensions'])
        annotations_dict['location'] = np.array(annotations['location'])
        annotations_dict['rotation_y'] = np.array(annotations['rotation_y'])
        annotations_list.append(annotations_dict)
    return annotations_list, calibrations_list, poses_list

# ---------------------------------------------------------------------------------------------------------------------
def load_specific_sequence_groundtruths_json(
    sequence, 
    aggregated_label_path, 
    label_rootdir, 
    dataset_rootdir):

    print('Load JSON file .. please wait')
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

    print('Load JSON file .. please wait')
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
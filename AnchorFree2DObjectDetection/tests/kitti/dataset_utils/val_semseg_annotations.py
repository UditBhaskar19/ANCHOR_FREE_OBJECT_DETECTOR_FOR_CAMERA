import argparse
import numpy as np
import sys, os, cv2
module_rootdir = '../../..'
dataset_rootdir = '../../../..'
label_rootdir = module_rootdir
sys.path.append(module_rootdir)

# import config
from modules.dataset_utils.kitti_dataset_utils.kitti_segment_utils import (
    parse_semantic_segmentation_label_image, 
    remap_semantic_segmentation_label, 
    new_semantic_labels_to_id )

from modules.dataset_utils.kitti_dataset_utils.kitti_remap_utils import load_specific_sequence_groundtruths_json

# color in BGR
Black_bgr = (0,0,0)
Yellow_bgr = (0,255,255)
Red_bgr = (0, 0, 255)
Green_bgr = (0, 255, 0)
Blue_bgr = (255, 0, 0)
Orange_bgr = (0, 128, 255)
Purple_bgr = (255, 51, 153)

# ---------------------------------------------------------------------------------------------------------------------
def overlay_segmentation_map_on_image(image_bgr, seg_map):
    H, W, C = image_bgr.shape
    road_map = (seg_map == new_semantic_labels_to_id['road']).flatten()
    vehicle_map = (seg_map == new_semantic_labels_to_id['vehicle']).flatten()
    person_map = (seg_map == new_semantic_labels_to_id['person']).flatten()
    twowheeler_map = (seg_map == new_semantic_labels_to_id['twowheeler']).flatten()

    label_image = np.full_like(image_bgr, 0).reshape((H*W, C))
    label_image[road_map] = np.array(Green_bgr, dtype=np.uint8)
    label_image[vehicle_map] = np.array(Red_bgr, dtype=np.uint8)
    label_image[person_map] = np.array(Purple_bgr, dtype=np.uint8)
    label_image[twowheeler_map] = np.array(Orange_bgr, dtype=np.uint8)
    label_image = label_image.reshape((H, W, C))
    image_bgr = cv2.addWeighted(image_bgr, 1, label_image, 1, 0)
    return image_bgr

# ---------------------------------------------------------------------------------------------------------------------
def visualize_annotations(annotations_list, play_video):
    def press_q_to_quit(key):
        return key == 113
    for annotation in annotations_list:
        image_path = annotation['image_path']
        seg_image_path = annotation['segmentation_image_path']
        image_bgr = cv2.imread(image_path)
        seg_map, _ = parse_semantic_segmentation_label_image(seg_image_path)
        seg_map = remap_semantic_segmentation_label(seg_map)
        image_bgr = overlay_segmentation_map_on_image(image_bgr, seg_map)
        print(image_path)
        cv2.imshow("Source", image_bgr)
        if play_video: key = cv2.waitKey(30)
        else: key = cv2.waitKey(0)
        if press_q_to_quit(key): break 
    cv2.destroyAllWindows()

# ---------------------------------------------------------------------------------------------------------------------
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == '__main__':
    import argparse
    import os
    import config_dataset

    parser = argparse.ArgumentParser(description="validate 2D annotated bounding boxes")
    parser.add_argument('--scene', type=str, help="scene that is being considered. Available scenes: '0000', '0001', ... '0020'")
    parser.add_argument('--play_video', type=boolean_string, default=True, help="flag to select if the frames are to be played like a video")
    args = parser.parse_args()

    aggregated_label_path = config_dataset.kitti_remapped_label_file_path
    annotations_list, calibrations_list, poses_list \
        = load_specific_sequence_groundtruths_json(args.scene, aggregated_label_path, label_rootdir, dataset_rootdir)

    visualize_annotations(annotations_list, args.play_video)
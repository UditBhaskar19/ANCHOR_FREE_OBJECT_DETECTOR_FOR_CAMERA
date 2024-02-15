import argparse
import sys, os, cv2
module_rootdir = '../../..'
dataset_rootdir = '../../../..'
label_rootdir = module_rootdir
sys.path.append(module_rootdir)

from modules.dataset_utils.kitti_dataset_utils.kitti_file_utils import \
    load_specific_sequence_groundtruths_json as load_specific_sequence_groundtruths_json1
from modules.dataset_utils.kitti_dataset_utils.kitti_remap_utils import \
    load_specific_sequence_groundtruths_json as load_specific_sequence_groundtruths_json2

# color in BGR
Black_bgr = (0,0,0)
Yellow_bgr = (0,255,255)
Red_bgr = (0, 0, 255)
Green_bgr = (0, 255, 0)

# ---------------------------------------------------------------------------------------------------------------------
def draw_annotations_on_image(image_bgr, bboxs, class_labels):
    # font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    font_thickness = 2
    dx = 20; dy = 8

    # objects
    for idx, box in enumerate(bboxs):
        # bounding box
        tl = (int(box[0]), int(box[1]))
        br = (int(box[2]), int(box[3]))
        cv2.rectangle(image_bgr, tl, br, Yellow_bgr, thickness=2)
        cv2.circle(image_bgr, tl, radius=4, color=Red_bgr, thickness=5)
        cv2.circle(image_bgr, br, radius=4, color=Green_bgr, thickness=5)

        # object catagory
        text = class_labels[idx]
        (w, h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        box_coords = ((tl[0], tl[1] - h - dy), (tl[0] + w, tl[1]))
        text_coords = (tl[0], tl[1] - dy // 2)
        cv2.rectangle(image_bgr, box_coords[0], box_coords[1], Yellow_bgr, cv2.FILLED)
        cv2.putText(image_bgr, text, text_coords, font, font_scale, Black_bgr, font_thickness)
    return image_bgr

# ---------------------------------------------------------------------------------------------------------------------
def visualize_annotations(annotations_list, play_video):
    def press_q_to_quit(key):
        return key == 113
    for annotation in annotations_list:
        class_labels = annotation['type']
        bboxs = annotation['bbox']
        image_path = annotation['image_path']
        image_bgr = cv2.imread(image_path)
        image_bgr = draw_annotations_on_image(image_bgr, bboxs, class_labels)
        print(image_path)
        cv2.imshow("Source", image_bgr)
        if play_video: key = cv2.waitKey(50)
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
    import os, sys
    import config_dataset

    parser = argparse.ArgumentParser(description="validate 2D annotated bounding boxes")
    parser.add_argument('--scene', type=str, help="scene that is being considered. Available scenes: '0000', '0001', ... '0020'")
    parser.add_argument('--play_video', type=boolean_string, default=True, help="flag to select if the frames are to be played like a video")
    args = parser.parse_args()

    # aggregated_label_path = config_dataset.kitti_label_file_path
    # annotations_list, calibrations_list, poses_list \
    #     = load_specific_sequence_groundtruths_json1(args.scene, aggregated_label_path, label_rootdir, dataset_rootdir)

    aggregated_label_path = config_dataset.kitti_remapped_label_file_path
    annotations_list, calibrations_list, poses_list \
        = load_specific_sequence_groundtruths_json2(args.scene, aggregated_label_path, label_rootdir, dataset_rootdir)

    visualize_annotations(annotations_list, args.play_video)
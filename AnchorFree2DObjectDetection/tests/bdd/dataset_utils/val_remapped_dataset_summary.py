import sys, os
module_rootdir = '../../..'
dataset_rootdir = '../../../../'
label_rootdir = module_rootdir
sys.path.append(module_rootdir)

import config_dataset
from modules.dataset_utils.bdd_dataset_utils.remapped_bdd_utils import load_ground_truths
from modules.dataset_utils.bdd_dataset_utils.remapped_dataset_summary import class_labels_summary, obj_box_summary, aggregated_bboxes
from modules.dataset_utils.bdd_dataset_utils.dataset_summary import sort_according_to_box_criteria

sel_train_labels_file = config_dataset.bdd_sel_train_labels_file
train_images_dir = config_dataset.bdd_train_images_dir

selected_labels = load_ground_truths(
    label_rootdir,
    sel_train_labels_file, 
    dataset_rootdir,
    train_images_dir, 
    verbose=True)

class_summary = class_labels_summary(selected_labels)
bbox_summary = obj_box_summary(selected_labels)
boxes_all, obj_class_all, area_all, aspect_ratio_all, image_names_all = aggregated_bboxes(bbox_summary)
boxes, obj_class, box_area, box_aspect_ratio, image_names = sort_according_to_box_criteria(
    boxes_all, obj_class_all, area_all, aspect_ratio_all, image_names_all
)

for key, val in class_summary.items():
    print('-' * 100)
    print(f'{key}')
    print('-' * 100)
    for catagory, value in val.items():
        print(f'{catagory}:  {value}')
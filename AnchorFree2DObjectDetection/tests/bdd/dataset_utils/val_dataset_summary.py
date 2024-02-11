import sys
module_rootdir = '../../..'
dataset_root_dir = '../../../../'
sys.path.append(module_rootdir)
import config_dataset, os
from modules.dataset_utils.bdd_dataset_utils.bdd_utils import aggregate_ground_truths
from modules.dataset_utils.bdd_dataset_utils.dataset_summary import class_labels_summary, obj_box_summary, aggregated_bboxes

train_labels_file = config_dataset.bdd_train_labels_file
train_images_dir = config_dataset.bdd_train_images_dir
train_lane_labels_file = config_dataset.bdd_train_lane_labels_file

selected_labels, label_names = aggregate_ground_truths(
    dataset_root_dir,
    train_labels_file, 
    train_images_dir, 
    train_lane_labels_file)

# print(label_names)
# print(selected_labels[0])

class_summary = class_labels_summary(selected_labels)
bbox_summary = obj_box_summary(selected_labels)
all_boxes = aggregated_bboxes(bbox_summary)

for key, val in class_summary.items():
    print('-' * 100)
    print(f'{key}')
    print('-' * 100)
    for catagory, value in val.items():
        print(f'{catagory}:  {value}')
# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Aggregate labels       
# ---------------------------------------------------------------------------------------------------------------------
import sys, os
module_rootdir = '.'
dataset_rootdir = '../'
label_rootdir = module_rootdir
sys.path.append(module_rootdir)

import config_dataset
from modules.dataset_utils.bdd_dataset_utils.remapped_bdd_utils \
    import aggregate_and_save_ground_truths as save_ground_truths_bdd
from modules.dataset_utils.bdd_dataset_utils.remapped_bdd_utils \
    import load_ground_truths as load_ground_truths_bdd
from modules.dataset_utils.kitti_dataset_utils.kitti_file_utils \
    import aggregate_and_save_groundtruths_json as save_groundtruths_json_kitti
from modules.dataset_utils.kitti_dataset_utils.kitti_remap_utils \
    import aggregate_and_save_remapped_groundtruths_json as save_remapped_groundtruths_json_kitti

# =======================================> BDD DATASET <==============================================
train_images_dir = config_dataset.bdd_train_images_dir
val_images_dir = config_dataset.bdd_val_images_dir

train_labels_file = config_dataset.bdd_train_labels_file
val_labels_file = config_dataset.bdd_val_labels_file

label_out_dir = os.path.join(label_rootdir, config_dataset.bdd_label_out_dir)
if not os.path.exists(label_out_dir): os.makedirs(label_out_dir, exist_ok=True)

sel_train_labels_file = config_dataset.bdd_sel_train_labels_file
sel_val_labels_file = config_dataset.bdd_sel_val_labels_file

print("========> Aggregating and Saving BDD Dataset <=========")

save_ground_truths_bdd(
    label_rootdir,
    sel_train_labels_file,
    dataset_rootdir,
    train_labels_file)

selected_labels = load_ground_truths_bdd(
    label_rootdir,
    sel_train_labels_file, 
    dataset_rootdir,
    train_images_dir, 
    verbose=True)

save_ground_truths_bdd(
    label_rootdir,
    sel_val_labels_file,
    dataset_rootdir,
    val_labels_file)

selected_labels = load_ground_truths_bdd(
    label_rootdir,
    sel_val_labels_file, 
    dataset_rootdir,
    val_images_dir, 
    verbose=True)

# =======================================> KITTI DATASET <==============================================
kitti_image_dir = config_dataset.kitti_image_dir
kitti_annotation_dir = config_dataset.kitti_annotation_dir
kitti_segmentation_dir = config_dataset.kitti_segmentation_dir
kitti_calibration_dir = config_dataset.kitti_calibration_dir
kitti_oxts_dir = config_dataset.kitti_oxts_dir

kitti_label_dir = os.path.join(label_rootdir, config_dataset.kitti_label_dir)
if not os.path.exists(kitti_label_dir): os.makedirs(kitti_label_dir, exist_ok=True)

kitti_label_file_path = config_dataset.kitti_label_file_path
kitti_remapped_label_file_path = config_dataset.kitti_remapped_label_file_path

print("========> Aggregating and Saving KITTI Dataset <=========")

save_groundtruths_json_kitti(
    dataset_rootdir,
    kitti_image_dir, 
    kitti_segmentation_dir, 
    kitti_annotation_dir, 
    kitti_calibration_dir, 
    kitti_oxts_dir,
    label_rootdir, 
    kitti_label_file_path)

save_remapped_groundtruths_json_kitti(
    label_rootdir,
    kitti_label_file_path, 
    kitti_remapped_label_file_path)
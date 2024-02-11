# ----------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Dataset related paths, folders and files
# ----------------------------------------------------------------------------------------------------------------
import os

# =======================================> BDD Dataset paths <====================================================
bdd_train_images_dir = 'dataset/bdd/bdd100k_images_100k/bdd100k/images/100k/train'
bdd_train_labels_file = 'dataset/bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json'
bdd_train_lane_labels_file = 'dataset/bdd/bdd100k_lane_labels_trainval/bdd100k/labels/lane/colormaps/train'

bdd_val_images_dir = 'dataset/bdd/bdd100k_images_100k/bdd100k/images/100k/val'
bdd_val_labels_file = 'dataset/bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json'
bdd_val_lane_labels_file = 'dataset/bdd/bdd100k_lane_labels_trainval/bdd100k/labels/lane/colormaps/val'

bdd_test_images_dir = 'dataset/bdd/bdd100k_images_100k/bdd100k/images/100k/test'

bdd_label_out_dir = 'labels/bdd'
bdd_sel_train_labels_file_name = 'labels_images_train.json'
bdd_sel_val_labels_file_name = 'labels_images_val.json'
bdd_sel_train_labels_file = os.path.join(bdd_label_out_dir, bdd_sel_train_labels_file_name)
bdd_sel_val_labels_file = os.path.join(bdd_label_out_dir, bdd_sel_val_labels_file_name)

# =======================================> KITTI Dataset paths <=================================================
kitti_image_dir = 'dataset/kitti/training/image_02'
kitti_annotation_dir = 'dataset/kitti/training/label_02'
kitti_segmentation_dir = 'dataset/kitti/training/panoptic_maps'
kitti_calibration_dir = 'dataset/kitti/training/calib'
kitti_oxts_dir = 'dataset/kitti/training/oxts'

kitti_label_dir = 'labels/kitti'
kitti_label_file = 'groundtruths.json'
kitti_remapped_label_file = 'remapped_gt.json'
kitti_label_file_path = os.path.join(kitti_label_dir, kitti_label_file)
kitti_remapped_label_file_path = os.path.join(kitti_label_dir, kitti_remapped_label_file)

kitti_all_sequences_folders = ['0000', '0001', '0002', '0003', '0004', '0005', '0006',
                               '0007', '0008', '0009', '0010', '0011', '0012', '0013',
                               '0014', '0015', '0016', '0017', '0018', '0019', '0020']
kitti_train_sequences_folders = ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0008', 
                                 '0009', '0011', '0012', '0015', '0016', '0017', '0019', '0020']
kitti_val_sequences_folders = ['0007', '0010', '0013', '0014', '0018']

kitti_image_dir_test =  'dataset/kitti/testing/image_02'
kitti_calibration_dir_test = 'dataset/kitti/testing/calib'
kitti_oxts_dir_test = 'dataset/kitti/testing/oxts'
kitti_all_sequences_folders_test = ['0000', '0001', '0002', '0003', '0004', '0005', '0006',
                                    '0007', '0008', '0009', '0010', '0011', '0012', '0013',
                                    '0014', '0015', '0016', '0017', '0018', '0019', '0020',
                                    '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028']
# kitti_all_sequences_folders_test = ['0016', '0017', '0018', '0019', '0020',
#                                     '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028']
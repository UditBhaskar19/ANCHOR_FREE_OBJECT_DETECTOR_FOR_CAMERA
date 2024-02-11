# ----------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : KITTI Segmentation related info
# ----------------------------------------------------------------------------------------------------------------
import cv2
import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
""" In this project we are aggregating the semantic labels """

all_semantic_labels = ['road', 
                       'sidewalk', 'building', 'wall', 
                       'fence', 'pole', 'traffic light', 'traffic sign', 
                       'vegetation', 'terrain', 
                       'sky', 
                       'person', 'rider', 
                       'car', 'truck', 'bus', 'train', 
                       'motorcycle', 'bicycle', 
                       'void']

semantic_labels_to_id = {}
semantic_labels_to_id['road'] = 0
semantic_labels_to_id['sidewalk'] = 1
semantic_labels_to_id['building'] = 2
semantic_labels_to_id['wall'] = 3
semantic_labels_to_id['fence'] = 4
semantic_labels_to_id['pole'] = 5
semantic_labels_to_id['traffic light'] = 6
semantic_labels_to_id['traffic sign'] = 7
semantic_labels_to_id['vegetation'] = 8
semantic_labels_to_id['terrain'] = 9
semantic_labels_to_id['sky'] = 10
semantic_labels_to_id['person'] = 11
semantic_labels_to_id['rider'] = 12
semantic_labels_to_id['car'] = 13
semantic_labels_to_id['truck'] = 14
semantic_labels_to_id['bus'] = 15
semantic_labels_to_id['train'] = 16
semantic_labels_to_id['motorcycle'] = 17
semantic_labels_to_id['bicycle'] = 18
semantic_labels_to_id['void'] = 255

new_semantic_labels = ['road', 'structure', 'object', 'nature', 'sky', 'person', 'vehicle', 'twowheeler', 'void']

new_semantic_labels_to_id = {}
new_semantic_labels_to_id['road'] = 0
new_semantic_labels_to_id['structure'] = 1
new_semantic_labels_to_id['object'] = 2
new_semantic_labels_to_id['nature'] = 3
new_semantic_labels_to_id['sky'] = 4
new_semantic_labels_to_id['person'] = 5
new_semantic_labels_to_id['vehicle'] = 6
new_semantic_labels_to_id['twowheeler'] = 7
new_semantic_labels_to_id['void'] = 255

semantic_labels_old_to_new = {}
semantic_labels_old_to_new['road'] = 'road'
semantic_labels_old_to_new['sidewalk'] = 'structure'
semantic_labels_old_to_new['building'] = 'structure'
semantic_labels_old_to_new['wall'] = 'structure'
semantic_labels_old_to_new['fence'] = 'object'
semantic_labels_old_to_new['pole'] = 'object'
semantic_labels_old_to_new['traffic light'] = 'object'
semantic_labels_old_to_new['traffic sign'] = 'object'
semantic_labels_old_to_new['vegetation'] = 'nature'
semantic_labels_old_to_new['terrain'] = 'nature'
semantic_labels_old_to_new['sky'] = 'sky'
semantic_labels_old_to_new['person'] = 'person'
semantic_labels_old_to_new['rider'] = 'person'
semantic_labels_old_to_new['car'] = 'vehicle'
semantic_labels_old_to_new['truck'] = 'vehicle'
semantic_labels_old_to_new['bus'] = 'vehicle'
semantic_labels_old_to_new['train'] = 'vehicle'
semantic_labels_old_to_new['motorcycle'] = 'twowheeler'
semantic_labels_old_to_new['bicycle'] = 'twowheeler'
semantic_labels_old_to_new['void'] = 'void'

all_semantic_ids = list(semantic_labels_to_id.values())
all_semantic_labels = list(semantic_labels_to_id.keys())

new_semantic_labels = [ semantic_labels_old_to_new[label] for label in all_semantic_labels ]
new_semantic_ids = [ new_semantic_labels_to_id[label] for label in new_semantic_labels ]

semantic_id_old_to_new = [-1] * (max(all_semantic_ids) + 1)
for (id_old, id_new) in zip(all_semantic_ids, new_semantic_ids): 
    semantic_id_old_to_new[id_old] = id_new

# ---------------------------------------------------------------------------------------------------------------------
def parse_semantic_segmentation_label_image(segmentation_image_path):
    image_bgr = cv2.imread(segmentation_image_path)
    semantic_id = image_bgr[:, :, 2]  # sematic id Red channel
    instance_id = image_bgr[:, :, 1] * 256 + image_bgr[:, :, 0]  # instance id Green and Blue channel
    return semantic_id, instance_id

# ---------------------------------------------------------------------------------------------------------------------
def remap_semantic_segmentation_label(semantic_idmap):
    H, W = semantic_idmap.shape
    ids_flattened = semantic_idmap.flatten()
    new_semantic_idmap = np.array(semantic_id_old_to_new)[ids_flattened]
    new_semantic_idmap = new_semantic_idmap.reshape(H, W)
    return new_semantic_idmap
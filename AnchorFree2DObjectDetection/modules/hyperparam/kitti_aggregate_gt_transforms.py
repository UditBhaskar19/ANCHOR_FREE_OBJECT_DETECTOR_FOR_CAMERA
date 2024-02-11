# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Compute box regression offset statistics for KITTI dataset              
# ---------------------------------------------------------------------------------------------------------------------
import torch, json, os
import config_dataset

from modules.proposal.box_association import fcos_match_locations_to_gt_main
from modules.proposal.prop_functions import get_bbox_deltas_normalized
from modules.proposal.box_functions import gen_grid_coord
from modules.dataset_utils.kitti_dataset_utils.kitti_remap_utils import load_all_sequence_groundtruths_json
from modules.dataset_utils.kitti_objdetect_dataloader import KittiTrackingDataset_2DObjectDetection, extract_valid_bbox
from modules.dataset_utils.kitti_dataset_utils.constants import _OBJ_CLASS_TO_IDX_
from modules.proposal.constants import _SHRINK_FACTOR_, _MATCH_CRITERIA_, _IGNORED_CLASS_DEFAULT_ID_
from modules.dataset_utils.kitti_dataset_utils.constants import (
    _IGNORED_CLASS_ID_, _IMG_D_, _IMG_RESIZED_H_, _IMG_RESIZED_W_, _OUT_FEAT_SIZE_H_, _OUT_FEAT_SIZE_W_, _STRIDE_H_, _STRIDE_W_)

hyperparam_out_dir = 'hyperparam/kitti'
filename_aggregated_transforms = 'aggregated_transforms.json'
filename_transforms_statistics = 'transforms_statistics.json'
filename_class_instance_count = 'class_instance_count.json'

# ---------------------------------------------------------------------------------------------------------------------
def write_data_json(data, out_dir, filename):
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    filename = os.path.join(out_dir, filename)
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def load_data_json(filepath):
    with open(filepath, 'r') as file: data = json.load(file)
    return data

# ---------------------------------------------------------------------------------------------------------------------
def aggregate_deltas(
    num_samples, 
    label_rootdir, 
    dataset_rootdir,
    out_dir, 
    write_deltas: bool = True, 
    write_deltas_statistics: bool = True,
    write_label_instance_count: bool = True):

    device = 'cuda'

    gt_labels_train, _, _ = load_all_sequence_groundtruths_json(
        config_dataset.kitti_train_sequences_folders, 
        config_dataset.kitti_remapped_label_file_path, 
        label_rootdir, 
        dataset_rootdir)

    gt_labels_val, _, _ = load_all_sequence_groundtruths_json(
        config_dataset.kitti_val_sequences_folders, 
        config_dataset.kitti_remapped_label_file_path, 
        label_rootdir, 
        dataset_rootdir)

    # init data-loader
    kitti_dataset = KittiTrackingDataset_2DObjectDetection(
        gt_labels = gt_labels_train, 
        image_new_shape = (_IMG_D_, _IMG_RESIZED_H_, _IMG_RESIZED_W_),
        device = device, 
        subset = num_samples,
        shuffle_dataset = True,
        augment = False)
    
    feat_h, feat_w = _OUT_FEAT_SIZE_H_, _OUT_FEAT_SIZE_W_
    strides_h, strides_w = _STRIDE_H_, _STRIDE_W_
    grid_coord = gen_grid_coord(feat_w, feat_h, strides_w, strides_h, device)

    print('strides width : ', strides_w)
    print('strides height: ', strides_h)
    print('feature width : ', feat_w)
    print('feature height: ', feat_h)

    aggregated_transforms = []
    num_transforms = 0
    class_instance_count = {}
    for label_id in _OBJ_CLASS_TO_IDX_.values():
        class_instance_count[str(label_id)] = 0

    aggregated_transforms = []
    num_transforms = 0
    class_instance_count = {}
    for label_id in _OBJ_CLASS_TO_IDX_.values():
        class_instance_count[str(label_id)] = 0

    for i in range(len(kitti_dataset)):
        _, labels = kitti_dataset.__getitem__(i)
        bbox, cls_label = extract_valid_bbox(labels['bbox'], labels['obj_class_label'])
        bbox = bbox.to(device)
        cls_label = cls_label.to(device)

        matched_gt_class, \
        matched_gt_boxes \
            = fcos_match_locations_to_gt_main(
                grid_coord, bbox, cls_label, device, 
                _IGNORED_CLASS_ID_, _SHRINK_FACTOR_, _MATCH_CRITERIA_)   # 'closest_box' or 'smallest_area'
        
        if bbox.shape[0] > 0:

            valid_labels = matched_gt_class >= 0
            matched_gt_class = matched_gt_class[valid_labels]
            for label_id in _OBJ_CLASS_TO_IDX_.values():
                flag = (matched_gt_class.to(torch.int16) == label_id)
                class_instance_count[str(label_id)] += torch.sum(flag).cpu().item()

            offsets_mean = torch.tensor([0,0,0,0], dtype=torch.float32, device=device)
            offsets_std = torch.tensor([1,1,1,1], dtype=torch.float32, device=device)
            # transforms = get_bbox_offsets_normalized(
            #     grid_coord, matched_gt_boxes, 
            #     strides_w, strides_h, 
            #     offsets_mean, offsets_std)
            transforms = get_bbox_deltas_normalized(
                grid_coord, matched_gt_boxes, 
                offsets_mean, offsets_std)
            aggregated_transforms += transforms[valid_labels]
            num_transforms += transforms.shape[0]

        if i % 100 == 1: print(f'{num_transforms} deltas accumulated,   {i}/{len(kitti_dataset)} images processed')
    print(f'{num_transforms} deltas accumulated,   {len(kitti_dataset)}/{len(kitti_dataset)} images processed')

    # compute statistical summary
    aggregated_transforms = torch.stack(aggregated_transforms, dim=0)
    mean = torch.mean(aggregated_transforms, dim=0).cpu().numpy().tolist()
    std = torch.std(aggregated_transforms, dim=0).cpu().numpy().tolist()
    aggregated_transforms = aggregated_transforms.cpu().numpy().tolist()

    if write_deltas:
        write_data_json(aggregated_transforms, out_dir, filename_aggregated_transforms)

    if write_deltas_statistics:
        transforms = {}
        transforms['mean'] = mean
        transforms['std'] = std
        write_data_json(transforms, out_dir, filename_transforms_statistics)

    if write_label_instance_count:
        write_data_json(class_instance_count, out_dir, filename_class_instance_count)

    return class_instance_count, aggregated_transforms, mean, std
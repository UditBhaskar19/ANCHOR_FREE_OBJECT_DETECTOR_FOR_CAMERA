# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : < Work in progress >        
# ---------------------------------------------------------------------------------------------------------------------
# import torch
# from torch.utils.data import DataLoader

# import sys, os
# rootdir = '../..'
# sys.path.append(rootdir)

# import config_dataset
# from get_datasets import infinite_loader 
# from get_parameters import (
#     get_device, bdd_parameters, kitti_parameters, net_config, print_parameters )
# from modules.dataset_utils.bdd_dataset_utils.remapped_bdd_utils import load_ground_truths
# from modules.dataset_utils.bdd_dataset_and_dataloader import BerkeleyDeepDriveDataset
# from modules.hyperparam.bdd_aggregate_gt_transforms import write_data_json, load_data_json

# from modules.neural_net.backbone.backbone_v2 import net_backbone
# from modules.neural_net.bifpn.bifpn_nblks_v2 import BiFPN
# from modules.neural_net.head.shared_head_v5 import SharedNet
# from modules.neural_net.detector.detector_v1 import FCOS
# from modules.neural_net.fcos import FCOS_train
# from modules.loss.fcos_loss import FCOS_Loss

# from modules.second_stage.proposal_extraction import proposal_extractor
# from modules.second_stage.get_param import net_config_stage2 as net_config
# from modules.second_stage.get_param import bdd_parameters_stage2 as bdd_parameters
# from modules.second_stage.get_param import kitti_parameters_stage2 as kitti_parameters
# from modules.second_stage.generate_gt import gen_training_gt

# BATCH_SIZE = 1
# num_samples = 10000
# DEVICE = get_device()

# write_deltas = True
# write_deltas_statistics = True
# write_label_instance_count = True

# hyperparam_out_dir = 'hyperparam/bdd'
# if not os.path.exists(hyperparam_out_dir): os.makedirs(hyperparam_out_dir, exist_ok=True)
# filename_aggregated_transforms = 'aggregated_transforms.json'
# filename_transforms_statistics = 'transforms_statistics.json'
# filename_class_instance_count = 'class_instance_count.json'

# # ================================================> SAVED MODEL WEIGHTS <========================================================
# weights_path = 'model_weights/1705990924432/anchor_free_detector.pt'
# weights_path = os.path.join(rootdir, weights_path)

# # ======================================> GET MODEL CONFIGURATION & OTHER PARAMETERS <===========================================
# net_config_obj = net_config()
# bdd_param_obj = bdd_parameters()
# kitti_param_obj = kitti_parameters()
# print_parameters(net_config_obj, bdd_param_obj, kitti_param_obj, DEVICE)

# # ============================================> BDD DATASET & DATALOADER TRAIN <=================================================
# # init train data-loader
# gt_labels_train = load_ground_truths(
#     os.path.join(rootdir, config_dataset.bdd_sel_train_labels_file),
#     os.path.join(rootdir, config_dataset.bdd_train_images_dir),
#     verbose = False)

# bdd_dataset_train = BerkeleyDeepDriveDataset(
#     gt_labels_train, 
#     (bdd_param_obj.IMG_D, bdd_param_obj.IMG_RESIZED_H, bdd_param_obj.IMG_RESIZED_W),
#     DEVICE, subset = -1, augment = False)

# bdd_train_args = dict(batch_size=BATCH_SIZE, shuffle=False, collate_fn=bdd_dataset_train.collate_fn)
# bdd_dataloader_train = DataLoader(bdd_dataset_train, **bdd_train_args)

# # ========================================> INIT DENSE PREDICTOR NETWORK STRUCTURE <=============================================
# backbone = net_backbone(net_config_obj)
# bifpn = BiFPN(net_config_obj, bdd_param_obj.feat_pyr_shapes)
# shared_head = SharedNet(net_config_obj, bdd_param_obj.out_feat_shape) 
# fcos = FCOS(backbone, bifpn, shared_head)

# loss = FCOS_Loss(net_config_obj, DEVICE)
# detector = FCOS_train(fcos, loss, bdd_param_obj, DEVICE)
# detector.load_state_dict(torch.load(weights_path, map_location="cpu"))
# detector = detector.to(DEVICE)

# # ==========================================> LOAD FEATURE EXTRACTOR NTEWORK <==================================================
# prop_extractor = proposal_extractor(
#     backbone = detector.detector.backbone, 
#     feataggregator = detector.detector.feataggregator, 
#     sharednet = detector.detector.sharednet,
#     netconfig_obj = net_config_obj,
#     param_obj = bdd_param_obj,
#     device = DEVICE)

# iter_start_offset = 0
# max_iters = iter_start_offset + num_samples
# bdd_dataloader_train = infinite_loader(bdd_dataloader_train)

# aggregated_transforms = []
# num_pos_samples = 0
# num_neg_samples = 0
# num_ignored_samples = 0

# for iter in range(iter_start_offset, max_iters):

#     img, labels = next(bdd_dataloader_train)
#     img_path = labels['img_path']
#     bboxes = labels['bbox_batch']
#     clslabels = labels['obj_class_label']

#     roi_features = prop_extractor(img)
#     features = roi_features['features']
#     queries = roi_features['queries']
#     pred_boxes = roi_features['pred_boxes']

#     groundtruths = gen_training_gt(
#         gt_boxes = bboxes, 
#         gt_class = clslabels,
#         pred_boxes = pred_boxes,
#         deltas_mean = torch.zeros((4, ), dtype=torch.float32, device=DEVICE), 
#         deltas_std = torch.ones((4, ), dtype=torch.float32, device=DEVICE), 
#         iou_threshold = bdd_param_obj.iou_threshold_stage2,
#         ignored_classId = bdd_param_obj.ignored_classId_stage2)
    
#     flag_ignored_mask = groundtruths.class_logits == -2
#     flag_neg_mask = groundtruths.class_logits == -1
#     flag_pos_mask = groundtruths.class_logits >= 0
    
#     aggregated_transforms += groundtruths.boxreg_deltas[flag_pos_mask]
#     num_pos_samples += torch.sum(flag_pos_mask)
#     num_neg_samples += torch.sum(flag_neg_mask)
#     num_ignored_samples += torch.sum(flag_ignored_mask)

#     if iter % 100 == 1: print(f'{num_pos_samples} deltas accumulated,   {iter}/{num_samples} images processed')
# print(f'{num_pos_samples} deltas accumulated,   {num_samples}/{num_samples} images processed')

# # compute statistical summary
# aggregated_transforms = torch.stack(aggregated_transforms, dim=0)
# mean = torch.mean(aggregated_transforms, dim=0).cpu().numpy().tolist()
# std = torch.std(aggregated_transforms, dim=0).cpu().numpy().tolist()
# aggregated_transforms = aggregated_transforms.cpu().numpy().tolist()

# print('mean', mean)
# print('std', std)
# print('num_pos_samples', num_pos_samples)
# print('num_neg_samples', num_neg_samples)
# print('num_ignored_samples', num_ignored_samples)

# if write_deltas:
#     write_data_json(aggregated_transforms, hyperparam_out_dir, filename_aggregated_transforms)

# if write_deltas_statistics:
#     transforms = {}
#     transforms['mean'] = mean
#     transforms['std'] = std
#     write_data_json(transforms, hyperparam_out_dir, filename_transforms_statistics)

# if write_label_instance_count:
#     class_instance_count = {}
#     class_instance_count['num_pos_samples'] = num_pos_samples.item()
#     class_instance_count['num_neg_samples'] = num_neg_samples.item()
#     class_instance_count['num_ignored_samples'] = num_ignored_samples.item()
#     write_data_json(class_instance_count, hyperparam_out_dir, filename_class_instance_count)
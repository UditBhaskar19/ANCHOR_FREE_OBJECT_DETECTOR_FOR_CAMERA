# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : < Work in progress >        
# ---------------------------------------------------------------------------------------------------------------------
# import torch

# import sys
# rootdir = '../..'
# sys.path.append(rootdir)

# from modules.first_stage.get_parameters import print_parameters, get_device, reset_seed
# from modules.second_stage.get_param import net_config_stage2, bdd_parameters_stage2, kitti_parameters_stage2
# from modules.first_stage.get_datasets import BDD_dataset, KITTI_dataset, DATSET_Selector

# from modules.neural_net.backbone.backbone_v2 import net_backbone
# from modules.neural_net.bifpn.bifpn_nblks_v2 import BiFPN
# from modules.neural_net.head.shared_head_v5 import SharedNet
# from modules.first_stage.detector import Detector
# from modules.first_stage.detector_train import Detector_Train
# from modules.first_stage.first_stage_loss import Loss

# from modules.second_stage.proposal_extraction import proposal_extractor
# from modules.second_stage.detector_v2 import second_stage_predictor, second_stage_detector_train
# from modules.second_stage.second_stage_loss import second_stage_loss

# reset_seed(0)
# BATCH_SIZE = 4
# DEVICE = get_device()
# # ================================================> SAVED MODEL WEIGHTS <========================================================
# # incase we would like to resume training from a model weight checkpoint, set 'load_model_weights' as True and
# # set the weights_path
# first_stage_weights_path = 'model_weights/1705990924432/anchor_free_detector.pt'
# load_second_stage_model_weights = False
# second_stage_weights_path = ''

# # ======================================> GET MODEL CONFIGURATION & OTHER PARAMETERS <===========================================
# net_config_obj = net_config_stage2()
# bdd_param_obj = bdd_parameters_stage2()
# kitti_param_obj = kitti_parameters_stage2()
# print_parameters(net_config_obj, bdd_param_obj, kitti_param_obj, DEVICE)

# # ===============================================> LOAD WEIGHTS FIRST STAGE <====================================================
# backbone = net_backbone(net_config_obj)
# bifpn = BiFPN(net_config_obj)
# shared_head = SharedNet(net_config_obj, bdd_param_obj.out_feat_shape) 
# detector = Detector(backbone, bifpn, shared_head)

# loss = Loss(net_config_obj, DEVICE)
# detector_train = Detector_Train(detector, loss, bdd_param_obj, DEVICE)
# detector_train.load_state_dict(torch.load(first_stage_weights_path, map_location="cpu"))
# detector_train = detector_train.to(DEVICE)

# # ===============================================> INIT NETWORK STRUCTURE <======================================================
# prop_extractor = proposal_extractor(
#     backbone = detector_train.detector.backbone, 
#     feataggregator = detector_train.detector.feataggregator, 
#     sharednet = detector_train.detector.sharednet,
#     netconfig_obj = net_config_obj,
#     param_obj = bdd_param_obj,
#     device = DEVICE)

# detector_second_stage = second_stage_predictor(
#     netconfig_obj = net_config_obj,
#     feat_pyr_shapes = bdd_param_obj.feat_pyr_shapes )

# loss_second_stage = second_stage_loss(device = DEVICE)

# detector_second_stage_train = second_stage_detector_train(
#     detector = detector_second_stage, 
#     loss_obj = loss_second_stage,
#     param_obj = bdd_param_obj,
#     device = DEVICE)

# if load_second_stage_model_weights:
#     detector_second_stage_train.load_state_dict(torch.load(second_stage_weights_path, map_location="cpu"))
# detector_second_stage_train = detector_second_stage_train.to(DEVICE)

# # ============================================> SET OPTIMIZATION PARAMETERS <==================================================
# # learning_rate = 8e-4
# # weight_decay = 1e-4
# # max_iters = 100000

# # params = [p for p in detector_second_stage_train.parameters() if p.requires_grad]
# # optimizer = torch.optim.SGD(params, momentum=0.9, lr=learning_rate, weight_decay=weight_decay)

# learning_rate = 1e-3
# weight_decay = 1e-4
# max_iters = 50000

# params = [p for p in detector_second_stage_train.parameters() if p.requires_grad]
# optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

# # in case we have to abruptly stop training and resume the training at a later time
# init_start = 0 # ==> start from this iteration  
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR( 
#     optimizer, 
#     gamma=0.1,
#     milestones=[int(0.6 * max_iters - init_start), 
#                 int(0.9 * max_iters - init_start)])

# # ==============================================> DATASET & DATALOADER <===================================================
# bdd_dataloader = BDD_dataset(
#     batch_size = BATCH_SIZE,
#     num_samples_val = 1000, 
#     bdd_param_obj = bdd_param_obj,
#     device = DEVICE,
#     shuffle_dataset = False,
#     perform_augmentation_train = False,
#     augmentation_prob_train = 0)

# kitti_dataloader = KITTI_dataset(
#     batch_size = BATCH_SIZE,
#     num_samples_val = 1000, 
#     kitti_param_obj = kitti_param_obj,
#     device = DEVICE,
#     shuffle_dataset = True,
#     perform_augmentation_train = False,
#     augmentation_prob_train = 0)

# dataloader_selector = DATSET_Selector(
#     bdd_dataset_obj = bdd_dataloader,
#     kitti_dataset_obj = kitti_dataloader,
#     max_training_iter = max_iters,
#     bdd_dataset_weight = 0.8)
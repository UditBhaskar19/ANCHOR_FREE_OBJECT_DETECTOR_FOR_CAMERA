# --------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : training parameters
# --------------------------------------------------------------------------------------------------------------
import torch, sys

def set_param_for_training(
    module_rootdir: str,
    dataset_rootdir: str,
    label_rootdir: str,
    batch_size: int,
    optim: str,
    max_train_iter: int,
    initial_lr: float,
    wt_decay: float,
    starting_iter_num: int = 0, 
    load_model_weights_train: bool = False, 
    trained_weights_path: str = ''):

    sys.path.append(module_rootdir)
    from modules.neural_net.backbone.backbone_v2 import net_backbone
    from modules.neural_net.bifpn.bifpn_nblks_v2 import BiFPN
    from modules.neural_net.head.shared_head_v5 import SharedNet
    from modules.first_stage.detector import Detector
    from modules.first_stage.detector_train import Detector_Train
    from modules.first_stage.first_stage_loss import Loss
    from modules.first_stage.get_parameters import (
        bdd_parameters, kitti_parameters, net_config, print_parameters, get_device, reset_seed )
    from modules.first_stage.get_datasets import BDD_dataset, KITTI_dataset, DATSET_Selector

    reset_seed(0)

    # ================================================> SAVED MODEL WEIGHTS <========================================
    # incase we would like to resume training from a model weight checkpoint, set 'load_model_weights' as True and
    # set the weights_path
    load_model_weights = load_model_weights_train
    weights_path = trained_weights_path

    # ======================================> GET MODEL CONFIGURATION & OTHER PARAMETERS <===========================================
    device = get_device()
    net_config_obj = net_config()
    bdd_param_obj = bdd_parameters()
    kitti_param_obj = kitti_parameters()
    print_parameters(net_config_obj, bdd_param_obj, kitti_param_obj, device)

    # ===============================================> INIT NETWORK STRUCTURE <======================================================
    backbone = net_backbone(net_config_obj)
    bifpn = BiFPN(net_config_obj)
    shared_head = SharedNet(net_config_obj, bdd_param_obj.out_feat_shape) 
    detector = Detector(backbone, bifpn, shared_head)

    loss = Loss(net_config_obj, bdd_param_obj, device)
    detector_train = Detector_Train(detector, loss, bdd_param_obj, device)
    if load_model_weights:
        detector_train.load_state_dict(torch.load(weights_path, map_location="cpu"))
    detector_train = detector_train.to(device)

    # ============================================> SET OPTIMIZATION PARAMETERS <==================================================
    max_iters = max_train_iter
    learning_rate = initial_lr
    weight_decay = wt_decay
    params = [p for p in detector_train.parameters() if p.requires_grad]

    if optim == 'sgd':
        optimizer = torch.optim.SGD(params, momentum=0.9, lr=learning_rate, weight_decay=weight_decay)

    if optim == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    # in case we have to abruptly stop training and resume the training at a later time
    init_start = starting_iter_num # ==> start from this iteration  
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR( 
        optimizer, 
        gamma=0.1,
        milestones=[int(0.65 * max_iters - init_start), 
                    int(0.95 * max_iters - init_start)])
    
    # ==============================================> DATASET & DATALOADER <===================================================
    bdd_dataloader = BDD_dataset(
        label_rootdir = label_rootdir,
        dataset_rootdir = dataset_rootdir,
        batch_size = batch_size,
        num_samples_val = 500, 
        bdd_param_obj = bdd_param_obj,
        device = device,
        shuffle_dataset = False,
        perform_augmentation_train = True,
        augmentation_prob_train = 0.99)

    kitti_dataloader = KITTI_dataset(
        label_rootdir = label_rootdir,
        dataset_rootdir = dataset_rootdir,
        batch_size = batch_size,
        num_samples_val = 500, 
        kitti_param_obj = kitti_param_obj,
        device = device,
        shuffle_dataset = True,
        perform_augmentation_train = True,
        augmentation_prob_train = 0.99)

    dataloader_selector = DATSET_Selector(
        bdd_dataset_obj = bdd_dataloader,
        kitti_dataset_obj = kitti_dataloader,
        max_training_iter = max_iters,
        bdd_dataset_weight = 0.8)
    
    return {
        'detector': detector_train,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler,
        'dataloader_selector': dataloader_selector}

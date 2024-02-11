# --------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : inference parameters
# --------------------------------------------------------------------------------------------------------------
import torch, sys

def set_param_for_inference(
    dataset_type: str,
    module_rootdir: str,
    dataset_rootdir: str,
    label_rootdir: str,
    batch_size: int,
    trained_weights_file: str,
    shuffle_dataset: bool = False,
    perform_augmentation_train: bool = False,
    num_samples_val: int = -1):

    sys.path.append(module_rootdir)
    from modules.neural_net.backbone.backbone_v2 import net_backbone
    from modules.neural_net.bifpn.bifpn_nblks_v2 import BiFPN
    from modules.neural_net.head.shared_head_v5 import SharedNet
    from modules.first_stage.detector import Detector
    from modules.first_stage.detector_train import Detector_Train
    from modules.first_stage.first_stage_loss import Loss
    from modules.first_stage.get_parameters import (
        bdd_parameters, kitti_parameters, net_config, print_parameters, get_device )
    from modules.first_stage.get_datasets import BDD_dataset, KITTI_dataset

    # ======================================> GET MODEL CONFIGURATION & OTHER PARAMETERS <===========================================
    device = get_device()
    net_config_obj = net_config()
    bdd_param_obj = bdd_parameters()
    kitti_param_obj = kitti_parameters()
    print_parameters(net_config_obj, bdd_param_obj, kitti_param_obj, device)

    # ===============================================> INIT NETWORK STRUCTURE <======================================================
    if dataset_type == 'bdd': datasetparam_obj = bdd_param_obj
    elif dataset_type == 'kitti': datasetparam_obj = kitti_param_obj
    
    backbone = net_backbone(net_config_obj)
    bifpn = BiFPN(net_config_obj)
    shared_head = SharedNet(net_config_obj, datasetparam_obj.out_feat_shape) 
    detector = Detector(backbone, bifpn, shared_head)

    loss = Loss(net_config_obj, datasetparam_obj, device)
    detector_train = Detector_Train(detector, loss, datasetparam_obj, device)
    detector_train.load_state_dict(torch.load(trained_weights_file, map_location="cpu"))
    detector_train = detector_train.to(device)

    # ==============================================> DATASET & DATALOADER <===================================================
    if dataset_type == 'bdd':
        dataloader = BDD_dataset(
            label_rootdir = label_rootdir,
            dataset_rootdir = dataset_rootdir,
            batch_size = batch_size,
            num_samples_val = num_samples_val, 
            bdd_param_obj = bdd_param_obj,
            device = device,
            shuffle_dataset = shuffle_dataset,
            perform_augmentation_train = perform_augmentation_train)

    elif dataset_type == 'kitti':
        dataloader = KITTI_dataset(
            label_rootdir = label_rootdir,
            dataset_rootdir = dataset_rootdir,
            batch_size = batch_size,
            num_samples_val = num_samples_val, 
            kitti_param_obj = kitti_param_obj,
            device = device,
            shuffle_dataset = shuffle_dataset,
            perform_augmentation_train = perform_augmentation_train)
        
    return {
        'device': device,
        'model_config': net_config_obj,
        'dataset_param': datasetparam_obj,
        'detector': detector_train.detector.eval(),
        'dataset_train': dataloader.dataset_train,
        'dataset_val': dataloader.dataset_val}


# --------------------------------------------------------------------------------------------------------------
def set_param_for_video_inference(
    dataset_type: str,
    module_rootdir: str,
    trained_weights_file: str):

    sys.path.append(module_rootdir)
    from modules.neural_net.backbone.backbone_v2 import net_backbone
    from modules.neural_net.bifpn.bifpn_nblks_v2 import BiFPN
    from modules.neural_net.head.shared_head_v5 import SharedNet
    from modules.first_stage.detector import Detector
    from modules.first_stage.detector_train import Detector_Train
    from modules.first_stage.first_stage_loss import Loss
    from modules.first_stage.get_parameters import (
        bdd_parameters, kitti_parameters, net_config, print_parameters, get_device )

    # ======================================> GET MODEL CONFIGURATION & OTHER PARAMETERS <===========================================
    device = get_device()
    net_config_obj = net_config()
    bdd_param_obj = bdd_parameters()
    kitti_param_obj = kitti_parameters()
    print_parameters(net_config_obj, bdd_param_obj, kitti_param_obj, device)

    # ===============================================> INIT NETWORK STRUCTURE <======================================================
    if dataset_type == 'bdd': datasetparam_obj = bdd_param_obj
    elif dataset_type == 'kitti': datasetparam_obj = kitti_param_obj
    
    backbone = net_backbone(net_config_obj)
    bifpn = BiFPN(net_config_obj)
    shared_head = SharedNet(net_config_obj, datasetparam_obj.out_feat_shape) 
    detector = Detector(backbone, bifpn, shared_head)

    loss = Loss(net_config_obj, datasetparam_obj, device)
    detector_train = Detector_Train(detector, loss, datasetparam_obj, device)
    detector_train.load_state_dict(torch.load(trained_weights_file, map_location="cpu"))
    detector_train = detector_train.to(device)

    return {
        'device': device,
        'dataset_param': datasetparam_obj,
        'detector': detector_train.detector.eval()}
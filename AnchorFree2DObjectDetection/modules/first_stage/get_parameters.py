# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Classes that encapsulates the model parameters, various dataset parameters
#               Depending on what dataset samples are in a specific batch, The forward propagation will need a 
#               different dataset related parameters. For e.g. the output feature map resolution is different for KITTI and BDD. 
#               We have to incorporate a procedure that will 'update' the feature map resolution values in the forward propagation 
#               of the network as per what dataset corrosponds to that batch         
# ---------------------------------------------------------------------------------------------------------------------
from modules.proposal.box_functions import gen_grid_coord
from modules.pretrained.utils_backbone_cfg import get_feat_shapes, extract_fpn_featmap_height_and_width
import modules.dataset_utils.bdd_dataset_utils.constants as bdd_dataset_utils_const
import modules.dataset_utils.kitti_dataset_utils.constants as kitti_dataset_utils_const
from modules.proposal.constants import _IGNORED_CLASS_DEFAULT_ID_
from modules.dataset_utils.kitti_dataset_utils.constants import _IGNORED_CLASS_ID_
import config_neuralnet_stage1 as config_neuralnet
import config_dataset
import torch
import os

# --------------------------------------------------------------------------------------------------------------
class net_config:
    def __init__(self):
        self.num_classes = config_neuralnet.num_classes
        self.basenet = config_neuralnet.basenet
        self.freeze_backbone_layers = config_neuralnet.freeze_backbone_layers
        self.num_backbone_nodes = config_neuralnet.num_backbone_nodes
        self.num_extra_blocks = config_neuralnet.num_extra_blocks
        self.in_channels_extra_blks = None
        self.out_channels_extra_blks = config_neuralnet.extra_blocks_feat_dim
        self.num_levels = config_neuralnet.num_backbone_nodes + config_neuralnet.num_extra_blocks
        self.num_fpn_blocks = config_neuralnet.num_fpn_blocks                                     # 2
        self.fpn_feat_dim = config_neuralnet.fpn_feat_dim                                         # 128
        self.stem_channels = [config_neuralnet.fpn_feat_dim] * config_neuralnet.num_stem_blocks   # 4  [n, n, n, n]
        self.activation = config_neuralnet.activation                                             # 'swish'
        self.deltas_mean = config_neuralnet.DELTAS_MEAN
        self.deltas_std = config_neuralnet.DELTAS_STD
        self.class_weights = config_neuralnet.CLASS_WEIGHTS
        self.in_channels_extra_blks = None
        self.feat_pyr_channels = None
        self.set_channels()

    def set_channels(self):
        input_image_shape = (bdd_dataset_utils_const._IMG_RESIZED_H_, 
                             bdd_dataset_utils_const._IMG_RESIZED_W_, 
                             bdd_dataset_utils_const._IMG_D_)
        feat_pyr_shapes = get_feat_shapes(           
                    self.basenet, 
                    input_image_shape, 
                    self.num_backbone_nodes, 
                    self.num_extra_blocks, 
                    self.out_channels_extra_blks)
        self.in_channels_extra_blks = feat_pyr_shapes[f'c{self.num_backbone_nodes - 1}'][0]

        feat_pyr_channels = {}
        for key, value in feat_pyr_shapes.items():
            feat_pyr_channels[key]  = value[0]
        self.feat_pyr_channels = feat_pyr_channels


# --------------------------------------------------------------------------------------------------------------
class bdd_parameters:
    def __init__(self):
        self.train_images_dir = config_dataset.bdd_train_images_dir
        self.train_labels_file = config_dataset.bdd_train_labels_file
        self.train_lane_labels_file = config_dataset.bdd_train_lane_labels_file

        self.val_images_dir = config_dataset.bdd_val_images_dir
        self.val_labels_file = config_dataset.bdd_val_labels_file
        self.val_lane_labels_file = config_dataset.bdd_val_lane_labels_file

        self.test_images_dir = config_dataset.bdd_test_images_dir
        self.label_out_dir = config_dataset.bdd_label_out_dir

        self.sel_train_labels_file_name = config_dataset.bdd_sel_train_labels_file_name
        self.sel_val_labels_file_name = config_dataset.bdd_sel_val_labels_file_name
        self.sel_train_labels_file = config_dataset.bdd_sel_train_labels_file
        self.sel_val_labels_file = config_dataset.bdd_sel_val_labels_file

        self.IMG_H = bdd_dataset_utils_const._IMG_H_
        self.IMG_W = bdd_dataset_utils_const._IMG_W_
        self.IMG_D = bdd_dataset_utils_const._IMG_D_
        self.IMG_RESIZED_H = bdd_dataset_utils_const._IMG_RESIZED_H_
        self.IMG_RESIZED_W = bdd_dataset_utils_const._IMG_RESIZED_W_
        self.OUT_FEAT_SIZE_H = bdd_dataset_utils_const._OUT_FEAT_SIZE_H_
        self.OUT_FEAT_SIZE_W = bdd_dataset_utils_const._OUT_FEAT_SIZE_W_
        self.STRIDE_H = bdd_dataset_utils_const._STRIDE_H_
        self.STRIDE_W = bdd_dataset_utils_const._STRIDE_W_
        self.input_image_shape = (self.IMG_RESIZED_H, self.IMG_RESIZED_W, self.IMG_D)
        self.out_feat_shape = (self.OUT_FEAT_SIZE_H, self.OUT_FEAT_SIZE_W)
        self.ignored_classId = _IGNORED_CLASS_DEFAULT_ID_

        self.deltas_mean = None
        self.deltas_std = None
        self.feat_pyr_shapes = None
        self.feat_pyr_h = None
        self.feat_pyr_w = None
        self.grid_coord = None

        self.set_deltas_statistic()
        self.set_feat_pyr_shapes()
        self.set_grid_coord()

    def set_feat_pyr_shapes(self):
        self.feat_pyr_shapes \
            = get_feat_shapes(           
                    config_neuralnet.basenet, 
                    self.input_image_shape, 
                    config_neuralnet.num_backbone_nodes, 
                    config_neuralnet.num_extra_blocks, 
                    config_neuralnet.extra_blocks_feat_dim)
        self.feat_pyr_h, self.feat_pyr_w \
            = extract_fpn_featmap_height_and_width(self.feat_pyr_shapes)

    def set_grid_coord(self):
        self.grid_coord \
            = gen_grid_coord(
                    self.OUT_FEAT_SIZE_W, self.OUT_FEAT_SIZE_H, 
                    self.STRIDE_W, self.STRIDE_H)
    
    def set_deltas_statistic(self):
        self.deltas_mean = config_neuralnet.DELTAS_MEAN
        self.deltas_std = config_neuralnet.DELTAS_STD

# --------------------------------------------------------------------------------------------------------------      
class kitti_parameters:
    def __init__(self):
        self.kitti_label_file_path = os.path.join(config_dataset.kitti_label_dir, config_dataset.kitti_label_file)
        self.kitti_remapped_label_file_path = os.path.join(config_dataset.kitti_label_dir, config_dataset.kitti_remapped_label_file)    
        self.kitti_all_sequences_folders = config_dataset.kitti_all_sequences_folders
        self.kitti_train_sequences_folders = config_dataset.kitti_train_sequences_folders
        self.kitti_val_sequences_folders = config_dataset.kitti_val_sequences_folders
        
        self.IMG_H = kitti_dataset_utils_const._IMG_H_
        self.IMG_W = kitti_dataset_utils_const._IMG_W_
        self.IMG_D = kitti_dataset_utils_const._IMG_D_
        self.IMG_RESIZED_H = kitti_dataset_utils_const._IMG_RESIZED_H_
        self.IMG_RESIZED_W = kitti_dataset_utils_const._IMG_RESIZED_W_
        self.OUT_FEAT_SIZE_H = kitti_dataset_utils_const._OUT_FEAT_SIZE_H_
        self.OUT_FEAT_SIZE_W = kitti_dataset_utils_const._OUT_FEAT_SIZE_W_
        self.STRIDE_H = kitti_dataset_utils_const._STRIDE_H_
        self.STRIDE_W = kitti_dataset_utils_const._STRIDE_W_
        self.input_image_shape = (self.IMG_RESIZED_H, self.IMG_RESIZED_W, self.IMG_D)
        self.out_feat_shape = (self.OUT_FEAT_SIZE_H, self.OUT_FEAT_SIZE_W)
        self.ignored_classId = _IGNORED_CLASS_ID_

        self.deltas_mean = None
        self.deltas_std = None
        self.feat_pyr_shapes = None
        self.feat_pyr_h = None
        self.feat_pyr_w = None
        self.grid_coord = None

        self.set_deltas_statistic()
        self.set_feat_pyr_shapes()
        self.set_grid_coord()

    def set_feat_pyr_shapes(self):
        self.feat_pyr_shapes \
            = get_feat_shapes(           
                    config_neuralnet.basenet, 
                    self.input_image_shape, 
                    config_neuralnet.num_backbone_nodes, 
                    config_neuralnet.num_extra_blocks, 
                    config_neuralnet.extra_blocks_feat_dim)
        self.feat_pyr_h, self.feat_pyr_w \
            = extract_fpn_featmap_height_and_width(self.feat_pyr_shapes)
        
    def set_grid_coord(self):
        self.grid_coord \
            = gen_grid_coord(
                    self.OUT_FEAT_SIZE_W, self.OUT_FEAT_SIZE_H, 
                    self.STRIDE_W, self.STRIDE_H)
    
    def set_deltas_statistic(self):
        self.deltas_mean = config_neuralnet.DELTAS_MEAN
        self.deltas_std = config_neuralnet.DELTAS_STD

# --------------------------------------------------------------------------------------------------------------
def reset_seed(number):
    """ Reset random seed to the specific number
    Inputs- number: A seed number to use
    """
    import random
    random.seed(number)
    torch.manual_seed(number)
    return

def get_device():
    if torch.cuda.is_available():
        print("GPU is available. Good to go!")
        DEVICE = torch.device("cuda")
    else:
        print("Only CPU is available.")
        DEVICE = torch.device("cpu")
    return DEVICE

# --------------------------------------------------------------------------------------------------------------
def print_parameters(
    net_config_obj: net_config, 
    bdd_parameters_obj: bdd_parameters, 
    kitti_parameters_obj: kitti_parameters, 
    device):

    basenet = net_config_obj.basenet
    num_backbone_nodes = net_config_obj.num_backbone_nodes
    num_extra_blocks = net_config_obj.num_extra_blocks
    out_channels_extra_blks = net_config_obj.out_channels_extra_blks
    num_levels = net_config_obj.num_backbone_nodes + net_config_obj.num_extra_blocks
    num_fpn_blocks = net_config_obj.num_fpn_blocks                                     # 2
    fpn_feat_dim = net_config_obj.fpn_feat_dim                                         # 128
    stem_channels = net_config_obj.stem_channels                                       # 3  [n, n, n]
    activation = net_config_obj.activation                                             # 'swish'

    bdd_img_h = bdd_parameters_obj.IMG_RESIZED_H
    bdd_img_w = bdd_parameters_obj.IMG_RESIZED_W
    kitti_img_h = kitti_parameters_obj.IMG_RESIZED_H
    kitti_img_w = kitti_parameters_obj.IMG_RESIZED_W
    img_d = bdd_parameters_obj.IMG_D
    num_classes = net_config_obj.num_classes

    print('printing model config parameters')
    print('-' * 100)
    print('backbone                        :', basenet)
    print('num_backbone_nodes              :', num_backbone_nodes)
    print('num_extra_blocks                :', num_extra_blocks)
    print('num_levels                      :', num_levels)
    if num_extra_blocks > 0:
        print('extra_blocks_feat_dim           :', out_channels_extra_blks)
    print('num_fpn_blocks                  :', num_fpn_blocks)
    print('fpn_feat_dim                    :', fpn_feat_dim)
    print('prediction head stem_channels   :', stem_channels)
    print('activation                      :', activation)
    print('image dimension BDD (H, W, D)   :', (bdd_img_h, bdd_img_w, img_d))
    print('image dimension KITTI (H, W, D) :', (kitti_img_h, kitti_img_w, img_d))
    print('num_classes                     :', num_classes)
    print('DEVICE                          :', device)
    print('*' * 100)
    print(' ')

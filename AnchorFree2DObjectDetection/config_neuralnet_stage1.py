# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Architecture parameters for dense predictor
# ---------------------------------------------------------------------------------------------------------------
# model architecture
basenet = 'efficientnet_b4'
num_backbone_nodes = 4
num_extra_blocks = 1
extra_blocks_feat_dim = 512
num_fpn_blocks = 2
fpn_feat_dim = 128
num_stem_blocks = 4
num_classes = 2         # vehicles and person
activation = 'swish'
freeze_backbone_layers = True

# ---------------------------------------------------------------------------------------------------------------
# box regression offset statistic and class weights 
DELTAS_MEAN = [3.940856695175171, 3.7502336502075195, 3.932513952255249, 3.731210231781006]
DELTAS_STD = [0.918531060218811, 0.9232671856880188, 0.9193117618560791, 0.9233105182647705]
CLASS_WEIGHTS = [0.62545866, 1.0]  # vehicles and person

# ---------------------------------------------------------------------------------------------------------------
# loss weights
CLS_LOSS_WT = 1.0
OBJ_LOSS_WT = 1.0
CTR_LOSS_WT = 1.0
BOX_LOSS_WT = 1.0

# ---------------------------------------------------------------------------------------------------------------
# directory and filename to store the trained model weights
model_weights_main_dir = 'model_weights'
weights_name = 'anchor_free_detector.pt'
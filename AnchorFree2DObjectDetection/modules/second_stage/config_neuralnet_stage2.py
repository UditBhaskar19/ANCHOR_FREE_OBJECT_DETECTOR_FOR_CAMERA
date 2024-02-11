# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : < Work in progress >        
# ---------------------------------------------------------------------------------------------------------------------
# model architecture
dropout = 0.1
freeze_singlestage_layers = True
iou_threshold_for_gt_matching = 0.3
nms_thr_for_proposal_extraction = 0.35
score_thr_for_proposal_extraction = [0.1, 0.05]   #[0.2, 0.1]
roi_size = 11
feat_embedding_nonshared_stem_channels = [128, 128]
feat_embedding_stem_channels = [128, 128, 128, 128]
output_dimension = 1

# ---------------------------------------------------------------------------------------------------------------
# box regression offset statistic and class weights
DELTAS_MEAN = [-0.002929835580289364, -0.008169739507138729, -0.05080288648605347, -0.007937428541481495]
DELTAS_STD = [0.12428951263427734, 0.09890233725309372, 0.24964886903762817, 0.2034754604101181]
CLASS_WEIGHTS = [0.62545866, 1.0]
NEG_POS_CLASS_WEIGHTS = [0.5, 1.0]

# ---------------------------------------------------------------------------------------------------------------
# loss weights
CLS_LOSS_WT = 1.0
OBJ_LOSS_WT = 1.0
BOX_LOSS_WT = 1.0

# ---------------------------------------------------------------------------------------------------------------
# directory and filename to store the trained model weights
model_weights_main_dir = 'model_weights_second_stage'
weights_name = 'detector_second_stage.pt'

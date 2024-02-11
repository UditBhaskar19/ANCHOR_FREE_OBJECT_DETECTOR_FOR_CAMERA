# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : < Work in progress >        
# ---------------------------------------------------------------------------------------------------------------------
import config_neuralnet_stage1
from modules.second_stage import config_neuralnet_stage2
from modules.first_stage.get_parameters import net_config as net_config_stage1
from modules.first_stage.get_parameters import bdd_parameters as bdd_parameters_stage1
from modules.first_stage.get_parameters import kitti_parameters as kittti_parameters_stage1
from modules.proposal.constants import _IGNORED_CLASS_DEFAULT_ID_
from modules.dataset_utils.kitti_dataset_utils.constants import _IGNORED_CLASS_ID_

# --------------------------------------------------------------------------------------------------------------
class net_config_stage2(net_config_stage1):
    def __init__(self):
        super().__init__()

        self.objness_weights_stage2 = config_neuralnet_stage2.NEG_POS_CLASS_WEIGHTS
        self.class_weights_stage2 = config_neuralnet_stage2.CLASS_WEIGHTS
        self.deltas_mean_stage2 = config_neuralnet_stage2.DELTAS_MEAN
        self.deltas_std_stage2 = config_neuralnet_stage2.DELTAS_STD

        self.feat_embedding_inchannels_stage2 = config_neuralnet_stage1.fpn_feat_dim   
        self.activation_stage2 = config_neuralnet_stage1.activation
        self.dropout_stage2 = config_neuralnet_stage2.dropout

        self.freeze_singlestage_layers_stage2 = config_neuralnet_stage2.freeze_singlestage_layers
        self.nms_thr_for_proposal_extraction_stage2 = config_neuralnet_stage2.nms_thr_for_proposal_extraction
        self.score_thr_for_proposal_extraction_stage2 = config_neuralnet_stage2.score_thr_for_proposal_extraction
        self.roi_size_stage2 = config_neuralnet_stage2.roi_size

        self.feat_embedding_nonshared_stem_channels_stage2 = config_neuralnet_stage2.feat_embedding_nonshared_stem_channels
        self.feat_embedding_stem_channels_stage2 = config_neuralnet_stage2.feat_embedding_stem_channels
        self.output_dimension_stage2 = config_neuralnet_stage2.output_dimension

# --------------------------------------------------------------------------------------------------------------
class bdd_parameters_stage2(bdd_parameters_stage1):
    def __init__(self):
        super().__init__()
        self.ignored_classId_stage2 = _IGNORED_CLASS_DEFAULT_ID_
        self.iou_threshold_stage2 = config_neuralnet_stage2.iou_threshold_for_gt_matching
        self.deltas_mean_stage2 = config_neuralnet_stage2.DELTAS_MEAN
        self.deltas_std_stage2 = config_neuralnet_stage2.DELTAS_STD

# --------------------------------------------------------------------------------------------------------------
class kitti_parameters_stage2(kittti_parameters_stage1):
    def __init__(self):
        super().__init__()
        self.ignored_classId_stage2 = _IGNORED_CLASS_ID_
        self.iou_threshold_stage2 = config_neuralnet_stage2.iou_threshold_for_gt_matching
        self.deltas_mean_stage2 = config_neuralnet_stage2.DELTAS_MEAN
        self.deltas_std_stage2 = config_neuralnet_stage2.DELTAS_STD
# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : < Work in progress >        
# ---------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
from typing import Tuple, List, Union
from collections import namedtuple
from modules.second_stage.roi_embedding import query_embedding, featmap_embedding
from modules.second_stage.attention import attention_network
from modules.second_stage.second_stage_loss import second_stage_loss
from modules.second_stage.generate_gt import gen_training_gt
from modules.second_stage.get_param import bdd_parameters_stage2 as bdd_parameters
from modules.second_stage.get_param import kitti_parameters_stage2 as kitti_parameters

# ---------------------------------------------------------------------------------------------------------------------
class second_stage_detector(nn.Module):
    def __init__(
        self, 
        feat_embedding_net: featmap_embedding, 
        query_embedding_net: query_embedding,
        attention_net: attention_network):
        super().__init__()
        self.feat_embedding_net = feat_embedding_net
        self.query_embedding_net = query_embedding_net
        self.attention_net = attention_net

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        roi_features, queries = x
        roi_embedding = self.feat_embedding_net(roi_features)
        query_embeddings = self.query_embedding_net(queries)
        predictions = self.attention_net((roi_embedding, query_embeddings))
        return predictions

# ---------------------------------------------------------------------------------------------------------------------
class second_stage_detector_train(nn.Module):
    def __init__(
        self,
        detector: second_stage_detector,
        loss_obj: second_stage_loss,
        param_obj: Union[bdd_parameters, kitti_parameters],
        device: str):
        super().__init__()

        self.device = device
        self.detector = detector
        self.loss_fn = loss_obj
        self.ignored_classId = param_obj.ignored_classId_stage2
        self.iou_threshold = param_obj.iou_threshold_stage2
        self.deltas_mean = torch.tensor(param_obj.deltas_mean_stage2, dtype=torch.float32, device=device)
        self.deltas_std = torch.tensor(param_obj.deltas_std_stage2, dtype=torch.float32, device=device)

    def reinit_const_parameters(
        self, 
        const_param_obj: Union[bdd_parameters, kitti_parameters]):
        self.ignored_classId = const_param_obj.ignored_classId_stage2
        self.iou_threshold = const_param_obj.iou_threshold_stage2
        self.deltas_mean = torch.tensor(const_param_obj.deltas_mean_stage2, dtype=torch.float32, device=self.device)
        self.deltas_std = torch.tensor(const_param_obj.deltas_std_stage2, dtype=torch.float32, device=self.device)

    def forward(
        self, 
        x: Tuple[torch.Tensor, torch.Tensor],
        gtbboxes: List[torch.Tensor],
        gtclslabels: List[torch.Tensor],
        proposals: List[torch.Tensor]):

        predictions = self.detector(x)

        groundtruths = gen_training_gt(
            gt_boxes = gtbboxes, 
            gt_class = gtclslabels,
            pred_boxes = proposals,
            deltas_mean = self.deltas_mean, 
            deltas_std = self.deltas_std, 
            iou_threshold = self.iou_threshold,
            ignored_classId = self.ignored_classId)
        
        loss = self.loss_fn(predictions, groundtruths)

        return {'loss_obj': loss['loss_obj'] }
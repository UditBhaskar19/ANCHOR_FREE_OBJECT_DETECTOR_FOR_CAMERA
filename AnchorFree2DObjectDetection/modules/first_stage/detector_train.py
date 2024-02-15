# --------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Model to be used during training
# --------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
from typing import List, Dict, Union
from modules.first_stage.generate_gt import gen_training_gt
from modules.first_stage.get_parameters import bdd_parameters, kitti_parameters

# --------------------------------------------------------------------------------------------------------------
class Detector_Train(nn.Module):
    def __init__(
        self, 
        detector_obj: nn.Module, 
        loss_obj: nn.Module,
        const_param_obj: Union[bdd_parameters, kitti_parameters],
        device: str):
        super().__init__()

        self.device = device
        self.detector = detector_obj
        self.loss_fn = loss_obj

        self.strides_width = const_param_obj.STRIDE_W
        self.strides_height = const_param_obj.STRIDE_H
        self.grid_coord = const_param_obj.grid_coord.to(device)
        self.ignored_classId = const_param_obj.ignored_classId

        self.deltas_mean = torch.tensor(const_param_obj.deltas_mean, dtype=torch.float32, device=device)
        self.deltas_std = torch.tensor(const_param_obj.deltas_std, dtype=torch.float32, device=device)

    
    def reinit_const_parameters(self, const_param_obj: Union[bdd_parameters, kitti_parameters]):

        self.strides_width = const_param_obj.STRIDE_W
        self.strides_height = const_param_obj.STRIDE_H

        self.grid_coord = const_param_obj.grid_coord.to(self.device)
        self.ignored_classId = const_param_obj.ignored_classId

        self.detector.sharednet.merge_blk.out_feat_h = const_param_obj.OUT_FEAT_SIZE_H
        self.detector.sharednet.merge_blk.out_feat_w = const_param_obj.OUT_FEAT_SIZE_W
        self.loss_fn.grid_coord = const_param_obj.grid_coord

        # num_feataggregator_blks = len(self.detector.feataggregator.BiFPN_layers)
        # for i in range(num_feataggregator_blks):
        #     self.detector.feataggregator.BiFPN_layers[i].feat_pyr_shapes = const_param_obj.feat_pyr_shapes


    def forward(
        self, 
        images: torch.Tensor, 
        bboxes: List[torch.Tensor],
        clslabels: List[torch.Tensor]):

        predictions = self.detector(images)

        groundtruths = gen_training_gt(
            gt_boxes = bboxes, 
            gt_class = clslabels, 
            strides_width = self.strides_width,
            strides_height = self.strides_height,
            grid_coord = self.grid_coord,
            deltas_mean = self.deltas_mean,
            deltas_std = self.deltas_std,
            device = self.device,
            ignored_classId = self.ignored_classId)
        
        loss = self.loss_fn(predictions, groundtruths)
        return {
            'loss_cls': loss['loss_cls'],
            'loss_box': loss['loss_box'],
            'loss_ctr': loss['loss_ctr'],
            'loss_obj': loss['loss_obj'],
        }
    

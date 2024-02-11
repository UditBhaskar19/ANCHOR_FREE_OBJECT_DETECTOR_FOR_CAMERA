# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : < Work in progress >        
# ---------------------------------------------------------------------------------------------------------------------
import torch 
import torch.nn as nn
from modules.loss.basic_loss import CE_loss, Smooth_L1_Loss, Focal_Loss
from modules.second_stage.get_param import net_config_stage2 as net_config

# --------------------------------------------------------------------------------------------------------------
class second_stage_loss(nn.Module):
    def __init__(self, device: str):
        super().__init__()

        self.obj_loss = Focal_Loss()
        self.device = device

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):

        # extract and gt data (note: the gt class labels are not in one-hot form)
        gt_class_logits = gt.class_logits.view(-1)
        gt_objness_logits = gt.objness_logits.view(-1)

        # extract and flatten predictions
        pred_objness_logits = pred.view(-1)

        # compute positive, negative and ignored mask
        ignored_mask = gt_class_logits == -2
        neg_mask = gt_class_logits == -1
        pos_mask = gt_class_logits >= 0

        # compute loss
        obj_loss = self.obj_loss(pred_objness_logits, gt_objness_logits)

        # set all losses to 0 if no +ve samples are present
        # set box and centreness loss to 0 for -ve samples
        N = pos_mask.sum().item()
        if N == 0: obj_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        else: obj_loss = torch.where(~ignored_mask, obj_loss, 0.0).sum() / N

        return { "loss_obj": obj_loss }


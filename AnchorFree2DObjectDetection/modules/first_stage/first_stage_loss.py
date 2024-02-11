# --------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Loss module
# --------------------------------------------------------------------------------------------------------------
from typing import Union
from modules.proposal.prop_functions import compute_bbox_from_offsets_normalized, compute_bbox_from_deltas_normalized
from modules.proposal.box_functions import gen_grid_coord
from modules.loss.basic_loss import *
from modules.first_stage.get_parameters import net_config, bdd_parameters, kitti_parameters
import torch
from torch import nn

# --------------------------------------------------------------------------------------------------------------
class Loss(nn.Module):
    def __init__(
        self, 
        net_config_obj: net_config, 
        dataset_config_obj: Union[bdd_parameters, kitti_parameters],
        device: str):
        super().__init__()

        self.cls_loss = CE_loss()
        # Distance_Box_IOU_Loss(), Generalized_Box_IOU_Loss(), Complete_Box_IOU_Loss(), Smooth_L1_Loss()
        self.box_loss = Smooth_L1_Loss() 
        self.cntr_loss = BCE_loss()
        self.obj_loss = Focal_Loss()
        self.num_classes = net_config_obj.num_classes
        self.class_weights = torch.tensor(net_config_obj.class_weights, dtype=torch.float32, device=device) 
        self.deltas_mean = torch.tensor(net_config_obj.deltas_mean, dtype=torch.float32, device=device)
        self.deltas_std = torch.tensor(net_config_obj.deltas_std, dtype=torch.float32, device=device)
        self.device = device
        self.grid_coord = dataset_config_obj.grid_coord


    def forward(self, pred: torch.Tensor, gt: torch.Tensor):

        # extract and flatten gt data (note: the gt class labels are not in one-hot form)
        gt_class_logits = gt.class_logits.view(-1)
        gt_boxreg_deltas = gt.boxreg_deltas.view(-1, gt.boxreg_deltas.shape[-1])
        gt_centerness_logits = gt.centerness_logits.view(-1)
        gt_objness_logits = gt.objness_logits.view(-1)
        gt_bbox = gt.bbox.view(-1, gt.bbox.shape[-1])

        # extract and flatten predictions
        pred_class_logits = pred.class_logits.view(-1, pred.class_logits.shape[-1])
        pred_boxreg_deltas = pred.boxreg_deltas.view(-1, pred.boxreg_deltas.shape[-1])
        pred_centerness_logits = pred.centerness_logits.view(-1)
        pred_objness_logits = pred.objness_logits.view(-1)

        # # compute predicted bounding box
        # B = gt.boxreg_deltas.shape[0]
        # grid_coord = self.grid_coord.unsqueeze(0).repeat(B, 1, 1).view(-1, 2)
        # pred_boxes = compute_bbox_from_deltas_normalized(
        #     grid_coord, pred_boxreg_deltas, self.deltas_mean, self.deltas_std)

        # compute positive, negative and ignored mask
        ignored_mask = gt_class_logits == -2
        neg_mask = gt_class_logits == -1
        pos_mask = gt_class_logits >= 0

        # compute class labels one-hot
        gt_class_logits[neg_mask] = 0
        gt_class_logits[ignored_mask] = 0
        gt_class_logits = torch.nn.functional.one_hot(gt_class_logits.to(int), self.num_classes).to(torch.float32)
        
        # compute loss
        cls_loss = self.cls_loss(pred_class_logits, gt_class_logits, self.class_weights)
        box_loss = 0.25 * self.box_loss(pred_boxreg_deltas, gt_boxreg_deltas).sum(-1) # for Smooth_L1_Loss()
        # box_loss = self.box_loss(pred_boxes, gt_bbox) # for IOU based box loss functions
        cntr_loss = self.cntr_loss(pred_centerness_logits, gt_centerness_logits)
        obj_loss = self.obj_loss(pred_objness_logits, gt_objness_logits)

        # set all losses to 0 if no +ve samples are present
        # set box and centreness loss to 0 for -ve samples
        N = pos_mask.sum().item()
        if N == 0:
            cls_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
            box_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
            cntr_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
            obj_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        else:
            cls_loss = torch.where(pos_mask, cls_loss, 0.0).sum() / N
            box_loss = torch.where(pos_mask, box_loss, 0.0).sum() / N
            cntr_loss = torch.where(pos_mask, cntr_loss, 0.0).sum() / N
            obj_loss = torch.where(~ignored_mask, obj_loss, 0.0).sum() / N

        return {
            "loss_cls": cls_loss,
            "loss_box": box_loss,
            "loss_ctr": cntr_loss,
            "loss_obj": obj_loss,
        }
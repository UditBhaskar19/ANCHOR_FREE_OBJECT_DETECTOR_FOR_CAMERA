# ------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : common loss functions
# ------------------------------------------------------------------------------------------------------------------
import torchvision
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

""" Here it is assumed that both y_pred and y_true have the shape: (num_samples, sample_dim). If y_pred or y_true 
have shapes (num_batches, num_samples, sample_dim), The tensors need to be reshaped to (num_batches*num_samples, sample_dim).
Inputs: y_pred - tensor of predicted values with shape (N, D)
      : y_true - tensor of ground-truth values with shape (N, D)
      : weight - tensor of weights with shape (N, D) or (N, ) : uunormalized weights
"""
# ------------------------------------------------------------------------------------------------------------------
class CE_loss(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weight: Optional[torch.Tensor] = None):
        if weight == None: return F.cross_entropy(y_pred, y_true, reduction=self.reduction)
        else: return F.cross_entropy(y_pred, y_true, weight, reduction=self.reduction)

# ------------------------------------------------------------------------------------------------------------------
class BCE_loss(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weight: Optional[torch.Tensor] = None):
        if weight == None: return F.binary_cross_entropy_with_logits(y_pred, y_true, reduction=self.reduction)
        else: return F.binary_cross_entropy_with_logits(y_pred, y_true, weight, reduction=self.reduction)

# ------------------------------------------------------------------------------------------------------------------
class Focal_Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return torchvision.ops.sigmoid_focal_loss(y_pred, y_true, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
    
# ------------------------------------------------------------------------------------------------------------------
class Smooth_L1_Loss(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return F.smooth_l1_loss(y_pred, y_true, reduction=self.reduction)
    
# ------------------------------------------------------------------------------------------------------------------
class L1_Loss(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return F.l1_loss(y_pred, y_true, reduction=self.reduction)
    
# ------------------------------------------------------------------------------------------------------------------
class MSE_loss(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return F.mse_loss(y_pred, y_true, reduction=self.reduction)
    
# ------------------------------------------------------------------------------------------------------------------
class Complete_Box_IOU_Loss(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return torchvision.ops.complete_box_iou_loss(y_pred, y_true, reduction=self.reduction)
    
# ------------------------------------------------------------------------------------------------------------------
class Distance_Box_IOU_Loss(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return torchvision.ops.distance_box_iou_loss(y_pred, y_true, reduction=self.reduction)
    
# ------------------------------------------------------------------------------------------------------------------
class Generalized_Box_IOU_Loss(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return torchvision.ops.generalized_box_iou_loss(y_pred, y_true, reduction=self.reduction)
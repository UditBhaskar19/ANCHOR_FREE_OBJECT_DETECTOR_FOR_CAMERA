# --------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Model to be used during inference
# --------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

# --------------------------------------------------------------------------------------------------------------
class Detector(nn.Module):
    def __init__(
        self, 
        backbone_obj: nn.Module, 
        feataggregator_obj: nn.Module, 
        sharednet_obj: nn.Module):
        super().__init__()

        self.backbone = backbone_obj
        self.feataggregator = feataggregator_obj
        self.sharednet = sharednet_obj
    
    def forward(
        self, 
        image: torch.Tensor):
        x = self.backbone(image)
        x = self.feataggregator(x)
        x = self.sharednet(x)
        return x
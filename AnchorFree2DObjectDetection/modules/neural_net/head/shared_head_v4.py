# ---------------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Shared Network Head
# ---------------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Dict, Tuple
from modules.neural_net.common import WSConv2dBlock, ResidualWSConv2dBlock, Conv2d
from modules.neural_net.constants import (
    _BOX_SCALING_WT_INIT_, _NUM_GROUPS_, _INTERP_MODE_, det_named_tuple,
    _CLS_CONV_MEAN_INIT_, _CLS_CONV_STD_INIT_, _CLS_CONV_BIAS_INIT_,
    _BOX_CONV_MEAN_INIT_, _BOX_CONV_STD_INIT_, _BOX_CONV_BIAS_INIT_,
    _CTR_CONV_MEAN_INIT_, _CTR_CONV_STD_INIT_, _CTR_CONV_BIAS_INIT_)

# ---------------------------------------------------------------------------------------------------------------------------
class BoxHeadExp(nn.Module):
    def __init__(self):
        super().__init__()
        self.wt = nn.Parameter(torch.tensor(_BOX_SCALING_WT_INIT_))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.wt * x)
    
# ---------------------------------------------------------------------------------------------------------------------------
class MergeFeatures(nn.Module):
    def __init__(
        self,
        num_levels: int,
        in_channels: int,
        out_channels: int,
        activation: str,
        out_feat_shape: Tuple[int, int]):  # (height, width)):
        super().__init__()

        self.out_feat_h, self.out_feat_w = out_feat_shape
        self.num_levels = num_levels

        # after upsampling
        _wsconv2dblk = {}
        for i in range(num_levels):
            _wsconv2dblk[f"c{i}"] = WSConv2dBlock(_NUM_GROUPS_, in_channels, out_channels, activation)
        self.weights = nn.ModuleDict(_wsconv2dblk)


    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        # upsample the feature maps to dimansion ( out_feat_h, out_feat_w )
        upsampled_maps = []
        for level, feat in x.items():
            feat = F.interpolate(feat, size=( self.out_feat_h, self.out_feat_w ), mode=_INTERP_MODE_)
            feat = self.weights[level](feat)
            upsampled_maps.append(feat)

        # add the upsampled feature maps
        featmap = upsampled_maps[0]
        for i in range(1, self.num_levels):
            featmap += upsampled_maps[i]
        return featmap
    
# ---------------------------------------------------------------------------------------------------------------------------
class StemBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        stem_channels: List[int],
        activation: str):
        super().__init__()
        
        stem_layers = []
        for stem_channel in stem_channels:
            stem_layer = WSConv2dBlock(_NUM_GROUPS_, in_channels, stem_channel, activation)
            stem_layers += [ stem_layer ]
            in_channels = stem_channel
        self.stem_layers = nn.Sequential(*stem_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_layers(x)
        return x
    
# ---------------------------------------------------------------------------------------------------------------------------
class ResidualStemBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        stem_channels: List[int],
        activation: str):
        super().__init__()

        # stem blocks
        stem_layers = []
        for stem_channel in stem_channels:
            stem_layer = ResidualWSConv2dBlock(_NUM_GROUPS_, in_channels, stem_channel, activation)
            stem_layers += [ stem_layer ]
            in_channels = stem_channel
        self.stem_layers = nn.Sequential(*stem_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_layers(x)
        return x

# ---------------------------------------------------------------------------------------------------------------------------
class TaskSpecificHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        init_weight_mu: torch.Tensor,
        init_weight_sigma: torch.Tensor,
        init_bias: torch.Tensor):
        super().__init__()

        _wsconv2dblk = WSConv2dBlock(_NUM_GROUPS_, in_channels, in_channels, activation)
        _conv2d = Conv2d(in_channels=in_channels, out_channels=out_channels)

        torch.nn.init.normal_(_conv2d.conv.weight, mean=init_weight_mu, std=init_weight_sigma)
        torch.nn.init.constant_(_conv2d.conv.bias, init_bias) 

        head = [ _wsconv2dblk, _conv2d ]
        self.head = nn.Sequential(*head)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x = self.head(x)
        x = torch.reshape(x, (B, -1, H*W)).contiguous()
        x = torch.permute(x, (0, 2, 1)).contiguous()
        return x

# ---------------------------------------------------------------------------------------------------------------------------
class SharedNet(nn.Module):
    def __init__(
        self,
        net_config_obj,
        out_feat_shape: Tuple[int, int]):  # (height, width)
        super().__init__()

        num_levels = net_config_obj.num_levels
        in_channels = net_config_obj.fpn_feat_dim
        stem_channels = net_config_obj.stem_channels
        num_classes = net_config_obj.num_classes
        activation = net_config_obj.activation

        # merge block
        self.merge_blk = MergeFeatures(
                num_levels=num_levels,
                in_channels=in_channels,
                out_channels=in_channels,
                activation=activation,
                out_feat_shape=out_feat_shape)

        # stem block
        self.stem_blk = ResidualStemBlock(
                in_channels=in_channels, 
                stem_channels=stem_channels,
                activation=activation)

        # Class prediction head
        self.pred_cls = TaskSpecificHead(
                            in_channels=in_channels,
                            out_channels=num_classes,
                            activation=activation,
                            init_weight_mu=_CLS_CONV_MEAN_INIT_,
                            init_weight_sigma=_CLS_CONV_STD_INIT_,
                            init_bias=_CLS_CONV_BIAS_INIT_)
        
        # Box regression head
        self.pred_box = TaskSpecificHead(
                            in_channels=stem_channels[-1],
                            out_channels=4,
                            activation=activation,
                            init_weight_mu=_BOX_CONV_MEAN_INIT_,
                            init_weight_sigma=_BOX_CONV_STD_INIT_,
                            init_bias=_BOX_CONV_BIAS_INIT_)

        # Centerness regression head
        self.pred_ctr = TaskSpecificHead(
                            in_channels=stem_channels[-1],
                            out_channels=1,
                            activation=activation,
                            init_weight_mu=_CTR_CONV_MEAN_INIT_,
                            init_weight_sigma=_CTR_CONV_STD_INIT_,
                            init_bias=_CTR_CONV_BIAS_INIT_)
        
        # Objectness class head
        self.pred_obj = TaskSpecificHead(
                            in_channels=stem_channels[-1],
                            out_channels=1,
                            activation=activation,
                            init_weight_mu=_CLS_CONV_MEAN_INIT_,
                            init_weight_sigma=_CLS_CONV_STD_INIT_,
                            init_bias=_CLS_CONV_BIAS_INIT_)
        
        # adjust the box deltas
        self.adjustbox = BoxHeadExp()


    def forward(self, x: Dict[str, torch.Tensor]):

        # merge and compute stem features
        featmap = self.merge_blk(x) 
        featmap = self.stem_blk(featmap)                 

        # task specific predictions
        class_logits = self.pred_cls(featmap)               # Class prediction
        objness_logits = self.pred_obj(featmap)             # Objectness prediction
        centerness_logits = self.pred_ctr(featmap)          # Centerness regression
        boxreg_deltas = self.pred_box(featmap)              # Box regression
        boxreg_deltas = self.adjustbox(boxreg_deltas)       # adjust box deltas
        
        return det_named_tuple(class_logits, boxreg_deltas, centerness_logits, objness_logits)
# ---------------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Shared Network Head
# ---------------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Dict
from modules.neural_net.common import WSConv2dBlock, Conv2d, Activation
from modules.neural_net.constants import (
    _BOX_SCALING_WT_INIT_, _NUM_GROUPS_, det_named_tuple,
    _CLS_CONV_MEAN_INIT_, _CLS_CONV_STD_INIT_, _CLS_CONV_BIAS_INIT_,
    _BOX_CONV_MEAN_INIT_, _BOX_CONV_STD_INIT_, _BOX_CONV_BIAS_INIT_,
    _CTR_CONV_MEAN_INIT_, _CTR_CONV_STD_INIT_, _CTR_CONV_BIAS_INIT_)

# --------------------------------------------------------------------------------------------------------------
class BoxHeadExp(nn.Module):
    def __init__(self):
        super().__init__()
        self.wt = nn.Parameter(torch.tensor(_BOX_SCALING_WT_INIT_))

    def forward(self, x: torch.Tensor):
        return F.leaky_relu(self.wt * x)

# --------------------------------------------------------------------------------------------------------------
class SharedNet(nn.Module):
    def __init__(
        self, 
        net_config_obj):
        super().__init__()

        num_levels = net_config_obj.num_levels
        in_channels = net_config_obj.fpn_feat_dim
        stem_channels = net_config_obj.stem_channels
        num_classes = net_config_obj.num_classes
        activation = net_config_obj.activation

        stem_cls = []
        stem_box = []

        for stem_channel in stem_channels:

            # classification stem 
            conv_layer_cls = WSConv2dBlock(_NUM_GROUPS_, in_channels, stem_channel, activation)
            if activation != None:
                act_layer_cls = Activation(activation)
                stem_cls += [ conv_layer_cls, act_layer_cls ]
            else: stem_cls += [ conv_layer_cls ]

            # regression stem
            conv_layer_box = WSConv2dBlock(_NUM_GROUPS_, in_channels, stem_channel, activation)
            if activation != None:
                act_layer_box = Activation(activation)
                stem_box += [ conv_layer_box, act_layer_box ]
            else: stem_box += [ conv_layer_box ]

            in_channels = stem_channel

        self.stem_cls = nn.Sequential(*tuple(stem_cls))
        self.stem_box = nn.Sequential(*tuple(stem_box))

        self.pred_cls = Conv2d(in_channels=in_channels, out_channels=num_classes)   # Class prediction conv
        self.pred_box = Conv2d(in_channels=in_channels, out_channels=4)             # Box regression conv
        self.pred_ctr = Conv2d(in_channels=in_channels, out_channels=1)             # Centerness conv 
        self.pred_obj = Conv2d(in_channels=in_channels, out_channels=1)             # objectness conv
        
        torch.nn.init.normal_(self.pred_cls.conv.weight, mean=_CLS_CONV_MEAN_INIT_, std=_CLS_CONV_STD_INIT_)
        torch.nn.init.constant_(self.pred_cls.conv.bias, _CLS_CONV_BIAS_INIT_)     

        torch.nn.init.normal_(self.pred_box.conv.weight, mean=_BOX_CONV_MEAN_INIT_, std=_BOX_CONV_STD_INIT_)
        torch.nn.init.constant_(self.pred_box.conv.bias, _BOX_CONV_BIAS_INIT_)

        torch.nn.init.normal_(self.pred_ctr.conv.weight, mean=_CTR_CONV_MEAN_INIT_, std=_CTR_CONV_STD_INIT_)
        torch.nn.init.constant_(self.pred_ctr.conv.bias, _CTR_CONV_BIAS_INIT_)

        torch.nn.init.normal_(self.pred_obj.conv.weight, mean=_CLS_CONV_MEAN_INIT_, std=_CLS_CONV_STD_INIT_)
        torch.nn.init.constant_(self.pred_obj.conv.bias, _CLS_CONV_BIAS_INIT_)

        # layer to automatically adjust the base of the exponential function for individial feature level
        adjustbox = {}
        for i in range(num_levels):
            adjustbox[f"c{i}"] = BoxHeadExp()
        self.adjustbox = nn.ModuleDict(adjustbox)


    def forward(self, x: Dict[str, torch.Tensor]):

        class_logits = {}
        boxreg_deltas = {}
        centerness_logits = {}
        objness_logits = {}

        for level, feat in x.items():
            B, _, H, W = feat.shape
            stem_cls = self.stem_cls(feat)
            stem_box = self.stem_box(feat)

            pred_cls = self.pred_cls(stem_cls)
            pred_cls = torch.reshape(pred_cls, (B, -1, H*W)).contiguous()
            class_logits[level] = torch.permute(pred_cls, (0, 2, 1)).contiguous()

            pred_box = self.pred_box(stem_box)
            pred_box = torch.reshape(pred_box, (B, -1, H*W)).contiguous()
            pred_box = torch.permute(pred_box, (0, 2, 1)).contiguous()
            boxreg_deltas[level] = self.adjustbox[level](pred_box)

            pred_ctr = self.pred_ctr(stem_box)
            pred_ctr = torch.reshape(pred_ctr, (B, -1, H*W)).contiguous()
            centerness_logits[level] = torch.permute(pred_ctr, (0, 2, 1)).contiguous()

            pred_obj = self.pred_obj(stem_box)
            pred_obj = torch.reshape(pred_obj, (B, -1, H*W)).contiguous()
            objness_logits[level] = torch.permute(pred_obj, (0, 2, 1)).contiguous()

        return det_named_tuple(class_logits, boxreg_deltas, centerness_logits, objness_logits)
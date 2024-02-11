# ---------------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Feature Pyramid Network (FPN) feature aggregator corrosponding to 'n' levels of the feature pyramid
# --------------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from modules.neural_net.constants import _NUM_GROUPS_
from modules.neural_net.common import WSConv2dBlock, Conv2d_v2
from modules.neural_net.constants import _INTERP_MODE_
from modules.first_stage.get_parameters import net_config

# ---------------------------------------------------------------------------------------------------------------------------
class FPN_block(nn.Module):
    def __init__(
        self,
        num_backbone_nodes: int,
        num_channels: int, 
        num_groups: int,
        activation: str):
        super().__init__()

        # top-down nodes
        top_down_nodes = {}
        for i in range(num_backbone_nodes):
            top_down_nodes[f"c{i}"] = WSConv2dBlock(num_groups, num_channels, num_channels, activation)
        self.top_down_nodes = nn.ModuleDict(top_down_nodes)

        # other info
        self.num_top_down_nodes = num_backbone_nodes


    def forward(self, x: Dict[str, torch.Tensor]):
        # top-down computation
        x_top_down = {}
        key = f"c{self.num_top_down_nodes - 1}"
        x_top_down[key] = x[key]
        prev_node = x[key]

        num_nodes = self.num_top_down_nodes - 1
        for i in range(num_nodes):
            key = f"c{num_nodes - i - 1}"
            x_interpolate = F.interpolate( 
                input = prev_node, 
                size = ( x[key].shape[2], x[key].shape[3] ), 
                mode = _INTERP_MODE_)
            x_top_down[key] = x[key] + x_interpolate
            prev_node = x_top_down[key]

        # return a dictionary of node outputs
        x_out = {}
        for i in range(self.num_top_down_nodes):
            key = f"c{i}"
            x_out[key] = self.top_down_nodes[key](x_top_down[key])
        return x_out
    
# ---------------------------------------------------------------------------------------------------------------------------
class FPN(nn.Module):
    def __init__(
        self, 
        net_config_obj: net_config):
        super().__init__()

        num_blks = net_config_obj.num_fpn_blocks
        num_channels = net_config_obj.fpn_feat_dim
        activation = net_config_obj.activation
        feat_pyr_channels = net_config_obj.feat_pyr_channels

        # conv layer to reduce the backbone feat dim
        reduced_feat_dim = {}
        num_backbone_nodes = len(list(feat_pyr_channels.keys()))
        for i in range(num_backbone_nodes):
            key = f"c{i}"
            reduced_feat_dim[key] = Conv2d_v2(in_channels = feat_pyr_channels[key], out_channels = num_channels, kernel_size = 1)
        self.reduced_feat_dim = nn.ModuleDict(reduced_feat_dim)

        # init number of BiFPN blocks
        FPN_layers = []
        for _ in range(num_blks):
            FPN_layer =  FPN_block(
                num_backbone_nodes,
                num_channels, 
                _NUM_GROUPS_,
                activation)            
            FPN_layers += [ FPN_layer ]
        self.FPN_layers = nn.Sequential(*tuple(FPN_layers))


    def forward(self, x: Dict[str, torch.Tensor]):
        x_dim_reduction = {}
        for key, val in x.items():
            x_dim_reduction[key] = self.reduced_feat_dim[key](val)
        x_out = self.FPN_layers(x_dim_reduction)
        return x_out
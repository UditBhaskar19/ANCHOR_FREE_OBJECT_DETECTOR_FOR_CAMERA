# ---------------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Bidirectional Feature Pyramid Network (BiFPN) feature aggregator 
#               corrosponding to 'n' levels of the feature pyramid
# ---------------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from modules.neural_net.constants import _NUM_GROUPS_
from modules.neural_net.common import WSConv2dBlock, Conv2d_v2
from modules.neural_net.constants import _EPS_, _INTERP_MODE_
from modules.first_stage.get_parameters import net_config

# ---------------------------------------------------------------------------------------------------------------------------
class BiFPN_block(nn.Module):
    def __init__(
        self, 
        num_backbone_nodes: int, 
        num_channels: int, 
        num_groups: int,
        activation: str):
        super().__init__()

        # top-down & bottom up weights
        num_top_down_nodes = num_backbone_nodes - 2
        top_down_weights = torch.rand(num_top_down_nodes, 2)
        bottom_up_weights = torch.rand(num_backbone_nodes, 3)
        bottom_up_weights[[0, -1], -1] = 0.0
        self.top_down_weights = nn.Parameter(top_down_weights)
        self.bottom_up_weights = nn.Parameter(bottom_up_weights)

        # for top down paths
        top_down_nodes = {}
        for i in range(num_top_down_nodes):
            top_down_nodes[f"c{num_top_down_nodes - i}"] = WSConv2dBlock(num_groups, num_channels, num_channels, activation)
        self.top_down_nodes = nn.ModuleDict(top_down_nodes)

        # for bottom up paths
        bottom_up_nodes = {}
        for i in range(num_backbone_nodes):
            bottom_up_nodes[f"c{i}"] = WSConv2dBlock(num_groups, num_channels, num_channels, activation)
        self.bottom_up_nodes = nn.ModuleDict(bottom_up_nodes) 

        # other info 
        # self.feat_pyr_shapes = feat_pyr_shapes
        self.num_top_down_nodes = num_top_down_nodes
        self.num_bottom_up_nodes = num_backbone_nodes


    def forward(self, x: Dict[str, torch.Tensor]):
        # top-down weight normalization
        top_down_weights_unnorm = F.relu(self.top_down_weights, inplace=False)
        top_down_weights = top_down_weights_unnorm / (torch.sum(top_down_weights_unnorm, dim=-1, keepdims=True) + _EPS_)

        # bottom-up weight normalization
        bottom_up_weights_unnorm = F.relu(self.bottom_up_weights, inplace=False)
        bottom_up_weights = bottom_up_weights_unnorm / (torch.sum(bottom_up_weights_unnorm, dim=-1, keepdims=True) + _EPS_)

        # top-down computation
        x_top_down = {}
        prev_node = x[f"c{self.num_bottom_up_nodes - 1}"]
        for i in range(self.num_top_down_nodes):
            key = f"c{self.num_top_down_nodes - i}"
            x_interpolate = F.interpolate(
                input = prev_node,
                size = ( x[key].shape[2], x[key].shape[3] ), 
                mode = _INTERP_MODE_) 
            x_top_down[key] = self.top_down_nodes[key](top_down_weights[i, 0] * x[key] + top_down_weights[i, 1] * x_interpolate)
            prev_node = x_top_down[key]

        # bottom-up computation
        x_bottom_up = {}
        x_interpolate = F.interpolate(
                input = x_top_down['c1'], 
                size = ( x['c0'].shape[2], x['c0'].shape[3] ), 
                mode = _INTERP_MODE_) 
        x_bottom_up['c0'] = self.bottom_up_nodes['c0']( bottom_up_weights[0, 0] * x['c0'] + bottom_up_weights[0, 1] * x_interpolate )

        prev_node = x_bottom_up['c0']
        for i in range(self.num_top_down_nodes):
            key = f"c{i+1}"
            x_interpolate = F.interpolate(
                input = prev_node, 
                size = ( x[key].shape[2], x[key].shape[3] ),
                mode = _INTERP_MODE_)
            x_aggregate = bottom_up_weights[i+1, 0] * x[key] + bottom_up_weights[i+1, 1] * x_top_down[key] + bottom_up_weights[i+1, 2] * x_interpolate
            x_bottom_up[key] = self.bottom_up_nodes[key](x_aggregate)
            prev_node = x_bottom_up[key]

        key = f"c{self.num_bottom_up_nodes - 1}"
        x_interpolate = F.interpolate(
            input = prev_node, 
            size = ( x[key].shape[2], x[key].shape[3] ),
            mode = _INTERP_MODE_)
        x_aggregate = bottom_up_weights[-1, 0] * x[key] + bottom_up_weights[-1, 1] * x_interpolate
        x_bottom_up[key] = self.bottom_up_nodes[key](x_aggregate)
        
        return x_bottom_up

# ---------------------------------------------------------------------------------------------------------------------------
class BiFPN(nn.Module):
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
            reduced_feat_dim[key] = Conv2d_v2(in_channels = feat_pyr_channels[key], out_channels=num_channels, kernel_size=1)
        self.reduced_feat_dim = nn.ModuleDict(reduced_feat_dim)

        # init number of BiFPN blocks
        BiFPN_layers = []
        for _ in range(num_blks):
            BiFPN_layer =  BiFPN_block(
                num_backbone_nodes, 
                num_channels, 
                _NUM_GROUPS_,
                activation)            
            BiFPN_layers += [ BiFPN_layer ]
        self.BiFPN_layers = nn.Sequential(*tuple(BiFPN_layers))


    def forward(self, x: Dict[str, torch.Tensor]):
        x_dim_reduction = {}
        for key, val in x.items():
            x_dim_reduction[key] = self.reduced_feat_dim[key](val)
        x_out = self.BiFPN_layers(x_dim_reduction)
        return x_out
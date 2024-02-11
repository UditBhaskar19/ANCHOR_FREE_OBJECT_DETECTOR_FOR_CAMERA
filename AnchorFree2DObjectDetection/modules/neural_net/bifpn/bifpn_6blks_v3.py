# ---------------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Bidirectional Feature Pyramid Network (BiFPN) feature aggregator corrosponding to 6 levels of the feature pyramid
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
        num_channels: int, 
        num_groups: int,
        activation: str):
        super().__init__()

        # top-down nodes
        self.c1_top_down_nodes = WSConv2dBlock(num_groups, num_channels, num_channels, activation)
        self.c2_top_down_nodes = WSConv2dBlock(num_groups, num_channels, num_channels, activation)
        self.c3_top_down_nodes = WSConv2dBlock(num_groups, num_channels, num_channels, activation)
        self.c4_top_down_nodes = WSConv2dBlock(num_groups, num_channels, num_channels, activation)

        # bottom-up nodes
        self.c0_bottom_up_nodes = WSConv2dBlock(num_groups, num_channels, num_channels, activation)
        self.c1_bottom_up_nodes = WSConv2dBlock(num_groups, num_channels, num_channels, activation)
        self.c2_bottom_up_nodes = WSConv2dBlock(num_groups, num_channels, num_channels, activation)
        self.c3_bottom_up_nodes = WSConv2dBlock(num_groups, num_channels, num_channels, activation)
        self.c4_bottom_up_nodes = WSConv2dBlock(num_groups, num_channels, num_channels, activation)
        self.c5_bottom_up_nodes = WSConv2dBlock(num_groups, num_channels, num_channels, activation)

        # top-down & bottom up weights
        self.top_down_weights = nn.Parameter(torch.rand(4, 2))
        wt_init_val = torch.rand(6, 3)
        wt_init_val[[0, -1], -1] = 0.0
        self.bottom_up_weights = nn.Parameter(wt_init_val)


    def forward(self, x: Dict[str, torch.Tensor]):
        # top-down weight normalization
        top_down_weights_unnorm = F.relu(self.top_down_weights, inplace=False)
        top_down_weights = top_down_weights_unnorm / (torch.sum(top_down_weights_unnorm, dim=-1, keepdims=True) + _EPS_)

        # bottom-up weight normalization
        bottom_up_weights_unnorm = F.relu(self.bottom_up_weights, inplace=False)
        bottom_up_weights = bottom_up_weights_unnorm / (torch.sum(bottom_up_weights_unnorm, dim=-1, keepdims=True) + _EPS_)

        # top-down path
        c4_top_down = F.interpolate(x['c5'], size=(x['c4'].shape[2], x['c4'].shape[3]), mode=_INTERP_MODE_) 
        c4_top_down = top_down_weights[0,0] * x['c4'] + top_down_weights[0,1] * c4_top_down

        c3_top_down = F.interpolate(c4_top_down, size=(x['c3'].shape[2], x['c3'].shape[3]), mode=_INTERP_MODE_) 
        c3_top_down = top_down_weights[1,0] * x['c3'] + top_down_weights[1,1] * c3_top_down

        c2_top_down = F.interpolate(c3_top_down, size=(x['c2'].shape[2], x['c2'].shape[3]), mode=_INTERP_MODE_) 
        c2_top_down = top_down_weights[2,0] * x['c2'] + top_down_weights[2,1] * c2_top_down

        c1_top_down = F.interpolate(c2_top_down, size=(x['c1'].shape[2], x['c1'].shape[3]), mode=_INTERP_MODE_) 
        c1_top_down = top_down_weights[3,0] * x['c1'] + top_down_weights[3,1] * c1_top_down

        c1_top_down = self.c1_top_down_nodes(c1_top_down)
        c2_top_down = self.c2_top_down_nodes(c2_top_down)
        c3_top_down = self.c3_top_down_nodes(c3_top_down)
        c4_top_down = self.c4_top_down_nodes(c4_top_down)

        # bottom-up path
        c0_bottom_up = F.interpolate(c1_top_down, size=(x['c0'].shape[2], x['c0'].shape[3]), mode=_INTERP_MODE_)
        c0_bottom_up = bottom_up_weights[5,0] * x['c0'] + bottom_up_weights[5,1] * c0_bottom_up

        c1_bottom_up = F.interpolate(c0_bottom_up, size=(x['c1'].shape[2], x['c1'].shape[3]), mode=_INTERP_MODE_)
        c1_bottom_up = bottom_up_weights[4,0] * x['c1'] + bottom_up_weights[4,1] * c1_bottom_up + bottom_up_weights[4,2] * c1_top_down

        c2_bottom_up = F.interpolate(c1_bottom_up, size=(x['c2'].shape[2], x['c2'].shape[3]), mode=_INTERP_MODE_)
        c2_bottom_up = bottom_up_weights[3,0] * x['c2'] + bottom_up_weights[3,1] * c2_bottom_up + bottom_up_weights[3,2] * c2_top_down

        c3_bottom_up = F.interpolate(c2_bottom_up, size=(x['c3'].shape[2], x['c3'].shape[3]), mode=_INTERP_MODE_)
        c3_bottom_up = bottom_up_weights[2,0] * x['c3'] + bottom_up_weights[2,1] * c3_bottom_up + bottom_up_weights[2,2] * c3_top_down

        c4_bottom_up = F.interpolate(c3_bottom_up, size=(x['c4'].shape[2], x['c4'].shape[3]), mode=_INTERP_MODE_)
        c4_bottom_up = bottom_up_weights[1,0] * x['c4'] + bottom_up_weights[1,1] * c4_bottom_up + bottom_up_weights[1,2] * c4_top_down

        c5_bottom_up = F.interpolate(c4_bottom_up, size=(x['c5'].shape[2], x['c5'].shape[3]), mode=_INTERP_MODE_)
        c5_bottom_up = bottom_up_weights[0,0] * x['c5'] + bottom_up_weights[0,1] * c5_bottom_up

        # return a dictionary of node outputs
        x_out = {}
        x_out['c0'] = self.c0_bottom_up_nodes(c0_bottom_up)
        x_out['c1'] = self.c1_bottom_up_nodes(c1_bottom_up)
        x_out['c2'] = self.c2_bottom_up_nodes(c2_bottom_up)
        x_out['c3'] = self.c3_bottom_up_nodes(c3_bottom_up)
        x_out['c4'] = self.c4_bottom_up_nodes(c4_bottom_up)
        x_out['c5'] = self.c5_bottom_up_nodes(c5_bottom_up)
        return x_out
    
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

        # reduce the feature dim from backbone nodes 
        size = 1
        self.c0_reduced_dim = Conv2d_v2(in_channels = feat_pyr_channels['c0'], out_channels = num_channels, kernel_size = size)
        self.c1_reduced_dim = Conv2d_v2(in_channels = feat_pyr_channels['c1'], out_channels = num_channels, kernel_size = size)
        self.c2_reduced_dim = Conv2d_v2(in_channels = feat_pyr_channels['c2'], out_channels = num_channels, kernel_size = size)
        self.c3_reduced_dim = Conv2d_v2(in_channels = feat_pyr_channels['c3'], out_channels = num_channels, kernel_size = size)
        self.c4_reduced_dim = Conv2d_v2(in_channels = feat_pyr_channels['c4'], out_channels = num_channels, kernel_size = size)
        self.c5_reduced_dim = Conv2d_v2(in_channels = feat_pyr_channels['c5'], out_channels = num_channels, kernel_size = size)

        # init number of BiFPN blocks
        BiFPN_layers = []
        for _ in range(num_blks):
            BiFPN_layer =  BiFPN_block(
                num_channels, 
                _NUM_GROUPS_,
                activation)           
            BiFPN_layers += [ BiFPN_layer ]
        self.BiFPN_layers = nn.Sequential(*tuple(BiFPN_layers))

    def forward(self, x: Dict[str, torch.Tensor]):
        x_in = {}
        x_in['c0'] = self.c0_reduced_dim(x['c0'])
        x_in['c1'] = self.c1_reduced_dim(x['c1'])
        x_in['c2'] = self.c2_reduced_dim(x['c2'])
        x_in['c3'] = self.c3_reduced_dim(x['c3'])
        x_in['c4'] = self.c4_reduced_dim(x['c4'])
        x_in['c5'] = self.c4_reduced_dim(x['c5'])
        x_out = self.BiFPN_layers(x_in)
        return x_out
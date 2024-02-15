# ---------------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Feature Pyramid Network (FPN) feature aggregator corrosponding to 6 levels of the feature pyramid
# ---------------------------------------------------------------------------------------------------------------------------
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
        num_channels: int, 
        num_groups: int,
        activation: str):
        super().__init__()

        # top-down nodes
        self.c0_top_down_nodes = WSConv2dBlock(num_groups, num_channels, num_channels, activation)
        self.c1_top_down_nodes = WSConv2dBlock(num_groups, num_channels, num_channels, activation)
        self.c2_top_down_nodes = WSConv2dBlock(num_groups, num_channels, num_channels, activation)
        self.c3_top_down_nodes = WSConv2dBlock(num_groups, num_channels, num_channels, activation)
        self.c4_top_down_nodes = WSConv2dBlock(num_groups, num_channels, num_channels, activation)
        self.c5_top_down_nodes = WSConv2dBlock(num_groups, num_channels, num_channels, activation)


    def forward(self, x: Dict[str, torch.Tensor]):

        # top-down path
        c5_top_down = self.c5_top_down_nodes(x['c5']) 

        c4_top_down = F.interpolate(c5_top_down, size=(x['c4'].shape[2], x['c4'].shape[3]), mode=_INTERP_MODE_)
        c4_top_down = self.c4_top_down_nodes( c4_top_down + x['c4'] )

        c3_top_down = F.interpolate(c4_top_down, size=(x['c3'].shape[2], x['c3'].shape[3]), mode=_INTERP_MODE_)
        c3_top_down = self.c3_top_down_nodes( c3_top_down + x['c3'] )

        c2_top_down = F.interpolate(c3_top_down, size=(x['c2'].shape[2], x['c2'].shape[3]), mode=_INTERP_MODE_)
        c2_top_down = self.c2_top_down_nodes( c2_top_down + x['c2'] )

        c1_top_down = F.interpolate(c2_top_down, size=(x['c1'].shape[2], x['c1'].shape[3]), mode=_INTERP_MODE_)
        c1_top_down = self.c1_top_down_nodes( c1_top_down + x['c1'] )

        c0_top_down = F.interpolate(c1_top_down, size=(x['c0'].shape[2], x['c0'].shape[3]), mode=_INTERP_MODE_)
        c0_top_down = self.c0_top_down_nodes( c0_top_down + x['c0'] )

        # return a dictionary of node outputs
        x_out = {}
        x_out['c0'] = c0_top_down
        x_out['c1'] = c1_top_down
        x_out['c2'] = c2_top_down
        x_out['c3'] = c3_top_down
        x_out['c4'] = c4_top_down
        x_out['c5'] = c5_top_down
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

        # reduce the feature dim from backbone nodes 
        size = 1
        self.c0_reduced_dim = Conv2d_v2(in_channels = feat_pyr_channels['c0'], out_channels = num_channels, kernel_size = size)
        self.c1_reduced_dim = Conv2d_v2(in_channels = feat_pyr_channels['c1'], out_channels = num_channels, kernel_size = size)
        self.c2_reduced_dim = Conv2d_v2(in_channels = feat_pyr_channels['c2'], out_channels = num_channels, kernel_size = size)
        self.c3_reduced_dim = Conv2d_v2(in_channels = feat_pyr_channels['c3'], out_channels = num_channels, kernel_size = size)
        self.c4_reduced_dim = Conv2d_v2(in_channels = feat_pyr_channels['c4'], out_channels = num_channels, kernel_size = size)
        self.c5_reduced_dim = Conv2d_v2(in_channels = feat_pyr_channels['c5'], out_channels = num_channels, kernel_size = size)

        # FPN blocks
        FPN_layers = []
        for _ in range(num_blks):
            FPN_layer =  FPN_block(
                num_channels, 
                _NUM_GROUPS_,
                activation)            
            FPN_layers += [ FPN_layer ]
        self.FPN_layers = nn.Sequential(*tuple(FPN_layers))

    def forward(self, x: Dict[str, torch.Tensor]):
        x_in = {}
        x_in['c0'] = self.c0_reduced_dim(x['c0'])
        x_in['c1'] = self.c1_reduced_dim(x['c1'])
        x_in['c2'] = self.c2_reduced_dim(x['c2'])
        x_in['c3'] = self.c3_reduced_dim(x['c3'])
        x_in['c4'] = self.c4_reduced_dim(x['c4'])
        x_in['c5'] = self.c4_reduced_dim(x['c5'])
        x_out = self.FPN_layers(x_in)
        return x_out
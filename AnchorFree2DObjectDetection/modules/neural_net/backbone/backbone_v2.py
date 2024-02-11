# --------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description: Neural Net backbone layers. Here pytorch pretrained base layers are extracted and if necessary more
#              blocks are added to create the backbone
# --------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
from modules.pretrained.utils_backbone_cfg import extract_backbone_layers, freeze_all_layers, freeze_bn_layers
from modules.neural_net.common import WSConv2d, GroupNorm, Activation
from modules.first_stage.get_parameters import net_config
from modules.neural_net.constants import _NUM_GROUPS_

# --------------------------------------------------------------------------------------------------------------
class block(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        num_groups: int,
        activation: str):
        super().__init__()

        conv_layer = WSConv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        gn_layer = GroupNorm(num_groups, out_channels)
        activ_layer = Activation(activation)
        layers = ( conv_layer, gn_layer, activ_layer )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.block(x)
    
# --------------------------------------------------------------------------------------------------------------
class extra_blocks_backbone(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        num_extra_blocks: int,
        num_backbone_nodes: int,
        num_groups: int,
        activation: str):
        super().__init__()

        extra_layers = {}
        for i in range(num_extra_blocks):
            extra_layers[f'c{num_backbone_nodes + i}'] = block(in_channels, out_channels, num_groups, activation)
            in_channels = out_channels
        self.features = nn.ModuleDict(extra_layers)

    def forward(self, x: torch.Tensor):
        outputs = {}
        for block_name, block in self.features.items():
            x = block(x)
            outputs[block_name] = x
        return outputs

# --------------------------------------------------------------------------------------------------------------
class net_backbone(nn.Module):
    def __init__(self, net_config_obj: net_config):
        super().__init__()

        basenet = net_config_obj.basenet
        num_extra_blocks = net_config_obj.num_extra_blocks
        num_backbone_nodes = net_config_obj.num_backbone_nodes
        in_channels_extra_blks = net_config_obj.in_channels_extra_blks
        out_channels_extra_blks = net_config_obj.out_channels_extra_blks
        freeze_backbone_layers = net_config_obj.freeze_backbone_layers
        activation = net_config_obj.activation

        backbone = extract_backbone_layers(basenet, num_backbone_nodes)
        if freeze_backbone_layers: 
            backbone = freeze_all_layers(backbone)
            # backbone = freeze_bn_layers(backbone)
        self.num_backbone_nodes = num_backbone_nodes
        self.num_extra_blocks = num_extra_blocks
        self.backbone = backbone
        if num_extra_blocks > 0:
            self.extra_blocks = extra_blocks_backbone(
                in_channels_extra_blks, 
                out_channels_extra_blks, 
                num_extra_blocks, 
                num_backbone_nodes,
                _NUM_GROUPS_,
                activation)

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        if self.num_extra_blocks > 0:
            key = f'c{self.num_backbone_nodes - 1}'
            x1 = self.extra_blocks(x[key])
            x.update(x1)
        return x
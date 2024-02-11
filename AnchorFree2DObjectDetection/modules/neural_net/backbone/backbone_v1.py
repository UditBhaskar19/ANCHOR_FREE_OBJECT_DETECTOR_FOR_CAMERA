# --------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description: Neural Net backbone layers. Here pytorch pretrained base layers are extracted and if necessary more
#              blocks are added to create the backbone
# --------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
from modules.pretrained.utils_backbone_cfg import extract_backbone_layers, freeze_all_layers
from modules.neural_net.common import Conv2d, BatchNorm, Activation
from modules.neural_net.constants import _BATCHNORM_MOMENTUM_

# --------------------------------------------------------------------------------------------------------------
class block(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        batch_norm: bool,
        activation: str):
        super().__init__()

        layers = []
        conv_layer = Conv2d(in_channels, out_channels, stride=2)
        layers.append(conv_layer)
        if batch_norm:
            bn_layer = BatchNorm(num_features=out_channels, momentum=_BATCHNORM_MOMENTUM_)
            layers.append(bn_layer)
        layers.append(Activation(activation))
        self.block = nn.Sequential(*tuple(layers))

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
        batch_norm: bool,
        activation: str):
        super().__init__()

        extra_layers = {}
        for i in range(num_extra_blocks):
            extra_layers[f'c{num_backbone_nodes + i}'] = block(in_channels, out_channels, batch_norm, activation)
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
    def __init__(self, net_conf):
        super().__init__()

        basenet = net_conf.basenet
        num_extra_blocks = net_conf.num_extra_blocks
        num_backbone_nodes = net_conf.num_backbone_nodes
        in_channels_extra_blks = net_conf.in_channels_extra_blks
        out_channels_extra_blks = net_conf.out_channels_extra_blks
        do_batch_norm_for_extra_blks = net_conf.do_batch_norm_for_extra_blks
        freeze_backbone_layers = net_conf.freeze_backbone_layers
        activation = net_conf.activation

        backbone = extract_backbone_layers(basenet, num_backbone_nodes)
        if freeze_backbone_layers: backbone = freeze_all_layers(backbone)
        self.num_backbone_nodes = num_backbone_nodes
        self.num_extra_blocks = num_extra_blocks
        self.backbone = backbone
        if num_extra_blocks > 0:
            self.extra_blocks = extra_blocks_backbone(
                in_channels_extra_blks, 
                out_channels_extra_blks, 
                num_extra_blocks, 
                num_backbone_nodes,
                do_batch_norm_for_extra_blks,
                activation)

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        if self.num_extra_blocks > 0:
            key = f'c{self.num_backbone_nodes - 1}'
            x1 = self.extra_blocks(x[key])
            x.update(x1)
        return x
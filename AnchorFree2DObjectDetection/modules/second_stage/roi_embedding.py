# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : < Work in progress >        
# ---------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
from modules.neural_net.common import WSConv2dBlock, Activation
from modules.neural_net.constants import _NUM_GROUPS_, _EPS_
from modules.second_stage.get_param import net_config_stage2 as net_config
from modules.neural_net.common import Conv2d_v2, WSConv2dBlock

# --------------------------------------------------------------------------------------------------------------
class average_pooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        B,C,H,W = x.shape
        x = torch.reshape(x, shape=(B, C, H*W))
        x = torch.mean(input=x, dim=-1, keepdims=True)
        return x

# --------------------------------------------------------------------------------------------------------------
class dimension_reduction(nn.Module):
    def __init__(
        self,
        out_channels: int,
        activation: str,
        feat_pyr_shapes: Dict[str, Tuple[int,int,int]]):
        super().__init__()

        # reduce the backbone feat dim
        reduced_feat_dim = {}
        num_backbone_nodes = len(list(feat_pyr_shapes.keys()))
        for i in range(num_backbone_nodes):
            key = f"c{i}"
            # reduced_feat_dim[key] = Conv2d_v2(
            #     in_channels = feat_pyr_shapes[key][0], 
            #     out_channels = out_channels, 
            #     kernel_size = 3)
            reduced_feat_dim[key] = WSConv2dBlock(
                num_groups = out_channels,
                in_channels = feat_pyr_shapes[key][0], 
                out_channels = out_channels, 
                activation = activation)
        self.reduced_feat_dim = nn.ModuleDict(reduced_feat_dim)

    def forward(self, x: Dict[str, torch.Tensor]):
        x_dim_reduction = {}
        for key, val in x.items():
            x_dim_reduction[key] = self.reduced_feat_dim[key](val)
        return x_dim_reduction

# --------------------------------------------------------------------------------------------------------------
class layer_normalization(nn.Module):
    def __init__(self, eps: float = _EPS_):
        super().__init__()
        self.eps = eps
        self.mu = nn.Parameter(torch.zeros(1))
        self.std = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor):
        mean = torch.mean(x, dim=-1, keepdims=True)
        std = torch.std(x, dim=-1, keepdims=True)
        x  = (x - mean) / (std + self.eps)
        x = self.std * x + self.mu
        return x

# --------------------------------------------------------------------------------------------------------------
class featmap_embedding_block(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        stem_channels: List[int],
        out_channels: int,
        activation: str,
        dropout: float):
        super().__init__()

        layers = []
        for stem_channel in stem_channels:
            layer = WSConv2dBlock(_NUM_GROUPS_, in_channels, stem_channel, activation)
            layers.append(layer)
            in_channels = stem_channel

        avg_pool = average_pooling()
        flatten = nn.Flatten(start_dim=1, end_dim=-1)
        fc_layers = nn.Linear(in_features=in_channels, out_features=out_channels, bias=True)
        act_layer = Activation(activation)
        layer_norm = layer_normalization()
        drop_layer= nn.Dropout(dropout)
        layers += [ avg_pool, flatten, fc_layers, act_layer, layer_norm, drop_layer ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)
    
# --------------------------------------------------------------------------------------------------------------
class featmap_embedding(nn.Module):
    def __init__(
        self,
        netconfig_obj: net_config):
        super().__init__()

        in_channels = netconfig_obj.feat_embedding_inchannels_stage2
        stem_channels = netconfig_obj.feat_embedding_stem_channels_stage2
        out_channels = netconfig_obj.feat_embedding_outchannels_stage2
        activation = netconfig_obj.activation_stage2
        dropout = netconfig_obj.dropout_stage2

        self.embedding_layer = featmap_embedding_block(
            in_channels = in_channels,
            stem_channels = stem_channels,
            out_channels = out_channels,
            activation = activation,
            dropout = dropout)
        
    def forward(self, x: Dict[str, torch.Tensor]):
        embeddings = {}
        for key, roi_maps in x.items():
            embeddings[key] = self.embedding_layer(roi_maps)
        embeddings = torch.stack(list(embeddings.values()), axis=1)
        return embeddings
    
# --------------------------------------------------------------------------------------------------------------
class query_embedding_block(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        activation: str,
        dropout: float):
        super().__init__()

        layers = []
        linear_layer = nn.Linear(in_features=in_channels, out_features=out_channels, bias=True)
        act_layer = Activation(activation)
        layer_norm = layer_normalization()
        drop_layer = nn.Dropout(dropout)
        layers += [ linear_layer, act_layer, layer_norm, drop_layer ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)
        
# --------------------------------------------------------------------------------------------------------------
class query_embedding(nn.Module):
    def __init__(
        self,
        netconfig_obj: net_config):
        super().__init__()

        in_channels = netconfig_obj.query_embedding_inchannels_stage2
        stem_channels = netconfig_obj.query_embedding_stem_channels_stage2
        activation = netconfig_obj.activation_stage2
        dropout = netconfig_obj.dropout_stage2

        layers = []
        for stem_channel in stem_channels:
            layer = query_embedding_block(in_channels, stem_channel, activation, dropout)
            layers.append(layer)
            in_channels = stem_channel
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        return self.layers(x)
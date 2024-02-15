# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : < Work in progress >        
# ---------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple
from modules.neural_net.common import Activation
from modules.second_stage.roi_embedding import layer_normalization
from modules.second_stage.get_param import net_config_stage2 as net_config

# --------------------------------------------------------------------------------------------------------------
def compute_attention_prob(queries, keys):
    dimension = keys.shape[-1]
    attention_score = queries @ keys.permute((0,2,1)) / math.sqrt(dimension)
    attention_prob =  F.softmax(input=attention_score, dim=-1)
    return attention_prob

# --------------------------------------------------------------------------------------------------------------
class feedforward_layer(nn.Module):
    def __init__(
        self, 
        in_dim: int,
        dropout: float,
        activation: str):
        super().__init__()

        base_layers = []
        layer0 = nn.Flatten(start_dim=1, end_dim=-1)
        layer1 = nn.Linear(in_features=in_dim, out_features=in_dim, bias=True)
        layer2 = Activation(activation)
        layer3 = layer_normalization()
        layer4 = nn.Dropout(dropout)
        base_layers +=  [ layer0, layer1, layer2, layer3, layer4 ]
        self.base_layers = nn.Sequential(*base_layers)
        self.obj = nn.Linear(in_features=in_dim, out_features=1, bias=True)

    def forward(self, x: torch.Tensor):
        return self.obj(self.base_layers(x))
    
# --------------------------------------------------------------------------------------------------------------
class single_head_attention(nn.Module):
    def __init__(
        self, 
        feat_dim: int, 
        query_dim: int, 
        hidden_dim: int,
        dropout: float):
        super().__init__()

        self.query = nn.Linear(in_features=query_dim, out_features=hidden_dim)
        self.key = nn.Linear(in_features=feat_dim, out_features=hidden_dim)
        self.value = nn.Linear(in_features=feat_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        feat_embedding, query_embedding = x
        queries = self.query(query_embedding) 
        keys = self.key(feat_embedding) 
        values = self.value(feat_embedding)
        attention_prob = compute_attention_prob(queries, keys)
        attention_prob = self.dropout(attention_prob)
        values = attention_prob @ values
        return values

# --------------------------------------------------------------------------------------------------------------
class multi_head_attention(nn.Module):  # sequential implementation
    def __init__(
        self, 
        feat_dim: int, 
        query_dim: int, 
        hidden_dim: int, 
        out_dim: int,
        num_heads: int,
        dropout: float):
        super().__init__()

        _attention_blocks = []
        for i in range(num_heads):
            _attention_blocks.append(
                single_head_attention(
                    feat_dim = feat_dim,
                    query_dim = query_dim,
                    hidden_dim = hidden_dim,
                    dropout = dropout))
        self.attention_blocks = nn.ModuleList(_attention_blocks)
        self.wo = nn.Linear(in_features=hidden_dim*num_heads, out_features=out_dim)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        features = []
        for attention_block in self.attention_blocks:
            features.append(attention_block(x))
        features = torch.concat(features, dim=-1)
        features = self.wo(features)
        return features
    
# --------------------------------------------------------------------------------------------------------------
class attention_network(nn.Module):
    def __init__(
        self, 
        netconfig_obj: net_config):
        super().__init__()

        feat_dim = netconfig_obj.feat_embedding_outchannels_stage2
        query_dim = netconfig_obj.query_embedding_outchannels_stage2
        hidden_dim = netconfig_obj.hidden_dim_attention_stage2
        num_heads = netconfig_obj.num_heads_attention_stage2
        dropout = netconfig_obj.dropout_stage2
        activation = netconfig_obj.activation_stage2

        self.mha = multi_head_attention(
            feat_dim = feat_dim, 
            query_dim = query_dim, 
            hidden_dim = hidden_dim, 
            out_dim = feat_dim,
            num_heads = num_heads,
            dropout = dropout)
        
        self.norm = layer_normalization()  

        self.ffn = feedforward_layer(
            in_dim = feat_dim,
            dropout = dropout,
            activation = activation)   
        
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        return self.ffn(self.norm(self.mha(x)))









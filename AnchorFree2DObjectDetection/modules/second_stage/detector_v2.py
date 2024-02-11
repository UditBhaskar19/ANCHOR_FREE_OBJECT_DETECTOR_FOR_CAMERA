# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : < Work in progress >        
# ---------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Union
from modules.neural_net.common import WSConv2dBlock, Activation
from modules.neural_net.constants import _NUM_GROUPS_, _EPS_
from modules.second_stage.get_param import net_config_stage2 as net_config
from modules.neural_net.common import WSConv2dBlock

from modules.second_stage.second_stage_loss import second_stage_loss
from modules.second_stage.generate_gt import gen_training_gt
from modules.second_stage.get_param import bdd_parameters_stage2 as bdd_parameters
from modules.second_stage.get_param import kitti_parameters_stage2 as kitti_parameters

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
class embedding_blk(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_channels: List[int],
        activation: str):
        super().__init__()

        layers = []
        for stem_channel in stem_channels:
            layer = WSConv2dBlock(
                num_groups = _NUM_GROUPS_,
                in_channels = in_channels, 
                out_channels = stem_channel, 
                activation = activation)
            in_channels = stem_channel
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)

# --------------------------------------------------------------------------------------------------------------
class nonshared_embedding_blocks(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_channels: List[int],
        activation: str,
        feat_pyr_shapes: Dict[str, Tuple[int,int,int]]):
        super().__init__()

        layer_dict = {}
        for key in feat_pyr_shapes.keys():
            layer_dict[key] = embedding_blk(
                in_channels = in_channels,
                stem_channels = stem_channels,
                activation = activation)
        self.layer_dict = nn.ModuleDict(layer_dict)

    def forward(self, x: Dict[str, torch.Tensor]):
        features = {}
        for key, val in x.items():
            features[key] = self.layer_dict[key](val)
        return features

# --------------------------------------------------------------------------------------------------------------
class shared_embedding_block(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        stem_channels: List[int],
        activation: str):
        super().__init__()

        self.layers = embedding_blk(
            in_channels = in_channels,
            stem_channels = stem_channels,
            activation = activation)
        
    def forward(self, x: torch.Tensor):
        return self.layers(x)

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
class feedforward_layer(nn.Module):
    def __init__(
        self, 
        in_dim: int,
        out_dim: int,
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
        self.obj = nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)

    def forward(self, x: torch.Tensor):
        return self.obj(self.base_layers(x))

# --------------------------------------------------------------------------------------------------------------
class second_stage_predictor(nn.Module):
    def __init__(
        self, 
        netconfig_obj: net_config,
        feat_pyr_shapes: Dict[str, Tuple[int,int,int]]):
        super().__init__()

        in_channels = netconfig_obj.feat_embedding_inchannels_stage2
        stem_channels_nonshared = netconfig_obj.feat_embedding_nonshared_stem_channels_stage2
        stem_channels = netconfig_obj.feat_embedding_stem_channels_stage2
        activation = netconfig_obj.activation_stage2
        dropout = netconfig_obj.dropout_stage2
        out_dim = netconfig_obj.output_dimension_stage2

        self.nonshared_feat = nonshared_embedding_blocks(
            in_channels = in_channels,
            stem_channels = stem_channels_nonshared,
            activation = activation,
            feat_pyr_shapes = feat_pyr_shapes)
        
        self.shared_feat = shared_embedding_block(
            in_channels = stem_channels_nonshared[-1],
            stem_channels = stem_channels,
            activation = activation)
        
        self.avg_pool = average_pooling()

        self.ffn = feedforward_layer(
            in_dim = stem_channels[-1],
            out_dim = out_dim,
            dropout = dropout,
            activation = activation)
        
    def forward(self, x: Dict[str, torch.Tensor]):
        x = self.nonshared_feat(x)
        values = list(x.values())
        x_sum = values[0]
        for i in range(1, len(values)):
            x_sum += values[i]
        return self.ffn(self.avg_pool(self.shared_feat(x_sum)))

# ---------------------------------------------------------------------------------------------------------------------
class second_stage_detector_train(nn.Module):
    def __init__(
        self,
        detector: second_stage_predictor,
        loss_obj: second_stage_loss,
        param_obj: Union[bdd_parameters, kitti_parameters],
        device: str):
        super().__init__()

        self.device = device
        self.detector = detector
        self.loss_fn = loss_obj
        self.ignored_classId = param_obj.ignored_classId_stage2
        self.iou_threshold = param_obj.iou_threshold_stage2
        self.deltas_mean = torch.tensor(param_obj.deltas_mean_stage2, dtype=torch.float32, device=device)
        self.deltas_std = torch.tensor(param_obj.deltas_std_stage2, dtype=torch.float32, device=device)

    def reinit_const_parameters(
        self, 
        const_param_obj: Union[bdd_parameters, kitti_parameters]):
        self.ignored_classId = const_param_obj.ignored_classId_stage2
        self.iou_threshold = const_param_obj.iou_threshold_stage2
        self.deltas_mean = torch.tensor(const_param_obj.deltas_mean_stage2, dtype=torch.float32, device=self.device)
        self.deltas_std = torch.tensor(const_param_obj.deltas_std_stage2, dtype=torch.float32, device=self.device)

    def forward(
        self, 
        x: Dict[str, torch.Tensor],
        gtbboxes: List[torch.Tensor],
        gtclslabels: List[torch.Tensor],
        proposals: List[torch.Tensor]):

        predictions = self.detector(x)

        groundtruths = gen_training_gt(
            gt_boxes = gtbboxes, 
            gt_class = gtclslabels,
            pred_boxes = proposals,
            deltas_mean = self.deltas_mean, 
            deltas_std = self.deltas_std, 
            iou_threshold = self.iou_threshold,
            ignored_classId = self.ignored_classId)
        
        loss = self.loss_fn(predictions, groundtruths)

        return {'loss_obj': loss['loss_obj'] }
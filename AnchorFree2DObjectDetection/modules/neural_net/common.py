# --------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : common modules used for model construction
# --------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.neural_net.constants import _EPS_, _LEAKY_RELU_NEG_SLOPE_

# --------------------------------------------------------------------------------------------------------------
class DepthwiseSeparableConv2d(nn.Module):
    """ Depthwise seperable convolution """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        stride: int = 1, 
        padding: int = 1, 
        bias: bool = True):
        super().__init__()
        # channel wise convolution
        self.channelwise =  nn.Conv2d( in_channels=in_channels, out_channels=in_channels,
                                       kernel_size=kernel_size, stride=stride, padding=padding, 
                                       dilation=1, groups=in_channels, bias=bias )
        # depth wise convolution
        self.depthwise = nn.Conv2d( in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=1, stride=1, padding=0, 
                                    dilation=1, groups=1, bias=bias )
    def forward(self, x: torch.Tensor):
        return self.depthwise(self.channelwise(x))
    
# --------------------------------------------------------------------------------------------------------------
class Conv2d(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        stride: int = 1, 
        padding: int = 1, 
        bias: bool = True):
        super().__init__()
        self.conv =  nn.Conv2d( in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding, 
                                dilation=1, groups=1, bias=bias )
    def forward(self, x: torch.Tensor):
        return self.conv(x)
    
# --------------------------------------------------------------------------------------------------------------
class WSConv2d(nn.Conv2d):
    """ Weight standardized convolution kernel """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        stride: int = 1, 
        padding: int = 1, 
        bias: int = True, 
        eps: float = _EPS_):
        super().__init__(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size, 
            stride=stride, padding=padding, 
            dilation=1, 
            groups=1, 
            bias=bias)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        out_channels, in_channels, H, W = self.weight.shape
        weight = self.weight.reshape(out_channels, -1)
        mean = torch.mean(weight, dim=1, keepdim=True)
        std = torch.std(weight, dim=1, keepdim=True)
        weight = (weight - mean) / (std + self.eps)
        weight = weight.reshape(out_channels, in_channels, H, W)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
# --------------------------------------------------------------------------------------------------------------
class GroupNorm(nn.Module):
    def __init__(
        self, 
        num_groups: int, 
        num_channels: int):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=_EPS_, affine=True)

    def forward(self, x: torch.Tensor):
        return self.gn(x)

# --------------------------------------------------------------------------------------------------------------
class BatchNorm(nn.Module):
    def __init__(
        self, 
        num_features: int, 
        momentum: float):
        super().__init__()
        self.bn =  nn.BatchNorm2d(num_features=num_features, eps=_EPS_, momentum=momentum, 
                                  affine=True, track_running_stats=True)
    def forward(self, x: torch.Tensor):
        return self.bn(x)
    
# --------------------------------------------------------------------------------------------------------------
class Conv2dBlock(nn.Module):
    """ A convolutional block which consists of a sequence of the below operation:
         1. convolution / depthwise separable convolution
         2. batch normalization (optional)
         3. non-linear activation
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        batch_norm: bool, 
        bn_momentum: float, 
        conv2dmode: str, 
        activation: str):  
        super().__init__()
        layers = []
        if conv2dmode == 'separable':  layers += [ DepthwiseSeparableConv2d(in_channels, out_channels) ]
        else:  layers += [ Conv2d(in_channels, out_channels) ]
        if batch_norm: layers += [ BatchNorm(out_channels, bn_momentum) ]
        if activation != None: layers += [ Activation(activation) ]
        self.block = nn.Sequential(*tuple(layers))

    def forward(self, x: torch.Tensor):
        return self.block(x)
    
# --------------------------------------------------------------------------------------------------------------
class WSConv2dBlock(nn.Module):
    """ A weight standardized convolutional block which consists of a sequence of the below operation:
         1. weight standardized convolution
         2. group normalization
         3. non-linear activation
    """
    def __init__(
        self, 
        num_groups: int, 
        in_channels: int, 
        out_channels: int, 
        activation: str):
        super().__init__()
        layers = []
        layers += [ WSConv2d(in_channels, out_channels) ]
        layers += [ GroupNorm(num_groups, out_channels) ]
        if activation != None: layers += [ Activation(activation) ]
        self.block = nn.Sequential(*tuple(layers))

    def forward(self, x: torch.Tensor):
        return self.block(x)
    
# --------------------------------------------------------------------------------------------------------------
class ResidualWSConv2dBlock(nn.Module):
    def __init__(
        self,
        num_groups: int, 
        in_channels: int, 
        out_channels: int, 
        activation: str):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.match_dimension = Conv2d(in_channels, out_channels, kernel_size=1)
        self.block = WSConv2dBlock(num_groups, in_channels, out_channels, activation)
    
    def forward(self, x: torch.Tensor):
        if self.in_channels != self.out_channels:
            x = self.match_dimension(x)
        x = x + self.block(x)
        return x

# --------------------------------------------------------------------------------------------------------------
class Conv2d_v2(nn.Module):
    """ A conv layer which is primarily used for converting a feature dimenion to another feature dimension """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int):
        super().__init__()
        if kernel_size == 3 : padding = 1
        elif kernel_size == 1 : padding = 0
        elif kernel_size not in [1, 3]:
            raise Exception("conv2d kernel size needs to be eith 1 or 3 for Conv2d_v2 module in common.py") 
        self.conv =  nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, 
                               dilation=1, groups=1, bias=True )
    def forward(self, x: torch.Tensor):
        return self.conv(x)
    
# --------------------------------------------------------------------------------------------------------------
class Activation(nn.Module):
    """ Activation Layer """
    def __init__(self, activation: str = 'relu'):
        super().__init__()
        if activation == 'relu': layer = nn.ReLU(inplace=False)
        elif activation == 'leakyrelu': layer = nn.LeakyReLU(negative_slope=_LEAKY_RELU_NEG_SLOPE_, inplace=False)
        elif activation == 'swish': layer = torch.nn.SiLU(inplace=False)
        else : layer = nn.ReLU(inplace=False)
        self.activation_layer = layer

    def forward(self, x: torch.Tensor):
        return self.activation_layer(x)
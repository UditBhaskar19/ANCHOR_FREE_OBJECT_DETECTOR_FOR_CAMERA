import torch
import sys
rootdir = '../..'
sys.path.append(rootdir)
from modules.pretrained.utils_backbone_cfg import get_feat_shapes
from modules.neural_net.backbone.backbone_v2 import net_backbone


def main():
    basenet = 'efficientnet_b4'
    num_backbone_nodes = 4
    num_extra_blocks = 0
    out_channels_extra_blks = 512
    img_h = 360
    img_w = 640
    img_d = 3

    dummy_out_shapes = get_feat_shapes(
        basenet, 
        img_h, img_w, img_d, 
        num_backbone_nodes, 
        num_extra_blocks, 
        out_channels_extra_blks)
    
    key = f'c{num_backbone_nodes - 1}'
    in_channels_extra_blks = dummy_out_shapes[key][0]

    net = net_backbone(
        basenet, 
        num_extra_blocks,
        num_backbone_nodes,
        in_channels_extra_blks, 
        out_channels_extra_blks,
        freeze_backbone_layers = True,
        activation = 'swish')

    input_data_shape = (1, img_d, img_h, img_w)   # (num batches, num_channels, height, width)
    dummy_in = torch.randn(input_data_shape)
    dummy_out = net(dummy_in)

    for key, val in dummy_out.items():
        print(f"key: {key}, shape: {val.shape}")



if __name__ == '__main__':
    main()
import sys
sys.path.append('../..')
import config
import torch
from modules.pretrained.utils_backbone_cfg import (
    get_feat_shapes, extract_fpn_featmap_height_and_width, 
    extract_stride, extract_backbone_layers, freeze_all_layers)


basenet = config.basenet
img_h = config.IMG_RESIZED_H
img_w = config.IMG_RESIZED_W
img_d = config.IMG_D
num_backbone_nodes = config.num_backbone_nodes
num_extra_blocks = config.num_extra_blocks
extra_blocks_feat_dim = config.extra_blocks_feat_dim

backbone_feat_shapes = get_feat_shapes(
    basenet, 
    img_h, img_w, img_d, 
    num_backbone_nodes, 
    num_extra_blocks, 
    extra_blocks_feat_dim)

fpn_feat_h, fpn_feat_w = extract_fpn_featmap_height_and_width(backbone_feat_shapes)
fpn_strides = extract_stride(img_h, fpn_feat_h)

print('-' * 100)
for key, val in backbone_feat_shapes.items():
    print(f"block: {key},  shape: {val}")

backbone = extract_backbone_layers(basenet, num_backbone_nodes)
backbone = freeze_all_layers(backbone)

input_data_shape = (1, img_d, img_h, img_w)   # (num batches, num_channels, height, width)
dummy_in = torch.randn(input_data_shape)
dummy_out = backbone(dummy_in)

print('-' * 100)
for key, val in dummy_out.items():
    print(f"block: {key},  shape: {val.shape}")

print('-' * 100)
for key, val in fpn_strides.items():
    print(f"block: {key},  stride: {val}")
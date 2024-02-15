import sys
rootdir = '../..'
sys.path.append(rootdir)
import torch
from modules.pretrained.utils_backbone_cfg import get_feat_shapes
from modules.neural_net.backbone.backbone_v2 import net_backbone
from modules.neural_net.fpn.fpn_nblks_v2 import FPN
from modules.neural_net.detector.detector import Backbone_and_BiFPN

def main():
    basenet = 'efficientnet_b4'
    num_backbone_nodes = 3
    num_extra_blocks = 1
    out_channels_extra_blks = 512
    num_fpn_blocks = 2
    fpn_feat_dim = 256
    img_h = 360
    img_w = 640
    img_d = 3
    activation = 'swish'

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
        activation = activation)
    
    fpn = FPN(
        num_blks = num_fpn_blocks, 
        feat_pyr_shapes = dummy_out_shapes, 
        num_channels = fpn_feat_dim,
        activation = activation)
    
    model = Backbone_and_BiFPN(net, fpn)
    input_data_shape = (1, img_d, img_h, img_w)   # (num batches, num_channels, height, width)
    dummy_in = torch.randn(input_data_shape)
    dummy_out = model(dummy_in)

    for key, val in dummy_out.items():
        print(f"key: {key}, shape: {val.shape}")


    # visualizing the model architecture in tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter("runs/bdd_dataset_bifpn_nblk")
    # traced_model = torch.jit.trace(model, dummy_in, strict=False)
    # writer.add_graph(traced_model, dummy_in)
    # writer.flush()
    # writer.close()

    # # onnx export model
    # torch.onnx.export(model, dummy_in, "feat_extractor_n.onnx", verbose=False)


if __name__ == '__main__':
    main()
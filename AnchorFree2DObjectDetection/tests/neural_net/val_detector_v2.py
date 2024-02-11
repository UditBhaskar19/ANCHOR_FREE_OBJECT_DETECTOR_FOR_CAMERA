import sys, torch
rootdir = '../..'
sys.path.append(rootdir)
from config import OUT_FEAT_SIZE_H, OUT_FEAT_SIZE_W
from modules.pretrained.utils_backbone_cfg import get_feat_shapes
from modules.neural_net.backbone.backbone_v2 import net_backbone
from modules.neural_net.bifpn.bifpn_nblks_v3 import BiFPN
from modules.neural_net.head.shared_head_v4 import SharedNet
from modules.neural_net.detector.detector_v1 import FCOS


def main():

    basenet = 'efficientnet_b4'
    num_backbone_nodes = 4
    num_extra_blocks = 3
    extra_blocks_feat_dim = 512
    fpn_feat_dim = 256
    num_fpn_blocks = 2
    stem_channels = [256, 256, 256, 256]
    num_classes = 4
    activation = 'swish'
    img_h = 360
    img_w = 640
    img_d = 3

    dummy_out_shapes = get_feat_shapes(
        basenet, 
        img_h, img_w, img_d, 
        num_backbone_nodes, 
        num_extra_blocks, 
        extra_blocks_feat_dim)
    
    backbone = net_backbone(
        basenet = basenet, 
        num_extra_blocks = num_extra_blocks,
        num_backbone_nodes = num_backbone_nodes,
        in_channels_extra_blks = dummy_out_shapes[f'c{num_backbone_nodes - 1}'][0], 
        out_channels_extra_blks = extra_blocks_feat_dim,
        freeze_backbone_layers = True,
        activation = activation)

    bifpn = BiFPN(
        num_blks = num_fpn_blocks, 
        feat_pyr_shapes = dummy_out_shapes, 
        num_channels = fpn_feat_dim,
        activation = activation)
    
    shared_head = SharedNet(
        num_levels = num_backbone_nodes + num_extra_blocks,
        in_channels = fpn_feat_dim, 
        stem_channels = stem_channels,
        num_classes = num_classes,
        activation = activation,
        out_feat_shape = (OUT_FEAT_SIZE_H, OUT_FEAT_SIZE_W))
    
    detector = FCOS(backbone, bifpn, shared_head)

    input_data_shape = (1, img_d, img_h, img_w)   # (num batches, num_channels, height, width)
    dummy_in = torch.randn(input_data_shape)
    preditions = detector(dummy_in)

    class_logits = preditions.class_logits
    boxreg_deltas = preditions.boxreg_deltas 
    centerness_logits = preditions.centerness_logits
    objness_logits = preditions.objness_logits

    print('=' * 100)
    print('class_logits shape      : ', class_logits.shape)
    print('boxreg_deltas shape     : ', boxreg_deltas.shape)
    print('centerness_logits shape : ', centerness_logits.shape)
    print('objness_logits shape    : ', objness_logits.shape)

    # from torch.utils.tensorboard import SummaryWriter
    # # type tensorboard --logdir=runs/bdd_dataset_entire_model_v6 in the cmd
    # writer = SummaryWriter("runs/bdd_dataset_entire_model_v6")
    # traced_model = torch.jit.trace(detector, dummy_in, strict=False)
    # writer.add_graph(traced_model, dummy_in)
    # writer.flush()
    # writer.close()

    # # onnx export model
    # torch.onnx.export(detector, dummy_in, "object_detector.onnx", verbose=False)



if __name__ == '__main__':
    main()
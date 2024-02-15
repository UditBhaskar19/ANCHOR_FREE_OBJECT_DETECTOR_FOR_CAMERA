import sys, torch
rootdir = '../..'
sys.path.append(rootdir)
from modules.pretrained.utils_backbone_cfg import get_feat_shapes
from modules.neural_net.backbone.backbone_v2 import net_backbone
from modules.neural_net.fpn.fpn_nblks_v2 import FPN
from modules.neural_net.head.shared_head_v3 import SharedNet
from modules.neural_net.detector.detector import FCOS


def main():

    basenet = 'efficientnet_b4'
    num_backbone_nodes = 3
    num_extra_blocks = 0
    out_channels_extra_blks = 512
    num_fpn_blocks = 1
    fpn_feat_dim = 256
    stem_channels = [256, 256, 256, 256]
    activation = 'relu'
    img_h = 360
    img_w = 640
    img_d = 3
    num_classes = 4

    dummy_out_shapes = get_feat_shapes(
        basenet, 
        img_h, img_w, img_d, 
        num_backbone_nodes, 
        num_extra_blocks, 
        out_channels_extra_blks)
    
    backbone = net_backbone(
        basenet=basenet, 
        num_extra_blocks=num_extra_blocks,
        num_backbone_nodes=num_backbone_nodes,
        in_channels_extra_blks=dummy_out_shapes[f'c{num_backbone_nodes - 1}'][0], 
        out_channels_extra_blks=out_channels_extra_blks,
        freeze_backbone_layers = True,
        activation = activation)
    
    fpn = FPN(
        num_blks = num_fpn_blocks, 
        feat_pyr_shapes = dummy_out_shapes, 
        num_channels = fpn_feat_dim,
        activation = activation) 
    
    shared_head = SharedNet(
        num_levels = num_backbone_nodes + num_extra_blocks,
        in_channels = fpn_feat_dim, 
        stem_channels = stem_channels,
        num_classes = num_classes,
        activation = activation)
    
    detector = FCOS(backbone, fpn, shared_head)

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
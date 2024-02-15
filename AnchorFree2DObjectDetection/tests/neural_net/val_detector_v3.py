import sys, torch
rootdir = '../..'
sys.path.append(rootdir)
from modules.pretrained.utils_backbone_cfg import get_feat_shapes
from modules.neural_net.backbone.backbone_v2 import net_backbone
from modules.neural_net.bifpn.bifpn_nblks_v2 import BiFPN
from modules.neural_net.head.shared_head_v5 import SharedNet
from modules.neural_net.detector.detector_v1 import FCOS
from get_parameters import bdd_parameters, kitti_parameters, net_config   

def main():

    net_config_obj = net_config()
    bdd_param_obj = bdd_parameters()
    kitti_param_obj = kitti_parameters()

    backbone = net_backbone(net_config_obj)
    bifpn = BiFPN(net_config_obj, kitti_param_obj.feat_pyr_shapes)
    shared_head = SharedNet(net_config_obj, kitti_param_obj.out_feat_shape) 
    detector = FCOS(backbone, bifpn, shared_head)
    
    input_data_shape = (
        1, 
        kitti_param_obj.IMG_D, 
        kitti_param_obj.IMG_RESIZED_H, 
        kitti_param_obj.IMG_RESIZED_W) 
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
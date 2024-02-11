import sys
rootdir = '../..'
sys.path.append(rootdir)
import torch
from modules.pretrained.utils_backbone_cfg import get_feat_shapes
from modules.neural_net.backbone.backbone_v2 import net_backbone
from modules.neural_net.bifpn.bifpn_nblks_v2 import BiFPN
from modules.neural_net.detector.detector import Backbone_and_BiFPN
from get_parameters import bdd_parameters, net_config     

def main():
    
    net_config_obj = net_config()
    bdd_param_obj = bdd_parameters()

    backbone = net_backbone(net_config_obj)
    bifpn = BiFPN(net_config_obj, bdd_param_obj.feat_pyr_shapes)
    model = Backbone_and_BiFPN(backbone, bifpn)
    
    input_data_shape = (
        1, 
        bdd_param_obj.IMG_D, 
        bdd_param_obj.IMG_RESIZED_H, 
        bdd_param_obj.IMG_RESIZED_W) 
    
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
    
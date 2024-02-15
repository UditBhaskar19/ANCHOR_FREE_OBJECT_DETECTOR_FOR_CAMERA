# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Backone intermediate feature extraction functions:
#               1) display_node_names_and_feature_map_shapes
#               2) display_feature_map_shapes_for_specific_node_names
#               3) freeze_all_layers
#               4) freeze_bn_layers
#               5) extract_backbone_layers
#               6) get_feat_shapes
#               7) extract_fpn_featmap_height_and_width
#               8) extract_stride
# ---------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

# ----------------------------------------------------------------------------------------------------
# Backbone options
vgg_options = ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
    
resnet_options = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

efficientnet_options = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 
                        'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
                        'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l']

regnet_options = ['regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 
                  'regnet_y_3_2gf', 'regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf',
                  'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf', 
                  'regnet_x_3_2gf', 'regnet_x_8gf', 'regnet_x_16gf', 'regnet_x_32gf']

# ----------------------------------------------------------------------------------------------------
def display_node_names_and_feature_map_shapes(device, model):
    """ display all the intermediate layer names and feature shapes """
    model = model.to(device)
    train_nodes, eval_nodes = get_graph_node_names(model)

    all_node = [ (name, 'N'+str(idx)) for idx, name in enumerate(train_nodes)]
    sel_node = dict(all_node[1:-2])
    backbone = create_feature_extractor(model, sel_node)
    
    input_data_shape = (1, 3, 256, 256)   # (num batches, num_channels, height, width)
    dummy_in = torch.randn(input_data_shape).to(device)
    dummy_out = backbone(dummy_in)

    nodes_and_feature_dim = [ ( key,  list(dummy_out[val].shape[1:]) ) for key, val in sel_node.items() ]
    nodes_and_feature_dim = dict(nodes_and_feature_dim)
    
    gap = 35
    for (key, value) in nodes_and_feature_dim.items():
        print(key , '-'*(gap-len(key)), value)

# ----------------------------------------------------------------------------------------------------
def display_feature_map_shapes_for_specific_node_names(device, model, node_names):
    """ display specific layer names and feature shapes """
    model = model.to(device)
    train_nodes, eval_nodes = get_graph_node_names(model)

    all_node = [ (name, 'N'+str(idx)) for idx, name in enumerate(train_nodes)]
    sel_node = dict(all_node[1:-2])
    backbone = create_feature_extractor(model, sel_node)

    input_data_shape = (1, 3, 256, 256)   # (num batches, num_channels, height, width)
    dummy_in = torch.randn(input_data_shape).to(device)
    dummy_out = backbone(dummy_in)

    nodes_and_feature_dim = []
    for key, val in sel_node.items():
        if key in node_names:
            nodes_and_feature_dim += [ ( key,  list(dummy_out[val].shape[1:]) ) ]

    if len(nodes_and_feature_dim) != len(node_names):
        print('one of the input node names is not compatible network configuration')
        return
    
    nodes_and_feature_dim = dict(nodes_and_feature_dim)

    gap = 35
    for (key, value) in nodes_and_feature_dim.items():
        print(key , '-'*(gap-len(key)), value)

# ----------------------------------------------------------------------------------------------------
def freeze_all_layers(backbone):
    """ Freeze all the layers of the backbone """
    backbone = backbone.requires_grad_(False)
    bn_modules = [ module for module in backbone.modules() if isinstance(module, nn.BatchNorm2d) ]
    for module in bn_modules:
        module.track_running_stats = False
    return backbone

# ----------------------------------------------------------------------------------------------------
def freeze_bn_layers(backbone):
    """ Freeze all the batch norm layers of the backbone """
    bn_modules = [ module for module in backbone.modules() if isinstance(module, nn.BatchNorm2d) ]
    for module in bn_modules:
        module.track_running_stats = False
        for parameter in module.parameters():
            parameter.requires_grad = False
    return backbone

# ----------------------------------------------------------------------------------------------------
def extract_backbone_layers(basenet, required_num_return_nodes):
    """ Create a backbone network from the backbone net type and required number of return nodes.
    From bottom-up the dictionary of returned layers of the backbone has keys : 'c0', 'c1', 'c2', ...
    ( i.e we are numbering it from 0, 1, 2, .. for convenience )
    """
    if basenet in vgg_options:
        from modules.pretrained.vgg_backbone_cfg import get_conv_base
        convnet, return_nodes = get_conv_base(basenet)
    elif basenet in resnet_options: 
        from modules.pretrained.resnet_backbone_cfg import get_conv_base
        convnet, return_nodes = get_conv_base(basenet)
    elif basenet in efficientnet_options: 
        from modules.pretrained.efficientnet_backbone_cfg import get_conv_base
        convnet, return_nodes = get_conv_base(basenet)
    elif basenet in regnet_options: 
        from modules.pretrained.regnet_backbone_cfg import get_conv_base
        convnet, return_nodes = get_conv_base(basenet)

    if required_num_return_nodes == 1: nodes = [return_nodes[3]]
    elif required_num_return_nodes == 2 : nodes = return_nodes[2:]
    elif required_num_return_nodes == 3 : nodes = return_nodes[1:]
    else: nodes = return_nodes
    return_nodes = {k: f"c{i}" for i, k in enumerate(nodes)}

    backbone = create_feature_extractor(convnet, return_nodes)
    return backbone

# ----------------------------------------------------------------------------------------------------
def get_feat_shapes(
    basenet, 
    input_image_shape, 
    required_num_return_nodes, 
    num_extra_blocks_fpn, 
    fpn_hidden_dim):

    """ Return a dictionary of feature shapes of selected backbone layer outputs """
    backbone = extract_backbone_layers(basenet, required_num_return_nodes)
    backbone = freeze_all_layers(backbone)

    img_h, img_w, img_d = input_image_shape
    input_data_shape = (1, img_d, img_h, img_w)   # (num batches, num_channels, height, width)
    dummy_in = torch.randn(input_data_shape)
    dummy_out = backbone(dummy_in)

    # set the layer names and shapes : for e.g c0, c1, c2, c3 or c0, c1, c2
    num_layers = len(dummy_out)
    layer_names = [f'c{i}' for i in range(num_layers)]
    dummy_out_shapes = {layer_names[idx] : tuple(v.shape[1:]) for idx, (k, v) in enumerate(dummy_out.items())}

    # extract the last layer dimensions 
    dummy_out_shapes_list = list(dummy_out_shapes.values())
    h = dummy_out_shapes_list[-1][1]
    w = dummy_out_shapes_list[-1][2]

    # set the layer names and shapes : for e.g. c6, c7 or c6
    for i in range(num_extra_blocks_fpn):
        layer_name = f'c{len(dummy_out_shapes_list) + i}'
        h = int(h / 2 + 0.5)
        w = int(w / 2 + 0.5) 
        shape = (fpn_hidden_dim, h, w)
        dummy_out_shapes[layer_name] = shape

    return dummy_out_shapes

# ----------------------------------------------------------------------------------------------------
def extract_fpn_featmap_height_and_width(backbone_feat_shapes):
    """ Return a dictionary of feature widths and heights of selected backbone layer outputs """
    fpn_feat_h = {}
    fpn_feat_w = {}
    for key, value in backbone_feat_shapes.items():
        fpn_feat_h[key] = value[1]
        fpn_feat_w[key] = value[2]
    return fpn_feat_h, fpn_feat_w

# ----------------------------------------------------------------------------------------------------
def extract_stride(img_h, fpn_feat_h):
    fpn_strides = {}
    for level, feat_h in fpn_feat_h.items():
        fpn_strides[level] = img_h / feat_h
    return fpn_strides
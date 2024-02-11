# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Extract last 4 intermediate pyramid layer outputs of resnet preloaded with imagenet weights
# ---------------------------------------------------------------------------------------------------------------
from torchvision import models

"""  Refer:   0_2_resnet_feat_shapes.ipynb """
# ==============================================================================================
resnet18_weights = 'IMAGENET1K_V1'
resnet18_return_nodes = ('layer1.1.relu_1', 
                         'layer2.1.relu_1', 
                         'layer3.1.relu_1',
                         'layer4.1.relu_1')
# ---------------------------------------------------------------------------------------------
resnet34_weights = 'IMAGENET1K_V1'
resnet34_return_nodes = ('layer1.2.relu_1', 
                         'layer2.3.relu_1', 
                         'layer3.5.relu_1',
                         'layer4.2.relu_1')
# ---------------------------------------------------------------------------------------------
resnet50_weights = 'IMAGENET1K_V1'  # 'IMAGENET1K_V1' or 'IMAGENET1K_V2'
resnet50_return_nodes = ('layer2.0.relu', 
                         'layer3.0.relu', 
                         'layer4.0.relu',
                         'layer4.2.relu_2')
# ---------------------------------------------------------------------------------------------
resnet101_weights = 'IMAGENET1K_V1'  # 'IMAGENET1K_V1' or 'IMAGENET1K_V2'
resnet101_return_nodes = ('layer2.0.relu', 
                          'layer3.0.relu', 
                          'layer4.0.relu',
                          'layer4.2.relu_2')
# ---------------------------------------------------------------------------------------------
resnet152_weights = 'IMAGENET1K_V1'  # 'IMAGENET1K_V1' or 'IMAGENET1K_V2'
resnet152_return_nodes = ('layer2.0.relu', 
                          'layer3.0.relu', 
                          'layer4.0.relu',
                          'layer4.2.relu_2')

# ==============================================================================================

resnet18_backbone_cfg = {'weights': resnet18_weights, 'return_nodes': resnet18_return_nodes}

resnet34_backbone_cfg = {'weights': resnet34_weights, 'return_nodes': resnet34_return_nodes}

resnet50_backbone_cfg = {'weights': resnet50_weights, 'return_nodes': resnet50_return_nodes}

resnet101_backbone_cfg = {'weights': resnet101_weights, 'return_nodes': resnet101_return_nodes}

resnet152_backbone_cfg = {'weights': resnet152_weights, 'return_nodes': resnet152_return_nodes}

# ==============================================================================================

def get_conv_base(basenet):

    if basenet == 'resnet18': 
        model = models.resnet18(weights=resnet18_weights)
        return_nodes = resnet18_return_nodes

    elif basenet == 'resnet34': 
        model = models.resnet34(weights=resnet34_weights)
        return_nodes = resnet34_return_nodes

    elif basenet == 'resnet50': 
        model = models.resnet50(weights=resnet50_weights)
        return_nodes = resnet50_return_nodes

    elif basenet == 'resnet101': 
        model = models.resnet101(weights=resnet101_weights)
        return_nodes = resnet101_return_nodes

    elif basenet == 'resnet152': 
        model = models.resnet152(weights=resnet152_weights)
        return_nodes = resnet152_return_nodes

    return model, return_nodes

# ==============================================================================================
# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Extract last 4 intermediate pyramid layer outputs of vgg preloaded with imagenet weights
# ---------------------------------------------------------------------------------------------------------------
from torchvision import models

"""  Refer:   0_1_vgg_feat_shapes.ipynb """
# ==============================================================================================
vgg11_weights = 'IMAGENET1K_V1'
vgg11_return_nodes = ('features.4', 
                      'features.9', 
                      'features.14',
                      'features.19')
# ---------------------------------------------------------------------------------------------
vgg11_bn_weights = 'IMAGENET1K_V1'
vgg11_bn_return_nodes = ('features.6', 
                         'features.13', 
                         'features.20',
                         'features.27')
# ---------------------------------------------------------------------------------------------
vgg13_weights = 'IMAGENET1K_V1'
vgg13_return_nodes = ('features.8', 
                      'features.13', 
                      'features.18',
                      'features.23')
# ---------------------------------------------------------------------------------------------
vgg13_bn_weights = 'IMAGENET1K_V1'
vgg13_bn_return_nodes = ('features.12', 
                         'features.19', 
                         'features.26',
                         'features.33')
# ---------------------------------------------------------------------------------------------
vgg16_weights = 'IMAGENET1K_V1'
vgg16_return_nodes = ('features.4', 
                      'features.9', 
                      'features.14',
                      'features.19')
# ---------------------------------------------------------------------------------------------
vgg16_bn_weights = 'IMAGENET1K_V1'
vgg16_bn_return_nodes = ('features.12', 
                         'features.22', 
                         'features.32',
                         'features.42')
# ---------------------------------------------------------------------------------------------
vgg19_weights = 'IMAGENET1K_V1'
vgg19_return_nodes = ('features.8', 
                      'features.17', 
                      'features.26',
                      'features.35')
# ---------------------------------------------------------------------------------------------
vgg19_bn_weights = 'IMAGENET1K_V1'
vgg19_bn_return_nodes = ('features.12', 
                         'features.25', 
                         'features.38',
                         'features.51')

# ==============================================================================================

vgg11_backbone_cfg = {'weights': vgg11_weights, 'return_nodes': vgg11_return_nodes}

vgg11_bn_backbone_cfg = {'weights': vgg11_bn_weights, 'return_nodes': vgg11_bn_return_nodes}

vgg13_backbone_cfg = {'weights': vgg13_weights, 'return_nodes': vgg13_return_nodes}

vgg13_bn_backbone_cfg = {'weights': vgg13_bn_weights, 'return_nodes': vgg13_bn_return_nodes}

vgg16_backbone_cfg = {'weights': vgg16_weights, 'return_nodes': vgg16_return_nodes}

vgg16_bn_backbone_cfg = {'weights': vgg16_bn_weights, 'return_nodes': vgg16_bn_return_nodes}

vgg19_backbone_cfg = {'weights': vgg19_weights, 'return_nodes': vgg19_return_nodes}

vgg19_bn_backbone_cfg = {'weights': vgg19_bn_weights, 'return_nodes': vgg19_bn_return_nodes}

# ==============================================================================================

def get_conv_base(basenet):

    if basenet == 'vgg11': 
        model = models.vgg11(weights=vgg11_weights)
        return_nodes = vgg11_return_nodes

    elif basenet == 'vgg11_bn': 
        model = models.vgg11_bn(weights=vgg11_bn_weights)
        return_nodes = vgg11_bn_return_nodes

    elif basenet == 'vgg13': 
        model = models.vgg13(weights=vgg13_weights)
        return_nodes = vgg13_return_nodes

    elif basenet == 'vgg13_bn': 
        model = models.vgg13_bn(weights=vgg13_bn_weights)
        return_nodes = vgg13_bn_return_nodes

    elif basenet == 'vgg16': 
        model = models.vgg16(weights=vgg16_weights)
        return_nodes = vgg16_return_nodes

    elif basenet == 'vgg16_bn': 
        model = models.vgg16_bn(weights=vgg16_bn_weights)
        return_nodes = vgg16_bn_return_nodes

    elif basenet == 'vgg19': 
        model = models.vgg19(weights=vgg19_weights)
        return_nodes = vgg19_return_nodes

    elif basenet == 'vgg19_bn': 
        model = models.vgg19_bn(weights=vgg19_bn_weights)
        return_nodes = vgg19_bn_return_nodes

    return model, return_nodes

# ==============================================================================================


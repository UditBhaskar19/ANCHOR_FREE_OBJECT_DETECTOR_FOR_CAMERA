# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Extract last 4 intermediate pyramid layer outputs of regnet preloaded with imagenet weights
# ---------------------------------------------------------------------------------------------------------------
from torchvision import models

"""  Refer: 0_4_regnet_feat_shapes.ipynb """
# ====================================================================================================================
regnet_y_400mf_weights = 'IMAGENET1K_V1'  # 'IMAGENET1K_V1' or 'IMAGENET1K_V2'
regnet_y_400mf_return_nodes = ('trunk_output.block1.block1-0.activation', 
                               'trunk_output.block2.block2-2.activation', 
                               'trunk_output.block3.block3-5.activation',
                               'trunk_output.block4.block4-5.activation')
# ---------------------------------------------------------------------------------------------
regnet_y_800mf_weights = 'IMAGENET1K_V1'  # 'IMAGENET1K_V1' or 'IMAGENET1K_V2'
regnet_y_800mf_return_nodes = ('trunk_output.block1.block1-0.activation', 
                               'trunk_output.block2.block2-2.activation', 
                               'trunk_output.block3.block3-7.activation',
                               'trunk_output.block4.block4-1.activation')
# ---------------------------------------------------------------------------------------------
regnet_y_1_6gf_weights = 'IMAGENET1K_V1'  # 'IMAGENET1K_V1' or 'IMAGENET1K_V2'
regnet_y_1_6gf_return_nodes = ('trunk_output.block1.block1-1.activation', 
                               'trunk_output.block2.block2-5.activation', 
                               'trunk_output.block3.block3-16.activation',
                               'trunk_output.block4.block4-1.activation')
# ---------------------------------------------------------------------------------------------
regnet_y_3_2gf_weights = 'IMAGENET1K_V1'  # 'IMAGENET1K_V1' or 'IMAGENET1K_V2'
regnet_y_3_2gf_return_nodes = ('trunk_output.block1.block1-1.activation', 
                               'trunk_output.block2.block2-4.activation', 
                               'trunk_output.block3.block3-12.activation',
                               'trunk_output.block4.block4-0.activation')
# ---------------------------------------------------------------------------------------------
regnet_y_8gf_weights = 'IMAGENET1K_V1'  # 'IMAGENET1K_V1' or 'IMAGENET1K_V2'
regnet_y_8gf_return_nodes = ('trunk_output.block1.block1-1.activation', 
                             'trunk_output.block2.block2-3.activation', 
                             'trunk_output.block3.block3-9.activation',
                             'trunk_output.block4.block4-0.activation')
# ---------------------------------------------------------------------------------------------
regnet_y_16gf_weights = 'IMAGENET1K_V1'  # 'IMAGENET1K_V1' or 'IMAGENET1K_V2'
regnet_y_16gf_return_nodes = ('trunk_output.block1.block1-1.activation', 
                              'trunk_output.block2.block2-3.activation', 
                              'trunk_output.block3.block3-10.activation',
                              'trunk_output.block4.block4-0.activation')
# ---------------------------------------------------------------------------------------------
regnet_y_32gf_weights = 'IMAGENET1K_V1'  # 'IMAGENET1K_V1' or 'IMAGENET1K_V2'
regnet_y_32gf_return_nodes = ('trunk_output.block1.block1-1.activation', 
                              'trunk_output.block2.block2-4.activation', 
                              'trunk_output.block3.block3-11.activation',
                              'trunk_output.block4.block4-0.activation')
# ====================================================================================================================

regnet_x_400mf_weights = 'IMAGENET1K_V1'  # 'IMAGENET1K_V1' or 'IMAGENET1K_V2'
regnet_x_400mf_return_nodes = ('trunk_output.block1.block1-0.activation', 
                               'trunk_output.block2.block2-1.activation', 
                               'trunk_output.block3.block3-6.activation',
                               'trunk_output.block4.block4-11.activation')
# ---------------------------------------------------------------------------------------------
regnet_x_800mf_weights = 'IMAGENET1K_V1'  # 'IMAGENET1K_V1' or 'IMAGENET1K_V2'
regnet_x_800mf_return_nodes = ('trunk_output.block1.block1-0.activation', 
                                'trunk_output.block2.block2-2.activation', 
                                'trunk_output.block3.block3-6.activation',
                                'trunk_output.block4.block4-4.activation')
# ---------------------------------------------------------------------------------------------
regnet_x_1_6gf_weights = 'IMAGENET1K_V1'  # 'IMAGENET1K_V1' or 'IMAGENET1K_V2'
regnet_x_1_6gf_return_nodes = ('trunk_output.block1.block1-1.activation', 
                               'trunk_output.block2.block2-1.activation', 
                               'trunk_output.block3.block3-9.activation',
                               'trunk_output.block4.block4-1.activation')
# ---------------------------------------------------------------------------------------------
regnet_x_3_2gf_weights = 'IMAGENET1K_V1'  # 'IMAGENET1K_V1' or 'IMAGENET1K_V2'
regnet_x_3_2gf_return_nodes = ('trunk_output.block1.block1-1.activation', 
                               'trunk_output.block2.block2-5.activation', 
                               'trunk_output.block3.block3-14.activation',
                               'trunk_output.block4.block4-1.activation')
# ---------------------------------------------------------------------------------------------
regnet_x_8gf_weights = 'IMAGENET1K_V1'  # 'IMAGENET1K_V1' or 'IMAGENET1K_V2'
regnet_x_8gf_return_nodes = ('trunk_output.block1.block1-1.activation', 
                             'trunk_output.block2.block2-4.activation', 
                             'trunk_output.block3.block3-14.activation',
                             'trunk_output.block4.block4-0.activation')
# ---------------------------------------------------------------------------------------------
regnet_x_16gf_weights = 'IMAGENET1K_V1'  # 'IMAGENET1K_V1' or 'IMAGENET1K_V2'
regnet_x_16gf_return_nodes = ('trunk_output.block1.block1-1.activation', 
                              'trunk_output.block2.block2-1.activation', 
                              'trunk_output.block3.block3-12.activation',
                              'trunk_output.block4.block4-0.activation')
# ---------------------------------------------------------------------------------------------
regnet_x_32gf_weights = 'IMAGENET1K_V1'  # 'IMAGENET1K_V1' or 'IMAGENET1K_V2'
regnet_x_32gf_return_nodes = ('trunk_output.block1.block1-1.activation', 
                              'trunk_output.block2.block2-6.activation', 
                              'trunk_output.block3.block3-12.activation',
                              'trunk_output.block4.block4-0.activation')
# ====================================================================================================================

regnet_y_400mf_backbone_cfg = {'weights': regnet_y_400mf_weights, 'return_nodes': regnet_y_400mf_return_nodes}

regnet_y_800mf_backbone_cfg = {'weights': regnet_y_800mf_weights, 'return_nodes': regnet_y_800mf_return_nodes}

regnet_y_1_6gf_backbone_cfg = {'weights': regnet_y_1_6gf_weights, 'return_nodes': regnet_y_1_6gf_return_nodes}

regnet_y_3_2gf_backbone_cfg = {'weights': regnet_y_3_2gf_weights, 'return_nodes': regnet_y_3_2gf_return_nodes}

regnet_y_8gf_backbone_cfg = {'weights': regnet_y_8gf_weights, 'return_nodes': regnet_y_8gf_return_nodes}

regnet_y_16gf_backbone_cfg = {'weights': regnet_y_16gf_weights, 'return_nodes': regnet_y_16gf_return_nodes}

regnet_y_32gf_backbone_cfg = {'weights': regnet_y_32gf_weights, 'return_nodes': regnet_y_32gf_return_nodes}

# ====================================================================================================================

regnet_x_400mf_backbone_cfg = {'weights': regnet_x_400mf_weights, 'return_nodes': regnet_x_400mf_return_nodes}

regnet_x_800mf_backbone_cfg = {'weights': regnet_x_800mf_weights, 'return_nodes': regnet_x_800mf_return_nodes}

regnet_x_1_6gf_backbone_cfg = {'weights': regnet_x_1_6gf_weights, 'return_nodes': regnet_x_1_6gf_return_nodes}

regnet_x_3_2gf_backbone_cfg = {'weights': regnet_x_3_2gf_weights, 'return_nodes': regnet_x_3_2gf_return_nodes}

regnet_x_8gf_backbone_cfg = {'weights': regnet_x_8gf_weights, 'return_nodes': regnet_x_8gf_return_nodes}

regnet_x_16gf_backbone_cfg = {'weights': regnet_x_16gf_weights, 'return_nodes': regnet_x_16gf_return_nodes}

regnet_x_32gf_backbone_cfg = {'weights': regnet_x_32gf_weights, 'return_nodes': regnet_x_32gf_return_nodes}

# ====================================================================================================================

def get_conv_base(basenet):

    if basenet == 'regnet_y_400mf': 
        model = models.regnet_y_400mf(weights=regnet_y_400mf_weights)
        return_nodes = regnet_y_400mf_return_nodes

    elif basenet == 'regnet_y_800mf': 
        model = models.regnet_y_800mf(weights=regnet_y_800mf_weights)
        return_nodes = regnet_y_800mf_return_nodes

    elif basenet == 'regnet_y_1_6gf': 
        model = models.regnet_y_1_6gf(weights=regnet_y_1_6gf_weights)
        return_nodes = regnet_y_1_6gf_return_nodes

    elif basenet == 'regnet_y_3_2gf': 
        model = models.regnet_y_3_2gf(weights=regnet_y_3_2gf_weights)
        return_nodes = regnet_y_3_2gf_return_nodes

    elif basenet == 'regnet_y_8gf': 
        model = models.regnet_y_8gf(weights=regnet_y_8gf_weights)
        return_nodes = regnet_y_8gf_return_nodes

    elif basenet == 'regnet_y_16gf': 
        model = models.regnet_y_16gf(weights=regnet_y_16gf_weights)
        return_nodes = regnet_y_16gf_return_nodes

    elif basenet == 'regnet_y_32gf': 
        model = models.regnet_y_32gf(weights=regnet_y_32gf_weights)
        return_nodes = regnet_y_32gf_return_nodes

    # ------------------------------------------------------------------------------------------------------
    elif basenet == 'regnet_x_400mf': 
        model = models.regnet_x_400mf(weights=regnet_x_400mf_weights)
        return_nodes = regnet_x_400mf_return_nodes

    elif basenet == 'regnet_x_800mf': 
        model = models.regnet_x_800mf(weights=regnet_x_800mf_weights)
        return_nodes = regnet_x_800mf_return_nodes

    elif basenet == 'regnet_x_1_6gf': 
        model = models.regnet_x_1_6gf(weights=regnet_x_1_6gf_weights)
        return_nodes = regnet_x_1_6gf_return_nodes

    elif basenet == 'regnet_x_3_2gf': 
        model = models.regnet_x_3_2gf(weights=regnet_x_3_2gf_weights)
        return_nodes = regnet_x_3_2gf_return_nodes

    elif basenet == 'regnet_x_8gf': 
        model = models.regnet_x_8gf(weights=regnet_x_8gf_weights)
        return_nodes = regnet_x_8gf_return_nodes

    elif basenet == 'regnet_x_16gf': 
        model = models.regnet_x_16gf(weights=regnet_x_16gf_weights)
        return_nodes = regnet_x_16gf_return_nodes

    elif basenet == 'regnet_x_32gf': 
        model = models.regnet_x_32gf(weights=regnet_x_32gf_weights)
        return_nodes = regnet_x_32gf_return_nodes

    return model, return_nodes
# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Extract last 4 intermediate pyramid layer outputs of efficientnet preloaded with imagenet weights
# ---------------------------------------------------------------------------------------------------------------
from torchvision import models

"""  Refer:   0_3_efficientnet_feat_shapes.ipynb """
# ====================================================================================================================
efficientnet_b0_weights = 'IMAGENET1K_V1'
efficientnet_b0_return_nodes = ('features.2.1.add', 
                                'features.3.1.add', 
                                'features.5.2.add',
                                'features.7.0.block.3')
# ---------------------------------------------------------------------------------------------
efficientnet_b1_weights = 'IMAGENET1K_V1'  # 'IMAGENET1K_V1' or 'IMAGENET1K_V2'
efficientnet_b1_return_nodes = ('features.2.2.add', 
                                'features.3.2.add', 
                                'features.5.3.add',
                                'features.7.1.add')
# ---------------------------------------------------------------------------------------------
efficientnet_b2_weights = 'IMAGENET1K_V1'
efficientnet_b2_return_nodes = ('features.2.2.add', 
                                'features.3.2.add', 
                                'features.5.3.add',
                                'features.7.1.add')
# ---------------------------------------------------------------------------------------------
efficientnet_b3_weights = 'IMAGENET1K_V1'
efficientnet_b3_return_nodes = ('features.2.2.add', 
                                'features.3.2.add', 
                                'features.5.4.add',
                                'features.7.1.add')     
# ---------------------------------------------------------------------------------------------
efficientnet_b4_weights = 'IMAGENET1K_V1'
efficientnet_b4_return_nodes = ('features.2.3.add', 
                                'features.3.3.add', 
                                'features.5.5.add',
                                'features.7.1.add')
# ---------------------------------------------------------------------------------------------
efficientnet_b5_weights = 'IMAGENET1K_V1'
efficientnet_b5_return_nodes = ('features.2.4.add', 
                                'features.3.4.add', 
                                'features.5.6.add',
                                'features.7.2.add')      
# ---------------------------------------------------------------------------------------------
efficientnet_b6_weights = 'IMAGENET1K_V1'
efficientnet_b6_return_nodes = ('features.2.5.add', 
                                'features.3.5.add', 
                                'features.5.7.add',
                                'features.7.2.add')
# ---------------------------------------------------------------------------------------------
efficientnet_b7_weights = 'IMAGENET1K_V1'
efficientnet_b7_return_nodes = ('features.2.6.add', 
                                'features.3.6.add', 
                                'features.5.9.add',
                                'features.7.3.add')
# ====================================================================================================================

efficientnet_v2_s_weights = 'IMAGENET1K_V1'
efficientnet_v2_s_return_nodes = ('features.2.3.add', 
                                  'features.3.3.add', 
                                  'features.5.8.add',
                                  'features.6.14.add')
# ---------------------------------------------------------------------------------------------
efficientnet_v2_m_weights = 'IMAGENET1K_V1'
efficientnet_v2_m_return_nodes = ('features.2.4.add', 
                                  'features.3.4.add', 
                                  'features.5.13.add',
                                  'features.7.4.add')
# ---------------------------------------------------------------------------------------------
efficientnet_v2_l_weights = 'IMAGENET1K_V1'
efficientnet_v2_l_return_nodes = ('features.2.6.add', 
                                  'features.3.6.add', 
                                  'features.5.18.add',
                                  'features.7.6.add')
# ====================================================================================================================

efficientnet_b0_backbone_cfg = {'weights': efficientnet_b0_weights, 'return_nodes': efficientnet_b0_return_nodes}

efficientnet_b1_backbone_cfg = {'weights': efficientnet_b1_weights, 'return_nodes': efficientnet_b1_return_nodes}

efficientnet_b2_backbone_cfg = {'weights': efficientnet_b2_weights, 'return_nodes': efficientnet_b2_return_nodes}

efficientnet_b3_backbone_cfg = {'weights': efficientnet_b3_weights, 'return_nodes': efficientnet_b3_return_nodes}

efficientnet_b4_backbone_cfg = {'weights': efficientnet_b4_weights, 'return_nodes': efficientnet_b4_return_nodes}

efficientnet_b5_backbone_cfg = {'weights': efficientnet_b5_weights, 'return_nodes': efficientnet_b5_return_nodes}

efficientnet_b6_backbone_cfg = {'weights': efficientnet_b6_weights, 'return_nodes': efficientnet_b6_return_nodes}

efficientnet_b7_backbone_cfg = {'weights': efficientnet_b7_weights, 'return_nodes': efficientnet_b7_return_nodes}

# =====================================================================================================================

efficientnet_v2_s_backbone_cfg = {'weights': efficientnet_v2_s_weights, 'return_nodes': efficientnet_v2_s_return_nodes}

efficientnet_v2_m_backbone_cfg = {'weights': efficientnet_v2_m_weights, 'return_nodes': efficientnet_v2_m_return_nodes}

efficientnet_v2_l_backbone_cfg = {'weights': efficientnet_v2_l_weights, 'return_nodes': efficientnet_v2_l_return_nodes}

# ====================================================================================================================

def get_conv_base(basenet):

    if basenet == 'efficientnet_b0': 
        model = models.efficientnet_b0(weights=efficientnet_b0_weights)
        return_nodes = efficientnet_b0_return_nodes

    elif basenet == 'efficientnet_b1': 
        model = models.efficientnet_b1(weights=efficientnet_b1_weights)
        return_nodes = efficientnet_b1_return_nodes

    elif basenet == 'efficientnet_b2': 
        model = models.efficientnet_b2(weights=efficientnet_b2_weights)
        return_nodes = efficientnet_b2_return_nodes

    elif basenet == 'efficientnet_b3': 
        model = models.efficientnet_b3(weights=efficientnet_b3_weights)
        return_nodes = efficientnet_b3_return_nodes

    elif basenet == 'efficientnet_b4': 
        model = models.efficientnet_b4(weights=efficientnet_b4_weights)
        return_nodes = efficientnet_b4_return_nodes

    elif basenet == 'efficientnet_b5': 
        model = models.efficientnet_b5(weights=efficientnet_b5_weights)
        return_nodes = efficientnet_b5_return_nodes

    elif basenet == 'efficientnet_b6': 
        model = models.efficientnet_b6(weights=efficientnet_b6_weights)
        return_nodes = efficientnet_b6_return_nodes

    elif basenet == 'efficientnet_b7': 
        model = models.efficientnet_b7(weights=efficientnet_b7_weights)
        return_nodes = efficientnet_b7_return_nodes

    # ------------------------------------------------------------------------------------------------------
    elif basenet == 'efficientnet_v2_s': 
        model = models.efficientnet_v2_s(weights=efficientnet_v2_s_weights)
        return_nodes = efficientnet_v2_s_return_nodes

    elif basenet == 'efficientnet_v2_m': 
        model = models.efficientnet_v2_m(weights=efficientnet_v2_m_weights)
        return_nodes = efficientnet_v2_m_return_nodes

    elif basenet == 'efficientnet_v2_l': 
        model = models.efficientnet_v2_l(weights=efficientnet_v2_l_weights)
        return_nodes = efficientnet_v2_l_return_nodes

    return model, return_nodes
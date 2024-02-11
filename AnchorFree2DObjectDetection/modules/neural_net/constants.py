# --------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : model architecture related constants
# --------------------------------------------------------------------------------------------------------------
import math
from collections import namedtuple
det_named_tuple = namedtuple('det_named_tuple', ['class_logits', 'boxreg_deltas', 'centerness_logits', 'objness_logits'])

_EPS_ = 1e-5

_LEAKY_RELU_NEG_SLOPE_ = 0.01

_BATCHNORM_MOMENTUM_ =  0.1

_NUM_GROUPS_ = 32

_INTERP_MODE_ = 'bilinear'  # 'nearest'

_STEM_CONV_MEAN_INIT_ = 0.0
_STEM_CONV_STD_INIT_ = 0.01
_STEM_CONV_BIAS_INIT_ = 0.0

_CLS_CONV_MEAN_INIT_ = _STEM_CONV_MEAN_INIT_
_CLS_CONV_STD_INIT_ = _STEM_CONV_STD_INIT_
# Use a negative bias in class prediction to improve training. Without this, the training can diverge
_CLS_CONV_BIAS_INIT_ = -math.log(99)    

_BOX_CONV_MEAN_INIT_ = _STEM_CONV_MEAN_INIT_
_BOX_CONV_STD_INIT_ = _STEM_CONV_STD_INIT_
_BOX_CONV_BIAS_INIT_ = _STEM_CONV_BIAS_INIT_

_CTR_CONV_MEAN_INIT_ = _STEM_CONV_MEAN_INIT_
_CTR_CONV_STD_INIT_ = _STEM_CONV_STD_INIT_
_CTR_CONV_BIAS_INIT_ = _STEM_CONV_BIAS_INIT_

_BOX_SCALING_WT_INIT_ = -0.01
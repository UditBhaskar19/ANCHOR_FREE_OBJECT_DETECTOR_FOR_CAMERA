# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Contants related to groundtruth computation        
# ---------------------------------------------------------------------------------------------------------------------
_LARGE_NUM_ = 1e6
_VERY_LARGE_NUM_ = 1e10
_VERY_VERY_LARGE_NUM_ = 1e12

_MATCH_CRITERIA_ = 'closest_box'   # 'closest_box' or 'smallest_area'
_REMOVE_FRINGE_ASSOCIATIONS_ = True
_SHRINK_FACTOR_ = 0.8
_SHRINK_FACTOR_CENTERNESS_ = 0.2
_CENTERNESS_FUNCTION_ = 'gaussian'  # 'gaussian' or 'geometric_mean'

_UPDATE_OBJECT_ID_ = True

# if a annotated bounding box has area less than a a certain threshold, that loss is ignored during training (confusing annotations)
_IGNORED_CLASS_DEFAULT_ID_ = -9999
_IGNORED_BOX_H_THR_ = 8
_IGNORED_BOX_W_THR_ = 8
_IGNORED_BOX_ASPECT_RATIO_UPPER_THR_ = 6.0
_IGNORED_BOX_ASPECT_RATIO_LOWER_THR_= 1 / _IGNORED_BOX_ASPECT_RATIO_UPPER_THR_
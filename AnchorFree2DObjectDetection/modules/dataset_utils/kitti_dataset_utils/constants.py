# ----------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : KITTI Dataset related parameters
# ----------------------------------------------------------------------------------------------------------------
import cv2, math

_INTERPOLATION_MODE_ = cv2.INTER_AREA

# ---------------------------------------------------------------------------------------------------------------
""" Resized image dimension of KITTI dataset """
_IMG_H_ = 375
_IMG_W_ = 1242
_IMG_D_ = 3

scale_factor = math.sqrt(( 360 * 640 ) / ( _IMG_H_ * _IMG_W_ ))
# scale_factor = 1

_IMG_RESIZED_H_ = int(scale_factor * _IMG_H_)
_IMG_RESIZED_W_ = int(scale_factor * _IMG_W_)

_OUT_FEAT_SIZE_H_ = _IMG_RESIZED_H_ // 4
_OUT_FEAT_SIZE_W_ = _IMG_RESIZED_W_ // 4

_STRIDE_H_ = _IMG_RESIZED_H_ / _OUT_FEAT_SIZE_H_
_STRIDE_W_ = _IMG_RESIZED_W_ / _OUT_FEAT_SIZE_W_

# ---------------------------------------------------------------------------------------------------------------
""" if a annotated bounding box has area less than a a certain threshold, that box is removed (annotation errors)"""
_BAD_BBOX_WIDTH_THR_ = 5 
_BAD_BBOX_HEIGHT_THR_ = 5
_BAD_BBOX_ASPECT_RATIO_HIGH_THR_ = 10.0 
_BAD_BBOX_ASPECT_RATIO_LOW_THR_ = 1 / _BAD_BBOX_ASPECT_RATIO_HIGH_THR_

# ---------------------------------------------------------------------------------------------------------------------
""" In this project we are aggregating the objects into 2 categories : 1) Vehicle 2) Pedestrian
    The objects marked with 'DontCare' is ignorned during.
    Optionally we could also ignore objects that are heavily occluded
    After object selection the selected data is saved as a JSON file
"""
_OBJ_LABELS_CONSIDERED_ = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'DontCare', 'Misc']
_NEW_OBJ_LABELS_ = ['vehicle', 'person', 'DontCare']

_LABEL_MAP_NEW_TO_OLD_ = {}
_LABEL_MAP_NEW_TO_OLD_['vehicle'] = ['Car', 'Van', 'Truck', 'Tram']
_LABEL_MAP_NEW_TO_OLD_['person'] = ['Pedestrian', 'Person_sitting', 'Cyclist']
_LABEL_MAP_NEW_TO_OLD_['DontCare'] = ['DontCare', 'Misc']

_LABEL_MAP_OLD_TO_NEW_ = {}
_LABEL_MAP_OLD_TO_NEW_['Car'] = 'vehicle'
_LABEL_MAP_OLD_TO_NEW_['Van'] = 'vehicle'
_LABEL_MAP_OLD_TO_NEW_['Truck'] = 'vehicle'
_LABEL_MAP_OLD_TO_NEW_['Tram'] = 'vehicle'
_LABEL_MAP_OLD_TO_NEW_['Pedestrian'] = 'person'
_LABEL_MAP_OLD_TO_NEW_['Person_sitting'] = 'person'
_LABEL_MAP_OLD_TO_NEW_['Cyclist'] = 'person'
_LABEL_MAP_OLD_TO_NEW_['DontCare'] = 'DontCare'
_LABEL_MAP_OLD_TO_NEW_['Misc'] = 'DontCare'

_TRUNCATED_PROP_CONSIDERED_ = [0, 1]
_OCCLUDED_PROP_CONSIDERED_ = [0, 1, 2]

# ---------------------------------------------------------------------------------------------------------------------
""" In this project we are considering one task 1) object detection. The class names are converted from names to ids """
_OBJ_CLASS_TO_IDX_ = { label: idx for idx, label in enumerate(_NEW_OBJ_LABELS_) }
_IDX_TO_OBJ_CLASS_ = { idx: label for idx, label in enumerate(_NEW_OBJ_LABELS_) }

# ---------------------------------------------------------------------------------------------------------------------
""" Ignored class """
_IGNORED_CLASS_ID_ = _OBJ_CLASS_TO_IDX_['DontCare']

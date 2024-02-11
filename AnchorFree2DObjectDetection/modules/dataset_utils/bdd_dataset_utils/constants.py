# ----------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : BDD Dataset related parameters
# ----------------------------------------------------------------------------------------------------------------
import cv2

_INTERPOLATION_MODE_ = cv2.INTER_AREA

# ---------------------------------------------------------------------------------------------------------------
""" Resized image dimension of BDD dataset """
_IMG_H_ = 720
_IMG_W_ = 1280
_IMG_D_ = 3

_IMG_RESIZED_H_ = _IMG_H_ // 2  
_IMG_RESIZED_W_ = _IMG_W_ // 2 

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

# ---------------------------------------------------------------------------------------------------------------
""" Below are all the available ground-truth labels. 
Out of these only a subset is used for object detection in version 1 of this project.
The considered labels and the corrosponding remappled label is in 'remapped_bdd_utils.py' """

_ALL_OBJ_LABELS_ = ['traffic light','traffic sign',
                    'car', 'bus', 'truck',
                    'rider', 'bike', 'person',
                    'motor','train']

_ALL_TRAFFIC_LIGHT_LABELS_ = ['green', 'red', 'yellow']

_ALL_WEATHER_ = ['clear', 'rainy', 'undefined', 'snowy', 'overcast', 'partly cloudy', 'foggy']
_ALL_SCENE_ = ['city street', 'highway', 'residential', 'parking lot', 'undefined', 'tunnel', 'gas stations']
_ALL_TimeOfDay_ = ['daytime', 'dawn/dusk', 'night', 'undefined']

_ALL_LaneDirection_Labels_ = ['parallel', 'vertical']
_ALL_LaneStyle_Labels_ = ['solid', 'dashed']
_ALL_LaneType_Labels_ = ['road curb',
                       'single white',
                       'double yellow',
                       'single yellow',
                       'crosswalk',
                       'double white',
                       'single other',
                       'double other']

# ---------------------------------------------------------------------------------------------------------------------
""" In this project we are aggregating the objects into 4 categories : 1) car 2) large_vehicle 3) bike 4) person.
    After object selection the selected data is saved as a JSON file
"""
_OBJ_LABELS_CONSIDERED_ = ['car', 'bus', 'truck', 'rider', 'person']
_OBJ_LABELS_NOT_CONSIDERED_ = ['drivable area', 'lane', 'traffic light', 'traffic sign', 'train', 'motor', 'bike']
_NEW_OBJ_LABELS_ = ['vehicle', 'person']

_LABEL_MAP_NEW_TO_OLD_ = {}
_LABEL_MAP_NEW_TO_OLD_['vehicle'] = ['car', 'bus', 'truck']
_LABEL_MAP_NEW_TO_OLD_['person'] = ['rider', 'person']

_LABEL_MAP_OLD_TO_NEW_ = {}
_LABEL_MAP_OLD_TO_NEW_['car'] = 'vehicle'
_LABEL_MAP_OLD_TO_NEW_['bus']   = 'vehicle'
_LABEL_MAP_OLD_TO_NEW_['truck'] = 'vehicle'
_LABEL_MAP_OLD_TO_NEW_['person'] = 'person'
_LABEL_MAP_OLD_TO_NEW_['rider']  = 'person'

# ---------------------------------------------------------------------------------------------------------------------
""" In this project we are considering one task 1) object detection. The class names are converted from names to ids """
_OBJ_CLASS_TO_IDX_ = { label: idx for idx, label in enumerate(_NEW_OBJ_LABELS_) }
_IDX_TO_OBJ_CLASS_ = { idx: label for idx, label in enumerate(_NEW_OBJ_LABELS_) }

_WEATHER_CLASS_TO_IDX_ = { label: idx for idx, label in enumerate(_ALL_WEATHER_) }
_IDX_TO_WEATHER_CLASS_ = { idx: label for idx, label in enumerate(_ALL_WEATHER_) }


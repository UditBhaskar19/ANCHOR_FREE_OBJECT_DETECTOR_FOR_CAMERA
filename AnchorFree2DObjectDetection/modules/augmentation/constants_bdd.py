# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : BDD dataset augmentation parameters/constants
# ---------------------------------------------------------------------------------------------------------------
import cv2

_BORDER_ = 114

_INTERPOLATION_MODE_ = cv2.INTER_AREA

_PROB_GEOMETRIC_  = 0.6

_PROB_MOSAIC_ = 0.8 
_PROB_MIXUP_ = 0.1
_PROB_FLIP_ = 0.5
_PROB_CROP_ = 0.7

_H_GAIN_ = 0.015
_S_GAIN_ = 0.7
_V_GAIN_ = 0.4

_MAX_DEGREE_ = 7
_MAX_TRANSLATE_ = 0.1
_MAX_SCALE_ = 0.25
_MAX_SHEAR_ = 0
_PERSPECTIVE_ = 0

_MIXUP_ALPHA_ = 32.0
_MIXUP_BETA_ = 32.0

_MAX_PIXEL_DROPOUT_ = 2000
_MIN_PIXEL_DROPOUT_ = 10000

_MAX_BLOCK_DROPOUT_ = 100
_MIN_BLOCK_DROPOUT_ = 20
_MAX_BLOCK_H_ = 25
_MIN_BLOCK_H_ = 10
_MAX_BLOCK_W_ = 25
_MIN_BLOCK_W_ = 10

_MAX_START_OFFSET_GRID_DROPOUT_ = 5
_MAX_CELL_H_ = 25
_MIN_CELL_H_ = 10
_MAX_CELL_W_ = 25
_MIN_CELL_W_ = 10
_MAX_GRID_ROWS_ = 10 
_MIN_GRID_ROWS_ = 5 
_MAX_GRID_COLS_ = 10
_MIN_GRID_COLS_ = 5

_SCALED_CROP_MAX_ = 1.0
_SCALED_CROP_MIN_ = 0.6
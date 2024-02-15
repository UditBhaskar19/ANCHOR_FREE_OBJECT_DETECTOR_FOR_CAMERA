# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Class definitions for mixup augmentation functions
# ---------------------------------------------------------------------------------------------------------------
import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------------------------------------------
class mixup():
    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta

    def create_mixup(
        self, 
        image1: np.ndarray, 
        image2: np.ndarray, 
        bbox1: np.ndarray, 
        bbox2: np.ndarray, 
        classlabels1: np.ndarray, 
        classlabels2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        r = np.random.beta(self.alpha, self.beta)
        image = (image1 * r + image2 * (1 - r)).astype(np.uint8)
        bbox = np.concatenate([bbox1, bbox2], axis=0)
        classlabels = np.concatenate([classlabels1, classlabels2], axis=0)
        return image, bbox, classlabels
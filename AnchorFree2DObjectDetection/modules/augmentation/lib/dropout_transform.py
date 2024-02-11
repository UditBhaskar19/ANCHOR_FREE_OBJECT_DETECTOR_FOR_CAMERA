# ---------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : Class definitions for dropout augmentation functions
# ---------------------------------------------------------------------------------------------------------------
import numpy as np

# ---------------------------------------------------------------------------------------------------------------
class pixel_dropout():
    """ 
    num_min : minimum number of pixels to drop (set value to 0)
    num_max : maximum number of pixels to drop
    """
    def __init__(self, num_min: int, num_max: int):
        self.num_min = num_min
        self.num_max = num_max

    def perform_dropout(self, img: np.ndarray) -> np.ndarray:
        n = np.random.randint(self.num_min, self.num_max + 1)
        x = np.random.randint(0, img.shape[1], size=n)
        y = np.random.randint(0, img.shape[0], size=n)
        img[y, x, :] = 0
        return img
    
# ---------------------------------------------------------------------------------------------------------------
class block_dropout():
    """ 
    num_min : minimum number of blocks
    num_max : maximum number of blocks
    h_min : minimum height of the block
    w_min : minimum width of the block
    h_max : maximum height of the block
    w_max : maximum width of the block
    """
    def __init__(
        self, 
        num_min: int, num_max: int, 
        h_min: int, w_min: int,
        h_max: int, w_max: int):

        self.num_min = num_min
        self.num_max = num_max
        self.h_min = h_min
        self.w_min = w_min
        self.h_max = h_max
        self.w_max = w_max


    def perform_dropout(self, img: np.ndarray) -> np.ndarray:
        n = np.random.randint(self.num_min, self.num_max + 1)
        x = np.random.randint(0, img.shape[1] - self.w_max, size=n)
        y = np.random.randint(0, img.shape[0] - self.h_max, size=n)
        h = np.random.randint(self.h_min, self.h_max + 1, size=n)
        w = np.random.randint(self.w_min, self.w_max + 1, size=n)

        x_coord_array = []
        y_coord_array = []

        for i in range(n):
            x_coord, y_coord = np.meshgrid(
                np.arange(start=x[i], stop=x[i] + w[i], step=1, dtype=np.int32), \
                np.arange(start=y[i], stop=y[i] + h[i], step=1, dtype=np.int32))
            x_coord_array.append(x_coord.flatten())
            y_coord_array.append(y_coord.flatten())

        x_coord_array = np.concatenate(x_coord_array, axis=0)
        y_coord_array = np.concatenate(y_coord_array, axis=0)
        img[y_coord_array, x_coord_array, :] = 0
        return img

# ---------------------------------------------------------------------------------------------------------------
class grid_dropout():
    """ 
    max_start_offset : maximum starting location of the block
    h_min : minimum height of the block
    w_min : minimum width of the block
    h_max : maximum height of the block
    w_max : maximum width of the block
    num_row_min : minimum number of rows
    num_row_max : maximum number of rows
    num_col_min : minimum number of cols
    num_col_max : maximum number of cols
    """
    def __init__(self, 
        max_start_offset: int, 
        h_min: int, w_min: int, 
        h_max: int, w_max: int, 
        num_row_min: int, num_row_max: int, 
        num_col_min: int, num_col_max: int):

        self.max_start_offset = max_start_offset
        self.h_min = h_min
        self.w_min = w_min
        self.h_max = h_max
        self.w_max = w_max
        self.num_row_min = num_row_min
        self.num_row_max = num_row_max
        self.num_col_min = num_col_min
        self.num_col_max = num_col_max


    def perform_dropout(self, img: np.ndarray) -> np.ndarray:
        num_rows = np.random.randint(self.num_row_min, self.num_row_max + 1)
        num_cols = np.random.randint(self.num_col_min, self.num_col_max + 1)
        
        h = np.random.randint(self.h_min, self.h_max + 1)
        w = np.random.randint(self.w_min, self.w_max + 1)
        xs = np.random.randint(0, self.max_start_offset)
        ys = np.random.randint(0, self.max_start_offset)
        
        x = np.linspace(start=xs, stop=img.shape[1] - self.w_max, num=num_cols, dtype=np.int32)
        y = np.linspace(start=ys, stop=img.shape[0] - self.h_max, num=num_rows, dtype=np.int32)
        
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()

        x_coord_array = []
        y_coord_array = []
        
        for i in range(x.shape[0]):
            x_coord, y_coord = np.meshgrid(
                np.arange(start=x[i], stop=x[i] + w, step=1, dtype=np.int32), \
                np.arange(start=y[i], stop=y[i] + h, step=1, dtype=np.int32))
            x_coord_array.append(x_coord.flatten())
            y_coord_array.append(y_coord.flatten())

        x_coord_array = np.concatenate(x_coord_array, axis=0)
        y_coord_array = np.concatenate(y_coord_array, axis=0)
        img[y_coord_array, x_coord_array, :] = 0
        return img
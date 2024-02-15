# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : box functions         
# ---------------------------------------------------------------------------------------------------------------------
import torch
from typing import Dict, Tuple

# ---------------------------------------------------------------------------------------------------------------------
def gen_grid_coord(
    width: int, height: int, 
    stride_width: int, stride_height: int, 
    device: str ='cpu') -> torch.Tensor:
    """ Generate a Grid of normalized anchor center coordinates.
    Input : batch size, image/feature_map height,  image/feature_map height
    Return a 4-d Tensor of grid cell coordinates (Batch, H, W, 2)
    """
    xcoord = torch.arange(start=0, end=width,  step=1, dtype=torch.float32, device=device) + 0.5
    ycoord = torch.arange(start=0, end=height, step=1, dtype=torch.float32, device=device) + 0.5
    xcoord *= stride_width
    ycoord *= stride_height
    xcoord = xcoord.unsqueeze(0).repeat(height, 1)
    ycoord = ycoord.unsqueeze(-1).repeat(1, width)
    grid_coords = torch.stack((xcoord, ycoord), dim=-1).reshape(-1, 2).contiguous()
    return grid_coords


def gen_grid_coord_fpn_level(
    feat_width_per_fpn_level: Dict[str, int],
    feat_height_per_fpn_level: Dict[str, int],
    strides_width_per_fpn_level: Dict[str, float],
    strides_height_per_fpn_level: Dict[str, float],
    device: str = 'cpu') -> Dict[str, torch.Tensor]:

    location_coords = { level: None for level in strides_width_per_fpn_level.keys() }
    for level, feat_w in feat_width_per_fpn_level.items():
        feat_h = feat_height_per_fpn_level[level]
        stride_w = strides_width_per_fpn_level[level]
        stride_h = strides_height_per_fpn_level[level]
        grid_coords = gen_grid_coord(feat_w, feat_h, stride_w, stride_h, device)
        location_coords[level] = grid_coords
    return location_coords

# ---------------------------------------------------------------------------------------------------------------------
def compute_box_aspect_ratio_corner(
    box: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    box_width = box[..., 2] - box[..., 0]
    box_height = box[..., 3] - box[..., 1]
    box_aspect_ratio = box_width / box_height
    return box_aspect_ratio, box_width, box_height


def compute_box_aspect_ratio_center(
    box: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    box_width = box[..., 2]
    box_height = box[..., 3]
    box_aspect_ratio = box_width / box_height
    return box_aspect_ratio, box_width, box_height

# ---------------------------------------------------------------------------------------------------------------------
def compute_box_area_corner(
    box: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    box_width = box[..., 2] - box[..., 0]
    box_height = box[..., 3] - box[..., 1]
    box_area = box_width * box_height
    return box_area, box_width, box_height


def compute_box_area_center(
    box: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    box_width = box[..., 2]
    box_height = box[..., 3]
    box_area = box_width * box_height
    return box_area, box_width, box_height

# ---------------------------------------------------------------------------------------------------------------------
def change_box_repr_corner(boxes: torch.Tensor) -> torch.Tensor:
    """ change box representation from centre to corner """
    half_wh = 0.5 * boxes[..., 2:4]
    x1y1 = boxes[..., :2] - half_wh
    x2y2 = boxes[..., :2] + half_wh
    return torch.concat([x1y1, x2y2], dim=-1)


def change_box_repr_center(boxes: torch.Tensor) -> torch.Tensor:
    """ change box representation from corner to centre """
    xy = 0.5*(boxes[..., :2] + boxes[..., 2:4])
    wh = boxes[..., 2:4] - boxes[...,  :2]
    return torch.concat([xy, wh], dim=-1)

# ---------------------------------------------------------------------------------------------------------------------
def compute_box_center(boxes: torch.Tensor) -> torch.Tensor:
    """ compute box center from box coordinates """
    xy = 0.5*(boxes[..., :2] + boxes[..., 2:4])
    return xy

# ---------------------------------------------------------------------------------------------------------------------
def clip_box(boxes: torch.Tensor, max_w: int, max_h: int) -> torch.Tensor:
    """ enforce anchor box coordinates to be within a valid range """
    return torch.stack([torch.clamp(boxes[..., 0], min=0, max=max_w), 
                        torch.clamp(boxes[..., 1], min=0, max=max_h), 
                        torch.clamp(boxes[..., 2], min=0, max=max_w), 
                        torch.clamp(boxes[..., 3], min=0, max=max_h)], axis=-1)

# ---------------------------------------------------------------------------------------------------------------------
def compute_iou_2d(proposals: torch.Tensor, boxes: torch.Tensor, eps: float = 1e-7):
    """ Compute intersection over union between sets of bounding boxes.
    Inputs:
    - proposals: Proposals of shape (A*H*W, 4)
    - bboxes: Ground-truth boxes of shape (N, 4).
    Each ground-truth box is represented as tuple (x_lr, y_lr, x_rb, y_rb).
    
    Outputs:
    - iou_mat: IoU matrix of shape (A*H*W, N) where iou_mat[b, i, n] gives
    the IoU between one element of proposals[b] and bboxes[b, n].
    """
    num_gt, _ = boxes.shape
    proposals_area = (proposals[:,2] - proposals[:,0]) * (proposals[:,3] - proposals[:,1])
    boxes_area = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])

    proposals_area = proposals_area.unsqueeze(dim=-1).repeat(1, num_gt)
    boxes_area = boxes_area.unsqueeze(dim=0)

    proposals = proposals.unsqueeze(dim=1).repeat(1, num_gt, 1)
    boxes = boxes.unsqueeze(dim=0)

    top_left  = torch.maximum(proposals[:,:,:2], boxes[:,:,:2])
    bot_right = torch.minimum(proposals[:,:,2:], boxes[:,:,2:])
    difference = torch.clamp(bot_right - top_left, min=0)

    area_intersection = difference[:,:,0] * difference[:,:,1]
    union = proposals_area + boxes_area - area_intersection + eps
    iou = area_intersection / union
    return iou

# ---------------------------------------------------------------------------------------------------------------------
def compute_pairwise_iou(boxsA: torch.Tensor, boxsB: torch.Tensor, eps: float = 1e-7):
    boxsA_area = (boxsA[..., 2] - boxsA[..., 0]) * (boxsA[..., 3] - boxsA[..., 1])
    boxsB_area = (boxsB[..., 2] - boxsB[..., 0]) * (boxsB[..., 3] - boxsB[..., 1])

    top_left  = torch.maximum(boxsA[..., :2], boxsB[..., :2]) 
    bot_right = torch.minimum(boxsA[..., 2:], boxsB[..., 2:])
    difference = torch.clamp(bot_right - top_left, min=0)

    area_intersection = difference[..., 0] * difference[..., 1]
    union = boxsA_area + boxsB_area - area_intersection + eps
    iou = area_intersection / union
    return iou

# ---------------------------------------------------------------------------------------------------------------------
def normalize_box(box: torch.Tensor, width: int, height: int):
    box_clone = box.clone()
    box_clone[..., [0,2]] /= width
    box_clone[..., [1,3]] /= height
    return box_clone

def uunormalize_box(box: torch.Tensor, width: int, height: int):
    box_clone = box.clone()
    box_clone[..., [0,2]] *= width
    box_clone[..., [1,3]] *= height
    return box_clone

def normalize_box_center(box_cntr: torch.Tensor, width: int, height: int):
    box_clone = box_cntr.clone()
    box_clone[..., 0] /= width
    box_clone[..., 1] /= height
    return box_clone

def uunormalize_box_center(box_cntr: torch.Tensor, width: int, height: int):
    box_clone = box_cntr.clone()
    box_clone[..., 0] *= width
    box_clone[..., 1] *= height
    return box_clone
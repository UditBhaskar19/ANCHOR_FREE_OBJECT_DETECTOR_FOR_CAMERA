# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : generate ground truths         
# ---------------------------------------------------------------------------------------------------------------------
import torch
from typing import List, Dict
from collections import namedtuple
from modules.proposal.box_association import fcos_match_locations_to_gt_main
from modules.proposal.prop_functions import (
    get_bbox_deltas_normalized, 
    compute_centerness_gaussian, shrink_bbox_and_compute_centerness)
from modules.proposal.constants import (
    _SHRINK_FACTOR_, _MATCH_CRITERIA_, _IGNORED_CLASS_DEFAULT_ID_,
    _SHRINK_FACTOR_CENTERNESS_, _CENTERNESS_FUNCTION_)

det_named_tuple \
    = namedtuple('det_named_tuple', 
                    ['class_logits', 
                    'boxreg_deltas', 
                    'centerness_logits', 
                    'objness_logits', 
                    'bbox'])

# ---------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def gen_training_gt(
    gt_boxes: List[torch.Tensor], 
    gt_class: List[torch.Tensor],
    strides_width: float,
    strides_height: float,
    grid_coord: torch.Tensor,
    deltas_mean: torch.Tensor, 
    deltas_std: torch.Tensor,
    device: str,
    ignored_classId: int = _IGNORED_CLASS_DEFAULT_ID_):
    
    B = len(gt_boxes)
    matched_gt_class = []
    matched_gt_boxes = []
    matched_gt_deltas = []
    matched_gt_objness = []
    matched_grid_coord = []

    for gtbox, gtcls in zip(gt_boxes, gt_class):

        matched_gt_class_b, matched_gt_boxes_b = fcos_match_locations_to_gt_main(
            locations = grid_coord,
            gt_boxes = gtbox,
            gt_class = gtcls,
            device = device,
            ignored_classId = ignored_classId,
            shrink_factor = _SHRINK_FACTOR_,
            match_criteria = _MATCH_CRITERIA_)
        
        neg_mask = matched_gt_class_b == -1
        ignored_mask = matched_gt_class_b == -2
        
        # for objectness gt
        matched_gt_objness_b = torch.full((matched_gt_class_b.shape[0],  ), 1.0, dtype=torch.float32, device=device)
        matched_gt_objness_b[neg_mask] = 0.0
        matched_gt_objness_b[ignored_mask] = 0.0
        
        matched_gt_deltas_b = get_bbox_deltas_normalized(
            grid_coord, matched_gt_boxes_b, 
            deltas_mean, deltas_std)
        
        matched_gt_deltas_b[neg_mask, :] = -1
        matched_gt_deltas_b[ignored_mask, :] = -1

        matched_gt_class.append(matched_gt_class_b)
        matched_gt_boxes.append(matched_gt_boxes_b)
        matched_gt_deltas.append(matched_gt_deltas_b)
        matched_gt_objness.append(matched_gt_objness_b)

    # tensors of shape (batch_size, locations_per_fpn_level, dimension) / (batch_size, locations_per_fpn_level, )
    matched_gt_class = torch.stack(matched_gt_class, dim=0)
    matched_gt_boxes = torch.stack(matched_gt_boxes, dim=0)
    matched_gt_deltas = torch.stack(matched_gt_deltas, dim=0)
    matched_gt_objness = torch.stack(matched_gt_objness, dim=0)
    matched_grid_coord = grid_coord.unsqueeze(0).repeat(B,1,1)
    
    if _CENTERNESS_FUNCTION_ == 'gaussian':
        matched_gt_centerness = compute_centerness_gaussian(
            matched_grid_coord, matched_gt_boxes, _SHRINK_FACTOR_CENTERNESS_)
    elif _CENTERNESS_FUNCTION_ == 'geometric_mean':
        matched_gt_centerness = shrink_bbox_and_compute_centerness(
            matched_grid_coord, matched_gt_boxes, 
            strides_width, strides_height, _SHRINK_FACTOR_)
    else: raise ValueError('wrong option.')

    return \
        det_named_tuple(
            matched_gt_class, 
            matched_gt_deltas, 
            matched_gt_centerness, 
            matched_gt_objness, 
            matched_gt_boxes)
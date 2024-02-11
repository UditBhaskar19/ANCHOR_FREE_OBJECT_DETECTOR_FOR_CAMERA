# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : anchor to box association functions
# ---------------------------------------------------------------------------------------------------------------------
import torch
from typing import Dict, Tuple
from modules.proposal import constants
from modules.proposal.box_functions import compute_iou_2d, change_box_repr_corner, change_box_repr_center

# ---------------------------------------------------------------------------------------------------------------------
def identify_prominent_objects(gt_boxes: torch.Tensor) -> torch.Tensor:
    box_width = gt_boxes[:,2] - gt_boxes[:,0]
    box_height = gt_boxes[:,3] - gt_boxes[:,1]
    box_height = torch.where(box_height > 0, box_height, 1e-10)
    box_aspect_ratio = box_width / box_height
    valid_condition = ( box_width >= constants._IGNORED_BOX_W_THR_ ) & \
                      ( box_height >= constants._IGNORED_BOX_H_THR_ ) & \
                      ( box_aspect_ratio >= constants._IGNORED_BOX_ASPECT_RATIO_LOWER_THR_ ) & \
                      ( box_aspect_ratio <= constants._IGNORED_BOX_ASPECT_RATIO_UPPER_THR_)    
    return valid_condition

# ---------------------------------------------------------------------------------------------------------------------
def check_if_points_are_within_bboxes(
    centers: torch.Tensor,
    gt_boxes: torch.Tensor):
    x, y = centers.unsqueeze(2).unbind(1) 
    x1, y1, x2, y2 = gt_boxes[:, :4].unsqueeze(0).unbind(2) 
    pairwise_dist = torch.stack([x - x1, y - y1, x2 - x, y2 - y], dim=-1)
    match_matrix = torch.min(pairwise_dist, dim=-1).values > 0
    return match_matrix, x, y, x1, y1, x2, y2

# ---------------------------------------------------------------------------------------------------------------------
def compute_match_quality(
    match_matrix: torch.Tensor, 
    distance: torch.Tensor):
    match_quality = match_matrix.to(torch.float32) * distance
    match_quality = torch.where(match_matrix, match_quality, constants._VERY_VERY_LARGE_NUM_)
    match_quality, matched_idxs = torch.min(match_quality, dim=-1)
    return match_quality, matched_idxs

# ---------------------------------------------------------------------------------------------------------------------
def compute_euclidean_distance(
    x: torch.Tensor, y: torch.Tensor, 
    x1: torch.Tensor, y1: torch.Tensor, 
    x2: torch.Tensor, y2: torch.Tensor):
    gt_center_x = (x1 + x2) * 0.5
    gt_center_y = (y1 + y2) * 0.5
    euclidean_dist = torch.sqrt((x - gt_center_x)**2 + (y - gt_center_y)**2)
    return euclidean_dist

def compute_box_area(
    x1: torch.Tensor, y1: torch.Tensor, 
    x2: torch.Tensor, y2: torch.Tensor):
    gt_areas = (x2 - x1) * (y2 - y1)
    return gt_areas

# ---------------------------------------------------------------------------------------------------------------------
def match_criteria_closest_gt_box_cntr(
    centers: torch.Tensor,
    gt_boxes: torch.Tensor):

    # anchor point must be inside GT.
    match_matrix, \
    x, y, \
    x1, y1, x2, y2 = check_if_points_are_within_bboxes(centers, gt_boxes)

    # euclidean distance between grid centers and bbox centers
    euclidean_dist = compute_euclidean_distance(x, y, x1, y1, x2, y2)

    # Match with the closest GT box center, if there are multiple GT matches.
    match_quality, matched_idxs = compute_match_quality(match_matrix, euclidean_dist)
    return match_quality, matched_idxs

# ---------------------------------------------------------------------------------------------------------------------
def match_criteria_smallest_gt_box_area(
    centers: torch.Tensor,
    gt_boxes: torch.Tensor):

    # anchor point must be inside GT.
    match_matrix, _, _, \
    x1, y1, x2, y2 = check_if_points_are_within_bboxes(centers, gt_boxes)

    # compute gt box area
    gt_areas = compute_box_area(x1, y1, x2, y2)

    # Match with the smallest GT box area, if there are multiple GT matches.
    match_quality, matched_idxs = compute_match_quality(match_matrix, gt_areas)
    return match_quality, matched_idxs

# ---------------------------------------------------------------------------------------------------------------------
def update_matched_gt(
    gt_boxes: torch.Tensor, 
    gt_class: torch.Tensor, 
    match_quality: torch.Tensor, 
    matched_idxs: torch.Tensor, 
    matched_gt: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

    pos_loc_flag = match_quality < constants._VERY_VERY_LARGE_NUM_
    neg_loc_flag = torch.logical_not(pos_loc_flag)

    # matched 
    matched_gt['matched_boxes'] = gt_boxes[matched_idxs]
    matched_gt['matched_class'] = gt_class[matched_idxs]
    matched_gt['matched_boxes'][neg_loc_flag, :] = -1
    matched_gt['matched_class'][neg_loc_flag] = -1

    if constants._UPDATE_OBJECT_ID_:
        object_ids = torch.arange(len(gt_class)).to(gt_boxes.device)
        matched_gt['matched_obj_ids'] = object_ids[matched_idxs]
        matched_gt['matched_obj_ids'][neg_loc_flag] = -1
    return matched_gt

# ---------------------------------------------------------------------------------------------------------------------
def match_locations_to_gt(
    centers: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_class: torch.Tensor,
    device: str,
    match_criteria: str) -> Dict[str, torch.Tensor]:

    matched_gt = { 'matched_boxes': None, 'matched_class': None }

    if gt_boxes.shape[0] == 0:
        matched_gt['matched_boxes'] = torch.full((centers.shape[0], 4), -1, dtype=torch.float32, device=device)
        matched_gt['matched_class'] = torch.full((centers.shape[0],  ), -1, dtype=torch.float32, device=device)
        return matched_gt
    
    if match_criteria == 'closest_box':
        match_quality, matched_idxs = match_criteria_closest_gt_box_cntr(centers, gt_boxes)
    elif match_criteria == 'smallest_area':
        match_quality, matched_idxs = match_criteria_smallest_gt_box_area(centers, gt_boxes)
    else: raise ValueError('wrong option.')

    matched_gt = update_matched_gt(gt_boxes, gt_class, match_quality, matched_idxs, matched_gt)
    return matched_gt

# ---------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def fcos_match_locations_to_gt(
    locations: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_class: torch.Tensor,
    device: str,
    match_criteria: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Match centers of the locations of FPN feature with a set of GT boundingboxes of the input image. """

    matched_gt \
        = match_locations_to_gt(
            centers = locations,
            gt_boxes = gt_boxes,
            gt_class = gt_class,
            device = device,
            match_criteria = match_criteria)

    matched_gt_boxes = matched_gt['matched_boxes']
    matched_gt_class = matched_gt['matched_class']
    return (matched_gt_class, matched_gt_boxes)

# ---------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def fcos_match_locations_to_gt_all_levels(
    locations_per_fpn_level: Dict[str, torch.Tensor],
    gt_boxes: torch.Tensor,
    gt_class: torch.Tensor,
    device: str,
    match_criteria: str = constants._MATCH_CRITERIA_) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """ Match centers of the locations of FPN feature with a set of GT boundingboxes of the input image. """

    matched_gt_boxes = { level_name: None for level_name in locations_per_fpn_level.keys() }
    matched_gt_class = { level_name: None for level_name in locations_per_fpn_level.keys() }

    for level, centers in locations_per_fpn_level.items():

       matched_gt \
        = match_locations_to_gt(
            centers = centers,
            gt_boxes = gt_boxes,
            gt_class = gt_class,
            device = device,
            match_criteria = match_criteria)
       
       matched_gt_boxes[level] = matched_gt['matched_boxes']
       matched_gt_class[level] = matched_gt['matched_class']
    return (matched_gt_class, matched_gt_boxes)

# ---------------------------------------------------------------------------------------------------------------------
def shrink_bbox(gt_boxes: torch.Tensor, shrink_factor: float):
    valid_boxes_xywh = change_box_repr_center(gt_boxes)
    valid_boxes_xywh[:, 2:4] = valid_boxes_xywh[:, 2:4] * shrink_factor
    valid_boxes_xyxy = change_box_repr_corner(valid_boxes_xywh)
    return valid_boxes_xyxy

# ---------------------------------------------------------------------------------------------------------------------
def shrink_gt_boxes_recompute_associations(
    locations: torch.Tensor,
    matched_gt_class: torch.Tensor,
    matched_gt_boxes: torch.Tensor,
    shrink_factor: float) -> Tuple[torch.Tensor, torch.Tensor]:

    # extract the valid associations : class and bbox
    valid_flag = matched_gt_class != -1
    valid_boxes = matched_gt_boxes[valid_flag]
    valid_class = matched_gt_class[valid_flag]
    valid_locations = locations[valid_flag]

    # shrink the bbox by a constant factor 'shrink_factor'
    valid_boxes_xyxy = shrink_bbox(valid_boxes, shrink_factor)

    # identify associations that are outside the shrinked bbox
    x1, y1, x2, y2 = valid_boxes_xyxy.unbind(-1)
    x, y = valid_locations.unbind(-1)
    outside_box_flag = (x < x1) | (y < y1) | (x > x2) | (y > y2)

    # update the new associations
    valid_class[outside_box_flag] = -1
    valid_boxes[outside_box_flag] = -1
    matched_gt_class[valid_flag] = valid_class
    matched_gt_boxes[valid_flag] = valid_boxes
    return (matched_gt_class, matched_gt_boxes)

# ---------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def fcos_match_locations_to_gt_main(
    locations: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_class: torch.Tensor,
    device: str,
    ignored_classId: int = constants._IGNORED_CLASS_DEFAULT_ID_,
    shrink_factor: float = constants._SHRINK_FACTOR_,
    match_criteria: str = constants._MATCH_CRITERIA_) -> Tuple[torch.Tensor, torch.Tensor]:

    # compute valid condition
    valid_condition1 = identify_prominent_objects(gt_boxes)
    valid_condition2 = (gt_class != ignored_classId)
    valid_condition = torch.logical_and(valid_condition1, valid_condition2)

    # compute associations valid bbox
    matched_gt_class1, matched_gt_boxes1 \
        = fcos_match_locations_to_gt(
            locations = locations,
            gt_boxes = gt_boxes[valid_condition],
            gt_class = gt_class[valid_condition],
            device = device,
            match_criteria = match_criteria)
    
    # optionaly recompute associations
    if constants._REMOVE_FRINGE_ASSOCIATIONS_:
        matched_gt_class1, matched_gt_boxes1 \
            = shrink_gt_boxes_recompute_associations(
                locations = locations, 
                matched_gt_class = matched_gt_class1, 
                matched_gt_boxes = matched_gt_boxes1, 
                shrink_factor = shrink_factor)

    # compute associations with all bbox (this will be used to compute ignored labels)
    matched_gt_class2, matched_gt_boxes2 \
        = fcos_match_locations_to_gt(
            locations = locations,
            gt_boxes = gt_boxes,
            gt_class = gt_class,
            device = device,
            match_criteria = match_criteria)
    
    # merge the associations
    flag = matched_gt_class2 != -1
    matched_gt_class2[flag] = -2
    matched_gt_boxes2[flag] = -1

    flag = matched_gt_class1 != -1
    matched_gt_class2[flag] = matched_gt_class1[flag]
    matched_gt_boxes2[flag] = matched_gt_boxes1[flag]
    return (matched_gt_class2, matched_gt_boxes2)

# ---------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def greedy_association(
    proposal_boxs: torch.Tensor, 
    gt_boxs: torch.Tensor,
    device: str):

    num_proposal_boxs, num_gt_boxs = proposal_boxs.shape[0], gt_boxs.shape[0]
    proposal_gt_map = torch.full((num_proposal_boxs, ), -1, dtype=torch.long, device=device)
    proposal_gt_iou = torch.full((num_proposal_boxs, ), -1, dtype=torch.float32, device=device)
    iou = compute_iou_2d(proposal_boxs, gt_boxs)

    for _ in range(min(num_gt_boxs, num_proposal_boxs)):
        max_idx = torch.argmax(iou)  # Find the largest IoU
        proposal_idx = (max_idx / num_gt_boxs).long()
        gtbox_idx = (max_idx % num_gt_boxs).long()
        proposal_gt_map[proposal_idx] = gtbox_idx
        proposal_gt_iou[proposal_idx] = iou[proposal_idx, gtbox_idx]
        iou[:, gtbox_idx] = -1.0
        iou[proposal_idx, :] = -1.0
    
    return {
        'proposal_gt_map': proposal_gt_map,
        'proposal_gt_iou': proposal_gt_iou,
    }
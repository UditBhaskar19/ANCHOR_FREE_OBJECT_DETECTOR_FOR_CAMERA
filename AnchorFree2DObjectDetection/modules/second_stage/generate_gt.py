# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : < Work in progress >        
# ---------------------------------------------------------------------------------------------------------------------
import torch
from typing import List
from collections import namedtuple
from modules.proposal.constants import _IGNORED_CLASS_DEFAULT_ID_
from modules.proposal.box_association import greedy_association, identify_prominent_objects
from modules.second_stage.prop_functions import get_bbox_offsets_normalized

det_named_tuple = \
    namedtuple('det_named_tuple', \
               ['class_logits', 'boxreg_deltas', 'objness_logits', 'bbox'])

# ---------------------------------------------------------------------------------------------------------------------
def gen_training_gt_single_image(
    predboxes: torch.Tensor,
    gtboxes: torch.Tensor,
    gtcls: torch.Tensor,
    deltas_mean: torch.Tensor, 
    deltas_std: torch.Tensor,
    iou_threshold: float,
    ignored_classId: int):

    if gtboxes.shape[0] == 0:
        matched_gt_objness_b = torch.full_like(input=predboxes[:,0], fill_value=0.0)
        matched_gt_box_cls_b = torch.full_like(input=predboxes[:,0], fill_value=-1.0)
        matched_gt_box_coord_b = torch.full_like(input=predboxes[:, :4], fill_value=-1.0)
        matched_gt_deltas_b = torch.full_like(input=predboxes[:, :4], fill_value=-1.0)

    else:
        # perform greedy association
        assoinfo = greedy_association(predboxes, gtboxes, predboxes.device)
        matched_gt_box_idx = assoinfo['proposal_gt_map']
        matched_gt_box_iou = assoinfo['proposal_gt_iou']

        # initially compute the positive and negative sample flags
        positive_flag = (matched_gt_box_iou >= iou_threshold) & (matched_gt_box_idx != -1)
        negative_flag = ~positive_flag

        # extract and update the associated info
        matched_gt_box_coord_b = gtboxes[matched_gt_box_idx]
        matched_gt_box_cls_b = gtcls[matched_gt_box_idx]
        matched_gt_box_coord_b[negative_flag] = -1.0
        matched_gt_box_cls_b[negative_flag] = -1.0

        # compute the ignored flag and update the info
        invalid_bbox_flag = ~identify_prominent_objects(gtboxes)
        invalid_bbox_flag = invalid_bbox_flag[matched_gt_box_idx]
        ignored_flag = (matched_gt_box_cls_b == ignored_classId) | ( invalid_bbox_flag & positive_flag )
        matched_gt_box_cls_b[ignored_flag] = -2.0

        # recompute the positive negative and ignored flags
        positive_flag = matched_gt_box_cls_b >= 0
        negative_flag = matched_gt_box_cls_b == -1
        ignored_flag = matched_gt_box_cls_b == -2

        # for objectness gt
        matched_gt_objness_b = torch.full_like(input=matched_gt_box_cls_b, fill_value=1.0)
        matched_gt_objness_b[negative_flag] = 0.0
        matched_gt_objness_b[ignored_flag] = 0.0

        # compute gt box regression offsets
        matched_gt_deltas_b = torch.full_like(input=matched_gt_box_coord_b, fill_value=-1.0)
        matched_gt_deltas_b[positive_flag] = get_bbox_offsets_normalized(
            predboxes[positive_flag], 
            matched_gt_box_coord_b[positive_flag], 
            deltas_mean, deltas_std)
    
    return \
        matched_gt_objness_b, \
        matched_gt_box_cls_b, \
        matched_gt_box_coord_b, \
        matched_gt_deltas_b

# ---------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def gen_training_gt(
    gt_boxes: List[torch.Tensor], 
    gt_class: List[torch.Tensor],
    pred_boxes: List[torch.Tensor],
    deltas_mean: torch.Tensor, 
    deltas_std: torch.Tensor,
    iou_threshold: float,
    ignored_classId: int = _IGNORED_CLASS_DEFAULT_ID_):

    B = len(gt_boxes)
    matched_gt_class = []
    matched_gt_boxes = []
    matched_gt_deltas = []
    matched_gt_objness = []

    for b, (gtboxes, gtcls, predboxes) in enumerate(zip(gt_boxes, gt_class, pred_boxes)):

        matched_gt_objness_b, \
        matched_gt_class_b, \
        matched_gt_boxes_b, \
        matched_gt_deltas_b \
            = gen_training_gt_single_image(
                    predboxes = predboxes, 
                    gtboxes = gtboxes, 
                    gtcls = gtcls, 
                    deltas_mean = deltas_mean, 
                    deltas_std = deltas_std, 
                    iou_threshold = iou_threshold, 
                    ignored_classId = ignored_classId)
        
        matched_gt_class.append(matched_gt_class_b)
        matched_gt_boxes.append(matched_gt_boxes_b)
        matched_gt_deltas.append(matched_gt_deltas_b)
        matched_gt_objness.append(matched_gt_objness_b)

    return \
        det_named_tuple(
            torch.concat(matched_gt_class, dim=0), 
            torch.concat(matched_gt_deltas, dim=0), 
            torch.concat(matched_gt_objness, dim=0), 
            torch.concat(matched_gt_boxes, dim=0))

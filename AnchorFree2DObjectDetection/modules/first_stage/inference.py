# --------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : various inference functions
# --------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.proposal.prop_functions import compute_bbox_from_deltas_normalized
from modules.proposal.nms import class_spec_nms
from modules.proposal.box_functions import compute_box_center

# --------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def inference(
    detector: nn.Module,
    images: torch.Tensor,
    grid_coord: torch.Tensor,
    deltas_mean: torch.Tensor, 
    deltas_std: torch.Tensor,
    test_score_thresh: torch.Tensor,
    test_nms_thresh: float):

    # We index predictions by `[0]` to remove batch dimension.
    predictions = detector(images)
    pred_boxreg_deltas = predictions.boxreg_deltas[0]
    pred_obj_logits = predictions.objness_logits[0]
    pred_cls_logits = predictions.class_logits[0]
    pred_ctr_logits = predictions.centerness_logits[0]

    # STEP 1: compute class prediction
    cls_prob = F.softmax(pred_cls_logits, dim=-1)
    cls_score, cls_idx = torch.max(cls_prob, dim=-1)
    pred_ctr_score = pred_ctr_logits.sigmoid_().reshape(-1)
    pred_obj_score = pred_obj_logits.sigmoid_().reshape(-1)
    pred_score = torch.sqrt( cls_score * pred_obj_score * pred_ctr_score )

    # STEP 2: compute class specific detection thresholds & extract positive detections
    score_thresh = test_score_thresh[cls_idx]
    valid_obj_mask = pred_score > score_thresh
    pred_boxreg_deltas = pred_boxreg_deltas[valid_obj_mask]
    grid_coord = grid_coord[valid_obj_mask]
    pred_score = pred_score[valid_obj_mask]
    cls_idx = cls_idx[valid_obj_mask]
    cls_prob = cls_score[valid_obj_mask]

    # STEP 3: compute bounding box and clip to valid dimensions
    # pred_boxes = compute_bbox_from_offsets_normalized(
    #     grid_coord, pred_boxreg_deltas, strides_width, strides_height, deltas_mean, deltas_std)
    pred_boxes = compute_bbox_from_deltas_normalized(
        grid_coord, pred_boxreg_deltas, deltas_mean, deltas_std)
    _, _, H, W = images.shape
    pred_boxes[:, [0,2]] = torch.clamp(pred_boxes[:, [0,2]], min=0, max=W)
    pred_boxes[:, [1,3]] = torch.clamp(pred_boxes[:, [1,3]], min=0, max=H)

    # STEP 4: perform class specific nms
    keep = class_spec_nms(
        pred_boxes,
        pred_score,
        cls_idx,
        test_nms_thresh,
    )

    return {
        'pred_box': pred_boxes[keep],
        'pred_class': cls_idx[keep],
        'pred_score': pred_score[keep],
        'cls_prob': cls_prob[keep]
    }

# --------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def inference_gt_coord_without_nms(
    detector: nn.Module,
    images: torch.Tensor,
    grid_coord: torch.Tensor,
    deltas_mean: torch.Tensor, 
    deltas_std: torch.Tensor,
    gt_objness_score: torch.Tensor):

    # We index predictions by `[0]` to remove batch dimension.
    predictions = detector(images)
    pred_boxreg_deltas = predictions.boxreg_deltas[0]
    pred_obj_logits = predictions.objness_logits[0]
    pred_cls_logits = predictions.class_logits[0]
    # pred_ctr_logits = predictions.centerness_logits[0]

    # extract the gt locations
    valid_obj_mask = gt_objness_score == 1
    pred_cls_logits = pred_cls_logits[valid_obj_mask] 
    obj_logits = pred_obj_logits[valid_obj_mask]
    deltas = pred_boxreg_deltas[valid_obj_mask]
    locations = grid_coord[valid_obj_mask]

    # compute class prediction
    cls_prob = F.softmax(pred_cls_logits, dim=-1)
    cls_score, cls_idx = torch.max(cls_prob, dim=-1)
    
    # compute 1) pred score 2) pred box coordinates, 3) pred box centers
    pred_score = obj_logits.sigmoid_()
    pred_boxes = compute_bbox_from_deltas_normalized(locations, deltas, deltas_mean, deltas_std)
    _, _, H, W = images.shape
    pred_boxes[:, [0,2]] = torch.clamp(pred_boxes[:, [0,2]], min=0, max=W)
    pred_boxes[:, [1,3]] = torch.clamp(pred_boxes[:, [1,3]], min=0, max=H)
    pred_boxes_center = compute_box_center(pred_boxes)

    return {
        'pred_score': pred_score.reshape(-1),
        'pred_boxes': pred_boxes,
        'pred_boxes_center': pred_boxes_center,
        'cls_score': cls_score,
        'cls_idx': cls_idx
    }

# --------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def inference_gt_coord_with_nms(
    detector: nn.Module,
    images: torch.Tensor,
    grid_coord: torch.Tensor,
    deltas_mean: torch.Tensor, 
    deltas_std: torch.Tensor,
    nms_thresh: float,
    gt_objness_score: torch.Tensor):

    # We index predictions by `[0]` to remove batch dimension.
    predictions = detector(images)
    pred_boxreg_deltas = predictions.boxreg_deltas[0]
    pred_obj_logits = predictions.objness_logits[0]
    pred_cls_logits = predictions.class_logits[0]
    pred_ctr_logits = predictions.centerness_logits[0]

    # extract the gt locations
    valid_obj_mask = gt_objness_score == 1
    pred_ctr_logits = pred_ctr_logits[valid_obj_mask]
    pred_cls_logits = pred_cls_logits[valid_obj_mask] 
    pred_obj_logits = pred_obj_logits[valid_obj_mask]
    deltas = pred_boxreg_deltas[valid_obj_mask]
    locations = grid_coord[valid_obj_mask]

    # compute class prediction
    cls_prob = F.softmax(pred_cls_logits, dim=-1)
    cls_score, cls_idx = torch.max(cls_prob, dim=-1)
    pred_ctr_score = pred_ctr_logits.sigmoid_().reshape(-1)
    pred_obj_score = pred_obj_logits.sigmoid_().reshape(-1)
    pred_score = torch.sqrt( cls_score * pred_obj_score * pred_ctr_score )

    # compute predicted box and clip to valid dimensions
    pred_boxes = compute_bbox_from_deltas_normalized(locations, deltas, deltas_mean, deltas_std)
    _, _, H, W = images.shape
    pred_boxes[:, [0,2]] = torch.clamp(pred_boxes[:, [0,2]], min=0, max=W)
    pred_boxes[:, [1,3]] = torch.clamp(pred_boxes[:, [1,3]], min=0, max=H)

    # perform class specific nms
    keep = class_spec_nms(
        pred_boxes,
        pred_score,
        cls_idx,
        nms_thresh,
    )

    return {
        'pred_score': pred_score[keep],
        'obj_score': pred_obj_score[keep],
        'ctr_score': pred_ctr_score[keep],
        'cls_score': cls_score[keep],
        'cls_idx': cls_idx[keep],
        'pred_boxes': pred_boxes[keep]
    }
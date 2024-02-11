# --------------------------------------------------------------------------------------------------------------
# Author Name: Udit Bhaskar
# Description: ROC computation functions
# --------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from modules.proposal.prop_functions import compute_bbox_from_deltas_normalized
from modules.first_stage.generate_gt import gen_training_gt
from modules.proposal.box_association import greedy_association
from modules.proposal.nms import class_spec_nms
from modules.proposal.box_association import identify_prominent_objects

# nms_threshs = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
nms_threshs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
# nms_threshs = np.arange(start=0.0, stop=1.0, step=0.025).tolist()

# --------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def inference_nms_tuning(
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

    pred_boxes = pred_boxes[keep]
    pred_classes = cls_idx[keep]
    pred_scores = pred_score[keep]

    inferences = {
        'pred_boxes': pred_boxes,
        'pred_classes': pred_classes,
        'pred_scores': pred_scores }
    return inferences

# --------------------------------------------------------------------------------------------------------------
def compute_roc_for_nms_param_tuning(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thresh: float,
    device: str):

    associations = greedy_association(pred_boxes, gt_boxes, device)
    proposal_gt_iou = associations['proposal_gt_iou']
    num_detections, num_false_positives, _, _  = compute_TP_FP(proposal_gt_iou, iou_thresh)

    return {
        'num_detections': num_detections.item(),
        'num_false_positives': num_false_positives.item()
    }

# --------------------------------------------------------------------------------------------------------------
def compute_TP_FP(
    proposal_gt_iou: torch.Tensor, 
    iou_thresh: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_mask = proposal_gt_iou >= iou_thresh
    neg_mask = ~pos_mask
    num_detections = pos_mask.sum()
    num_false_positives = neg_mask.sum()
    return num_detections, num_false_positives, pos_mask, neg_mask

# --------------------------------------------------------------------------------------------------------------
def ROC_for_nms_tuning(
    num_images, 
    image_start_idx, 
    detector, 
    dataset, 
    dataset_param,
    iou_thresh, 
    device):
    
    DETECTION_RATE_LIST = []
    FP_RATE_PER_IMAGE_LIST = []
    NMS_THRESH = []

    detector.eval()
    image_end_idx = image_start_idx + num_images
    grid_coord = dataset_param.grid_coord.to(device)
    deltas_mean = torch.tensor(dataset_param.deltas_mean, dtype=torch.float32, device=device)
    deltas_std = torch.tensor(dataset_param.deltas_std, dtype=torch.float32, device=device)

    for _, nms_thresh in enumerate(nms_threshs):
        print('generating detection rate & FP_per_image for nms score: ', round(nms_thresh, 3))
        num_detections = 0
        num_false_positives = 0
        num_gts = 0

        for i in range(image_start_idx, image_end_idx):

            img, labels = dataset.__getitem__(i)
            img = img.unsqueeze(0).to(device)
            bboxes = [labels['bbox'].to(device)]
            clslabels = [labels['obj_class_label'].to(device)]

            groundtruths = gen_training_gt(
                bboxes, 
                clslabels, 
                dataset_param.STRIDE_W,
                dataset_param.STRIDE_H,
                grid_coord,
                deltas_mean,
                deltas_std,
                device)
            
            infer_dict = inference_nms_tuning(
                detector, img, grid_coord,
                deltas_mean, deltas_std,
                nms_thresh, groundtruths.objness_logits[0])

            condition = identify_prominent_objects(bboxes[0])
            bboxes = bboxes[0][condition]
            roc_entries = compute_roc_for_nms_param_tuning(
                infer_dict['pred_boxes'], bboxes,
                iou_thresh, device)
            
            num_detections += roc_entries['num_detections']
            num_false_positives += roc_entries['num_false_positives']
            num_gts += bboxes.shape[0]

        detection_rate = num_detections / num_gts
        false_positives_per_image = num_false_positives / num_images
        
        decimal = 3
        DETECTION_RATE_LIST.append(round(detection_rate, decimal))
        FP_RATE_PER_IMAGE_LIST.append(round(false_positives_per_image, decimal))
        NMS_THRESH.append(round(nms_thresh, decimal))

    return NMS_THRESH, DETECTION_RATE_LIST, FP_RATE_PER_IMAGE_LIST

# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# Author Name: Udit Bhaskar
# Description: ROC computation functions
# --------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from modules.proposal.prop_functions import compute_bbox_from_deltas_normalized
from modules.proposal.box_association import greedy_association
from modules.proposal.nms import class_spec_nms
from modules.proposal.box_association import identify_prominent_objects

# score_threshs = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
score_threshs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
# score_threshs = np.arange(start=0.0, stop=1.0, step=0.025).tolist()

# --------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def inference_score_tuning(
    detector: nn.Module,
    images: torch.Tensor,
    grid_coord: torch.Tensor,
    deltas_mean: torch.Tensor,
    deltas_std: torch.Tensor,
    nms_thresh: float):

    # We index predictions by `[0]` to remove batch dimension.
    predictions = detector(images)
    pred_boxreg_deltas = predictions.boxreg_deltas[0]
    pred_obj_logits = predictions.objness_logits[0]
    pred_cls_logits = predictions.class_logits[0]
    pred_ctr_logits = predictions.centerness_logits[0]

    # compute class prediction
    cls_prob = F.softmax(pred_cls_logits, dim=-1)
    cls_score, cls_idx = torch.max(cls_prob, dim=-1)
    pred_ctr_score = pred_ctr_logits.sigmoid_().reshape(-1)
    pred_obj_score = pred_obj_logits.sigmoid_().reshape(-1)
    pred_score = torch.sqrt( cls_score * pred_obj_score * pred_ctr_score )

    # compute predicted box and clip to valid dimensions
    pred_boxes = compute_bbox_from_deltas_normalized(grid_coord, pred_boxreg_deltas, deltas_mean, deltas_std)
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
def compute_TP_FP(
    proposal_gt_iou: torch.Tensor, 
    iou_thresh: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_mask = proposal_gt_iou >= iou_thresh
    neg_mask = ~pos_mask
    num_detections = pos_mask.sum()
    num_false_positives = neg_mask.sum()
    return num_detections, num_false_positives, pos_mask, neg_mask

# --------------------------------------------------------------------------------------------------------------
def compute_roc_for_score_param_tuning(
    inferences: Dict[str, torch.Tensor],
    gt_box: torch.Tensor,
    gt_cls: torch.Tensor,
    score_thresh: float,
    iou_thresh: float,
    selected_clssid: int,
    device: str):

    mask = ( inferences['pred_scores'] >= score_thresh ) & ( inferences['pred_classes'] == selected_clssid )
    pred_boxes = inferences['pred_boxes'][mask]
    gt_box = gt_box[gt_cls == selected_clssid]

    associations = greedy_association(pred_boxes, gt_box, device)
    proposal_gt_iou = associations['proposal_gt_iou']
    num_detections, num_false_positives, _, _  = compute_TP_FP(proposal_gt_iou, iou_thresh)

    return {
        'num_detections': num_detections.item(),
        'num_false_positives': num_false_positives.item(),
        'num_gts': gt_box.shape[0]
    }

# --------------------------------------------------------------------------------------------------------------
def ROC_for_score_tuning(
    num_images, 
    image_start_idx, 
    selected_clssid, 
    detector, 
    dataset, 
    dataset_param,
    iou_thresh, 
    nms_thresh,
    device):
    
    DETECTION_RATE_LIST = []
    FP_RATE_PER_IMAGE_LIST = []
    SCORE_THRESH = []

    detector.eval()
    image_end_idx = image_start_idx + num_images
    grid_coord = dataset_param.grid_coord.to(device)   
    deltas_mean = torch.tensor(dataset_param.deltas_mean, dtype=torch.float32, device=device)
    deltas_std = torch.tensor(dataset_param.deltas_std, dtype=torch.float32, device=device)

    for _, score_thresh in enumerate(score_threshs):
        print('generating detection rate & FP_per_image for score: ', round(score_thresh, 3))
        num_detections = 0
        num_false_positives = 0
        num_gts = 0

        for i in range(image_start_idx, image_end_idx):

            img, labels = dataset.__getitem__(i)
            img = img.unsqueeze(0).to(device)
            bboxes = labels['bbox'].to(device)
            clslabels = labels['obj_class_label'].to(device)
            
            infer_dict = inference_score_tuning(
                detector, img, grid_coord,
                deltas_mean, deltas_std,
                nms_thresh)
            
            condition = identify_prominent_objects(bboxes)
            bboxes = bboxes[condition]
            clslabels = clslabels[condition]
            roc_entries = compute_roc_for_score_param_tuning(
                infer_dict,
                bboxes, clslabels,
                score_thresh, iou_thresh,
                selected_clssid,
                device=device)
            
            num_detections += roc_entries['num_detections']
            num_false_positives += roc_entries['num_false_positives']
            num_gts += roc_entries['num_gts']

        detection_rate = num_detections / num_gts
        false_positives_per_image = num_false_positives / num_images
        
        decimal = 3
        DETECTION_RATE_LIST.append(round(detection_rate, decimal))
        FP_RATE_PER_IMAGE_LIST.append(round(false_positives_per_image, decimal))
        SCORE_THRESH.append(round(score_thresh, decimal))

    return SCORE_THRESH, DETECTION_RATE_LIST, FP_RATE_PER_IMAGE_LIST

# --------------------------------------------------------------------------------------------------------------
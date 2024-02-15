# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : non maximal supression      
# ---------------------------------------------------------------------------------------------------------------------
import torch, torchvision

# ---------------------------------------------------------------------------------------------------------------------
def class_spec_nms_(boxes, scores, class_labels, iou_threshold=0.5) -> torch.Tensor:

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    
    unique_class_labels = class_labels.unique()
    keep = []
    index_tensor = torch.arange(boxes.shape[0])

    for class_label in unique_class_labels:
        class_mask = (class_labels == class_label)
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        class_index = index_tensor[class_mask]
        keep_indices = torchvision.ops.nms(class_boxes, class_scores, iou_threshold)
        keep.append(class_index[keep_indices])

    keep = torch.concat(keep, dim=0)
    return keep

# ---------------------------------------------------------------------------------------------------------------------
def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5) -> torch.Tensor:
    """
    Class specific NMS.
    Returns:
        keep: torch.long tensor with the indices of the elements that have been
              kept by NMS, sorted in decreasing order of scores;
              of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = torchvision.ops.nms(boxes_for_nms, scores, iou_threshold)
    return keep
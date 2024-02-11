# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : < Work in progress >        
# ---------------------------------------------------------------------------------------------------------------------
import torch
from modules.proposal.box_functions import change_box_repr_center, change_box_repr_corner

# ---------------------------------------------------------------------------------------------------------------------
def get_box_offsets(
    anchors: torch.Tensor, 
    boxes: torch.Tensor) -> torch.Tensor:

    anchors_xywh = change_box_repr_center(anchors)
    boxes_xywh = change_box_repr_center(boxes)
    x_gt, y_gt, w_gt, h_gt = boxes_xywh[..., :4].unbind(-1)
    x_anc, y_anc, w_anc, h_anc = anchors_xywh[..., :4].unbind(-1)
    x_offset = (x_gt - x_anc) / w_anc
    y_offset = (y_gt - y_anc) / h_anc
    w_offset = torch.log(w_gt/w_anc)
    h_offset = torch.log(h_gt/h_anc)
    transforms = torch.stack((x_offset, y_offset, w_offset, h_offset), dim=-1)
    return transforms

# ---------------------------------------------------------------------------------------------------------------------
def compute_box_from_offsets(
    anchors: torch.Tensor,
    box_offsets: torch.Tensor) -> torch.Tensor:

    anchors_xywh = change_box_repr_center(anchors)
    x_anc, y_anc, w_anc, h_anc = anchors_xywh[..., :4].unbind(-1)
    dx, dy, dw, dh = box_offsets[..., :4].unbind(-1)
    x_center = x_anc + dx * w_anc
    y_center = y_anc + dy * h_anc
    w = w_anc * torch.exp(dw)
    h = h_anc * torch.exp(dh)
    output_boxes = torch.stack((x_center, y_center, w, h), dim=-1)
    output_boxes = change_box_repr_corner(output_boxes)
    return output_boxes

# ---------------------------------------------------------------------------------------------------------------------
def get_bbox_offsets_normalized(
    anchors: torch.Tensor, 
    boxes: torch.Tensor,
    offsets_mean: torch.Tensor,
    offsets_std: torch.Tensor) -> torch.Tensor:

    anchors_xywh = change_box_repr_center(anchors)
    boxes_xywh = change_box_repr_center(boxes)
    x_gt, y_gt, w_gt, h_gt = boxes_xywh[..., :4].unbind(-1)
    x_anc, y_anc, w_anc, h_anc = anchors_xywh[..., :4].unbind(-1)
    mu_x, mu_y, mu_w, mu_h = offsets_mean.unbind(-1)
    std_x, std_y, std_w, std_h = offsets_std.unbind(-1)

    x_offset = ( (x_gt - x_anc) / w_anc - mu_x ) / std_x
    y_offset = ( (y_gt - y_anc) / h_anc - mu_y ) / std_y
    w_offset = ( torch.log(w_gt/w_anc) - mu_w ) / std_w
    h_offset = ( torch.log(h_gt/h_anc) - mu_h ) / std_h
    transforms = torch.stack((x_offset, y_offset, w_offset, h_offset), dim=-1)
    return transforms

# ---------------------------------------------------------------------------------------------------------------------
def compute_bbox_from_offsets_normalized(
    anchors: torch.Tensor,
    box_offsets: torch.Tensor,
    offsets_mean: torch.Tensor,
    offsets_std: torch.Tensor,) -> torch.Tensor:

    anchors_xywh = change_box_repr_center(anchors)
    x_anc, y_anc, w_anc, h_anc = anchors_xywh[..., :4].unbind(-1)
    dx, dy, dw, dh = box_offsets[..., :4].unbind(-1)
    mu_x, mu_y, mu_w, mu_h = offsets_mean.unbind(-1)
    std_x, std_y, std_w, std_h = offsets_std.unbind(-1)

    x_center = (dx * std_x + mu_x) * w_anc + x_anc
    y_center = (dy * std_y + mu_y) * h_anc + y_anc
    w = w_anc * torch.exp(dw * std_w + mu_w)
    h = h_anc * torch.exp(dh * std_h + mu_h)
    output_boxes = torch.stack((x_center, y_center, w, h), dim=-1)
    output_boxes = change_box_repr_corner(output_boxes)
    return output_boxes
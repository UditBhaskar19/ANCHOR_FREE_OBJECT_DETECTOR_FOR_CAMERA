# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : gt box transform functions     
# ---------------------------------------------------------------------------------------------------------------------
import torch
from modules.proposal.box_functions import change_box_repr_center, change_box_repr_corner

# ---------------------------------------------------------------------------------------------------------------------
def fcos_get_deltas_from_locations(
    locations: torch.Tensor, 
    boxes: torch.Tensor, 
    stride_width: int,
    stride_height: int) -> torch.Tensor:
    """
    Compute distances from feature locations to GT box edges. 
    These distances are called "deltas" - `(left, top, right, bottom)` or simply `LTRB`. 
    The feature locations and GT boxes are given in absolute image co-ordinates.

    Args:
        locations: Tensor of shape `(N, 2)` giving `(xc, yc)` feature locations.
        boxes: Tensor of shape `(N, 4)`.
        stride: Stride of the FPN feature map.

    Returns: deltas 
        Tensor of shape `(N, 4)` giving deltas from feature locations, that are normalized by feature stride.
    """
    # left, top, right, bottom
    x, y = locations.unbind(-1)
    x1, y1, x2, y2 = boxes[..., :4].unbind(-1)
    dL = ( x - x1 ) / stride_width
    dT = ( y - y1 ) / stride_height
    dR = ( x2 - x ) / stride_width
    dB = ( y2 - y ) / stride_height
    deltas = torch.stack((dL, dT, dR, dB), dim=-1)
    return deltas

# ---------------------------------------------------------------------------------------------------------------------
def fcos_apply_deltas_to_locations(
    locations: torch.Tensor, 
    deltas: torch.Tensor, 
    stride_width: int,
    stride_height: int) -> torch.Tensor:
    """
    Given edge deltas (left, top, right, bottom) and feature locations of FPN, get
    the resulting bounding box co-ordinates by applying deltas on locations. 
    Args:
        deltas: Tensor of shape `(N, 4)` giving edge deltas to apply to locations.
        locations: Locations to apply deltas on. shape: `(N, 2)`
        stride: Stride of the FPN feature map.

    Returns:
        output_boxes
            Same shape as deltas and locations, giving co-ordinates of the
            resulting boxes `(x1, y1, x2, y2)`, absolute in image dimensions.
    """
    x, y = locations.unbind(-1)
    deltas = torch.clamp(deltas, min=0)
    dL, dT, dR, dB = deltas.unbind(-1)
    x1 = x - stride_width * dL 
    y1 = y - stride_height * dT 
    x2 = x + stride_width * dR 
    y2 = y + stride_height * dB 
    output_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return output_boxes




# ---------------------------------------------------------------------------------------------------------------------
def compute_centerness(deltas: torch.Tensor) -> torch.Tensor:
    """
    Given LTRB deltas of boxes, compute the centerness.
    Args: deltas: Tensor of shape `(N, 4)` giving LTRB deltas for GT boxes.
    Returns: centerness: Tensor of shape `(N, )` giving centerness regression targets.
    """
    min_lr = torch.min(deltas[..., [0,2]], dim=-1).values
    max_lr = torch.max(deltas[..., [0,2]], dim=-1).values
    min_tb = torch.min(deltas[..., [1,3]], dim=-1).values
    max_tb = torch.max(deltas[..., [1,3]], dim=-1).values
    centerness = torch.sqrt( ( min_lr / max_lr ) * ( min_tb / max_tb ) )
    return centerness

# ---------------------------------------------------------------------------------------------------------------------
def shrink_bbox_and_compute_centerness(
    locations: torch.Tensor, 
    boxes: torch.Tensor, 
    stride_width: int,
    stride_height: int,
    shrink_factor: float) -> torch.Tensor:

    boxes_xywh = change_box_repr_center(boxes)
    boxes_xywh[:, 2:4] *= shrink_factor
    boxes = change_box_repr_corner(boxes_xywh)

    x, y = locations.unbind(-1)
    x1, y1, x2, y2 = boxes[..., :4].unbind(-1)
    dL = ( x - x1 ) / stride_width
    dT = ( y - y1 ) / stride_height
    dR = ( x2 - x ) / stride_width
    dB = ( y2 - y ) / stride_height
    deltas = torch.stack((dL, dT, dR, dB), dim=-1)
    
    min_lr = torch.min(deltas[..., [0,2]], dim=-1).values
    max_lr = torch.max(deltas[..., [0,2]], dim=-1).values
    min_tb = torch.min(deltas[..., [1,3]], dim=-1).values
    max_tb = torch.max(deltas[..., [1,3]], dim=-1).values
    centerness = torch.sqrt( ( min_lr / max_lr ) * ( min_tb / max_tb ) )
    return centerness

# ---------------------------------------------------------------------------------------------------------------------
def compute_centerness_gaussian(
    locations: torch.Tensor, 
    boxes: torch.Tensor,
    shrink_factor: float) -> torch.Tensor:

    boxes_xywh = change_box_repr_center(boxes)
    x_center, y_center, w, h = boxes_xywh[..., :4].unbind(-1)
    w, h = w*shrink_factor, h*shrink_factor
    x, y = locations.unbind(-1)
    centerness = torch.exp(-0.5*((( x - x_center ) / w) ** 2 + (( y - y_center ) / h) ** 2))
    return centerness




# ---------------------------------------------------------------------------------------------------------------------
def get_bbox_deltas(
    locations: torch.Tensor, 
    boxes: torch.Tensor) -> torch.Tensor :
    # left, top, right, bottom
    x, y = locations.unbind(-1)
    x1, y1, x2, y2 = boxes[..., :4].unbind(-1)
    dL = torch.log( x - x1 )
    dT = torch.log( y - y1 )
    dR = torch.log( x2 - x )
    dB = torch.log( y2 - y )
    deltas = torch.stack((dL, dT, dR, dB), dim=-1)
    return deltas

# ---------------------------------------------------------------------------------------------------------------------
def compute_bbox_from_deltas(
    locations: torch.Tensor, 
    deltas: torch.Tensor) -> torch.Tensor:
    x, y = locations.unbind(-1)
    dL, dT, dR, dB = deltas.unbind(-1)
    x1 = x - torch.exp(dL) 
    y1 = y - torch.exp(dT) 
    x2 = x + torch.exp(dR) 
    y2 = y + torch.exp(dB) 
    output_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return output_boxes

# ---------------------------------------------------------------------------------------------------------------------
def get_bbox_deltas_normalized(
    locations: torch.Tensor, 
    boxes: torch.Tensor,
    offsets_mean: torch.Tensor,
    offsets_std: torch.Tensor,) -> torch.Tensor :
    # left, top, right, bottom
    x, y = locations.unbind(-1)
    x1, y1, x2, y2 = boxes[..., :4].unbind(-1)
    mu_L, mu_T, mu_R, mu_B = offsets_mean.unbind(-1)
    std_L, std_T, std_R, std_B = offsets_std.unbind(-1)
    dL = ( torch.log( x - x1 ) - mu_L ) / std_L
    dT = ( torch.log( y - y1 ) - mu_T ) / std_T
    dR = ( torch.log( x2 - x ) - mu_R ) / std_R
    dB = ( torch.log( y2 - y ) - mu_B ) / std_B
    deltas = torch.stack((dL, dT, dR, dB), dim=-1)
    return deltas

# ---------------------------------------------------------------------------------------------------------------------
def compute_bbox_from_deltas_normalized(
    locations: torch.Tensor, 
    deltas: torch.Tensor,
    offsets_mean: torch.Tensor,
    offsets_std: torch.Tensor,) -> torch.Tensor:
    x, y = locations.unbind(-1)
    dL, dT, dR, dB = deltas.unbind(-1)
    mu_L, mu_T, mu_R, mu_B = offsets_mean.unbind(-1)
    std_L, std_T, std_R, std_B = offsets_std.unbind(-1)
    x1 = x - torch.exp(mu_L + std_L * dL) 
    y1 = y - torch.exp(mu_T + std_T * dT) 
    x2 = x + torch.exp(mu_R + std_R * dR) 
    y2 = y + torch.exp(mu_B + std_B * dB) 
    output_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return output_boxes




# ---------------------------------------------------------------------------------------------------------------------
def get_bbox_offsets(
    locations: torch.Tensor, 
    boxes: torch.Tensor, 
    stride_width: int,
    stride_height: int) -> torch.Tensor:

    x, y = locations.unbind(-1)
    boxes_xywh = change_box_repr_center(boxes)
    x_center, y_center, w, h = boxes_xywh[..., :4].unbind(-1)
    x_offset = ( x_center - x ) / stride_width
    y_offset = ( y_center - y ) / stride_height
    w_offset = torch.log( w / stride_width )
    h_offset = torch.log( h / stride_height )
    transforms = torch.stack((x_offset, y_offset, w_offset, h_offset), dim=-1)
    return transforms

# ---------------------------------------------------------------------------------------------------------------------
def compute_bbox_from_offsets(
    locations: torch.Tensor, 
    box_offsets: torch.Tensor, 
    stride_width: int,
    stride_height: int) -> torch.Tensor:

    x, y = locations.unbind(-1)
    dx, dy, dw, dh = box_offsets[..., :4].unbind(-1)
    x_center = x + dx * stride_width
    y_center = y + dy * stride_height
    w = stride_width * torch.exp(dw)
    h = stride_height * torch.exp(dh)
    output_boxes = torch.stack((x_center, y_center, w, h), dim=-1)
    output_boxes = change_box_repr_corner(output_boxes)
    return output_boxes

# ---------------------------------------------------------------------------------------------------------------------
def get_bbox_offsets_normalized(
    locations: torch.Tensor, 
    boxes: torch.Tensor, 
    stride_width: int,
    stride_height: int,
    offsets_mean: torch.Tensor,
    offsets_std: torch.Tensor,) -> torch.Tensor:

    x, y = locations.unbind(-1)
    boxes_xywh = change_box_repr_center(boxes)
    x_center, y_center, w, h = boxes_xywh[..., :4].unbind(-1)
    mu_x, mu_y, mu_w, mu_h = offsets_mean.unbind(-1)
    std_x, std_y, std_w, std_h = offsets_std.unbind(-1)

    x_offset = ( ( x_center - x ) / stride_width - mu_x ) / std_x
    y_offset = ( ( y_center - y ) / stride_height - mu_y ) / std_y
    w_offset = ( torch.log( w / stride_width ) - mu_w ) / std_w
    h_offset = ( torch.log( h / stride_height ) - mu_h ) / std_h
    transforms = torch.stack((x_offset, y_offset, w_offset, h_offset), dim=-1)
    return transforms

# ---------------------------------------------------------------------------------------------------------------------
def compute_bbox_from_offsets_normalized(
    locations: torch.Tensor, 
    box_offsets: torch.Tensor, 
    stride_width: int,
    stride_height: int,
    offsets_mean: torch.Tensor,
    offsets_std: torch.Tensor,) -> torch.Tensor:

    x, y = locations.unbind(-1)
    dx, dy, dw, dh = box_offsets[..., :4].unbind(-1)
    mu_x, mu_y, mu_w, mu_h = offsets_mean.unbind(-1)
    std_x, std_y, std_w, std_h = offsets_std.unbind(-1)

    x_center = x + (std_x * dx + mu_x) * stride_width
    y_center = y + (std_y * dy + mu_y) * stride_height
    w = stride_width * torch.exp(std_w * dw + mu_w)
    h = stride_height * torch.exp(std_h * dh + mu_h)
    output_boxes = torch.stack((x_center, y_center, w, h), dim=-1)
    output_boxes = change_box_repr_corner(output_boxes)
    return output_boxes

# ---------------------------------------------------------------------------------------------------------------------
# Author : Udit Bhaskar
# Description : < Work in progress >        
# ---------------------------------------------------------------------------------------------------------------------
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Dict
from modules.neural_net.backbone.backbone_v2 import net_backbone
from modules.neural_net.bifpn.bifpn_nblks_v2 import BiFPN
from modules.neural_net.head.shared_head_v5 import SharedNet
from modules.proposal.nms import class_spec_nms
from modules.proposal.prop_functions import compute_bbox_from_deltas_normalized
from modules.second_stage.get_param import net_config_stage2 as net_config
from modules.second_stage.get_param import bdd_parameters_stage2 as bdd_parameters
from modules.second_stage.get_param import kitti_parameters_stage2 as kitti_parameters

# --------------------------------------------------------------------------------------------------------------
class proposal_extractor(nn.Module):
    def __init__(
        self, 
        backbone: net_backbone, 
        feataggregator: BiFPN, 
        sharednet: SharedNet,
        netconfig_obj: net_config,
        param_obj: Union[bdd_parameters, kitti_parameters],
        device: str):
        super().__init__()

        self.backbone = backbone
        self.feataggregator = feataggregator
        self.sharednet = sharednet
        self.disable_training = netconfig_obj.freeze_singlestage_layers_stage2
        self.device = device
        self.nms_thr = netconfig_obj.nms_thr_for_proposal_extraction_stage2
        self.roi_size = netconfig_obj.roi_size_stage2
        self.score_thr = torch.tensor(netconfig_obj.score_thr_for_proposal_extraction_stage2, dtype=torch.float32, device=device)
        # self.feat_pyr_h = param_obj.feat_pyr_h
        # self.feat_pyr_w = param_obj.feat_pyr_w
        self.grid_coord = param_obj.grid_coord.to(device)
        self.deltas_mean = torch.tensor(param_obj.deltas_mean, dtype=torch.float32, device=device)
        self.deltas_std = torch.tensor(param_obj.deltas_std, dtype=torch.float32, device=device)
        if self.disable_training: self.eval()

    def reinit_const_parameters(self, param_obj: Union[bdd_parameters, kitti_parameters]):
        self.grid_coord = param_obj.grid_coord.to(self.device)
        self.sharednet.merge_blk.out_feat_h = param_obj.OUT_FEAT_SIZE_H
        self.sharednet.merge_blk.out_feat_w = param_obj.OUT_FEAT_SIZE_W
        # self.feat_pyr_h = param_obj.feat_pyr_h
        # self.feat_pyr_w = param_obj.feat_pyr_w

        num_feataggregator_blks = len(self.feataggregator.BiFPN_layers)
        for i in range(num_feataggregator_blks):
            self.feataggregator.BiFPN_layers[i].feat_pyr_shapes = param_obj.feat_pyr_shapes
        
    def forward(self, img: torch.Tensor):
        B, _, img_h, img_w = img.shape
        x = self.backbone(img)
        x_pyr = self.feataggregator(x)
        x_out = self.sharednet(x_pyr)

        roi_features = extract_rois(
            img_h = img_h, 
            img_w = img_w, 
            batch_size = B, 
            featmap_pyr = x_pyr, 
            dense_pred_output = x_out,
            grid_coord = self.grid_coord,
            deltas_mean = self.deltas_mean,
            deltas_std = self.deltas_std,
            score_thr = self.score_thr,
            nms_thr = self.nms_thr,
            roi_size = self.roi_size)

        pred_clsidx = roi_features['roi_features']
        pred_boxes = roi_features['pred_boxes']
        features = roi_features['roi_features']
        queries = torch.concat([
            roi_features['pred_qualities'], 
            roi_features['pred_boxes_norm_width'].unsqueeze(-1),
            roi_features['pred_boxes_norm_height'].unsqueeze(-1)], dim=-1).unsqueeze(1)
        
        return {
            'features': features,
            'queries': queries,
            'pred_boxes': pred_boxes,
            'pred_clsidx': pred_clsidx }
        
# --------------------------------------------------------------------------------------------------------------
def extract_rois(
    img_h: int, img_w: int, batch_size: int,
    featmap_pyr: Dict[str, torch.Tensor],
    dense_pred_output: torch.Tensor,
    grid_coord: torch.Tensor,
    deltas_mean: torch.Tensor,
    deltas_std: torch.Tensor,
    score_thr: torch.Tensor,
    nms_thr: float,
    roi_size: int):

    pred_boxes_list = []
    pred_class_list = []
    pred_boxes_norm_width_list = []
    pred_boxes_norm_height_list = []
    pred_qualities_list = []

    for b in range(batch_size):
        predictions = second_stage_proposals(
            img_h = img_h, img_w = img_w,
            boxreg_deltas = dense_pred_output.boxreg_deltas[b],
            objness_logits = dense_pred_output.objness_logits[b],
            class_logits = dense_pred_output.class_logits[b],
            centerness_logits = dense_pred_output.centerness_logits[b],
            grid_coord = grid_coord,
            deltas_mean = deltas_mean, 
            deltas_std = deltas_std, 
            score_thresh = score_thr,
            nms_thresh = nms_thr)
        
        # pred_class = predictions['pred_class']
        # pred_score = predictions['pred_score']
        pred_class_list.append(predictions['pred_class'])
        pred_boxes_list.append(predictions['pred_boxes'])
        norm_boxes_width = (predictions['pred_boxes'][:,2] - predictions['pred_boxes'][:,0]) / img_w
        norm_boxes_height = (predictions['pred_boxes'][:,3] - predictions['pred_boxes'][:,1]) / img_h
        pred_boxes_norm_width_list.append(norm_boxes_width)
        pred_boxes_norm_height_list.append(norm_boxes_height)

        pred_qualities_list.append(
            torch.concat(
                (predictions['cls_prob_softmax'], 
                 predictions['obj_prob'].unsqueeze(-1), 
                 predictions['ctr_score'].unsqueeze(-1)), axis=-1))
        
    roi_feat_pyr = { level_name: None for level_name in featmap_pyr.keys() }
    for key, featmap in featmap_pyr.items():
        _, _, feat_h, feat_w = featmap.shape
        scale_h = feat_h / img_h
        scale_w = feat_w / img_w

        pred_boxes_list_temp = []
        for boxes_ in pred_boxes_list:
            boxes = boxes_.clone()
            boxes[:, [0,2]] *= scale_w
            boxes[:, [1,3]] *= scale_h
            pred_boxes_list_temp.append(boxes)

        roi_features \
            = torchvision.ops.roi_align(
                input = featmap, 
                boxes = pred_boxes_list_temp, 
                output_size = roi_size, 
                spatial_scale = 1.0, 
                aligned = True)
        
        roi_feat_pyr[key] = roi_features

    return {
        'roi_features': roi_feat_pyr,
        'pred_clsidx': pred_class_list,
        'pred_boxes': pred_boxes_list,
        'pred_qualities': torch.concat(pred_qualities_list, axis=0),
        'pred_boxes_norm_width': torch.concat(pred_boxes_norm_width_list, axis=0),
        'pred_boxes_norm_height': torch.concat(pred_boxes_norm_height_list, axis=0)
    }
            
# --------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def second_stage_proposals(
    img_h: int, img_w: int, 
    boxreg_deltas: torch.Tensor,
    objness_logits: torch.Tensor,
    class_logits: torch.Tensor,
    centerness_logits: torch.Tensor,
    grid_coord: torch.Tensor,
    deltas_mean: torch.Tensor, 
    deltas_std: torch.Tensor,
    score_thresh: torch.Tensor,
    nms_thresh: float):

    # STEP 1: compute class prediction
    cls_prob_softmax = F.softmax(class_logits, dim=-1)
    cls_prob, cls_idx = torch.max(cls_prob_softmax, dim=-1)
    ctr_score = centerness_logits.sigmoid_().reshape(-1)
    obj_prob = objness_logits.sigmoid_().reshape(-1)
    pred_score = torch.sqrt( cls_prob * obj_prob * ctr_score )

    # STEP 2a: compute class specific detection thresholds & compute mask
    score_thresh = score_thresh[cls_idx]
    valid_obj_mask = pred_score > score_thresh

    # STEP 2b: extract positive detections and its attributes
    pred_score = pred_score[valid_obj_mask]
    cls_prob_softmax  = cls_prob_softmax[valid_obj_mask]
    obj_prob = obj_prob[valid_obj_mask]
    ctr_score = ctr_score[valid_obj_mask]
    cls_idx = cls_idx[valid_obj_mask]
    
    # STEP 2b: extract bbox info for positive detections
    boxreg_deltas = boxreg_deltas[valid_obj_mask]
    grid_coord = grid_coord[valid_obj_mask]

    # STEP 3: compute bounding box and clip to valid dimensions
    pred_boxes = compute_bbox_from_deltas_normalized(
        grid_coord, boxreg_deltas, deltas_mean, deltas_std)
    pred_boxes[:, [0,2]] = torch.clamp(pred_boxes[:, [0,2]], min=0, max=img_w)
    pred_boxes[:, [1,3]] = torch.clamp(pred_boxes[:, [1,3]], min=0, max=img_h)

    # STEP 4: perform class specific nms
    keep = class_spec_nms(
        pred_boxes,
        pred_score,
        cls_idx,
        nms_thresh,
    )

    return {
        'pred_boxes': pred_boxes[keep],
        'pred_class': cls_idx[keep],
        'pred_score': pred_score[keep],
        'cls_prob_softmax': cls_prob_softmax[keep],
        'obj_prob': obj_prob[keep],
        'ctr_score': ctr_score[keep]
    }



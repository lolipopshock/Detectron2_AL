from ..scoring_utils import *
from typing import List, Tuple, Dict, Any
from functools import partial
from detectron2.structures import Boxes, Instances
from detectron2.modeling.postprocessing import detector_postprocess
import torch
import numpy as np

__all__ = ['ObjectFusion']

def _quantile(t, q):
    # As we are using pytorch 1.4, there is no native
    # Pytorch support for the quantile function.  
    # This implementation is based on 
    # https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30
    return (t.view(-1)
             .kthvalue(1 + round(float(q) * (t.numel() - 1)))
             .values.item())

def _deselect(t, indices):
    """
    Select the elements in t with index not in indices
    """
    selected_indices = [i not in indices for i in range(len(t))]
    return t[selected_indices]


class ObjectFusionRatioScheduler:

    def __init__(self, cfg):
        self._rounds = cfg.AL.TRAINING.ROUNDS 
        self._init   = cfg.AL.OBJECT_FUSION.INITIAL_RATIO
        self._decay  = cfg.AL.OBJECT_FUSION.DECAY
        self._last   = cfg.AL.OBJECT_FUSION.LAST_RATIO

        if self._decay == 'linear':
            self._vals = np.linspace(self._init, self._last, self._rounds)
    
    def __getitem__(self, r):
        return self._vals[r]


class ObjectFusion:

    OVERLAPPING_METRICS = {
        'iou': pairwise_iou,
        'dice_coefficient': pairwise_dice_coefficient,
        'overlap_coefficient': pairwise_overlap_coefficient
    }

    OBJECT_SELECTOR = {
        'top': select_top,
        'above': select_above,
        'nonzero': select_nonzero
    }

    def __init__(self, cfg):
        self._init_overlapping_funcs(cfg)
        self.remove_duplicates = cfg.AL.OBJECT_FUSION.REMOVE_DUPLICATES
        self.remove_duplicates_th = cfg.AL.OBJECT_FUSION.REMOVE_DUPLICATES_TH
        self.recover_missing_objects = cfg.AL.OBJECT_FUSION.RECOVER_MISSING_OBJECTS

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.fusion_ratio = ObjectFusionRatioScheduler(cfg)

    def _init_overlapping_funcs(self, cfg):
        self.overlapping_metric = self.OVERLAPPING_METRICS[cfg.AL.OBJECT_FUSION.OVERLAPPING_METRIC]
        self.overlapping_th = cfg.AL.OBJECT_FUSION.OVERLAPPING_TH
        self.gt_selector = self.OBJECT_SELECTOR[cfg.AL.OBJECT_FUSION.SELECTION_METHOD]
        if cfg.AL.OBJECT_FUSION.SELECTION_METHOD == 'above':
            self.gt_selector = partial(select_above, threshold=self.overlapping_th)

    def combine(self, pred: Instances, 
                      gt: Dict, 
                      round: int):
        """
        Combine the model predictions with the ground-truth
        by replacing the objects in the pred with score_al of 
        top replace_ratio. It will automatically move all the boxes
        to self.device, and the output will be saved back to cpu.
        """
        fusion_ratio = self.fusion_ratio[round]
        gt_boxes = gt['instances'].gt_boxes.to(self.device)
        pred_boxes = pred.pred_boxes.to(self.device)

        score_al_th = _quantile(pred.scores_al, 1-fusion_ratio) 
        # If fusion_ratio is 0.8, then we want to find all objects that 
        # have scores more than 0.2(=1-0.8) quantile. 
        selected_pred_indices = torch.where(pred.scores_al > score_al_th)[0].to(self.device)
        aggregated_score = pred.scores_al[selected_pred_indices].mean().item()

        overlapping_scores = self.overlapping_metric(
                pred_boxes[selected_pred_indices], 
                gt_boxes)

        selected_gt_indices = self.gt_selector(overlapping_scores) 
        selected_gt_indices = list(set(sum(selected_gt_indices, []))) # Remove duplicates in gt boxes
        selected_gt_indices = torch.Tensor(selected_gt_indices).to(self.device).long()

        if self.remove_duplicates:

            selected_gt = gt_boxes[selected_gt_indices]
            selected_pred = _deselect(pred_boxes, selected_pred_indices)

            overlapping_scores = pairwise_overlap_coefficient(selected_pred, selected_gt)
            selected_pred_indices_extra, _ = torch.where(overlapping_scores>self.remove_duplicates_th)
            selected_pred_indices = torch.cat([selected_pred_indices, torch.unique(selected_pred_indices_extra)])

        if self.recover_missing_objects:
            
            combined_boxes = self._join_elements_pred_with_gt(pred_boxes, 
                                                              selected_pred_indices, 
                                                              gt_boxes, 
                                                              selected_gt_indices)

            max_overlapping_for_gt_boxes = pairwise_iou(combined_boxes, gt_boxes).max(dim=0).values
            missing_gt_boxes_indices = torch.where(max_overlapping_for_gt_boxes<=0.05)[0]
            selected_gt_indices = torch.cat([selected_gt_indices, missing_gt_boxes_indices])

        combined_instances = self._fuse_pred_with_gt(pred, selected_pred_indices,
                                                gt, selected_gt_indices)

        result = self._postprocess(combined_instances, gt)
        result['image_score'] = aggregated_score
        result['changed_inst'] = len(selected_gt_indices)
        
        del gt_boxes
        del pred_boxes

        return result

    def _fuse_pred_with_gt(self, pred, pred_indices, gt, gt_indices):
        boxes   = self._join_elements_pred_with_gt(pred.pred_boxes.to('cpu'), pred_indices.to('cpu'),
                                          gt['instances'].gt_boxes.to('cpu'), gt_indices.to('cpu'))
        
        classes = self._join_elements_pred_with_gt(pred.pred_classes.to('cpu'), pred_indices.to('cpu'),
                                          gt['instances'].gt_classes.to('cpu'), gt_indices.to('cpu'))

        return Instances(pred.image_size,
                         pred_boxes = boxes,
                         pred_classes = classes,
                         scores = torch.ones_like(classes).float())

    @staticmethod
    def _join_elements_pred_with_gt(pred_ele, pred_indices, gt_ele ,gt_indices):
        if isinstance(pred_ele, Boxes):
            return Boxes.cat([
                        _deselect(pred_ele, pred_indices), 
                        gt_ele[gt_indices]
                    ])
        else:
            return torch.cat([
                        _deselect(pred_ele, pred_indices), 
                        gt_ele[gt_indices]
                    ])

    @staticmethod
    def _postprocess(instances, gt):

        height = gt.get("height")
        width = gt.get("width")
        r = detector_postprocess(instances, height, width)
        return {'instances': r, 'image_id': gt['image_id']}

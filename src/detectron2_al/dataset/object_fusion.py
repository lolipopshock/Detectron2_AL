from typing import List, Tuple, Dict, Any
from functools import partial
import torch
import numpy as np

from detectron2.structures import Boxes, Instances
from detectron2.modeling.postprocessing import detector_postprocess

from ..scoring_utils import *
from ..scheduling_utils import DefaultSchedular

__all__ = ['ObjectFusion', 'ObjectFusionRatioScheduler', 'ObjectSelectionNumberScheuler']

def _quantile(t, q):
    # As we are using pytorch 1.4, there is no native
    # Pytorch support for the quantile function.  
    # This implementation is based on 
    # https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30
    return (t.view(-1)
             .kthvalue(1 + round(float(q) * (t.numel() - 1)))
             .values.item())

def _deselect(t, indices, return_mapping=False):
    """
    Select the elements in t with index not in indices
    """
    selected_indices = [i not in indices for i in range(len(t))]
    if not return_mapping:
        return t[selected_indices]
    else:
        return t[selected_indices], [i for i in range(len(t)) if i not in indices]


class ObjectFusionRatioScheduler(DefaultSchedular):

    def __init__(self, cfg):
        steps = cfg.AL.TRAINING.ROUNDS 
        start = cfg.AL.OBJECT_FUSION.INITIAL_RATIO
        mode  = cfg.AL.OBJECT_FUSION.DECAY
        end   = cfg.AL.OBJECT_FUSION.LAST_RATIO
        super().__init__(start, end, steps, mode)


class ObjectSelectionNumberScheuler(DefaultSchedular):
    def __init__(self, cfg):
        steps = cfg.AL.TRAINING.ROUNDS 
        start = cfg.AL.OBJECT_FUSION.PRESELECTION_RAIO
        mode  = cfg.AL.OBJECT_FUSION.SELECTION_RAIO_DECAY
        end   = cfg.AL.OBJECT_FUSION.ENDSELECTION_RAIO
        super().__init__(start, end, steps, mode)


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
        self.recover_almost_correct_predictions = cfg.AL.OBJECT_FUSION.RECOVER_ALMOST_CORRECT_PRED

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.fusion_ratio = ObjectFusionRatioScheduler(cfg)
        self.selection_ratio = ObjectSelectionNumberScheuler(cfg)

    def _init_overlapping_funcs(self, cfg):
        self.overlapping_metric = self.OVERLAPPING_METRICS[cfg.AL.OBJECT_FUSION.OVERLAPPING_METRIC]
        self.overlapping_th = cfg.AL.OBJECT_FUSION.OVERLAPPING_TH
        self.gt_selector = self.OBJECT_SELECTOR[cfg.AL.OBJECT_FUSION.SELECTION_METHOD]
        if cfg.AL.OBJECT_FUSION.SELECTION_METHOD == 'above':
            self.gt_selector = partial(select_above, threshold=self.overlapping_th)

    def combine(self, pred: Instances, 
                      gt: Dict, 
                      round: int,
                      ave_num_objects_per_image: int = None):
        """
        Combine the model predictions with the ground-truth
        by replacing the objects in the pred with score_al of 
        top replace_ratio. It will automatically move all the boxes
        to self.device, and the output will be saved back to cpu.
        """
        if len(pred) <= 0:
            # There is no prediction from the model
            # Which means this image is very challenging 
            # We will use all ground-truth data, and assgin high scores

            return self._duplicate_gt_as_output(gt)

        if ave_num_objects_per_image is not None:
            top_object_numbers = int(ave_num_objects_per_image*self.selection_ratio[round])
            pred = pred[:top_object_numbers]

        fusion_ratio = self.fusion_ratio[round]
        gt_boxes = gt['instances'].gt_boxes.to(self.device)
        pred_boxes = pred.pred_boxes.to(self.device)

        score_al_th = _quantile(pred.scores_al, 1-fusion_ratio) 
        # If fusion_ratio is 0.8, then we want to find all objects that 
        # have scores more than 0.2(=1-0.8) quantile. 
        selected_pred_indices = torch.where(pred.scores_al > score_al_th)[0].to(self.device)
        
        if len(selected_pred_indices)<= 0:
            # For some rounding issues, we might catch no indices. 
            # We just add all the boxes then. 
            selected_pred_indices = torch.arange(len(pred_boxes)).to(self.device)

        aggregated_score = pred.scores_al[selected_pred_indices].mean().item()

        overlapping_scores = self.overlapping_metric(
                pred_boxes[selected_pred_indices], 
                gt_boxes)

        selected_gt_indices = self.gt_selector(overlapping_scores) 
        selected_gt_indices = list(sum(selected_gt_indices, [])) 
        selected_gt_indices = torch.LongTensor(selected_gt_indices).to(self.device)
        
        # Remove duplicated gt_boxes from the output 
        selected_gt_indices, _cts = torch.unique(selected_gt_indices, return_counts=True)
        gt_indices_ct = {gt_idx.item():_ct.item() for (gt_idx, _ct) in zip(selected_gt_indices, _cts)}

        if self.recover_almost_correct_predictions:
            # Sometimes the model with generate pretty decent predictions. Therefore
            # annotators only need to visually check it without modifying the labels. 
            # Therefore the budget cost is not the "full cost", but a discounted 
            # partialy cost. In this step, we are trying to find these regions. 
            max_scores = overlapping_scores.max(dim=-1)
            to_recover_pred_indices = torch.where(max_scores.values>0.925)[0]
            
            recovered_pred_indices = []
            recovered_gt_indices   = []

            for _idx in to_recover_pred_indices:
                
                gt_idx = max_scores.indices[_idx]
                gt_class = gt['instances'].gt_classes[gt_idx].item()
                pred_idx = selected_pred_indices[_idx]
                pred_class = pred.pred_classes[pred_idx].item()
                if gt_class == pred_class:
                    if gt_indices_ct[gt_idx.item()] != 1: 
                        # If some gt box appears more than once, then 
                        # annotators still need to fix that. Thus we 
                        # don't include this box as the paritaly 
                        # discounted (or recovered) boxes 
                        continue 
                    else:
                        recovered_pred_indices.append(pred_idx)
                        recovered_gt_indices.append(gt_idx)
            
            recovered_pred_indices = torch.LongTensor(recovered_pred_indices).to(self.device)
            recovered_gt_indices = torch.LongTensor(recovered_gt_indices).to(self.device)

        if self.remove_duplicates:
            # Sometimes the model will generate duplicated predictions, and some of the 
            # duplicates won't be selected for matching with the gt. Thus, in this step,
            # we will find and eliminate these boxes using the identified ground-truth 
            # boxes.
            selected_gt = gt_boxes[selected_gt_indices]
            selected_pred, index_mapping = _deselect(pred_boxes, selected_pred_indices, return_mapping=True)

            overlapping_scores = pairwise_overlap_coefficient(selected_pred, selected_gt)
            selected_pred_indices_extra, _ = torch.where(overlapping_scores>self.remove_duplicates_th)
            selected_pred_indices_extra = torch.LongTensor([index_mapping[i] for i in 
                                                            torch.unique(selected_pred_indices_extra)]).to(self.device)
            selected_pred_indices = torch.cat([selected_pred_indices, selected_pred_indices_extra])

        if self.recover_missing_objects:
            # Sometimes the model won't generate predictions for some region. In this step,
            # we will reterive them by calculating the overlapping between the combined
            # boxes with all gt_boxes. If there's a gt_boxes without any combined boxes of 
            # overlapping higher than the threshold (0.05), we add them to the gt boxes list.
            combined_boxes = self._join_elements_pred_with_gt(pred_boxes, 
                                                              selected_pred_indices, 
                                                              gt_boxes, 
                                                              selected_gt_indices)

            if len(combined_boxes)<=0:
                return self._duplicate_gt_as_output(gt)

            max_overlapping_for_gt_boxes = pairwise_iou(combined_boxes, gt_boxes).max(dim=0).values
            missing_gt_boxes_indices = torch.where(max_overlapping_for_gt_boxes<=0.05)[0]
            selected_gt_indices = torch.cat([selected_gt_indices, missing_gt_boxes_indices])

        if self.recover_almost_correct_predictions:
            # During the remove_duplicates and recover_missing_objects process, we use the
            # gt boxes as an delegate for processing. And it's time to switch them back to
            # the pred boxes. This might cause very tiny inaccuray in these process, but as
            # we've set very high matching accuracy (0.925), the inaccuracy should be negligible.
            modified_pred_indices = torch.LongTensor([idx for idx in selected_pred_indices if idx not in recovered_pred_indices])
            modified_gt_indices = torch.LongTensor([idx for idx in selected_gt_indices if idx not in recovered_gt_indices])
            combined_instances = self._fuse_pred_with_gt(pred, modified_pred_indices,
                                                    gt, modified_gt_indices)
            result = self._postprocess(combined_instances, gt)
            result['image_score'] = aggregated_score
            result['labeled_inst_from_gt']   = len(modified_gt_indices)  
            result['dropped_inst_from_pred'] = len(modified_pred_indices)  
            result['recovered_inst']         = len(recovered_pred_indices)

        else:
            combined_instances = self._fuse_pred_with_gt(pred, selected_pred_indices,
                                                    gt, selected_gt_indices)
            if len(combined_instances)<=0:
                return self._duplicate_gt_as_output(gt)

            result = self._postprocess(combined_instances, gt)
            result['image_score'] = aggregated_score
            result['labeled_inst_from_gt']   = len(selected_gt_indices)
            result['dropped_inst_from_pred'] = len(selected_pred_indices)
            result['recovered_inst']         = 0

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

    def _duplicate_gt_as_output(self, gt):

        copied =  Instances(gt['instances'].image_size,
                    pred_boxes = gt['instances'].gt_boxes.to('cpu'),
                    pred_classes = gt['instances'].gt_classes.to('cpu'),
                    scores = torch.ones_like(gt['instances'].gt_classes).float())

        result = self._postprocess(copied, gt)
        result['image_score'] = 1
        result['labeled_inst_from_gt']   = len(gt['instances'])
        result['dropped_inst_from_pred'] = 0
        result['recovered_inst']         = 0
        return result
import layoutparser as lp
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
import random 
from copy import deepcopy
from itertools import product

from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.registry import Registry
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from detectron2.modeling.roi_heads.roi_heads import ROIHeads, StandardROIHeads, ROI_HEADS_REGISTRY
from detectron2.structures.boxes import Boxes
from .utils import *
from ..scoring_utils import elementwise_iou

__all__ = ['ROIHeadsAL']


def calculate_ce_scores(p, q, num_shifts):
    # use crossentropy for calculation diff
    diff = - (p * torch.log(q)).mean(dim=-1)
    # aggregate the statistics for each prediction
    diff = torch.Tensor([scores.mean() for scores in diff.split(num_shifts)]) 

    return diff

def calculate_kl_scores(p, q, num_shifts):
    # use kl divergence for calculation diff
    diff = - (p * torch.log(q/p)).mean(dim=-1) 
    # aggregate the statistics for each prediction
    diff = torch.Tensor([scores.mean() for scores in diff.split(num_shifts)])

    return diff

def calculate_iou_scores(perturbed_box, raw_det, num_shifts, num_bbox_reg_classes):
    reshaped_boxes = perturbed_box.reshape(-1, num_bbox_reg_classes, 4)
    cat_ids = raw_det.pred_classes.repeat_interleave(num_shifts, dim=0)
    perturbed_boxes = Boxes(torch.stack([reshaped_boxes[row_id, cat_id] for row_id, cat_id in enumerate(cat_ids)]))
    raw_boxes = Boxes(raw_det.pred_boxes.tensor.repeat_interleave(num_shifts, dim=0))
    ious = elementwise_iou(raw_boxes, perturbed_boxes)
    iou_scores = torch.Tensor([scores.mean() for scores in ious.split(num_shifts)])
    
    return iou_scores

@ROI_HEADS_REGISTRY.register()
class ROIHeadsAL(StandardROIHeads):

    def __init__(self, cfg, input_shape):
        
        super(ROIHeadsAL, self).__init__(cfg, input_shape)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.cfg = cfg
        self._init_al(cfg)

    def _init_al(self, cfg):
        
        # The scoring objective: max
        # The larger the more problematic the detection is
        if cfg.AL.OBJECT_SCORING == '1vs2':
            self.object_scoring_func = self._one_vs_two_scoring
        elif cfg.AL.OBJECT_SCORING == 'least_confidence':
            self.object_scoring_func = self._least_confidence_scoring
        elif cfg.AL.OBJECT_SCORING == 'random':
            self.object_scoring_func = self._random_scoring
        elif cfg.AL.OBJECT_SCORING == 'perturbation':
            self.object_scoring_func = self._perturbation_scoring
        else:
            raise NotImplementedError
        
        if cfg.AL.IMAGE_SCORE_AGGREGATION == 'avg':
            self.image_score_aggregation_func = torch.mean
        elif cfg.AL.IMAGE_SCORE_AGGREGATION == 'max':
            self.image_score_aggregation_func = torch.max
        elif cfg.AL.IMAGE_SCORE_AGGREGATION == 'sum':
            self.image_score_aggregation_func = torch.sum
        elif cfg.AL.IMAGE_SCORE_AGGREGATION == 'random':
            self.image_score_aggregation_func = lambda x: torch.rand(1)
        else:
            raise NotImplementedError

    def estimate_for_proposals(self, features, proposals):

        with torch.no_grad():
            features = [features[f] for f in self.in_features]
            box_features = self.box_pooler(features,
                                [x if isinstance(x, Boxes) \
                                    else x.proposal_boxes for x in proposals])
            box_features = self.box_head(box_features)
            pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
            del box_features

            outputs = FastRCNNOutputs(
                self.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                self.smooth_l1_beta)
        
        return outputs

    def generate_image_scores(self, features, proposals):

        detected_objects_with_given_scores = \
            self.generate_object_scores(features, proposals)

        image_scores = []

        for ds in detected_objects_with_given_scores:
            if len(ds) == 0:
                image_scores.append(1.)
            else:
                image_scores.append(self.image_score_aggregation_func(ds.scores_al).item())

        return image_scores
    
    def generate_object_scores(self, features, proposals, with_image_scores=False):
        
        outputs = self.estimate_for_proposals(features, proposals)            

        detected_objects_with_given_scores = self.object_scoring_func(outputs, features=features)

        if not with_image_scores:
            return detected_objects_with_given_scores
        else:
            image_scores = []

            for ds in detected_objects_with_given_scores:
                image_scores.append(self.image_score_aggregation_func(ds.scores_al).item())

            return image_scores, detected_objects_with_given_scores

    ########################################
    ### Class specific scoring functions ### 
    ########################################

    def _one_vs_two_scoring(self, outputs, **kwargs):
        """
        Comput the one_vs_two scores for the objects in the fasterrcnn outputs 
        """

        cur_detections, filtered_indices = \
            outputs.inference(self.test_score_thresh, self.test_nms_thresh, 
                              self.test_detections_per_img)

        pred_probs = outputs.predict_probs()
        # The predicted probabilities are a list of size batch_size

        object_scores = [one_vs_two_scoring(prob[idx]) for \
                            (idx, prob) in zip(filtered_indices, pred_probs)]
        
        for cur_detection, object_score in zip(cur_detections, object_scores):
            cur_detection.scores_al = object_score

        return cur_detections

    def _least_confidence_scoring(self, outputs, **kwargs):
        """
        Comput the least_confidence_scoring scores for the objects in the fasterrcnn outputs 
        """

        cur_detections, filtered_indices = \
            outputs.inference(self.test_score_thresh, self.test_nms_thresh, 
                              self.test_detections_per_img)

        for cur_detection in cur_detections:
            cur_detection.scores_al = (1-cur_detection.scores)**2

        return cur_detections

    def _random_scoring(self, outputs, **kwargs):
        """
        Assign random active learning scores for each object 
        """

        cur_detections, filtered_indices = \
            outputs.inference(self.test_score_thresh, self.test_nms_thresh, 
                              self.test_detections_per_img)

        device = cur_detections[0].scores.device
        for cur_detection in cur_detections:
            cur_detection.scores_al = torch.rand(cur_detection.scores.shape).to(device)

        return cur_detections

    def _create_translations(self, cfg):
        
        def _generate_individual_shift_matrix(alpha, beta):
            if cfg.AL.PERTURBATION.RANDOM:
                alpha = random.uniform(0, alpha)
                beta  = random.uniform(0, beta)
            return torch.Tensor([
                    [(1-alpha), 0,       -alpha,    0],
                    [0,        (1-beta), 0,         -beta],
                    [alpha,     0,       (1+alpha), 0],
                    [0,         beta,    0,         (1+beta)],
                ])
        
        alphas = cfg.AL.PERTURBATION.ALPHAS
        betas  = cfg.AL.PERTURBATION.BETAS

        derived_shift = [
            [
                [alpha, beta], 
                [alpha, -beta], 
                [-alpha, beta], 
                [-alpha, -beta]
            ] 
            for alpha, beta in product(alphas, betas)
        ]

        matrices = [_generate_individual_shift_matrix(*params) 
                        for params in sum(derived_shift, [])]
        return len(matrices), torch.stack(matrices, dim=-1).to(self.device)

    def _perturbation_scoring(self, raw_outputs, features):
        
        # Obtain the raw prediction boxes and probabilities
        raw_detections, raw_indices = \
            raw_outputs.inference(self.test_score_thresh, self.test_nms_thresh, 
                            self.test_detections_per_img)
        
        raw_probs = [prob[idx] for (idx, prob) 
                        in zip(raw_indices, raw_outputs.predict_probs())]

        is_empty_raw_detections = [len(det)==0 for det in raw_detections]

        # Generate perturbed boxes 
        num_shifts, shift_matrix = self._create_translations(self.cfg)

        all_new_proposals = []
        for is_empty, det in zip(is_empty_raw_detections, raw_detections):
            
            if is_empty:
                # Create a dummy box for empty detections
                used_boxes = torch.zeros((1,4)).to(self.device)
            else:
                used_boxes = det.pred_boxes.tensor

            new_proposals = Instances(
                det.image_size,
                proposal_boxes=Boxes(
                    torch.einsum('bi,ijc->bjc', 
                            used_boxes, 
                            shift_matrix)
                         .permute(0,2,1)
                         .reshape(-1,4)
                )
            )
            all_new_proposals.append(new_proposals)
        
        perturbed_outputs = self.estimate_for_proposals(features, all_new_proposals)
        perturbed_probs = perturbed_outputs.predict_probs()
        perturbed_boxes = perturbed_outputs.predict_boxes()
        num_bbox_reg_classes = perturbed_boxes[0].shape[1] // 4

        for is_empty, raw_det, raw_prob, perturbed_prob, perturbed_box in \
             zip(is_empty_raw_detections, raw_detections, raw_probs, perturbed_probs, perturbed_boxes):
            
            if is_empty:
                raw_det.scores_al = raw_det.scores
            else:
                p = raw_prob.repeat_interleave(num_shifts, dim=0)
                q = perturbed_prob
                
                if self.cfg.AL.PERTURBATION.VERSION == 1:
                    diff = calculate_ce_scores(p, q, num_shifts)

                elif self.cfg.AL.PERTURBATION.VERSION == 2:
                    diff = calculate_kl_scores(p, q, num_shifts)

                elif self.cfg.AL.PERTURBATION.VERSION == 3:
                    diff = calculate_iou_scores(perturbed_box, raw_det, num_shifts, num_bbox_reg_classes)

                elif self.cfg.AL.PERTURBATION.VERSION == 4:
                    diff1 = calculate_iou_scores(perturbed_box, raw_det, num_shifts, num_bbox_reg_classes)
                    diff2 = calculate_ce_scores(p, q, num_shifts)
                    diff = diff1 + diff2 * self.cfg.AL.PERTURBATION.LAMBDA

                elif self.cfg.AL.PERTURBATION.VERSION == 5:
                    diff1 = calculate_iou_scores(perturbed_box, raw_det, num_shifts, num_bbox_reg_classes)
                    diff2 = calculate_kl_scores(p, q, num_shifts)
                    diff = diff1 + diff2*3 * self.cfg.AL.PERTURBATION.LAMBDA

                raw_det.scores_al = diff
        
        return raw_detections
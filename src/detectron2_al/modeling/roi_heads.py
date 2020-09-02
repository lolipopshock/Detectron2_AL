import layoutparser as lp
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from copy import deepcopy
from itertools import product

from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.registry import Registry
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from detectron2.modeling.roi_heads.roi_heads import ROIHeads, StandardROIHeads, ROI_HEADS_REGISTRY
from detectron2.structures.boxes import Boxes
from .utils import *

__all__ = ['ROIHeadsAL']


@ROI_HEADS_REGISTRY.register()
class ROIHeadsAL(StandardROIHeads):

    def __init__(self, cfg, input_shape):
        
        super(ROIHeadsAL, self).__init__(cfg, input_shape)
        self._init_al(cfg)

    def _init_al(self, cfg):
        
        # The scoring objective: max
        # The larger the more problematic the detection is
        if cfg.AL.OBJECT_SCORING == '1vs2':
            self.object_scoring_func = self._one_vs_two_scoring
        elif cfg.AL.OBJECT_SCORING == 'least_confidence':
            self.object_scoring_func = self._least_confidence_scoring
        else:
            raise NotImplementedError
        
        if cfg.AL.IMAGE_SCORE_AGGREGATION == 'avg':
            self.image_score_aggregation_func = torch.mean
        elif cfg.AL.IMAGE_SCORE_AGGREGATION == 'max':
            self.image_score_aggregation_func = torch.max
        elif cfg.AL.IMAGE_SCORE_AGGREGATION == 'sum':
            self.image_score_aggregation_func = torch.sum
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

        detected_objects_with_given_scores = self.object_scoring_func(outputs)
        
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

    def _one_vs_two_scoring(self, outputs):
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

    def _least_confidence_scoring(self, outputs):
        """
        Comput the least_confidence_scoring scores for the objects in the fasterrcnn outputs 
        """

        cur_detections, filtered_indices = \
            outputs.inference(self.test_score_thresh, self.test_nms_thresh, 
                              self.test_detections_per_img)

        for cur_detection in cur_detections:
            cur_detection.scores_al = (1-cur_detection.scores)**2

        return cur_detections

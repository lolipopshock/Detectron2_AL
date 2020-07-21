import layoutparser as lp
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from copy import deepcopy

from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.registry import Registry
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from detectron2.modeling.roi_heads.roi_heads import ROIHeads, StandardROIHeads, ROI_HEADS_REGISTRY

from collections import namedtuple

JitterStatistics = namedtuple("JitterStatistics", ("box", "score", "cls"))

def transpose_nested_list(l):
    l = [[row[col] for row in l] for col in range(len(l[0]))]
    return l

def cal_iou(b1, b2):
    
    x_left   = max(b1.block.x_1, b2.block.x_1)
    y_top    = max(b1.block.y_1, b2.block.y_1)
    x_right  = min(b1.block.x_2, b2.block.x_2)
    y_bottom = min(b1.block.y_2, b2.block.y_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0 
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    b1_area = b1.height * b1.width
    b2_area = b2.height * b2.width
    
    return intersection_area / float(b1_area + b2_area - intersection_area)

def cvt_instance_to_textblock(cur_detection):
    pred_boxes = [lp.TextBlock(
                    block = lp.Rectangle(*cur_det.pred_boxes.tensor.to('cpu')[0].squeeze().detach().numpy()),
                    score = cur_det.scores.to('cpu')[0].item(),
                    type  = cur_det.pred_classes.to('cpu')[0].item()) 
                        if len(cur_det) > 0 else None
                            for cur_det in cur_detection ]
    return pred_boxes

def jitter_proposals(cur_proposals, jitter):
    
    new_proposals = deepcopy(cur_proposals)
    for idx in range(len(cur_proposals)):
        new_proposals[idx].proposal_boxes.tensor = \
            cur_proposals[idx].proposal_boxes.tensor + torch.Tensor([jitter]).to('cuda') 
    
    return new_proposals

def compute_jitter_statistics(origs, all_news):
    results = []
    for orig, news in zip(origs, all_news):
        box_diff  = torch.mean(torch.cat([pairwise_iou(orig.pred_boxes, new.pred_boxes) for new in news]))
        score_diff= torch.mean(torch.cat([orig.scores - new.scores for new in news]))
        cls_diff  = torch.mean(torch.cat([orig.pred_classes == new.pred_classes for new in news]).float())
        results.append(JitterStatistics(box_diff, score_diff, cls_diff))
    return results 

def compute_jitter_statistics_cpu(origs, all_news):
    
    results = []
    for orig, news in zip(cvt_instance_to_textblock(origs), all_news):
        if orig is not None:
            news = cvt_instance_to_textblock(news)        
            box_diff  = np.mean([cal_iou(orig, new) for new in news])
            score_diff= np.mean(np.abs([orig.score - new.score for new in news]))
            cls_diff  = np.mean([orig.type == new.type for new in news])
        else:
            box_diff, score_diff, cls_diff = None, None, None
        results.append(JitterStatistics(box_diff, score_diff, cls_diff))
    return results


@ROI_HEADS_REGISTRY.register()
class ALROIHeads(StandardROIHeads):

    def __init__(self, cfg, input_shape):
        super(ALROIHeads, self).__init__(cfg, input_shape)
        self.jitters = [
                [5, 0, 5, 0],
                [0, 5, 0, 5],
                [-5, 0, -5, 0],
                [0, -5, 0, -5],
                [-5, 0, 5, 0],
                [0, 5, 0, -5],
                [5, 0, -5, 0],
                [0, -5, 0, 5]]

    def estimate_for_proposals(self, features, proposals):
        features = [features[f] for f in self.in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
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

    def probe_proposals(self, features, rpn_proposals, orig_predictions): 
    
        all_jitter_statistics = []

        with torch.no_grad():
        
            for i in range(self.test_detections_per_img):

                cur_proposals = [rpn_proposal[i:i+1] for rpn_proposal in rpn_proposals]
                cur_predictions = [orig_prediction[i:i+1] for orig_prediction in orig_predictions]

                all_jitter_predictions = []
                for jitter_vector in self.jitters: 
                    new_proposals = jitter_proposals(cur_proposals, jitter_vector)
                    jittered_outputs = self.estimate_for_proposals(features, new_proposals)
                    jitter_predictions, _ = jittered_outputs.inference(score_thresh=0, nms_thresh=0, topk_per_image=1)
                    all_jitter_predictions.append(jitter_predictions)

                all_jitter_predictions = transpose_nested_list(all_jitter_predictions) # N'x4 -> 4xN'
                cur_jitter_statistics = compute_jitter_statistics_cpu(cur_predictions, all_jitter_predictions) # 1x4
                
                all_jitter_statistics.append(cur_jitter_statistics) # Nx4

        return transpose_nested_list(all_jitter_statistics) #4xN

    def _forward_al(
        self, 
        features: Dict[str, torch.Tensor], 
        proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:

        with torch.no_grad():
            outputs = self.estimate_for_proposals(features, proposals)
            cur_detection, filtered_indices = outputs.inference(
                    self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img)

            selected_proposals = [proposal[idx.to('cpu')] for proposal, idx in zip(proposals, filtered_indices)]
            detection_stats = self.probe_proposals(features, selected_proposals, cur_detection)

        return cur_detection, detection_stats
    
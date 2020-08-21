from detectron2.config.defaults import _C
from detectron2.config.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Additional Configuration for Active Learning
# ---------------------------------------------------------------------------- #

_C.AL = CN()
_C.AL.MODE = 'object' # {'image', 'object'}
# Perform active learning on whether image-level or object-level 
_C.AL.OBJECT_SCORING = '1vs2' # {'1vs2, 'least_confidence', 'jitter'}
# The method to compute the individual object scores  
_C.AL.IMAGE_SCORE_AGGREGATION = 'avg' # {'avg', 'max', 'sum'}
# The method to aggregate the individual object scores to the whole image score 

_C.AL.DATASET = CN()
# Specifies the configs for creating new datasets 
# It will also combines configs from DATASETS and DATALOADER 
# when creating the DynamicDataset for training.
_C.AL.DATASET.BUDGET_STYLE = 'object'
_C.AL.DATASET.IMAGE_BUDGET = 20
_C.AL.DATASET.OBJECT_BUDGET = 2000
# Specifies the way to calculate the budget 
# If specify the BUDGET_STYLE as image, while using the object-level
# Active Learning, we will convert the image_budget to object budget 
# by OBJECT_BUDGET = IMAGE_BUDGET * AVG_OBJ_IN_TRAINING. 
# Similarly, we have 
# IMAGE_BUDGET = OBJECT_BUDGET // AVG_OBJ_IN_TRAINING.
_C.AL.DATASET.SAMPLE_METHOD = 'top' # {'top', 'kmeans'}
# The method to sample images when labeling 

_C.AL.OBJECT_FUSION = CN()
# Specifies the configs to fuse model prediction and ground-truth (gt)
_C.AL.OBJECT_FUSION.OVERLAPPING_METRIC = 'iou' # {'iou', 'dice_coefficient', 'overlap_coefficient'}
# The function to calculate the overlapping between model pred and gt
_C.AL.OBJECT_FUSION.OVERLAPPING_TH = 0.25
# The threshold for selecting the boxes 
_C.AL.OBJECT_FUSION.SELECTION_METHOD = 'iou' # {'top', 'above', 'nonzero'}
# For gt boxes with non-zero overlapping with the pred box, specify the 
# way to choose the gt boxes. 
# top: choose the one with the highest overlapping
# above: choose the ones has the overlapping above the threshold specified above
# nonzero: choose the gt boxes with non-zero overlapping
_C.AL.OBJECT_FUSION.REMOVE_DUPLICATES = True
_C.AL.OBJECT_FUSION.REMOVE_DUPLICATEs_TH = 0.25
# For the fused dataset, remove duplicated boxes with overlapping more than 
# the given threshold
_C.AL.OBJECT_FUSION.RECOVER_MISSING = True
# If true, we recover the mis-identified objects during the process

_C.AL.TRAINING = CN()
_C.AL.TRAINING.NUM_EPOCHS = 5
_C.AL.TRAINING.UPDATE_FREQ = 3
_C.AL.TRAINING.MERGING_RATIO_INIT = 0.95
_C.AL.TRAINING.MERGING_RATIO_DECAY = 'exp'

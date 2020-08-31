"""
The script is based on https://github.com/facebookresearch/detectron2/blob/master/tools/train_net.py. 
"""

import logging
import os
import json
from collections import OrderedDict
import torch
import sys 
import pandas as pd 

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results

sys.path.append('../src')
from detectron2_al.configs import get_cfg
from detectron2_al.engine import build_al_trainer
from detectron2_al.modeling import *

def setup(args):
    """
    Create configs and perform basic setups.
    """
    
    # Add the val dataset to detectron2 directory
    dataset_name = args.dataset_name
    register_coco_instances(f"{dataset_name}-val",   {}, 
                            args.json_annotation_val,   
                            args.image_path_val)

    # Initialize the configurations
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # Ensure it uses appropriate names and architecture  
    cfg.MODEL.ROI_HEADS.NAME = 'ROIHeadsAL'
    cfg.MODEL.META_ARCHITECTURE = 'ActiveLearningRCNN'

    # Extra configurations for the active learning model
    # for better object selection  
    # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.25
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0

    # Initializethe dataset information
    cfg.AL.DATASET.NAME = args.dataset_name
    cfg.AL.DATASET.IMG_ROOT = args.image_path_train
    cfg.AL.DATASET.ANNO_PATH = args.json_annotation_train 
    cfg.DATASETS.TEST = (f"{args.dataset_name}-val",)

    with open(args.json_annotation_train, 'r') as fp:
        anno_file = json.load(fp)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(anno_file["categories"])
    del anno_file

    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    trainer = build_al_trainer(cfg)

    # if args.eval_only:
    #     model = Trainer.build_model(cfg)
    #     DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #         cfg.MODEL.WEIGHTS, resume=args.resume
    #     )
    #     res = Trainer.test(cfg, model)
    
    #     if cfg.TEST.AUG.ENABLED:
    #         res.update(Trainer.test_with_TTA(cfg, model))
    #     if comm.is_main_process():
    #         verify_results(cfg, res)

    #     # Save the evaluation results
    #     pd.DataFrame(res).to_csv(f'{cfg.OUTPUT_DIR}/eval.csv')
    #     return res

    return trainer.train_al()

if __name__ == "__main__":
    parser = default_argument_parser()

    # Extra Configurations for dataset names and paths
    parser.add_argument("--dataset_name",          default="", help="The Dataset Name")
    parser.add_argument("--json_annotation_train", default="", metavar="FILE", help="The path to the training set JSON annotation")
    parser.add_argument("--image_path_train",      default="", metavar="FILE", help="The path to the training set image folder")
    parser.add_argument("--json_annotation_val",   default="", metavar="FILE", help="The path to the validation set JSON annotation")
    parser.add_argument("--image_path_val",        default="", metavar="FILE", help="The path to the validation set image folder")

    args = parser.parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

import json
import random
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from dataclasses import dataclass, field

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.build import build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog
from detectron2.evaluation.coco_evaluation import instances_to_coco_json

from .object_fusion import ObjectFusion
from .dataset_mapper import DatasetMapperAL
from .utils import build_detection_train_loader_drop_ids
from ..scheduling_utils import *

__all__ =  ['build_al_dataset',
            'HandyCOCO',
            'Budget',
            'DatasetHistory',
            'EpochsPerRound',
            'ActiveLearningDataset',
            'ImageActiveLearningDataset',
            'ObjectActiveLearningDataset']


def _write_json(data, filename):
    
    with open(filename, 'w') as fp:
        json.dump(data, fp)

def _calculate_iterations(num_imgs, batch_size, num_epochs):
    return int(num_imgs*num_epochs/batch_size)

def build_al_dataset(cfg):

    if cfg.AL.MODE == 'image':
        return ImageActiveLearningDataset(cfg)
    elif cfg.AL.MODE == 'object':
        return ObjectActiveLearningDataset(cfg)
    else:
        raise ValueError(f'Unknown active learning mode {cfg.AL.MODE}')

class HandyCOCO(COCO):
    
    def subsample_with_image_ids(self, imgIds):
        
        imgs = [self.imgs[idx] for idx in imgIds]
        anns = self.loadAnns(ids=self.getAnnIds(imgIds=imgIds))
        
        dataset = {
            'images': imgs,
            'annotations': anns,
            'categories': self.dataset['categories'],
            'info': self.dataset.get('info', {}),
            'license': self.dataset.get('licenses', {})
        }
        
        return dataset 
    
    def subsample_and_save_with_image_ids(self, imgIds, filename):
        
        dataset = self.subsample_with_image_ids(imgIds)
        _write_json(dataset, filename)
        
        return dataset

    def avg_object_per_image(self):
        return np.mean([len(self.getAnnIds([img])) for img in self.imgs])


class Budget(object):
    
    def __init__(self, cfg, avg_object_per_image):
        """
        A handy class for controlling budgets
        """
        
        self.style = cfg.AL.MODE
        assert self.style in ['object', 'image']

        self.allocation_method = cfg.AL.DATASET.BUDGET_ALLOCATION
        self.total_rounds = cfg.AL.TRAINING.ROUNDS
        self.eta = cfg.AL.OBJECT_FUSION.BUDGET_ETA

        self.avg_object_per_image = avg_object_per_image
        if self.style == 'object':
            self._initial = self._remaining = cfg.AL.DATASET.IMAGE_BUDGET*avg_object_per_image
        elif self.style == 'image':
            self._initial = self._remaining = cfg.AL.DATASET.IMAGE_BUDGET

        if self.allocation_method == 'linear':
            self._allocations = np.ones(self.total_rounds)*round(self.initial / self.total_rounds)
        else:
            raise NotImplementedError 
        self._allocations = self._allocations.astype('int')

    @property
    def remaining_object_budget(self):
        if self.style == 'image':
            return self.remaining * self.avg_object_per_image
        else:
            return self.remaining

    @property
    def remaining_image_budget(self):
        if self.style == 'image':
            return self.remaining 
        else:
            return int(self.remaining // self.avg_object_per_image)

    @property
    def remaining(self):
        """
        The remaining budget 
        """
        return self._remaining

    @property
    def initial(self):
        """
        The initial budget
        """
        return self._initial

    def all_allocations(self, _as=None):

        if _as is not None and _as != self.style:
            if _as == 'object':
                # originally image
                return self._allocations * self.avg_object_per_image
            elif _as == 'image':
                # originally object
                return self._allocations // self.avg_object_per_image
        else:
            return self._allocations 

    def allocate(self, _round, _as=None):

        allocated = self._allocations[_round]
        
        self._remaining -= allocated
        if _as is not None and _as != self.style:
            if _as == 'object':
                # originally image
                allocated = int(allocated * self.avg_object_per_image)
            elif _as == 'image':
                # originally object
                allocated = int(round(allocated / self.avg_object_per_image))
        return allocated


@dataclass
class DatasetInfo:
    name: str
    json_path: str
    num_images: int
    num_objects: int
    image_ids: List
    anno_details: List = field(default_factory=list)
    training_iter: int = 0

class DatasetHistory(list):

    @property
    def all_dataset_names(self):
        return [ele.name for ele in self]

    def save(self, filename):
        _write_json([vars(ele) for ele in self], filename)


class EpochsPerRound(IntegerSchedular):

    def __init__(self, cfg):

        steps = cfg.AL.TRAINING.ROUNDS 
        start = cfg.AL.TRAINING.EPOCHS_PER_ROUND_INITIAL
        mode  = cfg.AL.TRAINING.EPOCHS_PER_ROUND_DECAY
        end   = cfg.AL.TRAINING.EPOCHS_PER_ROUND_LAST
        super().__init__(start, end, steps, mode)


class ActiveLearningDataset:

    def __init__(self, cfg):
        
        # Dataset configurations
        self.name = cfg.AL.DATASET.NAME
        self.image_root = cfg.AL.DATASET.IMG_ROOT
        self.anno_path = cfg.AL.DATASET.ANNO_PATH
        self.cache_dir = os.path.join(cfg.OUTPUT_DIR, cfg.AL.DATASET.CACHE_DIR)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.coco = HandyCOCO(self.anno_path)
        self.name_prefix = cfg.AL.DATASET.NAME_PREFIX

        # Add the base version of the dataset to the catalog
        self.add_new_dataset_into_catalogs(self.name, self.anno_path)

        # Scheduling
        self.total_rounds = cfg.AL.TRAINING.ROUNDS
        self.budget = Budget(cfg, self.coco.avg_object_per_image())
        self.epochs_per_round = EpochsPerRound(cfg)

        # Sampling method during AL
        self.sampling_method = cfg.AL.DATASET.SAMPLE_METHOD

        # Internal Storage
        self._history = DatasetHistory()
        self._round = -1
        self._cfg = cfg.clone()
        self._cfg.freeze()
        
    def calculate_iterations_for_cur_datasets(self):
        return _calculate_iterations(
            num_imgs   = sum([info.num_images for info in self._history]),
            batch_size = self._cfg.SOLVER.IMS_PER_BATCH,
            num_epochs = self.epochs_per_round[self._round]
        )

    def calculate_estimated_total_iterations(self):
        bs = self._cfg.SOLVER.IMS_PER_BATCH
        num_imgs = self.budget.all_allocations(_as='image')
        num_epochs = [self.epochs_per_round[i] for i in range(self.total_rounds)]
        return sum(
            [
                _calculate_iterations(num_img, bs, num_epoch) \
                    for (num_img, num_epoch) in zip(num_imgs, num_epochs)
            ]
        )

    def dataset_name_at(self, round):
        return f'{self.name}-{self.name_prefix}{round:02d}'

    def dataset_jsonpath_at(self, round):
        return os.path.join(
            self.cache_dir, self.dataset_name_at(round) + '.json'
        )

    @property
    def cur_dataset_name(self):
        return self.dataset_name_at(self._round)

    @property
    def cur_dataset_jsonpath(self):
        return self.dataset_jsonpath_at(self._round)

    def add_new_dataset_into_catalogs(self, name, json_path):
        
        if name not in DatasetCatalog.list():

            register_coco_instances(
                name = name,
                metadata = {}, 
                json_file = os.path.abspath(json_path),
                image_root = self.image_root
            )

    def get_training_dataloader(self, dataset_name=None):
        """
        Get the dataloader with the given datasetname.
        By default it will create the dataset using the last 
        created dataset. 
        """

        if dataset_name is None:
            dataset_name = self._history[-1].name
            max_iter = self.calculate_iterations_for_cur_datasets()
        else:
            assert dataset_name in self._history.all_dataset_names
            max_iter = self._cfg.SOLVER.MAX_ITER / self.total_rounds

        self._history[-1].training_iter = max_iter

        cfg = self._cfg.clone()
        cfg.defrost()
        cfg.DATASETS.TRAIN = tuple(self._history.all_dataset_names)
        dataloader = build_detection_train_loader(cfg)

        return dataloader, max_iter

    def get_oracle_dataloader(self, drop_existing=True):
        """
        Get the dataloader with all the images. Only used 
        for evaluating all the image/object scores and create 
        new datasets. 
        """
        cfg = self._cfg.clone()
        cfg.defrost()
        cfg.SOLVER.IMS_PER_BATCH = 1 # Avoid Dropped Images 
        cfg.DATASETS.TRAIN = (self.name,)

        if not drop_existing:
            dropped_image_ids = []
        else:
            dropped_image_ids = sum([info.image_ids for info in self._history], [])
        
        dataloader = build_detection_train_loader_drop_ids(cfg, dropped_image_ids, 
                                                            DatasetMapperAL(cfg, True))
        
        total_image_num = len(self.coco.imgs)-len(dropped_image_ids)
        max_iter = _calculate_iterations(total_image_num, cfg.SOLVER.IMS_PER_BATCH, 1)
        
        return dataloader, max_iter
        
    def create_dataset_with_image_ids(self, image_ids):

        cur_json_path = self.cur_dataset_jsonpath
        cur_data_name = self.cur_dataset_name
        
        dataset = self.coco.subsample_and_save_with_image_ids( 
            image_ids,
            cur_json_path
        )
        self.add_new_dataset_into_catalogs(cur_data_name, cur_json_path)

        self._history.append(
            DatasetInfo(
                name = cur_data_name,
                json_path = cur_json_path,
                num_images = len(dataset['images']),
                num_objects = len(dataset['annotations']), 
                image_ids = image_ids
            )
        )

    def create_dataset_with_annotations(self, annotations, image_ids, 
                                              labeling_history, num_objects=None):
        
        cur_json_path = self.cur_dataset_jsonpath
        cur_data_name = self.cur_dataset_name
        
        # Save an original dataset for further reference 
        dataset = self.coco.subsample_and_save_with_image_ids( 
            image_ids,
            cur_json_path.replace('.json', '-orig.json')
        )

        # Modify the annotations, save, and register
        dataset['annotations'] = self.coco.loadRes(annotations).dataset['annotations']
        # To ensure the annotations is compatible with the coco format
        _write_json(dataset, cur_json_path)
        self.add_new_dataset_into_catalogs(cur_data_name, cur_json_path)
        
        self._history.append(
            DatasetInfo(
                name = cur_data_name,
                json_path = cur_json_path,
                num_images = len(dataset['images']),
                num_objects = len(dataset['annotations']) if num_objects is None else num_objects,
                anno_details = labeling_history,
                image_ids = image_ids
            )
        )

    def create_initial_dataset(self):
        """
        Create the initial AL dataset.
        The implementation is depend on the active learning schema.
        """

        self._round += 1
        allocated_budget = self.budget.allocate(self._round, _as='image')
        selected_image_ids = random.sample(list(self.coco.imgs.keys()), allocated_budget)
        self.create_dataset_with_image_ids(selected_image_ids)

    def create_new_dataset(self):
        """
        Create a new dataset based on the round and the given budget.
        The implementation is depend on the active learning schema.
        """
        assert self._round <= self.total_rounds
        self._round += 1

    def save_history(self):
        self._history.save(os.path.join(self.cache_dir, 'labeling_history.json'))

class ImageActiveLearningDataset(ActiveLearningDataset):

    def create_new_dataset(self, image_scores:List[Tuple]):
        """
        args:
            images_scores:
                a list of tuples (image_score, image_id)
        """

        super().create_new_dataset()

        allocated_budget = self.budget.allocate(self._round, _as='object')
        if self.sampling_method == 'top':
            top_image_scores = sorted(image_scores)
            selected_image_ids = []

            while allocated_budget>0:
                score, idx = top_image_scores.pop()
                selected_image_ids.append(idx)

                num_objects = len(self.coco.getAnnIds([idx]))
                allocated_budget -= num_objects
        else:
            raise NotImplementedError
        
        self.create_dataset_with_image_ids(selected_image_ids)

class ObjectActiveLearningDataset(ActiveLearningDataset):

    def create_new_dataset(self, fused_results: List[Dict]):
        """
        Args:
            fused_results:
                the predictions results on the oracle dataset with 
        """
        super().create_new_dataset()

        image_scores = [fs['image_score'] for fs in fused_results]

        selected_image_ids = []
        selected_annotations = []
        allocated_budget = self.budget.allocate(self._round)
        
        used_budget = 0
        labeling_history = []
        if self.sampling_method == 'top':
            sorted_image_scores = np.argsort(image_scores).tolist()

            while allocated_budget>used_budget and sorted_image_scores!=[]:

                idx = sorted_image_scores.pop()
                image_id = fused_results[idx]['image_id']
                instances = fused_results[idx]['instances']
                annotations = instances_to_coco_json(instances, image_id)
                selected_image_ids.append(image_id)
                selected_annotations.extend(annotations)
                # Currently, there will be an 'score' field in each of the
                # annotations, and it will be saved in the JSON. The existence
                # of this field won't affect the coco loading, and will make 
                # it easier to compute the score.
                cur_cost =  fused_results[idx]['labeled_inst_from_gt'] + \
                            self.budget.eta * fused_results[idx]['recovered_inst']
                used_budget += round(cur_cost)
                
                labeling_history.append({
                    "image_id":            fused_results[idx]['image_id'],
                    "labeled_inst_from_gt":fused_results[idx]['labeled_inst_from_gt'],
                    "used_inst_from_pred": fused_results[idx]['dropped_inst_from_pred'],
                    "recovered_inst":      fused_results[idx]['recovered_inst']
                })
        else:
            raise NotImplementedError

        self.create_dataset_with_annotations(selected_annotations, 
                                             selected_image_ids, 
                                             labeling_history,
                                             num_objects=round(used_budget))
        dataset_eval = self.evaluate_merged_dataset(self._round)
        pd.Series(dataset_eval).to_csv(self.cur_dataset_jsonpath.replace('.json', 'eval.csv'))

    def evaluate_merged_dataset(self, round, iou_type='bbox'):
        
        dt_json_path = self.dataset_jsonpath_at(round)

        coco_dt = COCO(dt_json_path)
        coco_gt = COCO(dt_json_path.replace('.json', '-orig.json'))
        coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
        class_names = [val['name'] for _, val in coco_gt.cats.items()]

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # A simplified version of COCOEvaluator._derive_coco_results 
        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results
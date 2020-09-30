import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import numpy as np
import requests
import io
import hashlib
import urllib
import cv2

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from label_studio.ml import LabelStudioMLBase
from label_studio.ml.utils import get_single_tag_keys, get_choice, is_skipped


import sys
sys.path.append('./src')
from detectron2_al.configs import get_cfg
from detectron2_al.engine.al_engine import ActiveLearningPredictor
from detectron2_al.modeling import *
import layoutparser as lp
from fvcore.common.file_io import PathManager

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


image_cache_dir = os.path.join(os.path.dirname(__file__), 'image-cache')
os.makedirs(image_cache_dir, exist_ok=True)


def load_image_from_url(url):
    # is_local_file = url.startswith('http://localhost:') and '/data/' in url
    is_local_file = True
    if is_local_file:
        filename, dir_path = url.split('/data/')[1].split('?d=')
        dir_path = str(urllib.parse.unquote_plus(dir_path))
        filepath = os.path.join(dir_path, filename)
        return cv2.imread(filepath)
    else:
        cached_file = os.path.join(image_cache_dir, hashlib.md5(url.encode()).hexdigest())
        if os.path.exists(cached_file):
            with open(cached_file, mode='rb') as f:
                image = Image.open(f).convert('RGB')
        else:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with io.BytesIO(r.content) as f:
                image = Image.open(f).convert('RGB')
            with io.open(cached_file, mode='wb') as fout:
                fout.write(r.content)
        return image_transforms(image)

def convert_block_to_value(block, image_height, image_width):


    block.block.x_1 = max(0, block.block.x_1)
    block.block.x_2 = min(block.block.x_2, image_width)
    block.block.y_1 = max(0, block.block.y_1)
    block.block.y_2 = min(block.block.y_2, image_height)

    return  {
            "height": block.height / image_height*100,
            "rectanglelabels": [str(block.type)],
            "rotation": 0,
            "width":  block.width / image_width*100,
            "x":      block.coordinates[0] / image_width*100,
            "y":      block.coordinates[1] / image_height*100,
            "score":  block.score_al*100
        }


class Detectron2LayoutModel():

    def __init__(self, config_path,
                       model_path = None,
                       label_map  = None,
                       extra_config= []):

        cfg = get_cfg()
        config_path = PathManager.get_local_path(config_path)
        cfg.merge_from_file(config_path)
        cfg.merge_from_list(extra_config)
        
        cfg.MODEL.ROI_HEADS.NAME = 'ROIHeadsAL'
        cfg.MODEL.META_ARCHITECTURE = 'ActiveLearningRCNN'

        if model_path is not None:
            cfg.MODEL.WEIGHTS = model_path            
        cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.cfg = cfg
        self.label_map = label_map
        self._create_model()
    
    def _create_model(self):
        self.model = ActiveLearningPredictor(self.cfg)

    def gather_output(self, outputs):

        instance_pred = outputs['instances'].to("cpu")

        layout = lp.Layout()
        scores = instance_pred.scores.tolist()
        boxes  = instance_pred.pred_boxes.tensor.tolist()
        labels = instance_pred.pred_classes.tolist()

        for score, box, label in zip(scores, boxes, labels):
            x_1, y_1, x_2, y_2 = box

            if self.label_map is not None:
                label = self.label_map.get(label, label)

            cur_block = lp.TextBlock(
                    lp.Rectangle(x_1, y_1, x_2, y_2),
                    type=label, 
                    score=score)
            layout.append(cur_block)

        return layout

    def gather_output_with_stats(self, outputs):

        instance_pred = outputs['instances'].to("cpu")

        layout = lp.Layout()
        scores = instance_pred.scores.tolist()
        scores_al = instance_pred.scores_al.tolist()
        boxes  = instance_pred.pred_boxes.tensor.tolist()
        labels = instance_pred.pred_classes.tolist()

        for score, box, label, score_al in zip(scores, boxes, labels, scores_al):
            x_1, y_1, x_2, y_2 = box

            if self.label_map is not None:
                label = self.label_map.get(label, label)

            cur_block = lp.TextBlock(
                    lp.Rectangle(x_1, y_1, x_2, y_2),
                    type=label, 
                    score=score)
            cur_block.score_al = score_al
            layout.append(cur_block)

        return layout

    def detect(self, image):
        outputs, _ = self.model(image)
        layout  = self.gather_output(outputs)
        return layout

    def detect_al(self, image):
        pred = self.model(image)
        layout  = self.gather_output_with_stats(pred)
        return layout

class ObjectDetectionAPI(LabelStudioMLBase):

    def __init__(self, freeze_extractor=False, **kwargs):

        super(ObjectDetectionAPI, self).__init__(**kwargs)
        
        self.from_name, self.to_name, self.value, self.classes =\
            get_single_tag_keys(self.parsed_label_config, 'RectangleLabels', 'Image')
        self.freeze_extractor = freeze_extractor
        
        self.model = Detectron2LayoutModel(
                config_path="https://www.dropbox.com/s/ta4777i1g1jjj18/config.yml?dl=1",
                model_path ="https://www.dropbox.com/s/f261qar6f75b9c0/model_final.pth?dl=1",
                label_map={1: "title", 2: "address", 3: "text", 4:"number"},
                extra_config=["TEST.DETECTIONS_PER_IMAGE", 150, 
                            "MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5,
                            "MODEL.ROI_HEADS.NMS_THRESH_TEST", 0.75]
            )
            
    def reset_model(self):
        ## self.model = ImageClassifier(len(self.classes), self.freeze_extractor)
        pass

    def predict(self, tasks, **kwargs):

        image_urls = [task['data'][self.value] for task in tasks]
        images = [load_image_from_url(url) for url in image_urls]
        layouts = [self.model.detect_al(image) for image in images]  

        predictions = []
        for image, layout in zip(images, layouts):
            height, width = image.shape[:2]

            result = [
                {
                'from_name': self.from_name,
                'to_name': self.to_name,
                "original_height": height,
                "original_width": width,
                "source": "$image",
                'type': 'rectanglelabels',
                "value": convert_block_to_value(block, height, width)
                } for block in layout
            ]

            predictions.append({'result': result})

        return predictions

    def fit(self, completions, workdir=None, 
            batch_size=32, num_epochs=10, **kwargs):
        image_urls, image_classes = [], []
        print('Collecting completions...')
        # for completion in completions:
        #     if is_skipped(completion):
        #         continue
        #     image_urls.append(completion['data'][self.value])
        #     image_classes.append(get_choice(completion))

        print('Creating dataset...')
        # dataset = ImageClassifierDataset(image_urls, image_classes)
        # dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

        print('Train model...')
        # self.reset_model()
        # self.model.train(dataloader, num_epochs=num_epochs)

        print('Save model...')
        # model_path = os.path.join(workdir, 'model.pt')
        # self.model.save(model_path)

        return {'model_path': None, 'classes': None}
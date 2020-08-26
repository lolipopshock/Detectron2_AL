#!/bin/bash

cd ../tools

python train_al_model.py \
    --dataset_name          tkdata-v7 \
    --json_annotation_train ../data/tk1957-v7/train/annotations.json \
    --image_path_train      ../data/tk1957-v7/train/ \
    --json_annotation_val   ../data/tk1957-v7/val/annotations.json \
    --image_path_val        ../data/tk1957-v7/val/ \
    --config-file           ../configs/tk/faster_rcnn_X_101_32x8d_FPN_3x.yaml \
    OUTPUT_DIR  ../outputs/tk/object/faster_rcnn_X_101_32x8d_FPN_3x \
    AL.MODE object 

python train_al_model.py \
    --dataset_name          tkdata-v7 \
    --json_annotation_train ../data/tk1957-v7/train/annotations.json \
    --image_path_train      ../data/tk1957-v7/train/ \
    --json_annotation_val   ../data/tk1957-v7/val/annotations.json \
    --image_path_val        ../data/tk1957-v7/val/ \
    --config-file           ../configs/tk/faster_rcnn_X_101_32x8d_FPN_3x.yaml \
    OUTPUT_DIR  ../outputs/tk/image/faster_rcnn_X_101_32x8d_FPN_3x \
    AL.MODE image 
#!/bin/bash

config="$1"
gpu_num="$2"
shift 2

base_command="python train_al_model.py \
                    --num-gpus $gpu_num \
                    --dataset_name          HJDataset \
                    --json_annotation_train ../data/HJDataset/annotations/train.json \
                    --image_path_train      ../data/HJDataset/raw/train \
                    --json_annotation_val   ../data/HJDataset/annotations/val.json \
                    --image_path_val        ../data/HJDataset/raw/val \
                    --config-file           $config"

bash ./train_base.sh -c "$config" -b "$base_command" -n HJDataset "$@"
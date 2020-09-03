#!/bin/bash

config="$1"
gpu_num="$2"
shift 2

base_command="python train_al_model.py \
                    --num-gpus $gpu_num \
                    --dataset_name          coco-sml \
                    --json_annotation_train ../data/coco/annotations/train-sml.json \
                    --image_path_train      ../data/coco/raw/val2017 \
                    --json_annotation_val   ../data/coco/annotations/val-sml.json \
                    --image_path_val        ../data/coco/raw/val2017 \
                    --config-file           $config"

bash ./train_base.sh -c "$config" -b "$base_command" -n coco-sml "$@"
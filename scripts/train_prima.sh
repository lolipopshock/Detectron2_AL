#!/bin/bash

config="$1"
gpu_num="$2"
shift 2

base_command="python train_al_model.py \
                    --num-gpus $gpu_num \
                    --dataset_name          prima \
                    --json_annotation_train ../data/prima/annotations/train.json \
                    --image_path_train      ../data/prima/raw/Images \
                    --json_annotation_val   ../data/prima/annotations/val.json \
                    --image_path_val        ../data/prima/raw/Images \
                    --config-file           $config"

bash ./train_base.sh -c "$config" -b "$base_command" -n prima "$@"
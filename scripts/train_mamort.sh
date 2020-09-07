#!/bin/bash

config="$1"
gpu_num="$2"
shift 2

base_command="python train_al_model.py \
                    --num-gpus $gpu_num \
                    --dataset_name          mamort \
                    --json_annotation_train ../data/mamort/annotations/train.json \
                    --image_path_train      ../data/mamort/raw/images \
                    --json_annotation_val   ../data/mamort/annotations/val.json \
                    --image_path_val        ../data/mamort/raw/images \
                    --config-file           $config"

bash ./train_base.sh -c "$config" -b "$base_command" -n mamort "$@"
#!/bin/bash

config="$1"
gpu_num="$2"
shift 2

base_command="python train_al_model.py \
                    --num-gpus              $gpu_num\
                    --dataset_name          tkdata-v10 \
                    --json_annotation_train ../data/tk1957-v10/annotations/train.json \
                    --image_path_train      ../data/tk1957-v10/images \
                    --json_annotation_val   ../data/tk1957-v10/annotations/val.json \
                    --image_path_val        ../data/tk1957-v10/images \
                    --config-file           $config"

bash ./train_base.sh -c "$config" -b "$base_command" -n tk "$@"
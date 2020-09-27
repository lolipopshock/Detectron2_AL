#!/bin/bash

config="$1"
gpu_num="$2"
fold="$3"
shift 3

base_command="python train_al_model.py \
                    --num-gpus $gpu_num \
                    --dataset_name          prima \
                    --json_annotation_train ../data/prima/annotations/cv/$fold/train.json \
                    --image_path_train      ../data/prima/raw/Images \
                    --json_annotation_val   ../data/prima/annotations/cv/$fold/val.json \
                    --image_path_val        ../data/prima/raw/Images \
                    --config-file           $config"

bash ./train_base_cv.sh -c "$config" -b "$base_command" -n prima -f $fold "$@"
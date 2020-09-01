#!/bin/bash

config="$1"
gpu_num="$2"
shift 2

base_command="python train_al_model.py \
                    --num-gpus $gpu_num \
                    --dataset_name          publaynet-sml \
                    --json_annotation_train ../data/publaynet/annotations/annotation-train.json \
                    --image_path_train      ../data/publaynet/val \
                    --json_annotation_val   ../data/publaynet/annotations/annotation-val.json \
                    --image_path_val        ../data/publaynet/val \
                    --config-file           $config"

bash ./train_base.sh -c "$config" -b "$base_command" -n publaynet "$@"
#!/bin/bash

config="$1"
gpu_num="$2"
shift 2

base_command="python train_al_model.py \
                    --num-gpus $gpu_num \
                    --dataset_name          publaynet-sml \
                    --json_annotation_train ../data/publaynet/annotations/train-sml.json \
                    --image_path_train      ../data/publaynet/raw/val \
                    --json_annotation_val   ../data/publaynet/annotations/val-sml.json \
                    --image_path_val        ../data/publaynet/raw/val \
                    --config-file           $config"

bash ./train_base.sh -c "$config" -b "$base_command" -n publaynet-sml "$@"
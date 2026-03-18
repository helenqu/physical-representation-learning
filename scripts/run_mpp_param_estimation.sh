#!/bin/bash
source "$(dirname "$0")/env_setup.sh"

# $1: dataset_name, $2: path to pretrained MPP checkpoint
torchrun --nproc_per_node=2 --standalone \
    -m physics_jepa.baselines.mpp_param_estimation \
    --dataset_name $1 \
    --mpp_checkpoint_path $2

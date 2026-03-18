#!/bin/bash
source "$(dirname "$0")/../env_setup.sh"

torchrun --nproc_per_node=8 --standalone \
    -m physics_jepa.train_jepa \
    configs/train_shearflow_small.yaml \
    $1

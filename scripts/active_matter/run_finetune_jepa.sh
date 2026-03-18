#!/bin/bash
source "$(dirname "$0")/../env_setup.sh"

# Pass the pretrained checkpoint path as $1
python -m physics_jepa.finetune \
    configs/train_activematter_small.yaml \
    --trained_model_path $1

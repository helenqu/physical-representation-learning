#!/bin/bash
# Sourced by all scripts in scripts/**/

source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate jepa

cd /scratch/$USER/physical-representation-learning

export THE_WELL_DATA_DIR=/scratch/$USER/physical-representation-learning/datasets
export HDF5_USE_FILE_LOCKING=FALSE

# Keep HF cache off $HOME
export HF_HOME=/scratch/$USER/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

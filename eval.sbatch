#!/bin/bash
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --job-name=jepa-eval-frozen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err

# usage: sbatch eval.sbatch /scratch/$USER/physical-representation-learning/checkpoints/<run>/ConvEncoder_<epoch>.pth

set -euo pipefail

CKPT="${1:?pass the encoder .pth checkpoint path as argument}"

REPO=/scratch/$USER/physical-representation-learning
OVERLAY=$REPO/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif
CONFIG=configs/train_activematter_frozen.yaml

echo "[$(date)] Job $SLURM_JOB_ID on $(hostname) — checkpoint: $CKPT"
nvidia-smi

singularity exec --nv \
  --overlay "$OVERLAY:ro" \
  --bind /scratch/$USER:/scratch/$USER \
  "$IMAGE" bash -c "
    source /ext3/env.sh
    conda activate jepa
    cd $REPO
    export THE_WELL_DATA_DIR=$REPO/datasets
    export HDF5_USE_FILE_LOCKING=FALSE
    python -m physics_jepa.eval_frozen \
      --config $CONFIG \
      --checkpoint $CKPT
  "

echo "[$(date)] Done — results at $REPO/frozen_eval_out/results.json"

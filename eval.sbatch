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

# Derive a per-checkpoint tag: "<run_dir>_epoch<N>" from ".../noise-std-1.0_2026.../ConvEncoder_11.pth"
CKPT_BASE=$(basename "$CKPT" .pth)                   # e.g. ConvEncoder_11
EPOCH=${CKPT_BASE##*_}                                # e.g. 11
RUN_DIR=$(basename "$(dirname "$CKPT")")              # e.g. noise-std-1.0_2026-04-16_22:00:00
TAG="${RUN_DIR}_epoch${EPOCH}"
OUT_DIR="$REPO/frozen_eval_out/$TAG"

echo "[$(date)] Job $SLURM_JOB_ID on $(hostname) — checkpoint: $CKPT"
echo "results will land in: $OUT_DIR"
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
      --checkpoint $CKPT \
      ft.out_dir=$OUT_DIR \
      ft.run_name=frozen-$TAG
  "

echo "[$(date)] Done — results at $OUT_DIR/results.json"

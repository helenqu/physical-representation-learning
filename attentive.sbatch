#!/bin/bash
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --job-name=jepa-eval-attentive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=attentive_%j.out
#SBATCH --error=attentive_%j.err

# Paper-style attentive probe on a frozen encoder checkpoint.
# usage: sbatch attentive.sbatch /scratch/$USER/physical-representation-learning/checkpoints/<run>/ConvEncoder_<epoch>.pth

set -euo pipefail

CKPT="${1:?pass the encoder .pth checkpoint path as argument}"

REPO=/scratch/$USER/physical-representation-learning
OVERLAY=$REPO/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif
CONFIG=configs/train_activematter_small.yaml

# Per-checkpoint tag so outputs and wandb runs stay distinct.
CKPT_BASE=$(basename "$CKPT" .pth)                   # ConvEncoder_11
EPOCH=${CKPT_BASE##*_}                                # 11
RUN_DIR=$(basename "$(dirname "$CKPT")")              # noise-std-1.0_2026-04-16_...
TAG="${RUN_DIR}_epoch${EPOCH}"

# Separate root so attentive-probe artifacts don't mingle with frozen-probe artifacts.
ATT_ROOT=$REPO/attentive_eval_out
EMBED_DIR=$ATT_ROOT/embeddings
CKPT_OUT=$ATT_ROOT/ckpts

echo "[$(date)] Job $SLURM_JOB_ID on $(hostname) — checkpoint: $CKPT, tag: $TAG"
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
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python -m physics_jepa.finetune \
      $CONFIG \
      ft.run_name=attentive-$TAG \
      ft.embeddings_dir=$EMBED_DIR \
      out_path=$CKPT_OUT \
      --trained_model_path $CKPT
  "

echo "[$(date)] Done — attentive-probe artifacts under $ATT_ROOT"

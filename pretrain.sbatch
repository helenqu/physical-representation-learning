#!/bin/bash
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --job-name=jepa-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=pretrain_%j.out
#SBATCH --error=pretrain_%j.err
#SBATCH --signal=B:USR1@120

set -euo pipefail

REPO=/scratch/$USER/physical-representation-learning
OVERLAY=$REPO/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif
CONFIG=configs/train_activematter_small.yaml

trap 'echo "[$(date)] SIGUSR1 received — exiting for requeue"; exit 0' USR1

echo "[$(date)] Job $SLURM_JOB_ID on $(hostname)"
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
    export WANDB_API_KEY=\${WANDB_API_KEY:-}
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    torchrun --standalone --nproc_per_node=1 \
      -m physics_jepa.train_jepa \
      $CONFIG \
      out_path=$REPO/checkpoints
  "

echo "[$(date)] Done"

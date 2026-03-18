This is the official code repository for the paper [Representation Learning for Spatiotemporal Physical Systems](arxiv.org/pdf/2603.13227).

## Installation

**Requirements:** Python 3.10+, PyTorch 2.0+ with CUDA.

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-org/physics_jepa_public
cd physics_jepa_public
pip install torch torchvision einops omegaconf wandb tqdm h5py psutil scikit-learn timm the-well
pip install -e .
```

### Environment setup

Edit `scripts/env_setup.sh` to activate your virtual environment and set the path to [The Well](https://github.com/PolymathicAI/the_well) datasets. This file is sourced automatically by all scripts. The `THE_WELL_DATA_DIR` variable is required by all training and finetuning scripts that use The Well data.

## Training

### 1. JEPA pretraining

Pretrain a convolutional JEPA encoder on a physics dataset using the scripts in `scripts/<dataset>/`:

| Dataset | Script |
|---|---|
| Shear flow | `scripts/shear_flow/run_train_jepa.sh` |
| Rayleigh-Bénard | `scripts/rayleigh_benard/run_train_jepa.sh` |
| Active matter | `scripts/active_matter/run_train_jepa.sh` |

Config fields `out_path` and `cache_path` control where checkpoints and dataset caches are written. Key training hyperparameters (learning rate, number of epochs, noise level, etc.) are set in the `train:` block of the corresponding config. Config fields can be overridden from the command line by passing `key=value` arguments to the script, e.g.:

```bash
scripts/shear_flow/run_train_jepa.sh train.num_epochs=10 train.lr=5e-4
```

### 2. VideoMAE finetuning (baseline)

Fine-tune a pretrained [VideoMAE](https://github.com/MCG-NJU/VideoMAE) backbone for physical parameter estimation. Set the `CHECKPOINT_PATH` environment variable to the pretrained VideoMAE checkpoint and run the appropriate script:

| Dataset | Script |
|---|---|
| Shear flow | `scripts/shear_flow/run_finetune_videomae.sh` |
| Rayleigh-Bénard | `scripts/rayleigh_benard/run_finetune_videomae.sh` |
| Active matter | `scripts/active_matter/run_finetune_videomae.sh` |

### 3. JEPA finetuning (parameter estimation)

Fine-tune a pretrained JEPA encoder for physical parameter estimation. Set `CHECKPOINT_PATH` to a saved encoder checkpoint and run the appropriate script:

| Dataset | Script |
|---|---|
| Shear flow | `scripts/shear_flow/run_finetune_jepa.sh` |
| Rayleigh-Bénard | `scripts/rayleigh_benard/run_finetune_jepa.sh` |
| Active matter | `scripts/active_matter/run_finetune_jepa.sh` |

The same configs used for pretraining are reused here; the `ft:` block controls finetuning hyperparameters. A multi-GPU variant is available at `scripts/shear_flow/run_finetune_jepa_ddp.sh`.

## Baselines

### 4. DISCO finetuning

[DISCO](https://arxiv.org/abs/2401.09246) is a latent-space parameter estimation baseline. It operates on precomputed DISCO latent representations rather than raw data. Pass the path to a directory of DISCO inference outputs as the first argument:

```bash
scripts/run_finetune_disco.sh /path/to/disco_inference_shear_flow
```

The data directory name must match one of the dataset keys in `physics_jepa/baselines/disco.py` (e.g. `disco_inference_shear_flow`, `disco_inference_rayleigh_benard`, `disco_inference_active_matter`).

### 5. MPP finetuning

Fine-tune a pretrained [MPP](https://github.com/PolymathicAI/multiple_physics_pretraining) (Multiple Physics Pretraining) model for physical parameter estimation. Requires the `avit`, `spatial_modules`, and `YParams` packages from the MPP codebase. Pass the dataset name and path to a pretrained MPP checkpoint:

```bash
scripts/run_mpp_param_estimation.sh shear_flow /path/to/MPP_AViT_Ti
```

`--dataset_name` should match the corresponding dataset directory name in `THE_WELL_DATA_DIR`. The checkpoint save directory can be controlled via the `CHECKPOINT_DIR` environment variable (defaults to `./checkpoints`).

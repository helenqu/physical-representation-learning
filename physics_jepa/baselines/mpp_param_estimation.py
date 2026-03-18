import os
import torch
import torch.nn as nn
from pathlib import Path
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
import h5py
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import wandb
import argparse

from physics_jepa.data import WellDatasetForMPP
from physics_jepa.attentive_pooler import AttentiveClassifier
from avit import build_avit
from spatial_modules import SubsampledLinear
from YParams import YParams

class ParameterEstimationModel(nn.Module):
    def __init__(self, pretrained_model, num_outputs=2, freeze_encoder=True, hidden_dim=32):
        super().__init__()
        self.encoder = pretrained_model
        
        # Freeze the encoder (everything up to debed block)
        if freeze_encoder:
            self._freeze_encoder()

        # Add global average pooling and regression head
        embed_dim = pretrained_model.embed.embed_dim
        self.regression_head = AttentiveClassifier(
            embed_dim=embed_dim, 
            num_classes=num_outputs,
            num_heads=8
        )
        
    def _freeze_encoder(self):
        """Freeze all parameters in the encoder (space_bag, embed, blocks)"""
        # Freeze space_bag
        for param in self.encoder.space_bag.parameters():
            param.requires_grad = False
            
        # Freeze embed
        for param in self.encoder.embed.parameters():
            param.requires_grad = False
            
        # Freeze processor blocks
        for param in self.encoder.blocks.parameters():
            param.requires_grad = False
            
    def forward(self, x, state_labels, bcs):
        """Default forward pass that includes regression head"""
        x = self.forward_encoder_only(x, state_labels, bcs)
        return self.regression_head(x)
    
    def forward_encoder_only(self, x, state_labels, bcs):
        """Forward pass through encoder only (without regression head)"""
        T = x.shape[0]
        
        # Normalize (time + space per sample) - same as original -- may already be taken care of by the well
        with torch.no_grad():
            data_std, data_mean = torch.std_mean(x, dim=(0, -2, -1), keepdims=True)
            data_std = data_std + 1e-7
        x = (x - data_mean) / (data_std)

        # Sparse proj
        x = rearrange(x, 't b c h w -> t b h w c')
        x = self.encoder.space_bag(x, state_labels)
        # Encode
        x = rearrange(x, 't b h w c -> (t b) c h w')
        x = self.encoder.embed(x)            
        x = rearrange(x, '(t b) c h w -> t b c h w', t=T)

        # Process through encoder blocks
        for blk in self.encoder.blocks:
            x = blk(x, bcs)
        
        # Take the last time step for parameter estimation
        x = rearrange(x[-1], 'b c h w -> b (h w) c')  # Shape: (B, num_tokens, C)
        
        return x
    
    def forward_regression_only(self, embeddings):
        """Forward pass using pre-computed embeddings"""
        return self.regression_head(embeddings)
    
def normalize_labels(x, stats={}):
    if 'mins' in stats and 'maxes' in stats:
        mins = torch.tensor(stats['mins'])
        maxes = torch.tensor(stats['maxes'])
        return (x - mins) / (maxes - mins)
    elif 'means' in stats and 'stds' in stats:
        if 'compression' in stats:
            for i, compression_type in enumerate(stats['compression']):
                if compression_type == 'log':
                    x[:, i] = torch.log10(x[:, i])
                elif compression_type == None:
                    continue
        means = torch.tensor(stats['means'])
        stds = torch.tensor(stats['stds'])
        return (x - means) / stds
    else:
        return x

def train_parameter_estimation(model, train_loader, val_loader, dataset_name, num_epochs=50, lr=1e-3, rank=0, world_size=1, wandb_run=None):
    """
    Full weight fine-tuning with DDP support and wandb logging
    """
    STATS = {
        "gray_scott_reaction_diffusion": {
            "means": [0.086, 0.058], # F, k
            "stds": [0.1, 0.005]
        },
        "active_matter": {
            "means": [-3.0, 9.0], # alpha, zeta
            "stds": [1.41, 5.16],
        },
        "shear_flow": {
            "means": [4.85, 2.69], # rayleigh, schmidt
            "stds": [0.61, 3.38],
            "compression": ["log", None],
        },
        "rayleigh_benard": {
            "means": [2.69,8.0], # prandtl, rayleigh
            "stds": [3.38, 1.41],
            "compression": [None, "log"],
        },
    }
    label_stats = STATS[dataset_name]

    model.train()
    
    # Loss function and optimizer (for all parameters)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    if rank == 0:
        print(f"Starting full weight fine-tuning for {num_epochs} epochs...")
        print(f"Learning rate: {lr}")
        print(f"World size: {world_size}")
        
        # Initialize wandb if provided
        if wandb_run is not None:
            wandb_run.watch(model, log="all", log_freq=100)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        skipped_batches = 0
        
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', disable=rank != 0)
        for batch_idx, (ctx, physical_params, state_labels, bcs) in enumerate(train_pbar):
            ctx = rearrange(ctx, 'b c t h w -> t b c h w')
            # Move data to device
            ctx = ctx.to(rank)
            physical_params = normalize_labels(physical_params, stats=label_stats).to(rank)
            state_labels = state_labels.to(rank)
            bcs = bcs.to(rank)
            
            # Forward pass (full model with regression head)
            optimizer.zero_grad()
            predictions = model(ctx, state_labels, bcs)
        
            if torch.isnan(predictions).any():
                print(f"Found NaN in predictions at batch {batch_idx}")
                skipped_batches += 1
                continue
            
            if batch_idx % 100 == 0:
                print(f"preds: {predictions[:5]}, labels: {physical_params[:5]}")
            
            # Compute loss
            loss = criterion(predictions, physical_params)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_batches += 1
            
            # Update progress bar (only on rank 0)
            if rank == 0:
                train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
                # Log batch metrics to wandb
                if wandb_run is not None and batch_idx % 100 == 0:  # Log every 10 batches
                    wandb_run.log({
                        'train/batch_loss': loss.item(),
                        'train/learning_rate': optimizer.param_groups[0]['lr'],
                        'epoch': epoch + 1,
                        'batch': batch_idx
                    })
        
        if rank == 0:
            print(f"Skipped {skipped_batches} / {train_batches} batches")

        avg_train_loss = train_loss / train_batches
        if wandb_run is not None:
            wandb_run.log({
                'train/epoch_loss': avg_train_loss,
            })
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', disable=rank != 0)
            for ctx, physical_params, state_labels, bcs in val_pbar:
                ctx = rearrange(ctx, 'b c t h w -> t b c h w')
                # Move data to device
                ctx = ctx.to(rank)
                physical_params = normalize_labels(physical_params, stats=label_stats).to(rank)
                state_labels = state_labels.to(rank)
                bcs = bcs.to(rank)
                
                # Forward pass (full model with regression head)
                predictions = model(ctx, state_labels, bcs)
                
                # Compute loss
                loss = criterion(predictions, physical_params)
                
                # Update metrics
                val_loss += loss.item()
                val_batches += 1
                
                # Update progress bar (only on rank 0)
                if rank == 0:
                    val_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_val_loss = val_loss / val_batches
        if wandb_run is not None:
            wandb_run.log({
                'val/epoch_loss': avg_val_loss,
            })
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model (only on rank 0)
        if rank == 0 and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the underlying model state dict for DDP
            model_state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            
            # Save local checkpoint
            checkpoint_path = Path(os.environ.get("CHECKPOINT_DIR", "./checkpoints")) / f"{config.params['dataset_name']}_mpp_full_finetuning"
            if not checkpoint_path.exists():
                checkpoint_path.mkdir(parents=True, exist_ok=False)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
            }, checkpoint_path / f'best_param_estimation_model.pth')
            print(f"Saved best model with validation loss: {best_val_loss:.6f}")
            
            # Log model artifact to wandb
            if wandb_run is not None:
                artifact = wandb.Artifact(
                    name=f"best-model-{wandb_run.id}", 
                    type="model",
                    description=f"Best model from epoch {epoch+1} with validation loss {best_val_loss:.6f}"
                )
                artifact.add_file(checkpoint_path / f'best_param_estimation_model.pth')
                wandb_run.log_artifact(artifact)
        
        # Log epoch metrics to wandb (only on rank 0)
        if rank == 0 and wandb_run is not None:
            wandb_run.log({
                'epoch': epoch + 1,
                'train/epoch_loss': avg_train_loss,
                'val/epoch_loss': avg_val_loss,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'best_val_loss': best_val_loss,
            })

        # Print epoch summary (only on rank 0)
        if rank == 0:
            print(train_losses, val_losses)
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            print(f"  Val Loss: {avg_val_loss:.6f}")
            print(f"  Best Val Loss: {best_val_loss:.6f}")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            print("-" * 50)
    
    return train_losses, val_losses

def setup_distributed():
    """
    Set up distributed training environment
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return 0, 1
    
    print(f"Rank: {rank}, World size: {world_size}, Local rank: {local_rank}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    dist.barrier()
    
    return rank, world_size

def cleanup_distributed():
    """
    Clean up distributed training environment
    """
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    # Set up distributed training
    rank, world_size = setup_distributed()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--mpp_checkpoint_path', type=str, required=True,
                        help='Path to the pretrained MPP checkpoint')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    # Load configuration and pretrained model
    config = YParams(Path('config/mpp_avit_ti_config.yaml').resolve(), 'basic_config')
    config.params.update({
        'batch_size': 32, 
        'dataset_name': args.dataset_name, 
        'num_frames': 16, 
        'freeze_encoder': False,
    })
    
    batch_size = 32
    freeze_encoder = False  # Set to False for full weight fine-tuning

    well_data_dir = os.environ.get("THE_WELL_DATA_DIR")
    if well_data_dir is None:
        raise ValueError("THE_WELL_DATA_DIR environment variable is not set.")
    train_dataset = WellDatasetForMPP(
        data_dir=Path(well_data_dir) / config.params['dataset_name'],
        split="train",
        num_frames=16,
        resolution=(224, 224),
        stride=4
    )
    val_dataset = WellDatasetForMPP(
        data_dir=Path(well_data_dir) / config.params['dataset_name'],
        split="val",
        num_frames=16,
        resolution=(224, 224),
        stride=4
    ) 

    # Create distributed samplers and dataloaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        sampler=val_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    wandb_run = None
    if rank == 0 and not args.dry_run:
        wandb.init(
            project="mpp-baseline",
            name=f"{config.params['dataset_name']}-full-finetuning",
            config=config.params,
            tags=["full-finetuning", "ddp", "parameter-estimation"]
        )
        wandb_run = wandb.run

    state = torch.load(args.mpp_checkpoint_path)

    # Build the base model and load pretrained weights
    base_model = build_avit(config)
    base_model.load_state_dict(state)

    # Create the parameter estimation model
    param_model = ParameterEstimationModel(base_model, num_outputs=2, freeze_encoder=freeze_encoder)

    # Print total and trainable parameters
    total_params = sum(p.numel() for p in param_model.parameters())
    trainable_params = sum(p.numel() for p in param_model.parameters() if p.requires_grad)
    print(f"Total parameters in param_model: {total_params}")
    print(f"Trainable parameters in param_model: {trainable_params}")

    unused_params = [4, 7, 10, 14, 16, 31, 49, 51, 66, 84, 86, 101, 119, 121, 136, 154, 156, 171, 189, 191, 206, 224, 226, 241, 259, 261, 276, 294, 296, 311, 329, 331, 346, 364, 366, 381, 399, 401, 416, 431, 432, 433, 434, 435, 436, 437, 438, 446, 447]
    for i, (name, param) in enumerate(param_model.named_parameters()):
        if i in unused_params:
            # print(f"param {i}: {name}: {param.shape}")
            param.requires_grad = False
    if config.params['dataset_name'] == "active_matter":
        param_model.encoder.space_bag = SubsampledLinear(11, param_model.encoder.embed.embed_dim//4)

    # Wrap model with DDP
    if world_size > 1:
        param_model = DDP(param_model.to(rank), device_ids=[rank])
    else:
        param_model = param_model.to(rank)

    if rank == 0:
        print("Parameter Estimation Model:")
        print(param_model)

    if rank == 0:
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")

    # Training parameters for full weight fine-tuning
    num_epochs = 30
    base_lr = 1e-5  # Lower learning rate for full fine-tuning

    # Start full weight fine-tuning
    if rank == 0:
        print("\nStarting full weight fine-tuning...")
    train_losses, val_losses = train_parameter_estimation(
        model=param_model,
        train_loader=train_loader,
        val_loader=val_loader,
        dataset_name=config.params['dataset_name'],
        num_epochs=num_epochs,
        lr=base_lr,
        rank=rank,
        world_size=world_size,
        wandb_run=wandb_run
    )

    if rank == 0:
        print("Training completed!")
        print(f"Best validation loss: {min(val_losses):.6f}")
    
    # Clean up distributed training
    cleanup_distributed()
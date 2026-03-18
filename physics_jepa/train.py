import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset
from torch.nn import MSELoss

import os
from pathlib import Path
from tqdm import tqdm
from einops import rearrange
import wandb
from omegaconf import OmegaConf
from collections import defaultdict
import datetime
import gc

from .data import get_train_dataloader_from_cfg, get_val_dataloader_from_cfg, get_dataset_metadata
from .model import get_model_and_loss_cnn, get_autoencoder
from .utils.model_utils import CosineLRScheduler
from .utils.data_utils import mae
from .utils.hydra import compose
from .utils.misc import distprint
from .utils.train_utils import ddp_setup, gather_losses_and_report
from .utils.model_summary import summarize_convs
from .attentive_pooler import AttentiveClassifier

class Trainer:
    def __init__(self, cfg, stage="train"):
        self.cfg = cfg
        self.train_cfg = cfg[stage]

        if self.cfg.model.get("vit_equivalency", None) == 'tiny':
            assert self.cfg.model.dims[-1] == 384, "dims must be [48, 96, 192, 384] for tiny vit equivalency"
            assert self.cfg.dataset.get('resolution', None) == 224, "resolution must be 224 for tiny vit equivalency"
            assert self.cfg.dataset.get('num_frames', None) == 4, "num_frames must be 4 for current implementation of tiny vit equivalency"

        if os.environ.get("LOCAL_RANK", None) is not None:
            ddp_setup()
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
            torch.cuda.set_device(0)

        distprint(OmegaConf.to_yaml(self.cfg, resolve=True), local_rank=self.rank)

        self.train_loader = get_train_dataloader_from_cfg(self.cfg, stage=stage, rank=self.rank, world_size=self.world_size)
        self.val_loader = get_val_dataloader_from_cfg(self.cfg, stage=stage, rank=self.rank, world_size=self.world_size)
        self.is_iterable_dataset = isinstance(self.train_loader.dataset, IterableDataset)

    def train(self):
        if self.is_iterable_dataset:
            distprint(f"Running on {self.world_size} devices, batch size {self.train_cfg.batch_size}", local_rank=self.rank)
        else:
            distprint(f"Loaded {len(self.train_loader)} training batches per device on {self.world_size} devices, batch size {self.train_cfg.batch_size}", local_rank=self.rank)

        model_components, loss_fn = self.get_model_components()
        
        # Get weight decay from config, default to 0.05 if not specified
        weight_decay = self.train_cfg.get("weight_decay", 0.05)
        
        optimizer = torch.optim.AdamW(
            [p for component in model_components for p in list(component.parameters())], 
            lr=self.train_cfg.lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )

        run_name = f"{self.cfg.dataset.name}-{self.cfg.dataset.num_frames}frames-{self.cfg.model.name}-{self.cfg.model.objective}"
        if self.train_cfg.get("run_name", None) is not None:
            run_name = f"{run_name}-{self.train_cfg.run_name}"
        if self.rank == 0 and not self.cfg.dry_run:
            wandb.init(project="physics-jepa" if 'seed' not in self.cfg.out_path else "physics-jepa-seeds",
                name=run_name,
                config=OmegaConf.to_container(self.cfg))

        self.training_loop(model_components, loss_fn, optimizer, run_name)

        if self.world_size > 1:
            dist.destroy_process_group()
    
    def set_up_gradient_accumulation(self):
        actual_global_batch_size = self.train_cfg.batch_size * self.world_size
        return max(self.train_cfg.get("target_global_batch_size", 256) // actual_global_batch_size, 1)

    def training_loop(self, model_components, loss_fn, optimizer, run_name):
        # set up gradient accumulation
        grad_accum_steps = self.set_up_gradient_accumulation()
        distprint(f"using gradient accumulation with {grad_accum_steps} steps", local_rank=self.rank)

        steps = self.train_cfg.num_epochs * self.train_cfg.steps if 'steps' in self.train_cfg else self.train_cfg.num_epochs * len(self.train_loader)

        # Setup cosine LR scheduler with warmup
        if self.train_cfg.get("lr_scheduler", None) == "cosine":
            max_lr = self.train_cfg.get("lr", self.train_cfg.lr)
            min_lr = self.train_cfg.get("min_lr", 1e-6)
            warmup_steps_from_epochs = self.train_cfg.get("lr_scheduler_warmup_epochs", 0) * len(self.train_loader) if isinstance(self.train_loader.dataset, IterableDataset) else 0
            warmup_steps = max(self.train_cfg.get("lr_scheduler_warmup_steps", 0), warmup_steps_from_epochs)
            warmup_updates = (warmup_steps + grad_accum_steps - 1) // grad_accum_steps
            total_updates = (steps + grad_accum_steps - 1) // grad_accum_steps
            
            distprint(f"using cosine scheduler with max_lr {max_lr}, min_lr {min_lr}, warmup_steps {warmup_updates}, total_updates {total_updates}", local_rank=self.rank)

            # Use the existing cosine_scheduler function
            lr_scheduler = CosineLRScheduler(
                optimizer,
                step=self.train_cfg.get("start_step", 0) // grad_accum_steps,
                base_value=max_lr,
                final_value=min_lr,
                steps=total_updates,
                warmup_steps=warmup_updates,
                start_warmup_value=min_lr
            )
        else:
            lr_scheduler = None

        if not self.is_iterable_dataset:
            distprint(f"starting to train w/ {len(self.train_loader)} batches per device", local_rank=self.rank)

        date_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        out_path = Path(self.cfg.out_path) / f"{run_name}_{date_str}"

        if self.rank == 0:
            epochs = tqdm(range(self.train_cfg.num_epochs))
        else:
            epochs = range(self.train_cfg.num_epochs)

        for epoch in epochs:
            if self.train_cfg.get("not_from_embeddings", False): # compute embeddings at each epoch
                model_components[0].eval()
                model_components[1].train()
            else:
                for component in model_components:
                    component.train()

            epoch_losses_dict = defaultdict(list)
            if self.rank == 0:
                start_time = datetime.datetime.now()

            for i, batch in enumerate(self.train_loader):
                i += self.train_cfg.get("start_step", 0) # add start step to the global step count

                pred, loss_dict = self.step(batch, model_components, loss_fn, self.rank, log=(i == 0 and epoch % 10 == 0))
                if i == 0:
                    distprint(f"train batch {i} pred: {pred[:5]}", local_rank=self.rank)
                    if 'label' in batch:
                        distprint(f"train batch {i} label: {batch['label'][:5]}", local_rank=self.rank)

                del (batch, pred)

                loss = loss_dict['loss'] / grad_accum_steps
                loss.backward()

                if i % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    # Set learning rate from schedule
                    if lr_scheduler is not None:
                        lr_scheduler.step()

                for loss_name, loss_value in loss_dict.items():
                    epoch_losses_dict[loss_name].append(loss_value.detach())

                if i == 0:
                    distprint(f"batch {i} loss: {loss_dict['loss'].detach().cpu()}", local_rank=self.rank)

                if i % self.train_cfg.report_every == 0:
                    # Get current learning rate from schedule
                    distprint(f"step {i}", local_rank=self.rank)
                    current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else self.train_cfg.lr
                    other_metrics = {'train/lr': current_lr, 'train/epoch': epoch}
                    gather_losses_and_report(epoch_losses_dict, other_metrics, self.rank, self.world_size, split='train', num_steps=self.train_cfg.report_every, dry_run=self.cfg.dry_run)
                    epoch_losses_dict = defaultdict(list)

                    if self.rank == 0:
                        elapsed_time = (datetime.datetime.now() - start_time).total_seconds() / 60
                        if i > 0 and not self.cfg.dry_run:
                            wandb.log({f'train/mins_per_{self.train_cfg.report_every}_steps': elapsed_time})
                        if i / self.train_cfg.report_every == 1 and epoch == 0:
                            # Calculate expected time to completion at first report
                            self.time_to_completion(start_time, i, steps)
                        start_time = datetime.datetime.now()
                
                if self.train_cfg.get("save_every_steps", None) is not None and i % self.train_cfg.save_every_steps == 0:
                    distprint(f"save_every_steps: {self.train_cfg.save_every_steps}, i: {i}", local_rank=self.rank)
                    if self.rank == 0:
                        if not out_path.exists():
                            out_path.mkdir(parents=True)
                            cfg_path = out_path / f"config.yaml"
                            OmegaConf.save(self.cfg, cfg_path)
                        for component in model_components:
                            torch.save(component.state_dict(), out_path / f"{component.__class__.__name__}_step{i}.pth")
                        print(f"checkpoint at step {i} saved to {out_path}")

                if self.train_cfg.get("steps", None) is not None and i > self.train_cfg.steps:
                    break

            for loss_name, loss_values in epoch_losses_dict.items():
                epoch_losses_dict[loss_name] = torch.stack(loss_values).mean().item()

            val_losses_dict = self.val(model_components, loss_fn, epoch)
            if val_losses_dict is not None:
                distprint(f"Epoch {epoch} val loss: {val_losses_dict['val/loss']}", local_rank=self.rank)

            if self.rank == 0 and (epoch+1) % self.train_cfg.save_every == 0 and epoch > 0:
                if not out_path.exists():
                    out_path.mkdir(parents=True)
                    cfg_path = out_path / f"config.yaml"
                    OmegaConf.save(self.cfg, cfg_path)
                for i, component in enumerate(model_components):
                    torch.save(component.state_dict(), out_path / f"{component.__class__.__name__}_{epoch}.pth")

        distprint(f"all checkpoints saved to {out_path}", local_rank=self.rank)

    def step(self, batch, model_components, loss_fn, device, log=False):
        if 'context' in batch: # B C T H W
            ctx = batch['context'].to(device)
            if log:
                distprint(f"ctx shape before pad (b c t h w): {ctx.shape}", local_rank=self.rank)
            if ctx.shape[2] < 4:
                # pad time dimension to 16 frames to get shapes to work out
                ctx = F.pad(ctx, (0, 0, 0, 0, 0, 4 - ctx.shape[2]))
            batch['context'] = ctx
        else:
            ctx = batch['embeddings']
            del ctx

        if 'target' in batch:
            tgt = batch['target'].to(device)
            if tgt.shape[2] < 4:
                # pad time dimension to 16 frames to get shapes to work out
                tgt = F.pad(tgt, (0, 0, 0, 0, 0, 4 - tgt.shape[2]))
            batch['target'] = tgt
            del tgt

        pred, loss_dict = self.pred_fn(batch, model_components, loss_fn)

        if log:
            distprint(f"pred shape: {pred.shape}", local_rank=self.rank)
            if 'label' in batch:
                distprint(f"label: {batch['label']}", local_rank=self.rank)
        
        del batch

        return pred, loss_dict

    def pred_fn(self, batch, model_components, loss_fn):
        raise NotImplementedError("pred_fn must be implemented in subclass")

    def val(self, model_components, loss_fn, epoch):
        distprint(f"Loaded {len(self.val_loader)} validation batches, batch size {self.train_cfg.batch_size}", local_rank=self.rank)

        for component in model_components:
            component.eval()

        val_losses_dict = defaultdict(list)
        for i, batch in enumerate(self.val_loader):
            with torch.no_grad():
                pred, loss_dict = self.step(batch, model_components, loss_fn, self.rank)
                if i == 0:
                    distprint(f"val batch {i} pred: {pred[:5]}", local_rank=self.rank)
                    if 'label' in batch:
                        distprint(f"val batch {i} label: {batch['label'][:5]}", local_rank=self.rank)
                for loss_name, loss_value in loss_dict.items():
                    val_losses_dict[loss_name].append(loss_value.detach())
                
                if self.train_cfg.get("val_steps", None) is not None and i % self.train_cfg.val_steps == 0:
                    break
            del batch, pred

        torch.cuda.empty_cache()
        gc.collect()

        val_losses_dict = gather_losses_and_report(val_losses_dict, {'val/epoch': epoch}, self.rank, self.world_size, split='val', dry_run=self.cfg.dry_run)

        return val_losses_dict

    def get_model_components(self):
        if self.cfg.model.objective == 'jepa':
            encoder, predictor, loss_fn = get_model_and_loss_cnn(
                self.cfg.model.dims,
                self.cfg.model.num_res_blocks,
                self.cfg.dataset.num_frames,
                in_chans=self.cfg.dataset.num_chans if 'fields' not in self.train_cfg else len(self.train_cfg.fields),
                sim_coeff=self.train_cfg.sim_coeff,
                std_coeff=self.train_cfg.std_coeff,
                cov_coeff=self.train_cfg.cov_coeff,
            )

            if 'encoder_path' in self.train_cfg and self.train_cfg.encoder_path is not None:
                distprint(f"loading encoder from {self.train_cfg.encoder_path}", local_rank=self.rank)
                state_dict = torch.load(self.train_cfg.encoder_path)
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                encoder.load_state_dict(state_dict)
            if 'predictor_path' in self.train_cfg and self.train_cfg.predictor_path is not None:
                distprint(f"loading predictor from {self.train_cfg.predictor_path}", local_rank=self.rank)
                state_dict = torch.load(self.train_cfg.predictor_path)
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                predictor.load_state_dict(state_dict)

            distprint(f"num encoder parameters: {sum(p.numel() for p in encoder.parameters())}", local_rank=self.rank)
            distprint(f"num predictor parameters: {sum(p.numel() for p in predictor.parameters())}", local_rank=self.rank)
            distprint(summarize_convs(encoder), local_rank=self.rank)

            model_components = [encoder, predictor]

        elif self.cfg.model.objective == 'ae':
            encoder, decoder = get_autoencoder(
                self.cfg.model.dims,
                in_chans=self.cfg.dataset.num_chans if 'fields' not in self.train_cfg else len(self.train_cfg.fields),
            )
            if 'encoder_path' in self.train_cfg and self.train_cfg.encoder_path is not None:
                state_dict = torch.load(self.train_cfg.encoder_path)
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                encoder.load_state_dict(state_dict)
            if 'decoder_path' in self.train_cfg and self.train_cfg.decoder_path is not None:
                state_dict = torch.load(self.train_cfg.decoder_path)
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                decoder.load_state_dict(state_dict)

            distprint(f"num encoder parameters: {sum(p.numel() for p in encoder.parameters())}", local_rank=self.rank)
            distprint(f"num decoder parameters: {sum(p.numel() for p in decoder.parameters())}", local_rank=self.rank)

            model_components = [encoder, decoder]
            loss_fn = mae
        elif self.cfg.model.objective == 'supervised':
            encoder, _, _ = get_model_and_loss_cnn(
                self.cfg.model.dims,
                self.cfg.model.num_res_blocks,
                self.cfg.dataset.num_frames,
                in_chans=self.cfg.dataset.num_chans if 'fields' not in self.train_cfg else len(self.train_cfg.fields),
            )
            metadata = get_dataset_metadata(self.cfg.dataset.name)
            head = AttentiveClassifier(
                embed_dim=self.cfg.model.dims[-1],
                num_classes=len(metadata.constant_scalar_names),
                num_heads=8,
            )

            distprint(f"num encoder parameters: {sum(p.numel() for p in encoder.parameters())}", local_rank=self.rank)
            distprint(summarize_convs(encoder), local_rank=self.rank)

            model_components = [encoder, head]
            loss_fn = MSELoss() # add mse loss on params
        else:
            raise ValueError(f"Invalid model objective: {self.cfg.model.objective}")
        
        if 'loss' in self.cfg.model and self.cfg.model.loss == 'gaussian_matching':
            from .model import vicreg_loss_bcs
            from functools import partial
            loss_fn = partial(vicreg_loss_bcs, sim_coeff=self.train_cfg.sim_coeff, bcs_coeff=self.train_cfg.bcs_coeff, num_slices=self.train_cfg.num_slices)

        if self.world_size > 1:
            for component in model_components:
                component = DDP(component.to(self.rank), device_ids=[self.rank])
        else:
            for component in model_components:
                component = component.to(self.rank)

        return model_components, loss_fn

    def time_to_completion(self, start_time, i, total_steps):
        steps_per_sec = (i + 1) / (datetime.datetime.now() - start_time).total_seconds()
        expected_sec = total_steps / steps_per_sec
        expected_time = datetime.timedelta(seconds=int(expected_sec))
        distprint(f"Expected time to completion: {expected_time}", local_rank=self.rank)

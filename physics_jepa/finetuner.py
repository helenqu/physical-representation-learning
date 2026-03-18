from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import IterableDataset
from abc import ABC, abstractmethod
import json
from einops import rearrange

import numpy as np
from pathlib import Path
import wandb
from tqdm import tqdm
import h5py
import psutil
import os
import gc
import time
from typing import List, Sequence
import re
from sklearn.metrics import f1_score

from .data import EmbeddingsDataset, get_dataset_metadata, get_train_dataloader, get_val_dataloader
from .model import get_model_and_loss_cnn, get_autoencoder
from .utils.data_utils import normalize_labels
from .utils.model_utils import RegressionHead, RegressionMLP
from .attentive_pooler import AttentiveClassifier
from .utils.train_utils import accuracy
from .train import Trainer
from .videomae import vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224, vit_huge_patch16_224

class BaseFinetuner(Trainer, ABC):
    def __init__(self, cfg, trained_model_path=None, rank=0, world_size=1):
        super().__init__(cfg, stage="ft")
        self.trained_model_path = trained_model_path
        self.cfg.ft.trained_model_path = trained_model_path
        self.train_cfg = cfg.ft
        self.loss_for_task = {"regression": nn.MSELoss(), "binary_classification": nn.BCEWithLogitsLoss(), 'classification': nn.CrossEntropyLoss()}
        self.label_name = 'physical_params'
        print(f"populating rank from args: {rank}", flush=True)
        self.rank = rank
        self.world_size = world_size

        STATS = {
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
        assert self.cfg.dataset.name in STATS, f"label stats for {self.cfg.dataset.name} not found"
        self.label_stats = STATS[self.cfg.dataset.name]

        self.seed = self.cfg.seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
    
    @abstractmethod
    def load_model(self):
        """Load the model and return the encoder component"""
        pass
    
    @abstractmethod
    def create_head(self, metadata):
        """Create the appropriate head for the model type"""
        pass

    @abstractmethod
    def _model_inference(self, ctx, encoder):
        """Model inference"""
        pass
    
    def inference_step(self, batch, encoder):
        """
        Default inference step implementation.
        Subclasses can override this to customize the inference process.
        """
        ctx = batch['context'].to(self.rank)
        if ctx.shape[2] < 4:
            # pad time dimension to 16 frames to get shapes to work out
            ctx = F.pad(ctx, (0, 0, 0, 0, 0, 4 - ctx.shape[2]))
        labels = normalize_labels(batch[self.label_name], stats=self.label_stats).to(self.rank)
        enc_ctx = self._model_inference(ctx, encoder)
        return enc_ctx, labels
    
    def train(self):
        run_name = self.cfg.ft.get("run_name", f"{self.cfg.dataset.name}-{self.cfg.dataset.num_frames}frames-{self.cfg.model.objective + '-FT' if not self.cfg.ft.get('not_from_embeddings', False) else 'supervised'}-{self.cfg.ft.task}{f'-randominit' if self.trained_model_path is None else ''}")
        if self.rank == 0 and not self.cfg.dry_run:
            wandb.init(project="physics-jepa",
                name=run_name,
                config=OmegaConf.to_container(self.cfg))
        
        if self.cfg.ft.get("not_from_embeddings", False):
            encoder = self.get_encoder_and_raw_loaders()
            model_components = [encoder]
        else:
            train_dataset, train_labels, val_dataset, val_labels = self.get_embeddings()
            train_dataset = EmbeddingsDataset(train_dataset, train_labels)
            val_dataset = EmbeddingsDataset(val_dataset, val_labels)
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.cfg.ft.batch_size,
                shuffle=True,
                num_workers=4,
                prefetch_factor=2
            )
            self.val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.cfg.ft.batch_size,
                shuffle=False,
                num_workers=4,
                prefetch_factor=2
            )
            model_components = []
            
        metadata = get_dataset_metadata(self.cfg.dataset.name)
        head = self.create_head(metadata)

        model_components.append(head)
        if self.world_size > 1:
            model_components[-1] = DDP(model_components[-1].to(self.rank), device_ids=[self.rank])
            if len(model_components) > 1:
                model_components[0].to(self.rank) # encoder doesn't need DDP
        else:
            for m in model_components:
                m.to(self.rank)

        optimizer = torch.optim.AdamW(
            [p for m in model_components for p in m.parameters()],
            lr=self.cfg.ft.lr, 
            weight_decay=self.cfg.ft.get('weight_decay', 0.01)
        )

        if self.cfg.ft.task == "binary_classification":
            assert self.cfg.ft.num_classes == 1, "binary classification must have 1 class"
        if self.cfg.ft.task == "classification" and self.cfg.ft.num_classes == 1:
            self.cfg.ft.task = "binary_classification"
        loss_fn = self.loss_for_task[self.cfg.ft.task] if self.cfg.ft.task in self.loss_for_task else None
        if loss_fn is None:
            raise ValueError(f"loss function not found for task {self.cfg.ft.task}")

        self.training_loop(model_components, loss_fn, optimizer, run_name)

        # Clean up HDF5 file handles if they exist
        self.cleanup_embedding_files()

        if self.world_size > 1:
            dist.destroy_process_group()

    def pred_fn(self, batch, model_components, loss_fn):
        if self.cfg.ft.get("not_from_embeddings", False):
            ctx = self._model_inference(batch['context'].to(self.rank), model_components[0])
            head = model_components[1]
            labels = normalize_labels(batch[self.label_name], stats=self.label_stats).to(self.rank)
        else:
            ctx = batch['embeddings'].to(self.rank)
            head = model_components[0]
            labels = batch['label'].to(self.rank) # don't normalize here since it's already done when saving embeddings, embeddings save as 'label'

        pred = head(ctx)

        # Check for NaN values in context and predictions
        if torch.isnan(ctx).any():
            print(f"WARNING: NaN detected in context tensor: {torch.isnan(ctx).sum()} NaN values out of {ctx.numel()} total")
            
        if torch.isnan(pred).any():
            print(f"WARNING: NaN detected in predictions: {torch.isnan(pred).sum()} NaN values out of {pred.numel()} total")

        loss_dict = {"loss": loss_fn(pred, labels)}
        if "classification" in self.cfg.ft.task:
            loss_dict["acc"] = accuracy(pred.detach(), labels)
            
            # Convert predictions to class predictions
            if self.cfg.ft.task == "binary_classification":
                pred_classes = (torch.sigmoid(pred.detach()) > 0.5).cpu().numpy()
            else:  # multiclass classification
                pred_classes = torch.argmax(pred.detach(), dim=1).cpu().numpy()
            true_classes = labels.cpu().numpy()
            
            # Calculate macro-F1 score
            macro_f1 = f1_score(true_classes, pred_classes, average='macro', zero_division=0)
            loss_dict["macro_f1"] = torch.tensor(macro_f1)
        
        return pred, loss_dict

    def get_encoder_and_raw_loaders(self):
        # make new loaders that have larger batch size for calculating embeddings
        self.train_loader = get_train_dataloader(
            self.cfg.dataset.name,
            self.cfg.dataset.num_frames,
            self.cfg.dataset.get("num_examples", None),
            self.cfg.ft.batch_size,
            shuffle=True,
            include_labels=True,
            predict_n_steps=False,
            rank=self.rank,
            world_size=self.world_size,
            task=self.cfg.ft.task,
            fields=self.cfg.ft.get("fields", None),
            balance_classes=self.cfg.ft.get("balance_classes", False),
            resolution=self.cfg.dataset.get("resolution", None),
            offset=self.cfg.dataset.get("offset", None),
            noise_std=self.cfg.ft.get("noise_std", 0.0),
        )
        self.val_loader = get_val_dataloader(
            self.cfg.dataset.name,
            self.cfg.dataset.num_frames,
            self.cfg.dataset.get("num_examples", None),
            self.cfg.ft.batch_size,
            shuffle=True,
            include_labels=True,
            predict_n_steps=False,
            rank=self.rank,
            world_size=self.world_size,
            task=self.cfg.ft.task,
            fields=self.cfg.ft.get("fields", None),
            balance_classes=False,
            resolution=self.cfg.dataset.get("resolution", None),
            offset=self.cfg.dataset.get("offset", None),
            noise_std=self.cfg.ft.get("noise_std", 0.0),
        )
        
        encoder = self.load_model()

        for param in encoder.parameters():
            param.requires_grad = False
        encoder.to(self.rank)

        return encoder

    def get_embeddings(self):
        # try to avoid issues reading right after writing
        import os
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

        runpath = Path(self.trained_model_path) if self.trained_model_path is not None else Path(f"{self.cfg.dataset.name}-{self.cfg.dataset.num_frames}frames-{str(self.cfg.ft.get('fields', None)) + 'fields-' if self.cfg.ft.get('fields', None) is not None else ''}-{self.cfg.model.objective}-{self.cfg.ft.task}/randominit-seed{self.seed}")
        if runpath.parent.name == "checkpoints": # backwards compatibility w/ old model paths that weren't nested
            train_embeddings_path = Path(self.cfg.ft.embeddings_dir) / f"{runpath.name}_embeddings_train.h5"
            val_embeddings_path = Path(self.cfg.ft.embeddings_dir) / f"{runpath.name}_embeddings_val.h5" 
        else:
            train_embeddings_path = Path(self.cfg.ft.embeddings_dir) / f"{runpath.parent.name}_{runpath.name}_{'noise-' + str(self.cfg.ft.get('noise_std', 0.0)) + '_' if self.cfg.ft.get('noise_std', 0.0) > 0.0 else ''}embeddings_train.h5"
            val_embeddings_path = Path(self.cfg.ft.embeddings_dir) / f"{runpath.parent.name}_{runpath.name}_{'noise-' + str(self.cfg.ft.get('noise_std', 0.0)) + '_' if self.cfg.ft.get('noise_std', 0.0) > 0.0 else ''}embeddings_val.h5"
        
        self.cfg.ft.embeddings_path = str(train_embeddings_path)
        if not train_embeddings_path.exists():
            print(f"no embeddings found at {train_embeddings_path}, calculating", flush=True)

            encoder = self.get_encoder_and_raw_loaders()
            train_batches_per_epoch = min(self.train_cfg.num_train_steps if 'num_train_steps' in self.train_cfg else len(self.train_loader), len(self.train_loader))

            # Create HDF5 files for train and val embeddings
            embeddings_dir = Path(self.cfg.ft.embeddings_dir)
            if not embeddings_dir.exists():
                embeddings_dir.mkdir(parents=True, exist_ok=True)

            # Initialize HDF5 files with train embeddings
            with h5py.File(train_embeddings_path, 'w') as train_file:
                # We'll create datasets after getting the first batch to know the shapes
                train_embeddings_buffer = []
                train_labels_buffer = []
                buffer_size = 100
                total_train_count = 0
                
                # Memory monitoring
                process = psutil.Process(os.getpid())
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                print(f"Initial memory usage: {initial_memory:.1f} MB", flush=True)

                # make sure not to allocate a more rows in the hdf5 dataset than dataset size
                train_dataset_size = train_batches_per_epoch * self.train_cfg.batch_size if 'num_train_steps' in self.train_cfg else len(self.train_loader.dataset)
                if not isinstance(self.train_loader.dataset, IterableDataset):
                    assert train_dataset_size <= len(self.train_loader.dataset), "num_train_examples is greater than the size of the train dataset"
                for i, batch in tqdm(enumerate(self.train_loader), desc="train embeddings", total=train_batches_per_epoch):
                    if i >= train_batches_per_epoch:
                        break

                    enc_ctx, labels = self.inference_step(batch, encoder)

                    train_embeddings_buffer.append(enc_ctx.detach().cpu().numpy())
                    train_labels_buffer.append(labels.detach().cpu().numpy())
                    
                    if i == 0:
                        # Create datasets after first batch to know the shapes
                        embedding_shape = (train_dataset_size,) + enc_ctx.shape[1:]
                        label_shape = (train_dataset_size,) + labels.shape[1:]
                        train_file.create_dataset('embeddings', shape=embedding_shape, dtype=np.float32, chunks=True)
                        train_file.create_dataset('labels', shape=label_shape, dtype=np.float32, chunks=True)

                    # Write to HDF5 when buffer is full
                    if len(train_embeddings_buffer) >= buffer_size:
                        all_train_embeddings = np.concatenate(train_embeddings_buffer, axis=0)
                        all_train_labels = np.concatenate(train_labels_buffer, axis=0)
                        start_idx = total_train_count
                        end_idx = total_train_count + len(all_train_embeddings)
                        train_file['embeddings'][start_idx:end_idx] = all_train_embeddings
                        train_file['labels'][start_idx:end_idx] = all_train_labels
                        total_train_count = end_idx
                        
                        # Memory monitoring after buffer write
                        current_memory = process.memory_info().rss / 1024 / 1024 / 1024 # GB
                        buffer_memory = sum(arr.nbytes for arr in train_embeddings_buffer) / 1024 / 1024 / 1024 # GB
                        print(f"Buffer written at step {i}: Memory: {current_memory:.1f} GB, Buffer size: {buffer_memory:.1f} GB", flush=True)
                        
                        train_embeddings_buffer = []
                        train_labels_buffer = []
                    
                    # Monitor memory every 100 steps
                    if i % 100 == 0 and i > 0:
                        current_memory = process.memory_info().rss / 1024 / 1024 / 1024 # GB
                        buffer_memory = sum(arr.nbytes for arr in train_embeddings_buffer) / 1024 / 1024 / 1024 # GB
                        print(f"Step {i}: Memory: {current_memory:.1f} GB, Buffer: {buffer_memory:.1f} GB, Buffer count: {len(train_embeddings_buffer)}", flush=True)
                
                # Write remaining items in buffer
                if train_embeddings_buffer:
                    all_train_embeddings = np.concatenate(train_embeddings_buffer, axis=0)
                    all_train_labels = np.concatenate(train_labels_buffer, axis=0)
                    start_idx = total_train_count
                    end_idx = total_train_count + len(all_train_embeddings)
                    train_file['embeddings'][start_idx:end_idx] = all_train_embeddings
                    train_file['labels'][start_idx:end_idx] = all_train_labels
                    total_train_count = end_idx
                
                print(f"num train embeddings: {total_train_count}, size of each embedding: {train_file['embeddings'].shape[1:]}")

        if not val_embeddings_path.exists():
            print(f"saving val embeddings to {val_embeddings_path}", flush=True)
            
            encoder = self.get_encoder_and_raw_loaders()
            val_batches_per_epoch = self.train_cfg.num_val_steps if 'num_val_steps' in self.train_cfg else len(self.val_loader)

            # Create HDF5 files for train and val embeddings
            embeddings_dir = Path(self.cfg.ft.embeddings_dir)
            if not embeddings_dir.exists():
                embeddings_dir.mkdir(parents=True, exist_ok=True)

            # Initialize HDF5 files with val embeddings
            with h5py.File(val_embeddings_path, 'w') as val_file:
                val_embeddings_buffer = []
                val_labels_buffer = []
                total_val_count = 0
                buffer_size = 100

                val_dataset_size = val_batches_per_epoch * self.train_cfg.batch_size if 'num_val_steps' in self.train_cfg else len(self.val_loader.dataset)
                if not isinstance(self.val_loader.dataset, IterableDataset):
                    val_dataset_size = min(val_dataset_size, len(self.val_loader.dataset))
                process = psutil.Process(os.getpid())
                
                for i, batch in tqdm(enumerate(self.val_loader), desc="val embeddings", total=val_batches_per_epoch):
                    if i >= val_batches_per_epoch:
                        break

                    enc_ctx, labels = self.inference_step(batch, encoder)

                    val_embeddings_buffer.append(enc_ctx.detach().cpu().numpy())
                    val_labels_buffer.append(labels.detach().cpu().numpy())
                    
                    if i == 0:
                        # Create datasets after first batch to know the shapes
                        embedding_shape = (val_dataset_size,) + enc_ctx.shape[1:]
                        label_shape = (val_dataset_size,) + labels.shape[1:]
                        val_file.create_dataset('embeddings', shape=embedding_shape, dtype=np.float32, chunks=True)
                        val_file.create_dataset('labels', shape=label_shape, dtype=np.float32, chunks=True)
                    
                    # Write to HDF5 when buffer is full
                    if len(val_embeddings_buffer) >= buffer_size:
                        all_val_embeddings = np.concatenate(val_embeddings_buffer, axis=0)
                        all_val_labels = np.concatenate(val_labels_buffer, axis=0)
                        start_idx = total_val_count
                        end_idx = total_val_count + len(all_val_embeddings)
                        val_file['embeddings'][start_idx:end_idx] = all_val_embeddings
                        val_file['labels'][start_idx:end_idx] = all_val_labels
                        total_val_count = end_idx
                        
                        # Memory monitoring after buffer write
                        current_memory = process.memory_info().rss / 1024 / 1024 / 1024 # GB
                        buffer_memory = sum(arr.nbytes for arr in val_embeddings_buffer) / 1024 / 1024 / 1024 # GB
                        print(f"Val buffer written at step {i}: Memory: {current_memory:.1f} GB, Buffer size: {buffer_memory:.1f} GB", flush=True)
                        
                        val_embeddings_buffer = []
                        val_labels_buffer = []
                    
                    # Monitor memory every 25 steps for validation (usually smaller)
                    if i % 50 == 0 and i > 0:
                        current_memory = process.memory_info().rss / 1024 / 1024 / 1024 # GB
                        buffer_memory = sum(arr.nbytes for arr in val_embeddings_buffer) / 1024 / 1024 / 1024 # GB
                        print(f"Val step {i}: Memory: {current_memory:.1f} GB, Buffer: {buffer_memory:.1f} GB, Buffer count: {len(val_embeddings_buffer)}", flush=True)

                # Write remaining items in buffer
                if val_embeddings_buffer:
                    all_val_embeddings = np.concatenate(val_embeddings_buffer, axis=0)
                    all_val_labels = np.concatenate(val_labels_buffer, axis=0)
                    start_idx = total_val_count
                    end_idx = total_val_count + len(all_val_embeddings)
                    val_file['embeddings'][start_idx:end_idx] = all_val_embeddings
                    val_file['labels'][start_idx:end_idx] = all_val_labels
                    total_val_count = end_idx

                print(f"num val embeddings: {total_val_count}, size of each embedding: {val_file['embeddings'].shape[1:]}")

            print(f"finished saving embeddings", flush=True)

            del encoder
            torch.cuda.empty_cache()
            gc.collect()
        
        if train_embeddings_path.exists():
            print(f"loading embeddings from {train_embeddings_path}", flush=True)
            
            # Add retry mechanism for file opening to handle temporary locking issues
            max_retries = 5
            retry_delay = 2  # seconds
            
            for attempt in range(max_retries):
                try:
                    # Load from HDF5 files using memory mapping for efficient memory usage
                    # Data is loaded on-demand when accessed, not all at once
                    train_file = h5py.File(train_embeddings_path, 'r', locking=False)
                    val_file = h5py.File(val_embeddings_path, 'r', locking=False)
                    
                    # Use memory mapping - don't load data into memory, just get references
                    embeddings = train_file['embeddings'][:]
                    all_labels = train_file['labels'][:]
                    val_embeddings = val_file['embeddings'][:]
                    val_labels = val_file['labels'][:]
                    
                    # Store file handles so they can be closed later
                    self._train_file = train_file
                    self._val_file = val_file
                    break  # Success, exit retry loop
                    
                except (BlockingIOError, OSError) as e:
                    print(f"Attempt {attempt + 1}/{max_retries} failed to open HDF5 files: {e}", flush=True)
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...", flush=True)
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise e
        
        return embeddings, all_labels, val_embeddings, val_labels
    
    def cleanup_embedding_files(self):
        """Close HDF5 file handles to free up system resources"""
        if hasattr(self, '_train_file'):
            self._train_file.close()
            delattr(self, '_train_file')
        if hasattr(self, '_val_file'):
            self._val_file.close()
            delattr(self, '_val_file')
    
    def __del__(self):
        """Destructor to ensure HDF5 files are closed"""
        self.cleanup_embedding_files()

# JEPA Finetuner
class JepaFinetuner(BaseFinetuner):
    def load_model(self):
        encoder, _, _ = get_model_and_loss_cnn(
            self.cfg.model.dims,
            self.cfg.model.num_res_blocks,
            self.cfg.dataset.num_frames,
            in_chans=self.cfg.dataset.num_chans if 'fields' not in self.cfg.ft else len(self.cfg.ft.fields),
        )
        if self.trained_model_path is not None:
            print(f"loading state dict from {self.trained_model_path}", flush=True)
            state_dict = torch.load(self.trained_model_path)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            encoder.load_state_dict(state_dict)
        else:
            print(f"no pretrained model path provided, randomly initializing encoder", flush=True)
        
        encoder.eval()
        return encoder
    
    def create_head(self, metadata):
        embed_dim = self.cfg.model.dims[-1]

        if self.cfg.ft.get("use_attentive_pooling", False):
            # Use attentive pooling
            if self.cfg.ft.task == "regression":
                head = AttentiveClassifier(
                    embed_dim=embed_dim,
                    num_classes=len(metadata.constant_scalar_names),
                    num_heads=8, # needs to be divisible by embed_dim
                    mlp_ratio=12.0,
                    dropout_rate=self.cfg.ft.get("dropout_rate", 0.0)
                )
            elif "classification" in self.cfg.ft.task:
                head = AttentiveClassifier(
                    embed_dim=embed_dim,
                    num_classes=self.cfg.ft.num_classes,
                    num_heads=8, # needs to be divisible by embed_dim
                    mlp_ratio=12.0,
                    dropout_rate=self.cfg.ft.get("dropout_rate", 0.0)
                )
        else:
            # Use traditional pooling
            if self.cfg.ft.task == "regression":
                if self.cfg.ft.head_type == "linear":
                    head = RegressionHead(
                        in_dim=embed_dim,
                        out_dim=len(metadata.constant_scalar_names),
                        flatten_first=True
                    )
                elif self.cfg.ft.head_type == "mlp":
                    head = RegressionMLP(
                        in_dim=embed_dim,
                        out_dim=len(metadata.constant_scalar_names),
                        flatten_first=True,
                        add_dropout=self.cfg.ft.get("add_dropout", False),
                        dropout_rate=self.cfg.ft.get("dropout_rate", 0.2)
                    )
            elif "classification" in self.cfg.ft.task:
                head = RegressionHead(
                    in_dim=embed_dim,
                    out_dim=self.cfg.ft.num_classes,
                    flatten_first=True,
                    add_dropout=self.cfg.ft.get("add_dropout", False),
                    dropout_rate=self.cfg.ft.get("dropout_rate", 0.8)
                )
        return head

    def _model_inference(self, ctx, encoder):
        with torch.no_grad():
            enc_ctx = encoder(ctx)
            if self.cfg.ft.get("use_attentive_pooling", False):
                # reshape to (batch_size, num_tokens, embed_dim)
                enc_ctx = rearrange(enc_ctx, 'b c h w -> b (h w) c')
            # Check for NaN values in the encoded context
            if torch.isnan(enc_ctx).any():
                raise ValueError(f"NaN values detected in encoded context. Shape: {enc_ctx.shape}, NaN count: {torch.isnan(enc_ctx).sum()}")
        return enc_ctx


# VideoMAE Finetuner
class VideoMAEFinetuner(BaseFinetuner):
    def load_model(self):
        if self.trained_model_path is not None:
            model_config = json.load(open(Path(self.trained_model_path).parent / "config.json"))
            model_arch = model_config["model"] 
        else:
            model_arch = 'pretrain_videomae_small_patch16_224'

        model_functions = {
            'pretrain_videomae_small_patch16_224': vit_small_patch16_224,
            'pretrain_videomae_base_patch16_224': vit_base_patch16_224,
            'pretrain_videomae_large_patch16_224': vit_large_patch16_224,
            'pretrain_videomae_huge_patch16_224': vit_huge_patch16_224,
        }
        
        if model_arch not in model_functions:
            raise ValueError(f"Unknown model: {model_arch}")
        
        # Create encoder (without classification head)
        encoder = model_functions[model_arch](
            pretrained=False,
            drop_path_rate=self.cfg.ft.get("drop_path_rate", 0.0),
            in_chans=self.cfg.dataset.num_chans if 'fields' not in self.cfg.ft else len(self.cfg.ft.fields),
            all_frames=self.cfg.dataset.num_frames,
            num_classes=0,
            use_mean_pooling=False,
        )

        if self.trained_model_path is not None:
            print(f"Loading pretrained weights from: {self.trained_model_path}")
    
            checkpoint = torch.load(self.trained_model_path, map_location='cpu')
            pretrained_state_dict = checkpoint['model']
            
            # Filter encoder weights (exclude decoder and MAE-specific components)
            encoder_state_dict = {}
            for key, value in pretrained_state_dict.items():
                if key.startswith('encoder.'):
                    # Remove 'encoder.' prefix
                    new_key = key[8:]  # Remove 'encoder.' prefix
                    if new_key in encoder.state_dict():
                        encoder_state_dict[new_key] = value
            
            # Load encoder weights
            missing_keys, unexpected_keys = encoder.load_state_dict(encoder_state_dict, strict=False)
            
            print(f"Loaded {len(encoder_state_dict)} encoder layers")
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")
        else:
            print(f"no pretrained model path provided, randomly initializing encoder", flush=True)

        for param in encoder.parameters():
            param.requires_grad = False

        encoder.eval()
        return encoder

    def create_head(self, metadata):
        if self.trained_model_path is not None:
            model_config = json.load(open(Path(self.trained_model_path).parent / "config.json"))
            model_arch = model_config["model"] 
        else:
            model_arch = 'pretrain_videomae_small_patch16_224'

        embed_dim = {
            'pretrain_videomae_small_patch16_224': 384,
            'pretrain_videomae_base_patch16_224': 768,
            'pretrain_videomae_large_patch16_224': 1024,
            'pretrain_videomae_huge_patch16_224': 1280,
        }[model_arch]

        if self.cfg.ft.get("use_attentive_pooling", False):
            # Use attentive pooling over patch embeddings
            if self.cfg.ft.task == "regression":
                head = AttentiveClassifier(
                    embed_dim=embed_dim,
                    num_classes=len(metadata.constant_scalar_names),
                    # use defaults for other values
                    num_heads=8, # needs to be divisible by embed_dim
                    mlp_ratio=0.25,
                    dropout_rate=self.cfg.ft.get("dropout_rate", 0.0)
                )
            elif "classification" in self.cfg.ft.task:
                head = AttentiveClassifier(
                    embed_dim=embed_dim,
                    num_classes=self.cfg.ft.num_classes,
                    num_heads=8, # needs to be divisible by embed_dim
                    mlp_ratio=0.25,
                    dropout_rate=self.cfg.ft.get("dropout_rate", 0.0)
                )
        else:
            # Use traditional pooling with CLS token
            if self.cfg.ft.task == "regression":
                if self.cfg.ft.head_type == "linear":
                    head = RegressionHead(
                        in_dim=embed_dim,
                        out_dim=len(metadata.constant_scalar_names),
                        flatten_first=True
                    )
                elif self.cfg.ft.head_type == "mlp":
                    head = RegressionMLP(
                        in_dim=embed_dim,
                        out_dim=len(metadata.constant_scalar_names),
                        flatten_first=True,
                        add_dropout=self.cfg.ft.get("add_dropout", False),
                        dropout_rate=self.cfg.ft.get("dropout_rate", 0.2)
                    )
            elif "classification" in self.cfg.ft.task:
                head = RegressionHead(
                    in_dim=embed_dim,
                    out_dim=self.cfg.ft.num_classes,
                    flatten_first=True,
                )
            else:
                raise ValueError(f"task {self.cfg.ft.task} not supported for VideoMAE finetuning")
        return head
    
    def _model_inference(self, ctx, encoder):
        with torch.no_grad():
            if self.cfg.ft.get("use_attentive_pooling", False):
                # Get patch embeddings for attentive pooling
                patch_embeddings = encoder.get_patch_embeddings(ctx)
                return patch_embeddings
            else:
                # Return CLS token for traditional pooling
                cls_token = encoder.forward_features(ctx)  # (B, embed_dim) - already the CLS token
                return cls_token
 
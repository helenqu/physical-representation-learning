"""Frozen-encoder evaluation: linear probe + kNN.

Standalone alternative to finetuner.py's attentive-probe path. The existing
attentive-probe flow in finetuner.py is untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .data import get_dataset
from .model import get_model_and_loss_cnn
from .utils.hydra import compose


PARAM_NAMES = ["zeta", "alpha"]  # matches data.py:659 metadata.constant_scalar_names


class FrozenEvaluator:
    def __init__(self, cfg, checkpoint_path: str):
        self.cfg = cfg
        self.checkpoint_path = checkpoint_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval_mode = cfg.ft.eval_mode
        if self.eval_mode == "attentive":
            raise SystemExit("use physics_jepa.finetune for attentive mode")
        if self.eval_mode not in ("linear", "knn", "linear_and_knn"):
            raise ValueError(f"unknown eval_mode: {self.eval_mode}")

        self.feature_pool = cfg.ft.get("feature_pool", "gap")
        assert self.feature_pool in ("gap", "flatten")

        self.cache_dir = Path(cfg.ft.feature_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = Path(cfg.ft.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def load_encoder(self) -> nn.Module:
        encoder, _, _ = get_model_and_loss_cnn(
            self.cfg.model.dims,
            self.cfg.model.num_res_blocks,
            self.cfg.dataset.num_frames,
            in_chans=self.cfg.dataset.num_chans,
        )
        print(f"loading encoder state dict from {self.checkpoint_path}", flush=True)
        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        encoder.load_state_dict(state_dict)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
        assert all(not p.requires_grad for p in encoder.parameters()), \
            "encoder params must be frozen"
        return encoder.to(self.device)

    def _cache_key(self, split: str) -> str:
        s = "|".join([
            str(Path(self.checkpoint_path).resolve()),
            split,
            self.cfg.dataset.name,
            str(self.cfg.dataset.num_frames),
            str(self.cfg.dataset.get("resolution", None)),
            self.feature_pool,
        ])
        return hashlib.sha1(s.encode()).hexdigest()[:16]

    def _make_loader(self, split: str) -> DataLoader:
        dataset = get_dataset(
            self.cfg.dataset.name,
            self.cfg.dataset.num_frames,
            split=split,
            include_labels=True,
            resolution=self.cfg.dataset.get("resolution", None),
            offset=self.cfg.dataset.get("offset", None),
            noise_std=0.0,
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.ft.batch_size,
            shuffle=False,
            num_workers=self.cfg.ft.get("num_workers", 4),
            pin_memory=True,
            drop_last=False,
        )

    def _pool(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, C, H, W)
        if self.feature_pool == "gap":
            return z.mean(dim=(-2, -1))
        return z.flatten(1)

    def extract_features(self, encoder: nn.Module, split: str) -> Dict[str, torch.Tensor]:
        cache_path = self.cache_dir / f"{self._cache_key(split)}.pt"
        if cache_path.exists():
            print(f"loading cached features from {cache_path}", flush=True)
            return torch.load(cache_path)

        print(f"extracting features for split={split}", flush=True)
        loader = self._make_loader(split)
        feats, raw_labels = [], []
        for batch in tqdm(loader, desc=f"encode {split}"):
            ctx = batch["context"].to(self.device, non_blocking=True)
            if ctx.shape[2] < 4:
                ctx = F.pad(ctx, (0, 0, 0, 0, 0, 4 - ctx.shape[2]))
            with torch.no_grad():
                z = encoder(ctx)
                if torch.isnan(z).any():
                    raise ValueError(f"NaN in encoder output: split={split}")
                z = self._pool(z)
            feats.append(z.float().cpu())
            raw_labels.append(batch["physical_params"].float())

        out = {
            "features": torch.cat(feats, dim=0),
            "labels_raw": torch.cat(raw_labels, dim=0),
        }
        torch.save(out, cache_path)
        print(
            f"cached {split}: features={tuple(out['features'].shape)} "
            f"labels={tuple(out['labels_raw'].shape)} -> {cache_path}",
            flush=True,
        )
        return out

    def _init_wandb(self):
        if self.cfg.get("dry_run", False):
            self._wandb_on = False
            return
        run_name = self.cfg.ft.get("run_name") or (
            f"{self.cfg.dataset.name}-frozen-{self.eval_mode}-"
            f"{Path(self.checkpoint_path).stem}"
        )
        wandb.init(
            project="physics-jepa",
            name=run_name,
            config=OmegaConf.to_container(self.cfg),
        )
        self._wandb_on = True

    def run(self):
        self._init_wandb()
        encoder = self.load_encoder()
        train_f = self.extract_features(encoder, "train")
        val_all = self.extract_features(encoder, "val")
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        N = val_all["features"].shape[0]
        g = torch.Generator().manual_seed(self.cfg.seed)
        perm = torch.randperm(N, generator=g)
        half = N // 2
        val_f = {k: v[perm[:half]] for k, v in val_all.items()}
        test_f = {k: v[perm[half:]] for k, v in val_all.items()}

        mean = train_f["labels_raw"].mean(0)
        std = train_f["labels_raw"].std(0).clamp_min(1e-8)
        for s in (train_f, val_f, test_f):
            s["labels"] = (s["labels_raw"] - mean) / std

        print(
            f"label z-score: mean={mean.tolist()} std={std.tolist()}",
            flush=True,
        )
        print(
            f"train={tuple(train_f['features'].shape)} "
            f"val={tuple(val_f['features'].shape)} "
            f"test={tuple(test_f['features'].shape)}",
            flush=True,
        )

        results: List[Dict] = []
        if self.eval_mode in ("linear", "linear_and_knn"):
            results += self.run_linear(train_f, val_f, test_f)
        return results

    # ------------------------------------------------------------------ linear
    @staticmethod
    def _per_param_mse(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        err = (pred - target) ** 2
        per = err.mean(dim=0).tolist()
        out = {f"mse_{PARAM_NAMES[i]}": float(per[i]) for i in range(len(PARAM_NAMES))}
        out["mse_mean"] = float(err.mean().item())
        return out

    def run_linear(self, train_f, val_f, test_f) -> List[Dict]:
        lin_cfg = self.cfg.ft.linear
        in_dim = train_f["features"].shape[1]
        out_dim = train_f["labels"].shape[1]
        head = nn.Linear(in_dim, out_dim, bias=lin_cfg.get("bias", True)).to(self.device)

        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=lin_cfg.lr,
            weight_decay=lin_cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=lin_cfg.num_epochs,
            eta_min=lin_cfg.get("min_lr", 0.0),
        )
        loss_fn = nn.MSELoss()

        train_loader = DataLoader(
            TensorDataset(train_f["features"], train_f["labels"]),
            batch_size=lin_cfg.batch_size,
            shuffle=True,
            drop_last=False,
        )

        val_x = val_f["features"].to(self.device)
        val_y = val_f["labels"].to(self.device)

        best_val = float("inf")
        best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        for epoch in range(lin_cfg.num_epochs):
            head.train()
            epoch_losses = []
            for x, y in train_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                pred = head(x)
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            scheduler.step()

            head.eval()
            with torch.no_grad():
                val_pred = head(val_x)
                val_metrics = self._per_param_mse(val_pred, val_y)
            train_mse = float(np.mean(epoch_losses))
            log = {
                "linear/epoch": epoch,
                "linear/train_mse": train_mse,
                "linear/lr": scheduler.get_last_lr()[0],
                **{f"linear/val_{k}": v for k, v in val_metrics.items()},
            }
            if self._wandb_on:
                wandb.log(log)
            if epoch % max(1, lin_cfg.num_epochs // 20) == 0 or epoch == lin_cfg.num_epochs - 1:
                print(
                    f"[linear] epoch {epoch:03d} train_mse={train_mse:.4f} "
                    f"val_mse_mean={val_metrics['mse_mean']:.4f}",
                    flush=True,
                )
            if val_metrics["mse_mean"] < best_val:
                best_val = val_metrics["mse_mean"]
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

        ckpt_path = self.out_dir / "linear_best.pt"
        torch.save(best_state, ckpt_path)
        print(f"[linear] best val mse_mean={best_val:.4f} saved to {ckpt_path}", flush=True)

        head.load_state_dict(best_state)
        head.eval()
        out: List[Dict] = []
        with torch.no_grad():
            for name, f in (("val", val_f), ("test", test_f)):
                x = f["features"].to(self.device)
                y = f["labels"].to(self.device)
                metrics = self._per_param_mse(head(x), y)
                out.append({"probe_type": "linear", "k": None, "metric": None,
                            "split": name, **metrics})
        return out


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg = compose(args.config, args.overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.seed = args.seed
    cfg.dry_run = args.dry_run
    print(OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.model.objective != "jepa":
        raise ValueError(
            f"eval_frozen currently supports jepa only, got {cfg.model.objective}"
        )

    evaluator = FrozenEvaluator(cfg, checkpoint_path=args.checkpoint)
    evaluator.run()


if __name__ == "__main__":
    main()

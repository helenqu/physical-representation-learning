"""Frozen-encoder evaluation: linear probe + kNN.

Standalone alternative to finetuner.py's attentive-probe path. The existing
attentive-probe flow in finetuner.py is untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import get_dataset
from .model import get_model_and_loss_cnn
from .utils.hydra import compose


PARAM_NAMES = ["alpha", "zeta"]  # scalars are sorted alphabetically in data.py:_build_index


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
            results += self.run_linear(train_f, val_f, test_f, mean=mean, std=std)
        if self.eval_mode in ("knn", "linear_and_knn"):
            results += self.run_knn(train_f, val_f, test_f, mean=mean, std=std)

        self._report(results, mean=mean.tolist(), std=std.tolist())
        if self._wandb_on:
            wandb.finish()
        return results

    def _report(self, results: List[Dict], mean, std):
        header = [
            "probe_type", "k", "metric", "split",
            "mse_alpha", "mse_zeta", "mse_mean",
            "mse_alpha_raw", "mse_zeta_raw", "mse_mean_raw",
        ]
        widths = [10, 5, 10, 5, 10, 10, 10, 14, 14, 14]

        def fmt(v, w):
            if v is None:
                return f"{'-':<{w}}"
            if isinstance(v, float):
                return f"{v:<{w}.4f}"
            return f"{str(v):<{w}}"

        line = "  ".join(f"{h:<{w}}" for h, w in zip(header, widths))
        sep = "-" * len(line)
        print()
        print(sep)
        print(line)
        print(sep)
        for r in results:
            row = [r.get(h) for h in header]
            print("  ".join(fmt(v, w) for v, w in zip(row, widths)))
        print(sep)

        payload = {
            "checkpoint": str(Path(self.checkpoint_path).resolve()),
            "dataset": self.cfg.dataset.name,
            "eval_mode": self.eval_mode,
            "feature_pool": self.feature_pool,
            "label_order": PARAM_NAMES,
            "label_mean": mean,
            "label_std": std,
            "results": results,
        }
        out_path = self.out_dir / "results.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"results saved to {out_path}", flush=True)

    # ------------------------------------------------------------------ linear
    @staticmethod
    def _per_param_mse(
        pred_n: torch.Tensor,
        target_n: torch.Tensor,
        mean: torch.Tensor = None,
        std: torch.Tensor = None,
    ) -> Dict[str, float]:
        err_n = (pred_n - target_n) ** 2
        per_n = err_n.mean(dim=0).tolist()
        out = {f"mse_{PARAM_NAMES[i]}": float(per_n[i]) for i in range(len(PARAM_NAMES))}
        out["mse_mean"] = float(err_n.mean().item())
        if mean is not None and std is not None:
            pred_raw = pred_n * std + mean
            target_raw = target_n * std + mean
            err_raw = (pred_raw - target_raw) ** 2
            per_raw = err_raw.mean(dim=0).tolist()
            for i in range(len(PARAM_NAMES)):
                out[f"mse_{PARAM_NAMES[i]}_raw"] = float(per_raw[i])
            out["mse_mean_raw"] = float(err_raw.mean().item())
        return out

    def run_linear(self, train_f, val_f, test_f, mean, std) -> List[Dict]:
        """Closed-form least-squares linear probe on normalized targets."""
        lin_cfg = self.cfg.ft.linear
        use_bias = lin_cfg.get("bias", True)
        in_dim = train_f["features"].shape[1]
        out_dim = train_f["labels"].shape[1]

        def design(X):
            if not use_bias:
                return X
            return torch.cat([X, torch.ones(X.size(0), 1, dtype=X.dtype)], dim=1)

        X_tr = train_f["features"].cpu()
        Y_tr = train_f["labels"].cpu()
        A = design(X_tr)
        W = torch.linalg.lstsq(A, Y_tr).solution  # (in_dim [+1], out_dim)

        # Save as a materialized nn.Linear for symmetry with future resumability.
        head = nn.Linear(in_dim, out_dim, bias=use_bias)
        with torch.no_grad():
            if use_bias:
                head.weight.copy_(W[:-1].T)
                head.bias.copy_(W[-1])
            else:
                head.weight.copy_(W.T)
        torch.save(head.state_dict(), self.out_dir / "linear_best.pt")

        def predict(X):
            X = X.cpu()
            if use_bias:
                return X @ W[:-1] + W[-1]
            return X @ W

        # Report on train, val, test. Train included so user can sanity-check that
        # lstsq's train MSE is the global minimum (and ≤ val/test MSE).
        out: List[Dict] = []
        for name, f in (("train", train_f), ("val", val_f), ("test", test_f)):
            pred_n = predict(f["features"])
            metrics = self._per_param_mse(pred_n, f["labels"].cpu(), mean=mean, std=std)
            out.append({"probe_type": "linear", "k": None, "metric": None,
                        "split": name, **metrics})
            print(
                f"[linear] split={name:5s} "
                f"mse_mean={metrics['mse_mean']:.4f} "
                f"mse_mean_raw={metrics['mse_mean_raw']:.4f}",
                flush=True,
            )
            if self._wandb_on and name in ("val", "test"):
                wandb.log({
                    f"linear/{name}/mse_mean": metrics["mse_mean"],
                    f"linear/{name}/mse_mean_raw": metrics["mse_mean_raw"],
                    **{f"linear/{name}/mse_{p}": metrics[f"mse_{p}"] for p in PARAM_NAMES},
                })
        return out

    # --------------------------------------------------------------------- knn
    def run_knn(self, train_f, val_f, test_f, mean, std) -> List[Dict]:
        from sklearn.neighbors import KNeighborsRegressor

        knn_cfg = self.cfg.ft.knn
        x_tr = train_f["features"].numpy()
        y_tr = train_f["labels"].numpy()
        x_va = val_f["features"].numpy()
        y_va = val_f["labels"].numpy()
        x_te = test_f["features"].numpy()
        y_te = test_f["labels"].numpy()

        out: List[Dict] = []
        best = None
        for metric in knn_cfg.metrics:
            algorithm = "brute" if metric == "cosine" else "auto"
            for k in knn_cfg.ks:
                if k > len(x_tr):
                    print(f"[knn] skipping k={k} > n_train={len(x_tr)}", flush=True)
                    continue
                model = KNeighborsRegressor(
                    n_neighbors=k, metric=metric, algorithm=algorithm, n_jobs=-1,
                )
                model.fit(x_tr, y_tr)
                for name, x, y in (("val", x_va, y_va), ("test", x_te, y_te)):
                    pred = torch.from_numpy(model.predict(x))
                    target = torch.from_numpy(y)
                    metrics = self._per_param_mse(pred, target, mean=mean, std=std)
                    row = {"probe_type": "knn", "k": k, "metric": metric,
                           "split": name, **metrics}
                    out.append(row)
                    if name == "val":
                        if self._wandb_on:
                            wandb.log({f"knn/{metric}/k{k}_val_mse_mean": metrics["mse_mean"]})
                        if best is None or metrics["mse_mean"] < best["val_mse_mean"]:
                            best = {
                                "k": k, "metric": metric,
                                "val_mse_mean": metrics["mse_mean"],
                            }
                    print(
                        f"[knn] k={k:3d} metric={metric:9s} split={name:4s} "
                        f"mse_mean={metrics['mse_mean']:.4f}",
                        flush=True,
                    )

        if best is not None:
            # Add a knn_best row for val and test using the best val config
            for r in out:
                if (
                    r["probe_type"] == "knn"
                    and r["k"] == best["k"]
                    and r["metric"] == best["metric"]
                ):
                    out.append({**r, "probe_type": "knn_best"})
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

import argparse
from pathlib import Path
from omegaconf import OmegaConf

from .finetuner import JepaFinetuner, VideoMAEFinetuner
from .utils.hydra import compose

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default=f"{Path(__file__).parent.parent}/configs/train_grayscott.yml")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--trained_model_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = compose(args.config, args.overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.dry_run = args.dry_run
    cfg.seed = args.seed

    print(OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.model.objective == "jepa":
        finetuner = JepaFinetuner(cfg, trained_model_path=args.trained_model_path)
    elif cfg.model.objective == "videomae":
        finetuner = VideoMAEFinetuner(cfg, trained_model_path=args.trained_model_path)
    else:
        raise ValueError(f"Unknown objective: {cfg.model.objective}")

    finetuner.train()
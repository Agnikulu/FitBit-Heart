# src/config.py

import argparse
from pathlib import Path

from omegaconf import OmegaConf, DictConfig


def load_config() -> DictConfig:
    """
    Load a YAML config via `--config path/to.yaml` plus any number of
    dotlist overrides (like Hydra).

    Usage:
      python -m src.pretrain --config configs/pretrain.yaml \
            model.attn_heads=2 pretrain.lr=5e-4

    Returns:
      OmegaConf DictConfig with interpolation resolved.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="path to your YAML config file"
    )
    # take the rest as overrides
    args, overrides = parser.parse_known_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        parser.error(f"Config file not found: {cfg_path}")

    # 1) Load the main YAML
    cfg: DictConfig = OmegaConf.load(str(cfg_path))

    # 2) Apply CLI overrides (dotlist)
    if overrides:
        dotlist = [ov for ov in overrides if not ov.startswith("-")]
        if dotlist:
            oc_override = OmegaConf.from_dotlist(dotlist)
            cfg = OmegaConf.merge(cfg, oc_override)

    # 3) Resolve interpolations
    OmegaConf.resolve(cfg)

    return cfg
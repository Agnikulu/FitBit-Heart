"""Hierarchical YAML loader (Hydra‑style `defaults:`) + CLI overrides."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import Set
from omegaconf import OmegaConf, DictConfig

# ──────────────────────────────────────────────────────────────────────

_DEF_KEY = "defaults"

def _load_recursive(path: Path, seen: Set[Path] | None = None) -> DictConfig:
    """Depth‑first load with cycle detection."""
    seen = seen or set()
    path = path.resolve()
    if path in seen:
        raise RuntimeError(f"Cyclic config reference: {path}")
    seen.add(path)

    cfg: DictConfig = OmegaConf.load(str(path))
    parents = cfg.pop(_DEF_KEY, [])
    if parents:
        merged = OmegaConf.create()
        for item in parents:
            if isinstance(item, str):
                p = (path.parent / f"{item}.yaml").resolve()
            else:  # dict syntax – rarely used, but we support it
                k, v = next(iter(item.items()))
                p = (path.parent / k / f"{v}.yaml").resolve()
            merged = OmegaConf.merge(merged, _load_recursive(p, seen))
        cfg = OmegaConf.merge(merged, cfg)
    return cfg


def load_config() -> DictConfig:
    """Entry‑point used by every script."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-c", "--config", required=True, help="YAML experiment file")
    args, overrides = parser.parse_known_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        sys.exit(f"Config not found: {cfg_path}")

    cfg = _load_recursive(cfg_path)
    if overrides:  # CLI dot‑overrides
        dotlist = [x for x in overrides if not x.startswith("-")]
        if dotlist:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    OmegaConf.resolve(cfg)
    return cfg
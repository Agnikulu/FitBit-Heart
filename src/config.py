"""
YAML Loader.

Usage examples
--------------
python -m src.pretrain --config configs/pretrain.yaml device=cpu
python -m src.train    --config configs/train.yaml   train.lr=5e-4
"""
from __future__ import annotations
import sys, pathlib, yaml
from types import SimpleNamespace

def _to_ns(d: dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _to_ns(v) if isinstance(v, dict) else v)
    return ns

def _cast(s: str):
    if s.lower() in {"true","false"}: return s.lower()=="true"
    try: return int(s)
    except ValueError:
        try: return float(s)
        except ValueError: return s

def _set(d: dict, dotted: str, v):
    ks=dotted.split(".")
    for k in ks[:-1]: d=d.setdefault(k,{})
    d[ks[-1]]=v

def _load(p: pathlib.Path):
    with p.open() as f: cfg=yaml.safe_load(f) or {}
    for inc in cfg.pop("defaults", []):
        parent=_load(p.parent/f"{inc}.yaml"); parent.update(cfg); cfg=parent
    return cfg

def load_config():
    argv=sys.argv[1:]
    if len(argv)<2 or argv[0] not in {"-c","--config"}:
        raise SystemExit("Usage: ... --config cfg.yaml [key=value ...]")
    cfg=_load(pathlib.Path(argv[1]))
    for ov in argv[2:]:
        k,v=ov.split("=",1); _set(cfg,k,_cast(v))
    return _to_ns(cfg)

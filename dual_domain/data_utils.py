"""Dual-domain dataset yaml schema and dataset builder.

Yaml schema (sibling of ultralytics' own flat train/val schema):

    path: /data/aerial
    source:                     # domain the model is already pretrained on
      train: source/images/train
      val: source/images/val
    target:                     # domain being feature-aligned + fine-tuned (the bigger set)
      train: target/images/train
      val: target/images/val
    names: [vehicle, person]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from ultralytics.utils import IterableSimpleNamespace, colorstr

from .dataset import DualDomainYOLODataset

_DOMAINS = ("source", "target")
_SUBSETS = ("train", "val")


def check_domain_adapt_dataset(data: str | Path) -> dict[str, Any]:
    """Validate and resolve a dual-domain dataset yaml into a data dict.

    The returned dict also carries top-level `train`/`val` keys pointing at the target
    domain's paths, purely so it satisfies ultralytics internals that index
    `self.data["train"]` / `self.data.get("val")` directly; `build_dual_domain_yolo_dataset`
    ignores those and reads `data["source"][mode]` / `data["target"][mode]` itself.
    """
    path = Path(data)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    for domain in _DOMAINS:
        if domain not in raw:
            raise SyntaxError(f"Dual-domain dataset yaml must define an '{domain}:' key with train/val paths ({path}).")

    root = Path(raw.get("path", path.parent))
    resolved: dict[str, Any] = {"channels": raw.get("channels", 3)}

    names = raw["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    resolved["names"] = names
    resolved["nc"] = len(names)

    for domain in _DOMAINS:
        domain_cfg = raw[domain]
        for subset in _SUBSETS:
            if subset not in domain_cfg:
                raise SyntaxError(f"Dual-domain dataset yaml '{domain}:' block is missing '{subset}:' ({path}).")
        resolved[domain] = {subset: str((root / domain_cfg[subset]).resolve()) for subset in domain_cfg}

    resolved["train"] = resolved["target"]["train"]
    resolved["val"] = resolved["target"]["val"]
    return resolved


def build_dual_domain_yolo_dataset(
    cfg: IterableSimpleNamespace,
    data: dict[str, Any],
    mode: str = "train",
    batch: int | None = None,
    stride: int = 32,
) -> DualDomainYOLODataset:
    """Build a DualDomainYOLODataset for 'train' or 'val' mode."""
    common_kwargs = dict(
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=cfg.rect or mode == "val",
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )
    return DualDomainYOLODataset(
        img_path=data["source"][mode],
        target_dataset_kwargs={**common_kwargs, "img_path": data["target"][mode]},
        **common_kwargs,
    )

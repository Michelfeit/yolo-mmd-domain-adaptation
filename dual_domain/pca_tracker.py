"""Fixed-basis PCA tracker for backbone feature-space movement across training.

Fits a PCA basis once, on a fixed sample of source + target features taken before any
dual-domain training happens, then re-projects the SAME fixed sample of images through
that frozen basis at the end of every epoch. Movement in the resulting plot reflects the
features moving under training, not the basis shifting under them.

Reuses the same hooked layer the model's MMD term already reads (`model._captured_features`)
rather than registering a second hook. Purely diagnostic — not wired into the trainer by
default; see `callbacks.attach_pca_tracker` to opt in.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.torch_utils import unwrap_model


def _gap(features: torch.Tensor) -> torch.Tensor:
    return features.mean(dim=tuple(range(2, features.ndim)))


class FeaturePCATracker:
    def __init__(
        self,
        data: dict[str, Any],
        cfg: Any,
        csv_path: str | Path,
        n_samples_per_domain: int = 64,
        n_components: int = 2,
        extract_batch_size: int = 16,
    ):
        self.csv_path = Path(csv_path)
        self.n_components = n_components
        self.extract_batch_size = extract_batch_size
        self._source_ds = self._build_dataset(data, cfg, data["source"]["train"], "pca-source: ")
        self._target_ds = self._build_dataset(data, cfg, data["target"]["train"], "pca-target: ")
        self._source_indices = list(range(min(n_samples_per_domain, len(self._source_ds))))
        self._target_indices = list(range(min(n_samples_per_domain, len(self._target_ds))))
        self._pca: PCA | None = None

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "domain", "sample_index", *(f"pc{i + 1}" for i in range(n_components))])

    @staticmethod
    def _build_dataset(data: dict[str, Any], cfg: Any, img_path: str, prefix: str) -> YOLODataset:
        return YOLODataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=1,
            augment=False,  # fixed, unaugmented images so movement reflects training, not augmentation noise
            hyp=cfg,
            rect=False,
            cache=None,
            single_cls=False,
            stride=32,
            pad=0.5,
            prefix=prefix,
            task=cfg.task,
            classes=cfg.classes,
            data=data,
            fraction=1.0,
        )

    @torch.no_grad()
    def _extract(self, model: torch.nn.Module, dataset: YOLODataset, indices: list[int], device: torch.device) -> np.ndarray:
        model = unwrap_model(model)
        was_training = model.training
        model.eval()
        try:
            feats = []
            for start in range(0, len(indices), self.extract_batch_size):
                chunk = indices[start : start + self.extract_batch_size]
                imgs = torch.stack([dataset[i]["img"] for i in chunk]).to(device).float() / 255
                model.predict(imgs)
                feats.append(_gap(model._captured_features).cpu().numpy())
            return np.concatenate(feats, axis=0)
        finally:
            model.train(was_training)

    def fit_initial(self, model: torch.nn.Module, device: torch.device) -> None:
        source_feats = self._extract(model, self._source_ds, self._source_indices, device)
        target_feats = self._extract(model, self._target_ds, self._target_indices, device)
        self._pca = PCA(n_components=self.n_components)
        self._pca.fit(np.concatenate([source_feats, target_feats], axis=0))
        self._log(0, source_feats, target_feats)

    def snapshot(self, epoch: int, model: torch.nn.Module, device: torch.device) -> None:
        if self._pca is None:
            raise RuntimeError("fit_initial() must be called before snapshot().")
        source_feats = self._extract(model, self._source_ds, self._source_indices, device)
        target_feats = self._extract(model, self._target_ds, self._target_indices, device)
        self._log(epoch, source_feats, target_feats)

    def _log(self, epoch: int, source_feats: np.ndarray, target_feats: np.ndarray) -> None:
        source_proj = self._pca.transform(source_feats)
        target_proj = self._pca.transform(target_feats)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for domain, proj in (("source", source_proj), ("target", target_proj)):
                for idx, coords in enumerate(proj):
                    writer.writerow([epoch, domain, idx, *coords.tolist()])

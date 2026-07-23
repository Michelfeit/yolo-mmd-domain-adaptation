"""DetectionModel + an MMD alignment term between hooked backbone features.

Detection loss is computed only on the target-domain batch (the bigger set, being
fine-tuned). The source domain (already pretrained on) is forward-only; its backbone
features are the reference the target domain's features are pulled toward via MMD.
"""

from __future__ import annotations

from typing import Any

import torch
from ultralytics.nn.tasks import DetectionModel

from .config import MMDConfig
from .mmd import build_mmd


class DualDomainDetectionModel(DetectionModel):
    def __init__(self, *args: Any, mmd_cfg: MMDConfig | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.mmd_cfg = mmd_cfg or MMDConfig()
        self.mmd_weight = self.mmd_cfg.mmd_weight
        self._mmd = build_mmd(self.mmd_cfg.kernel, self.mmd_cfg.preprocess, self.mmd_cfg.momentum)
        self._captured_features: torch.Tensor | None = None
        # Registered once; the buffer above is simply overwritten on every forward pass,
        # so there's no per-epoch re-registration or manual reset bookkeeping.
        self.model[self.mmd_cfg.mmd_target_layer].register_forward_hook(self._capture_hook)

    def _capture_hook(self, module: torch.nn.Module, inputs: Any, output: torch.Tensor) -> None:
        self._captured_features = output

    def loss(self, batch: dict[str, Any], preds: Any = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if "domain_target" not in batch:
            # Plain single-domain call (e.g. stock validation forward). Pad a zero
            # "mmd_distance" entry so loss_items has the same keys as the dual-domain
            # path below — BaseValidator preallocates its running-loss accumulator
            # from this dict's keys and would otherwise KeyError on later batches.
            loss, loss_items = super().loss(batch, preds)
            loss_items = {**loss_items, "mmd_distance": loss.new_zeros(())}
            return torch.cat([loss.view(-1), loss.new_zeros(1)]), loss_items

        loss, loss_items = super().loss(batch["domain_target"], preds)
        feat_target = self._captured_features

        if self.mmd_cfg.detach_source_features:
            with torch.no_grad():
                self.predict(batch["domain_source"]["img"])
            feat_source = self._captured_features.detach()
        else:
            self.predict(batch["domain_source"]["img"])
            feat_source = self._captured_features

        mmd, distance = self._mmd(feat_source, feat_target)
        loss = torch.cat([loss.view(-1), (self.mmd_weight * mmd).view(1)])
        loss_items = {**loss_items, "mmd_distance": distance.detach()}
        return loss, loss_items

"""DetectionModel + an MMD alignment term between hooked backbone features.

Detection loss is normally computed only on the target-domain batch (the bigger set,
being fine-tuned); the source domain (already pretrained on) is forward-only, and its
backbone features are the reference the target domain's features are pulled toward via
MMD. Setting mmd_cfg.joint_detection_loss=True additionally computes detection loss on
the source batch too (mutual supervision instead of a frozen reference), scaled by
mmd_cfg.source_loss_weight (default 1.0, i.e. equal to the target's).
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
        # NOTE: ultralytics' BaseModel.loss() returns (loss, loss_items) where loss_items
        # is a dict[str, Tensor] (e.g. {"box_loss": ..., "cls_loss": ..., "dfl_loss": ...}),
        # not a positionally-matched tensor -- merged by key below, with an extra
        # "mmd_distance" entry added the same way.
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

        if self.mmd_cfg.joint_detection_loss:
            # Source needs a real (non-detached) forward pass for its own detection loss
            # regardless of detach_source_features -- that flag only controls whether the
            # *MMD term* also sees a gradient path into source, independent of the
            # detection-loss gradient it now always gets. This forward pass is the same
            # one detach_source_features=True would otherwise skip via no_grad, so joint
            # mode costs no extra forward pass over the mutual-pull (non-detached) case.
            source_loss, source_loss_items = super().loss(batch["domain_source"], preds)
            feat_source = self._captured_features
            if self.mmd_cfg.detach_source_features:
                feat_source = feat_source.detach()
            w = self.mmd_cfg.source_loss_weight
            loss = loss + w * source_loss
            loss_items = {k: loss_items[k] + w * source_loss_items[k] for k in loss_items}
        elif self.mmd_cfg.detach_source_features:
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

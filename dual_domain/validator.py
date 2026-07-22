"""DetectionValidator + a diagnostic val-time MMD distance.

Detection metrics (mAP, val loss) are computed only on the target-domain batch, exactly
as in stock DetectionValidator. The source batch is only used for an extra forward pass
so we can log how far apart the two domains' hooked features currently are; this number
is purely a diagnostic (mirrors the training-time distance stat) and never feeds into
fitness/checkpoint selection.
"""

from __future__ import annotations

from typing import Any

import torch
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.torch_utils import unwrap_model


class DualDomainDetectionValidator(DetectionValidator):
    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        self._source_batch = super().preprocess(batch["domain_source"])
        return super().preprocess(batch["domain_target"])

    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        self._dd_model = unwrap_model(model)
        self._mmd_distances: list[torch.Tensor] = []

    def update_metrics(self, preds: Any, batch: dict[str, Any]) -> None:
        feat_target = self._dd_model._captured_features
        self._dd_model.predict(self._source_batch["img"])
        feat_source = self._dd_model._captured_features
        _, distance = self._dd_model._mmd(feat_source, feat_target)
        self._mmd_distances.append(distance.detach())
        super().update_metrics(preds, batch)

    def get_stats(self) -> dict[str, Any]:
        stats = super().get_stats()
        if self._mmd_distances:
            # This 5th "metrics/" column makes ultralytics' own plot_results() choke: it
            # buckets results.csv columns by substring match on "loss"/"metric", then does
            # `plt.subplots(2, len(columns) // 2)` — integer division silently drops the
            # last slot whenever the combined count is odd (it always was even before this
            # column existed). The loop then indexes one past the end and plot_results()
            # logs (non-fatal) "Plotting error ... index N is out of bounds" and saves
            # results.png missing its last subplot. results.csv itself is unaffected; this
            # is an upstream ultralytics bug, not something to work around by renaming/
            # dropping this key.
            stats["metrics/mmd_distance"] = float(torch.stack(self._mmd_distances).mean())
        return stats

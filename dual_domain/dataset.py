"""Composes two real YOLODatasets instead of forking BaseDataset."""

from __future__ import annotations

import random
from typing import Any

from ultralytics.data.dataset import YOLODataset


class DualDomainYOLODataset(YOLODataset):
    """A target-domain YOLODataset (self, via the usual __init__) paired with a second,
    independent source-domain YOLODataset.

    Epoch length is defined by the target domain -- the one actually getting the
    detection loss + MMD alignment -- so a full pass over it happens every epoch, same
    as a normal single-domain training run. Each access draws a fresh random index into
    the source domain (the fixed/detached reference, forward-only) instead of a fixed
    `index % len(source)` cycle, so a given target image isn't paired with the same
    source image on every epoch.
    """

    def __init__(self, *args: Any, source_dataset_kwargs: dict[str, Any], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.source_dataset = YOLODataset(**source_dataset_kwargs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        target_sample = super().__getitem__(index)
        source_sample = self.source_dataset[random.randrange(len(self.source_dataset))]
        return {"domain_source": source_sample, "domain_target": target_sample}

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "domain_source": YOLODataset.collate_fn([sample["domain_source"] for sample in batch]),
            "domain_target": YOLODataset.collate_fn([sample["domain_target"] for sample in batch]),
        }

"""Composes two real YOLODatasets instead of forking BaseDataset."""

from __future__ import annotations

import random
from typing import Any

from ultralytics.data.dataset import YOLODataset


class DualDomainYOLODataset(YOLODataset):
    """A source-domain YOLODataset (self, via the usual __init__) paired with a second,
    independent target-domain YOLODataset.

    Epoch length is defined by the source domain. Each access draws a fresh random index
    into the target domain, so a given source image isn't paired with the same target
    image on every epoch (unlike a fixed `index % len(target)` cycle).
    """

    def __init__(self, *args: Any, target_dataset_kwargs: dict[str, Any], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.target_dataset = YOLODataset(**target_dataset_kwargs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_sample = super().__getitem__(index)
        target_sample = self.target_dataset[random.randrange(len(self.target_dataset))]
        return {"domain_source": source_sample, "domain_target": target_sample}

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "domain_source": YOLODataset.collate_fn([sample["domain_source"] for sample in batch]),
            "domain_target": YOLODataset.collate_fn([sample["domain_target"] for sample in batch]),
        }

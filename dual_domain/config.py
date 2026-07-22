"""Single flat config for dual-domain MMD training (no nested yaml-of-yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MMDWeightSchedule:
    type: str = "constant"  # "constant" or "linear"
    end_weight: float = 0.0
    end_epoch: int | None = None  # defaults to trainer.epochs when type == "linear"

    def weight_at(self, epoch: int, start_weight: float, total_epochs: int) -> float:
        if self.type == "constant":
            return start_weight
        if self.type == "linear":
            end_epoch = self.end_epoch or total_epochs
            if end_epoch <= 0:
                return start_weight
            t = min(max(epoch / end_epoch, 0.0), 1.0)
            return start_weight + t * (self.end_weight - start_weight)
        raise NotImplementedError(f"Unknown MMD weight schedule type: {self.type!r}")


@dataclass
class MMDConfig:
    kernel: str = "rbf"
    preprocess: str = "flatten"  # "flatten" or "gap"
    momentum: float = 0.9
    mmd_weight: float = 1.0
    mmd_target_layer: int = 10
    detach_source_features: bool = False
    weight_schedule: MMDWeightSchedule = field(default_factory=MMDWeightSchedule)
    bandwidth_freeze_epoch: int | None = None  # None = never freeze (EMA keeps updating, old behavior)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> MMDConfig:
        data = dict(data or {})
        schedule = data.pop("weight_schedule", None)
        return cls(**data, weight_schedule=MMDWeightSchedule(**(schedule or {})))

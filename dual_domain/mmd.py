"""Maximum Mean Discrepancy between source- and target-domain feature maps.

Pure PyTorch, no ultralytics coupling. The RBF bandwidth uses the common
"median heuristic" approximation: an exponential moving average of the batch's
mean pairwise squared distance / 2, rather than the literal median (cheaper,
avoids sorting the full pairwise distance matrix every step).
"""

from __future__ import annotations

import torch
from torch import nn


class MomentumRBFMMD(nn.Module):
    """Two-sample RBF-kernel MMD with an EMA-smoothed bandwidth."""

    def __init__(self, momentum: float = 0.9):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("running_scale", torch.tensor(1.0), persistent=False)
        self._initialized = False
        self._frozen = False

    def freeze_bandwidth(self) -> None:
        """Stop updating the EMA bandwidth from now on; keep whatever value it currently
        holds. Breaks a feedback loop: as the two domains' features get closer, the
        EMA bandwidth shrinks too, which sharpens the kernel and rewards collapsing
        even further. Freezing it removes that runaway incentive."""
        self._frozen = True

    def preprocess(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} must implement preprocess()")

    def _rbf_kernel(self, joint: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        l2_distance = (joint.unsqueeze(0) - joint.unsqueeze(1)).pow(2).sum(-1)
        batch_scale = (l2_distance.mean() / 2).detach()
        if not self._frozen:
            if not self._initialized:
                self.running_scale = batch_scale.clamp_min(1e-6)
                self._initialized = True
            else:
                self.running_scale = self.momentum * self.running_scale + (1 - self.momentum) * batch_scale
        bandwidth = self.running_scale.clamp_min(1e-6)
        kernel = torch.exp(-l2_distance / bandwidth)
        return kernel, batch_scale

    def forward(self, source_features: torch.Tensor, target_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mmd, distance) where `distance` is the raw batch bandwidth stat, for logging.

        Always computed in fp32 regardless of the caller's dtype: squared pairwise
        distances over a flattened feature vector easily overflow fp16's ~65504 range,
        which callers can hand us either via torch.amp.autocast (training) or a blanket
        model.half() (ultralytics casts the whole model to fp16 for validation whenever
        AMP is enabled) — autocast keeps sensitive reductions in fp32 automatically, a
        raw .half() cast does not.
        """
        source = self.preprocess(source_features.float())
        target = self.preprocess(target_features.float())
        n_source = source.shape[0]
        kernel, distance = self._rbf_kernel(torch.cat([source, target], dim=0))
        k_ss = kernel[:n_source, :n_source]
        k_tt = kernel[n_source:, n_source:]
        k_st = kernel[:n_source, n_source:]
        k_ts = kernel[n_source:, :n_source]
        mmd = k_ss.mean() + k_tt.mean() - k_st.mean() - k_ts.mean()
        return mmd, distance


class MmdFlattenRBF(MomentumRBFMMD):
    """Flattens all spatial locations into one feature vector per image."""

    def preprocess(self, features: torch.Tensor) -> torch.Tensor:
        return features.flatten(1)


class MmdGapRBF(MomentumRBFMMD):
    """Global-average-pools each channel into one feature vector per image."""

    def preprocess(self, features: torch.Tensor) -> torch.Tensor:
        return features.mean(dim=tuple(range(2, features.ndim)))


_REGISTRY: dict[tuple[str, str], type[MomentumRBFMMD]] = {
    ("rbf", "flatten"): MmdFlattenRBF,
    ("rbf", "gap"): MmdGapRBF,
}


def build_mmd(kernel: str, preprocess: str, momentum: float = 0.9) -> MomentumRBFMMD:
    key = (kernel.lower(), preprocess.lower())
    try:
        cls = _REGISTRY[key]
    except KeyError as e:
        raise NotImplementedError(f"No MMD implementation for kernel={kernel!r}, preprocess={preprocess!r}") from e
    return cls(momentum=momentum)

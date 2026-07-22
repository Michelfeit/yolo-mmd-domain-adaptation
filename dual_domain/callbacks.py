"""Opt-in callback wiring for the PCA feature-space tracker.

Kept separate from DualDomainTrainer: it's a diagnostic/visualization aid, not part of
training itself, and attaching it is a single explicit call rather than a trainer
side-effect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .pca_tracker import FeaturePCATracker


def attach_pca_tracker(
    trainer: Any,
    csv_path: str | Path | None = None,
    n_samples_per_domain: int = 64,
    n_components: int = 2,
    extract_batch_size: int = 16,
) -> None:
    """Fit a PCA basis right before the first training batch (on the freshly loaded/
    pretrained weights), then re-project the same fixed sample of images into that
    frozen basis at the end of every epoch, logging to `csv_path`.

    Safe to call any time after constructing the trainer (even before `.train()`):
    actual construction of the tracker is deferred to `on_pretrain_routine_end`, by
    which point `trainer.data`/`trainer.model`/`trainer.device` are guaranteed to exist.

    `csv_path` defaults to `trainer.save_dir / "pca_features.csv"`, resolved lazily
    inside the callback rather than by the caller building a path from the requested
    run name up front -- ultralytics may auto-increment `save_dir` (e.g.
    `my_run` -> `my_run2`) if the requested directory already exists, and a path built
    from the requested name would silently point at the wrong (non-incremented)
    directory, split from every other artifact of that same run.
    """
    state: dict[str, FeaturePCATracker] = {}

    def _fit_initial(trainer: Any) -> None:
        tracker = FeaturePCATracker(
            data=trainer.data,
            cfg=trainer.args,
            csv_path=csv_path if csv_path is not None else trainer.save_dir / "pca_features.csv",
            n_samples_per_domain=n_samples_per_domain,
            n_components=n_components,
            extract_batch_size=extract_batch_size,
        )
        tracker.fit_initial(trainer.model, trainer.device)
        state["tracker"] = tracker

    def _snapshot(trainer: Any) -> None:
        state["tracker"].snapshot(trainer.epoch + 1, trainer.model, trainer.device)

    trainer.add_callback("on_pretrain_routine_end", _fit_initial)
    trainer.add_callback("on_train_epoch_end", _snapshot)

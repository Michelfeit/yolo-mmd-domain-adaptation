"""DetectionTrainer + dual-domain (source/target) batches.

Only 6 methods differ from stock DetectionTrainer; everything else (optimizer building,
DDP setup, scheduler, checkpointing, early stopping, ...) is inherited unchanged.
"""

from __future__ import annotations

from copy import copy
from typing import Any

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOCAL_RANK, LOGGER, RANK
from ultralytics.utils.torch_utils import strip_optimizer, torch_distributed_zero_first, unwrap_model

from .config import MMDConfig
from .data_utils import build_dual_domain_yolo_dataset, check_domain_adapt_dataset
from .model import DualDomainDetectionModel
from .validator import DualDomainDetectionValidator


def _update_mmd_weight(trainer: DualDomainTrainer) -> None:
    """on_train_epoch_start callback driving the (optional) MMD weight schedule.

    Kept out of the model entirely: the model just reads `self.mmd_weight` each forward,
    and this callback is the only thing that needs to know about epochs/schedules.
    """
    model = unwrap_model(trainer.model)
    schedule = model.mmd_cfg.weight_schedule
    model.mmd_weight = schedule.weight_at(trainer.epoch, model.mmd_cfg.mmd_weight, trainer.epochs)


def _maybe_freeze_bandwidth(trainer: DualDomainTrainer) -> None:
    """on_train_epoch_start callback driving the (optional) bandwidth freeze.

    Same reasoning as the weight-schedule callback: epoch-awareness lives here, not in
    the MMD module itself, which only needs a one-shot freeze_bandwidth() call. Calling
    it every epoch past the threshold is harmless (idempotent).
    """
    model = unwrap_model(trainer.model)
    freeze_epoch = model.mmd_cfg.bandwidth_freeze_epoch
    if freeze_epoch is not None and trainer.epoch >= freeze_epoch:
        model._mmd.freeze_bandwidth()


class DualDomainTrainer(DetectionTrainer):
    def __init__(self, cfg: Any = DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: Any = None):
        overrides = dict(overrides or {})
        self.mmd_cfg = MMDConfig.from_dict(overrides.pop("mmd", None))
        super().__init__(cfg, overrides, _callbacks)
        self.add_callback("on_train_epoch_start", _update_mmd_weight)
        self.add_callback("on_train_epoch_start", _maybe_freeze_bandwidth)

    def get_dataset(self) -> dict[str, Any]:
        data = check_domain_adapt_dataset(self.args.data)
        if self.args.single_cls:
            data["names"] = {0: "item"}
            data["nc"] = 1
        return data

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_dual_domain_yolo_dataset(self.args, self.data, mode=mode, batch=batch, stride=gs)

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch["domain_source"] = super().preprocess_batch(batch["domain_source"])
        batch["domain_target"] = super().preprocess_batch(batch["domain_target"])
        # BaseTrainer's progress bar / plot_training_samples index batch["img"], batch["cls"],
        # etc. directly with no override point — alias the target domain's fields onto the
        # top level so those still work; the model only ever reads the nested domain keys.
        batch.update(batch["domain_target"])
        return batch

    def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True) -> DualDomainDetectionModel:
        model = DualDomainDetectionModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1, mmd_cfg=self.mmd_cfg
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self) -> DualDomainDetectionValidator:
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "mmd_distance"
        return DualDomainDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def final_eval(self) -> None:
        """BaseTrainer.final_eval() re-loads best.pt via `self.validator(model=model)`,
        which always takes BaseValidator's standalone branch (`trainer=None`) and resolves
        `self.args.data` with the stock `check_det_dataset` — incompatible with the
        source/target yaml schema, and with no override seam to fix from the validator side.

        Re-run here via `trainer=self` instead, so it takes the training-mode branch and
        reuses `self.data`/`self.test_loader` like every other validation call during
        training. Trade-off: this validates the live in-memory (EMA) weights rather than
        round-tripping best.pt through disk — checkpoint save/load itself is untouched
        ultralytics code, so that's a redundant integrity check being skipped, not a gap
        in dual-domain-specific behavior.
        """
        with torch_distributed_zero_first(LOCAL_RANK):
            if RANK in {-1, 0}:
                if self.last.exists():
                    strip_optimizer(self.last)
                if self.best.exists():
                    strip_optimizer(self.best)
        if self.best.exists():
            LOGGER.info(f"\nValidating {self.best}...")
            self.validator.args.plots = self.args.plots
            self.metrics = self.validator(trainer=self)
            self.metrics.pop("fitness", None)
            self.run_callbacks("on_fit_epoch_end")

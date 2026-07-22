"""Example dual-domain training run using the tiny synthetic dummy dataset (see
make_dummy_data.py) -- runs end to end with no external data required.

    python examples/make_dummy_data.py   # only needed once, or to regenerate
    python examples/train.py

Two named variants (see the handoff discussion this repo grew out of):
  - "as is": detection loss only on the target domain + MMD alignment (mmd_weight > 0
    is the only thing that couples the two domains; source is forward-only).
  - detach_source_features toggles whether the source's forward pass contributes
    gradient to the MMD term (mutual pull, matching prior results) or is frozen
    relative to MMD (target is pulled toward a fixed source, source unaffected by MMD).

This dummy dataset (8 train / 2 val images per domain) is only large enough to prove the
pipeline runs; point `data` at your own dataset yaml (same source/target schema) and scale
up imgsz/batch/epochs for anything resembling real training.
"""

from dual_domain import DualDomainTrainer, attach_pca_tracker

overrides = {
    "data": "examples/dummy_data.yaml",
    "model": "yolov10n.yaml",
    "epochs": 5,
    "imgsz": 128,
    "batch": 4,
    "workers": 0,
    "mmd": {
        "kernel": "rbf",
        "preprocess": "flatten",  # or "gap"
        "momentum": 0.9,
        "mmd_weight": 1.0,
        "mmd_target_layer": 4,  # yolov10n.yaml's backbone is shallower than this repo's main results used (layer 10)
        "detach_source_features": False,
        "weight_schedule": {"type": "constant"},  # or {"type": "linear", "end_weight": 0.0}
    },
}

trainer = DualDomainTrainer(overrides=overrides)
attach_pca_tracker(trainer, n_samples_per_domain=8)
trainer.train()

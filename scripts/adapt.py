#!/usr/bin/env python3
"""Run a single dual-domain (MMD) adaptation run against one configs/adapt/*.yaml.

GPU is selected via CUDA_VISIBLE_DEVICES in front of the command, same as
scripts/pretrain.py. --model must be the pretrained checkpoint for whichever domain
plays `source` in --data's yaml (the fixed/detached reference domain) -- i.e. the
checkpoint scripts/pretrain.py produced for that same dataset variant.

    CUDA_VISIBLE_DEVICES=0 python scripts/adapt.py \\
        --data configs/adapt/real_small_source_syn_large_target.yaml \\
        --model runs/yolov10n/pretrain/real_small/weights/best.pt &
    CUDA_VISIBLE_DEVICES=1 python scripts/adapt.py \\
        --data configs/adapt/syn_small_source_real_large_target.yaml \\
        --model runs/yolov10n/pretrain/syn_small/weights/best.pt &
    wait

Saves to runs/<model-tag>/adapt/<name>/ (name defaults to the data yaml's stem).
"""

import argparse
from pathlib import Path

from dual_domain import DualDomainTrainer, attach_pca_tracker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="path to an adapt dataset yaml, e.g. configs/adapt/real_small_source_syn_large_target.yaml")
    parser.add_argument("--model", required=True, help="pretrained checkpoint for the source domain (see scripts/pretrain.py)")
    parser.add_argument("--model-tag", default="yolov10n", help="run-directory tag")
    parser.add_argument("--name", default=None, help="run name under the project dir, defaults to the data yaml's stem")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=2080)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default=None, help="defaults to runs/<model-tag>/adapt")

    parser.add_argument("--mmd-weight", type=float, default=0.8)
    parser.add_argument("--mmd-target-layer", type=int, default=10)
    parser.add_argument("--preprocess", choices=["flatten", "gap"], default="flatten")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--mmd-weight-schedule",
        choices=["constant", "linear"],
        default="constant",
        help="constant: mmd_weight stays fixed. linear: decays from --mmd-weight to --mmd-weight-end "
        "by --mmd-weight-end-epoch (defaults to the full run).",
    )
    parser.add_argument("--mmd-weight-end", type=float, default=0.0, help="only used if --mmd-weight-schedule=linear")
    parser.add_argument(
        "--mmd-weight-end-epoch",
        type=int,
        default=None,
        help="only used if --mmd-weight-schedule=linear; defaults to --epochs (decay over the whole run)",
    )
    parser.add_argument(
        "--bandwidth-freeze-epoch",
        type=int,
        default=None,
        help="stop updating the EMA kernel bandwidth from this epoch on; default: never freeze",
    )
    parser.add_argument("--pca-samples", type=int, default=64, help="fixed sample size per domain for PCA tracking")
    parser.add_argument("--pca-components", type=int, default=3)
    parser.add_argument("--pca-batch-size", type=int, default=16)
    args = parser.parse_args()

    name = args.name or Path(args.data).stem
    # Must be absolute: ultralytics silently nests *relative* project paths under its own
    # default runs/<task>/ root (see get_save_dir).
    project = args.project or str(Path(f"runs/{args.model_tag}/adapt").resolve())

    print(f"=== Dual-domain adaptation: {name} (starting from {args.model}) ===")
    overrides = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": project,
        "name": name,
        "mmd": {
            "kernel": "rbf",
            "preprocess": args.preprocess,
            "momentum": args.momentum,
            "mmd_weight": args.mmd_weight,
            "mmd_target_layer": args.mmd_target_layer,
            "detach_source_features": True,
            "weight_schedule": {
                "type": args.mmd_weight_schedule,
                "end_weight": args.mmd_weight_end,
                "end_epoch": args.mmd_weight_end_epoch,
            },
            "bandwidth_freeze_epoch": args.bandwidth_freeze_epoch,
        },
    }
    trainer = DualDomainTrainer(overrides=overrides)
    attach_pca_tracker(
        trainer,
        n_samples_per_domain=args.pca_samples,
        n_components=args.pca_components,
        extract_batch_size=args.pca_batch_size,
    )
    trainer.train()
    print(f"\nDone. Checkpoint: {Path(project) / name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()

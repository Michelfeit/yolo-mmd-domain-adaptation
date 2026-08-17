#!/usr/bin/env python3
"""Dual-domain training where BOTH domains receive their own detection loss (not just
the target), in addition to the usual MMD alignment term.

One invocation trains exactly one model. The warm-start checkpoint is entirely your
call via --model -- pass the raw COCO-pretrained weights (e.g. yolov10n.pt) for no
warm-up at all, or any already-domain-pretrained checkpoint if you want one; this
script has no opinion on which. --data points at a single dual-domain yaml (source:/
target: keys, see configs/adapt/*.yaml) -- since both domains get real detection loss
here and MMD is symmetric (see dual_domain/mmd.py), which physical domain is labeled
"source" vs "target" in that yaml is just a data-loading formality, not an asymmetric
training role.

    python scripts/joint_train.py --model yolov10n.pt \\
        --data configs/adapt/small/syn_source_real_target.yaml --name coco_joint_mmd
    python scripts/joint_train.py --model yolov10n.pt \\
        --data configs/adapt/small/syn_source_real_target.yaml --name coco_joint_no_mmd --no-mmd
    python scripts/joint_train.py --model runs/yolov10n/pretrain/syn_large/weights/best.pt \\
        --data configs/adapt/small/syn_source_real_target.yaml --name syn_large_warmup_joint_mmd
"""

import argparse
import csv
import sys
from pathlib import Path

from dual_domain import DualDomainTrainer, attach_pca_tracker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model", required=True,
        help="warm-start checkpoint -- e.g. yolov10n.pt for no warm-up at all, or any "
        "domain-pretrained .pt if you want one",
    )
    parser.add_argument("--data", required=True, help="dual-domain yaml (source:/target: keys)")
    parser.add_argument("--model-tag", default="joint", help="output grouping: runs/<model-tag>/<variant>/<name>")
    parser.add_argument("--name", default=None, help="run name under runs/<model-tag>/<variant>/; defaults to --data's filename stem")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=2080)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="e.g. '0,1' for 2-GPU DDP, or a single index like '0'")
    parser.add_argument(
        "--no-mmd", action="store_true",
        help="joint detection loss on both domains, MMD weight forced to 0 (still computed each "
        "step, just zero-weighted -- no extra cost since joint mode already forwards source)",
    )
    parser.add_argument("--mmd-weight", type=float, default=0.8)
    parser.add_argument("--mmd-target-layer", type=int, default=10)
    parser.add_argument("--preprocess", choices=["flatten", "gap"], default="flatten")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--mmd-weight-schedule",
        choices=["constant", "linear"],
        default="constant",
        help="ignored if --no-mmd. constant: mmd_weight stays fixed. linear: decays from "
        "--mmd-weight to --mmd-weight-end by --mmd-weight-end-epoch (defaults to the full run).",
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
        help="ignored if --no-mmd. stop updating the EMA kernel bandwidth from this epoch on; "
        "default: never freeze",
    )
    parser.add_argument(
        "--detach-source-features", action="store_true",
        help="detach the features fed into the MMD term specifically; source's own detection-loss "
        "gradient is unaffected either way (this only controls the MMD term's gradient path)",
    )
    parser.add_argument(
        "--source-loss-weight", type=float, default=1.0,
        help="scales the source-domain detection loss relative to the target's (which always stays "
        "at its natural scale, weight 1.0). Use < 1.0 to keep the source domain as a light supervised "
        "anchor -- e.g. when the warm-start checkpoint is already a full pretrain and you mainly want "
        "the target domain to actually learn, with source detection loss just preventing drift.",
    )
    parser.add_argument("--pca-samples", type=int, default=64, help="fixed sample size per domain for PCA tracking")
    parser.add_argument("--pca-components", type=int, default=3)
    parser.add_argument("--pca-batch-size", type=int, default=16)
    args = parser.parse_args()

    ckpt = Path(args.model)
    if not ckpt.exists():
        sys.exit(f"Checkpoint not found: {ckpt}")
    if not Path(args.data).exists():
        sys.exit(f"Data yaml not found: {args.data}")

    variant = "joint_no_mmd" if args.no_mmd else "joint_mmd"
    name = args.name or Path(args.data).stem
    project = str(Path(f"runs/{args.model_tag}/{variant}").resolve())

    label = "no MMD" if args.no_mmd else "+ MMD"
    print(f"\n=== Joint detection loss ({label}): {name} (model={ckpt}, data={args.data}) ===")

    overrides = {
        "model": str(ckpt),
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": project,
        "name": name,
        "mmd": {
            "kernel": "rbf",
            "preprocess": args.preprocess,
            "momentum": args.momentum,
            "mmd_weight": 0.0 if args.no_mmd else args.mmd_weight,
            "mmd_target_layer": args.mmd_target_layer,
            "detach_source_features": args.detach_source_features,
            "joint_detection_loss": True,
            "source_loss_weight": args.source_loss_weight,
            "weight_schedule": {
                "type": "constant" if args.no_mmd else args.mmd_weight_schedule,
                "end_weight": args.mmd_weight_end,
                "end_epoch": args.mmd_weight_end_epoch,
            },
            "bandwidth_freeze_epoch": None if args.no_mmd else args.bandwidth_freeze_epoch,
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

    summary_path = Path(project) / "joint_summary.csv"
    if trainer.metrics is None:
        # Multi-GPU (--device with 2+ ids): BaseTrainer.train() only spawns a DDP
        # subprocess from this process and returns -- the actual training/final_eval
        # (and this trainer's self.metrics) happen in that subprocess's own trainer
        # instance, never in this one. Checkpoints/results.csv on disk are unaffected;
        # only this process's summary-row bookkeeping has nothing to read.
        print(f"  (metrics unavailable in the DDP launcher process; "
              f"see {project}/{name}/results.csv for per-epoch numbers)")
        return

    row = {"run": name, "model": str(ckpt), "variant": variant, **trainer.metrics}
    # Merge with any existing summary rather than overwrite, so repeated invocations
    # into the same --model-tag/variant accumulate one row per run name.
    existing_rows = {}
    if summary_path.exists():
        with open(summary_path, newline="", encoding="utf-8") as f:
            existing_rows = {r["run"]: r for r in csv.DictReader(f)}
    existing_rows[name] = row

    fieldnames = sorted({k for r in existing_rows.values() for k in r})
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows.values())

    print(f"\nSummary table written to {summary_path}")
    print("  ".join(f"{h:>14}" for h in fieldnames))
    for r in existing_rows.values():
        print("  ".join(f"{r.get(h, '')!s:>14.14}" for h in fieldnames))


if __name__ == "__main__":
    main()

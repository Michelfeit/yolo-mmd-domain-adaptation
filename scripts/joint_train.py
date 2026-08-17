#!/usr/bin/env python3
"""Dual-domain training where BOTH domains receive their own detection loss (not just
the target), in addition to the usual MMD alignment term.

Tests whether source-domain forgetting can be avoided by supervising both domains
directly, and whether MMD contributes anything on top of that: run once with --no-mmd
for the joint-detection-loss-only comparison point, and once without it for the
joint-loss-plus-MMD run. Both start from the same pretrained checkpoints as
scripts/adapt_train.py and use the same source/target config yamls.

    python scripts/joint_train.py --no-mmd
    python scripts/joint_train.py
    python scripts/joint_train.py --only syn_source_real_target --no-mmd
"""

import argparse
import csv
import sys
from pathlib import Path

from dual_domain import DualDomainTrainer, attach_pca_tracker

RUN_SPECS = [
    {"name": "real_source_syn_target", "source_domain": "real"},
    {"name": "syn_source_real_target", "source_domain": "syn"},
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs-dir", default="configs/adapt")
    parser.add_argument("--pretrain-model-tag", default="yolov10n", help="model tag used in scripts/pretrain_baselines.py")
    parser.add_argument("--model-tag", default=None, help="run-directory tag, defaults to --pretrain-model-tag")
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
    parser.add_argument("--pca-samples", type=int, default=64, help="fixed sample size per domain for PCA tracking")
    parser.add_argument("--pca-components", type=int, default=3)
    parser.add_argument("--pca-batch-size", type=int, default=16)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[spec["name"] for spec in RUN_SPECS],
        default=None,
        help="run only these directions instead of both (e.g. after one direction crashed)",
    )
    args = parser.parse_args()

    model_tag = args.model_tag or args.pretrain_model_tag
    pretrain_project = Path(f"runs/{args.pretrain_model_tag}/pretrain")
    checkpoints = {
        domain: pretrain_project / domain / "weights" / "best.pt"
        for domain in ("syn", "real")
    }
    for domain, ckpt in checkpoints.items():
        if not ckpt.exists():
            sys.exit(f"Missing {domain} baseline checkpoint: {ckpt} (run scripts/pretrain_baselines.py first)")

    variant = "joint_no_mmd" if args.no_mmd else "joint_mmd"
    joint_project = str(Path(f"runs/{model_tag}/{variant}").resolve())
    specs = [spec for spec in RUN_SPECS if args.only is None or spec["name"] in args.only]

    rows = []
    for spec in specs:
        name = spec["name"]
        ckpt = checkpoints[spec["source_domain"]]
        data = f"{args.configs_dir}/{name}.yaml"
        label = "no MMD" if args.no_mmd else "+ MMD"
        print(f"\n=== Joint detection loss ({label}): {name} (source={spec['source_domain']}, starting from {ckpt}) ===")

        overrides = {
            "model": str(ckpt),
            "data": data,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "project": joint_project,
            "name": name,
            "mmd": {
                "kernel": "rbf",
                "preprocess": args.preprocess,
                "momentum": args.momentum,
                "mmd_weight": 0.0 if args.no_mmd else args.mmd_weight,
                "mmd_target_layer": args.mmd_target_layer,
                "detach_source_features": args.detach_source_features,
                "joint_detection_loss": True,
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
        if trainer.metrics is None:
            # Multi-GPU (--device with 2+ ids): BaseTrainer.train() only spawns a DDP
            # subprocess from this process and returns -- the actual training/final_eval
            # (and this trainer's self.metrics) happen in that subprocess's own trainer
            # instance, never in this one. Checkpoints/results.csv on disk are unaffected;
            # only this process's summary-row bookkeeping has nothing to read.
            print(f"  (metrics unavailable in the DDP launcher process for {name}; "
                  f"see {joint_project}/{name}/results.csv for per-epoch numbers)")
        else:
            rows.append({"run": name, "source_domain": spec["source_domain"], "variant": variant, **trainer.metrics})

    # Merge with any existing summary rather than overwrite, so re-running just one
    # direction (e.g. --only after a crash) doesn't lose the other direction's row.
    summary_path = Path(joint_project) / "joint_summary.csv"
    existing_rows = {}
    if summary_path.exists():
        with open(summary_path, newline="", encoding="utf-8") as f:
            existing_rows = {row["run"]: row for row in csv.DictReader(f)}
    for row in rows:
        existing_rows[row["run"]] = row

    fieldnames = sorted({k for row in existing_rows.values() for k in row})
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows.values())

    print(f"\nSummary table written to {summary_path}")
    print("  ".join(f"{h:>14}" for h in fieldnames))
    for row in existing_rows.values():
        print("  ".join(f"{row.get(h, '')!s:>14.14}" for h in fieldnames))


if __name__ == "__main__":
    main()

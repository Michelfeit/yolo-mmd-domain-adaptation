#!/usr/bin/env python3
"""Queue both dual-domain (MMD) adaptation directions, each starting from its
pretrained source-domain baseline checkpoint (see scripts/pretrain_baselines.py):

  - real_source_syn_target: source=real (pretrained), target=syn (fine-tuned + aligned)
  - syn_source_real_target: source=syn (pretrained),  target=real (fine-tuned + aligned)

Uses the FULL datasets (configs/adapt/*.yaml), not the stride-subsampled source/ dirs
used for baseline pretraining. The source domain's forward pass is detached from the
MMD gradient (detach_source_features=True) so its weights only move via the target
domain's own detection-loss + MMD-into-target-branch gradients, never via a gradient
computed from processing the source batch itself.

    python scripts/adapt_train.py
    python scripts/adapt_train.py --epochs 150 --mmd-weight 2.0
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
        help="stop updating the EMA kernel bandwidth from this epoch on (breaks the shrinking-bandwidth "
        "feedback loop that rewards feature collapse); default: never freeze, EMA keeps updating",
    )
    parser.add_argument("--pca-samples", type=int, default=64, help="fixed sample size per domain for PCA tracking")
    parser.add_argument(
        "--pca-components",
        type=int,
        default=3,
        help="PCA components to fit; the first 2 columns are the 2D view for free (PCA is nested/hierarchical)",
    )
    parser.add_argument(
        "--pca-batch-size",
        type=int,
        default=16,
        help="batch size for the (unrelated to training) PCA feature-extraction forward passes",
    )
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

    adapt_project = str(Path(f"runs/{model_tag}/adapt").resolve())
    specs = [spec for spec in RUN_SPECS if args.only is None or spec["name"] in args.only]

    rows = []
    for spec in specs:
        name = spec["name"]
        ckpt = checkpoints[spec["source_domain"]]
        data = f"{args.configs_dir}/{name}.yaml"
        print(f"\n=== Dual-domain adaptation: {name} (source={spec['source_domain']}, starting from {ckpt}) ===")

        overrides = {
            "model": str(ckpt),
            "data": data,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "project": adapt_project,
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
        # csv_path defaults to trainer.save_dir / "pca_features.csv" -- correct even if
        # ultralytics auto-incremented save_dir away from the requested `name` (e.g. a
        # collision with a prior crashed attempt), unlike building the path from `name`.
        attach_pca_tracker(
            trainer,
            n_samples_per_domain=args.pca_samples,
            n_components=args.pca_components,
            extract_batch_size=args.pca_batch_size,
        )
        trainer.train()
        # trainer.metrics holds the final post-training validation result (target-domain
        # mAP + the diagnostic val-time metrics/mmd_distance), same dict our final_eval()
        # override populates every training run — this is what's missing from just the
        # per-run results.csv: one row per direction, side by side.
        rows.append({"run": name, "source_domain": spec["source_domain"], **trainer.metrics})

    # Merge with any existing summary rather than overwrite, so re-running just one
    # direction (e.g. --only after a crash) doesn't lose the other direction's row.
    summary_path = Path(adapt_project) / "adapt_summary.csv"
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

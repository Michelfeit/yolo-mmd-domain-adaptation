#!/usr/bin/env python3
"""Cross-evaluate dual-domain adapted checkpoints (plus the two pretrain baselines) on
the exact same eval data used for the pretrain baselines
(configs/pretrain/{syn,real}_source.yaml -> the stride-3 subsampled source/val sets), so
all numbers are directly comparable -- same val images, same metric columns.

Loads each adapted checkpoint via stock ultralytics.YOLO (no dual_domain machinery
needed for plain single-domain inference/validation -- DualDomainDetectionModel only
overrides loss(), which a standalone .val() call never invokes).

Note: DEFAULT_ADAPT_RUNS uses the current (post-rename) run-directory naming produced
by scripts/adapt_train.py. Runs completed before the anchor/adapt -> source/target
rename kept their original directory names (e.g. real_anchor_syn_adapt) -- evaluate
those via --extra-checkpoint instead.

    python scripts/eval_adapted_vs_baseline.py
    python scripts/eval_adapted_vs_baseline.py --extra-checkpoint syn_source_real_target_reg=runs/yolov10n_reg/adapt/syn_anchor_real_adapt4/weights/best.pt
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

EVAL_DOMAINS = ("syn", "real")
DEFAULT_ADAPT_RUNS = ("real_source_syn_target", "syn_source_real_target")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapt-model-tag", default="yolov10n", help="model tag used in scripts/adapt_train.py")
    parser.add_argument("--pretrain-model-tag", default="yolov10n", help="model tag used in scripts/pretrain_baselines.py")
    parser.add_argument("--configs-dir", default="configs/pretrain", help="same eval yamls baseline_summary.csv used")
    parser.add_argument("--imgsz", type=int, default=2080)
    parser.add_argument(
        "--extra-checkpoint",
        nargs="+",
        default=[],
        metavar="NAME=PATH",
        help="additional checkpoints to evaluate, e.g. syn_source_real_target_reg=runs/yolov10n_reg/adapt/syn_anchor_real_adapt4/weights/best.pt",
    )
    args = parser.parse_args()

    adapt_project = Path(f"runs/{args.adapt_model_tag}/adapt")

    checkpoints = {name: adapt_project / name / "weights" / "best.pt" for name in DEFAULT_ADAPT_RUNS}
    for spec in args.extra_checkpoint:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--extra-checkpoint must be NAME=PATH, got {spec!r}")
        checkpoints[name] = Path(path)

    rows = []

    baseline_path = Path(f"runs/{args.pretrain_model_tag}/pretrain/baseline_summary.csv")
    if baseline_path.exists():
        with open(baseline_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append({"model": f"baseline_{row['trained_on']}", "eval_on": row["eval_on"], **row})
    else:
        print(f"No baseline summary at {baseline_path}, skipping baselines")

    for run_name, ckpt in checkpoints.items():
        if not ckpt.exists():
            print(f"Skipping {run_name}: no checkpoint at {ckpt}")
            continue
        model = YOLO(str(ckpt))
        for eval_on in EVAL_DOMAINS:
            data = f"{args.configs_dir}/{eval_on}_source.yaml"
            name = f"{run_name}_on_{eval_on}"
            print(f"  {name}: {ckpt} on {data}")
            metrics = model.val(data=data, imgsz=args.imgsz, project=str(adapt_project / "eval_vs_baseline"), name=name)
            rows.append({"model": run_name, "eval_on": eval_on, **metrics.results_dict})

    if not rows:
        raise SystemExit("No checkpoints found to evaluate.")

    summary_path = adapt_project / "full_comparison_summary.csv"
    fieldnames = sorted({k for row in rows for k in row})
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary written to {summary_path}")
    print("  ".join(f"{h:>16}" for h in fieldnames))
    for row in rows:
        print("  ".join(f"{row.get(h, '')!s:>16.16}" for h in fieldnames))


if __name__ == "__main__":
    main()

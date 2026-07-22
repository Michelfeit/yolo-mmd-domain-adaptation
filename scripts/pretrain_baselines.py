#!/usr/bin/env python3
"""Train + cross-evaluate the two single-domain pretrain baselines (syn, real).

Runs configs/pretrain/{syn,real}_source.yaml through stock ultralytics YOLO training
(saving to runs/<model_tag>/pretrain/{syn,real}), then evaluates each trained model on
BOTH domains' val sets — in-domain and cross-domain — to get the baseline numbers the
dual-domain (MMD) results get compared against later.

    python scripts/pretrain_baselines.py
    python scripts/pretrain_baselines.py --model yolov10s.yaml --epochs 150
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

DOMAINS = ("syn", "real")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="yolov10n.yaml")
    parser.add_argument("--model-tag", default=None, help="run-directory tag, defaults to the model stem")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=2080)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--configs-dir", default="configs/pretrain")
    args = parser.parse_args()

    model_tag = args.model_tag or Path(args.model).stem
    # Must be absolute: ultralytics silently nests *relative* project paths under its own
    # default runs/<task>/ root (see get_save_dir), which would break the checkpoint-path
    # assumptions below.
    project = str(Path(f"runs/{model_tag}/pretrain").resolve())

    checkpoints = {}
    for domain in DOMAINS:
        data = f"{args.configs_dir}/{domain}_source.yaml"
        print(f"\n=== Training {domain} baseline ({data}) ===")
        model = YOLO(args.model)
        model.train(data=data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, project=project, name=domain)
        checkpoints[domain] = Path(project) / domain / "weights" / "best.pt"

    print("\n=== Cross-evaluation (in-domain + cross-domain) ===")
    rows = []
    for trained_on, ckpt in checkpoints.items():
        model = YOLO(ckpt)
        for eval_on in DOMAINS:
            data = f"{args.configs_dir}/{eval_on}_source.yaml"
            name = f"{trained_on}_on_{eval_on}"
            print(f"  {name}: {ckpt} on {data}")
            metrics = model.val(data=data, imgsz=args.imgsz, project=f"{project}/eval", name=name)
            row = {"trained_on": trained_on, "eval_on": eval_on, **metrics.results_dict}
            rows.append(row)

    summary_path = Path(project) / "baseline_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary table written to {summary_path}")
    header = list(rows[0].keys())
    print("  ".join(f"{h:>14}" for h in header))
    for row in rows:
        print("  ".join(f"{row[h]!s:>14.14}" for h in header))


if __name__ == "__main__":
    main()

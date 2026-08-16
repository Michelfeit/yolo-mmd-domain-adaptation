#!/usr/bin/env python3
"""Evaluate every adapted/no-MMD checkpoint that contributed a result to the paper
on its own SOURCE domain's val split, to measure source-domain retention after
adaptation/fine-tuning on the target domain. Baseline/Solo checkpoints are excluded
-- they never see a source domain during training, so retention doesn't apply.

Checkpoint list and source-domain assignment cross-referenced against
docs/experiment_summary_v6.tex, docs/experiment_summary_v7.tex, and
docs/freeze_epoch_ablation.tex.

    python scripts/eval_source_domain_retention.py
"""

import csv
from pathlib import Path

from ultralytics import YOLO

IMGSZ = 1920

# (label, checkpoint path, source-domain name, source-domain eval yaml)
RUNS = [
    # source = real_large
    ("real_large_source_syn_small_target_naive",
     "runs/final_eval/adapt/real_large_source_syn_small_target_naive/weights/best.pt",
     "real_large", "configs/pretrain/real_large.yaml"),
    ("real_large_source_syn_small_target_reg-4",
     "runs/final_eval/adapt/real_large_source_syn_small_target_reg-4/weights/best.pt",
     "real_large", "configs/pretrain/real_large.yaml"),
    ("real_large_source_syn_small_target_reg_smaller_weight-4",
     "runs/final_eval/adapt/real_large_source_syn_small_target_reg_smaller_weight-4/weights/best.pt",
     "real_large", "configs/pretrain/real_large.yaml"),
    ("syn_small_from_real_large_1920_no_mmd",
     "runs/final_eval/finetune_no_mmd/syn_small_from_real_large_1920/weights/best.pt",
     "real_large", "configs/pretrain/real_large.yaml"),

    # source = syn_large_clearnoon
    ("syn_large_clearnoon_source_real_small_target_naive",
     "runs/final_eval/adapt/syn_large_clearnoon_source_real_small_target_naive/weights/best.pt",
     "syn_large_clearnoon", "configs/pretrain/syn_large_clearnoon.yaml"),
    ("syn_large_clearnoon_source_real_small_target_reg",
     "runs/final_eval/adapt/syn_large_clearnoon_source_real_small_target_reg/weights/best.pt",
     "syn_large_clearnoon", "configs/pretrain/syn_large_clearnoon.yaml"),
    ("syn_large_clearnoon_source_real_small_target_reg_smaller_weight",
     "runs/final_eval/adapt/syn_large_clearnoon_source_real_small_target_reg_smaller_weight/weights/best.pt",
     "syn_large_clearnoon", "configs/pretrain/syn_large_clearnoon.yaml"),
    ("real_small_from_syn_large_clearnoon_no_mmd",
     "runs/final_eval/finetune_no_mmd/real_small_from_syn_large_clearnoon/weights/best.pt",
     "syn_large_clearnoon", "configs/pretrain/syn_large_clearnoon.yaml"),

    # source = syn_large
    ("syn_large_1920_source_real_small_target_naive",
     "runs/final_eval/adapt/syn_large_1920_source_real_small_target_naive/weights/best.pt",
     "syn_large", "configs/pretrain/syn_large.yaml"),
    ("syn_large_1920_source_real_small_target_reg_fr1",
     "runs/final_eval/adapt/syn_large_1920_source_real_small_target_reg/weights/best.pt",
     "syn_large", "configs/pretrain/syn_large.yaml"),
    ("syn_large_1920_source_real_small_target_reg_fr5",
     "runs/final_eval/adapt/syn_large_1920_source_real_small_target_reg_fr5/weights/best.pt",
     "syn_large", "configs/pretrain/syn_large.yaml"),
    ("syn_large_1920_source_real_small_target_reg_smaller_weight_fr1",
     "runs/final_eval/adapt/syn_large_1920_source_real_small_target_reg_smaller_weight/weights/best.pt",
     "syn_large", "configs/pretrain/syn_large.yaml"),
    ("syn_large_1920_source_real_small_target_reg_smaller_weight_fr5",
     "runs/final_eval/adapt/syn_large_1920_source_real_small_target_reg_smaller_weight_fr5/weights/best.pt",
     "syn_large", "configs/pretrain/syn_large.yaml"),
    ("real_small_from_syn_large_1920_no_mmd",
     "runs/final_eval/finetune_no_mmd/real_small_from_syn_large_1920/weights/best.pt",
     "syn_large", "configs/pretrain/syn_large.yaml"),

    # source = syn_large_vehicle_only
    ("syn_large_vehicle_only_source_real_small_vehicle_only_target_naive",
     "runs/final_eval/adapt/syn_large_vehicle_only_source_real_small_vehicle_only_target_naive/weights/best.pt",
     "syn_large_vehicle_only", "configs/pretrain/syn_large_vehicle_only.yaml"),
    ("syn_large_vehicle_only_source_real_small_vehicle_only_target_reg",
     "runs/final_eval/adapt/syn_large_vehicle_only_source_real_small_vehicle_only_target_reg/weights/best.pt",
     "syn_large_vehicle_only", "configs/pretrain/syn_large_vehicle_only.yaml"),
    ("syn_large_vehicle_only_source_real_small_vehicle_only_target_reg_smaller_weight",
     "runs/final_eval/adapt/syn_large_vehicle_only_source_real_small_vehicle_only_target_reg_smaller_weight/weights/best.pt",
     "syn_large_vehicle_only", "configs/pretrain/syn_large_vehicle_only.yaml"),
    ("real_small_vehicle_only_from_syn_large_vehicle_only_no_mmd",
     "runs/final_eval/finetune_no_mmd/real_small_vehicle_only_from_syn_large_vehicle_only/weights/best.pt",
     "syn_large_vehicle_only", "configs/pretrain/syn_large_vehicle_only.yaml"),

    # source = syn_xlarge
    ("syn_xlarge_yolo_n_source_real_small_target_reg_fr5",
     "runs/final_eval/adapt/syn_xlarge_yolo_n_source_real_small_target_reg_fr5/weights/best.pt",
     "syn_xlarge", "configs/pretrain/syn_xlarge.yaml"),
    ("syn_xlarge_source_real_large_target_reg_fr5_freeze6",
     "runs/yolov10n/adapt/syn_xlarge_source_real_large_target_reg_fr5/weights/best.pt",
     "syn_xlarge", "configs/pretrain/syn_xlarge.yaml"),
    ("syn_xlarge_source_real_large_target_reg_fr5_stronger_weight-5_freeze25",
     "runs/yolov10n/adapt/syn_xlarge_source_real_large_target_reg_fr5_stronger_weight-5/weights/best.pt",
     "syn_xlarge", "configs/pretrain/syn_xlarge.yaml"),
    ("syn_xlarge_source_real_small_target_yolov10m",
     "runs/yolov10m/adapt/syn_xlarge_source_real_small_target/weights/best.pt",
     "syn_xlarge", "configs/pretrain/syn_xlarge.yaml"),
]


def main() -> None:
    project = Path("runs/source_domain_retention")
    project.mkdir(parents=True, exist_ok=True)
    rows = []

    for label, ckpt, source_domain, data_yaml in RUNS:
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            print(f"SKIP {label}: checkpoint not found at {ckpt}")
            continue
        if not Path(data_yaml).exists():
            print(f"SKIP {label}: data yaml not found at {data_yaml}")
            continue

        print(f"\n=== {label} ===\n  checkpoint: {ckpt}\n  source domain: {source_domain} ({data_yaml})")
        model = YOLO(str(ckpt_path))
        metrics = model.val(data=data_yaml, imgsz=IMGSZ, project=str(project), name=label, plots=False)
        row = {"label": label, "source_domain": source_domain, "checkpoint": ckpt, **metrics.results_dict}
        rows.append(row)

    if not rows:
        raise SystemExit("No checkpoints evaluated.")

    summary_path = project / "source_domain_retention_summary.csv"
    fieldnames = sorted({k for row in rows for k in row})
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train a single pretrain baseline (stock ultralytics supervised training, no domain
adaptation) on one dataset config.

GPU is selected via CUDA_VISIBLE_DEVICES in front of the command. Meant to be launched
once per config/GPU, e.g. to train all 4 size/domain variants in parallel on 4 GPUs:

    CUDA_VISIBLE_DEVICES=0 python scripts/pretrain.py --data configs/pretrain/real_small.yaml &
    CUDA_VISIBLE_DEVICES=1 python scripts/pretrain.py --data configs/pretrain/real_large.yaml &
    CUDA_VISIBLE_DEVICES=2 python scripts/pretrain.py --data configs/pretrain/syn_small.yaml &
    CUDA_VISIBLE_DEVICES=3 python scripts/pretrain.py --data configs/pretrain/syn_large.yaml &
    wait

For multi-GPU DDP on a single run instead (e.g. to raise the effective batch size
while keeping the per-GPU batch, and thus memory footprint, unchanged from a
single-GPU run), pass --device explicitly -- ultralytics only enables DDP when device
is a comma-separated multi-index string, CUDA_VISIBLE_DEVICES masking alone does not
trigger it. --batch is the TOTAL batch across all listed devices, split evenly
(ultralytics does batch_size // world_size internally), so to keep e.g. 8 per GPU
across 4 GPUs, pass --batch 32 --device 0,1,2,3:

    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/pretrain.py --data configs/pretrain/syn_xlarge.yaml --batch 32 --device 0,1,2,3

Saves to runs/<model_tag>/pretrain/<name>/ (name defaults to the data yaml's stem), the
same layout scripts/pretrain_baselines.py uses, so results across runs are easy to
collect and compare afterwards.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="path to a pretrain dataset yaml, e.g. configs/pretrain/real_small.yaml")
    parser.add_argument("--model", default="yolov10n.pt", help="'.pt' = COCO-pretrained weights, '.yaml' = random-init architecture only")
    parser.add_argument("--model-tag", default=None, help="run-directory tag, defaults to the model stem")
    parser.add_argument("--name", default=None, help="run name under the project dir, defaults to the data yaml's stem")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=2080)
    parser.add_argument("--batch", type=int, default=16, help="TOTAL batch size across all --device GPUs, not per-GPU")
    parser.add_argument("--device", default=None, help="e.g. 0,1,2,3 for multi-GPU DDP; omit to use whatever CUDA_VISIBLE_DEVICES exposes on a single GPU")
    parser.add_argument("--project", default=None, help="defaults to runs/<model_tag>/pretrain")
    args = parser.parse_args()

    model_tag = args.model_tag or Path(args.model).stem
    name = args.name or Path(args.data).stem
    # Must be absolute: ultralytics silently nests *relative* project paths under its own
    # default runs/<task>/ root (see get_save_dir), which would break the checkpoint-path
    # assumptions collect-results tooling relies on. Applies to an explicitly-passed
    # --project too, not just the default -- a bare relative path there hits the exact
    # same nesting bug.
    project = str(Path(args.project).resolve()) if args.project else str(Path(f"runs/{model_tag}/pretrain").resolve())

    print(f"=== Training {name} ({args.data}) ===")
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=project,
        name=name,
    )
    print(f"\nDone. Checkpoint: {Path(project) / name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()

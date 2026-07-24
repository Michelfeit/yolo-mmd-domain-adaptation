#!/usr/bin/env python3
"""Build a bigger synthetic (FlyAwareV2) dataset, ~10k images, to better support the
"synthetic is cheap and abundant" premise of the study. Independent addition alongside
syn_small/syn_large (built by scripts/build_domain_datasets.py) -- not a superset or
subset relationship, just another evenly-spaced sample at a larger per-combo count.

Same sampling approach as build_domain_datasets.py's synthetic path: for each of the
32 (town, weather) combinations, sort images (frame sequence order), pick
--train-per-combo evenly-spaced frames for train, then --val-per-combo evenly-spaced
frames from the remaining (non-train) pool for val.

Non-destructive: only reads from --synth-root, only writes under
<out-root>/<out-name>/. Safe to re-run: existing symlinks/label files at the
destination are left in place.

    python scripts/build_syn_xlarge_dataset.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

DEFAULT_SYNTH_ROOT = Path("/mnt/15TB-NVME/chr65046/FlyAwareV2/synth")
DEFAULT_OUT_ROOT = Path("/mnt/15TB-NVME/chr65046/datasets")
DEFAULT_OUT_NAME = "syn_xlarge"

DEFAULT_TOWNS = [
    "Town01_Opt_120",
    "Town02_Opt_120",
    "Town03_Opt_120",
    "Town04_Opt_120",
    "Town05_Opt_120",
    "Town06_Opt_120",
    "Town07_Opt_120",
    "Town10HD_Opt_120",
]
DEFAULT_WEATHERS = ["ClearNight", "ClearNoon", "HardRainNoon", "MidFoggyNoon"]


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def evenly_spaced_indices(n: int, k: int) -> list[int]:
    """k evenly-spaced, strictly increasing indices in range(n). Requires k <= n."""
    if k > n:
        raise ValueError(f"cannot pick {k} evenly-spaced items from {n}")
    return [int(i * n / k) for i in range(k)]


def stage_split(pairs: list[tuple[Path, Path]], images_out: Path, labels_out: Path) -> tuple[int, int]:
    """Symlink images + copy labels for `pairs` (img_path, label_path), renumbered
    sequentially starting at 00001. Skips files that already exist at the destination."""
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    n_missing = 0
    for i, (img, label) in enumerate(pairs, start=1):
        stem = f"{i:05d}"
        dst_img = images_out / f"{stem}{img.suffix.lower()}"
        if not dst_img.exists():
            dst_img.symlink_to(img.resolve())

        dst_label = labels_out / f"{stem}.txt"
        if label.exists():
            if not dst_label.exists():
                shutil.copy2(label, dst_label)
        else:
            n_missing += 1

    return len(pairs), n_missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synth-root", type=Path, default=DEFAULT_SYNTH_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--out-name", default=DEFAULT_OUT_NAME)
    parser.add_argument("--towns", nargs="+", default=DEFAULT_TOWNS)
    parser.add_argument("--weathers", nargs="+", default=DEFAULT_WEATHERS)
    parser.add_argument("--height", default="height50m")
    parser.add_argument("--train-per-combo", type=int, default=320, help="32 combos x 320 = 10240 train images")
    parser.add_argument("--val-per-combo", type=int, default=32, help="32 combos x 32 = 1024 val images")
    args = parser.parse_args()

    train_pairs: list[tuple[Path, Path]] = []
    val_pairs: list[tuple[Path, Path]] = []

    print(f"=== {args.out_name}: {len(args.towns)} towns x {len(args.weathers)} weathers ===")
    for town in args.towns:
        for weather in args.weathers:
            combo_dir = args.synth_root / town / weather / args.height
            rgb_dir = combo_dir / "rgb"
            label_dir = combo_dir / "bb_small"

            images = list_images(rgb_dir)
            n = len(images)
            if n == 0:
                print(f"  [{town}/{weather}] no images found at {rgb_dir}, skipping")
                continue

            train_idx = evenly_spaced_indices(n, args.train_per_combo)
            remaining = [i for i in range(n) if i not in set(train_idx)]
            val_idx = [remaining[p] for p in evenly_spaced_indices(len(remaining), args.val_per_combo)]

            train_pairs.extend((images[i], label_dir / f"{images[i].stem}.txt") for i in train_idx)
            val_pairs.extend((images[i], label_dir / f"{images[i].stem}.txt") for i in val_idx)

            print(f"  [{town}/{weather}] available={n} train={len(train_idx)} val={len(val_idx)}")

    out_root = args.out_root / args.out_name
    n_train, missing_train = stage_split(train_pairs, out_root / "images" / "train", out_root / "labels" / "train")
    n_val, missing_val = stage_split(val_pairs, out_root / "images" / "val", out_root / "labels" / "val")
    print(f"\n-> {args.out_name}: train={n_train} ({missing_train} missing labels), val={n_val} ({missing_val} missing labels)")


if __name__ == "__main__":
    main()

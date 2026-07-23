#!/usr/bin/env python3
"""Build four subsampled train/val YOLO-format dataset trees for domain adaptation
experiments: syn_small, syn_large (from FlyAwareV2 synthetic) and real_small, real_large
(from VisDrone real).

Each output tree follows the standard ultralytics sibling layout so `img2label_paths`
(string-replace of "/images/" -> "/labels/") works unmodified:

    <out_root>/<variant>/images/train/00001.jpg
    <out_root>/<variant>/images/val/00001.jpg
    <out_root>/<variant>/labels/train/00001.txt
    <out_root>/<variant>/labels/val/00001.txt

Images are symlinked to the originals (never copied); labels are copied (tiny, and
avoids fragility if the source label trees are reorganized later). Filenames are
renumbered sequentially per split (00001, 00002, ...) since synthetic frames from
different (town, weather) combinations share filenames and would otherwise collide
when merged into one flat directory.

Non-destructive: only ever reads from the source trees, only ever writes under
`--out-root`. Safe to re-run: existing symlinks/label files at the destination are
left in place rather than recreated, so re-running with the same parameters is a no-op,
and re-running after widening a count only adds the newly-selected files.

Synthetic (FlyAwareV2) sampling, per (town, weather) combination:
  - list `rgb/*.jpg` sorted (frame sequence order), N frames (~3000; verified per-combo,
    not assumed)
  - pick `--syn-large-train-per-combo` (default 200) evenly-spaced frames -> that
    combo's contribution to syn_large/train
  - take every `--syn-small-train-ratio`-th (default 4) frame of that same selection
    (50) -> that combo's contribution to syn_small/train (strict subset, not an
    independent sample)
  - from the remaining (non-train) frames, pick `--syn-large-val-per-combo`
    (default 20) evenly-spaced frames -> syn_large/val contribution
  - take every `--syn-small-val-ratio`-th (default 4) frame of that val selection (5)
    -> syn_small/val contribution (also a strict subset)

Real (VisDrone) sampling:
  - real_large: full existing train split as-is, full existing val split as-is
  - real_small: evenly-spaced stride subsample of train down to
    `--real-small-train-target` (default 1600); val left at full size (already small
    relative to the target, see README discussion)

Usage:
    python scripts/build_domain_datasets.py
    python scripts/build_domain_datasets.py --only synth
    python scripts/build_domain_datasets.py --syn-large-train-per-combo 300  # re-run with new params
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

DEFAULT_SYNTH_ROOT = Path("/mnt/15TB-NVME/chr65046/FlyAwareV2/synth")
DEFAULT_REAL_ROOT = Path("/mnt/15TB-NVME/vca-shared/VisDrone-Dataset/Visdrone_2021/yolo_splits_no_bikes")
DEFAULT_OUT_ROOT = Path("/mnt/15TB-NVME/chr65046/datasets")

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


def build_synthetic(
    synth_root: Path,
    out_root: Path,
    towns: list[str],
    weathers: list[str],
    height: str,
    large_train_per_combo: int,
    small_train_ratio: int,
    large_val_per_combo: int,
    small_val_ratio: int,
) -> None:
    large_train: list[tuple[Path, Path]] = []
    small_train: list[tuple[Path, Path]] = []
    large_val: list[tuple[Path, Path]] = []
    small_val: list[tuple[Path, Path]] = []

    print(f"\n=== synthetic (FlyAwareV2): {len(towns)} towns x {len(weathers)} weathers ===")
    for town in towns:
        for weather in weathers:
            combo_dir = synth_root / town / weather / height
            rgb_dir = combo_dir / "rgb"
            label_dir = combo_dir / "bb_small"

            images = list_images(rgb_dir)
            n = len(images)
            if n == 0:
                print(f"  [{town}/{weather}] no images found at {rgb_dir}, skipping")
                continue

            large_train_idx = evenly_spaced_indices(n, large_train_per_combo)
            small_train_idx = large_train_idx[::small_train_ratio]

            remaining = [i for i in range(n) if i not in set(large_train_idx)]
            large_val_positions = evenly_spaced_indices(len(remaining), large_val_per_combo)
            large_val_idx = [remaining[p] for p in large_val_positions]
            small_val_idx = large_val_idx[::small_val_ratio]

            def to_pairs(idxs: list[int]) -> list[tuple[Path, Path]]:
                return [(images[i], label_dir / f"{images[i].stem}.txt") for i in idxs]

            large_train.extend(to_pairs(large_train_idx))
            small_train.extend(to_pairs(small_train_idx))
            large_val.extend(to_pairs(large_val_idx))
            small_val.extend(to_pairs(small_val_idx))

            print(
                f"  [{town}/{weather}] available={n} "
                f"large_train={len(large_train_idx)} small_train={len(small_train_idx)} "
                f"large_val={len(large_val_idx)} small_val={len(small_val_idx)}"
            )

    variants = {
        "syn_large": (large_train, large_val),
        "syn_small": (small_train, small_val),
    }
    for variant, (train_pairs, val_pairs) in variants.items():
        n_train, missing_train = stage_split(
            train_pairs, out_root / variant / "images" / "train", out_root / variant / "labels" / "train"
        )
        n_val, missing_val = stage_split(
            val_pairs, out_root / variant / "images" / "val", out_root / variant / "labels" / "val"
        )
        print(
            f"  -> {variant}: train={n_train} ({missing_train} missing labels), "
            f"val={n_val} ({missing_val} missing labels)"
        )


def build_real(real_root: Path, out_root: Path, small_train_target: int) -> None:
    print("\n=== real (VisDrone) ===")

    train_images_dir = real_root / "train" / "images"
    train_labels_dir = real_root / "train" / "labels"
    val_images_dir = real_root / "val" / "images"
    val_labels_dir = real_root / "val" / "labels"

    train_images = list_images(train_images_dir)
    val_images = list_images(val_images_dir)
    print(f"  train available={len(train_images)}, val available={len(val_images)}")

    large_train = [(img, train_labels_dir / f"{img.stem}.txt") for img in train_images]
    large_val = [(img, val_labels_dir / f"{img.stem}.txt") for img in val_images]

    small_train_idx = evenly_spaced_indices(len(train_images), min(small_train_target, len(train_images)))
    small_train = [large_train[i] for i in small_train_idx]
    # val is already small relative to the target train sizes; reuse it as-is for
    # real_small rather than subsampling it further (see script docstring).
    small_val = large_val

    print(f"  real_large: train={len(large_train)} selected of {len(train_images)}, val={len(large_val)}")
    print(f"  real_small: train={len(small_train)} selected of {len(train_images)}, val={len(small_val)}")

    variants = {
        "real_large": (large_train, large_val),
        "real_small": (small_train, small_val),
    }
    for variant, (train_pairs, val_pairs) in variants.items():
        n_train, missing_train = stage_split(
            train_pairs, out_root / variant / "images" / "train", out_root / variant / "labels" / "train"
        )
        n_val, missing_val = stage_split(
            val_pairs, out_root / variant / "images" / "val", out_root / variant / "labels" / "val"
        )
        print(
            f"  -> {variant}: train={n_train} ({missing_train} missing labels), "
            f"val={n_val} ({missing_val} missing labels)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synth-root", type=Path, default=DEFAULT_SYNTH_ROOT)
    parser.add_argument("--real-root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--only", choices=["all", "synth", "real"], default="all")

    parser.add_argument("--towns", nargs="+", default=DEFAULT_TOWNS)
    parser.add_argument("--weathers", nargs="+", default=DEFAULT_WEATHERS)
    parser.add_argument("--height", default="height50m")
    parser.add_argument("--syn-large-train-per-combo", type=int, default=200)
    parser.add_argument("--syn-small-train-ratio", type=int, default=4)
    parser.add_argument("--syn-large-val-per-combo", type=int, default=20)
    parser.add_argument("--syn-small-val-ratio", type=int, default=4)

    parser.add_argument("--real-small-train-target", type=int, default=1600)

    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    if args.only in ("all", "synth"):
        build_synthetic(
            args.synth_root,
            args.out_root,
            args.towns,
            args.weathers,
            args.height,
            args.syn_large_train_per_combo,
            args.syn_small_train_ratio,
            args.syn_large_val_per_combo,
            args.syn_small_val_ratio,
        )

    if args.only in ("all", "real"):
        build_real(args.real_root, args.out_root, args.real_small_train_target)


if __name__ == "__main__":
    main()

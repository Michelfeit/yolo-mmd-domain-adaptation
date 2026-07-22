#!/usr/bin/env python3
"""Build a stride-subsampled dataset split from an existing on-disk layout of
{root}/{split}/images + {root}/{split}/<labels-dir>/labels.

Non-destructive: only creates a new {root}/<out-name>/ tree (images symlinked back to
the originals, labels copied — mirrors the existing convention where each per-class
label folder, e.g. labels_vehicle, is its own real `labels/` dir sitting next to an
`images` symlink). Safe to re-run: existing symlinks/files in the destination are left
in place rather than recreated.

    python scripts/make_anchor_subset.py /media/chr65046/4TB-SSD/VisDrone_20250110 --labels-dir labels_vehicle
    python scripts/make_anchor_subset.py /media/chr65046/4TB-SSD/Syndrone --labels-dir labels_vehicle
"""

import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_split(root: Path, split: str, labels_dir: str, out_name: str, stride: int) -> None:
    src_images = root / split / "images"
    src_labels = root / split / labels_dir / "labels"
    if not src_images.is_dir():
        print(f"  [{split}] no images dir at {src_images}, skipping")
        return

    dst_images = root / out_name / split / "images"
    dst_labels = root / out_name / split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    image_files = sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    selected = image_files[::stride]

    n_missing_label = 0
    for img in selected:
        dst_img = dst_images / img.name
        if not dst_img.exists():
            dst_img.symlink_to(img.resolve())

        label = src_labels / f"{img.stem}.txt"
        dst_label = dst_labels / label.name
        if label.exists():
            if not dst_label.exists():
                shutil.copy2(label, dst_label)
        else:
            n_missing_label += 1

    print(f"  [{split}] {len(image_files)} -> {len(selected)} images (stride={stride}), {n_missing_label} missing labels")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="dataset root containing train/val/test splits")
    parser.add_argument("--labels-dir", default="labels_vehicle", help="per-class label folder name")
    parser.add_argument("--out-name", default="anchor", help="name of the new top-level subset directory")
    parser.add_argument("--stride", type=int, default=3, help="keep every Nth image")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    print(f"{args.root} -> {args.out_name}/ (stride={args.stride})")
    for split in args.splits:
        build_split(args.root, split, args.labels_dir, args.out_name, args.stride)


if __name__ == "__main__":
    main()

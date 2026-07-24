#!/usr/bin/env python3
"""Build vehicle-only copies of the 4 existing dataset trees (syn_small, syn_large,
real_small, real_large): same images (symlinked, resolved to the true original file
rather than double-linked through the existing dataset's own symlink), labels
filtered down to the vehicle class only, remapped from class id 1 (in the
person=0/vehicle=1 source labels) to class id 0 -- i.e. single-class output
(nc=1, names=[vehicle]).

Non-destructive: only ever reads from the existing dataset trees under
--datasets-root, never modifies them; only writes new <variant>_vehicle_only/ trees.
Safe to re-run: existing symlinks/label files at the destination are left in place.
Images with no vehicle instances still get an (empty) label file, same as any
standard YOLO background image -- they are not dropped.

    python scripts/build_vehicle_only_datasets.py
    python scripts/build_vehicle_only_datasets.py --variants syn_large real_large
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_DATASETS_ROOT = Path("/mnt/15TB-NVME/chr65046/datasets")
DEFAULT_VARIANTS = ["syn_small", "syn_large", "real_small", "real_large"]
SPLITS = ["train", "val"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

PERSON_CLASS = 0
VEHICLE_CLASS = 1


def filter_and_remap_label(src_label: Path) -> list[str]:
    """Keep only vehicle-class lines, remapped to class id 0. Person lines dropped."""
    if not src_label.exists():
        return []
    lines = []
    for line in src_label.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if int(parts[0]) == VEHICLE_CLASS:
            lines.append(" ".join(["0", *parts[1:]]))
    return lines


def build_variant(datasets_root: Path, variant: str) -> None:
    src_root = datasets_root / variant
    dst_root = datasets_root / f"{variant}_vehicle_only"
    print(f"\n=== {variant} -> {dst_root.name} ===")

    for split in SPLITS:
        src_images = src_root / "images" / split
        src_labels = src_root / "labels" / split
        if not src_images.is_dir():
            print(f"  [{split}] no images dir at {src_images}, skipping")
            continue

        dst_images = dst_root / "images" / split
        dst_labels = dst_root / "labels" / split
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        images = sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        n_with_vehicle = 0
        for img in images:
            dst_img = dst_images / img.name
            if not dst_img.exists():
                dst_img.symlink_to(img.resolve())

            lines = filter_and_remap_label(src_labels / f"{img.stem}.txt")
            if lines:
                n_with_vehicle += 1

            dst_label = dst_labels / f"{img.stem}.txt"
            if not dst_label.exists():
                dst_label.write_text("\n".join(lines) + ("\n" if lines else ""))

        print(f"  [{split}] {len(images)} images, {n_with_vehicle} with >=1 vehicle instance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    args = parser.parse_args()

    for variant in args.variants:
        build_variant(args.datasets_root, variant)


if __name__ == "__main__":
    main()

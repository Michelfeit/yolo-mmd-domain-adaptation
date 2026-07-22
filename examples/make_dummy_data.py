#!/usr/bin/env python3
"""Generate a tiny, fully synthetic dummy dataset so the pipeline can be run end-to-end
with zero external data and zero licensing concerns (no real dataset images are used or
redistributed anywhere in this repo).

Two domains, deliberately drawn with different background palettes to give a real (if
easy) visual domain gap: "real" (green/brown, grass-like) and "synthetic" (blue/gray,
rendered-look). Each image has 1-3 solid-color rectangles standing in for "vehicle",
with matching YOLO-format labels -- an easy but genuine detection task, not random
noise with meaningless labels.

    python examples/make_dummy_data.py

Regenerate anytime; this script is the source of truth, not the generated files.
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image

OUT_DIR = Path(__file__).parent / "dummy_data"
IMG_SIZE = 128
SPLITS = {"train": 8, "val": 2}

# (background low, background high) RGB ranges per domain -- gives each domain a
# distinct look while keeping the "vehicle" rectangles visually consistent across both.
DOMAIN_BG = {
    "real": ((40, 70, 30), (90, 120, 70)),  # green/brown, grass-like
    "synthetic": ((60, 70, 110), (110, 120, 170)),  # blue/gray, rendered-look
}
VEHICLE_COLOR = (200, 40, 40)


def make_image(rng: random.Random, bg_lo, bg_hi):
    img = np.stack(
        [rng_uniform_channel(rng, lo, hi) for lo, hi in zip(bg_lo, bg_hi)],
        axis=-1,
    ).astype(np.uint8)
    boxes = []
    n_boxes = rng.randint(1, 3)
    for _ in range(n_boxes):
        w = rng.randint(12, 28)
        h = rng.randint(12, 28)
        cx = rng.randint(w // 2 + 2, IMG_SIZE - w // 2 - 2)
        cy = rng.randint(h // 2 + 2, IMG_SIZE - h // 2 - 2)
        x0, y0, x1, y1 = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2
        img[y0:y1, x0:x1] = VEHICLE_COLOR
        boxes.append((cx / IMG_SIZE, cy / IMG_SIZE, w / IMG_SIZE, h / IMG_SIZE))
    return img, boxes


def rng_uniform_channel(rng: random.Random, lo: int, hi: int) -> np.ndarray:
    return np.full((IMG_SIZE, IMG_SIZE), rng.randint(lo, hi), dtype=np.uint8)


def main() -> None:
    rng = random.Random(0)
    for domain, (bg_lo, bg_hi) in DOMAIN_BG.items():
        for split, count in SPLITS.items():
            img_dir = OUT_DIR / domain / "images" / split
            lbl_dir = OUT_DIR / domain / "labels" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                img, boxes = make_image(rng, bg_lo, bg_hi)
                Image.fromarray(img).save(img_dir / f"{i:03d}.jpg")
                lines = [f"0 {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}" for cx, cy, w, h in boxes]
                (lbl_dir / f"{i:03d}.txt").write_text("\n".join(lines))
            print(f"{domain}/{split}: {count} images")


if __name__ == "__main__":
    main()

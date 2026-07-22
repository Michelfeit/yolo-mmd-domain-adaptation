#!/usr/bin/env python3
"""2D timelapse animation of the PCA feature-space tracker's output.

Reads a pca_features.csv (see dual_domain/pca_tracker.py) and renders an animated GIF
of the source/target point clouds' pc1-pc2 positions evolving over training, using
whatever epochs currently exist in the file. Safe to run while training is still
appending to the same file — a truncated trailing row (from reading mid-write) is
silently dropped rather than crashing.

By default each epoch is one frame (choppy but exact). Pass --duration to instead
render a fixed-length, fixed-fps animation with each point's position linearly
interpolated between its own epoch keyframes -- smooth motion instead of one jump per
epoch, independent of how many epochs actually exist.

    python scripts/pca_timelapse.py runs/yolov10n/adapt/real_source_syn_target/pca_features.csv
    python scripts/pca_timelapse.py <csv> --duration 3 --fps 30 --max-points-per-domain 100
"""

import argparse
import bisect
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

COLORS = {"source": "#5659f0", "target": "#e84865"}


def load_series(
    csv_path: Path, max_points_per_domain: int | None
) -> dict[tuple[str, int], list[tuple[int, float, float]]]:
    """(domain, sample_index) -> sorted [(epoch, pc1, pc2), ...], one entry per point's
    trajectory across epochs. Tolerates a truncated trailing row (concurrent write)."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        parsed = []
        for row in reader:
            if len(row) != len(header):
                continue  # truncated trailing row from a concurrent write
            try:
                parsed.append((
                    int(row[idx["epoch"]]),
                    row[idx["domain"]],
                    int(row[idx["sample_index"]]),
                    float(row[idx["pc1"]]),
                    float(row[idx["pc2"]]),
                ))
            except ValueError:
                continue  # truncated numeric field mid-write

    series: dict[tuple[str, int], list[tuple[int, float, float]]] = defaultdict(list)
    for epoch, domain, sample_index, pc1, pc2 in parsed:
        if max_points_per_domain is not None and sample_index >= max_points_per_domain:
            continue
        series[(domain, sample_index)].append((epoch, pc1, pc2))
    for points in series.values():
        points.sort(key=lambda p: p[0])
    return series


def interpolate_frame(
    series: dict[tuple[str, int], list[tuple[int, float, float]]], t: float
) -> dict[tuple[str, int], tuple[float, float]]:
    """Each point's (pc1, pc2) at continuous "virtual epoch" t, linearly interpolated
    between its own two bracketing keyframes (clamped at the ends)."""
    frame = {}
    for key, points in series.items():
        epochs = [p[0] for p in points]
        if t <= epochs[0]:
            frame[key] = (points[0][1], points[0][2])
        elif t >= epochs[-1]:
            frame[key] = (points[-1][1], points[-1][2])
        else:
            i = bisect.bisect_right(epochs, t) - 1
            e0, x0, y0 = points[i]
            e1, x1, y1 = points[i + 1]
            frac = (t - e0) / (e1 - e0)
            frame[key] = (x0 + frac * (x1 - x0), y0 + frac * (y1 - y0))
    return frame


def render_frame(ax, series, t: float, xlim, ylim, labels: dict[str, str]) -> None:
    frame = interpolate_frame(series, t)
    ax.clear()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("pc1")
    ax.set_ylabel("pc2")
    ax.set_title(f"Epoch {int(t)}")
    for domain, color in COLORS.items():
        pts = [xy for (dom, _idx), xy in frame.items() if dom == domain]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=14, alpha=0.6, color=color, label=labels[domain])
    ax.legend(loc="upper right")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="GIF path (default <csv-dir>/pca_timelapse.gif), or with --epochs, the output "
        "directory for the PNGs (default <csv-dir>/pca_frames)",
    )
    parser.add_argument(
        "--max-points-per-domain",
        type=int,
        default=None,
        help="keep only sample_index < N per domain (same fixed points every epoch), default: all",
    )
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="if set, render this many seconds at --fps with smooth interpolated motion, "
        "instead of one (choppy) frame per epoch",
    )
    parser.add_argument("--max-epoch", type=int, default=None, help="only show epochs up to this one")
    parser.add_argument("--target-label", default="target", help="legend label for the target domain's points")
    parser.add_argument("--source-label", default="source", help="legend label for the source domain's points")
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=None,
        help="if set, save these specific epochs as individual static PNGs (into --out as a "
        "directory) instead of building a GIF animation",
    )
    args = parser.parse_args()
    labels = {"target": args.target_label, "source": args.source_label}

    series = load_series(args.csv, args.max_points_per_domain)
    if not series:
        raise SystemExit(f"No complete rows found in {args.csv} yet.")

    if args.max_epoch is not None:
        series = {key: [p for p in points if p[0] <= args.max_epoch] for key, points in series.items()}
        series = {key: points for key, points in series.items() if points}
        if not series:
            raise SystemExit(f"No rows at or before epoch {args.max_epoch} in {args.csv}.")

    all_epochs = sorted({e for points in series.values() for e, _, _ in points})
    min_e, max_e = all_epochs[0], all_epochs[-1]

    if args.duration:
        n_frames = max(2, round(args.duration * args.fps))
        virtual_epochs = [min_e + (max_e - min_e) * i / (n_frames - 1) for i in range(n_frames)]
    else:
        virtual_epochs = [float(e) for e in all_epochs]

    all_pc1 = [pt[1] for points in series.values() for pt in points]
    all_pc2 = [pt[2] for points in series.values() for pt in points]
    pad_x = 0.05 * (max(all_pc1) - min(all_pc1) or 1.0)
    pad_y = 0.05 * (max(all_pc2) - min(all_pc2) or 1.0)
    xlim = (min(all_pc1) - pad_x, max(all_pc1) + pad_x)
    ylim = (min(all_pc2) - pad_y, max(all_pc2) + pad_y)

    fig, ax = plt.subplots(figsize=(6, 6))

    if args.epochs is not None:
        out_dir = args.out or args.csv.parent / "pca_frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        for epoch in args.epochs:
            render_frame(ax, series, float(epoch), xlim, ylim, labels)
            frame_path = out_dir / f"epoch_{epoch}.png"
            fig.savefig(frame_path, dpi=150)
            print(f"Saved {frame_path}")
        return

    anim = FuncAnimation(fig, lambda t: render_frame(ax, series, t, xlim, ylim, labels), frames=virtual_epochs, interval=1000 / args.fps)
    out_path = args.out or args.csv.parent / "pca_timelapse.gif"
    anim.save(out_path, writer=PillowWriter(fps=args.fps))
    print(f"Saved {len(virtual_epochs)} frames (epoch range {min_e}-{max_e}) to {out_path}")


if __name__ == "__main__":
    main()

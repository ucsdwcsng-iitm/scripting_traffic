#!/usr/bin/env python3
"""Generate per-dataset figures for UE power-estimation captures.

For each capture dataset directory this produces:

  * ``synthetic_traffic/schedule.pdf`` (vector) and ``synthetic_traffic/
    schedule.jpg`` (raster preview) -- the synthetic schedule's DL/UL data
    rate plotted segment-exact (only when ``synthetic_traffic/schedule.json``
    exists). The .jpg is emitted alongside the .pdf by the canonical plotter.

  * ``ground_truth_power/`` at the dataset root -- the three ground-truth power
    files grouped together: the raw ``ina219_power_log_*.csv`` (moved in if it
    was still at the root) plus the INA219 power linearly interpolated onto a
    uniform 0.5 ms grid as ``ground_truth_power_0p5ms.csv`` and a matching
    ``ground_truth_power_0p5ms.pdf`` figure (plus a ``.jpg`` raster preview
    alongside it).

A "dataset" is any directory that holds an ``ina219_power_log_*.csv`` (at the
root or already under ``ground_truth_power/``) and/or a
``synthetic_traffic/schedule.json``. Any path under a directory named
``archive`` is skipped.

Usage:
  python3 generate_dataset_figures.py [PATH ...]

PATH may be a single dataset directory or a parent (e.g. data/qualcomm) under
which datasets are discovered recursively. With no PATH, defaults to the
qualcomm and pixel data roots of this repository.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

POWER_SAMPLE_INTERVAL_S = 0.0005  # 0.5 ms
POWER_BASENAME = "ground_truth_power_0p5ms"
SCHEDULE_FIG_NAME = "schedule.pdf"
# All ground-truth power files (raw INA219 log + interpolated CSV + figure) are
# grouped under this subfolder at the dataset root.
GROUND_TRUTH_SUBDIR = "ground_truth_power"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def find_datasets(root: str):
    """Yield dataset directories under ``root`` (skipping ``archive`` trees)."""
    root = os.path.abspath(root)
    if is_dataset_dir(root):
        yield root
        return
    for dirpath, dirnames, _ in os.walk(root):
        # Never descend into archive directories.
        dirnames[:] = [d for d in dirnames if d != "archive"]
        if is_dataset_dir(dirpath):
            yield dirpath
            # A dataset dir's own subfolders (processed/, srsRAN/, ...) are not
            # themselves datasets; stop descending.
            dirnames[:] = []


def is_dataset_dir(path: str) -> bool:
    if os.path.basename(path) == "archive":
        return False
    if find_ina219_csv(path):
        return True
    if os.path.isfile(os.path.join(path, "synthetic_traffic", "schedule.json")):
        return True
    return False


def find_ina219_csv(path: str):
    """Find the raw INA219 log, whether already grouped or still at the root."""
    for base in (os.path.join(path, GROUND_TRUTH_SUBDIR), path):
        matches = sorted(glob.glob(os.path.join(base, "ina219_power_log_*.csv")))
        if matches:
            return matches[0]
    return None


# --------------------------------------------------------------------------- #
# Schedule figure
# --------------------------------------------------------------------------- #
def plot_schedule(schedule_path: str, out_path: str) -> bool:
    """Render the schedule to PDF via the canonical segment-exact plotter in
    synthetic_traffic/plot_schedule.py, so fine (sub-second) schedules keep their
    true texture instead of being averaged onto a 1 s grid."""
    project_root = os.path.dirname(REPO_ROOT)  # repo root holds synthetic_traffic/
    plotter_dir = os.path.join(project_root, "synthetic_traffic")
    if plotter_dir not in sys.path:
        sys.path.insert(0, plotter_dir)
    try:
        import plot_schedule as canonical
    except ImportError:
        print(f"  ! synthetic_traffic/plot_schedule.py not importable from "
              f"{plotter_dir}, skipping schedule figure", file=sys.stderr)
        return False
    return canonical.render(schedule_path, out_path) is not None


# --------------------------------------------------------------------------- #
# Ground-truth power figure + interpolated CSV
# --------------------------------------------------------------------------- #
def load_ina219_power(csv_path: str):
    """Return (epoch_s, power_w) arrays, sorted and de-duplicated by time."""
    epochs, powers = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        ecol = "Timestamp_Epoch(s)"
        pcol = "Power (W)"
        if reader.fieldnames is None or ecol not in reader.fieldnames or pcol not in reader.fieldnames:
            raise ValueError(
                f"unexpected INA219 columns {reader.fieldnames}; "
                f"need '{ecol}' and '{pcol}'"
            )
        for row in reader:
            try:
                epochs.append(float(row[ecol]))
                powers.append(float(row[pcol]))
            except (TypeError, ValueError):
                continue
    epoch = np.asarray(epochs, dtype=float)
    power = np.asarray(powers, dtype=float)
    order = np.argsort(epoch, kind="stable")
    epoch, power = epoch[order], power[order]
    keep = np.concatenate(([True], np.diff(epoch) > 0))
    return epoch[keep], power[keep]


def interpolate_power(epoch: np.ndarray, power: np.ndarray):
    """Linearly interpolate power onto a uniform 0.5 ms grid (relative time)."""
    t0 = epoch[0]
    rel = epoch - t0
    span = rel[-1]
    n = int(np.floor(span / POWER_SAMPLE_INTERVAL_S)) + 1
    grid = np.arange(n) * POWER_SAMPLE_INTERVAL_S
    grid = grid[grid <= span]
    interp = np.interp(grid, rel, power)
    return t0, grid, interp


def write_power_csv(out_path: str, t0: float, grid: np.ndarray, power: np.ndarray):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time_s", "Power (W)", "Timestamp_Epoch(s)"])
        for ts, p in zip(grid, power):
            writer.writerow([f"{ts:.4f}", f"{p:.6f}", f"{t0 + ts:.6f}"])


def plot_power(grid: np.ndarray, power: np.ndarray, out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grid, power, color="#2ca02c", linewidth=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    ax.set_xlim(0, grid[-1] if len(grid) else 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    mean_w = float(np.mean(power)) if len(power) else 0.0
    ax.set_title(
        f"Ground-truth power (INA219, linear-interp @ 0.5 ms)\n"
        f"{title}  mean={mean_w:.3f} W",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    # A raster sibling for quick previews/thumbnails (galleries, READMEs); the PDF
    # stays the vector source of truth. Mirrors the schedule plotter's behaviour.
    fig.savefig(os.path.splitext(out_path)[0] + ".jpg", dpi=150)
    plt.close(fig)


def process_power(dataset_dir: str) -> bool:
    csv_path = find_ina219_csv(dataset_dir)
    if not csv_path:
        return False

    # Group all ground-truth power files under ground_truth_power/: move the raw
    # INA219 log in (if still at the root) and write the outputs alongside it.
    gt_dir = os.path.join(dataset_dir, GROUND_TRUTH_SUBDIR)
    os.makedirs(gt_dir, exist_ok=True)
    if os.path.dirname(os.path.abspath(csv_path)) != os.path.abspath(gt_dir):
        grouped = os.path.join(gt_dir, os.path.basename(csv_path))
        os.replace(csv_path, grouped)
        csv_path = grouped

    epoch, power = load_ina219_power(csv_path)
    if len(epoch) < 2:
        print(f"  ! too few INA219 samples, skipping: {csv_path}", file=sys.stderr)
        return False
    t0, grid, interp = interpolate_power(epoch, power)
    out_csv = os.path.join(gt_dir, POWER_BASENAME + ".csv")
    out_pdf = os.path.join(gt_dir, POWER_BASENAME + ".pdf")
    write_power_csv(out_csv, t0, grid, interp)
    plot_power(grid, interp, out_pdf, os.path.basename(dataset_dir.rstrip("/")))

    # Remove stale copies left at the dataset root by an earlier (ungrouped) run.
    for stale in (POWER_BASENAME + ".csv", POWER_BASENAME + ".pdf", POWER_BASENAME + ".jpg"):
        stale_path = os.path.join(dataset_dir, stale)
        if os.path.isfile(stale_path):
            os.remove(stale_path)

    rel = os.path.join(GROUND_TRUTH_SUBDIR, "")
    print(f"  power: {rel}{POWER_BASENAME}.pdf  +  .jpg  +  .csv  +  raw log  ({len(grid)} samples)")
    return True


def process_dataset(dataset_dir: str) -> None:
    print(f"[dataset] {dataset_dir}")
    sched_json = os.path.join(dataset_dir, "synthetic_traffic", "schedule.json")
    if os.path.isfile(sched_json):
        out_pdf = os.path.join(dataset_dir, "synthetic_traffic", SCHEDULE_FIG_NAME)
        if plot_schedule(sched_json, out_pdf):
            out_jpg = os.path.splitext(out_pdf)[0] + ".jpg"
            print(f"  schedule: {out_pdf}  +  {os.path.basename(out_jpg)}")
    process_power(dataset_dir)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="dataset dir(s) or parent root(s)")
    args = ap.parse_args(argv)

    roots = args.paths or [
        os.path.join(REPO_ROOT, "data", "qualcomm"),
        os.path.join(REPO_ROOT, "data", "pixel"),
    ]

    datasets = []
    seen = set()
    for root in roots:
        if not os.path.exists(root):
            print(f"(skip, missing) {root}", file=sys.stderr)
            continue
        for ds in find_datasets(root):
            if ds not in seen:
                seen.add(ds)
                datasets.append(ds)

    if not datasets:
        print("No datasets found.", file=sys.stderr)
        return 1
    for ds in datasets:
        process_dataset(ds)
    print(f"\nDone: {len(datasets)} dataset(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

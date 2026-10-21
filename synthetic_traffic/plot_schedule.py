#!/usr/bin/env python3
"""Plot a synthetic traffic schedule's DL/UL data rate to PDF.

Segment-exact: the schedule is a piecewise-constant rate function, so the figure
steps directly on segment boundaries (t0, t0+dur) with no resampling. A 10ms-tick
schedule therefore shows its true sub-second texture instead of being averaged to
a 1s grid. Each render also drops a matching .jpg next to the .pdf (raster preview
for galleries/READMEs; the PDF stays the vector source). Used by schedule_gen.py
(--pdf) and by testbed/capture.sh to populate the schedules/ gallery; also runnable standalone.

  python3 plot_schedule.py schedules/sweep_s1.json            # -> sweep_s1.pdf + sweep_s1.jpg
  python3 plot_schedule.py schedules/sweep_s1.json out.pdf    # -> out.pdf + out.jpg
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DL_COLOR = "#1f77b4"
UL_COLOR = "#d62728"


def _step_arrays(segments):
    """Edges (len m+1) and per-segment DL/UL Mbps (len m) for a step plot."""
    if not segments:
        return None
    edges = [float(segments[0]["t0"])]
    dl, ul = [], []
    for seg in segments:
        edges.append(float(seg["t0"]) + float(seg["dur"]))
        dl.append(float(seg.get("dl_kbps", 0.0)) / 1000.0)
        ul.append(float(seg.get("ul_kbps", 0.0)) / 1000.0)
    return np.asarray(edges), np.asarray(dl), np.asarray(ul)


def render(schedule_path: str, out_path: Optional[str] = None,
           jpg: bool = True) -> Optional[str]:
    """Render schedule JSON at `schedule_path` to a PDF (and, when `jpg`, a
    matching .jpg alongside it). Returns the PDF output path, or None if the
    schedule has no segments."""
    with open(schedule_path) as f:
        schedule = json.load(f)
    arrs = _step_arrays(schedule.get("segments", []))
    if arrs is None:
        print(f"  ! empty/invalid schedule, skipping: {schedule_path}", file=sys.stderr)
        return None
    edges, dl, ul = arrs
    if out_path is None:
        out_path = os.path.splitext(schedule_path)[0] + ".pdf"

    # one extra trailing sample so the last segment draws as a full block (post)
    dl_y = np.append(dl, dl[-1])
    ul_y = np.append(ul, ul[-1])
    caps = schedule.get("caps", {})
    tick_ms = float(schedule.get("tick_s", 0.0)) * 1000.0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(edges, dl_y, where="post", color=DL_COLOR, label="DL", linewidth=1.0)
    ax.step(edges, ul_y, where="post", color=UL_COLOR, label="UL", linewidth=1.0)
    ax.fill_between(edges, dl_y, step="post", color=DL_COLOR, alpha=0.12)
    ax.fill_between(edges, ul_y, step="post", color=UL_COLOR, alpha=0.12)

    # Autoscale to the actual traffic; only draw a cap reference line when it is
    # in range, so a low-rate profile (e.g. voip at ~64 kbps) is not flattened by
    # a 90 Mbps cap line dominating the axis.
    data_max = float(max(dl_y.max(), ul_y.max()))
    top = data_max * 1.18 if data_max > 0 else 1.0
    for key, color in (("dl_kbps", DL_COLOR), ("ul_kbps", UL_COLOR)):
        cap_mbps = caps.get(key, 0) / 1000.0
        if cap_mbps and cap_mbps <= top:
            ax.axhline(cap_mbps, color=color, ls=":", lw=0.9, alpha=0.7)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Data rate (Mbps)")
    ax.set_xlim(float(edges[0]), float(edges[-1]))
    ax.set_ylim(0, top)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    res = f"{tick_ms:g} ms tick" if tick_ms else "segment resolution"
    ax.set_title(
        f"Synthetic schedule data rate ({res}, {len(dl)} segments)\n"
        f"profile={schedule.get('profile')}  seed={schedule.get('seed')}  "
        f"hash={schedule.get('schedule_hash', '')}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    # A raster sibling for quick previews/thumbnails (galleries, READMEs); the PDF
    # stays the vector source of truth.
    if jpg:
        fig.savefig(os.path.splitext(out_path)[0] + ".jpg", dpi=150)
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    schedule_path = argv[0]
    out_path = argv[1] if len(argv) > 1 else None
    result = render(schedule_path, out_path)
    if result:
        print(f"wrote {result}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Paper Fig. 7: synthetic traffic schedules (make_deliverables/figures/fig7_synthetic_schedules.pdf).

"Representative scripted downlink and uplink IP-rate schedules from the synthetic
training corpus": five stacked panels (YouTube, Social scroll, Web browse, Video
call, Rate sweep) sharing a 0-250 s time axis, DL blue / UL red.

No capture data is needed: schedule_gen is deterministic in (profile, seed,
duration, tick, caps), so each panel is rebuilt in memory exactly as
`schedule_gen.py --profile <p> --seed 1 --duration 250` (default 40/10 Mbps
caps, 10 ms tick, 25 kbps segment step) would write it, and drawn segment-exact
like plot_schedule.py.

  python3 make_deliverables/fig7_synthetic_schedules.py [--out path.pdf]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "synthetic_traffic"))
from schedule_gen import (DEFAULT_DL_CAP, DEFAULT_UL_CAP,  # noqa: E402
                          build_envelopes, segment)

DL_C = "#1f77b4"
UL_C = "#d62728"

DURATION_S = 250
TICK_S = 0.01
STEP_KBPS = 25.0
# (title, profile, seed, dl_cap_kbps, ul_cap_kbps)
PANELS = [
    ("YouTube",       "youtube",       1, DEFAULT_DL_CAP, DEFAULT_UL_CAP),
    ("Social scroll", "social_scroll", 1, DEFAULT_DL_CAP, DEFAULT_UL_CAP),
    ("Web browse",    "web_browse",    1, DEFAULT_DL_CAP, DEFAULT_UL_CAP),
    ("Video call",    "video_call",    1, DEFAULT_DL_CAP, DEFAULT_UL_CAP),
    ("Rate sweep",    "sweep",         1, DEFAULT_DL_CAP, DEFAULT_UL_CAP),
]


def schedule_steps(profile, seed, dl_cap, ul_cap):
    """Segment edges (s) and DL/UL Mbps (with a trailing repeat for where='post')."""
    n = int(round(DURATION_S / TICK_S))
    caps = {"dl_kbps": float(dl_cap), "ul_kbps": float(ul_cap)}
    dl, ul = build_envelopes(profile, seed, n, caps, TICK_S)
    segs = segment(dl, ul, TICK_S, STEP_KBPS)
    t = np.array([s["t0"] for s in segs] + [segs[-1]["t0"] + segs[-1]["dur"]])
    d = np.array([s["dl_kbps"] for s in segs] + [segs[-1]["dl_kbps"]]) / 1000.0
    u = np.array([s["ul_kbps"] for s in segs] + [segs[-1]["ul_kbps"]]) / 1000.0
    return t, d, u


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "make_deliverables" / "figures" / "fig7_synthetic_schedules.pdf"))
    args = ap.parse_args()

    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "lines.linewidth": 0.8,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "legend.frameon": False, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 200,
    })

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(3.4, 3.7), sharex=True)
    for ax, (title, profile, seed, dl_cap, ul_cap) in zip(axes, PANELS):
        t, dl, ul = schedule_steps(profile, seed, dl_cap, ul_cap)
        ax.fill_between(t, 0, dl, step="post", color=DL_C, alpha=0.10, linewidth=0)
        ax.fill_between(t, 0, ul, step="post", color=UL_C, alpha=0.10, linewidth=0)
        ax.step(t, dl, where="post", color=DL_C, label="DL")
        ax.step(t, ul, where="post", color=UL_C, label="UL")
        ax.set_title(title, loc="left", pad=1.5)
        ax.set_ylabel("Mbps")
        ax.set_xlim(0, DURATION_S)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(3, steps=[1, 2, 5, 10]))
        ax.grid(True, alpha=0.3, linewidth=0.4)

    axes[0].legend(loc="upper right", ncol=2, handlelength=3.0, columnspacing=2.0,
                   borderaxespad=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout(h_pad=0.3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()

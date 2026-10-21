"""Paper Fig. 4: programmed traffic probes and the UE power they induce.

Two rows x five columns:
  top    -> offered DL (blue) / UL (red) IP rate from <capture>/synthetic_traffic/schedule.json
  bottom -> measured UE power from <capture>/ground_truth_power/ground_truth_power_0p5ms.csv

Panels: DL burst, UL burst, UL triangle, Dual sine, mix. Output:
make_deliverables/figures/fig4_traffic_probes.pdf
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
SYN = REPO / "ue_power_estimation" / "data" / "qualcomm" / "synthetic"
OUT = REPO / "make_deliverables" / "figures"

# paper palette (main.tex): gnbblue, meterorange-ish red, modelpurple
DL_C = "#1f77b4"
UL_C = "#d62728"
PWR_C = "#7256A4"

# (title, capture dir, xlim) -- xlim=None means full duration
PANELS = [
    ("DL burst",    SYN / "synth_dl_pulse_s1_20260623_182448",        None),
    ("UL burst",    SYN / "synth_ul_pulse_s1_20260623_182113",        None),
    ("UL triangle", SYN / "synth_ul_triangle_60s_s1_20260628_131638", None),
    ("Dual sine",   SYN / "synth_dual_sine_120s_s1_20260624_125957",  (0, 60)),
    ("mix",         SYN / "synth_mix_60s_s1_20260628_133610",         None),
]

PWR_YLIM = (250, 5000)   # shared UE-power axis (mW); spans global min (~280) to
                         # sustained max (UL burst ~4500); transient noise spikes clipped


def load_traffic(cap: Path):
    sched = json.loads((cap / "synthetic_traffic" / "schedule.json").read_text())
    segs = sched["segments"]
    t, dl, ul = [], [], []
    for s in segs:
        t.append(s["t0"]); dl.append(s["dl_kbps"] / 1000.0); ul.append(s["ul_kbps"] / 1000.0)
    last = segs[-1]
    t.append(last["t0"] + last["dur"]); dl.append(dl[-1]); ul.append(ul[-1])
    return np.array(t), np.array(dl), np.array(ul), float(sched["duration_s"])


def load_power(cap: Path):
    df = pd.read_csv(cap / "ground_truth_power" / "ground_truth_power_0p5ms.csv")
    t = df["Time_s"].to_numpy()
    p = df["Power (W)"].to_numpy() * 1000.0
    return t[::4], p[::4]   # ~2 ms; keeps spikes, lighter PDF


plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
    "font.family": "serif", "mathtext.fontset": "cm",
    "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "legend.frameon": False, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 200,
})

fig, axes = plt.subplots(2, 5, figsize=(11.0, 3.6), sharey=False)

for col, (title, cap, xlim) in enumerate(PANELS):
    ax_t = axes[0][col]
    ax_p = axes[1][col]

    # --- top: offered IP rate ---
    t, dl, ul, dur = load_traffic(cap)
    ax_t.fill_between(t, 0, dl, step="post", color=DL_C, alpha=0.20, linewidth=0)
    ax_t.fill_between(t, 0, ul, step="post", color=UL_C, alpha=0.18, linewidth=0)
    ax_t.step(t, dl, where="post", color=DL_C, linewidth=1.0, label="DL")
    ax_t.step(t, ul, where="post", color=UL_C, linewidth=1.0, label="UL")
    ax_t.set_title(title)
    ax_t.set_xlim(xlim if xlim else (0, dur))
    ax_t.set_ylim(0, max(dl.max(), ul.max()) * 1.10 + 1e-6)
    ax_t.margins(x=0)
    ax_t.grid(True, alpha=0.25, linewidth=0.4)
    ax_t.tick_params(labelbottom=False)

    # --- bottom: measured UE power ---
    tp, pp = load_power(cap)
    ax_p.plot(tp, pp, color=PWR_C, linewidth=0.5, rasterized=True)
    ax_p.set_xlim(xlim if xlim else (0, dur))
    ax_p.set_ylim(*PWR_YLIM)
    ax_p.grid(True, alpha=0.25, linewidth=0.4)
    ax_p.set_xlabel("Time (s)")

    if col == 0:
        ax_t.set_ylabel("IP rate\n(Mbps)")
        ax_p.set_ylabel("UE power\n(mW)")
    else:
        ax_p.tick_params(labelleft=False)

axes[0][4].legend(loc="upper right", handlelength=1.2, borderaxespad=0.3)

fig.tight_layout(w_pad=0.6, h_pad=0.4)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig4_traffic_probes.pdf")
print("wrote", OUT / "fig4_traffic_probes.pdf")

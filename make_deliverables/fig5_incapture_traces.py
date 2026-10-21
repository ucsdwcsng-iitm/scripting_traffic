"""Paper Fig. 5: in-capture holdout traces, measured UE power vs. XGBoost estimate.

Per-capture protocol: model trained and evaluated on the same capture's
chronological holdout. Reads the holdout-prediction CSVs written by the
experiment runner (the ones behind the in-capture table), so no retraining:
  <results>/in_capture_<app>/default/<capture>/srsran_eval/xgboost/
      eval_epoch_offset_<X>ms/xgboost_holdout_predictions_<N>s.csv

3x2 grid: five app panels + a legend cell. Each panel shows a 3 s window of the
holdout with a 15 ms moving average (display only); the MAE box is computed over
the FULL unsmoothed holdout. Output: make_deliverables/figures/fig5_incapture_traces.pdf
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "ue_power_estimation" / "results"
OUT = REPO / "make_deliverables" / "figures" / "fig5_incapture_traces.pdf"

# (app, panel title, window start in s relative to the holdout start).
# Starts were recovered by matching the vector paths of the published figure
# against every candidate window of the holdout: all five panels are the
# holdout's opening 3 s (start 0.0; next-best candidate is >20 mW off).
PANELS = [
    ("youtube",   "(a) Youtube (720p video streaming)",              0.0),
    ("instagram", "(b) Instagram (shorts)",                          0.0),
    ("slither",   "(c) Slither.io (continuous multi-player gaming)", 0.0),
    ("x",         "(d) X (social posts)",                            0.0),
    ("zoom",      "(e) Zoom (video call)",                           0.0),
]
WINDOW = 3.0             # seconds shown per panel
SMOOTH = 30              # 15 ms moving average at 0.5 ms/slot (display only)
YLIM = (400, 1237)       # shared y range of the published figure

plt.rcParams.update({"xtick.labelsize": 12, "ytick.labelsize": 12,
                     "axes.labelsize": 14, "axes.titlesize": 15,
                     "legend.fontsize": 13, "lines.linewidth": 1.0,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def load(results: Path, app: str) -> pd.DataFrame:
    # fresh training writes eval_epoch_offset_<X>ms/...; a reused (frozen) model
    # writes the CSV directly under xgboost/
    hits = sorted(results.glob(f"in_capture_{app}/default/*/srsran_eval/xgboost/"
                               "**/xgboost_holdout_predictions_*s.csv"))
    if not hits:
        raise SystemExit(f"no holdout predictions for {app} under {results}")
    return pd.read_csv(hits[-1])


def smooth(x):
    # applied to the displayed window only (zero-padded edges, as published)
    return np.convolve(x, np.ones(SMOOTH) / SMOOTH, mode="same")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=RES)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    fig, axes = plt.subplots(3, 2, figsize=(12.1, 6.75), sharey=True)
    axes = axes.ravel()
    for i, (app, title, start) in enumerate(PANELS):
        ax, df = axes[i], load(args.results, app)
        t = df["slot_offset_from_t0_s"].to_numpy() - df["slot_offset_from_t0_s"].min()
        y = df["target_power_mw"].to_numpy(np.float64)
        p = df["predicted_power_mw"].to_numpy(np.float64)
        mae = float(np.mean(np.abs(p - y)))            # full holdout, unsmoothed
        w = (t >= start) & (t <= start + WINDOW)
        ax.plot(t[w] - start, smooth(y[w]), color="C0", label="measured (ground truth)")
        ax.plot(t[w] - start, smooth(p[w]), color="C1", label="gNB-Power estimate (XGBoost)")
        ax.set(title=title, xlim=(0, WINDOW), ylim=YLIM)
        ax.grid(alpha=0.3)
        if i % 2 == 0:
            ax.set_ylabel("power (mW)")
        ax.text(0.03, 0.95, f"MAE {mae:.0f} mW", transform=ax.transAxes, ha="left",
                va="top", fontsize=13,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9))
        print(f"  {app:10s} holdout MAE = {mae:6.1f} mW  (N={len(t)})")
    axes[-1].axis("off")
    axes[-1].legend(*axes[0].get_legend_handles_labels(), loc="center", frameon=False)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

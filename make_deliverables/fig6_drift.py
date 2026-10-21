"""Paper Fig. 6: per-chunk MAE over the 200 s drift holdout.

For each app, an XGBoost model is trained on the first 50 s of a single 250 s
capture and evaluated, frozen, on the remaining 200 s sliced into 20 contiguous
10 s chunks. Reads the per-chunk summaries written by the experiment runner:
  <results>/drift_<app>/default/<capture>/srsran_eval_<a>s_to_<b>s/xgboost/xgboost_summary.json
x = chunk index 1..20 (sorted by <a>), y = XGBoost MAE (mW).
Output: make_deliverables/figures/fig6_drift.pdf
"""
import argparse
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "ue_power_estimation" / "results"
OUT = REPO / "make_deliverables" / "figures" / "fig6_drift.pdf"

APPS = [("youtube", "YouTube"), ("instagram", "Instagram"), ("slither", "slither.io"),
        ("x", "X"), ("zoom", "Zoom")]      # colors C0..C4 (tab10), as published
YLIM = (60, 270)                           # y range of the published figure
_START = re.compile(r"srsran_eval_(\d+(?:p\d+)?)s_to_")

plt.rcParams.update({"xtick.labelsize": 11, "ytick.labelsize": 11,
                     "axes.labelsize": 14, "legend.fontsize": 12,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def load_mae(results: Path, app: str) -> list[float]:
    rows = []
    for f in results.glob(f"drift_{app}/default/*/srsran_eval_*s_to_*s/xgboost/xgboost_summary.json"):
        start = float(_START.search(f.parents[1].name).group(1).replace("p", "."))
        rows.append((start, json.loads(f.read_text())["metrics"]["XGBoost"]["mae_mw"]))
    if not rows:
        raise SystemExit(f"no drift chunk summaries for {app} under {results}")
    return [mae for _, mae in sorted(rows)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=RES)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(6, 2.25))
    for i, (app, label) in enumerate(APPS):
        mae = load_mae(args.results, app)
        ax.plot(range(1, len(mae) + 1), mae, color=f"C{i}", lw=1.25, marker="o",
                ms=2.6, label=label)
        print(f"  {label:10s} chunks={len(mae):2d}  MAE " + " ".join(f"{m:.0f}" for m in mae))
    ax.set(xlabel="test chunk", ylabel="MAE (mW)", xlim=(1, 20), ylim=YLIM,
           xticks=[1, 5, 10, 15, 20])
    ax.grid(alpha=0.3)
    # default subplot params; the one-row legend overhangs the axes on both sides
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=len(APPS), frameon=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

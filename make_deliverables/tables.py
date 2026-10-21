"""Paper Tables 3 and 4 from the experiment runner's results.

Each cell is MAE / MAPE / interval energy error (mW / % / %) over the final 10 s
holdout. MAE and MAPE are per-slot; the energy error integrates the estimate
and the raw INA219 trace over the holdout span. Prints Markdown tables.

  Table 3 (per-capture):  results/in_capture_<app>/...
  Table 4 (synth->real):  results/synthetic_to_<app>/... (+ Table 3 columns)
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
APPS = [("youtube", "YouTube"), ("instagram", "Instagram"), ("slither", "slither.io"),
        ("x", "X"), ("zoom", "Zoom")]
# prediction-CSV column per estimator
COLUMNS = {"xgb": "predicted_power_mw", "raw": "rule_based_power_raw_mw",
           "fit": "rule_based_power_mw"}


def _unwrap_sod(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    day, previous = 0.0, out[0] if len(out) else 0.0
    for i in range(1, len(out)):
        current = out[i]
        if current < previous - 43200.0:
            day += 86400.0
        out[i] = current + day
        previous = current
    return out


def _parse_hms(value: object) -> float:
    h, m, s = str(value).strip().split(":")
    return int(h) * 3600.0 + int(m) * 60.0 + float(s)


def _ground_truth(capture: Path):
    """Raw INA219 trace on local seconds-of-day, plus the epoch->local offset."""
    csv = sorted((capture / "ground_truth_power").glob("ina219_power_log*.csv")) or \
        sorted(capture.glob("ina219_power_log*.csv"))
    frame = pd.read_csv(csv[0])
    epoch_sod = pd.to_numeric(frame["Timestamp_Epoch(s)"], errors="coerce").to_numpy(float) % 86400.0
    times = frame["Timestamp_ISO"].map(_parse_hms).to_numpy(float)
    power = pd.to_numeric(frame["Power (mW)"], errors="coerce").to_numpy(float)
    ok = np.isfinite(times) & np.isfinite(epoch_sod) & np.isfinite(power)
    local = _unwrap_sod(times[ok])
    return local, power[ok], float(local[0] - epoch_sod[ok][0])


def _integrate(times, values_mw, start, end) -> float:
    order = np.argsort(times)
    t, idx = np.unique(np.asarray(times, float)[order], return_index=True)
    v = np.asarray(values_mw, float)[order][idx]
    inner = (t > start) & (t < end)
    grid = np.concatenate(([start], t[inner], [end]))
    return float(np.trapezoid(np.interp(grid, t, v), grid) / 1000.0)


def evaluate(result_dir: Path, data_root: Path) -> dict[str, tuple[float, float, float]]:
    """{estimator: (MAE mW, MAPE %, |energy error| %)} for one result folder."""
    manifest = json.loads((result_dir / "manifest.json").read_text())
    capture = data_root / manifest["captures"]["test"]["capture_id"]
    gt_t, gt_p, offset = _ground_truth(capture)
    pred_csv = glob.glob(str(result_dir / "srsran_eval" / "xgboost" / "**" /
                             "xgboost_holdout_predictions_*.csv"), recursive=True)[0]
    frame = pd.read_csv(pred_csv, low_memory=False)
    frame["t"] = _unwrap_sod(pd.to_numeric(frame["target_epoch_s"], errors="coerce")
                             .to_numpy(float) % 86400.0 + offset)
    frame = frame.dropna(subset=["t", "target_power_mw", *COLUMNS.values()])
    frame = frame.sort_values("t")
    start, end = float(frame["t"].min()), float(frame["t"].max())
    while start - gt_t[0] > 43200.0:
        gt_t = gt_t + 86400.0
    while gt_t[0] - start > 43200.0:
        gt_t = gt_t - 86400.0
    gt_energy = _integrate(gt_t, gt_p, start, end)
    truth = frame["target_power_mw"].to_numpy(float)
    out = {}
    for key, col in COLUMNS.items():
        est = frame[col].to_numpy(float)
        err = np.abs(truth - est)
        energy = _integrate(frame["t"].to_numpy(float), est, start, end)
        out[key] = (float(err.mean()),
                    float(np.mean(err / np.maximum(np.abs(truth), 1e-6)) * 100.0),
                    abs(energy - gt_energy) / gt_energy * 100.0)
    return out


def _result_dir(results: Path, experiment_id: str) -> Path:
    dirs = glob.glob(str(results / experiment_id / "*" / "*" / "manifest.json"))
    if len(dirs) != 1:
        raise SystemExit(f"expected one result under {results / experiment_id}, found {len(dirs)}")
    return Path(dirs[0]).parent


def _cell(m) -> str:
    return f"{m[0]:.0f}/{m[1]:.1f}/{m[2]:.1f}"


def _table(header: list[str], rows: dict[str, list[tuple]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, cells in rows.items():
        lines.append("| " + " | ".join([name, *map(_cell, cells)]) + " |")
    macro = [tuple(np.mean([r[i][k] for r in rows.values()]) for k in range(3))
             for i in range(len(header) - 1)]
    lines.append("| " + " | ".join(["**Macro**", *map(_cell, macro)]) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=REPO / "ue_power_estimation" / "results")
    ap.add_argument("--data", type=Path, default=REPO / "ue_power_estimation" / "data" / "qualcomm")
    args = ap.parse_args()

    incap = {app: evaluate(_result_dir(args.results, f"in_capture_{app}"), args.data)
             for app, _ in APPS}
    synth = {app: evaluate(_result_dir(args.results, f"synthetic_to_{app}"), args.data)
             for app, _ in APPS}

    print("Table 3: per-capture power estimation (MAE / MAPE / energy error, mW / % / %)\n")
    print(_table(["Application", "TR 38.840 raw", "TR 38.840 aff.", "gNB-Power"],
                 {name: [incap[a]["raw"], incap[a]["fit"], incap[a]["xgb"]] for a, name in APPS}))
    print("\nTable 4: synthetic-to-real generalization (MAE / MAPE / energy error, mW / % / %)\n")
    print(_table(["App", "In-cap. (ref.)", "TR 38.840 synth-fit", "TR 38.840 target-raw",
                  "TR 38.840 target-fit", "XGBoost (synth-trained)"],
                 {name: [incap[a]["xgb"], synth[a]["fit"], synth[a]["raw"], incap[a]["fit"],
                         synth[a]["xgb"]] for a, name in APPS}))


if __name__ == "__main__":
    main()

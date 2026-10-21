"""Paper Sec. 5.1 inference cost: per-slot XGBoost latency of the per-capture
(3GPP-free) gNB-Power models, single CPU thread, replaying each application's
held-out slots.

  batched:     one predict() over all held-out slots, divided by the slot count
  single-slot: one predict() call per slot (per-call overhead dominates)

Run from the repo root after the in_capture experiments; absolute numbers depend
on the host CPU.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ue_power_estimation"))
from scripts.power_estimation.srsRAN.run_xgboost import _prepare_frozen  # noqa: E402
from scripts.power_estimation.capture_layout import CaptureLayout  # noqa: E402

APPS = ["youtube", "instagram", "slither", "x", "zoom"]
HOLDOUT_START_S = 50.0


def measure(result_dir: Path, repeats: int, single_slots: int) -> tuple[float, float, int]:
    manifest = json.loads((result_dir / "manifest.json").read_text())
    srsran = manifest["outputs"]["srsran"]
    capture = Path(srsran["eval_capture_dir"])
    features_csv = Path(srsran["features_csv"])
    prep = _prepare_frozen(
        capture_dir=capture,
        features_csv=features_csv,
        ground_truth_csv=CaptureLayout(capture).require_ground_truth_csv(),
        model_bundle=Path(manifest["model_dir"]),
        gpp_output_dir=features_csv.parent / "3gpp",
    )
    matrix, model = prep["matrix"], prep["model"]
    offset = matrix["slot_offset_from_t0_s"] - matrix["slot_offset_from_t0_s"].min()
    X = matrix.loc[offset >= HOLDOUT_START_S, prep["expected_features"]]
    X = X.astype(np.float32).fillna(0.0).to_numpy()
    model.set_params(n_jobs=1)
    model.predict(X[:256])  # warm-up

    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(X)
    batched_us = (time.perf_counter() - start) / (repeats * len(X)) * 1e6

    rows = X[: min(single_slots, len(X))]
    start = time.perf_counter()
    for i in range(len(rows)):
        model.predict(rows[i : i + 1])
    single_us = (time.perf_counter() - start) / len(rows) * 1e6
    return batched_us, single_us, len(X)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=REPO / "ue_power_estimation" / "results")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--single-slots", type=int, default=2000)
    args = ap.parse_args()
    print("| App | held-out slots | batched (us/slot) | single-slot (us) |")
    print("|---|---:|---:|---:|")
    for app in APPS:
        [manifest] = glob.glob(str(args.results / f"in_capture_{app}" / "*" / "*" / "manifest.json"))
        batched, single, n = measure(Path(manifest).parent, args.repeats, args.single_slots)
        print(f"| {app} | {n} | {batched:.1f} | {single:.0f} |")
    print("\nSlot period at 30 kHz SCS: 500 us.")


if __name__ == "__main__":
    main()

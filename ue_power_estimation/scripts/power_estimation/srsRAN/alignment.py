"""Per-capture clock alignment for srsRAN gNB logs vs. INA219 ground truth.

The residual offset between the FAPI slot timestamps and the INA219 power-meter
clock is a *physical property of a capture* (clock skew at recording time), not a
property of any model or train/test split. It is therefore estimated once, as
soon as the capture's slot features and ground truth exist, and cached as capture
metadata (``srsRAN/alignment/alignment.json``). All downstream power
estimation reads and applies this cached offset instead of recomputing it.

Selection criterion: minimum bounded-affine 3GPP calibration MAE over a fixed
early calibration window (decoupled from the experiment's train/validation
windows so the offset is stable across experiments). The correlation curve is
retained for diagnostics only.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.power_estimation.capture_layout import REPO_ROOT, CaptureLayout
from .run_3gpp import run as run_3gpp


ALIGNMENT_SCHEMA_VERSION = 2
DEFAULT_OFFSET_SEARCH_MS = 250.0
DEFAULT_OFFSET_STEP_MS = 1.0
# Fixed calibration window used to select the offset. Decoupled from any
# experiment's train/validation windows so the cached offset is one stable
# per-capture constant. Capped internally to the available overlap.
DEFAULT_CALIBRATION_WINDOW_S = 50.0


def _first_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in frame:
            return name
    raise RuntimeError(f"Missing required column; expected one of {tuple(candidates)}")


def load_ground_truth(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    epoch_col = _first_column(frame, ("Timestamp_Epoch(s)", "timestamp_epoch_s", "epoch_s"))
    power_col = _first_column(frame, ("Power (mW)", "Power_mW", "power_mw"))
    out = pd.DataFrame(
        {
            "epoch_s": pd.to_numeric(frame[epoch_col], errors="coerce"),
            "power_mw": pd.to_numeric(frame[power_col], errors="coerce"),
        }
    ).dropna()
    out = out.sort_values("epoch_s").drop_duplicates("epoch_s")
    if len(out) < 2:
        raise RuntimeError(f"Ground-truth trace has too few valid rows: {path}")
    return out


def load_slot_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False).copy()
    if "slot_time_point_ns" not in frame:
        raise RuntimeError("srsRAN feature CSV is missing slot_time_point_ns.")
    frame["slot_epoch_s"] = pd.to_numeric(frame["slot_time_point_ns"], errors="coerce") / 1e9
    frame = frame.dropna(subset=["slot_epoch_s"]).sort_values("global_slot").reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("srsRAN feature CSV has no timestamped slot rows.")
    return frame


def _activity_proxy(frame: pd.DataFrame) -> np.ndarray:
    proxy = np.zeros(len(frame), dtype=np.float64)
    weights = {
        "pdcch_count": 1.0,
        "pdsch_count": 3.0,
        "pucch_count": 1.5,
        "pusch_count": 4.0,
        "srs_count": 1.5,
        "csi_rs_nzp_count": 0.5,
        "pdcch_monitoring_required": 0.25,
    }
    for name, weight in weights.items():
        if name in frame:
            proxy += weight * pd.to_numeric(frame[name], errors="coerce").fillna(0.0).to_numpy()
    for rb_name, sym_name, weight in (
        ("pdsch_n_prb", "pdsch_n_sym", 0.01),
        ("pusch_n_prb", "pusch_n_sym", 0.015),
    ):
        if rb_name in frame and sym_name in frame:
            proxy += (
                pd.to_numeric(frame[rb_name], errors="coerce").fillna(0.0).to_numpy()
                * pd.to_numeric(frame[sym_name], errors="coerce").fillna(0.0).to_numpy()
                * weight
            )
    return proxy


def estimate_residual_offset(
    slots: pd.DataFrame,
    gt: pd.DataFrame,
    *,
    search_ms: float,
    step_ms: float,
    fit_window_s: float,
) -> Tuple[float, pd.DataFrame]:
    """Search a small residual clock offset using 10 ms activity/power changes."""

    if search_ms <= 0:
        return 0.0, pd.DataFrame([{"offset_s": 0.0, "correlation": np.nan}])
    slot_t = slots["slot_epoch_s"].to_numpy(dtype=np.float64)
    proxy = _activity_proxy(slots)
    gt_t = gt["epoch_s"].to_numpy(dtype=np.float64)
    gt_p = gt["power_mw"].to_numpy(dtype=np.float64)

    overlap_start = max(float(slot_t.min()), float(gt_t.min()) - search_ms / 1000.0)
    overlap_end = min(float(slot_t.max()), float(gt_t.max()) + search_ms / 1000.0)
    overlap_end = min(overlap_end, overlap_start + fit_window_s)
    bin_edges = np.arange(overlap_start, overlap_end + 0.010, 0.010)
    if len(bin_edges) < 20:
        raise RuntimeError("Too little overlap for residual offset search.")
    slot_bins = np.clip(np.digitize(slot_t, bin_edges) - 1, 0, len(bin_edges) - 2)
    proxy_sum = np.bincount(slot_bins, weights=proxy, minlength=len(bin_edges) - 1)
    proxy_diff = np.diff(proxy_sum)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    rows = []
    for offset_ms in np.arange(-search_ms, search_ms + step_ms / 2.0, step_ms):
        offset_s = float(offset_ms / 1000.0)
        power = np.interp(centers + offset_s, gt_t, gt_p)
        power_diff = np.diff(power)
        if np.std(proxy_diff) == 0 or np.std(power_diff) == 0:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(proxy_diff, power_diff)[0, 1])
        rows.append({"offset_s": offset_s, "correlation": corr})
    scores = pd.DataFrame(rows)
    finite = scores["correlation"].replace([np.inf, -np.inf], np.nan).dropna()
    best = 0.0 if finite.empty else float(scores.loc[finite.idxmax(), "offset_s"])
    return best, scores


def score_rule_baseline_offsets(
    slots: pd.DataFrame,
    gt: pd.DataFrame,
    rule_based: pd.DataFrame,
    offsets: Sequence[float],
    *,
    fit_window_s: float,
) -> pd.DataFrame:
    merged = slots[["global_slot", "slot_epoch_s"]].merge(
        rule_based[["global_slot", "power_assumed_mw"]],
        on="global_slot",
        how="inner",
    )
    slot_t = merged["slot_epoch_s"].to_numpy(dtype=np.float64)
    raw = merged["power_assumed_mw"].to_numpy(dtype=np.float64)
    gt_t = gt["epoch_s"].to_numpy(dtype=np.float64)
    gt_p = gt["power_mw"].to_numpy(dtype=np.float64)
    rows = []
    for offset in offsets:
        target_t = slot_t + float(offset)
        valid = (target_t >= gt_t.min()) & (target_t <= gt_t.max())
        if not valid.any():
            rows.append({"offset_s": float(offset), "cal_mae_mw": np.nan, "a": np.nan, "b_mw": np.nan})
            continue
        first_t = float(slot_t[valid][0])
        fit = valid & (slot_t < first_t + float(fit_window_s))
        x = raw[fit]
        y = np.interp(target_t[fit], gt_t, gt_p)
        if len(x) < 100:
            rows.append({"offset_s": float(offset), "cal_mae_mw": np.nan, "a": np.nan, "b_mw": np.nan})
            continue
        variance = float(np.var(x))
        a = 1.0 if variance <= 1e-12 else float(np.cov(x, y, ddof=0)[0, 1] / variance)
        a = float(np.clip(a, 0.15, 2.0))
        b = float(np.mean(y - a * x))
        # Keep the bounded-affine (a, b) fit at this offset: at the *selected*
        # offset these are the rule-baseline calibration, so downstream code can
        # reuse them instead of refitting the same OLS (see select_offset).
        rows.append(
            {
                "offset_s": float(offset),
                "cal_mae_mw": float(np.mean(np.abs(y - (a * x + b)))),
                "a": a,
                "b_mw": b,
            }
        )
    return pd.DataFrame(rows)


def select_offset(
    slots: pd.DataFrame,
    gt: pd.DataFrame,
    rule_based: pd.DataFrame,
    *,
    calibration_window_s: float,
    search_ms: float,
    step_ms: float,
) -> Dict[str, object]:
    """Select the residual offset by minimum 3GPP calibration MAE.

    Returns a dict with the scalar selection result plus the full diagnostic
    score curves (``offset_scores`` correlation, ``offset_mae`` calibration MAE).
    """

    corr_offset_s, offset_scores = estimate_residual_offset(
        slots,
        gt,
        search_ms=search_ms,
        step_ms=step_ms,
        fit_window_s=calibration_window_s,
    )
    offset_mae = score_rule_baseline_offsets(
        slots,
        gt,
        rule_based,
        offset_scores["offset_s"].to_numpy(dtype=float),
        fit_window_s=calibration_window_s,
    )
    finite_mae = offset_mae.dropna(subset=["cal_mae_mw"])
    if finite_mae.empty:
        offset_s = float(corr_offset_s)
        selected_by = "correlation"
        cal_mae_mw: Optional[float] = None
        # No usable affine fit (degenerate capture); identity calibration.
        rule_affine_a, rule_affine_b = 1.0, 0.0
    else:
        best_row = finite_mae.loc[finite_mae["cal_mae_mw"].idxmin()]
        offset_s = float(best_row["offset_s"])
        selected_by = "gpp_calibration_mae"
        cal_mae_mw = float(best_row["cal_mae_mw"])
        # The bounded-affine (a, b) at the selected offset *is* the rule-baseline
        # calibration. Carry it out so the training path and the diagnostic plot
        # reuse it rather than refitting the identical OLS.
        rule_affine_a = float(best_row["a"])
        rule_affine_b = float(best_row["b_mw"])
    return {
        "residual_offset_s": offset_s,
        "selected_by": selected_by,
        "correlation_offset_s": float(corr_offset_s),
        "cal_mae_mw": cal_mae_mw,
        "rule_affine_a": rule_affine_a,
        "rule_affine_b": rule_affine_b,
        "offset_scores": offset_scores,
        "offset_mae": offset_mae,
    }


def _git_rev() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _ensure_rule_based(
    *,
    features_csv: Path,
    gpp_csv: Optional[Path],
    alignment_dir: Path,
) -> pd.DataFrame:
    if gpp_csv is not None and Path(gpp_csv).exists():
        return pd.read_csv(gpp_csv, low_memory=False)
    gpp_dir = alignment_dir / "3gpp"
    gpp_out = gpp_dir / "power_per_slot.csv"
    if not gpp_out.exists():
        run_3gpp(features_csv=features_csv, output_dir=gpp_dir)
    return pd.read_csv(gpp_out, low_memory=False)


def _plot_gt_vs_estimate(
    slots: pd.DataFrame,
    gt: pd.DataFrame,
    rule_based: pd.DataFrame,
    *,
    residual_offset_s: float,
    a: float,
    b: float,
    output_path: Path,
) -> None:
    """Plot INA219 ground truth against the srsRAN 3GPP estimate (raw and
    bounded-affine calibrated), time-aligned by the residual offset, over the
    full overlap. The affine (a, b) is the one selected with the offset
    (score_rule_baseline_offsets) and is passed in rather than refit here."""

    merged = slots[["global_slot", "slot_epoch_s"]].merge(
        rule_based[["global_slot", "power_assumed_mw"]],
        on="global_slot",
        how="inner",
    )
    slot_t = merged["slot_epoch_s"].to_numpy(dtype=np.float64)
    raw = merged["power_assumed_mw"].to_numpy(dtype=np.float64)
    gt_t = gt["epoch_s"].to_numpy(dtype=np.float64)
    gt_p = gt["power_mw"].to_numpy(dtype=np.float64)

    target_t = slot_t + float(residual_offset_s)
    valid = (target_t >= gt_t.min()) & (target_t <= gt_t.max())
    if not valid.any():
        return

    target_valid = target_t[valid]
    raw_valid = raw[valid]
    truth = np.interp(target_valid, gt_t, gt_p)
    calibrated = a * raw_valid + b
    x = target_valid - float(target_valid[0])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, truth, label="INA219 ground truth", linewidth=1.0)
    ax.plot(x, raw_valid, label="3GPP raw", linewidth=1.0, alpha=0.7)
    ax.plot(
        x,
        calibrated,
        label=f"3GPP affine-calibrated (a={a:.3f}, b={b:.1f} mW)",
        linewidth=1.0,
        alpha=0.9,
    )
    ax.set_xlabel("Seconds from aligned overlap start")
    ax.set_ylabel("Power (mW)")
    ax.set_title(f"srsRAN 3GPP vs ground truth (residual offset {residual_offset_s:+.4f}s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def compute_alignment(
    capture_dir: Path,
    *,
    features_csv: Path,
    ground_truth_csv: Path,
    gpp_csv: Optional[Path] = None,
    slots: Optional[pd.DataFrame] = None,
    gt: Optional[pd.DataFrame] = None,
    rule_based: Optional[pd.DataFrame] = None,
    calibration_window_s: float = DEFAULT_CALIBRATION_WINDOW_S,
    search_ms: float = DEFAULT_OFFSET_SEARCH_MS,
    step_ms: float = DEFAULT_OFFSET_STEP_MS,
    write: bool = True,
) -> Dict[str, object]:
    """Estimate and (by default) persist the per-capture residual offset."""

    layout = CaptureLayout(capture_dir)
    alignment_dir = layout.srsran_alignment_dir
    if slots is None:
        slots = load_slot_features(Path(features_csv))
    if gt is None:
        gt = load_ground_truth(Path(ground_truth_csv))
    if rule_based is None:
        rule_based = _ensure_rule_based(
            features_csv=Path(features_csv),
            gpp_csv=gpp_csv,
            alignment_dir=alignment_dir,
        )
    # The 3GPP gt-vs-estimate figure goes next to the 3GPP CSV.
    gpp_dir = Path(gpp_csv).parent if gpp_csv is not None else (alignment_dir / "3gpp")

    selection = select_offset(
        slots,
        gt,
        rule_based,
        calibration_window_s=calibration_window_s,
        search_ms=search_ms,
        step_ms=step_ms,
    )
    offset_scores = selection.pop("offset_scores")
    offset_mae = selection.pop("offset_mae")

    corr_csv = alignment_dir / "correlation_offset_scores.csv"
    mae_csv = alignment_dir / "gpp_calibration_mae_offset_scores.csv"
    gt_vs_estimate_pdf = gpp_dir / "gt_vs_estimate.pdf"
    n_cal_slots = int(
        (slots["slot_epoch_s"] < float(slots["slot_epoch_s"].iloc[0]) + calibration_window_s).sum()
    )
    metadata: Dict[str, object] = {
        "schema_version": ALIGNMENT_SCHEMA_VERSION,
        "pipeline": "srsran",
        "residual_offset_s": float(selection["residual_offset_s"]),
        "selected_by": selection["selected_by"],
        "correlation_offset_s": float(selection["correlation_offset_s"]),
        "cal_mae_mw": selection["cal_mae_mw"],
        # Bounded-affine rule-baseline calibration at the selected offset, fit on
        # the same calibration window. Persisted so power estimation reuses it as
        # the rule_based_power_mw calibration without refitting.
        "rule_affine_a": float(selection["rule_affine_a"]),
        "rule_affine_b": float(selection["rule_affine_b"]),
        "calibration_window_s": float(calibration_window_s),
        "offset_search_ms": float(search_ms),
        "offset_step_ms": float(step_ms),
        "n_calibration_slots": n_cal_slots,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_rev": _git_rev(),
        "selection_metric": (
            "Minimum bounded-affine 3GPP calibration MAE over a fixed early "
            "calibration window. This is a 3GPP-baseline calibration MAE, not the "
            "XGBoost model MAE. Correlation offset is diagnostic only."
        ),
        "sources": {
            "capture_dir": str(capture_dir),
            "features_csv": str(features_csv),
            "ground_truth_csv": str(ground_truth_csv),
        },
        "diagnostics": {
            "correlation_offset_scores_csv": str(corr_csv),
            "gpp_calibration_mae_offset_scores_csv": str(mae_csv),
            "gt_vs_estimate_pdf": str(gt_vs_estimate_pdf),
        },
    }

    if write:
        alignment_dir.mkdir(parents=True, exist_ok=True)
        offset_scores.to_csv(corr_csv, index=False)
        offset_mae.to_csv(mae_csv, index=False)
        layout.srsran_alignment_json.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _plot_gt_vs_estimate(
            slots,
            gt,
            rule_based,
            residual_offset_s=float(selection["residual_offset_s"]),
            a=float(selection["rule_affine_a"]),
            b=float(selection["rule_affine_b"]),
            output_path=gt_vs_estimate_pdf,
        )
    return metadata


def load_alignment(capture_dir: Path) -> Optional[Dict[str, object]]:
    """Read cached alignment metadata, or None if absent/incompatible."""

    path = CaptureLayout(capture_dir).srsran_alignment_json
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if meta.get("schema_version") != ALIGNMENT_SCHEMA_VERSION or "residual_offset_s" not in meta:
        return None
    return meta


def load_or_compute_alignment(
    capture_dir: Path,
    **kwargs: object,
) -> Dict[str, object]:
    """Return cached alignment metadata, computing and caching it if absent."""

    existing = load_alignment(capture_dir)
    if existing is not None:
        return existing
    return compute_alignment(capture_dir, **kwargs)  # type: ignore[arg-type]


# Alias used by the experiment runner at data-prep time.
ensure_alignment = load_or_compute_alignment


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Estimate and cache the per-capture srsRAN residual clock offset "
            "(capture metadata). Run once when a capture's features and ground "
            "truth exist; power estimation reads the cached offset."
        )
    )
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, default=None)
    parser.add_argument("--ground-truth-csv", type=Path, default=None)
    parser.add_argument("--gpp-csv", type=Path, default=None)
    parser.add_argument(
        "--calibration-window-s", type=float, default=DEFAULT_CALIBRATION_WINDOW_S
    )
    parser.add_argument("--offset-search-ms", type=float, default=DEFAULT_OFFSET_SEARCH_MS)
    parser.add_argument("--offset-step-ms", type=float, default=DEFAULT_OFFSET_STEP_MS)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if cached alignment metadata already exists.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    capture_dir = args.capture_dir.resolve()
    layout = CaptureLayout(capture_dir)
    features_csv = args.features_csv or layout.srsran_parsed_dir / "slot_features.csv"
    ground_truth_csv = args.ground_truth_csv or layout.require_ground_truth_csv()
    if not args.force:
        existing = load_alignment(capture_dir)
        if existing is not None:
            print(
                f"Alignment already cached: residual_offset_s="
                f"{existing['residual_offset_s']:.6f}s "
                f"(selected_by={existing['selected_by']}). Use --force to recompute."
            )
            return
    metadata = compute_alignment(
        capture_dir,
        features_csv=features_csv,
        ground_truth_csv=ground_truth_csv,
        gpp_csv=args.gpp_csv,
        calibration_window_s=args.calibration_window_s,
        search_ms=args.offset_search_ms,
        step_ms=args.offset_step_ms,
    )
    print(
        f"Wrote {layout.srsran_alignment_json}: "
        f"residual_offset_s={metadata['residual_offset_s']:.6f}s "
        f"(selected_by={metadata['selected_by']}, "
        f"cal_mae_mw={metadata['cal_mae_mw']})."
    )


if __name__ == "__main__":
    main()

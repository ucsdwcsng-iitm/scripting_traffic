"""Pooled XGBoost training over many synthetic captures (paper Table 4).

Reuses the canonical full-feature builder (``build_labeled_feature_matrix``) and
the shared bundle writer (``_write_bundle``) so the resulting model bundle is
evaluated by the existing ``run_frozen`` with no changes. The only differences
from ``train_only`` are that (1) many training captures are pooled at full
resolution and (2) a single global affine TR 38.840 calibration is fit on the
pooled data. Building the pool needs a high-memory host (tens of GB).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.power_estimation.capture_layout import CaptureLayout
from scripts.power_estimation.srsRAN.alignment import load_slot_features, load_ground_truth
from scripts.power_estimation.srsRAN.run_xgboost import (
    build_labeled_feature_matrix, apply_rule_calibration, _write_bundle,
)

_KEEP_META = ("target_power_mw", "rule_based_power_raw_mw", "slot_offset_from_t0_s")


def _capture_paths(capture: Path):
    layout = CaptureLayout(Path(capture))
    return (
        layout.srsran_parsed_dir / "slot_features.csv",
        layout.srsran_3gpp_dir / "power_per_slot.csv",
        layout.srsran_alignment_json,
        layout.ground_truth_csv(),
    )


def _build_reduced(capture: Path):
    """Build one capture's full-feature matrix (canonical builder) and reduce it
    to float32 + the kept columns. Returns (DataFrame, feature_columns,
    residual_offset_s)."""
    feat_csv, gpp_csv, align_json, gt_csv = _capture_paths(capture)
    slots = load_slot_features(feat_csv)
    gt = load_ground_truth(gt_csv)
    rule_based = pd.read_csv(gpp_csv, low_memory=False)
    offset = float(json.loads(Path(align_json).read_text())["residual_offset_s"])
    matrix, feature_cols = build_labeled_feature_matrix(
        slots, gt, residual_offset_s=offset, rule_based=rule_based)
    keep = [c for c in feature_cols if c in matrix.columns] + [c for c in _KEEP_META if c in matrix.columns]
    reduced = matrix[keep].copy()
    for col in reduced.columns:
        reduced[col] = pd.to_numeric(reduced[col], errors="coerce").astype(np.float32)
    return reduced, feature_cols, offset


def _fit_global_affine(raw: np.ndarray, target: np.ndarray):
    ok = np.isfinite(raw) & np.isfinite(target)
    raw, target = raw[ok], target[ok]
    a, b = np.linalg.lstsq(np.vstack([raw, np.ones_like(raw)]).T, target, rcond=None)[0]
    if not (0.05 <= a <= 5.0):
        a = float(np.clip(a, 0.05, 5.0))
        b = float(np.mean(target - a * raw))
    return float(a), float(b)


def _cached_reduced(capture: Path, cache_dir: Path | None):
    """Build (or load from cache) one capture's reduced feature matrix. The cache
    is keyed by capture name so the per-capture build is shared across the
    per-app configs, which all train the same pooled model."""
    if cache_dir is not None:
        cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
        pkl, meta = cache_dir / f"{Path(capture).name}.pkl", cache_dir / f"{Path(capture).name}.json"
        if pkl.exists() and meta.exists():
            info = json.loads(meta.read_text())
            return pd.read_pickle(pkl), info["feature_cols"], float(info["offset_s"])
    reduced, cols, offset = _build_reduced(Path(capture))
    if cache_dir is not None:
        reduced.to_pickle(pkl)
        meta.write_text(json.dumps({"feature_cols": cols, "offset_s": offset}))
    return reduced, cols, offset


def train_pooled(pool: list[Path], model_dir: Path, *, train_window_s: float,
                 validation_window_s: float, n_estimators: int,
                 xgb_params: dict | None = None, cache_dir: Path | None = None,
                 residual_learning: bool = True) -> dict:
    """Train XGBoost on the pooled synthetic captures and write a
    ``run_frozen``-compatible bundle to ``model_dir``. When
    ``residual_learning=True`` (default) the model regresses the residual over the
    calibrated 3GPP baseline (``target − rule_based_power_mw``) and the
    ``rule_based_*`` columns are included as input features. When
    ``residual_learning=False`` it regresses absolute ``target_power_mw`` directly
    and the ``rule_based_*`` columns are excluded from model inputs."""
    parts, feature_union, offsets = [], set(), []
    for capture in pool:
        reduced, cols, offset = _cached_reduced(Path(capture), cache_dir)
        parts.append(reduced)
        feature_union.update(cols)
        offsets.append(offset)

    feature_cols = sorted(feature_union)
    pooled = pd.concat(parts, ignore_index=True)
    del parts
    for col in feature_cols:
        if col not in pooled.columns:
            pooled[col] = np.float32(0.0)

    # One global affine TR 38.840 calibration over the pooled raw vs target.
    a, b = _fit_global_affine(pooled["rule_based_power_raw_mw"].to_numpy(np.float64),
                              pooled["target_power_mw"].to_numpy(np.float64))
    pooled, rule_fit = apply_rule_calibration(pooled, a=a, b=b)

    # When residual_learning is False the pooled model is fully 3GPP-free: it
    # drops the rule_based_* columns from the inputs AND regresses absolute
    # target_power_mw directly (no residual over the calibrated 3GPP baseline).
    if residual_learning:
        model_features = feature_cols + ["rule_based_power_raw_mw", "rule_based_power_mw"]
    else:
        model_features = list(feature_cols)
    X = pooled[model_features].astype(np.float32).fillna(0.0)
    train_target = pooled["target_power_mw"].to_numpy(np.float64)
    if residual_learning:
        label = train_target - pooled["rule_based_power_mw"].to_numpy(np.float64)
    else:
        label = train_target
    ok = np.isfinite(label)
    rng = np.random.RandomState(42)
    val = rng.rand(len(X)) < 0.12
    params = {
        "objective": "reg:squarederror", "n_estimators": int(n_estimators),
        "learning_rate": 0.03, "max_depth": 7, "min_child_weight": 2.0,
        "subsample": 0.9, "colsample_bytree": 0.8, "reg_lambda": 2.0,
        "tree_method": "hist", "n_jobs": 6, "random_state": 42,
        "early_stopping_rounds": 100,
    }
    params.update(xgb_params or {})
    params["n_estimators"] = int(params.get("n_estimators", n_estimators))
    model = xgb.XGBRegressor(**params)
    tr, va = ok & ~val, ok & val
    model.fit(X[tr], label[tr], eval_set=[(X[va], label[va])], verbose=False)

    feat_csv0, _, _, gt0 = _capture_paths(Path(pool[0]))
    model_dir = Path(model_dir)
    _write_bundle(
        bundle_dir=model_dir, model=model, features=model_features,
        offset_s=float(np.median(offsets)), rule_based_fit=rule_fit,
        train_window_s=train_window_s, validation_window_s=validation_window_s,
        capture_dir=Path(pool[0]), features_csv=feat_csv0, ground_truth_csv=gt0,
        residual_learning=residual_learning)
    summary = {
        "mode": "train_pooled", "n_pool": len(pool), "n_rows": int(len(X)),
        "rule_based_fit": rule_fit, "model_input_feature_count": len(model_features),
        "best_iteration": int(getattr(model, "best_iteration", n_estimators - 1)),
    }
    (model_dir / "pooled_summary.json").write_text(json.dumps(summary, indent=2))
    return summary

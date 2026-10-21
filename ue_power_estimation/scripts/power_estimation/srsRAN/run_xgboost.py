"""Train and evaluate XGBoost power estimation from srsRAN slot features."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scripts.power_estimation.capture_layout import CaptureLayout
from .feature_ablation import is_full, normalize_scheme
from .run_3gpp import run as run_3gpp
from .alignment import (
    DEFAULT_CALIBRATION_WINDOW_S,
    DEFAULT_OFFSET_SEARCH_MS,
    DEFAULT_OFFSET_STEP_MS,
    _activity_proxy,
    _first_column,
    estimate_residual_offset,
    load_alignment,
    load_ground_truth,
    load_or_compute_alignment,
    load_slot_features,
    score_rule_baseline_offsets,
)


# Shared curve colors across the srsRAN figures so the same series reads the same
# everywhere. Ground truth stays matplotlib blue; XGBoost and 3GPP use the srsRAN
# colors, distinct from each other and from ground truth.
GROUND_TRUTH_COLOR = "tab:blue"
XGBOOST_COLOR = "#2f8f6b"  # teal-green (srsRAN XGBoost)
GPP_COLOR = "#9a4f86"  # purple (srsRAN 3GPP)

DEFAULT_LAGS = (1, 2, 4, 8, 16, 40, 80, 160, 320)
DEFAULT_ROLLING_WINDOWS = (4, 10, 20, 40, 80, 160)
# Baseline micro-sleep power floor (mW) for the srsRAN 3GPP rule-based estimate.
# Matches the run_3gpp.py default; callers (experiment runner) override it from
# the preprocessing config.
DEFAULT_MICROSLEEP_POWER_MW = 250.0
TEMPORAL_BASES = (
    "pdcch_present",
    "pdcch_count",
    "pdsch_present",
    "pdsch_count",
    "pdsch_n_prb",
    "pdsch_n_sym",
    "pdsch_tbs_bytes",
    "pucch_present",
    "pucch_count",
    "pusch_present",
    "pusch_count",
    "pusch_n_prb",
    "pusch_n_sym",
    "pusch_tbs_bytes",
    "srs_present",
    "srs_count",
    "csi_rs_present",
    "csi_rs_nzp_count",
    "pdcch_monitoring_required",
    "pdcch_candidate_count_total",
    "drx_awake",
    "drx_sleep_opportunity",
    "has_rx_activity",
    "has_tx_activity",
    "has_data_rx",
    "has_data_tx",
)
CATEGORICAL_FEATURES = (
    "activity_class",
    "sleep_depth_candidate",
    "pdcch_direction",
    "pdcch_format",
    "pdsch_mod_order",
    "pusch_mod_order",
    "drx_timer_semantics",
)
EXCLUDED_FEATURES = {
    "capture_id",
    "rnti",
    "rnti_int",
    "ue_id",
    "frame",
    "slot",
    "global_slot",
    "slot_wall_time_iso",
    "slot_time_point_ns",
    "slot_start_time_abs_sod",
    "slot_end_time_abs_sod",
    "target_ue_selected",
    "target_epoch_s",
    "target_power_mw",
    "slot_offset_from_t0_s",
    "rule_based_power_raw_mw",
    "rule_based_power_shifted_mw",
    "rule_based_power_mw",
}


# Ground-truth/slot loaders and the residual-offset search live in alignment.py
# (imported above and re-exported here for callers and tests) so the offset can
# be estimated once as capture metadata, independent of any model or split.


def build_labeled_feature_matrix(
    slots: pd.DataFrame,
    gt: pd.DataFrame,
    *,
    residual_offset_s: float,
    rule_based: pd.DataFrame | None = None,
    lag_slots: Sequence[int] = DEFAULT_LAGS,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    feature_ablation: Mapping[str, object] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    # The optional ablation scheme switches whole feature families off; the
    # default (no block) keeps every family, so the build is byte-for-byte the
    # legacy one. Gating generation (not just dropping columns afterwards) keeps
    # the dropped lag/roll/dummy columns from ever being materialized, which also
    # matters for the memory-bounded pooled build.
    scheme = normalize_scheme(feature_ablation)
    frame = slots.copy()
    frame["target_epoch_s"] = frame["slot_epoch_s"] + float(residual_offset_s)
    gt_t = gt["epoch_s"].to_numpy(dtype=np.float64)
    valid = frame["target_epoch_s"].between(float(gt_t.min()), float(gt_t.max()))
    frame = frame.loc[valid].reset_index(drop=True)
    frame["target_power_mw"] = np.interp(
        frame["target_epoch_s"].to_numpy(dtype=np.float64),
        gt_t,
        gt["power_mw"].to_numpy(dtype=np.float64),
    )
    frame["slot_offset_from_t0_s"] = frame["slot_epoch_s"] - float(frame["slot_epoch_s"].iloc[0])
    if rule_based is not None:
        rule = rule_based[["global_slot", "power_assumed_mw", "dominant_state"]].copy()
        rule = rule.rename(
            columns={
                "power_assumed_mw": "rule_based_power_raw_mw",
                "dominant_state": "rule_based_state",
            }
        )
        frame = frame.merge(rule, on="global_slot", how="left")
        frame["rule_based_power_raw_mw"] = (
            pd.to_numeric(frame["rule_based_power_raw_mw"], errors="coerce")
            .ffill()
            .bfill()
        )

    temporal_bases = [name for name in TEMPORAL_BASES if name in frame]
    temporal_columns: Dict[str, pd.Series] = {}
    if scheme["include_temporal_aggregates"]:
        for name in temporal_bases:
            values = pd.to_numeric(frame[name], errors="coerce").fillna(0.0)
            for lag in lag_slots:
                temporal_columns[f"lag__{name}__s{lag}"] = values.shift(int(lag)).fillna(0.0)
            prior = values.shift(1)
            for window in rolling_windows:
                rolling = prior.rolling(int(window), min_periods=1)
                temporal_columns[f"roll__{name}__w{window}__mean"] = rolling.mean().fillna(0.0)
                temporal_columns[f"roll__{name}__w{window}__max"] = rolling.max().fillna(0.0)
    if temporal_columns:
        frame = pd.concat([frame, pd.DataFrame(temporal_columns, index=frame.index)], axis=1)

    # Categorical one-hots are only generated when the family is kept. When
    # dropped, the source string columns stay in `frame` (used for export) but are
    # skipped by the numeric feature filter below, so they never become features.
    if scheme["include_categorical"]:
        categorical = [name for name in CATEGORICAL_FEATURES if name in frame]
        frame = pd.get_dummies(frame, columns=categorical, prefix=categorical, dummy_na=True, dtype=np.uint8)

    base_set = set(TEMPORAL_BASES)
    feature_columns: List[str] = []
    for name in frame.columns:
        if name in EXCLUDED_FEATURES or name == "slot_epoch_s":
            continue
        # Drop the current-slot raw base values when that family is ablated; their
        # lag/roll derivatives (lag__*/roll__*, distinct names) are governed by the
        # temporal-aggregate flag's generation gate above.
        if not scheme["include_temporal_bases"] and name in base_set:
            continue
        if pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(frame[name]):
            values = pd.to_numeric(frame[name], errors="coerce")
            if values.notna().any() and values.nunique(dropna=True) > 1:
                frame[name] = values.fillna(values.median()).fillna(0.0)
                feature_columns.append(name)
    if not feature_columns and not is_full(scheme):
        raise ValueError(
            f"feature_ablation scheme {scheme.get('scheme_id')!r} removed every "
            "feature column; keep at least one feature family"
        )
    return frame, feature_columns


def _fit_residual_model(
    matrix: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    train_indices: pd.Index,
    validation_indices: pd.Index,
    n_estimators: int,
    xgb_params: Dict[str, object] | None = None,
    residual_learning: bool = True,
) -> Tuple[xgb.XGBRegressor, pd.DataFrame, pd.Series]:
    """Fit the XGBoost model on the train split, using the validation split for
    early stopping. When ``residual_learning=True`` (default), the model is
    trained on the residual ``target_power_mw - rule_based_power_mw``; when
    ``False``, it is trained on the absolute ``target_power_mw``. Returns the
    model plus the feature matrix X and label series y for reuse by the
    caller's evaluation. Shared by train_and_evaluate (in-capture eval) and
    train_only (cross-capture).
    """
    X = matrix[list(feature_columns)].astype(np.float32)
    y = matrix["target_power_mw"].astype(np.float32)
    effective_params: Dict[str, object] = {
        "objective": "reg:squarederror",
        "n_estimators": int(n_estimators),
        "learning_rate": 0.03,
        "max_depth": 7,
        "min_child_weight": 2.0,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
        "early_stopping_rounds": 100,
    }
    effective_params.update(xgb_params or {})
    effective_params["n_estimators"] = int(
        effective_params.get("n_estimators", n_estimators)
    )
    model = xgb.XGBRegressor(**effective_params)
    train_target = y.loc[train_indices].astype(np.float32)
    validation_target = y.loc[validation_indices].astype(np.float32)
    if residual_learning:
        train_target = train_target - matrix.loc[train_indices, "rule_based_power_mw"].astype(np.float32)
        validation_target = validation_target - matrix.loc[validation_indices, "rule_based_power_mw"].astype(np.float32)
    # Evaluate on the train split first and the validation split second so
    # model.evals_result() carries BOTH loss curves (validation_0=train,
    # validation_1=validation) for the train_info diagnostics. Early stopping
    # still keys on the last eval_set entry (validation), so behaviour is
    # unchanged.
    model.fit(
        X.loc[train_indices],
        train_target,
        eval_set=[
            (X.loc[train_indices], train_target),
            (X.loc[validation_indices], validation_target),
        ],
        verbose=False,
    )
    return model, X, y


def train_and_evaluate(
    matrix: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    train_window_s: float,
    validation_window_s: float,
    n_estimators: int,
    xgb_params: Dict[str, object] | None = None,
    residual_learning: bool = True,
) -> Tuple[xgb.XGBRegressor, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    train_indices, validation_indices, test_indices = _time_split_indices(
        matrix,
        train_window_s=train_window_s,
        validation_window_s=validation_window_s,
    )
    test_mask = matrix.index.isin(test_indices)
    if min(len(train_indices), len(validation_indices), int(test_mask.sum())) < 1000:
        raise RuntimeError(
            "Insufficient split rows: "
            f"train={len(train_indices)}, validation={len(validation_indices)}, "
            f"test={int(test_mask.sum())}."
        )

    model, X, y = _fit_residual_model(
        matrix,
        feature_columns,
        train_indices=train_indices,
        validation_indices=validation_indices,
        n_estimators=n_estimators,
        xgb_params=xgb_params,
        residual_learning=residual_learning,
    )
    model_output = model.predict(X.loc[test_indices])
    fitted_rule = matrix.loc[test_indices, "rule_based_power_mw"].to_numpy(dtype=np.float64)
    if residual_learning:
        prediction = model_output + fitted_rule
    else:
        prediction = model_output.astype(np.float64)
    truth = y.loc[test_indices].to_numpy(dtype=np.float64)
    predictions = matrix.loc[
        test_indices,
        [
            "global_slot",
            "frame",
            "slot",
            "slot_epoch_s",
            "target_epoch_s",
            "slot_offset_from_t0_s",
            "target_power_mw",
            "rule_based_state",
            "rule_based_power_raw_mw",
            "rule_based_power_shifted_mw",
            "rule_based_power_mw",
        ],
    ].copy()
    predictions["predicted_power_mw_raw"] = prediction
    predictions["predicted_power_mw"] = prediction
    predictions["predicted_power_mw_base_model"] = prediction
    # The model emits a residual only in residual-learning mode; otherwise its
    # output IS the absolute prediction, so the residual column is masked off.
    if residual_learning:
        predictions["residual_predicted_by_model_mw"] = model_output
    predictions["residual_mw"] = predictions["target_power_mw"] - predictions["predicted_power_mw"]
    predictions["residual_pred_mw"] = predictions["residual_mw"]
    predictions["residual_rule_based_mw"] = (
        predictions["target_power_mw"] - predictions["rule_based_power_mw"]
    )

    importance = pd.DataFrame(
        {
            "feature": list(feature_columns),
            "importance": model.feature_importances_.astype(float),
        }
    ).sort_values("importance", ascending=False)
    metrics = {
        "XGBoost": _metric_block(truth, prediction),
        "3GPP (shift+affine)": _metric_block(truth, fitted_rule),
        "3GPP raw": _metric_block(
            truth,
            predictions["rule_based_power_raw_mw"].to_numpy(dtype=np.float64),
        ),
        "r2": float(r2_score(truth, prediction)),
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "test_rows": int(len(test_indices)),
        "feature_count": int(len(feature_columns)),
        "best_iteration": int(getattr(model, "best_iteration", n_estimators - 1)),
    }
    return model, predictions, importance, metrics


def _time_split_indices(
    matrix: pd.DataFrame,
    *,
    train_window_s: float,
    validation_window_s: float,
) -> Tuple[pd.Index, pd.Index, pd.Index]:
    offsets = matrix["slot_offset_from_t0_s"].astype(float)
    validation_end_s = float(train_window_s + validation_window_s)
    train_indices = matrix.index[offsets < float(train_window_s)]
    validation_indices = matrix.index[
        (offsets >= float(train_window_s)) & (offsets < validation_end_s)
    ]
    test_indices = matrix.index[offsets >= validation_end_s]
    return train_indices, validation_indices, test_indices


def apply_rule_calibration(
    matrix: pd.DataFrame,
    *,
    a: float,
    b: float,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Build the ``rule_based_power_mw`` residual-learning feature by applying the
    per-capture bounded-affine calibration ``(a, b)``.

    ``(a, b)`` is the OLS fit that ``score_rule_baseline_offsets`` already
    computed at the selected clock offset (cached in the capture's alignment
    metadata) -- it is *not* refit here. No additional time shift is applied: the
    single residual offset, applied to the labels in
    ``build_labeled_feature_matrix``, already aligns the 3GPP column to ground
    truth, so ``shift_slots`` is fixed at 0. Shifting the feature too would
    double-apply that one alignment constant. For cross-capture inference the
    training capture's ``(a, b)`` is frozen into the model bundle and reused on
    the test capture (see ``_apply_frozen_rule_calibration``).
    """
    raw = matrix["rule_based_power_raw_mw"].to_numpy(dtype=np.float64)
    out = matrix.copy()
    # No shift: the "shifted" column equals raw so downstream column references
    # (predictions/plots) keep working with shift_slots == 0.
    out["rule_based_power_shifted_mw"] = raw
    out["rule_based_power_mw"] = float(a) * raw + float(b)
    calibration: Dict[str, object] = {
        "applied": True,
        "strategy": "offset_affine",
        "shift_slots": 0,
        "a": float(a),
        "b_mw": float(b),
    }
    return out, calibration


def _write_bundle(
    *,
    bundle_dir: Path,
    model: xgb.XGBRegressor,
    features: Sequence[str],
    offset_s: float,
    rule_based_fit: Dict[str, object],
    train_window_s: float,
    validation_window_s: float,
    capture_dir: Path,
    features_csv: Path,
    ground_truth_csv: Path,
    also_model_json: Path | None = None,
    residual_learning: bool = True,
) -> Path:
    """Write the frozen srsRAN model bundle (model.json + bundle.json). Shared by
    run (in-capture training) and train_only (cross-capture training) so both
    emit an identical bundle that run_frozen can consume."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(bundle_dir / "model.json")
    if also_model_json is not None:
        model.save_model(also_model_json)
    bundle = {
        "schema_version": 1,
        "pipeline": "srsran-xgboost",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_file": "model.json",
        "feature_columns": list(features),
        "feature_generation": {
            "categorical_source_columns": list(CATEGORICAL_FEATURES),
            "temporal_base_columns": list(TEMPORAL_BASES),
            "lag_slots": list(DEFAULT_LAGS),
            "rolling_windows": list(DEFAULT_ROLLING_WINDOWS),
        },
        "alignment": {"residual_offset_s": float(offset_s)},
        "rule_based_calibration": rule_based_fit,
        "residual_learning": bool(residual_learning),
        "training_windows": {
            "train_window_s": float(train_window_s),
            "validation_window_s": float(validation_window_s),
        },
        "xgboost_params": model.get_params(),
        "postprocessing": {"prediction_clip": None},
        "provenance": {
            "capture_dir": str(capture_dir),
            "features_csv": str(features_csv),
            "ground_truth_csv": str(ground_truth_csv),
        },
    }
    (bundle_dir / "bundle.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bundle_dir


def train_only(
    *,
    capture_dir: Path,
    features_csv: Path,
    ground_truth_csv: Path,
    output_dir: Path,
    model_dir: Path,
    train_window_s: float,
    validation_window_s: float,
    offset_search_ms: float,
    offset_step_ms: float,
    n_estimators: int,
    calibration_window_s: float = DEFAULT_CALIBRATION_WINDOW_S,
    gpp_output_dir: Path | None = None,
    train_info_dir: Path | None = None,
    xgb_params: Dict[str, object] | None = None,
    microsleep_power_mw: float = DEFAULT_MICROSLEEP_POWER_MW,
    feature_ablation: Mapping[str, object] | None = None,
    residual_learning: bool = True,
) -> Dict[str, object]:
    """Train the residual model and save the bundle, WITHOUT an in-capture
    holdout evaluation. Used for cross-capture experiments where the whole train
    capture is consumed by train+validation and the test capture is evaluated
    separately (run_frozen). The validation split is still used for early
    stopping; no in-capture test region is required."""
    output_dir.mkdir(parents=True, exist_ok=True)
    slots = load_slot_features(features_csv)
    gt = load_ground_truth(ground_truth_csv)
    layout = CaptureLayout(capture_dir)
    gpp_dir = gpp_output_dir or layout.srsran_3gpp_dir
    gpp_csv = gpp_dir / "power_per_slot.csv"
    if not gpp_csv.exists():
        run_3gpp(
            features_csv=features_csv,
            output_dir=gpp_dir,
            microsleep_power_mw=microsleep_power_mw,
        )
    rule_based = pd.read_csv(gpp_csv, low_memory=False)
    alignment = load_or_compute_alignment(
        capture_dir,
        features_csv=features_csv,
        ground_truth_csv=ground_truth_csv,
        gpp_csv=gpp_csv,
        slots=slots,
        gt=gt,
        rule_based=rule_based,
        calibration_window_s=calibration_window_s,
        search_ms=offset_search_ms,
        step_ms=offset_step_ms,
    )
    offset_s = float(alignment["residual_offset_s"])
    matrix, features = build_labeled_feature_matrix(
        slots,
        gt,
        residual_offset_s=offset_s,
        rule_based=rule_based,
        feature_ablation=feature_ablation,
    )
    matrix, rule_based_fit = apply_rule_calibration(
        matrix,
        a=float(alignment["rule_affine_a"]),
        b=float(alignment["rule_affine_b"]),
    )
    if normalize_scheme(feature_ablation)["include_rule_based_context"]:
        for column in ("rule_based_power_raw_mw", "rule_based_power_mw"):
            if column not in features:
                features.append(column)
    available_s = float(matrix["slot_offset_from_t0_s"].max())
    fit_window_s = float(train_window_s + validation_window_s)
    if available_s + 0.5 < fit_window_s:
        raise RuntimeError(
            f"Train capture aligned overlap is {available_s:.3f}s, shorter than "
            f"train ({train_window_s:.3f}s) + validation ({validation_window_s:.3f}s)."
        )
    train_indices, validation_indices, _ = _time_split_indices(
        matrix,
        train_window_s=train_window_s,
        validation_window_s=validation_window_s,
    )
    if min(len(train_indices), len(validation_indices)) < 1000:
        raise RuntimeError(
            "Insufficient train/validation rows: "
            f"train={len(train_indices)}, validation={len(validation_indices)}."
        )
    model, _, _ = _fit_residual_model(
        matrix,
        features,
        train_indices=train_indices,
        validation_indices=validation_indices,
        n_estimators=n_estimators,
        xgb_params=xgb_params,
        residual_learning=residual_learning,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_bundle(
        bundle_dir=model_dir,
        model=model,
        features=features,
        offset_s=offset_s,
        rule_based_fit=rule_based_fit,
        train_window_s=train_window_s,
        validation_window_s=validation_window_s,
        capture_dir=capture_dir,
        features_csv=features_csv,
        ground_truth_csv=ground_truth_csv,
        residual_learning=residual_learning,
    )
    summary = {
        "mode": "train_only",
        "capture_dir": str(capture_dir),
        "model_bundle": str(model_dir),
        "residual_alignment_offset_s": offset_s,
        "train_window_s": float(train_window_s),
        "validation_window_s": float(validation_window_s),
        "model_input_feature_count": int(len(features)),
        "rule_based_fit": rule_based_fit,
        "split_counts": {
            "train_rows": int(len(train_indices)),
            "validation_rows": int(len(validation_indices)),
        },
    }
    if train_info_dir is not None:
        summary["train_info"] = write_train_info(
            train_info_dir=train_info_dir,
            matrix=matrix,
            model=model,
            train_window_s=train_window_s,
            validation_window_s=validation_window_s,
            capture_dir=capture_dir,
        )
    (output_dir / "xgboost_train_only_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def run(
    *,
    capture_dir: Path,
    features_csv: Path,
    ground_truth_csv: Path,
    output_dir: Path,
    train_window_s: float,
    validation_window_s: float,
    offset_search_ms: float,
    offset_step_ms: float,
    n_estimators: int,
    calibration_window_s: float = DEFAULT_CALIBRATION_WINDOW_S,
    gpp_output_dir: Path | None = None,
    model_dir: Path | None = None,
    feature_output_dir: Path | None = None,
    train_info_dir: Path | None = None,
    xgb_params: Dict[str, object] | None = None,
    microsleep_power_mw: float = DEFAULT_MICROSLEEP_POWER_MW,
    feature_ablation: Mapping[str, object] | None = None,
    residual_learning: bool = True,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    slots = load_slot_features(features_csv)
    gt = load_ground_truth(ground_truth_csv)
    layout = CaptureLayout(capture_dir)
    gpp_dir = gpp_output_dir or layout.srsran_3gpp_dir
    gpp_csv = gpp_dir / "power_per_slot.csv"
    if not gpp_csv.exists():
        run_3gpp(
            features_csv=features_csv,
            output_dir=gpp_dir,
            microsleep_power_mw=microsleep_power_mw,
        )
    rule_based = pd.read_csv(gpp_csv, low_memory=False)
    fit_window_s = float(train_window_s + validation_window_s)
    # The residual clock offset is a per-capture constant, estimated once as
    # capture metadata (srsRAN/alignment/alignment.json) over a fixed
    # calibration window and read back here. It is selected by minimum 3GPP
    # calibration MAE and is independent of this experiment's train/validation
    # split. Computed lazily if a capture predates the data-prep alignment step.
    alignment = load_or_compute_alignment(
        capture_dir,
        features_csv=features_csv,
        ground_truth_csv=ground_truth_csv,
        gpp_csv=gpp_csv,
        slots=slots,
        gt=gt,
        rule_based=rule_based,
        calibration_window_s=calibration_window_s,
        search_ms=offset_search_ms,
        step_ms=offset_step_ms,
    )
    offset_s = float(alignment["residual_offset_s"])
    corr_offset_s = float(alignment["correlation_offset_s"])
    matrix, features = build_labeled_feature_matrix(
        slots,
        gt,
        residual_offset_s=offset_s,
        rule_based=rule_based,
        feature_ablation=feature_ablation,
    )
    available_s = float(matrix["slot_offset_from_t0_s"].max())
    if available_s <= fit_window_s + 0.5:
        raise RuntimeError(
            f"Aligned overlap is {available_s:.3f}s, but train "
            f"({train_window_s:.3f}s) + validation ({validation_window_s:.3f}s) "
            "must leave at least 0.5s for test."
        )
    effective_fit_s = fit_window_s
    # Reuse the bounded-affine (a, b) selected with the clock offset instead of
    # refitting it here; no extra feature shift (shift is redundant with the
    # offset already applied to the labels).
    matrix, rule_based_fit = apply_rule_calibration(
        matrix,
        a=float(alignment["rule_affine_a"]),
        b=float(alignment["rule_affine_b"]),
    )
    if normalize_scheme(feature_ablation)["include_rule_based_context"]:
        for column in ("rule_based_power_raw_mw", "rule_based_power_mw"):
            if column not in features:
                features.append(column)
    model, predictions, importance, metrics = train_and_evaluate(
        matrix,
        features,
        train_window_s=train_window_s,
        validation_window_s=validation_window_s,
        n_estimators=n_estimators,
        xgb_params=xgb_params,
        residual_learning=residual_learning,
    )

    total_label = _seconds_label(available_s)
    test_window_s = max(0.0, available_s - effective_fit_s)
    test_label = _seconds_label(test_window_s)
    fit_label = _seconds_label(effective_fit_s)
    eval_dir = output_dir / f"eval_epoch_offset_{_offset_label(offset_s)}"
    comparison_dir = output_dir.parent / "comparison"
    others_dir = eval_dir / "others"
    feature_root = feature_output_dir or eval_dir
    feature_cols_dir = feature_root / "feature_cols"
    offset_dir = output_dir / "offset_search"
    for path in (
        eval_dir,
        comparison_dir,
        others_dir,
        feature_cols_dir,
        offset_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = feature_root / f"xgboost_feature_rows_first_{total_label}.csv"
    predictions_path = eval_dir / f"xgboost_holdout_predictions_{test_label}.csv"
    importance_path = eval_dir / f"xgboost_feature_importance_{fit_label}_train_{test_label}_test.csv"
    model_path = (
        model_dir / "model.json"
        if model_dir is not None
        else eval_dir / "xgboost_model.json"
    )
    schema_path = (
        model_dir / "feature_schema.json"
        if model_dir is not None
        else eval_dir / "xgboost_feature_schema.json"
    )
    summary_path = eval_dir / f"xgboost_eval_summary_{fit_label}_train_{test_label}_test.json"

    export_columns = [
        "global_slot",
        "frame",
        "slot",
        "slot_epoch_s",
        "target_epoch_s",
        "slot_offset_from_t0_s",
        "activity_class" if "activity_class" in matrix else None,
        "rule_based_state",
        "target_power_mw",
        "rule_based_power_raw_mw",
        "rule_based_power_shifted_mw",
        "rule_based_power_mw",
        *features,
    ]
    export_columns = list(dict.fromkeys(column for column in export_columns if column is not None))
    matrix[export_columns].to_csv(matrix_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    importance.to_csv(importance_path, index=False)
    # Offset diagnostics are produced once with the cached alignment metadata;
    # copy them into this result for self-containedness.
    _diag = alignment.get("diagnostics", {}) if isinstance(alignment, dict) else {}
    for src_key, dst_name in (
        ("correlation_offset_scores_csv", "correlation_offset_scores.csv"),
        ("gpp_calibration_mae_offset_scores_csv", "gpp_calibration_mae_offset_scores.csv"),
    ):
        src = _diag.get(src_key)
        if src and Path(src).exists():
            shutil.copyfile(src, offset_dir / dst_name)
    model.save_model(model_path)
    schema_path.write_text(
        json.dumps(
            {
                "feature_columns": list(features),
                "categorical_source_columns": list(CATEGORICAL_FEATURES),
                "temporal_base_columns": list(TEMPORAL_BASES),
                "lag_slots": list(DEFAULT_LAGS),
                "rolling_windows": list(DEFAULT_ROLLING_WINDOWS),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    holdout_plot = output_dir / f"xgboost_holdout_gt_vs_estimate_{test_label}.pdf"
    residual_xgb = output_dir / f"xgboost_holdout_residuals_{test_label}.pdf"
    holdout_with_3gpp = (
        comparison_dir / f"srsran_xgboost_3gpp_gt_vs_estimate_{test_label}.pdf"
    )
    holdout_unfitted = comparison_dir / (
        f"srsran_xgboost_3gpp_gt_vs_estimate_{test_label}_unfitted_3gpp.pdf"
    )
    residual_plot = (
        comparison_dir / f"srsran_xgboost_3gpp_residuals_{test_label}.pdf"
    )
    residual_gpp = comparison_dir / (
        f"srsran_xgboost_3gpp_residuals_{test_label}_3gpp_fitted.pdf"
    )
    fit_gt_plot = others_dir / f"xgboost_fit_window_ground_truth_{fit_label}.pdf"
    importance_plot = others_dir / f"xgboost_feature_importance_top25_{fit_label}_train_{test_label}_test.pdf"
    _plot_predictions(
        predictions,
        holdout_plot,
        include_rule=False,
        rule_column="rule_based_power_mw",
    )
    _plot_predictions(
        predictions,
        holdout_with_3gpp,
        include_rule=True,
        rule_column="rule_based_power_mw",
    )
    _plot_predictions(predictions, holdout_unfitted, include_rule=True, rule_column="rule_based_power_raw_mw")
    _plot_residuals(predictions, residual_plot, ("residual_pred_mw", "residual_rule_based_mw"))
    _plot_residuals(predictions, residual_xgb, ("residual_pred_mw",))
    _plot_residuals(predictions, residual_gpp, ("residual_rule_based_mw",))
    _plot_fit_ground_truth(
        matrix,
        train_window_s,
        validation_window_s,
        fit_gt_plot,
    )
    _plot_feature_importance(importance, importance_plot)
    _export_feature_families(matrix, matrix_path, feature_cols_dir)
    # 3GPP-vs-ground-truth figures over the hold-out interval (same rows the
    # XGBoost comparison uses), written next to the 3GPP CSV.
    if gpp_output_dir is not None:
        write_gpp_vs_ground_truth(
            predictions,
            gpp_dir,
            a=rule_based_fit.get("a"),
            b=rule_based_fit.get("b_mw"),
        )
    # Hold-out test data on its own, at the root of the srsRAN eval folder.
    write_holdout_test_data(predictions, output_dir.parent)

    selected_by = str(alignment.get("selected_by", "gpp_calibration_mae"))
    mae_best = {"offset_s": offset_s, "cal_mae_mw": alignment.get("cal_mae_mw")}
    offset_summary = {
        "capture_dir": str(capture_dir),
        "slot_csv": str(features_csv),
        "ground_truth_csv": str(ground_truth_csv),
        "timezone_offset_sec": 0.0,
        "selected_offset_s": offset_s,
        "selected_by": selected_by,
        "alignment_metadata_json": str(layout.srsran_alignment_json),
        "calibration_window_s": alignment.get("calibration_window_s"),
        "correlation": {
            "best": {"offset_s": corr_offset_s},
            "score_description": "Pearson correlation of 10 ms activity and INA219 first differences (diagnostic only).",
        },
        "gpp_calibration_mae": {
            "best": mae_best,
            "selection_metric": "Minimum bounded-affine 3GPP calibration MAE; selected as the residual offset (robust to train/validation window). Note: this is a 3GPP-baseline calibration MAE, not the XGBoost model MAE.",
        },
    }
    (offset_dir / "offset_search_summary.json").write_text(
        json.dumps(offset_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (offset_dir / "offset_search.log").write_text(
        f"Selected residual offset {offset_s:.6f}s by minimum 3GPP calibration MAE "
        f"(correlation argmax was {corr_offset_s:.6f}s, diagnostic only).\n",
        encoding="utf-8",
    )

    summary = {
        "capture_dir": str(capture_dir),
        "features_csv": str(features_csv),
        "ground_truth_csv": str(ground_truth_csv),
        "output_dir": str(eval_dir),
        "residual_alignment_offset_s": offset_s,
        "offset_search_ms": offset_search_ms,
        "offset_step_ms": offset_step_ms,
        "fit_window_s": effective_fit_s,
        "train_window_s": float(train_window_s),
        "validation_window_s": float(validation_window_s),
        "overlap_duration_s": available_s,
        "lag_slots": list(DEFAULT_LAGS),
        "rolling_windows": list(DEFAULT_ROLLING_WINDOWS),
        "test_window_s": test_window_s,
        "feature_matrix_rows": int(len(matrix)),
        "feature_matrix_columns": int(len(matrix.columns)),
        "model_input_feature_count": int(len(features)),
        "split_counts": {
            "train_rows": metrics["train_rows"],
            "validation_rows": metrics["validation_rows"],
            "test_rows": metrics["test_rows"],
        },
        "metrics": {
            "XGBoost": metrics["XGBoost"],
            "3GPP (shift+affine)": metrics["3GPP (shift+affine)"],
            "3GPP raw": metrics["3GPP raw"],
            "training": {
                "train_rows_total": metrics["train_rows"],
                "train_rows_used": metrics["train_rows"],
                "train_rows_dropped_idle": 0,
            },
        },
        "rule_based_fit": rule_based_fit,
        "alignment": {
            "timestamp_source": "FAPI Slot.indication time_point",
            "manual_coarse_shift_sec": offset_s,
            "description": "FAPI epoch timestamps plus residual offset before INA219 interpolation.",
        },
        "figure_dirs": {
            "comparison": str(comparison_dir),
            "xgboost": str(output_dir),
            "others": str(others_dir),
        },
        "feature_matrix_csv": str(matrix_path),
        "predictions_csv": str(predictions_path),
        "feature_importance_csv": str(importance_path),
        "holdout_plot_pdf": str(holdout_plot),
        "holdout_with_fitted_3gpp_pdf": str(holdout_with_3gpp),
        "holdout_plot_unfitted_3gpp_pdf": str(holdout_unfitted),
        "holdout_residual_pdf": str(residual_plot),
        "holdout_residual_xgboost_pdf": str(residual_xgb),
        "holdout_residual_3gpp_fitted_pdf": str(residual_gpp),
        "fit_window_ground_truth_plot_pdf": str(fit_gt_plot),
        "feature_importance_plot_pdf": str(importance_plot),
        "artifacts": {
            "feature_matrix_csv": str(matrix_path),
            "predictions_csv": str(predictions_path),
            "feature_importance_csv": str(importance_path),
            "offset_scores_csv": str(offset_dir / "correlation_offset_scores.csv"),
            "model_json": str(model_path),
            "feature_schema_json": str(schema_path),
            "holdout_plot_pdf": str(holdout_plot),
            "holdout_with_fitted_3gpp_pdf": str(holdout_with_3gpp),
            "holdout_plot_unfitted_3gpp_pdf": str(holdout_unfitted),
            "holdout_residual_pdf": str(residual_plot),
            "holdout_residual_xgboost_pdf": str(residual_xgb),
            "holdout_residual_3gpp_fitted_pdf": str(residual_gpp),
            "fit_window_ground_truth_plot_pdf": str(fit_gt_plot),
            "feature_importance_plot_pdf": str(importance_plot),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (eval_dir / "xgboost_eval.log").write_text(
        (
            f"rows={len(matrix)} features={len(features)} "
            f"offset_s={offset_s:.6f} "
            f"xgboost_mae_mw={metrics['XGBoost']['mae_mw']:.6f} "
            f"fitted_3gpp_mae_mw={metrics['3GPP (shift+affine)']['mae_mw']:.6f}\n"
        ),
        encoding="utf-8",
    )
    (output_dir / "xgboost_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    bundle_dir = model_dir or output_dir / "model_bundle"
    _write_bundle(
        bundle_dir=bundle_dir,
        model=model,
        features=features,
        offset_s=offset_s,
        rule_based_fit=rule_based_fit,
        train_window_s=train_window_s,
        validation_window_s=validation_window_s,
        capture_dir=capture_dir,
        features_csv=features_csv,
        ground_truth_csv=ground_truth_csv,
        also_model_json=(None if model_dir is not None else output_dir / "xgboost_model.json"),
        residual_learning=residual_learning,
    )
    if train_info_dir is not None:
        summary["train_info"] = write_train_info(
            train_info_dir=train_info_dir,
            matrix=matrix,
            model=model,
            train_window_s=train_window_s,
            validation_window_s=validation_window_s,
            capture_dir=capture_dir,
        )
    return summary


def _apply_frozen_rule_calibration(
    matrix: pd.DataFrame,
    calibration: Dict[str, object],
) -> pd.DataFrame:
    if not calibration.get("applied"):
        raise RuntimeError("Frozen bundle does not contain an applied rule-based calibration.")
    raw = matrix["rule_based_power_raw_mw"].to_numpy(dtype=np.float64)
    shift = int(calibration["shift_slots"])
    shifted = pd.Series(raw).shift(shift).ffill().bfill().to_numpy(dtype=np.float64)
    out = matrix.copy()
    out["rule_based_power_shifted_mw"] = shifted
    out["rule_based_power_mw"] = (
        float(calibration["a"]) * shifted + float(calibration["b_mw"])
    )
    return out


def _prepare_frozen(
    *,
    capture_dir: Path,
    features_csv: Path,
    ground_truth_csv: Path,
    model_bundle: Path,
    gpp_output_dir: Path | None = None,
    microsleep_power_mw: float = DEFAULT_MICROSLEEP_POWER_MW,
) -> Dict[str, object]:
    """ONE-TIME heavy work: validate bundle, build feature matrix, load model.

    Returns a ``prep`` dict that can be passed to ``_evaluate_frozen_window``
    one or more times without rebuilding the feature matrix.
    """
    bundle_path = model_bundle / "bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != 1 or bundle.get("pipeline") != "srsran-xgboost":
        raise RuntimeError(f"Incompatible srsRAN model bundle: {bundle_path}")
    generation = bundle["feature_generation"]
    if tuple(generation["lag_slots"]) != DEFAULT_LAGS or tuple(generation["rolling_windows"]) != DEFAULT_ROLLING_WINDOWS:
        raise RuntimeError("Frozen bundle temporal feature settings do not match this code version.")

    slots = load_slot_features(features_csv)
    gt = load_ground_truth(ground_truth_csv)
    layout = CaptureLayout(capture_dir)
    gpp_dir = gpp_output_dir or layout.srsran_3gpp_dir
    gpp_csv = gpp_dir / "power_per_slot.csv"
    if not gpp_csv.exists():
        run_3gpp(
            features_csv=features_csv,
            output_dir=gpp_dir,
            microsleep_power_mw=microsleep_power_mw,
        )
    rule_based = pd.read_csv(gpp_csv, low_memory=False)
    # The residual clock offset is a property of THIS capture, not of the model.
    # For cross-capture inference (train on A, test on B) we must align B with
    # B's own offset, so prefer the capture's cached alignment metadata; only
    # fall back to the bundle's offset for captures lacking it.
    capture_alignment = load_alignment(capture_dir)
    residual_offset_s = (
        float(capture_alignment["residual_offset_s"])
        if capture_alignment is not None
        else float(bundle["alignment"]["residual_offset_s"])
    )
    matrix, current_features = build_labeled_feature_matrix(
        slots,
        gt,
        residual_offset_s=residual_offset_s,
        rule_based=rule_based,
    )
    matrix = _apply_frozen_rule_calibration(
        matrix,
        bundle["rule_based_calibration"],
    )
    expected_features = list(bundle["feature_columns"])
    missing = [name for name in expected_features if name not in matrix]
    non_dummy_missing = [
        name for name in missing
        if not any(name.startswith(f"{column}_") for column in CATEGORICAL_FEATURES)
    ]
    if non_dummy_missing:
        raise RuntimeError(
            "Frozen srsRAN feature schema is incompatible; missing columns: "
            + ", ".join(non_dummy_missing)
        )
    for name in missing:
        matrix[name] = 0.0

    model = xgb.XGBRegressor()
    model.load_model(model_bundle / str(bundle["model_file"]))

    return {
        "bundle": bundle,
        "matrix": matrix,
        "model": model,
        "expected_features": expected_features,
        "current_features": current_features,
        "missing": missing,
        "gpp_dir": gpp_dir,
        "residual_learning": bool(bundle.get("residual_learning", True)),
        "capture_alignment": capture_alignment,
        "residual_offset_s": residual_offset_s,
        "capture_dir": capture_dir,
        "features_csv": features_csv,
        "ground_truth_csv": ground_truth_csv,
    }


def _evaluate_frozen_window(
    prep: Dict[str, object],
    *,
    output_dir: Path,
    model_bundle: Path,
    train_window_s: float,
    validation_window_s: float,
    test_start_s: float | None = None,
    test_end_s: float | None = None,
    gpp_output_dir: Path | None = None,
) -> Dict[str, object]:
    """PER-WINDOW evaluation using a pre-built ``prep`` dict from ``_prepare_frozen``.

    Reads matrix/model/bundle from ``prep`` and writes predictions for the
    time window [test_start_s, test_end_s) (or the leftover region after
    train+validation when both are None). Does NOT mutate ``prep["matrix"]``.
    """
    bundle = prep["bundle"]
    matrix = prep["matrix"]
    model = prep["model"]
    expected_features = prep["expected_features"]
    current_features = prep["current_features"]
    missing = prep["missing"]
    gpp_dir = gpp_output_dir or prep["gpp_dir"]
    residual_learning = prep["residual_learning"]
    capture_alignment = prep["capture_alignment"]
    residual_offset_s = prep["residual_offset_s"]
    capture_dir = prep["capture_dir"]
    features_csv = prep["features_csv"]
    ground_truth_csv = prep["ground_truth_csv"]

    output_dir.mkdir(parents=True, exist_ok=True)

    if test_start_s is not None or test_end_s is not None:
        # Explicit test window on this capture (cross-capture eval): evaluate
        # [test_start_s, test_end_s) of the capture rather than the leftover
        # region after train+validation.
        offsets = matrix["slot_offset_from_t0_s"].astype(float)
        mask = offsets >= float(test_start_s or 0.0)
        if test_end_s is not None:
            mask = mask & (offsets < float(test_end_s))
        test_indices = matrix.index[mask]
    else:
        _, _, test_indices = _time_split_indices(
            matrix,
            train_window_s=train_window_s,
            validation_window_s=validation_window_s,
        )
    if len(test_indices) < 1000:
        raise RuntimeError(f"Insufficient frozen-inference test rows: {len(test_indices)}")
    X_test = matrix.loc[test_indices, expected_features].astype(np.float32)
    model_output = model.predict(X_test)
    fitted_rule = matrix.loc[test_indices, "rule_based_power_mw"].to_numpy(dtype=np.float64)
    if residual_learning:
        prediction = model_output + fitted_rule
    else:
        prediction = model_output.astype(np.float64)
    truth = matrix.loc[test_indices, "target_power_mw"].to_numpy(dtype=np.float64)
    predictions = matrix.loc[
        test_indices,
        [
            "global_slot", "frame", "slot", "slot_epoch_s", "target_epoch_s",
            "slot_offset_from_t0_s", "target_power_mw", "rule_based_state",
            "rule_based_power_raw_mw", "rule_based_power_shifted_mw",
            "rule_based_power_mw",
        ],
    ].copy()
    predictions["predicted_power_mw_raw"] = prediction
    predictions["predicted_power_mw"] = prediction
    predictions["predicted_power_mw_base_model"] = prediction
    if residual_learning:
        predictions["residual_predicted_by_model_mw"] = model_output
    predictions["residual_mw"] = truth - prediction
    predictions["residual_pred_mw"] = truth - prediction
    predictions["residual_rule_based_mw"] = truth - fitted_rule

    test_window_s = float(
        predictions["slot_offset_from_t0_s"].max()
        - predictions["slot_offset_from_t0_s"].min()
    )
    test_label = _seconds_label(test_window_s)
    predictions_path = output_dir / f"xgboost_holdout_predictions_{test_label}.csv"
    holdout_plot = output_dir / f"xgboost_holdout_gt_vs_estimate_{test_label}.pdf"
    residual_plot = output_dir / f"xgboost_holdout_residuals_{test_label}.pdf"
    predictions.to_csv(predictions_path, index=False)
    _plot_predictions(predictions, holdout_plot, include_rule=False, rule_column="rule_based_power_mw")
    _plot_residuals(predictions, residual_plot, ("residual_pred_mw",))
    # 3GPP-vs-ground-truth figures over the hold-out interval. The fitted column
    # uses the (a, b) frozen from the TRAIN capture (bundle calibration), so it
    # matches what the residual model actually applied to this test capture.
    if gpp_output_dir is not None:
        _frozen_cal = bundle["rule_based_calibration"]
        write_gpp_vs_ground_truth(
            predictions,
            gpp_dir,
            a=_frozen_cal.get("a"),
            b=_frozen_cal.get("b_mw"),
        )
    # Hold-out test data on its own, at the root of the srsRAN eval folder.
    write_holdout_test_data(predictions, output_dir.parent)
    summary = {
        "mode": "frozen",
        "capture_dir": str(capture_dir),
        "model_bundle": str(model_bundle),
        "features_csv": str(features_csv),
        "ground_truth_csv": str(ground_truth_csv),
        "predictions_csv": str(predictions_path),
        "holdout_plot_pdf": str(holdout_plot),
        "holdout_residual_xgboost_pdf": str(residual_plot),
        "train_window_s": float(train_window_s),
        "validation_window_s": float(validation_window_s),
        "test_window_s": test_window_s,
        "model_input_feature_count": len(expected_features),
        "schema": {
            "missing_one_hot_columns_filled_zero": missing,
            "current_candidate_feature_count": len(current_features),
        },
        "metrics": {
            "XGBoost": _metric_block(truth, prediction),
            "3GPP (frozen shift+affine)": _metric_block(truth, fitted_rule),
        },
        "energy": {
            "ground_truth_j": float(np.trapezoid(truth, predictions["target_epoch_s"]) / 1000.0),
            "predicted_j": float(np.trapezoid(prediction, predictions["target_epoch_s"]) / 1000.0),
        },
        "alignment": {
            "residual_offset_s": residual_offset_s,
            "offset_source": "capture_metadata" if capture_alignment is not None else "model_bundle",
            "bundle_residual_offset_s": float(bundle["alignment"]["residual_offset_s"]),
        },
        "rule_based_fit": bundle["rule_based_calibration"],
    }
    ground_energy = summary["energy"]["ground_truth_j"]
    summary["energy"]["deviation_pct"] = (
        (summary["energy"]["predicted_j"] - ground_energy) / ground_energy * 100.0
        if ground_energy
        else None
    )
    (output_dir / "xgboost_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def run_frozen(
    *,
    capture_dir: Path,
    features_csv: Path,
    ground_truth_csv: Path,
    output_dir: Path,
    model_bundle: Path,
    train_window_s: float,
    validation_window_s: float,
    gpp_output_dir: Path | None = None,
    test_start_s: float | None = None,
    test_end_s: float | None = None,
    microsleep_power_mw: float = DEFAULT_MICROSLEEP_POWER_MW,
) -> Dict[str, object]:
    """Thin wrapper: prepare once, evaluate one window. Preserves existing API."""
    prep = _prepare_frozen(
        capture_dir=capture_dir,
        features_csv=features_csv,
        ground_truth_csv=ground_truth_csv,
        model_bundle=model_bundle,
        gpp_output_dir=gpp_output_dir,
        microsleep_power_mw=microsleep_power_mw,
    )
    return _evaluate_frozen_window(
        prep,
        output_dir=output_dir,
        model_bundle=model_bundle,
        train_window_s=train_window_s,
        validation_window_s=validation_window_s,
        test_start_s=test_start_s,
        test_end_s=test_end_s,
        gpp_output_dir=gpp_output_dir,
    )


def _metric_block(truth: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    return {
        "mae_mw": float(mean_absolute_error(truth, prediction)),
        "rmse_mw": float(math.sqrt(mean_squared_error(truth, prediction))),
        "mape_pct": float(
            np.mean(np.abs((truth - prediction) / np.maximum(np.abs(truth), 1e-6)))
            * 100.0
        ),
    }


def _plot_predictions(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    include_rule: bool,
    rule_column: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    x = frame["slot_offset_from_t0_s"].to_numpy(dtype=float)
    ax.plot(x, frame["target_power_mw"], label="INA219", linewidth=1.0, color=GROUND_TRUTH_COLOR)
    ax.plot(x, frame["predicted_power_mw"], label="srsRAN XGBoost", linewidth=1.0, color=XGBOOST_COLOR)
    if include_rule:
        label = "3GPP fitted" if rule_column == "rule_based_power_mw" else "3GPP raw"
        ax.plot(x, frame[rule_column], label=label, linewidth=1.0, alpha=0.8, color=GPP_COLOR)
    ax.set_xlabel("Seconds from aligned overlap start")
    ax.set_ylabel("Power (mW)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_residuals(
    frame: pd.DataFrame,
    output_path: Path,
    columns: Sequence[str],
) -> None:
    labels = {
        "residual_pred_mw": "GT - XGBoost",
        "residual_rule_based_mw": "GT - 3GPP fitted",
    }
    colors = {
        "residual_pred_mw": XGBOOST_COLOR,
        "residual_rule_based_mw": GPP_COLOR,
    }
    fig, ax = plt.subplots(figsize=(12, 3.5))
    x = frame["slot_offset_from_t0_s"].to_numpy(dtype=float)
    for column in columns:
        ax.plot(x, frame[column], label=labels[column], linewidth=1.0, color=colors[column])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Seconds from aligned overlap start")
    ax.set_ylabel("Residual (mW)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_fit_ground_truth(
    matrix: pd.DataFrame,
    train_window_s: float,
    validation_window_s: float,
    output_path: Path,
) -> None:
    fit_window_s = float(train_window_s + validation_window_s)
    fit = matrix[matrix["slot_offset_from_t0_s"] < fit_window_s]
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(fit["slot_offset_from_t0_s"], fit["target_power_mw"], linewidth=1.0)
    ax.axvspan(train_window_s, fit_window_s, color="tab:orange", alpha=0.1)
    ax.set_xlabel("Seconds from aligned overlap start")
    ax.set_ylabel("Power (mW)")
    ax.set_title("Fit Window Ground Truth")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_train_info(
    *,
    train_info_dir: Path,
    matrix: pd.DataFrame,
    model: xgb.XGBRegressor,
    train_window_s: float,
    validation_window_s: float,
    capture_dir: Path,
) -> Dict[str, object]:
    """Emit the per-result training diagnostics into ``train_info_dir``.

    Writes three things, each as both a vector PDF and a raster JPG plus the raw
    numbers behind them:

    * ``train_validation_data.{pdf,jpg}`` -- the ground-truth power timeline over
      the train + validation windows actually used to fit the model, shaded by
      split. The whole window comes from a single capture (enforced upstream by
      the train/validation same-capture config check), so the figure is one
      continuous trace.
    * ``loss_curves.{pdf,jpg}`` and ``loss_curves.csv`` -- the XGBoost train and
      validation loss per boosting round, with the early-stopping best iteration
      marked, and the actual per-round values as a CSV.
    """
    train_info_dir.mkdir(parents=True, exist_ok=True)
    capture_id = Path(capture_dir).name
    fit_window_s = float(train_window_s + validation_window_s)

    # --- Train/validation ground-truth data figure ---
    fit = matrix[matrix["slot_offset_from_t0_s"] < fit_window_s]
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(
        fit["slot_offset_from_t0_s"],
        fit["target_power_mw"],
        linewidth=1.0,
        color="tab:blue",
    )
    ax.axvspan(0.0, train_window_s, color="tab:green", alpha=0.10,
               label=f"train ({train_window_s:g}s)")
    ax.axvspan(train_window_s, fit_window_s, color="tab:orange", alpha=0.18,
               label=f"validation ({validation_window_s:g}s)")
    ax.set_xlabel("Seconds from aligned overlap start")
    ax.set_ylabel("Power (mW)")
    ax.set_title(f"Train + validation data (capture {capture_id})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    data_pdf = train_info_dir / "train_validation_data.pdf"
    fig.savefig(data_pdf)
    fig.savefig(train_info_dir / "train_validation_data.jpg", dpi=150)
    plt.close(fig)

    # --- Loss curves (train + validation) ---
    evals = model.evals_result()
    curves: Dict[str, List[float]] = {}
    metric_label = "rmse"
    for split_key, label in (("validation_0", "train"), ("validation_1", "validation")):
        split_metrics = evals.get(split_key)
        if split_metrics:
            metric_label = next(iter(split_metrics))
            curves[label] = [float(v) for v in split_metrics[metric_label]]

    loss_csv = train_info_dir / "loss_curves.csv"
    rounds = max((len(values) for values in curves.values()), default=0)
    order = [name for name in ("train", "validation") if name in curves]
    with loss_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["boosting_round", *[f"{name}_{metric_label}" for name in order]])
        for i in range(rounds):
            writer.writerow(
                [i, *[(curves[name][i] if i < len(curves[name]) else "") for name in order]]
            )

    raw_best = getattr(model, "best_iteration", None)
    best_iteration = int(raw_best) if raw_best is not None else -1
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, color in (("train", "tab:blue"), ("validation", "tab:orange")):
        if name in curves:
            ax.plot(range(len(curves[name])), curves[name], label=name,
                    linewidth=1.2, color=color)
    if best_iteration >= 0:
        ax.axvline(best_iteration, color="black", linestyle="--", linewidth=0.8,
                   label=f"best_iteration={best_iteration}")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel(f"{metric_label} of residual (mW)")
    ax.set_title(f"Training and validation loss (capture {capture_id})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(train_info_dir / "loss_curves.pdf")
    fig.savefig(train_info_dir / "loss_curves.jpg", dpi=150)
    plt.close(fig)

    return {
        "train_info_dir": str(train_info_dir),
        "capture_id": capture_id,
        "train_window_s": float(train_window_s),
        "validation_window_s": float(validation_window_s),
        "loss_metric": metric_label,
        "best_iteration": best_iteration,
        "boosting_rounds_run": rounds,
        "train_validation_data_pdf": str(data_pdf),
        "loss_curves_csv": str(loss_csv),
    }


def write_gpp_vs_ground_truth(
    predictions: pd.DataFrame,
    output_dir: Path,
    *,
    a: float | None = None,
    b: float | None = None,
    also_jpg: bool = True,
) -> Dict[str, str]:
    """Write the raw and fitted 3GPP-vs-ground-truth figures over the HOLD-OUT
    (test) interval, drawn from the model's hold-out predictions so they line up
    exactly with the XGBoost hold-out comparison.

    ``predictions`` is the hold-out frame and must contain
    ``slot_offset_from_t0_s``, ``target_power_mw``, ``rule_based_power_raw_mw``
    and ``rule_based_power_mw``. The fitted column already carries the TRAINING
    ``(a, b)`` calibration the residual model used -- for a cross-capture run that
    is the calibration frozen from the train capture, not the test capture's own
    -- so the comparison is apples-to-apples with the model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    x = predictions["slot_offset_from_t0_s"].to_numpy(dtype=float)
    truth = predictions["target_power_mw"].to_numpy(dtype=float)
    written: Dict[str, str] = {}

    def _emit(column: str, estimate_label: str, title: str, stem: str) -> None:
        estimate = predictions[column].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(x, truth, label="INA219 ground truth", linewidth=1.0, color=GROUND_TRUTH_COLOR)
        ax.plot(x, estimate, label=estimate_label, linewidth=1.0, alpha=0.85, color=GPP_COLOR)
        ax.set_xlabel("Seconds from aligned overlap start")
        ax.set_ylabel("Power (mW)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        pdf_path = output_dir / f"{stem}.pdf"
        fig.savefig(pdf_path)
        if also_jpg:
            fig.savefig(output_dir / f"{stem}.jpg", dpi=150)
        plt.close(fig)
        written[stem] = str(pdf_path)

    fitted_label = "3GPP fitted (train-set calibration)"
    if a is not None and b is not None:
        fitted_label = f"3GPP fitted (train a={a:.3f}, b={b:.1f} mW)"
    _emit(
        "rule_based_power_raw_mw",
        "3GPP raw",
        "srsRAN 3GPP raw vs ground truth (hold-out)",
        "gpp_raw_vs_ground_truth",
    )
    _emit(
        "rule_based_power_mw",
        fitted_label,
        "srsRAN 3GPP fitted vs ground truth (hold-out)",
        "gpp_fitted_vs_ground_truth",
    )
    return written


def write_holdout_test_data(
    predictions: pd.DataFrame,
    output_dir: Path,
    *,
    also_jpg: bool = True,
) -> Dict[str, str]:
    """Plot the hold-out (test) data on its own -- the INA219 ground-truth power
    the model is evaluated against, over the hold-out interval, with no estimate
    overlaid. Written at the root of the srsRAN eval folder as
    ``holdout_test_data.{pdf,jpg}``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    x = predictions["slot_offset_from_t0_s"].to_numpy(dtype=float)
    truth = predictions["target_power_mw"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, truth, label="INA219 ground truth", linewidth=1.0, color=GROUND_TRUTH_COLOR)
    ax.set_xlabel("Seconds from aligned overlap start")
    ax.set_ylabel("Power (mW)")
    ax.set_title("Hold-out test data (ground truth)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    pdf_path = output_dir / "holdout_test_data.pdf"
    fig.savefig(pdf_path)
    if also_jpg:
        fig.savefig(output_dir / "holdout_test_data.jpg", dpi=150)
    plt.close(fig)
    return {"holdout_test_data_pdf": str(pdf_path)}


def _plot_feature_importance(frame: pd.DataFrame, output_path: Path) -> None:
    top = frame.head(25).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Importance")
    ax.set_title("XGBoost Feature Importance")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _export_feature_families(
    matrix: pd.DataFrame,
    source_path: Path,
    output_dir: Path,
) -> None:
    metadata = [
        name
        for name in (
            "global_slot",
            "frame",
            "slot",
            "slot_epoch_s",
            "target_epoch_s",
            "slot_offset_from_t0_s",
            "rule_based_state",
        )
        if name in matrix
    ]
    temporal = [name for name in matrix if name.startswith(("lag__", "roll__"))]
    geometry = [
        name
        for name in matrix
        if any(token in name for token in ("_prb", "_sym", "_crb", "_bwp", "_bw_mhz"))
        and name not in temporal
    ]
    resolved = [
        name
        for name in matrix
        if name.startswith(("drx_", "pdcch_monitor", "is_dl_", "is_ul_", "is_special_", "has_"))
        and name not in temporal
    ]
    rule = [
        name
        for name in (
            "target_power_mw",
            "rule_based_power_raw_mw",
            "rule_based_power_shifted_mw",
            "rule_based_power_mw",
        )
        if name in matrix
    ]
    assigned = set(metadata + temporal + geometry + resolved + rule)
    raw = [
        name
        for name in matrix
        if name not in assigned
        and name not in {"capture_id", "rnti", "slot_wall_time_iso"}
        and (pd.api.types.is_numeric_dtype(matrix[name]) or pd.api.types.is_bool_dtype(matrix[name]))
    ]
    families = {
        "metadata": metadata,
        "geometry": geometry,
        "raw_srsran_phy_sched": raw,
        "resolved_state": resolved,
        "temporal": temporal,
        "3gpp_context": rule,
    }
    stem = source_path.stem
    splits = []
    for family, columns in families.items():
        path = output_dir / f"{stem}_{family}.csv"
        selected = list(dict.fromkeys(metadata + columns))
        matrix[selected].to_csv(path, index=False)
        splits.append(
            {
                "file": path.name,
                "feature_family": family,
                "row_count": int(len(matrix)),
                "column_count": int(len(selected)),
            }
        )
    manifest = {
        "source_file": source_path.name,
        "leading_metadata_columns": metadata,
        "splits": splits,
    }
    (output_dir / f"{stem}_split_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def _seconds_label(value: float) -> str:
    rounded = round(float(value), 3)
    text = f"{rounded:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p") + "s"


def _offset_label(value: float) -> str:
    milliseconds = round(float(value) * 1000.0, 3)
    sign = "+" if milliseconds >= 0.0 else "-"
    magnitude = f"{abs(milliseconds):.3f}".rstrip("0").rstrip(".")
    return f"{sign}{magnitude.replace('.', 'p')}ms"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/test XGBoost power estimation from srsRAN features.")
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, default=None)
    parser.add_argument("--ground-truth-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-window-s", type=float, default=18.0)
    parser.add_argument("--validation-window-s", type=float, default=2.0)
    parser.add_argument("--offset-search-ms", type=float, default=250.0)
    parser.add_argument("--offset-step-ms", type=float, default=1.0)
    parser.add_argument(
        "--calibration-window-s", type=float, default=DEFAULT_CALIBRATION_WINDOW_S,
        help="Early window used to estimate the residual offset if not already cached.",
    )
    parser.add_argument("--n-estimators", type=int, default=1500)
    parser.add_argument("--microsleep-power-mw", type=float, default=DEFAULT_MICROSLEEP_POWER_MW)
    parser.add_argument("--mode", choices=("train-per-config", "frozen"), default="train-per-config")
    parser.add_argument("--model-bundle", type=Path, default=None)
    parser.add_argument("--gpp-output-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--feature-output-dir", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.train_window_s <= 0.0 or args.validation_window_s <= 0.0:
        parser.error("train and validation durations must both be positive")
    layout = CaptureLayout(args.capture_dir.resolve())
    gt = args.ground_truth_csv or layout.ground_truth_csv()
    if gt is None:
        raise FileNotFoundError(f"No INA219 CSV found under {layout.capture_dir}")
    common = {
        "capture_dir": layout.capture_dir,
        "features_csv": args.features_csv or layout.srsran_parsed_dir / "slot_features.csv",
        "ground_truth_csv": gt,
        "output_dir": args.output_dir or layout.srsran_xgboost_dir,
        "train_window_s": args.train_window_s,
        "validation_window_s": args.validation_window_s,
        "gpp_output_dir": args.gpp_output_dir,
        "microsleep_power_mw": args.microsleep_power_mw,
    }
    if args.mode == "frozen":
        if args.model_bundle is None:
            parser.error("--model-bundle is required in frozen mode")
        summary = run_frozen(model_bundle=args.model_bundle.resolve(), **common)
    else:
        summary = run(
            offset_search_ms=args.offset_search_ms,
            offset_step_ms=args.offset_step_ms,
            calibration_window_s=args.calibration_window_s,
            n_estimators=args.n_estimators,
            model_dir=args.model_dir,
            feature_output_dir=args.feature_output_dir,
            **common,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

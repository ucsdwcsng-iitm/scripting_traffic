"""Offline experiment orchestration with explicit data/model/result paths."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from scripts.experiments.config import REPO_ROOT, ExperimentConfig, capture_dir
from scripts.power_estimation.capture_layout import CaptureLayout


# Result subfolder holding the srsRAN power-estimation evaluation artifacts. The
# config pipeline key stays "srsran"; only the on-disk result folder is named
# "srsran_eval" to mark it as evaluation output.
SRSRAN_RESULT_SUBDIR = "srsran_eval"


def _slice_num(value: float) -> str:
    """Render a slice bound for a folder name: ``50`` not ``50.0``; ``2p5`` for 2.5."""
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value)).replace(".", "p")


def _slice_subdir(start_s: float, end_s: float) -> str:
    """Per-slice result folder name, e.g. ``srsran_eval_50s_to_60s``."""
    return f"{SRSRAN_RESULT_SUBDIR}_{_slice_num(start_s)}s_to_{_slice_num(end_s)}s"


# Progress lines are mirrored here so the per-result logs/ folder gets a durable
# copy of the run narration (stderr is otherwise the only place it lands). One
# experiment runs per process, so a module-level buffer is safe.
_LOG_LINES: list[str] = []


def _log(message: str) -> None:
    """Emit a timestamped progress line to stderr (stdout stays manifest JSON)."""
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    _LOG_LINES.append(line)
    print(line, file=sys.stderr, flush=True)


def _rel(path: Path) -> str:
    """Repo-relative POSIX path for portable provenance records."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _write_config_provenance(config: ExperimentConfig, target_dir: Path) -> None:
    """Store the exact config that produced the artifacts in `target_dir`.

    Two copies are written so a stored model or result is self-describing:
    `resolved_config.yaml` is the inheritance-merged, self-contained config used
    by the runner, and `config.source.yaml` is the verbatim file the user edited.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.resolved, sort_keys=False),
        encoding="utf-8",
    )
    shutil.copyfile(config.path, target_dir / "config.source.yaml")


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_result_dir(config: ExperimentConfig) -> Path:
    # One result slot per config identity: each run regenerates it so only the
    # newest result for a config is kept. The model is in a separate tree and is
    # never touched here.
    result_dir = config.result_dir
    if result_dir.exists():
        for name in (
            SRSRAN_RESULT_SUBDIR,
            "srsran_train",
            "train_info",
            "logs",
            "resolved_config.yaml",
            "config.source.yaml",
            "model_ref.json",
            "manifest.json",
        ):
            path = result_dir / name
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        # Sliced runs emit one srsran_eval_<a>s_to_<b>s/ folder per test slice;
        # clear any from a prior run so only the current slices remain.
        for path in result_dir.glob(f"{SRSRAN_RESULT_SUBDIR}_*"):
            if path.is_dir():
                shutil.rmtree(path)
    result_dir.mkdir(parents=True, exist_ok=True)
    # Sliced runs pre-create a per-slice xgboost/ folder instead of the bare
    # srsran_eval/; the shared 3GPP baseline lives in the feature cache.
    if config.test_sliced:
        for slc in config.tests:
            (result_dir / _slice_subdir(slc.start_s, slc.end_s) / "xgboost").mkdir(
                parents=True, exist_ok=True
            )
    else:
        for child in ("3gpp", "xgboost"):
            (result_dir / SRSRAN_RESULT_SUBDIR / child).mkdir(parents=True, exist_ok=True)
    (result_dir / "logs").mkdir(exist_ok=True)
    _write_config_provenance(config, result_dir)
    return result_dir


def _can_reuse_model(config: ExperimentConfig) -> bool:
    """True if an existing model can be reused (training-relevant config matches)."""
    specs_path = config.model_dir / "specs.json"
    if not specs_path.is_file():
        return False
    try:
        specs = json.loads(specs_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if specs.get("train_hash") != config.train_hash:
        return False
    return (config.model_dir / "bundle.json").is_file() and (
        config.model_dir / "model.json"
    ).is_file()


def _prepare_model_dir(config: ExperimentConfig, *, reuse: bool) -> None:
    if reuse:
        # Keep the already-trained model untouched; only evaluation re-runs.
        return
    if config.model_dir.exists():
        # Replace any prior model at this identity (a lone .gitkeep scaffold
        # placeholder is fine to drop). Newest training wins.
        leftovers = [p for p in config.model_dir.iterdir() if p.name != ".gitkeep"]
        if leftovers:
            shutil.rmtree(config.model_dir)
    config.model_dir.mkdir(parents=True, exist_ok=True)


def _canonical_input(config: ExperimentConfig) -> tuple[Path, CaptureLayout]:
    device = str(config.resolved["data"].get("device", "qualcomm"))
    capture = capture_dir(config.train.capture_id, device)
    return capture, CaptureLayout(capture)


def _assert_cached_microsleep(power_summary: Path, expected_mw: float) -> None:
    """Guard a reused 3GPP baseline against a config/cache micro-sleep mismatch.

    The 3GPP baseline is cached per-capture and reused as-is when present, so the
    config's preprocessing.microsleep_power_mw is NOT re-applied on a cache hit.
    The config value is therefore an assertion about how the cached baseline was
    produced: it must equal the micro-sleep floor recorded in the cache's
    power_summary.json (written by run_3gpp). If they differ, the cached 3GPP
    power -- and the XGBoost residuals derived from it -- do not correspond to the
    indicated micro-sleep power, so we refuse to proceed. Delete the cached 3gpp/
    folder to recompute it at the configured value.
    """
    if not power_summary.exists():
        raise RuntimeError(
            "3GPP power model was not run with the indicated microsleep power "
            f"value: cannot verify cached baseline, {_rel(power_summary)} is "
            f"missing (config indicates microsleep_power_mw={expected_mw:.0f}). "
            f"Delete {_rel(power_summary.parent)} to recompute it."
        )
    cached_mw = float(json.loads(power_summary.read_text())["microsleep_power_mw"])
    if abs(cached_mw - expected_mw) > 1e-6:
        raise RuntimeError(
            "3GPP power model was not run with the indicated microsleep power "
            f"value: config indicates microsleep_power_mw={expected_mw:.0f}, but "
            f"the cached 3GPP baseline {_rel(power_summary)} was run at "
            f"{cached_mw:.0f}. Delete {_rel(power_summary.parent)} to recompute it."
        )


def _prepare_srsran_capture(
    config: ExperimentConfig,
    capture: Path,
    layout: CaptureLayout,
    *,
    gpp_dir: Path,
    feature_subdir: str,
) -> tuple[Path, Path, dict[str, Any], float]:
    """Extract slot features, run the 3GPP baseline, and ensure the per-capture
    residual alignment for one capture. Returns (processed_dir, features_csv,
    alignment, calibration_window_s). Shared by the train and (cross-capture)
    test captures so each is aligned with its OWN clock skew."""
    from scripts.power_estimation.srsRAN.alignment import (
        DEFAULT_CALIBRATION_WINDOW_S,
        ensure_alignment,
    )
    from scripts.power_estimation.srsRAN.extract_features import extract_to_paths
    from scripts.power_estimation.srsRAN.run_3gpp import run as run_3gpp

    processed = layout.srsran_dir / "feature_sets" / feature_subdir
    features = processed / "slot_features.csv"
    summary = processed / "slot_feature_summary.json"
    sparse = processed / "parsed_rows.jsonl"
    if not features.exists():
        _log(f"srsRAN: extracting slot features from {layout.srsran_log.name} ...")
        extract_to_paths(
            log_path=layout.srsran_log,
            output_csv=features,
            summary_json=summary,
            sparse_rows_jsonl=sparse,
        )
        _log(f"srsRAN: features written to {_rel(features)}")
    else:
        _log(f"srsRAN: reusing cached features at {_rel(features)}")

    # The 3GPP baseline is a deterministic function of the slot features (fixed
    # power coefficients, no fitting), so cache it next to them and recompute only
    # when missing. When the requested gpp_dir is a different directory (e.g. the
    # result folder, which is wiped each run), copy the cached files in so reruns
    # do zero 3GPP work on a capture whose features already exist.
    preprocessing = config.resolved.get("preprocessing", {})
    microsleep_power_mw = float(preprocessing.get("microsleep_power_mw", 250.0))
    cache_gpp = processed / "3gpp"
    if (cache_gpp / "power_per_slot.csv").exists():
        _assert_cached_microsleep(cache_gpp / "power_summary.json", microsleep_power_mw)
        _log(f"srsRAN: reusing cached 3GPP baseline at {_rel(cache_gpp / 'power_per_slot.csv')}")
    else:
        _log(f"srsRAN: running 3GPP baseline (microsleep={microsleep_power_mw:.0f}mW) ...")
        run_3gpp(
            features_csv=features,
            output_dir=cache_gpp,
            microsleep_power_mw=microsleep_power_mw,
        )

    calibration_window_s = float(
        preprocessing.get("alignment_calibration_window_s", DEFAULT_CALIBRATION_WINDOW_S)
    )
    # Align against the cached 3GPP baseline. ensure_alignment caches the residual
    # offset + (a, b) per capture, and on first compute it also drops the
    # gt_vs_estimate diagnostic next to the cached baseline, so all 3GPP artifacts
    # live in the cache and survive the per-run wipe of the result folder.
    alignment = ensure_alignment(
        capture,
        features_csv=features,
        ground_truth_csv=layout.require_ground_truth_csv(),
        gpp_csv=cache_gpp / "power_per_slot.csv",
        calibration_window_s=calibration_window_s,
        search_ms=float(preprocessing.get("offset_search_ms", 250.0)),
        step_ms=float(preprocessing.get("offset_step_ms", 1.0)),
    )

    # Materialize the (now fully cached) 3GPP folder into the requested result
    # location. When gpp_dir IS the cache (cross-capture train baseline), there is
    # nothing to copy and it stays out of the result folder by design.
    if gpp_dir.resolve() != cache_gpp.resolve():
        gpp_dir.mkdir(parents=True, exist_ok=True)
        for name in ("power_per_slot.csv", "power_summary.json", "gt_vs_estimate.pdf"):
            source = cache_gpp / name
            if source.exists():
                shutil.copyfile(source, gpp_dir / name)
    _log(
        "srsRAN: residual offset "
        f"{float(alignment['residual_offset_s']) * 1000.0:+.1f}ms "
        f"(selected_by={alignment['selected_by']})"
    )
    return processed, features, alignment, calibration_window_s


def _uses_pool(config: ExperimentConfig) -> bool:
    """True when the config requests pooled synthetic training (data.train_pool)."""
    return bool(getattr(config, "train_pool", None))


def _train_pool(config: ExperimentConfig, model_params: dict[str, Any], device: str) -> None:
    from scripts.power_estimation.srsRAN.pooled_training import train_pooled

    pool = [capture_dir(cid, device) for cid in config.train_pool]
    _log(f"srsRAN: pooled training on {len(pool)} synthetic captures ...")
    train_pooled(
        pool,
        config.model_dir,
        train_window_s=config.train.duration_s,
        validation_window_s=config.validation.duration_s,
        n_estimators=int(model_params.get("n_estimators", 1500)),
        xgb_params=dict(model_params),
        cache_dir=config.model_dir.parents[2] / "_pool_feature_cache",
        residual_learning=config.residual_learning,
    )


def _run_srsran(
    config: ExperimentConfig,
    capture: Path,
    layout: CaptureLayout,
    result_dir: Path,
    *,
    reuse: bool,
) -> dict[str, Any]:
    from scripts.power_estimation.srsRAN.run_xgboost import run as run_xgboost
    from scripts.power_estimation.srsRAN.run_xgboost import run_frozen, train_only

    preprocessing = config.resolved.get("preprocessing", {})
    offset_search_ms = float(preprocessing.get("offset_search_ms", 250.0))
    offset_step_ms = float(preprocessing.get("offset_step_ms", 1.0))
    microsleep_power_mw = float(preprocessing.get("microsleep_power_mw", 250.0))
    model_params = config.resolved["model"].get("params", {})
    device = str(config.resolved["data"].get("device", "qualcomm"))
    ground_truth_csv = layout.require_ground_truth_csv()
    cross_capture = config.test.capture_id != config.train.capture_id

    # Prepare the (train) canonical capture. Feature extraction depends only on
    # training-relevant inputs, so cache it by train_hash. In a cross-capture run
    # the result baseline is the TEST capture's 3GPP output; the train capture's
    # baseline is only an intermediate for residual alignment, so keep it in the
    # train capture's feature cache rather than the result folder.
    if cross_capture or config.test_sliced:
        # Cross-capture and sliced runs both keep the train 3GPP baseline in the
        # feature cache rather than the result folder (sliced has no bare
        # srsran_eval/; cross-capture's result baseline is the test capture's).
        gpp_dir = layout.srsran_dir / "feature_sets" / config.train_hash[:16] / "3gpp"
    else:
        gpp_dir = result_dir / SRSRAN_RESULT_SUBDIR / "3gpp"
    processed, features, alignment, calibration_window_s = _prepare_srsran_capture(
        config, capture, layout, gpp_dir=gpp_dir, feature_subdir=config.train_hash[:16]
    )
    residual_offset_ms = float(alignment["residual_offset_s"]) * 1000.0

    if config.test_sliced:
        return _run_srsran_sliced(
            config,
            capture,
            layout,
            result_dir,
            reuse=reuse,
            train_features=features,
            train_processed=processed,
            ground_truth_csv=ground_truth_csv,
            cross_capture=cross_capture,
            calibration_window_s=calibration_window_s,
            residual_offset_ms=residual_offset_ms,
            offset_search_ms=offset_search_ms,
            offset_step_ms=offset_step_ms,
            microsleep_power_mw=microsleep_power_mw,
            model_params=model_params,
            device=device,
        )

    if not cross_capture:
        if reuse:
            _log("srsRAN: evaluating with frozen (reused) XGBoost model ...")
            xgb_summary = run_frozen(
                capture_dir=capture,
                features_csv=features,
                ground_truth_csv=ground_truth_csv,
                output_dir=result_dir / SRSRAN_RESULT_SUBDIR / "xgboost",
                model_bundle=config.model_dir,
                train_window_s=config.train.duration_s,
                validation_window_s=config.validation.duration_s,
                gpp_output_dir=gpp_dir,
                microsleep_power_mw=microsleep_power_mw,
            )
        else:
            _log(
                "srsRAN: training XGBoost "
                f"(n_estimators={model_params.get('n_estimators', 1500)}, "
                f"residual offset {residual_offset_ms:+.1f}ms reused) ..."
            )
            xgb_summary = run_xgboost(
                capture_dir=capture,
                features_csv=features,
                ground_truth_csv=ground_truth_csv,
                output_dir=result_dir / SRSRAN_RESULT_SUBDIR / "xgboost",
                gpp_output_dir=gpp_dir,
                model_dir=config.model_dir,
                feature_output_dir=processed / "training",
                train_info_dir=config.model_dir / "train_info",
                train_window_s=config.train.duration_s,
                validation_window_s=config.validation.duration_s,
                offset_search_ms=offset_search_ms,
                offset_step_ms=offset_step_ms,
                calibration_window_s=calibration_window_s,
                n_estimators=int(model_params.get("n_estimators", 1500)),
                xgb_params=dict(model_params),
                microsleep_power_mw=microsleep_power_mw,
                feature_ablation=config.feature_ablation,
                residual_learning=config.residual_learning,
            )
        eval_capture = capture
    else:
        # Cross-capture: train on the train capture (whole capture consumed by
        # train+validation, so no in-capture holdout) and evaluate config.test's
        # window on the SEPARATE test capture with the frozen bundle. The test
        # capture is aligned with its own clock skew (its own offset); only the
        # model's (a, b) calibration is frozen from training.
        if not reuse:
            _log(
                "srsRAN: training XGBoost (cross-capture, no in-capture holdout; "
                f"n_estimators={model_params.get('n_estimators', 1500)}; "
                f"residual offset {residual_offset_ms:+.1f}ms reused) ..."
            )
            if _uses_pool(config):
                # Pooled synthetic training: writes a run_frozen-compatible
                # bundle to config.model_dir.
                _train_pool(config, model_params, device)
            else:
                train_only(
                    capture_dir=capture,
                    features_csv=features,
                    ground_truth_csv=ground_truth_csv,
                    output_dir=result_dir / SRSRAN_RESULT_SUBDIR / "xgboost" / "train",
                    model_dir=config.model_dir,
                    train_info_dir=config.model_dir / "train_info",
                    train_window_s=config.train.duration_s,
                    validation_window_s=config.validation.duration_s,
                    offset_search_ms=offset_search_ms,
                    offset_step_ms=offset_step_ms,
                    calibration_window_s=calibration_window_s,
                    n_estimators=int(model_params.get("n_estimators", 1500)),
                    xgb_params=dict(model_params),
                    microsleep_power_mw=microsleep_power_mw,
                    feature_ablation=config.feature_ablation,
                    residual_learning=config.residual_learning,
                )
        else:
            _log("srsRAN: reusing existing XGBoost model (cross-capture) ...")

        test_capture = capture_dir(config.test.capture_id, device)
        test_layout = CaptureLayout(test_capture)
        # The test capture is the evaluation capture here, so its 3GPP baseline is
        # the one that belongs in the result folder, under the canonical "3gpp".
        test_gpp_dir = result_dir / SRSRAN_RESULT_SUBDIR / "3gpp"
        _, test_features, _test_alignment, _ = _prepare_srsran_capture(
            config, test_capture, test_layout, gpp_dir=test_gpp_dir, feature_subdir="frozen_eval"
        )
        _log(
            f"srsRAN: evaluating test capture {config.test.capture_id} "
            f"[{config.test.start_s}-{config.test.end_s}]s with frozen bundle ..."
        )
        xgb_summary = run_frozen(
            capture_dir=test_capture,
            features_csv=test_features,
            ground_truth_csv=test_layout.require_ground_truth_csv(),
            output_dir=result_dir / SRSRAN_RESULT_SUBDIR / "xgboost",
            model_bundle=config.model_dir,
            train_window_s=config.train.duration_s,
            validation_window_s=config.validation.duration_s,
            gpp_output_dir=test_gpp_dir,
            test_start_s=config.test.start_s,
            test_end_s=config.test.end_s,
            microsleep_power_mw=microsleep_power_mw,
        )
        eval_capture = test_capture

    _log("srsRAN: pipeline complete")
    return {
        "processed_dir": str(processed),
        "features_csv": str(features),
        "reused_model": reuse,
        "cross_capture": cross_capture,
        "eval_capture_dir": str(eval_capture),
        "xgboost": xgb_summary,
    }


def _run_srsran_sliced(
    config: ExperimentConfig,
    capture: Path,
    layout: CaptureLayout,
    result_dir: Path,
    *,
    reuse: bool,
    train_features: Path,
    train_processed: Path,
    ground_truth_csv: Path,
    cross_capture: bool,
    calibration_window_s: float,
    residual_offset_ms: float,
    offset_search_ms: float,
    offset_step_ms: float,
    microsleep_power_mw: float,
    model_params: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Train the model once, then evaluate each test slice with the frozen bundle
    into its own ``srsran_eval_<a>s_to_<b>s/`` folder.

    Works for in-capture (eval capture == train capture) and cross-capture (a
    separate test capture) alike; the slices differ only by their [start, end)
    window into the evaluation capture's timeline. The whole-capture 3GPP baseline
    is deterministic, so it is built once and shared across every slice.
    """
    from scripts.power_estimation.srsRAN.run_xgboost import (
        _prepare_frozen,
        _evaluate_frozen_window,
        run_frozen,
        train_only,
    )

    if not reuse:
        _log(
            "srsRAN: training XGBoost once for sliced eval "
            f"(n_estimators={model_params.get('n_estimators', 1500)}; "
            f"residual offset {residual_offset_ms:+.1f}ms reused) ..."
        )
        if _uses_pool(config):
            _train_pool(config, model_params, device)
        else:
            train_only(
                capture_dir=capture,
                features_csv=train_features,
                ground_truth_csv=ground_truth_csv,
                output_dir=result_dir / "srsran_train" / "xgboost",
                model_dir=config.model_dir,
                train_info_dir=config.model_dir / "train_info",
                train_window_s=config.train.duration_s,
                validation_window_s=config.validation.duration_s,
                offset_search_ms=offset_search_ms,
                offset_step_ms=offset_step_ms,
                calibration_window_s=calibration_window_s,
                n_estimators=int(model_params.get("n_estimators", 1500)),
                xgb_params=dict(model_params),
                microsleep_power_mw=microsleep_power_mw,
                feature_ablation=config.feature_ablation,
                residual_learning=config.residual_learning,
            )
    else:
        _log("srsRAN: reusing existing XGBoost model (sliced eval) ...")

    # Resolve the evaluation capture and its features/3GPP once. The cached 3GPP
    # baseline (in the feature cache, alongside features) is passed to every slice
    # so run_frozen reads it instead of recomputing it 20 times.
    if cross_capture:
        eval_capture = capture_dir(config.test.capture_id, device)
        eval_layout = CaptureLayout(eval_capture)
        eval_processed, eval_features, _, _ = _prepare_srsran_capture(
            config,
            eval_capture,
            eval_layout,
            gpp_dir=eval_layout.srsran_dir / "feature_sets" / "frozen_eval" / "3gpp",
            feature_subdir="frozen_eval",
        )
        eval_gt = eval_layout.require_ground_truth_csv()
    else:
        eval_capture = capture
        eval_features = train_features
        eval_processed = train_processed
        eval_gt = ground_truth_csv
    eval_gpp_dir = eval_processed / "3gpp"

    # Build the feature matrix ONCE for the entire evaluation capture, then
    # reuse it across every slice window — avoids rebuilding a 500k-row matrix
    # with ~700 lag/roll columns for each of the 20 slices (~1.5 min each).
    _log("srsRAN: building frozen-eval feature matrix once for all slices ...")
    frozen_prep = _prepare_frozen(
        capture_dir=eval_capture,
        features_csv=eval_features,
        ground_truth_csv=eval_gt,
        model_bundle=config.model_dir,
        gpp_output_dir=eval_gpp_dir,
        microsleep_power_mw=microsleep_power_mw,
    )

    slice_results: list[dict[str, Any]] = []
    for slc in config.tests:
        subdir = _slice_subdir(slc.start_s, slc.end_s)
        _log(
            f"srsRAN: frozen eval slice [{slc.start_s}-{slc.end_s}]s -> {subdir}/ ..."
        )
        summary = _evaluate_frozen_window(
            frozen_prep,
            output_dir=result_dir / subdir / "xgboost",
            model_bundle=config.model_dir,
            train_window_s=config.train.duration_s,
            validation_window_s=config.validation.duration_s,
            test_start_s=slc.start_s,
            test_end_s=slc.end_s,
            gpp_output_dir=eval_gpp_dir,
        )
        slice_results.append(
            {
                "start_s": slc.start_s,
                "end_s": slc.end_s,
                "subdir": subdir,
                "xgboost": summary,
            }
        )
    _log(f"srsRAN: sliced pipeline complete ({len(slice_results)} slices)")
    return {
        "processed_dir": str(eval_processed),
        "features_csv": str(eval_features),
        "reused_model": reuse,
        "cross_capture": cross_capture,
        "eval_capture_dir": str(eval_capture),
        "sliced": True,
        "slices": slice_results,
    }


def _write_model_specs(config: ExperimentConfig) -> None:
    config.model_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model_id": config.model_id,
        "source": "srsran",
        "algorithm": config.resolved["model"]["name"],
        "params": config.resolved["model"].get("params", {}),
        "other": config.resolved["model"].get("other", {}),
        "config_hash": config.config_hash,
        "train_hash": config.train_hash,
        "train": config.train.__dict__,
        "validation": config.validation.__dict__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }
    (config.model_dir / "specs.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_config_provenance(config, config.model_dir)


def _write_model_ref(config: ExperimentConfig, result_dir: Path, *, reused: bool) -> None:
    """Explicit, durable pointer from a result to the model it used."""
    ref = {
        "experiment_id": config.experiment_id,
        "variant_id": config.variant_id,
        "config": _rel(config.path),
        "model_dir": _rel(config.model_dir),
        "config_hash": config.config_hash,
        "train_hash": config.train_hash,
        "status": "reused" if reused else "trained",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (result_dir / "model_ref.json").write_text(
        json.dumps(ref, indent=2) + "\n",
        encoding="utf-8",
    )


def run_experiment(config: ExperimentConfig, *, force: bool = False) -> dict[str, Any]:
    reuse = (not force) and _can_reuse_model(config)
    _log(
        f"experiment {config.experiment_id}/{config.variant_id}: "
        f"{'reusing existing model' if reuse else 'training new model'}"
    )
    result_dir = _prepare_result_dir(config)
    _log(f"result dir prepared: {_rel(result_dir)}")
    _prepare_model_dir(config, reuse=reuse)
    capture, layout = _canonical_input(config)
    outputs: dict[str, Any] = {
        "srsran": _run_srsran(config, capture, layout, result_dir, reuse=reuse)
    }
    if not reuse:
        _write_model_specs(config)

    # Surface the training diagnostics (train/validation data figure + loss
    # curves and their raw values) next to the result. They are written into the
    # model bundle at train time, so a reused model still carries them; copy
    # whatever the bundle has.
    bundle_train_info = config.model_dir / "train_info"
    if bundle_train_info.is_dir():
        result_train_info = result_dir / "train_info"
        if result_train_info.exists():
            shutil.rmtree(result_train_info)
        shutil.copytree(bundle_train_info, result_train_info)

    manifest = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "variant_id": config.variant_id,
        "config_path": str(config.path),
        "config_hash": config.config_hash,
        "train_hash": config.train_hash,
        "model_id": config.model_id,
        "model_dir": str(config.model_dir),
        "model_reused": reuse,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "captures": {},
        "outputs": outputs,
    }
    device = str(config.resolved["data"].get("device", "qualcomm"))
    for split_name, split in (
        ("train", config.train),
        ("validation", config.validation),
        ("test", config.test),
    ):
        root = capture_dir(split.capture_id, device)
        metadata = root / "metadata.json"
        manifest["captures"][split_name] = {
            **split.__dict__,
            "metadata": str(metadata),
            "metadata_sha256": _sha256(metadata),
        }
    if config.test_sliced:
        # Record every test slice (the "test" entry above is just the first one).
        manifest["captures"]["test_slices"] = [
            {**slc.__dict__, "subdir": _slice_subdir(slc.start_s, slc.end_s)}
            for slc in config.tests
        ]
    (result_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_model_ref(config, result_dir, reused=reuse)
    _log(f"done — manifest written to {_rel(result_dir / 'manifest.json')}")
    # Persist the run narration so the result's logs/ folder isn't empty.
    (result_dir / "logs" / "runner.log").write_text(
        "\n".join(_LOG_LINES) + "\n", encoding="utf-8"
    )
    return manifest

"""Typed loading, inheritance, and validation for experiment YAML files."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
CONFIG_ROOT = REPO_ROOT / "experiment_configs"
SCHEMA_PATH = CONFIG_ROOT / "schema.yaml"
MODELS_ROOT = REPO_ROOT / "models"
RESULTS_ROOT = REPO_ROOT / "results"


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


@dataclass(frozen=True)
class TimeRange:
    capture_id: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    resolved: dict[str, Any]
    config_hash: str
    experiment_id: str
    variant_id: str
    train: TimeRange
    validation: TimeRange
    test: TimeRange
    # All test ranges. For a legacy single-mapping `data.test` this is just
    # `(test,)` and `test_sliced` is False. For a list-valued `data.test` it
    # holds every slice and `test_sliced` is True, which switches the runner to
    # per-slice `srsran_eval_<a>s_to_<b>s/` output folders. `test` stays the
    # first slice so existing consumers (result_slug, manifest) are unchanged.
    tests: tuple[TimeRange, ...] = ()
    test_sliced: bool = False

    @property
    def capture_slug(self) -> str:
        # Only the basename of the training capture_id forms the path coordinate.
        # Any subfolder prefix (e.g. "synthetic/") mirrors the data layout and is
        # dropped so results/models aren't nested one level deeper under it.
        return Path(self.train.capture_id).name

    @property
    def result_slug(self) -> str:
        # The result folder is keyed by BOTH the train and test captures so that
        # holding the train capture fixed while changing only the test capture
        # appends a new result instead of overwriting the previous one. For a
        # same-capture (in-capture holdout) run the test adds nothing, so the
        # slug stays the bare train capture name; for a cross-capture run it
        # becomes "<train>_to_<test>".
        test_slug = Path(self.test.capture_id).name
        if test_slug == self.capture_slug:
            return self.capture_slug
        return f"{self.capture_slug}_to_{test_slug}"

    @property
    def result_dir(self) -> Path:
        # Keyed by the train+test captures (see result_slug) so re-running the
        # same config against a different test capture appends a new result
        # rather than overwriting; reruns on the same captures still replace in
        # place (newest wins). The model tree stays keyed by train alone.
        return RESULTS_ROOT / self.experiment_id / self.variant_id / self.result_slug

    @property
    def train_hash(self) -> str:
        """Hash over only training-relevant config.

        A re-invoked config whose training-relevant fields are unchanged reuses
        the already-trained model and only re-runs evaluation. Training-relevant:
        data.device, data.train, data.validation, model.* (excluding the cosmetic
        model.id label), preprocessing.*, and pipelines. Eval-only fields
        (data.test, evaluation.*, retention, description/tags) are excluded.
        """
        data = self.resolved.get("data", {})
        model = {k: v for k, v in self.resolved.get("model", {}).items() if k != "id"}
        subset = {
            "device": data.get("device", "qualcomm"),
            "train": dict(self.train.__dict__),
            "validation": dict(self.validation.__dict__),
            "train_pool": data.get("train_pool"),
            "model": model,
            "preprocessing": self.resolved.get("preprocessing", {}),
            "pipelines": self.resolved.get("pipelines", {}),
            "feature_ablation": self.resolved.get("feature_ablation"),
        }
        canonical = yaml.safe_dump(subset, sort_keys=True).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @property
    def model_id(self) -> str:
        configured = self.resolved.get("model", {}).get("id")
        if configured:
            return str(configured)
        return f"{self.experiment_id}/{self.variant_id}/{self.capture_slug}"

    @property
    def model_dir(self) -> Path:
        # One model per config+capture identity, mirroring result_dir, so the
        # config, its model, and its result share the same
        # <experiment_id>/<variant_id>/<capture_id> coordinate. Re-running the
        # same config on the same capture overwrites in place (newest wins); a
        # different capture appends a new model. The content hash lives in the
        # model's manifest, not in the path.
        return MODELS_ROOT / self.experiment_id / self.variant_id / self.capture_slug

    @property
    def train_pool(self) -> list[str] | None:
        """Optional list of synthetic capture_ids to pool for training. ``None``
        for ordinary single-capture configs (the existing code path)."""
        pool = self.resolved.get("data", {}).get("train_pool")
        return [str(c) for c in pool] if pool else None

    @property
    def feature_ablation(self) -> dict[str, Any] | None:
        """Optional feature-family ablation block ({scheme_id, include_*}). ``None``
        for ordinary configs, which keep every feature family. Passed verbatim to
        the srsRAN training entry points, which normalize it (feature_ablation.py)."""
        ablation = self.resolved.get("feature_ablation")
        return dict(ablation) if isinstance(ablation, dict) else None

    @property
    def residual_learning(self) -> bool:
        """Whether the srsRAN XGBoost model learns the residual over the calibrated
        3GPP baseline (``target - rule_based_power_mw``, default) or fits absolute
        ``target_power_mw`` directly. ``model.other.residual_learning``; defaults to
        ``True`` so ordinary configs are unchanged."""
        other = self.resolved.get("model", {}).get("other", {})
        if isinstance(other, dict):
            return bool(other.get("residual_learning", True))
        return True


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {path}")
    return dict(payload)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge mappings; scalar values and lists replace their base values."""

    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_config(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigError(f"Configuration inheritance cycle: {cycle}")
    payload = _read_yaml(path)
    parent_name = payload.pop("extends", None)
    if parent_name is None:
        return payload
    if not isinstance(parent_name, str) or not parent_name.strip():
        raise ConfigError(f"'extends' must be a non-empty relative path in {path}")
    parent = Path(parent_name)
    if parent.is_absolute():
        raise ConfigError(f"'extends' must be relative to the child config: {path}")
    base = resolve_config(path.parent / parent, (*stack, path))
    return _deep_merge(base, payload)


def _require_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be a mapping")
    return value


def _require_string(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{context}.{key}' must be a non-empty string")
    return value


def _parse_range(data: Mapping[str, Any], key: str) -> TimeRange:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"'data.{key}' must be a mapping")
    capture_id = _require_string(value, "capture_id", f"data.{key}")
    try:
        start_s = float(value["start_s"])
        end_s = float(value["end_s"])
    except KeyError as exc:
        raise ConfigError(f"'data.{key}' requires start_s and end_s") from exc
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'data.{key}' bounds must be numeric") from exc
    if start_s < 0 or end_s <= start_s:
        raise ConfigError(
            f"'data.{key}' must satisfy 0 <= start_s < end_s; got "
            f"[{start_s}, {end_s})"
        )
    return TimeRange(capture_id=capture_id, start_s=start_s, end_s=end_s)


def _parse_range_value(value: Any, context: str) -> TimeRange:
    """Parse a single ``{capture_id, start_s, end_s}`` mapping into a TimeRange."""
    if not isinstance(value, dict):
        raise ConfigError(f"'{context}' must be a mapping")
    capture_id = _require_string(value, "capture_id", context)
    try:
        start_s = float(value["start_s"])
        end_s = float(value["end_s"])
    except KeyError as exc:
        raise ConfigError(f"'{context}' requires start_s and end_s") from exc
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'{context}' bounds must be numeric") from exc
    if start_s < 0 or end_s <= start_s:
        raise ConfigError(
            f"'{context}' must satisfy 0 <= start_s < end_s; got "
            f"[{start_s}, {end_s})"
        )
    return TimeRange(capture_id=capture_id, start_s=start_s, end_s=end_s)


def _parse_test_ranges(data: Mapping[str, Any]) -> tuple[tuple[TimeRange, ...], bool]:
    """Parse ``data.test`` as either a single mapping or a list of mappings.

    A single mapping is the legacy form and yields ``((range,), False)``. A list
    yields one TimeRange per entry and ``sliced=True``, which the runner uses to
    emit a separate ``srsran_eval_<a>s_to_<b>s/`` folder per slice. Every slice in
    a list must reference the same capture_id (the slices carve one capture's
    timeline), since the per-slice result folder is keyed by start/end alone.
    """
    value = data.get("test")
    if isinstance(value, list):
        if not value:
            raise ConfigError("'data.test' list must not be empty")
        ranges = tuple(
            _parse_range_value(item, f"data.test[{i}]")
            for i, item in enumerate(value)
        )
        capture_ids = {item.capture_id for item in ranges}
        if len(capture_ids) > 1:
            raise ConfigError(
                "all 'data.test' slices must use the same capture_id; got "
                f"{sorted(capture_ids)}"
            )
        return ranges, True
    return (_parse_range_value(value, "data.test"),), False


def capture_dir(capture_id: str, device: str = "qualcomm") -> Path:
    """Resolve a capture directory.

    Captures normally live at ``data/<device>/<capture_id>``. As a fallback we
    also accept ``data/<device>/archive/<capture_id>`` so configs keep resolving
    after captures are moved into an ``archive/`` cold-storage subfolder. The
    primary location is returned when neither exists, so error messages point at
    the canonical path.
    """
    primary = DATA_ROOT / device / capture_id
    if primary.is_dir():
        return primary
    archived = DATA_ROOT / device / "archive" / capture_id
    if archived.is_dir():
        return archived
    return primary


def _validate_capture(capture_id: str, device: str) -> None:
    path = capture_dir(capture_id, device)
    if not path.is_dir():
        raise ConfigError(f"Unknown {device} capture_id '{capture_id}': {path}")
    metadata = path / "metadata.json"
    if not metadata.is_file():
        raise ConfigError(f"Capture is missing metadata.json: {path}")


def _validate_link_config(
    resolved: Mapping[str, Any],
    ranges: tuple[TimeRange, ...],
    device: str,
) -> None:
    expected = resolved.get("link_config", {})
    if not expected:
        return
    if not isinstance(expected, dict):
        raise ConfigError("'link_config' must be a mapping")
    strict = bool(resolved.get("validation", {}).get("require_link_match", True))
    if not strict:
        return
    for item in ranges:
        metadata_path = capture_dir(item.capture_id, device) / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        observed = metadata.get("link_config")
        if observed is None:
            raise ConfigError(
                f"Capture {item.capture_id} has no link_config metadata; "
                "set validation.require_link_match: false only for legacy analysis"
            )
        for group, values in expected.items():
            if not isinstance(values, dict):
                continue
            observed_group = observed.get(group, {})
            for field, expected_value in values.items():
                if expected_value is None:
                    continue
                if observed_group.get(field) != expected_value:
                    raise ConfigError(
                        f"Capture {item.capture_id} link mismatch at "
                        f"{group}.{field}: expected {expected_value!r}, "
                        f"observed {observed_group.get(field)!r}"
                    )


def _overlap(left: TimeRange, right: TimeRange) -> bool:
    return (
        left.capture_id == right.capture_id
        and max(left.start_s, right.start_s) < min(left.end_s, right.end_s)
    )


def validate_config(path: Path, resolved: dict[str, Any]) -> ExperimentConfig:
    schema = _read_yaml(SCHEMA_PATH)
    allowed = {"schema_version", *schema.get("sections", {}).keys()}
    unknown = sorted(set(resolved) - allowed)
    if unknown:
        raise ConfigError(f"Unknown top-level fields: {', '.join(unknown)}")
    missing = sorted(set(schema.get("required", [])) - set(resolved))
    if missing:
        raise ConfigError(f"Missing required top-level fields: {', '.join(missing)}")
    expected_version = schema.get("schema_version", 1)
    if resolved.get("schema_version") != expected_version:
        raise ConfigError(f"'schema_version' must be {expected_version}")

    experiment = _require_mapping(resolved, "experiment")
    experiment_id = _require_string(experiment, "id", "experiment")
    variant_id = str(experiment.get("variant_id", "default"))
    if not variant_id:
        raise ConfigError("'experiment.variant_id' must not be empty")

    data = _require_mapping(resolved, "data")
    device = str(data.get("device", "qualcomm"))
    if device not in {"qualcomm", "pixel"}:
        raise ConfigError("'data.device' must be 'qualcomm' or 'pixel'")
    train = _parse_range(data, "train")
    validation = _parse_range(data, "validation")
    tests, test_sliced = _parse_test_ranges(data)
    test = tests[0]
    for item in (train, validation, *tests):
        _validate_capture(item.capture_id, device)

    # Optional pooled training (additive; absent for ordinary configs).
    train_pool = data.get("train_pool")
    if train_pool is not None:
        if not isinstance(train_pool, list) or not train_pool:
            raise ConfigError("data.train_pool must be a non-empty list of capture_ids")
        for capture_id in train_pool:
            _validate_capture(str(capture_id), device)

    # Optional feature-family ablation (additive; absent for ordinary configs).
    # normalize_scheme validates keys/types and fills defaults; we only need it
    # here for early rejection of malformed or unsupported combinations.
    feature_ablation = resolved.get("feature_ablation")
    if feature_ablation is not None:
        from scripts.power_estimation.srsRAN.feature_ablation import (
            is_full,
            normalize_scheme,
        )

        try:
            scheme = normalize_scheme(feature_ablation)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        # Pooled training builds features through a separate cached path
        # that does not yet thread the ablation mask; refuse rather than silently
        # train a full-feature model under an ablation variant_id.
        if not is_full(scheme) and train_pool is not None:
            raise ConfigError(
                "feature_ablation (non-baseline) is not supported with "
                "data.train_pool; ablations run on single-capture training"
            )

    # Training and validation must come from one capture. The srsRAN pipeline
    # builds a single feature matrix from the train capture and carves the
    # validation split out of that same timeline (by time window), so a
    # different validation capture_id would be silently ignored. Refuse to start
    # rather than train against a misleading config.
    if train.capture_id != validation.capture_id:
        raise ConfigError(
            "data.train and data.validation must use the same capture_id "
            f"(train={train.capture_id!r}, validation={validation.capture_id!r}); "
            "the validation split is drawn from the train capture's own timeline."
        )

    allow_overlap = bool(
        resolved.get("validation", {}).get("allow_split_overlap", False)
    )
    if not allow_overlap:
        checks = [
            ("train", train, "validation", validation),
        ]
        for i, slice_range in enumerate(tests):
            label = f"test[{i}]" if test_sliced else "test"
            checks.append(("train", train, label, slice_range))
            checks.append(("validation", validation, label, slice_range))
        for left_name, left, right_name, right in checks:
            if _overlap(left, right):
                raise ConfigError(
                    f"data.{left_name} overlaps data.{right_name} on "
                    f"capture {left.capture_id}"
                )

    pipelines = _require_mapping(resolved, "pipelines")
    if not pipelines.get("srsran", False):
        raise ConfigError("pipelines.srsran must be enabled")
    if pipelines.get("3gpp") is False:
        raise ConfigError(
            "pipelines.3gpp must be enabled for the currently supported residual model"
        )

    model = _require_mapping(resolved, "model")
    if model.get("name") != "xgboost":
        raise ConfigError("Only model.name: xgboost is currently supported")
    if model.get("source", "srsran") != "srsran":
        raise ConfigError("model.source must be 'srsran'")
    other = model.get("other", {})
    if other and not isinstance(other, dict):
        raise ConfigError("'model.other' must be a mapping")
    residual_learning = other.get("residual_learning", True) if isinstance(other, dict) else True
    if not isinstance(residual_learning, bool):
        raise ConfigError("'model.other.residual_learning' must be a boolean")

    retention = resolved.get("retention", {})
    if retention and not isinstance(retention, dict):
        raise ConfigError("'retention' must be a mapping")
    if isinstance(retention, dict):
        if retention.get("predictions", True) is not True:
            raise ConfigError("retention.predictions must currently be true")
        if retention.get("feature_matrices", "processed_data") != "processed_data":
            raise ConfigError(
                "retention.feature_matrices must be 'processed_data'"
            )

    _validate_link_config(resolved, (train, validation, *tests), device)
    canonical = yaml.safe_dump(resolved, sort_keys=True).encode("utf-8")
    config_hash = hashlib.sha256(canonical).hexdigest()
    return ExperimentConfig(
        path=path.resolve(),
        resolved=resolved,
        config_hash=config_hash,
        experiment_id=experiment_id,
        variant_id=variant_id,
        train=train,
        validation=validation,
        test=test,
        tests=tests,
        test_sliced=test_sliced,
    )


def load_experiment_config(path: Path) -> ExperimentConfig:
    resolved = resolve_config(path)
    return validate_config(Path(path), resolved)

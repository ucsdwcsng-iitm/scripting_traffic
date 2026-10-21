"""Feature-family ablation for the srsRAN XGBoost pipeline.

An ablation *scheme* switches whole feature families on or off so a config can
ask "how well does the residual model do without temporal aggregates / without
categoricals / with raw bases only?". Schemes are declared in the experiment YAML
under a top-level ``feature_ablation`` block and applied inside
``build_labeled_feature_matrix`` (the single feature chokepoint shared by every
srsRAN training path). When the block is absent the default scheme keeps every
family, so ordinary configs build exactly the feature matrix they always did.

Each scheme is addressed by ``scheme_id``, which a study uses as its
``experiment.variant_id`` so results/models land at
``<experiment_id>/<scheme_id>/<capture>`` with no new path machinery. The
normalized scheme is also folded into the model ``train_hash`` (see config.py) so
two schemes never share a cached model.
"""

from __future__ import annotations

from typing import Any, Mapping


# Each flag gates one family produced by build_labeled_feature_matrix:
#   include_temporal_bases      -> current-slot raw PHY values (the TEMPORAL_BASES
#                                  columns themselves, un-lagged)
#   include_temporal_aggregates -> derived lag__* and roll__* columns
#   include_categorical         -> one-hot expansions of CATEGORICAL_FEATURES
#   include_rule_based_context  -> the rule_based_* columns fed in as features
SCHEME_FLAGS = (
    "include_temporal_bases",
    "include_temporal_aggregates",
    "include_categorical",
    "include_rule_based_context",
)

# scheme_id of the all-on baseline; also the default variant_id for non-ablation
# configs, so a bare config and the baseline scheme address the same model tree.
FULL_FEATURES_SCHEME_ID = "full_features"


def normalize_scheme(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a complete scheme dict (every flag present) from a config block.

    Missing flags default to ``True`` (keep the family), so a block only needs to
    name the families it drops. ``scheme_id`` defaults to ``full_features`` and is
    carried through for provenance. ``raw`` may be ``None`` (no ablation), in
    which case the all-on baseline scheme is returned.
    """
    scheme: dict[str, Any] = {"scheme_id": FULL_FEATURES_SCHEME_ID}
    for flag in SCHEME_FLAGS:
        scheme[flag] = True
    if not raw:
        return scheme
    if not isinstance(raw, Mapping):
        raise ValueError("'feature_ablation' must be a mapping")
    unknown = sorted(set(raw) - ({"scheme_id"} | set(SCHEME_FLAGS)))
    if unknown:
        raise ValueError(f"unknown feature_ablation keys: {', '.join(unknown)}")
    if "scheme_id" in raw:
        scheme_id = raw["scheme_id"]
        if not isinstance(scheme_id, str) or not scheme_id.strip():
            raise ValueError("'feature_ablation.scheme_id' must be a non-empty string")
        scheme["scheme_id"] = scheme_id
    for flag in SCHEME_FLAGS:
        if flag in raw:
            scheme[flag] = bool(raw[flag])
    return scheme


def is_full(scheme: Mapping[str, Any]) -> bool:
    """True when every family is kept (the build is identical to the no-ablation
    path). Callers use this to skip masking entirely on the common case."""
    return all(bool(scheme.get(flag, True)) for flag in SCHEME_FLAGS)

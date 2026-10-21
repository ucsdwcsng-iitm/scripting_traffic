"""Build per-slot feature matrices from parsed srsRAN facts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .log_parser import ParsedLog, SearchSpaceConfig


BOOLEAN_DEFAULTS = [
    "pdcch_present",
    "pdsch_present",
    "pusch_present",
    "pucch_present",
    "srs_present",
    "csi_rs_present",
    "csi_report_present",
    "csi_report_valid",
    "phr_present",
    "pusch_contains_bsr",
    "pusch_contains_phr",
    "pdsch_ta_cmd_present",
    "sr_detected",
]

COUNT_DEFAULTS = [
    "pdcch_count",
    "pdsch_count",
    "pusch_count",
    "pucch_count",
    "srs_count",
    "csi_rs_nzp_count",
    "csi_rs_zp_count",
    "harq_ack_count",
    "harq_nack_count",
]


def build_feature_frame(
    parsed: ParsedLog,
    *,
    rnti: Optional[int] = None,
    slot_start: Optional[int] = None,
    slot_end: Optional[int] = None,
    max_slots: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Return a per-slot feature matrix and summary metadata."""

    selected_rnti = rnti or parsed.config.get("selected_rnti")
    if selected_rnti is None:
        raise RuntimeError("No UE RNTI found. Pass --rnti explicitly.")
    selected_rnti = int(selected_rnti)

    slots = _select_slot_axis(parsed, slot_start=slot_start, slot_end=slot_end, max_slots=max_slots)
    inactivity_windows = _build_timer_windows(parsed, selected_rnti, "ctx_drx_inactivity_slots", "pdcch_present")
    retx_dl_windows = _build_timer_windows(parsed, selected_rnti, "ctx_drx_retx_dl_slots", "pdsch_retx_count")
    retx_ul_windows = _build_timer_windows(parsed, selected_rnti, "ctx_drx_retx_ul_slots", "pusch_retx_count")
    drx_timer_semantics = (
        "sched_dci_assisted"
        if _has_sched_dci(parsed, selected_rnti)
        else "conservative_log_inferred"
    )

    rows: List[Dict[str, object]] = []
    for global_slot in slots:
        slot_rec = parsed.slots.get(global_slot)
        base = {
            "capture_id": parsed.capture_id,
            "rnti": f"0x{selected_rnti:04x}",
            "rnti_int": selected_rnti,
            "ue_id": parsed.ue_by_rnti.get(selected_rnti),
            "pci": parsed.config.get("ctx_pci"),
            "frame": slot_rec.frame if slot_rec else _frame_from_global(parsed, global_slot),
            "slot": slot_rec.slot if slot_rec else _slot_from_global(parsed, global_slot),
            "global_slot": global_slot,
            "slot_wall_time_iso": slot_rec.wall_time_iso if slot_rec else None,
            "slot_time_point_ns": slot_rec.slot_time_point_ns if slot_rec else None,
            "scs_khz": parsed.config.get("ctx_common_scs_khz", 30),
            "slot_duration_ms": _slot_duration_ms(parsed),
            "target_ue_selected": 1,
        }
        _add_wall_time_sod(base)

        raw = dict(parsed.rows.get((global_slot, 0), {}))
        raw.update(parsed.rows.get((global_slot, selected_rnti), {}))
        for count_name in COUNT_DEFAULTS:
            raw[count_name] = (
                parsed.row_counts.get((global_slot, 0), {}).get(count_name, 0)
                + parsed.row_counts.get((global_slot, selected_rnti), {}).get(count_name, raw.get(count_name, 0))
            )
        for name in BOOLEAN_DEFAULTS:
            raw.setdefault(name, 0)
        for name in COUNT_DEFAULTS:
            raw.setdefault(name, 0)
        base.update(raw)

        _add_tdd_features(base, parsed, global_slot)
        _add_bwp_features(base, parsed)
        _add_drx_features(
            base,
            parsed,
            global_slot,
            inactivity_windows=inactivity_windows,
            retx_dl_windows=retx_dl_windows,
            retx_ul_windows=retx_ul_windows,
            timer_semantics=drx_timer_semantics,
        )
        _add_pdcch_monitoring_features(base, parsed, global_slot)
        _add_combined_activity_features(base)
        _add_static_context(base, parsed)
        rows.append(base)

    df = pd.DataFrame(rows)
    summary = _build_summary(parsed, df, selected_rnti)
    return df, summary


def _select_slot_axis(
    parsed: ParsedLog,
    *,
    slot_start: Optional[int],
    slot_end: Optional[int],
    max_slots: Optional[int],
) -> range:
    if parsed.slots:
        start = min(parsed.slots) if slot_start is None else slot_start
        end = max(parsed.slots) + 1 if slot_end is None else slot_end
    elif parsed.first_global_slot is not None and parsed.last_global_slot is not None:
        start = parsed.first_global_slot if slot_start is None else slot_start
        end = parsed.last_global_slot + 1 if slot_end is None else slot_end
    else:
        raise RuntimeError("No slot-prefixed records found.")

    if max_slots is not None:
        end = min(end, start + max_slots)
    return range(start, end)


def _build_timer_windows(parsed: ParsedLog, rnti: int, timer_config_key: str, trigger_feature: str) -> set[int]:
    timer_len = int(parsed.config.get(timer_config_key) or 0)
    if timer_len <= 0:
        return set()
    active: set[int] = set()
    for global_slot, row_rnti in parsed.rows:
        if row_rnti != rnti:
            continue
        row = parsed.rows[(global_slot, row_rnti)]
        triggered = bool(row.get(trigger_feature))
        if trigger_feature.endswith("retx_count"):
            triggered = int(row.get(trigger_feature) or 0) > 0
        if not triggered:
            continue
        active.update(range(global_slot, global_slot + timer_len))
    return active


def _add_drx_features(
    row: Dict[str, object],
    parsed: ParsedLog,
    global_slot: int,
    *,
    inactivity_windows: set[int],
    retx_dl_windows: set[int],
    retx_ul_windows: set[int],
    timer_semantics: str,
) -> None:
    activation_slot = parsed.drx_activation_slot
    configured = int(
        bool(parsed.config.get("ctx_drx_configured"))
        and (activation_slot is None or global_slot >= activation_slot)
    )
    row["drx_configured"] = configured
    for src, dst in [
        ("ctx_drx_on_duration_slots", "drx_on_duration_slots"),
        ("ctx_drx_inactivity_slots", "drx_inactivity_slots"),
        ("ctx_drx_long_cycle_slots", "drx_long_cycle_slots"),
        ("ctx_drx_retx_dl_slots", "drx_retx_dl_slots"),
        ("ctx_drx_retx_ul_slots", "drx_retx_ul_slots"),
    ]:
        row[dst] = parsed.config.get(src)

    long_cycle = int(parsed.config.get("ctx_drx_long_cycle_slots") or 0)
    on_duration = int(parsed.config.get("ctx_drx_on_duration_slots") or 0)
    offset = int(parsed.config.get("ctx_drx_long_cycle_offset_slots") or 0)
    slot_offset = int(parsed.config.get("ctx_drx_slot_offset_slots") or 0)

    if configured and long_cycle > 0:
        phase = (global_slot - offset - slot_offset) % long_cycle
        in_on_duration = int(phase < on_duration)
        row["drx_cycle_phase_slots"] = phase
        row["drx_in_on_duration"] = in_on_duration
        row["drx_in_inactivity_timer"] = int(global_slot in inactivity_windows)
        row["drx_in_retx_dl"] = int(global_slot in retx_dl_windows)
        row["drx_in_retx_ul"] = int(global_slot in retx_ul_windows)
        row["drx_timer_semantics"] = timer_semantics
    else:
        row["drx_cycle_phase_slots"] = None
        row["drx_in_on_duration"] = 1
        row["drx_in_inactivity_timer"] = 0
        row["drx_in_retx_dl"] = 0
        row["drx_in_retx_ul"] = 0
        row["drx_timer_semantics"] = "not_configured"

    activity = int(_has_any_activity(row))
    row["drx_awake"] = int(
        bool(row["drx_in_on_duration"])
        or bool(row["drx_in_inactivity_timer"])
        or bool(row["drx_in_retx_dl"])
        or bool(row["drx_in_retx_ul"])
        or activity
    )
    row["drx_sleep_opportunity"] = int(configured and not row["drx_awake"] and not activity)
    row["sleep_depth_candidate"] = _sleep_depth(row, global_slot)


def _has_sched_dci(parsed: ParsedLog, rnti: int) -> bool:
    for _, row_rnti in parsed.rows:
        if row_rnti == rnti and "pdcch_dci_mcs" in parsed.rows[(_, row_rnti)]:
            return True
    return False


def _sleep_depth(row: Dict[str, object], global_slot: int) -> str:
    if row["drx_awake"]:
        return "awake"
    phase = row.get("drx_cycle_phase_slots")
    if phase is None:
        return "unknown"
    if int(phase) < 40:
        return "micro_sleep_candidate"
    if int(phase) < 120:
        return "light_sleep_candidate"
    return "deep_sleep_candidate"


def _add_pdcch_monitoring_features(row: Dict[str, object], parsed: ParsedLog, global_slot: int) -> None:
    active = _active_search_spaces(parsed.search_spaces, global_slot)
    candidate_total = sum(ss.candidate_count_total for ss in active)
    symbol_bitmap = _or_symbol_bitmaps([ss.monitoring_symbols_bitmap for ss in active])
    monitored_bw = max((ss.coreset_bw_prbs for ss in active), default=0)
    duration_symbols = max((ss.coreset_duration_symbols for ss in active), default=0)

    drx_active = bool(row.get("drx_in_on_duration", 1) or row.get("drx_in_inactivity_timer", 0))
    row["pdcch_monitoring_required"] = int(bool(active) and drx_active)
    row["pdcch_monitoring_symbols"] = symbol_bitmap
    row["pdcch_monitoring_symbol_count"] = symbol_bitmap.count("1")
    row["pdcch_candidate_count_total"] = candidate_total
    row["pdcch_monitored_bw_prbs"] = monitored_bw
    row["pdcch_monitoring_duration_symbols"] = duration_symbols
    row["pdcch_search_space_id_active"] = ",".join(str(ss.search_space_id) for ss in active)
    row["pdcch_monitor_only_no_dci"] = int(row["pdcch_monitoring_required"] and not row.get("pdcch_present"))


def _active_search_spaces(search_spaces: Sequence[SearchSpaceConfig], global_slot: int) -> List[SearchSpaceConfig]:
    active: List[SearchSpaceConfig] = []
    for ss in search_spaces:
        period = max(int(ss.period_slots or 1), 1)
        duration = max(int(ss.duration_slots or 1), 1)
        if ((global_slot - int(ss.offset_slots or 0)) % period) < duration:
            active.append(ss)
    return active


def _or_symbol_bitmaps(bitmaps: Sequence[str]) -> str:
    if not bitmaps:
        return ""
    width = max(len(b) for b in bitmaps)
    out = ["0"] * width
    for bitmap in bitmaps:
        for idx, value in enumerate(bitmap):
            if value == "1":
                out[idx] = "1"
    return "".join(out)


def _add_bwp_features(row: Dict[str, object], parsed: ParsedLog) -> None:
    scs_khz = int(parsed.config.get("ctx_common_scs_khz") or 30)
    fallback_prbs = _bandwidth_mhz_to_prbs(parsed.config.get("ctx_channel_bandwidth_mhz"), scs_khz)
    dl_prbs = int(parsed.config.get("ctx_dl_bwp_n_prb") or fallback_prbs or 0)
    ul_prbs = int(parsed.config.get("ctx_ul_bwp_n_prb") or fallback_prbs or 0)
    row["active_dl_bwp_start_prb"] = 0
    row["active_dl_bwp_n_prb"] = dl_prbs
    row["active_ul_bwp_start_prb"] = 0
    row["active_ul_bwp_n_prb"] = ul_prbs
    row["active_dl_bw_mhz"] = dl_prbs * 12 * scs_khz / 1000.0
    row["active_ul_bw_mhz"] = ul_prbs * 12 * scs_khz / 1000.0
    row["bwp_decode_source"] = "rrc_or_carrier_fallback" if fallback_prbs else "unknown"


def _bandwidth_mhz_to_prbs(bandwidth_mhz: object, scs_khz: int) -> int:
    if bandwidth_mhz is None:
        return 0
    # FR1 30 kHz common cases used in these captures. Fall back to arithmetic
    # lower bound if an unseen bandwidth appears.
    table = {
        (5, 30): 11,
        (10, 30): 24,
        (15, 30): 38,
        (20, 30): 51,
        (40, 30): 106,
        (100, 30): 273,
    }
    bw = int(float(bandwidth_mhz))
    return table.get((bw, scs_khz), int((bw * 1000) // (12 * scs_khz)))


def _add_tdd_features(row: Dict[str, object], parsed: ParsedLog, global_slot: int) -> None:
    period_ms = float(parsed.config.get("ctx_tdd_period_ms") or 0)
    slot_duration_ms = _slot_duration_ms(parsed)
    slots_per_period = int(round(period_ms / slot_duration_ms)) if period_ms else 0
    dl_slots = int(parsed.config.get("ctx_tdd_nof_dl_slots") or 0)
    dl_symbols = int(parsed.config.get("ctx_tdd_nof_dl_symbols") or 0)
    ul_slots = int(parsed.config.get("ctx_tdd_nof_ul_slots") or 0)
    ul_symbols = int(parsed.config.get("ctx_tdd_nof_ul_symbols") or 0)
    phase = global_slot % slots_per_period if slots_per_period else None
    row["tdd_phase_slot"] = phase
    row["tdd_mapping_source"] = "startup_config_tdd_ul_dl_cfg" if slots_per_period else "missing"

    row["is_dl_slot"] = 0
    row["is_ul_slot"] = 0
    row["is_special_slot"] = 0
    row["dl_symbols_available"] = 0
    row["ul_symbols_available"] = 0
    if phase is None:
        return
    if phase < dl_slots:
        row["is_dl_slot"] = 1
        row["dl_symbols_available"] = 14
    elif phase == dl_slots and dl_symbols > 0:
        row["is_dl_slot"] = 1
        row["is_special_slot"] = 1
        row["dl_symbols_available"] = dl_symbols
    elif phase >= slots_per_period - ul_slots:
        row["is_ul_slot"] = 1
        row["ul_symbols_available"] = 14
    elif phase == slots_per_period - ul_slots - 1 and ul_symbols > 0:
        row["is_ul_slot"] = 1
        row["is_special_slot"] = 1
        row["ul_symbols_available"] = ul_symbols
    else:
        row["is_special_slot"] = 1


def _add_combined_activity_features(row: Dict[str, object]) -> None:
    rx = bool(row.get("pdcch_present") or row.get("pdsch_present") or row.get("csi_rs_present"))
    tx = bool(row.get("pusch_present") or row.get("pucch_present") or row.get("srs_present"))
    row["has_rx_activity"] = int(rx)
    row["has_tx_activity"] = int(tx)
    row["has_data_rx"] = int(bool(row.get("pdsch_present")))
    row["has_data_tx"] = int(bool(row.get("pusch_present")))
    row["has_control_only"] = int(
        bool(row.get("pdcch_present") or row.get("pucch_present"))
        and not bool(row.get("pdsch_present") or row.get("pusch_present"))
    )
    row["has_reference_signal_rx"] = int(bool(row.get("csi_rs_present")))
    row["has_reference_signal_tx"] = int(bool(row.get("srs_present")))
    row["activity_class"] = _activity_class(row)


def _activity_class(row: Dict[str, object]) -> str:
    if row.get("pdsch_present") and row.get("pusch_present"):
        return "mixed"
    if row.get("pdsch_present"):
        return "dl_data"
    if row.get("pusch_present"):
        return "ul_data"
    if row.get("pucch_present") or row.get("srs_present"):
        return "ul_control"
    if row.get("pdcch_present"):
        return "dl_control"
    if row.get("pdcch_monitoring_required"):
        return "monitor_only"
    if row.get("csi_rs_present"):
        return "reference_rx"
    return "idle"


def _has_any_activity(row: Dict[str, object]) -> bool:
    return any(bool(row.get(name)) for name in BOOLEAN_DEFAULTS[:6])


def _add_static_context(row: Dict[str, object], parsed: ParsedLog) -> None:
    for key, value in parsed.config.items():
        if key.startswith("ctx_"):
            row[key] = value


def _slot_duration_ms(parsed: ParsedLog) -> float:
    scs = int(parsed.config.get("ctx_common_scs_khz") or 30)
    mu = {15: 0, 30: 1, 60: 2, 120: 3, 240: 4}.get(scs, 1)
    return 1.0 / (2 ** mu)


def _frame_from_global(parsed: ParsedLog, global_slot: int) -> int:
    slots_per_frame = int(parsed.config.get("slots_per_frame") or 20)
    return (global_slot // slots_per_frame) % 1024


def _slot_from_global(parsed: ParsedLog, global_slot: int) -> int:
    slots_per_frame = int(parsed.config.get("slots_per_frame") or 20)
    return global_slot % slots_per_frame


def _add_wall_time_sod(row: Dict[str, object]) -> None:
    ts = row.get("slot_wall_time_iso")
    if not ts:
        row["slot_start_time_abs_sod"] = None
        row["slot_end_time_abs_sod"] = None
        return
    time_part = str(ts).split("T", 1)[1]
    hour = int(time_part[0:2])
    minute = int(time_part[3:5])
    seconds = float(time_part[6:])
    sod = hour * 3600 + minute * 60 + seconds
    row["slot_start_time_abs_sod"] = sod
    row["slot_end_time_abs_sod"] = sod + float(row.get("slot_duration_ms") or 0) / 1000.0


def _build_summary(parsed: ParsedLog, df: pd.DataFrame, selected_rnti: int) -> Dict[str, object]:
    activity_cols = [col for col in BOOLEAN_DEFAULTS if col in df]
    counts = {col: int(df[col].fillna(0).sum()) for col in activity_cols}
    counts.update({col: int(df[col].fillna(0).sum()) for col in COUNT_DEFAULTS if col in df})
    return {
        "capture_id": parsed.capture_id,
        "log_path": str(parsed.log_path),
        "selected_rnti": f"0x{selected_rnti:04x}",
        "selected_rnti_int": selected_rnti,
        "row_count": int(len(df)),
        "global_slot_min": int(df["global_slot"].min()) if len(df) else None,
        "global_slot_max": int(df["global_slot"].max()) if len(df) else None,
        "sfn_wraps_detected": parsed.sfn_wraps,
        "sfn_extra_cycles_inferred_from_timestamps": parsed.sfn_extra_cycles_inferred,
        "global_slot_definition": "unwrapped_sfn_times_slots_per_frame_plus_slot",
        "slots_per_frame": int(parsed.config.get("slots_per_frame") or 20),
        "drx_config": {
            key: parsed.config.get(key)
            for key in sorted(parsed.config)
            if key.startswith("ctx_drx_")
        },
        "tdd_config": {
            key: parsed.config.get(key)
            for key in sorted(parsed.config)
            if key.startswith("ctx_tdd_")
        },
        "search_space_count": len(parsed.search_spaces),
        "search_spaces": [asdict(ss) for ss in parsed.search_spaces],
        "feature_activity_counts": counts,
        "warnings": parsed.warnings,
    }

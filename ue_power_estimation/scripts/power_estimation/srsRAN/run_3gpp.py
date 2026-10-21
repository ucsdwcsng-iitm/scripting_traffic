"""3GPP TS 38.840-style power model over srsRAN per-slot features."""

from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from scripts.power_estimation.capture_layout import CaptureLayout


_constants = importlib.import_module("scripts.power_estimation.tr38840.constants")
_models = importlib.import_module("scripts.power_estimation.tr38840.models")
_power_logic = importlib.import_module("scripts.power_estimation.tr38840.power_logic")

REL_MICRO_SLEEP = float(_constants.REL_MICRO_SLEEP)
REL_PDCCH_ONLY = float(_constants.REL_PDCCH_ONLY)
SlotFlags = _models.SlotFlags


def build_3gpp_power_frame(
    features: pd.DataFrame,
    *,
    microsleep_power_mw: float = 250.0,
    ul_power_mode: str = "0dbm",
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rows = []
    state_counts: Counter[str] = Counter()

    for record in features.to_dict(orient="records"):
        pdcch_present = bool(record.get("pdcch_present")) or bool(
            record.get("pdcch_monitoring_required")
        )
        pdcch_direction = str(record.get("pdcch_direction") or "")
        pucch_format = _optional_int(record.get("pucch_format"))
        flags = SlotFlags(
            has_pdcch_dl=pdcch_present and pdcch_direction != "ul_grant",
            has_pdcch_ul=pdcch_present and pdcch_direction == "ul_grant",
            has_pdsch=bool(record.get("pdsch_present")),
            has_pusch=bool(record.get("pusch_present")),
            has_srs=bool(record.get("srs_present")),
            has_prach=False,
            has_pucch_short=bool(record.get("pucch_present")) and pucch_format in (0, 2),
            has_pucch_long=bool(record.get("pucch_present")) and pucch_format not in (0, 2),
            has_k0_gt0=False,
            has_pdsch_inferred_from_k0_zero=False,
            pdsch_rx_2x2_seen=False,
            pdsch_rx_4x4_seen=False,
        )
        dl_bw_mhz = _optional_int(record.get("active_dl_bw_mhz"))
        dl_scale = float(_power_logic.dl_bwp_scaling_factor(dl_bw_mhz))
        state, relative = _power_logic.pick_state(
            flags,
            ul_power_mode,
            dl_scale,
        )

        if (
            state == "MICRO_SLEEP"
            and bool(record.get("csi_rs_present"))
            and not bool(record.get("drx_sleep_opportunity"))
        ):
            state = "CSI_RS_RX"
            relative = max(REL_MICRO_SLEEP, REL_PDCCH_ONLY * dl_scale)
        elif state == "MICRO_SLEEP" and bool(record.get("pdcch_monitoring_required")):
            state = "PDCCH_MONITOR_ONLY"
            relative = REL_PDCCH_ONLY

        absolute_mw = float(_power_logic.rel_to_abs(relative, microsleep_power_mw))
        state_counts[state] += 1
        rows.append(
            {
                "capture_id": record.get("capture_id"),
                "rnti": record.get("rnti"),
                "frame": record.get("frame"),
                "slot": record.get("slot"),
                "global_slot": record.get("global_slot"),
                "slot_wall_time_iso": record.get("slot_wall_time_iso"),
                "slot_time_point_ns": record.get("slot_time_point_ns"),
                "slot_start_time_abs_sod": record.get("slot_start_time_abs_sod"),
                "slot_end_time_abs_sod": record.get("slot_end_time_abs_sod"),
                "activities": _activity_tokens(record),
                "dominant_state": state,
                "power_relative": relative,
                "power_assumed_mw": absolute_mw,
                "dl_bwp_scaling_factor": dl_scale,
                "active_dl_bw_mhz": record.get("active_dl_bw_mhz"),
                "has_pdcch_dl": int(flags.has_pdcch_dl),
                "has_pdcch_ul": int(flags.has_pdcch_ul),
                "has_pdsch": int(flags.has_pdsch),
                "has_pusch": int(flags.has_pusch),
                "has_pucch_short": int(flags.has_pucch_short),
                "has_pucch_long": int(flags.has_pucch_long),
                "has_srs": int(flags.has_srs),
                "has_csi_rs": int(bool(record.get("csi_rs_present"))),
                "pdcch_monitoring_required": int(bool(record.get("pdcch_monitoring_required"))),
                "drx_awake": int(bool(record.get("drx_awake"))),
                "drx_sleep_opportunity": int(bool(record.get("drx_sleep_opportunity"))),
            }
        )

    result = pd.DataFrame(rows)
    summary = {
        "model": "3GPP TS 38.840 section 8.1 adapted to canonical srsRAN PHY features",
        "row_count": int(len(result)),
        "microsleep_power_mw": float(microsleep_power_mw),
        "ul_power_mode": ul_power_mode,
        "state_counts": dict(sorted(state_counts.items())),
        "mean_power_mw": float(result["power_assumed_mw"].mean()),
        "min_power_mw": float(result["power_assumed_mw"].min()),
        "max_power_mw": float(result["power_assumed_mw"].max()),
        "assumptions": {
            "pdcch_monitoring_without_dci": "PDCCH_ONLY",
            "csi_rs_only_awake_receive": "PDCCH_ONLY scaled by active DL BWP",
            "unknown_ul_tx_power": ul_power_mode,
            "pdsch_rx_antennas": "1Rx/no antenna reduction unless explicitly logged",
            "mixed_undefined_states": "maximum component relative power",
        },
    }
    return result, summary


def run(
    *,
    features_csv: Path,
    output_dir: Path,
    microsleep_power_mw: float = 250.0,
    ul_power_mode: str = "0dbm",
) -> Dict[str, object]:
    header = pd.read_csv(features_csv, nrows=0).columns
    required = {
        "capture_id",
        "rnti",
        "frame",
        "slot",
        "global_slot",
        "slot_wall_time_iso",
        "slot_time_point_ns",
        "slot_start_time_abs_sod",
        "slot_end_time_abs_sod",
        "pdcch_present",
        "pdcch_monitoring_required",
        "pdcch_direction",
        "pdsch_present",
        "pusch_present",
        "pucch_present",
        "pucch_format",
        "srs_present",
        "csi_rs_present",
        "drx_awake",
        "drx_sleep_opportunity",
        "active_dl_bw_mhz",
    }
    features = pd.read_csv(
        features_csv,
        usecols=[name for name in header if name in required],
        low_memory=False,
    )
    frame, summary = build_3gpp_power_frame(
        features,
        microsleep_power_mw=microsleep_power_mw,
        ul_power_mode=ul_power_mode,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "power_per_slot.csv"
    summary_json = output_dir / "power_summary.json"
    frame.to_csv(output_csv, index=False)
    summary.update(
        {
            "features_csv": str(features_csv),
            "output_csv": str(output_csv),
            "summary_json": str(summary_json),
        }
    )
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _activity_tokens(record: Dict[str, object]) -> str:
    names = []
    for flag, label in (
        ("pdcch_present", "PDCCH"),
        ("pdcch_monitoring_required", "PDCCH_MONITOR"),
        ("pdsch_present", "PDSCH"),
        ("pucch_present", "PUCCH"),
        ("pusch_present", "PUSCH"),
        ("srs_present", "SRS"),
        ("csi_rs_present", "CSI_RS"),
    ):
        if bool(record.get(flag)):
            names.append(label)
    return "+".join(names) if names else "MICRO_SLEEP"


def _optional_int(value: object) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 3GPP power model on srsRAN slot features.")
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--microsleep-power-mw", type=float, default=250.0)
    parser.add_argument("--ul-power-mode", choices=("0dbm", "23dbm"), default="0dbm")
    args = parser.parse_args()
    layout = CaptureLayout(args.capture_dir.resolve())
    summary = run(
        features_csv=args.features_csv or layout.srsran_parsed_dir / "slot_features.csv",
        output_dir=args.output_dir or layout.srsran_3gpp_dir,
        microsleep_power_mw=args.microsleep_power_mw,
        ul_power_mode=args.ul_power_mode,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

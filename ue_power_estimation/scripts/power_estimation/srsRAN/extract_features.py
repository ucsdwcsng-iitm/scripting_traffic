"""CLI for srsRAN per-slot feature extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from scripts.power_estimation.capture_layout import CaptureLayout, capture_root_from_source

from .feature_builder import BOOLEAN_DEFAULTS, COUNT_DEFAULTS, build_feature_frame
from .log_parser import parse_srsran_log


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract per-slot XGBoost features from an srsRAN gNB log."
    )
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Path to gnb_n41.log or another srsRAN gNB log.",
    )
    parser.add_argument(
        "--rnti",
        type=str,
        default=None,
        help="Target C-RNTI, for example 0x4601. Defaults to the most active UE RNTI.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Per-slot feature CSV. Defaults to <capture>/srsRAN/parsed/slot_features.csv.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Summary JSON. Defaults to <capture>/srsRAN/parsed/slot_feature_summary.json.",
    )
    parser.add_argument(
        "--sparse-rows-jsonl",
        type=Path,
        default=None,
        help="Sparse parsed slot/RNTI rows. Defaults to <capture>/srsRAN/parsed/parsed_rows.jsonl.",
    )
    parser.add_argument(
        "--slot-start",
        type=int,
        default=None,
        help="Optional first global_slot to emit.",
    )
    parser.add_argument(
        "--slot-end",
        type=int,
        default=None,
        help="Optional exclusive last global_slot to emit.",
    )
    parser.add_argument(
        "--max-slots",
        type=int,
        default=None,
        help="Optional maximum number of slots to emit from the selected start.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Developer/testing option to parse only the first N log lines.",
    )
    parser.add_argument(
        "--chunk-slots",
        type=int,
        default=10000,
        help="Number of slot rows built and written at once. Default: 10000.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print summary after writing outputs.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    layout = CaptureLayout(capture_root_from_source(args.log.parent))
    output_csv = args.output_csv or layout.srsran_parsed_dir / "slot_features.csv"
    summary_json = args.summary_json or layout.srsran_parsed_dir / "slot_feature_summary.json"
    sparse_rows_jsonl = args.sparse_rows_jsonl or layout.srsran_sparse_rows
    summary = extract_to_paths(
        log_path=args.log,
        output_csv=output_csv,
        summary_json=summary_json,
        sparse_rows_jsonl=sparse_rows_jsonl,
        rnti=args.rnti,
        slot_start=args.slot_start,
        slot_end=args.slot_end,
        max_slots=args.max_slots,
        max_lines=args.max_lines,
        chunk_slots=args.chunk_slots,
        verbose=args.verbose,
    )
    if args.verbose:
        print(json.dumps(summary, indent=2, sort_keys=True)[:4000])


def extract_to_paths(
    *,
    log_path: Path,
    output_csv: Path,
    summary_json: Path,
    sparse_rows_jsonl: Path,
    rnti: str | None = None,
    slot_start: int | None = None,
    slot_end: int | None = None,
    max_slots: int | None = None,
    max_lines: int | None = None,
    chunk_slots: int = 10000,
    verbose: bool = False,
) -> Dict[str, object]:
    """Extract one gNB log into explicitly selected processed-data paths."""

    parsed = parse_srsran_log(log_path, max_lines=max_lines)
    parsed_rnti = _parse_rnti(rnti) if rnti else None
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    sparse_rows_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _write_sparse_rows(parsed, sparse_rows_jsonl)

    start, end = _output_bounds(parsed, slot_start, slot_end, max_slots)
    selected_rnti = parsed_rnti or int(parsed.config["selected_rnti"])
    columns = _feature_columns(parsed, selected_rnti, start)
    summary: Optional[Dict[str, object]] = None
    row_count = 0

    for chunk_start in range(start, end, chunk_slots):
        chunk_end = min(end, chunk_start + chunk_slots)
        df, chunk_summary = build_feature_frame(
            parsed,
            rnti=selected_rnti,
            slot_start=chunk_start,
            slot_end=chunk_end,
        )
        df = df.reindex(columns=columns)
        df.to_csv(
            output_csv,
            index=False,
            mode="w" if row_count == 0 else "a",
            header=row_count == 0,
        )
        summary = _merge_summary(summary, chunk_summary)
        row_count += len(df)
        if verbose:
            print(f"Wrote slots [{chunk_start}, {chunk_end}) ({row_count}/{end - start})", flush=True)

    if summary is None:
        raise RuntimeError("No slot rows selected.")
    with summary_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    if verbose:
        print(f"Wrote {row_count} slot rows to {output_csv}")
        print(f"Wrote sparse parsed rows to {sparse_rows_jsonl}")
        print(f"Wrote summary to {summary_json}")
    return summary


def _parse_rnti(value: str) -> int:
    value = value.strip()
    return int(value, 16) if value.lower().startswith("0x") else int(value)


def _write_sparse_rows(parsed, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as fh:
        for (global_slot, rnti), values in sorted(parsed.rows.items()):
            slot = parsed.slots.get(global_slot)
            line_numbers = sorted(parsed.source_line_numbers.get((global_slot, rnti), set()))
            payload = {
                "global_slot": global_slot,
                "frame": slot.frame if slot else (global_slot // int(parsed.config["slots_per_frame"])) % 1024,
                "slot": slot.slot if slot else global_slot % int(parsed.config["slots_per_frame"]),
                "rnti": f"0x{rnti:04x}" if rnti else None,
                "gnb_log_line_numbers": line_numbers,
                "gnb_log_first_line": line_numbers[0] if line_numbers else None,
                "gnb_log_last_line": line_numbers[-1] if line_numbers else None,
                **values,
            }
            fh.write(json.dumps(payload, sort_keys=True))
            fh.write("\n")


def _output_bounds(
    parsed,
    slot_start: Optional[int],
    slot_end: Optional[int],
    max_slots: Optional[int],
) -> tuple[int, int]:
    start = min(parsed.slots) if slot_start is None else slot_start
    end = max(parsed.slots) + 1 if slot_end is None else slot_end
    if max_slots is not None:
        end = min(end, start + max_slots)
    return start, end


def _feature_columns(parsed, rnti: int, start: int) -> List[str]:
    preview, _ = build_feature_frame(parsed, rnti=rnti, slot_start=start, slot_end=start + 1)
    columns = list(preview.columns)
    seen = set(columns)
    for raw in parsed.rows.values():
        for key in raw:
            if key not in seen:
                columns.append(key)
                seen.add(key)
    for key in BOOLEAN_DEFAULTS + COUNT_DEFAULTS:
        if key not in seen:
            columns.append(key)
            seen.add(key)
    return columns


def _merge_summary(
    aggregate: Optional[Dict[str, object]],
    chunk: Dict[str, object],
) -> Dict[str, object]:
    if aggregate is None:
        return chunk
    aggregate["row_count"] = int(aggregate["row_count"]) + int(chunk["row_count"])
    aggregate["global_slot_min"] = min(int(aggregate["global_slot_min"]), int(chunk["global_slot_min"]))
    aggregate["global_slot_max"] = max(int(aggregate["global_slot_max"]), int(chunk["global_slot_max"]))
    aggregate_counts = aggregate["feature_activity_counts"]
    chunk_counts = chunk["feature_activity_counts"]
    for key, value in chunk_counts.items():
        aggregate_counts[key] = int(aggregate_counts.get(key, 0)) + int(value)
    return aggregate


if __name__ == "__main__":
    main()

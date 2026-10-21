"""Streaming parser for srsRAN gNB logs used by power estimation."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple


LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\S+)\s+"
    r"\[(?P<layer>[^\]]+)\]\s+\[(?P<level>[^\]]+)\]"
    r"(?:\s+\[\s*(?P<frame>\d+)\.(?P<slot>\d+)\])?\s*"
    r"(?P<msg>.*)$"
)
SLOT_IN_TEXT_RE = re.compile(r"\b(?:slot|rx_slot|slot_rx)=(?P<frame>\d+)\.(?P<slot>\d+)")
HEX_RNTI_RE = re.compile(r"\b(?:c-rnti|rnti)=0x([0-9a-fA-F]+)")


@dataclass
class SlotRecord:
    frame: int
    slot: int
    global_slot: int
    wall_time_iso: Optional[str] = None
    slot_time_point_ns: Optional[int] = None


@dataclass
class SearchSpaceConfig:
    search_space_id: int
    coreset_id: int = 0
    period_slots: int = 1
    offset_slots: int = 0
    duration_slots: int = 1
    monitoring_symbols_bitmap: str = ""
    candidate_count_total: int = 0
    coreset_bw_prbs: int = 0
    coreset_duration_symbols: int = 0


@dataclass
class ParsedLog:
    log_path: Path
    capture_id: str
    config: Dict[str, object] = field(default_factory=dict)
    slots: Dict[int, SlotRecord] = field(default_factory=dict)
    rows: DefaultDict[Tuple[int, int], Dict[str, object]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    row_counts: DefaultDict[Tuple[int, int], Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    source_line_numbers: DefaultDict[Tuple[int, int], Set[int]] = field(
        default_factory=lambda: defaultdict(set)
    )
    current_record_line: Optional[int] = None
    rnti_counts: Counter = field(default_factory=Counter)
    ue_by_rnti: Dict[int, int] = field(default_factory=dict)
    search_spaces: List[SearchSpaceConfig] = field(default_factory=list)
    drx_activation_slot: Optional[int] = None
    first_global_slot: Optional[int] = None
    last_global_slot: Optional[int] = None
    first_wall_time_iso: Optional[str] = None
    last_wall_time_iso: Optional[str] = None
    sfn_wraps: int = 0
    sfn_extra_cycles_inferred: int = 0
    warnings: List[str] = field(default_factory=list)


class SlotIndexer:
    """Map logged SFN/slot pairs to monotonic NR radio slots with SFN wrap."""

    def __init__(self, slots_per_frame: int) -> None:
        self.slots_per_frame = slots_per_frame
        self.cycle_slots = 1024 * slots_per_frame
        self.slot_duration_s = 0.01 / slots_per_frame
        self._last_raw: Optional[int] = None
        self._last_global: Optional[int] = None
        self._last_timestamp_s: Optional[float] = None
        self._cycle = 0
        self.wraps = 0
        self.extra_cycles_inferred = 0

    def observe_indication(
        self,
        frame: int,
        slot: int,
        timestamp_s: Optional[float] = None,
    ) -> int:
        """Advance radio time using the canonical FAPI Slot.indication stream."""
        raw = frame * self.slots_per_frame + slot
        if self._last_raw is None or self._last_global is None:
            self._last_raw = raw
            self._last_global = raw
            self._last_timestamp_s = timestamp_s
            return raw

        minimum_forward_delta = (raw - self._last_raw) % self.cycle_slots
        extra_cycles = 0
        if (
            timestamp_s is not None
            and self._last_timestamp_s is not None
            and timestamp_s > self._last_timestamp_s
        ):
            elapsed_slots = (timestamp_s - self._last_timestamp_s) / self.slot_duration_s
            extra_cycles = max(
                0,
                round((elapsed_slots - minimum_forward_delta) / self.cycle_slots),
            )

        previous_cycle = self._last_global // self.cycle_slots
        delta = minimum_forward_delta + extra_cycles * self.cycle_slots
        self._last_global += delta
        self._cycle = self._last_global // self.cycle_slots
        self.wraps += self._cycle - previous_cycle
        self.extra_cycles_inferred += extra_cycles
        self._last_raw = raw
        self._last_timestamp_s = timestamp_s
        return self._last_global

    def resolve_near_current(self, frame: int, slot: int) -> int:
        """Resolve delayed PHY/SCHED/FAPI records without advancing wrap state."""
        raw = frame * self.slots_per_frame + slot
        if self._last_global is None:
            return self._cycle * self.cycle_slots + raw
        candidates = [
            cycle * self.cycle_slots + raw
            for cycle in range(max(0, self._cycle - 1), self._cycle + 2)
        ]
        return min(candidates, key=lambda value: abs(value - self._last_global))


def parse_srsran_log(
    log_path: Path,
    *,
    max_lines: Optional[int] = None,
) -> ParsedLog:
    """Parse one srsRAN gNB log into slot-indexed raw facts."""

    log_path = Path(log_path)
    parsed = ParsedLog(log_path=log_path, capture_id=log_path.parent.name)
    indexer: Optional[SlotIndexer] = None
    current: List[str] = []
    current_start_line: Optional[int] = None
    radio_time_started = False
    pending_drx_activation = False

    def flush() -> None:
        nonlocal indexer, radio_time_started, pending_drx_activation
        if not current:
            return
        record = "\n".join(current)
        first = current[0]
        m = LINE_RE.match(first)
        if not m:
            return

        if not parsed.config:
            _parse_startup_config(record, parsed.config)
        else:
            _parse_startup_config(record, parsed.config)

        scs_khz = int(parsed.config.get("ctx_common_scs_khz") or 30)
        slots_per_frame = int(10 * (2 ** _scs_to_mu(scs_khz)))
        if indexer is None:
            indexer = SlotIndexer(slots_per_frame)
            parsed.config["slots_per_frame"] = slots_per_frame
            parsed.config["sfn_cycle_slots"] = 1024 * slots_per_frame

        frame, slot = _record_slot(m, first)
        global_slot: Optional[int] = None
        is_slot_indication = "Slot.indication" in record
        if frame is not None and slot is not None and (radio_time_started or is_slot_indication):
            if is_slot_indication:
                radio_time_started = True
                global_slot = indexer.observe_indication(
                    frame,
                    slot,
                    _slot_observation_timestamp_s(record, m.group("ts")),
                )
                if pending_drx_activation:
                    parsed.drx_activation_slot = global_slot
                    pending_drx_activation = False
            else:
                global_slot = indexer.resolve_near_current(frame, slot)
            parsed.sfn_wraps = indexer.wraps
            parsed.sfn_extra_cycles_inferred = indexer.extra_cycles_inferred
            if is_slot_indication:
                _remember_slot(parsed, frame, slot, global_slot, m.group("ts"), record)

        _parse_rrc_config(record, parsed, global_slot)
        if "drx-Config" in record and parsed.config.get("ctx_drx_configured"):
            if global_slot is None:
                pending_drx_activation = True
            else:
                parsed.drx_activation_slot = global_slot
        if global_slot is not None:
            parsed.current_record_line = current_start_line
            _parse_activity_record(record, m.group("layer").strip(), global_slot, parsed)
            parsed.current_record_line = None

    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if LINE_RE.match(line):
                flush()
                current = [line]
                current_start_line = line_no
            else:
                current.append(line)
            if max_lines is not None and line_no >= max_lines:
                break
    flush()

    if parsed.rnti_counts and "selected_rnti" not in parsed.config:
        parsed.config["selected_rnti"] = _select_default_rnti(parsed.rnti_counts)
    return parsed


def _scs_to_mu(scs_khz: int) -> int:
    mapping = {15: 0, 30: 1, 60: 2, 120: 3, 240: 4}
    return mapping.get(int(scs_khz), 1)


def _record_slot(match: re.Match[str], first_line: str) -> Tuple[Optional[int], Optional[int]]:
    if match.group("frame") is not None:
        return int(match.group("frame")), int(match.group("slot"))
    return None, None


def _slot_observation_timestamp_s(record: str, wall_time_iso: str) -> float:
    time_point = re.search(r"\btime_point=(\d+)", record)
    if time_point:
        return int(time_point.group(1)) / 1e9
    return datetime.fromisoformat(wall_time_iso.replace("Z", "+00:00")).timestamp()


def _remember_slot(
    parsed: ParsedLog,
    frame: int,
    slot: int,
    global_slot: int,
    wall_time_iso: str,
    record: str,
) -> None:
    rec = parsed.slots.get(global_slot)
    if rec is None:
        rec = SlotRecord(frame=frame, slot=slot, global_slot=global_slot, wall_time_iso=wall_time_iso)
        parsed.slots[global_slot] = rec
    elif rec.wall_time_iso is None:
        rec.wall_time_iso = wall_time_iso

    tp_m = re.search(r"time_point=(\d+)", record)
    if tp_m:
        rec.slot_time_point_ns = int(tp_m.group(1))

    parsed.first_global_slot = global_slot if parsed.first_global_slot is None else min(parsed.first_global_slot, global_slot)
    parsed.last_global_slot = global_slot if parsed.last_global_slot is None else max(parsed.last_global_slot, global_slot)
    parsed.first_wall_time_iso = wall_time_iso if parsed.first_wall_time_iso is None else parsed.first_wall_time_iso
    parsed.last_wall_time_iso = wall_time_iso


def _parse_startup_config(record: str, config: Dict[str, object]) -> None:
    if "Input configuration" not in record and not any(k in record for k in ("common_scs:", "tdd_ul_dl_cfg:")):
        return
    scalar_map = {
        "ctx_band": r"\bband:\s*(\d+)",
        "ctx_dl_arfcn": r"\bdl_arfcn:\s*(\d+)",
        "ctx_channel_bandwidth_mhz": r"\bchannel_bandwidth_MHz:\s*([0-9.]+)",
        "ctx_common_scs_khz": r"\bcommon_scs:\s*(\d+)",
        "ctx_pci": r"\bpci:\s*(\d+)",
        "ctx_nof_antennas_ul": r"\bnof_antennas_ul:\s*(\d+)",
        "ctx_nof_antennas_dl": r"\bnof_antennas_dl:\s*(\d+)",
        "ctx_ssb_period_ms": r"\bssb_period:\s*([0-9.]+)",
        "ctx_ssb_block_power_dbm": r"\bssb_block_power_dbm:\s*(-?[0-9.]+)",
        "ctx_pdsch_nof_harqs": r"\bnof_harqs:\s*(\d+)",
        "ctx_pusch_target_sinr": r"\btarget_sinr:\s*(-?[0-9.]+)",
        "ctx_pusch_ref_path_loss": r"\bref_path_loss:\s*(-?[0-9.]+)",
        "ctx_pucch_p0_nominal": r"\bp0_nominal:\s*(-?[0-9.]+)",
        "ctx_sr_period_ms": r"\bsr_period_ms:\s*([0-9.]+)",
    }
    for key, pattern in scalar_map.items():
        if key not in config:
            m = re.search(pattern, record)
            if m:
                config[key] = _num(m.group(1))

    tdd_block = _block_after(record, "tdd_ul_dl_cfg:")
    if tdd_block:
        tdd_fields = {
            "ctx_tdd_period_ms": r"dl_ul_tx_period:\s*([0-9.]+)",
            "ctx_tdd_nof_dl_slots": r"nof_dl_slots:\s*(\d+)",
            "ctx_tdd_nof_dl_symbols": r"nof_dl_symbols:\s*(\d+)",
            "ctx_tdd_nof_ul_slots": r"nof_ul_slots:\s*(\d+)",
            "ctx_tdd_nof_ul_symbols": r"nof_ul_symbols:\s*(\d+)",
        }
        for key, pattern in tdd_fields.items():
            m = re.search(pattern, tdd_block)
            if m:
                config[key] = _num(m.group(1))

    pdcch_block = _block_after(record, "pdcch:")
    if pdcch_block and "ctx_common_search_space_candidates" not in config:
        m = re.search(r"ss1_n_candidates:\s*\[([^\]]+)\]", pdcch_block)
        if m:
            config["ctx_common_search_space_candidates"] = _sum_int_list(m.group(1))
        m = re.search(r"ss2_n_candidates:\s*\[([^\]]+)\]", pdcch_block)
        if m:
            config["ctx_dedicated_search_space_candidates"] = _sum_int_list(m.group(1))

    for key, pattern in {
        "ctx_pdsch_mcs_table": r"\bpdsch:\n.*?\bmcs_table:\s*([A-Za-z0-9_]+)",
        "ctx_pusch_mcs_table": r"\bpusch:\n.*?\bmcs_table:\s*([A-Za-z0-9_]+)",
        "ctx_pusch_max_rank": r"\bmax_rank:\s*(\d+)",
        "ctx_pusch_p0_nominal_with_grant": r"\bp0_nominal_with_grant:\s*(-?\d+)",
    }.items():
        if key not in config:
            m = re.search(pattern, record, re.S)
            if m:
                config[key] = _num(m.group(1)) if re.fullmatch(r"-?\d+(\.\d+)?", m.group(1)) else m.group(1)


def _parse_rrc_config(record: str, parsed: ParsedLog, global_slot: Optional[int]) -> None:
    if "drx-Config" in record:
        drx = _parse_drx(record, int(parsed.config.get("ctx_common_scs_khz") or 30))
        if drx:
            parsed.config.update(drx)
            parsed.config["ctx_drx_configured"] = 1
            parsed.drx_activation_slot = global_slot
    if "pdcch-Config" in record or "pdcch-ConfigCommon" in record:
        parsed.search_spaces.extend(_parse_search_spaces(record))
    if "initialDownlinkBWP" in record or "initialUplinkBWP" in record:
        _parse_bwp(record, parsed.config)


def _parse_drx(record: str, scs_khz: int) -> Dict[str, object]:
    slots_per_ms = 1.0 / (1.0 / (2 ** _scs_to_mu(scs_khz)))
    out: Dict[str, object] = {}

    m = re.search(r'"drx-onDurationTimer"\s*:\s*\{[^}]*"milliSeconds"\s*:\s*"([^"]+)"', record, re.S)
    if m:
        out["ctx_drx_on_duration_slots"] = _timer_to_slots(m.group(1), slots_per_ms)
    m = re.search(r'"drx-InactivityTimer"\s*:\s*"([^"]+)"', record)
    if m:
        out["ctx_drx_inactivity_slots"] = _timer_to_slots(m.group(1), slots_per_ms)
    for src, dst in [
        ("drx-HARQ-RTT-TimerDL", "ctx_drx_harq_rtt_dl_slots"),
        ("drx-HARQ-RTT-TimerUL", "ctx_drx_harq_rtt_ul_slots"),
    ]:
        m = re.search(rf'"{src}"\s*:\s*(\d+)', record)
        if m:
            out[dst] = int(m.group(1))
    for src, dst in [
        ("drx-RetransmissionTimerDL", "ctx_drx_retx_dl_slots"),
        ("drx-RetransmissionTimerUL", "ctx_drx_retx_ul_slots"),
    ]:
        m = re.search(rf'"{src}"\s*:\s*"([^"]+)"', record)
        if m:
            out[dst] = _timer_to_slots(m.group(1), slots_per_ms)
    m = re.search(r'"drx-LongCycleStartOffset"\s*:\s*\{\s*"([^"]+)"\s*:\s*(\d+)', record, re.S)
    if m:
        out["ctx_drx_long_cycle_slots"] = _timer_to_slots(m.group(1), slots_per_ms)
        out["ctx_drx_long_cycle_offset_slots"] = int(m.group(2))
    m = re.search(r'"drx-SlotOffset"\s*:\s*(\d+)', record)
    if m:
        out["ctx_drx_slot_offset_slots"] = int(m.group(1))
    return out


def _timer_to_slots(value: str, slots_per_ms: float) -> int:
    if value.startswith("ms"):
        return int(round(float(value[2:]) * slots_per_ms))
    if value.startswith("sl"):
        return int(value[2:])
    if value.startswith("sf"):
        return int(round(float(value[2:]) * slots_per_ms))
    return int(value)


def _parse_search_spaces(record: str) -> List[SearchSpaceConfig]:
    coreset_bw: Dict[int, int] = {}
    coreset_duration: Dict[int, int] = {}
    for block in re.finditer(r'"controlResourceSetId"\s*:\s*(\d+)(?P<body>.*?)(?="controlResourceSetId"|\]\s*,?\s*"searchSpacesToAddModList"|$)', record, re.S):
        cid = int(block.group(1))
        body = block.group("body")
        fd = re.search(r'"frequencyDomainResources"\s*:\s*"([01]+)"', body)
        dur = re.search(r'"duration"\s*:\s*(\d+)', body)
        if fd:
            coreset_bw[cid] = fd.group(1).count("1") * 6
        if dur:
            coreset_duration[cid] = int(dur.group(1))

    spaces: List[SearchSpaceConfig] = []
    for block in re.finditer(r'"searchSpaceId"\s*:\s*(\d+)(?P<body>.*?)(?="searchSpaceId"|\]\s*,?\s*"searchSpace"|$)', record, re.S):
        ss_id = int(block.group(1))
        body = block.group("body")
        coreset_id = _first_int(body, r'"controlResourceSetId"\s*:\s*(\d+)') or 0
        period, offset = _parse_monitoring_period(body)
        duration = _first_int(body, r'"duration"\s*:\s*(\d+)') or 1
        symbols = _first_str(body, r'"monitoringSymbolsWithinSlot"\s*:\s*"([^"]+)"') or ""
        candidates = _sum_candidate_counts(body)
        spaces.append(
            SearchSpaceConfig(
                search_space_id=ss_id,
                coreset_id=coreset_id,
                period_slots=period,
                offset_slots=offset,
                duration_slots=duration,
                monitoring_symbols_bitmap=symbols,
                candidate_count_total=candidates,
                coreset_bw_prbs=coreset_bw.get(coreset_id, 0),
                coreset_duration_symbols=coreset_duration.get(coreset_id, 0),
            )
        )
    return spaces


def _parse_monitoring_period(body: str) -> Tuple[int, int]:
    m = re.search(r'"monitoringSlotPeriodicityAndOffset"\s*:\s*\{\s*"([^"]+)"\s*:\s*(\d+)', body, re.S)
    if not m:
        return 1, 0
    key, offset = m.group(1), int(m.group(2))
    key_m = re.search(r"sl(\d+)", key)
    if key_m:
        return int(key_m.group(1)), offset
    return 1, offset


def _sum_candidate_counts(body: str) -> int:
    candidates = 0
    for value in re.findall(r'"aggregationLevel\d+"\s*:\s*"n(\d+)"', body):
        candidates += int(value)
    if candidates:
        return candidates
    m = re.search(r'"nrofCandidates"\s*:\s*\{(?P<body>.*?)\}', body, re.S)
    if m:
        for value in re.findall(r'"n(\d+)"', m.group("body")):
            candidates += int(value)
    return candidates


def _parse_bwp(record: str, config: Dict[str, object]) -> None:
    for prefix, key in [("Downlink", "dl"), ("Uplink", "ul")]:
        block = _block_after(record, f"initial{prefix}BWP")
        if not block:
            continue
        m = re.search(r'"carrierBandwidth"\s*:\s*(\d+)', block)
        if m:
            config[f"ctx_{key}_bwp_n_prb"] = int(m.group(1))
        scs = re.search(r'"subcarrierSpacing"\s*:\s*"kHz(\d+)"', block)
        if scs:
            config[f"ctx_{key}_bwp_scs_khz"] = int(scs.group(1))


def _parse_activity_record(record: str, layer: str, global_slot: int, parsed: ParsedLog) -> None:
    first = record.splitlines()[0]
    rntis = [int(x, 16) for x in HEX_RNTI_RE.findall(record)]
    for rnti in rntis:
        if _is_candidate_ue_rnti(rnti):
            parsed.rnti_counts[rnti] += 1
    ue_match = re.search(r"\bue=(\d+)\s+(?:c-rnti|rnti)=0x([0-9a-fA-F]+)", record)
    if ue_match:
        parsed.ue_by_rnti[int(ue_match.group(2), 16)] = int(ue_match.group(1))

    if "Slot.indication" in first:
        return
    if "PDCCH:" in first and layer == "PHY":
        _parse_phy_pdcch(record, global_slot, parsed)
    elif "PDSCH:" in first and layer == "PHY":
        _parse_phy_pdsch(record, global_slot, parsed)
    elif "PUSCH:" in first and layer == "PHY":
        _parse_phy_pusch(record, global_slot, parsed)
    elif "PUCCH:" in first and layer == "PHY":
        _parse_phy_pucch(record, global_slot, parsed)
    elif "SRS:" in first and layer == "PHY":
        _parse_phy_srs(record, global_slot, parsed)
    elif layer == "FAPI":
        _parse_fapi(record, global_slot, parsed)
    elif layer == "SCHED":
        _parse_sched(record, global_slot, parsed)
    elif layer == "MAC":
        _parse_mac(record, global_slot, parsed)


def _row(
    parsed: ParsedLog,
    global_slot: int,
    rnti: int,
    source_line: Optional[int] = None,
) -> Dict[str, object]:
    line_number = source_line if source_line is not None else parsed.current_record_line
    if line_number is not None:
        parsed.source_line_numbers[(global_slot, rnti)].add(int(line_number))
    return parsed.rows[(global_slot, rnti)]


def _inc(parsed: ParsedLog, global_slot: int, rnti: int, name: str, amount: int = 1) -> None:
    parsed.row_counts[(global_slot, rnti)][name] += amount


def _parse_phy_pdcch(record: str, global_slot: int, parsed: ParsedLog) -> None:
    rnti = _rnti(record)
    if rnti is None:
        return
    row = _row(parsed, global_slot, rnti)
    _inc(parsed, global_slot, rnti, "pdcch_count")
    row["pdcch_present"] = 1
    _assign(row, "pdcch_ss_id", _first_int(record, r"\bss_id=(\d+)"))
    _assign(row, "pdcch_format", _first_str(record, r"\bformat=([0-9_]+)"))
    _assign(row, "pdcch_cce", _first_int(record, r"\bcce=(\d+)"))
    _assign(row, "pdcch_aggregation_level", _first_int(record, r"\bal=(\d+)"))
    fmt = row.get("pdcch_format")
    if isinstance(fmt, str):
        row["pdcch_direction"] = "ul_grant" if fmt.startswith("0_") else "dl_grant"


def _parse_phy_pdsch(record: str, global_slot: int, parsed: ParsedLog) -> None:
    rnti = _rnti(record)
    if rnti is None:
        return
    row = _row(parsed, global_slot, rnti)
    _inc(parsed, global_slot, rnti, "pdsch_count")
    row["pdsch_present"] = 1
    _assign_range(row, "pdsch_prb_start", "pdsch_n_prb", _first_range(record, r"\bprb=\[(\d+),\s*(\d+)\)"))
    _assign_range(row, "pdsch_sym_start", "pdsch_n_sym", _first_range(record, r"\bsymb=\[(\d+),\s*(\d+)\)"))
    for feature, pattern in {
        "pdsch_mod_order": r"\bmod=([A-Za-z0-9]+)",
        "pdsch_rv": r"\brv=(\d+)",
        "pdsch_tbs_bytes": r"\btbs=(\d+)",
        "pdsch_harq_id": r"\bh_id=(\d+)",
        "pdsch_k1": r"\bk1=(\d+)",
        "pdsch_dmrs_type": r"\bdmrs_type=(\d+)",
        "pdsch_power_dmrs_db": r"\bpower_dmrs=([+-]?[0-9.]+)dB",
        "pdsch_power_data_db": r"\bpower_data=([+-]?[0-9.]+)dB",
    }.items():
        value = _first_str(record, pattern)
        _assign(row, feature, _num(value) if value is not None and re.fullmatch(r"[+-]?[0-9.]+", value) else value)
    dmrs = re.search(r"\bdmrs_mask=\{([^}]*)\}", record)
    if dmrs:
        row["pdsch_dmrs_symbol_count"] = len([x for x in re.split(r"[\s,]+", dmrs.group(1).strip()) if x])


def _parse_phy_pusch(record: str, global_slot: int, parsed: ParsedLog) -> None:
    rnti = _rnti(record)
    if rnti is None:
        return
    row = _row(parsed, global_slot, rnti)
    _inc(parsed, global_slot, rnti, "pusch_count")
    row["pusch_present"] = 1
    _assign_range(row, "pusch_prb_start", "pusch_n_prb", _first_range(record, r"\bprb=\[(\d+),\s*(\d+)\)"))
    _assign_range(row, "pusch_sym_start", "pusch_n_sym", _first_range(record, r"\bsymb=\[(\d+),\s*(\d+)\)"))
    for feature, pattern in {
        "pusch_mod_order": r"\bmod=([A-Za-z0-9]+)",
        "pusch_rv": r"\brv=(\d+)",
        "pusch_tbs_bytes": r"\btbs=(\d+)",
        "pusch_harq_id": r"\bh_id=(\d+)",
        "pusch_iter": r"\biter=([0-9.]+)",
        "pusch_sinr_db": r"\bsinr=([+-]?[0-9.]+)dB",
        "pusch_nof_layers": r"\bnof_layers=(\d+)",
    }.items():
        value = _first_str(record, pattern)
        _assign(row, feature, _num(value) if value is not None and re.fullmatch(r"[+-]?[0-9.]+", value) else value)
    crc = _first_str(record, r"\bcrc=([A-Za-z]+)")
    if crc:
        row["pusch_crc_ok"] = 1 if crc.upper() == "OK" else 0


def _parse_phy_pucch(record: str, global_slot: int, parsed: ParsedLog) -> None:
    rnti = _rnti(record)
    if rnti is None:
        return
    row = _row(parsed, global_slot, rnti)
    _inc(parsed, global_slot, rnti, "pucch_count")
    row["pucch_present"] = 1
    _assign(row, "pucch_format", _first_int(record, r"\bformat=(\d+)"))
    _assign(row, "pucch_prb1", _first_int(record, r"\bprb1?=(\d+)"))
    _assign(row, "pucch_prb2", _first_int(record, r"\bprb2=(\d+)"))
    _assign_range(row, "pucch_sym_start", "pucch_n_sym", _first_range(record, r"\bsymb=\[(\d+),\s*(\d+)\)"))
    for feature, pattern in {
        "pucch_ack_value": r"\back=([01]+)",
        "pucch_sr_detected": r"\bsr=(yes|no|0|1)",
        "pucch_csi_payload": r"\bcsi1=([01]+)",
        "pucch_sinr_db": r"\bsinr=([+-]?[0-9.]+)dB",
        "pucch_metric": r"\bmetric=([+-]?[0-9.]+)",
    }.items():
        value = _first_str(record, pattern)
        if value is None:
            continue
        if feature == "pucch_sr_detected":
            row[feature] = 1 if value in ("yes", "1") else 0
        elif re.fullmatch(r"[+-]?[0-9.]+", value):
            row[feature] = _num(value)
        else:
            row[feature] = value
    n_sym = row.get("pucch_n_sym")
    if isinstance(n_sym, int):
        row["pucch_is_short"] = 1 if n_sym <= 2 else 0
        row["pucch_is_long"] = 1 if n_sym > 2 else 0


def _parse_phy_srs(record: str, global_slot: int, parsed: ParsedLog) -> None:
    rnti = _rnti(record)
    if rnti is None:
        return
    row = _row(parsed, global_slot, rnti)
    _inc(parsed, global_slot, rnti, "srs_count")
    row["srs_present"] = 1
    _assign_range(row, "srs_crb_start", "srs_n_crb", _first_range(record, r"\bcrb=\[(\d+)\.\.(\d+)\)"))
    _assign_range(row, "srs_sym_start", "srs_n_sym", _first_range(record, r"\bsymb=\[(\d+),\s*(\d+)\)"))
    for feature, pattern in {
        "srs_re_pattern": r"\bre=\{([^}]*)\}",
        "srs_sequence_id": r"\bsequence_id=(\d+)",
        "srs_timing_align_ns": r"\bt_align=([+-]?[0-9.]+)ns",
        "srs_epre_db": r"\bepre=([+-]?[0-9.]+)dB",
        "srs_rsrp_db": r"\brsrp=([+-]?[0-9.]+)dB",
        "srs_noise_var_db": r"\bnoise_var=([+-]?[0-9.]+)dB",
        "srs_nof_ports": r"\bnof_antenna_ports=(\d+)",
        "srs_comb_size": r"\bcomb_size=(\d+)",
        "srs_comb_offset": r"\bcomb_offset=(\d+)",
        "srs_cyclic_shift": r"\bcyclic_shift=(\d+)",
        "srs_freq_shift": r"\bfreq_shift=(\d+)",
    }.items():
        value = _first_str(record, pattern)
        _assign(row, feature, _num(value) if value is not None and re.fullmatch(r"[+-]?[0-9.]+", value) else value)


def _parse_fapi(record: str, global_slot: int, parsed: ParsedLog) -> None:
    target_rnti = _rnti(record)
    for line_offset, line in enumerate(record.splitlines()[1:], start=1):
        source_line = (
            parsed.current_record_line + line_offset
            if parsed.current_record_line is not None
            else None
        )
        if "NZP-CSI-RS" in line or "ZP-CSI-RS" in line:
            rnti = target_rnti or int(parsed.config.get("selected_rnti") or 0)
            if not rnti:
                rnti = 0
            row = _row(parsed, global_slot, rnti, source_line)
            row["csi_rs_present"] = 1
            if "NZP-CSI-RS" in line:
                _inc(parsed, global_slot, rnti, "csi_rs_nzp_count")
                _assign(row, "csi_rs_scrambling_id", _first_int(line, r"\bscramb_id=(\d+)"))
            else:
                _inc(parsed, global_slot, rnti, "csi_rs_zp_count")
            _assign_range(row, "csi_rs_crb_start", "csi_rs_n_crb", _first_range(line, r"\bcrbs=(\d+):(\d+)"))
            _assign(row, "csi_rs_row", _first_int(line, r"\brow=(\d+)"))
            _assign(row, "csi_rs_symb_l0", _first_int(line, r"\bsymbL0=(\d+)"))
            _assign(row, "csi_rs_symb_l1", _first_int(line, r"\bsymbL1=(\d+)"))
        if "PUCCH" in line:
            rnti = _rnti(line)
            if rnti is None:
                continue
            row = _row(parsed, global_slot, rnti, source_line)
            for feature, pattern in {
                "pucch_harq_bit_len": r"\bharq_bit_len=(\d+)",
                "pucch_sr_bit_len": r"\bsr_bit_len=(\d+)",
                "pucch_csi1_bit_len": r"\bcsi1_bit_len=(\d+)",
            }.items():
                _assign(row, feature, _first_int(line, pattern))
        if "SRS" in line:
            rnti = _rnti(line)
            if rnti is None:
                continue
            row = _row(parsed, global_slot, rnti, source_line)
            _assign(row, "srs_channel_matrix_requested", _first_int(line, r"\bnormalized_channel_iq_matrix_req=(\d+)"))
            _assign(row, "srs_positioning_report_requested", _first_int(line, r"\bpositioning_report_req=(\d+)"))


def _parse_sched(record: str, global_slot: int, parsed: ParsedLog) -> None:
    for line_offset, line in enumerate(record.splitlines()):
        rnti = _rnti(line)
        if rnti is None:
            continue
        source_line = (
            parsed.current_record_line + line_offset
            if parsed.current_record_line is not None
            else None
        )
        row = _row(parsed, global_slot, rnti, source_line)
        if "DL PDCCH:" in line or "UL PDCCH:" in line:
            row["pdcch_rnti_type"] = _first_str(line, r"\btype=([A-Za-z0-9_-]+)")
            for feature, pattern in {
                "pdcch_dci_harq_id": r"\bdci:.*?\bh_id=(\d+)",
                "pdcch_dci_rv": r"\bdci:.*?\brv=(\d+)",
                "pdcch_dci_mcs": r"\bdci:.*?\bmcs=(\d+)",
                "pdcch_dci_tpc": r"\bdci:.*?\btpc=(\d+)",
                "pdcch_dci_dai": r"\bdci:.*?\bdai=(\d+)",
                "pdcch_dci_antenna_ports": r"\bdci:.*?\bant=(\d+)",
            }.items():
                _assign(row, feature, _first_int(line, pattern))
            ndi = _first_str(line, r"\bdci:.*?\bndi=([A-Za-z0-9]+)")
            if ndi is not None:
                row["pdcch_dci_ndi"] = 1 if ndi.lower() in ("true", "1") else 0
        elif "UE PDSCH:" in line:
            for feature, pattern in {
                "pdsch_mcs": r"\bmcs=(\d+)",
                "pdsch_retx_count": r"\bnrtx=(\d+)",
                "pdsch_olla": r"\bolla=([+-]?[0-9.]+)",
                "pdsch_dl_bo": r"\bdl_bo=(\d+)",
                "pdsch_nof_layers": r"\bnof_layers=(\d+)",
            }.items():
                value = _first_str(line, pattern)
                _assign(row, feature, _num(value) if value is not None else None)
            grants = re.findall(r"lcid=\d+:\s*size=(\d+)", line)
            if grants:
                row["pdsch_dl_buffer_bytes"] = sum(int(x) for x in grants)
        elif "UE PUSCH:" in line:
            for feature, pattern in {
                "pusch_mcs": r"\bmcs=(\d+)",
                "pusch_k2": r"\bk2=(\d+)",
                "pusch_nof_layers": r"\bnof_layers=(\d+)",
                "pusch_retx_count": r"\bnrtx=(\d+)",
                "pusch_olla": r"\bolla=([+-]?[0-9.]+)",
            }.items():
                value = _first_str(line, pattern)
                _assign(row, feature, _num(value) if value is not None else None)
        elif "PUCCH:" in line:
            for feature, pattern in {
                "pucch_scheduled_harq_bits": r"\bharq_bits=(\d+)",
                "pucch_scheduled_sr_bits": r"\bsr=(\d+)",
                "pucch_scheduled_csi1_bits": r"\bcsi-1_bits=(\d+)",
            }.items():
                _assign(row, feature, _first_int(line, pattern))
        elif "SRS:" in line:
            _assign(row, "srs_tpmi", _first_int(line, r"\btpmi=(\d+)"))
            _assign(row, "srs_report_sinr_db", _first_float(line, r"\bsinr=\(([+-]?[0-9.]+)\)dB"))
        elif "CSI:" in line:
            row["csi_report_present"] = 1
            row["csi_report_valid"] = 0 if "invalid" in line else 1
            _assign(row, "csi_cqi", _first_int(line, r"\bcqi=(\d+)"))
            _assign(row, "csi_ri", _first_int(line, r"\bri=(\d+)"))
        elif "PHR:" in line:
            row["phr_present"] = 1
            _assign(row, "phr_ph_db_mid", _range_mid(_first_range(line, r"\bph=\[([+-]?\d+)\.\.([+-]?\d+)\)dB")))
            _assign(row, "phr_p_cmax_dbm_mid", _range_mid(_first_range(line, r"\bp_cmax=\[([+-]?\d+)\.\.([+-]?\d+)\)dBm")))
            if "phr_ph_db_mid" in row and "phr_p_cmax_dbm_mid" in row:
                row["tx_power_est_dbm"] = row["phr_p_cmax_dbm_mid"] - row["phr_ph_db_mid"]
        elif "BSR:" in line:
            _assign(row, "bsr_pending_bytes", _first_int(line, r"\bpending_bytes=(\d+)"))
        elif "RLC Buffer State:" in line:
            _assign(row, "rlc_pending_bytes", _first_int(line, r"\bpending_bytes=(\d+)"))
        elif "HARQ-ACK:" in line:
            _inc(parsed, global_slot, rnti, "harq_ack_count", line.count("ACK"))
            _inc(parsed, global_slot, rnti, "harq_nack_count", line.count("NACK"))
        elif "SR:" in line:
            row["sr_detected"] = 1


def _parse_mac(record: str, global_slot: int, parsed: ParsedLog) -> None:
    for line_offset, line in enumerate(record.splitlines()):
        rnti = _rnti(line)
        if rnti is None:
            continue
        source_line = (
            parsed.current_record_line + line_offset
            if parsed.current_record_line is not None
            else None
        )
        row = _row(parsed, global_slot, rnti, source_line)
        if "UL rnti=" in line:
            sdu_bytes = sum(int(x) for x in re.findall(r"lcid=\d+:\s*len=(\d+)", line))
            pad_bytes = sum(int(x) for x in re.findall(r"PAD:\s*len=(\d+)", line))
            if sdu_bytes:
                row["pusch_mac_sdu_bytes"] = sdu_bytes
                row["ul_mac_bytes"] = sdu_bytes + pad_bytes
            if pad_bytes:
                row["pusch_mac_padding_bytes"] = pad_bytes
            if "SBSR" in line or "BSR" in line:
                row["pusch_contains_bsr"] = 1
            if "SE_PHR" in line or "PHR" in line:
                row["pusch_contains_phr"] = 1
        elif "DL" in line:
            size = _first_int(line, r"\bsize=(\d+)")
            if size:
                row["pdsch_mac_pdu_bytes"] = size
                row["dl_mac_bytes"] = size
            if "TA_CMD" in line:
                row["pdsch_ta_cmd_present"] = 1


def _select_default_rnti(counts: Counter) -> Optional[int]:
    candidates = [(rnti, count) for rnti, count in counts.items() if _is_candidate_ue_rnti(rnti)]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])[0]


def _is_candidate_ue_rnti(rnti: int) -> bool:
    return rnti not in (0, 0xFFFF) and rnti >= 0x1000


def _rnti(text: str) -> Optional[int]:
    m = HEX_RNTI_RE.search(text)
    return int(m.group(1), 16) if m else None


def _first_int(text: str, pattern: str) -> Optional[int]:
    m = re.search(pattern, text, re.S)
    return int(m.group(1)) if m else None


def _first_float(text: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, text, re.S)
    return float(m.group(1)) if m else None


def _first_str(text: str, pattern: str) -> Optional[str]:
    m = re.search(pattern, text, re.S)
    return m.group(1) if m else None


def _first_range(text: str, pattern: str) -> Optional[Tuple[int, int]]:
    m = re.search(pattern, text, re.S)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _assign(row: Dict[str, object], key: str, value: object) -> None:
    if value is not None:
        row[key] = value


def _assign_range(row: Dict[str, object], start_key: str, n_key: str, value: Optional[Tuple[int, int]]) -> None:
    if value is None:
        return
    start, end = value
    row[start_key] = start
    row[n_key] = end - start


def _range_mid(value: Optional[Tuple[int, int]]) -> Optional[float]:
    if value is None:
        return None
    return (value[0] + value[1]) / 2.0


def _num(value: str) -> object:
    if "." in value:
        return float(value)
    return int(value)


def _sum_int_list(text: str) -> int:
    return sum(int(part.strip()) for part in text.split(",") if part.strip())


def _block_after(text: str, marker: str) -> str:
    idx = text.find(marker)
    return "" if idx < 0 else text[idx:]

"""Shared data models and type aliases for 3GPP power estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

BinKey = Tuple[int, int, int]  # (cycle_idx, frame_idx, slot_idx@30kHz grid)
RowBinLookupKey = Tuple[str, int]  # (source filename, zero-based CSV data row index)
RowBinLookupValue = Tuple[int, int, float]  # (start_bin, duration_bins, row_timestamp_sod)

@dataclass
class BinEvidence:
    """Row-level evidence accumulated per bin before state resolution.

    Parsers only increment these counters. The final bin-by-bin loop converts
    them to boolean flags (`SlotFlags`) and then applies section 8.1 logic.
    """

    pdcch_dl_rows: int = 0
    pdcch_ul_rows: int = 0
    pdsch_rows: int = 0
    pusch_rows: int = 0
    srs_rows: int = 0
    prach_rows: int = 0
    pucch_short_rows: int = 0
    pucch_long_rows: int = 0
    k0_gt0_rows: int = 0
    pdsch_inferred_from_k0_zero_rows: int = 0
    pdsch_rx_2x2_rows: int = 0
    pdsch_rx_4x4_rows: int = 0

@dataclass
class SlotFlags:
    """Per-bin channel activity booleans consumed by state selection."""

    has_pdcch_dl: bool = False
    has_pdcch_ul: bool = False
    has_pdsch: bool = False
    has_pusch: bool = False
    has_srs: bool = False
    has_prach: bool = False
    has_pucch_short: bool = False
    has_pucch_long: bool = False
    has_k0_gt0: bool = False
    has_pdsch_inferred_from_k0_zero: bool = False
    pdsch_rx_2x2_seen: bool = False
    pdsch_rx_4x4_seen: bool = False

@dataclass
class WarningLog:
    """Collect non-fatal parse/model warnings for final reporting."""

    unknown_scs_values: Set[str] = field(default_factory=set)
    unknown_pucch_formats: Set[str] = field(default_factory=set)
    unknown_pucch_symbol_values: Set[str] = field(default_factory=set)
    unknown_rrc_scs_values: Set[str] = field(default_factory=set)
    missing_pucch_symbol_count_rows: int = 0
    skipped_files: List[str] = field(default_factory=list)

    def to_messages(self) -> List[str]:
        """
        Flatten warning buckets into human-readable strings.
        
        Args:
            None.
        
        Returns:
            List[str]: Flattened warning message strings.
        
        Example:
            >>> WarningLog(unknown_scs_values={"60KHZ"}).to_messages()[0].startswith("Unknown SCS values")
            True
        """
        messages: List[str] = []
        if self.unknown_scs_values:
            values = ", ".join(sorted(self.unknown_scs_values))
            messages.append(
                f"Unknown SCS values encountered; defaulted to 30kHz/1-bin behavior: {values}"
            )
        if self.unknown_pucch_formats:
            values = ", ".join(sorted(self.unknown_pucch_formats))
            messages.append(
                f"Unknown PUCCH formats encountered; defaulted to long PUCCH: {values}"
            )
        if self.unknown_pucch_symbol_values:
            values = ", ".join(sorted(self.unknown_pucch_symbol_values))
            messages.append(
                "Unparseable PUCCH Num Symbols values encountered; "
                f"fell back to format-based classification: {values}"
            )
        if self.missing_pucch_symbol_count_rows:
            messages.append(
                "PUCCH Num Symbols missing in "
                f"{self.missing_pucch_symbol_count_rows} rows; "
                "fell back to format-based classification."
            )
        if self.unknown_rrc_scs_values:
            values = ", ".join(sorted(self.unknown_rrc_scs_values))
            messages.append(
                f"Unknown RRC subcarrierSpacing values; BWP bandwidth mapping skipped: {values}"
            )
        if self.skipped_files:
            messages.extend(self.skipped_files)
        return messages

@dataclass
class RrcSrsResource:
    """RRC SRS resource fields needed for audit columns in per-slot CSV."""

    srs_resource_id: int
    nrof_srs_ports: Optional[str] = None
    nrof_srs_ports_count: Optional[int] = None
    nrof_symbols: Optional[str] = None
    start_position: Optional[int] = None
    freq_domain_position: Optional[int] = None
    freq_domain_shift: Optional[int] = None
    c_srs: Optional[int] = None
    b_srs: Optional[int] = None
    b_hop: Optional[int] = None
    transmission_comb_value: Optional[int] = None
    transmission_comb_offset: Optional[int] = None

@dataclass
class RrcSearchSpaceConfig:
    """RRC search-space config used to infer PDCCH monitoring schedule."""

    search_space_id: int
    periodicity_slots: int
    offset_slots: int
    duration_slots: int
    monitoring_symbols_within_slot: Optional[str] = None
    has_monitoring_symbols: bool = True
    candidate_count: int = 36
    candidate_counts_by_level: Dict[int, int] = field(default_factory=dict)
    search_space_type: Optional[str] = None
    control_resource_set_id: Optional[int] = None

@dataclass
class RrcControlResourceSetConfig:
    """One DL control resource set used for CORESET-level PDCCH geometry."""

    control_resource_set_id: int
    frequency_domain_resources: Optional[str] = None
    duration: Optional[int] = None

@dataclass
class RrcTimeDomainAllocation:
    """One decoded TDRA list entry used for DCI geometry decoding."""

    allocation_index: int
    mapping_type: Optional[str] = None
    start_symbol: Optional[int] = None
    num_symbols: Optional[int] = None
    k0: Optional[int] = None
    k2: Optional[int] = None

@dataclass
class RrcMibConfig:
    """MIB-derived FR1 SSB context used for SSB schedule interpretation."""

    ssb_scs: Optional[str] = None
    ssb_scs_source: Optional[str] = None
    absolute_frequency_ssb: Optional[int] = None
    freq_band_indicator_nr: Optional[int] = None
    freq_band_indicator_source: Optional[str] = None

@dataclass
class RrcSsbConfig:
    """Resolved SSB schedule in slot-domain form."""

    ssb_scs: Optional[str] = None
    ssb_band: Optional[int] = None
    ssb_case: Optional[str] = None
    periodicity_slots: Optional[int] = None
    sync_periodicity_slots: Optional[int] = None
    offset_slots_global: Optional[int] = None
    ssb_case_source: Optional[str] = None
    phase_anchor_source: Optional[str] = None
    # Key: slot offset within one SSB burst period. Value: number of SSBs in that slot.
    slot_ssb_count_map: Dict[int, int] = field(default_factory=dict)
    raw_positions_text: Optional[str] = None
    raw_periodicity_text: Optional[str] = None

@dataclass
class RrcNzpCsiRsResourceConfig:
    """One NZP CSI-RS resource periodic schedule."""

    resource_id: int
    periodicity_slots: int
    offset_slots: int

@dataclass
class RrcNzpCsiRsResourceSetConfig:
    """One NZP CSI-RS resource set containing NZP resource IDs."""

    resource_set_id: int
    nzp_resource_ids: List[int] = field(default_factory=list)
    trs_info: bool = False

@dataclass
class RrcCsiResourceConfig:
    """CSI resource-config entry, mapping report usage to NZP resource sets."""

    resource_config_id: int
    bwp_id: Optional[int] = None
    resource_type: Optional[str] = None
    resource_domain: Optional[str] = None
    nzp_resource_set_ids: List[int] = field(default_factory=list)
    csi_ssb_resource_set_ids: List[int] = field(default_factory=list)
    csi_im_resource_set_ids: List[int] = field(default_factory=list)

@dataclass
class RrcCsiReportConfig:
    """CSI report-config entry used to gate monitored channel-measurement resources."""

    report_config_id: int
    resources_for_channel_measurement: Optional[int] = None
    is_periodic: bool = False

@dataclass
class RrcBwpConfig:
    """Resolved BWP definition tracked by BWP ID (DL or UL)."""

    bwp_id: int
    location_and_bandwidth: Optional[int] = None
    scs: Optional[str] = None
    start_rb: Optional[int] = None
    size_rb: Optional[int] = None
    bw_mhz: Optional[int] = None
    srs_resources: Dict[int, RrcSrsResource] = field(default_factory=dict)
    common_search_spaces: Dict[int, RrcSearchSpaceConfig] = field(default_factory=dict)
    dedicated_search_spaces: Dict[int, RrcSearchSpaceConfig] = field(default_factory=dict)
    control_resource_sets: Dict[int, RrcControlResourceSetConfig] = field(default_factory=dict)
    pdsch_time_domain_allocations: List[RrcTimeDomainAllocation] = field(default_factory=list)
    pusch_time_domain_allocations: List[RrcTimeDomainAllocation] = field(default_factory=list)
    # UL-side PUSCH capability fields are stored on the BWP config so the
    # scorer can infer a conservative UL-TX capability even when no same-slot
    # PHY SRS row is present.
    pusch_tx_config: Optional[str] = None
    pusch_codebook_subset: Optional[str] = None
    pusch_max_rank: Optional[int] = None
    pusch_transform_precoding_disabled: Optional[bool] = None
    pusch_nr_of_antenna_ports: Optional[str] = None
    ul_tx_capability: Optional[str] = None

@dataclass
class RrcState:
    """Fill-forward RRC configuration state active at a given event bin."""

    dl_bwps: Dict[int, RrcBwpConfig] = field(default_factory=dict)
    ul_bwps: Dict[int, RrcBwpConfig] = field(default_factory=dict)
    active_dl_bwp_id: Optional[int] = None
    active_ul_bwp_id: Optional[int] = None
    mib_config: RrcMibConfig = field(default_factory=RrcMibConfig)
    ssb_config: RrcSsbConfig = field(default_factory=RrcSsbConfig)
    nzp_csi_rs_resources: Dict[int, RrcNzpCsiRsResourceConfig] = field(default_factory=dict)
    nzp_csi_rs_resource_sets: Dict[int, RrcNzpCsiRsResourceSetConfig] = field(default_factory=dict)
    csi_resource_configs: Dict[int, RrcCsiResourceConfig] = field(default_factory=dict)
    csi_report_configs: Dict[int, RrcCsiReportConfig] = field(default_factory=dict)

@dataclass
class RrcParseArtifacts:
    """Outputs produced by `parse_rrc_json()`."""

    events: List[Tuple[int, int, RrcState]]
    parse_stats: Dict[str, object]
    fallback_counts: Dict[str, int]

@dataclass
class TimestampMergeSpec:
    """Column-selection spec for one CSV participating in timestamp merge."""

    filename: str
    frame_cols: Tuple[str, ...]
    slot_cols: Tuple[str, ...]
    scs_cols: Tuple[str, ...]


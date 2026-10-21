"""Constants for the 3GPP TS 38.840 section 8.1 model."""

from __future__ import annotations

from typing import Dict, Tuple

# FR1 relative power references from 38.840 section 8.1
REL_MICRO_SLEEP = 45.0
REL_PDCCH_ONLY = 100.0
REL_PDCCH_ONLY_CROSS_SLOT = 0.7 * REL_PDCCH_ONLY
REL_PDCCH_PDSCH = 300.0
REL_PDSCH_ONLY = 280.0
REL_UL_LONG_0DBM = 250.0
REL_UL_LONG_23DBM = 700.0
# TODO: where is PUCCH power defnition?

BASE_BIN_SEC = 0.0005  # 0.5 ms
SLOTS_PER_30KHZ_FRAME = 20
FRAME_MODULUS = 1024
CYCLE_BINS = FRAME_MODULUS * SLOTS_PER_30KHZ_FRAME

SCS_TO_BINS = {
    "30KHZ": 1,
    "15KHZ": 2,
}

BinKey = Tuple[int, int, int]  # (cycle_idx, frame_idx, slot_idx@30kHz grid)
RowBinLookupKey = Tuple[str, int]  # (source filename, zero-based CSV data row index)
RowBinLookupValue = Tuple[int, int, float]  # (start_bin, duration_bins, row_timestamp_sod)

SHORT_PUCCH_FORMATS = {"PUCCH_FORMAT_F0", "PUCCH_FORMAT_F2"}
LONG_PUCCH_FORMATS = {"PUCCH_FORMAT_F1", "PUCCH_FORMAT_F3", "PUCCH_FORMAT_F4"}

# RRC locationAndBandwidth decoding assumes maximum FR1 BWP indexing basis.
N_RB_LOCATION_AND_BANDWIDTH = 275

# FR1 anchors for BWP size -> nominal MHz mapping. E.g., 15kHz x 52 x 12 => 10 MHz, 30kHz x 106 x 106 x 12 => 40 MHz, etc. Used for bandwidth-based scaling.
FR1_BW_ANCHORS = {
    "15KHZ": {52: 10, 106: 20, 216: 40},
    "30KHZ": {51: 20, 106: 40, 217: 80, 273: 100},
}

# TODO: check if RRC setup can also update list of assigned BWP
RRC_RELEVANT_MESSAGE_NAMES = {
    "BCCH_BCH / Mib",
    "BCCH_DL_SCH / SystemInformationBlockType1",
    "DL_CCCH / RRC Setup",
    "DL_DCCH / RRCReconfiguration",
}
# Name substrings that should be ignored even if the upstream message-type
# filter is broadened later. Matching is case-insensitive substring.
# NOTE: `BCCH_BCH / Mib` is an explicit allowlisted exception in parser logic.
RRC_OMIT_MESSAGE_NAME_SUBSTRINGS = ("UL_DCCH", "UL_CCCH", "BCCH_BCH")
# Candidate timestamp fields in parsed 0xB821 JSON.
# The parser checks these in order and uses the first parseable value.
RRC_TIMESTAMP_FIELD_CANDIDATES = (
    "timestamp",
    "Timestamp",
    "timestamp_line",
    "ota_timestamp",
    "summary_timestamp",
)

# 38.840 section 8.1.3: when DRX is not modeled, periodic sync/tracking
# activities use the 160 ms baseline directly.
TRACKING_BASELINE_MS = 160
TRACKING_BASELINE_SLOTS_BY_SCS = {
    "15KHZ": 160,
    "30KHZ": 320,
}

# 38.104 v17.6.0 Table 5.4.3.3-1 condensed to FR1 `(band, ssb_scs) -> case`.
# Most band/SCS pairs map to exactly one case. The value is still a tuple so the
# parser can preserve the "use absoluteFrequencySSB as tie-breaker if multiple
# cases exist" workflow without changing the data shape later.
FR1_SSB_CASES_BY_BAND_AND_SCS: Dict[str, Dict[int, Tuple[str, ...]]] = {
    "15KHZ": {
        1: ("A",),
        2: ("A",),
        3: ("A",),
        5: ("A",),
        7: ("A",),
        8: ("A",),
        12: ("A",),
        13: ("A",),
        14: ("A",),
        18: ("A",),
        20: ("A",),
        24: ("A",),
        25: ("A",),
        26: ("A",),
        28: ("A",),
        29: ("A",),
        30: ("A",),
        41: ("A",),
        50: ("A",),
        51: ("A",),
        53: ("A",),
        65: ("A",),
        66: ("A",),
        67: ("A",),
        70: ("A",),
        71: ("A",),
        74: ("A",),
        75: ("A",),
        76: ("A",),
        85: ("A",),
        90: ("A",),
        91: ("A",),
        92: ("A",),
        93: ("A",),
        94: ("A",),
        100: ("A",),
        101: ("A",),
    },
    "30KHZ": {
        5: ("B",),
        7: ("B",),
        34: ("A",),
        38: ("A",),
        39: ("A",),
        40: ("B",),
        41: ("C",),
        46: ("C",),
        48: ("C",),
        50: ("A",),
        51: ("A",),
        53: ("A",),
        66: ("B",),
        70: ("B",),
        75: ("B",),
        76: ("B",),
        77: ("C",),
        78: ("C",),
        79: ("C",),
        85: ("A",),
        90: ("C",),
        91: ("A",),
        92: ("A",),
        93: ("A",),
        94: ("A",),
        96: ("C",),
        100: ("A",),
        101: ("C",),
        102: ("C",),
        104: ("C",),
    },
}

SSB_PHASE_ANCHOR_SOURCE = "FIRST_SERVING_SSB_CONFIG_EVENT"
CSI_SYNC_SOURCE = "TRS"

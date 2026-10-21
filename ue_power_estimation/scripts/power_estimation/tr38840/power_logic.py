"""3GPP section 8.1 state classification and power math."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

try:
    from .constants import (
        REL_MICRO_SLEEP,
        REL_PDCCH_ONLY,
        REL_PDCCH_ONLY_CROSS_SLOT,
        REL_PDCCH_PDSCH,
        REL_PDSCH_ONLY,
        REL_UL_LONG_0DBM,
        REL_UL_LONG_23DBM,
    )
    from .models import BinEvidence, SlotFlags
except ImportError:
    from constants import (
        REL_MICRO_SLEEP,
        REL_PDCCH_ONLY,
        REL_PDCCH_ONLY_CROSS_SLOT,
        REL_PDCCH_PDSCH,
        REL_PDSCH_ONLY,
        REL_UL_LONG_0DBM,
        REL_UL_LONG_23DBM,
    )
    from models import BinEvidence, SlotFlags

def ul_long_relative(ul_power_mode: str) -> float:
    """
    Return UL long-slot relative power baseline for selected UL mode.
    
    Args:
        ul_power_mode (str): UL power mode (`0dbm` or `23dbm`).
    
    Returns:
        float: Relative UL-long power baseline for the selected mode.
    
    Example:
        >>> ul_long_relative("23dbm")
        700.0
    """

    return REL_UL_LONG_23DBM if ul_power_mode == "23dbm" else REL_UL_LONG_0DBM

def ul_antenna_scale_factor(
    ul_power_mode: str,
    ul_tx_count_inferred: Optional[int],
) -> float:
    """
    Return FR1 UL antenna scaling factor for inferred UL TX count.

    The current FR1 model only distinguishes 1TX and 2TX. Anything above 2TX
    should have been clamped by the caller before reaching this helper.
    """

    if ul_tx_count_inferred is None or ul_tx_count_inferred <= 1:
        return 1.0
    if ul_power_mode == "23dbm":
        return 1.2
    return 1.4

def dl_bwp_scaling_factor(dl_bwp_bw_mhz: Optional[int]) -> float:
    """
    Compute section-8.1.3 DL-BWP bandwidth scaling factor for PDSCH states.

    Scaling formula:
    `scale = 0.4 + 0.6 * (X - 20) / 80`, where `X` is DL BWP BW in MHz,
    clamped into the FR1 nominal range [10, 100]. If bandwidth is unknown,
    return neutral scale 1.0.

    Args:
        dl_bwp_bw_mhz (Optional[int]): Effective DL BWP bandwidth in MHz.

    Returns:
        float: DL BWP scaling multiplier.

    Example:
        >>> round(dl_bwp_scaling_factor(20), 4)
        0.4
    """

    if dl_bwp_bw_mhz is None:
        return 1.0
    x = float(max(10, min(100, dl_bwp_bw_mhz)))
    return 0.4 + 0.6 * ((x - 20.0) / 80.0)

def pdsch_scaling_factor(flags: SlotFlags, dl_bwp_scale: float) -> float:
    """
    Combine DL-BWP bandwidth scale and optional 2Rx antenna scale.
    
    Args:
        flags (SlotFlags): Per-bin activity flags used for state selection.
        dl_bwp_scale (float): DL-BWP bandwidth scaling factor.
    
    Returns:
        float: Combined PDSCH scaling multiplier.
    
    Example:
        >>> pdsch_scaling_factor(SlotFlags(has_pdsch=True, pdsch_rx_2x2_seen=True), 1.0)
        0.7
    """

    # Apply FR1 antenna scaling only when a bin is clearly 2Rx-only for PDSCH.
    antenna_scale = 1.0
    if flags.has_pdsch and flags.pdsch_rx_2x2_seen and not flags.pdsch_rx_4x4_seen:
        antenna_scale = 0.7
    return antenna_scale * dl_bwp_scale

def build_candidates(
    flags: SlotFlags,
    ul_mode: str,
    dl_bwp_scale: float,
    ul_antenna_scale: float = 1.0,
    pdcch_only_same_slot_rel_override: Optional[float] = None,
) -> List[Tuple[str, float]]:
    """
    Build candidate `(state_name, relative_power)` values for one bin.
    
    This function encodes explicit section 8.1 rules. If multiple candidates
    remain and no explicit combined state exists, `pick_state()` applies the
    agreed policy: choose the max-power candidate.
    
    Args:
        flags (SlotFlags): Per-bin activity flags used for state selection.
        ul_mode (str): ul mode input.
        dl_bwp_scale (float): DL-BWP bandwidth scaling factor.
        pdcch_only_same_slot_rel_override (Optional[float]): Optional override
            for same-slot PDCCH-only relative power (cross-slot remains fixed).
    
    Returns:
        List[Tuple[str, float]]: List of `(state_name, relative_power)` candidates for the bin.
    
    Example:
        >>> build_candidates(SlotFlags(has_pdcch_dl=True), "0dbm", 1.0)
        [('PDCCH_ONLY', 100.0)]
    """

    candidates: List[Tuple[str, float]] = []
    ul_rel = ul_long_relative(ul_mode) * ul_antenna_scale
    short_rel = 0.3 * ul_rel
    srs_rel = 0.3 * ul_rel

    pdcch_present = flags.has_pdcch_dl or flags.has_pdcch_ul
    cross_slot_pdcch = flags.has_pdcch_dl and flags.has_k0_gt0 and not flags.has_pdsch
    if cross_slot_pdcch:
        pdcch_only_rel = REL_PDCCH_ONLY_CROSS_SLOT
    elif pdcch_only_same_slot_rel_override is not None:
        pdcch_only_rel = pdcch_only_same_slot_rel_override
    else:
        pdcch_only_rel = REL_PDCCH_ONLY

    pdsch_scale = pdsch_scaling_factor(flags, dl_bwp_scale)

    if flags.has_pdsch and flags.has_pdcch_dl:
        candidates.append(("PDCCH_PDSCH", REL_PDCCH_PDSCH * pdsch_scale))
    elif flags.has_pdsch:
        candidates.append(("PDSCH_ONLY", REL_PDSCH_ONLY * pdsch_scale))

    if pdcch_present and not flags.has_pdsch:
        state = "PDCCH_ONLY_CROSS_SLOT" if cross_slot_pdcch else "PDCCH_ONLY"
        candidates.append((state, pdcch_only_rel))

    # Apply explicit 8.1 simplification:
    # - PDCCH + PUCCH(short) = PDCCH-only + short PUCCH
    if pdcch_present and not flags.has_pdsch and flags.has_pucch_short:
        candidates.append(("PDCCH_PUCCH_SHORT_COMBO", pdcch_only_rel + short_rel))

    # If PDSCH is present, 8.1 simplification keeps PDSCH or PDCCH+PDSCH power;
    # we suppress explicit PUCCH contribution in that case.
    if not flags.has_pdsch:
        if flags.has_pucch_short:
            candidates.append(("PUCCH_SHORT", short_rel))
        if flags.has_pucch_long:
            candidates.append(("PUCCH_LONG", ul_rel))

    if flags.has_pusch:
        candidates.append(("PUSCH_UL_LONG", ul_rel))
    if flags.has_srs:
        candidates.append(("SRS", srs_rel))
    if flags.has_prach:
        candidates.append(("PRACH_AS_UL_LONG", ul_rel))

    return candidates

def pick_state(
    flags: SlotFlags,
    ul_mode: str,
    dl_bwp_scale: float,
    ul_antenna_scale: float = 1.0,
    pdcch_only_same_slot_rel_override: Optional[float] = None,
) -> Tuple[str, float]:
    """
    Resolve one dominant state/power for a bin from candidate states.
    
    Args:
        flags (SlotFlags): Per-bin activity flags used for state selection.
        ul_mode (str): ul mode input.
        dl_bwp_scale (float): DL-BWP bandwidth scaling factor.
        pdcch_only_same_slot_rel_override (Optional[float]): Optional override
            for same-slot PDCCH-only relative power (cross-slot remains fixed).
    
    Returns:
        Tuple[str, float]: Chosen `(state_name, relative_power)` for the bin.
    
    Example:
        >>> pick_state(SlotFlags(has_pdcch_dl=True), "0dbm", 1.0)
        ('PDCCH_ONLY', 100.0)
    """

    candidates = build_candidates(
        flags,
        ul_mode,
        dl_bwp_scale,
        ul_antenna_scale=ul_antenna_scale,
        pdcch_only_same_slot_rel_override=pdcch_only_same_slot_rel_override,
    )
    if not candidates:
        return "MICRO_SLEEP", REL_MICRO_SLEEP
    # Undefined mixed combinations use max component power policy.
    state, rel = max(candidates, key=lambda x: x[1])
    return state, rel

def rel_to_abs(rel_power: float, microsleep_power: float) -> float:
    """
    Convert section-8.1 relative power into assumed absolute power.
    
    Args:
        rel_power (float): Relative section-8.1 power value.
        microsleep_power (float): Assumed micro-sleep absolute power baseline in mW.
    
    Returns:
        float: Absolute power value derived from relative power.
    
    Example:
        >>> rel_to_abs(90.0, 1.0)
        2.0
    """

    return microsleep_power * (rel_power / REL_MICRO_SLEEP)

def evidence_to_slot_flags(evidence: Optional[BinEvidence]) -> SlotFlags:
    """
    Convert accumulated per-bin evidence counters into boolean flags.
    
    Args:
        evidence (Optional[BinEvidence]): Per-bin evidence counters (or `None` when absent).
    
    Returns:
        SlotFlags: Boolean slot flags derived from evidence counters.
    
    Example:
        >>> evidence_to_slot_flags(BinEvidence(pdsch_rows=1)).has_pdsch
        True
    """

    if evidence is None:
        return SlotFlags()
    return SlotFlags(
        has_pdcch_dl=evidence.pdcch_dl_rows > 0,
        has_pdcch_ul=evidence.pdcch_ul_rows > 0,
        has_pdsch=evidence.pdsch_rows > 0,
        has_pusch=evidence.pusch_rows > 0,
        has_srs=evidence.srs_rows > 0,
        has_prach=evidence.prach_rows > 0,
        has_pucch_short=evidence.pucch_short_rows > 0,
        has_pucch_long=evidence.pucch_long_rows > 0,
        has_k0_gt0=evidence.k0_gt0_rows > 0,
        has_pdsch_inferred_from_k0_zero=evidence.pdsch_inferred_from_k0_zero_rows > 0,
        pdsch_rx_2x2_seen=evidence.pdsch_rx_2x2_rows > 0,
        pdsch_rx_4x4_seen=evidence.pdsch_rx_4x4_rows > 0,
    )

def bool_to_int(v: bool) -> int:
    """
    CSV helper: store booleans as 0/1 for easy downstream processing.
    
    Args:
        v (bool): Boolean value to convert.
    
    Returns:
        int: `1` when input is `True`, otherwise `0`.
    
    Example:
        >>> bool_to_int(True)
        1
    """

    return 1 if v else 0

def resolve_effective_bwp_id(
    same_bin_bwp_id: Optional[int],
    last_seen_dci_bwp_id: Optional[int],
    active_rrc_bwp_id: Optional[int],
    fallback_counter_key: str,
    fallback_counts: Dict[str, int],
) -> Tuple[int, Optional[int]]:
    """
    Resolve effective BWP ID for a bin using priority chain.
    
    Priority:
    1) same-bin DCI BWP ID,
    2) last-seen DCI BWP ID (fill-forward from DCI),
    3) active RRC BWP ID,
    4) default 0 (counted as fallback).
    
    Args:
        same_bin_bwp_id (Optional[int]): BWP ID seen in DCI for the current bin.
        last_seen_dci_bwp_id (Optional[int]): Last DCI-derived BWP ID from previous bins.
        active_rrc_bwp_id (Optional[int]): Active BWP ID from current RRC state.
        fallback_counter_key (str): Fallback counter name incremented on default selection.
        fallback_counts (Dict[str, int]): Counter map for parser fallback conditions.
    
    Returns:
        Tuple[int, Optional[int]]: Tuple `(effective_bwp_id, updated_last_seen_dci_bwp_id)`.
    
    Example:
        >>> resolve_effective_bwp_id(2, None, None, "x", {})
        (2, 2)
    """

    if same_bin_bwp_id is not None:
        return same_bin_bwp_id, same_bin_bwp_id
    if last_seen_dci_bwp_id is not None:
        return last_seen_dci_bwp_id, last_seen_dci_bwp_id
    if active_rrc_bwp_id is not None:
        return active_rrc_bwp_id, last_seen_dci_bwp_id
    fallback_counts[fallback_counter_key] += 1
    return 0, last_seen_dci_bwp_id


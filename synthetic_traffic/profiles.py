"""App-like profiles: compose patterns.py primitives into (DL, UL) envelopes.

A *profile* is the shape recipe for one app type. A *seed* (passed in as the
Random instance) fixes every random draw, so (profile, seed) is reproducible at
a given tick.

Each builder takes (n, rng, caps, tick_s) and returns (dl_kbps, ul_kbps): two
length-n lists, one value per tick. Rates are clamped to caps by the caller
(schedule_gen).

Time-constants here are written in SECONDS and converted to ticks via tick_s, so
the *shape* is independent of resolution -- lowering --tick just samples the same
envelope more finely (e.g. with the persistent-UDP player at tick=0.1 you keep a
~4s YouTube buffer-fill burst, you just get it in 0.1s steps). The patterns.py
primitives still count in ticks; the helpers below do the seconds->ticks and
per-second->per-tick conversions. Diffusion (random_walk) scales its per-step
sigma by sqrt(tick_s) so the variance accumulated per real second is constant
across resolutions.
"""

from __future__ import annotations

import math
from random import Random
from typing import Callable, Dict, Tuple

import patterns as P

Envelope = list
Pair = Tuple[Envelope, Envelope]


def _dur_ticks(seconds: float, tick_s: float) -> float:
    """A duration in seconds expressed in ticks (kept fractional for means)."""
    return seconds / tick_s


def _per_tick(per_second: float, tick_s: float) -> float:
    """A per-second event rate expressed as a per-tick probability."""
    return per_second * tick_s


def _walk_sigma(sigma_per_s: float, tick_s: float) -> float:
    """Per-tick step sigma that preserves diffusion (variance/second) across
    resolutions: Var over T seconds = sigma_tick^2 * (T/tick_s) is constant when
    sigma_tick = sigma_per_s * sqrt(tick_s)."""
    return sigma_per_s * math.sqrt(tick_s)


def _halton(i: int, base: int) -> float:
    """i-th value of the 1-D van der Corput / Halton low-discrepancy sequence."""
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def _log_rate(u: float, cap: float, gamma: float = 2.2) -> float:
    """Map u in [0,1] onto [0, cap] with extra density at low rates (gamma>1),
    so a uniform sequence spends more points where real apps live."""
    return cap * (max(0.0, min(1.0, u)) ** gamma)


def youtube(n: int, rng: Random, caps: Dict[str, float], tick_s: float) -> Pair:
    """Big periodic DL buffer-fill bursts + thin steady UL (acks/telemetry)."""
    dl = P.burst(
        n, rng,
        hi=caps["dl_kbps"] * rng.uniform(0.12, 0.30),  # 720p-ish fill bursts
        lo=0.0,
        on_mean=_dur_ticks(4.0, tick_s), off_mean=_dur_ticks(10.0, tick_s),
        jitter=0.5,
    )
    # deliver each fill burst as ~60ms grant bursts with ~40ms idle gaps so the
    # modem cycles in/out of DRX inside a buffer-fill instead of staying pinned on
    dl = P.duty_gate(dl, _dur_ticks(0.10, tick_s), duty=0.6)
    ul = P.random_walk(n, rng, start=120, sigma=_walk_sigma(30, tick_s), lo=40, hi=300)
    return dl, ul


def social_scroll(n: int, rng: Random, caps: Dict[str, float], tick_s: float) -> Pair:
    """Frequent shorter DL bursts (reels/feed) + occasional UL spikes."""
    dl = P.burst(
        n, rng,
        hi=caps["dl_kbps"] * rng.uniform(0.06, 0.18),
        lo=0.0,
        on_mean=_dur_ticks(2.5, tick_s), off_mean=_dur_ticks(4.0, tick_s),
        jitter=0.6,
    )
    # reels arrive frame-paced: ~50ms bursts with short idle gaps within a scroll
    dl = P.duty_gate(dl, _dur_ticks(0.08, tick_s), duty=0.55)
    ul = P.poisson_bursts(
        n, rng,
        arrival_per_tick=_per_tick(0.05, tick_s),
        burst_hi=caps["ul_kbps"] * rng.uniform(0.10, 0.30),
        burst_len_mean=_dur_ticks(2.0, tick_s),
        base=60,
    )
    return dl, ul


def web_browse(n: int, rng: Random, caps: Dict[str, float], tick_s: float) -> Pair:
    """Poisson page-load DL bursts over a low random-walk background + UL requests."""
    bg = P.random_walk(n, rng, start=200, sigma=_walk_sigma(80, tick_s),
                       lo=0, hi=caps["dl_kbps"] * 0.05)
    loads = P.poisson_bursts(
        n, rng,
        arrival_per_tick=_per_tick(0.10, tick_s),
        burst_hi=caps["dl_kbps"] * rng.uniform(0.10, 0.25),
        burst_len_mean=_dur_ticks(2.5, tick_s),
        base=0,
    )
    dl = [max(a, b) for a, b in zip(bg, loads)]
    # a page load is a sub-second flurry of parallel object fetches: chop it into
    # ~120ms request/response bursts so the fetch-then-idle micro-structure shows
    dl = P.duty_gate(dl, _dur_ticks(0.12, tick_s), duty=0.6)
    ul = P.poisson_bursts(
        n, rng,
        arrival_per_tick=_per_tick(0.12, tick_s),
        burst_hi=caps["ul_kbps"] * rng.uniform(0.05, 0.15),
        burst_len_mean=_dur_ticks(1.5, tick_s),
        base=40,
    )
    return dl, ul


def video_call(n: int, rng: Random, caps: Dict[str, float], tick_s: float) -> Pair:
    """Roughly symmetric DL+UL, sinusoidal load + random-walk jitter."""
    mid_dl = caps["dl_kbps"] * rng.uniform(0.03, 0.08)
    mid_ul = caps["ul_kbps"] * rng.uniform(0.03, 0.08)
    dl_s = P.sine(n, rng, mid=mid_dl, amp=mid_dl * 0.4,
                  period_ticks=_dur_ticks(rng.uniform(20, 45), tick_s))
    ul_s = P.sine(n, rng, mid=mid_ul, amp=mid_ul * 0.4,
                  period_ticks=_dur_ticks(rng.uniform(20, 45), tick_s))
    dl_j = P.random_walk(n, rng, start=0, sigma=_walk_sigma(mid_dl * 0.1, tick_s),
                         lo=-mid_dl * 0.3, hi=mid_dl * 0.3)
    ul_j = P.random_walk(n, rng, start=0, sigma=_walk_sigma(mid_ul * 0.1, tick_s),
                         lo=-mid_ul * 0.3, hi=mid_ul * 0.3)
    dl = [max(0.0, a + b) for a, b in zip(dl_s, dl_j)]
    ul = [max(0.0, a + b) for a, b in zip(ul_s, ul_j)]
    # real-time media is frame-paced both ways: ~30ms frame bursts (~33 fps)
    dl = P.duty_gate(dl, _dur_ticks(0.03, tick_s), duty=0.6)
    ul = P.duty_gate(ul, _dur_ticks(0.03, tick_s), duty=0.6)
    return dl, ul


def voip(n: int, rng: Random, caps: Dict[str, float], tick_s: float) -> Pair:
    """Two-way VoIP: ~64 kbps talk-spurts with DTX comfort-noise in the silences,
    packetised at 20ms. The talk/silence (VAD) structure and the 20ms cadence are
    sub-second features the old 1s/3s grid flattened into a constant trickle."""
    talk_dl = P.burst(n, rng, hi=64.0, lo=3.0,  # AMR-WB active vs DTX SID
                      on_mean=_dur_ticks(1.4, tick_s), off_mean=_dur_ticks(1.6, tick_s),
                      jitter=0.5)
    talk_ul = P.burst(n, rng, hi=64.0, lo=3.0,
                      on_mean=_dur_ticks(1.4, tick_s), off_mean=_dur_ticks(1.6, tick_s),
                      jitter=0.5)
    # No sub-datagram duty gate: at 64 kbps the pacer already emits ~1 datagram
    # per ~150 ms, so the packet cadence itself is the sub-second texture; the
    # VAD talk/silence bursts above carry the structure that matters for power.
    return talk_dl, talk_ul


def cloud_gaming(n: int, rng: Random, caps: Dict[str, float], tick_s: float) -> Pair:
    """High-rate DL video delivered at ~33ms frame cadence + a tight ~16ms UL
    control stream. Both are pure sub-second structure -- a frame-bursty downlink
    and a high-frequency uplink that only a sub-frame tick can represent."""
    base = caps["dl_kbps"] * rng.uniform(0.15, 0.35)
    dl_env = P.random_walk(n, rng, start=base, sigma=_walk_sigma(base * 0.15, tick_s),
                           lo=base * 0.5, hi=base * 1.3)
    dl = P.duty_gate(dl_env, _dur_ticks(1.0 / 30.0, tick_s), duty=0.5)   # ~33ms frames
    # UL is a thin control stream (~200 kbps); the pacer's own ~40 ms datagram
    # spacing is its cadence, so gating it below one datagram only sheds bytes.
    ul = P.steady(n, caps["ul_kbps"] * rng.uniform(0.02, 0.05))
    return dl, ul


def dl_pulse(n: int, rng: Random, caps: Dict[str, float], tick_s: float) -> Pair:
    """Calibration square wave (NOT app-realistic): DL alternates 10s at 40 Mbps
    and 10s idle, repeating; UL stays silent. A deliberately clean on/off pattern
    to read the DL-active vs idle swing directly in the power trace."""
    hold = max(1, round(10.0 / tick_s))
    dl = P.staircase(n, [40_000.0, 0.0], hold)
    ul = P.idle(n)
    return dl, ul


def ul_pulse(n: int, rng: Random, caps: Dict[str, float], tick_s: float) -> Pair:
    """Calibration square wave (NOT app-realistic): UL alternates 10s at 10 Mbps
    and 10s idle, repeating; DL stays silent. Mirror of dl_pulse for the uplink."""
    hold = max(1, round(10.0 / tick_s))
    ul = P.staircase(n, [10_000.0, 0.0], hold)
    dl = P.idle(n)
    return dl, ul


def dl_triangle(n: int, rng: Random, caps: Dict[str, float], tick_s: float) -> Pair:
    """Calibration triangle (NOT app-realistic): DL ramps linearly 0->40 Mbps
    over the first half, then symmetrically back to 0; UL stays silent. A slow,
    edge-free sweep of the DL rate axis (peak at mid-capture)."""
    half = n // 2
    dl = P.ramp(half, 0.0, 40_000.0) + P.ramp(n - half, 40_000.0, 0.0)
    ul = P.idle(n)
    return dl, ul


def ul_triangle(n: int, rng: Random, caps: Dict[str, float], tick_s: float) -> Pair:
    """Calibration triangle (NOT app-realistic): UL ramps linearly 0->10 Mbps
    over the first half, then symmetrically back to 0; DL stays silent. Mirror
    of dl_triangle for the uplink."""
    half = n // 2
    ul = P.ramp(half, 0.0, 10_000.0) + P.ramp(n - half, 10_000.0, 0.0)
    dl = P.idle(n)
    return dl, ul


def dual_sine(n: int, rng: Random, caps: Dict[str, float], tick_s: float) -> Pair:
    """Two quadrature sinusoids (NOT app-realistic): DL oscillates about mean
    20 Mbps with amplitude 20 Mbps (so it grazes 0 and 40 Mbps); UL about mean
    5 Mbps with amplitude 5 Mbps. Both have period 12s (frequency 1/12 Hz, i.e.
    10 full cycles over a 120s capture) and are 90 deg out of phase."""
    period = _dur_ticks(12.0, tick_s)
    dl = P.sine(n, rng, mid=20_000.0, amp=20_000.0, period_ticks=period, phase=0.0)
    ul = P.sine(n, rng, mid=5_000.0, amp=5_000.0, period_ticks=period,
                phase=math.pi / 2)
    return dl, ul


# --- mix profile block budget (seconds at scale=1) -------------------------
# Fixed blocks keep absolute timescales: A idle anchor (3s), D idle-gap ladder
# (~35.2s: 4 ladders x (5 on-pulses of 1s + gaps 2+1+0.5+0.2+0.1s)), and the 5s
# idle tail. The scalable blocks B,C,E,F,G share whatever duration is left,
# proportionally to these natural weights, so every block survives at any
# duration >= the floor below (where the fixed cost alone is ~43s).
_MIX_FIXED_S = 3.0 + 35.2 + 5.0                    # A anchor + D ladder + tail
_MIX_SCALABLE_S = 40.0 + 30.0 + 26.0 + 18.0 + 27.0  # B + C + E + F + G dwell
_MIX_MIN_DURATION_S = 60.0


def _mix_blocks(n: int, rng: Random, caps: Dict[str, float], tick_s: float):
    """Build the `mix` profile's labelled coverage blocks (pre-shuffle, pre-tail).

    Returns an ordered list of (name, dl, ul) for blocks A..G. Fixed blocks (A
    anchor, D idle-gap ladder) are built in absolute seconds; scalable blocks
    (B, C, E, F, G) share the remaining duration budget proportionally so every
    block is present at any duration. Raises ValueError below the 60s floor,
    where the fixed blocks alone would crowd out the scalable coverage.

    Micro-timescales stay absolute even inside scalable blocks: the ~33ms burst
    duty-gate (E), the 6-9s sine periods (F), and G's burst/arrival constants
    are tied to real time via _dur_ticks/_per_tick; only the dwell (hold length,
    ramp length, number of sine cycles, stochastic span) scales.
    """
    duration_s = n * tick_s
    if duration_s < _MIX_MIN_DURATION_S:
        raise ValueError(
            f"mix profile requires duration >= {_MIX_MIN_DURATION_S:g}s for full "
            f"coverage; got {duration_s:g}s")

    DL, UL = caps["dl_kbps"], caps["ul_kbps"]

    def T(seconds: float) -> int:                 # absolute seconds -> ticks
        return max(1, round(seconds / tick_s))

    scale = (duration_s - _MIX_FIXED_S) / _MIX_SCALABLE_S

    def S(seconds: float) -> int:                 # scalable seconds -> ticks
        return max(1, round(seconds * scale / tick_s))

    # ---- Block A: idle baseline anchor (fixed) ----
    A_dl, A_ul = P.idle(T(3.0)), P.idle(T(3.0))

    # ---- Block B: joint (DL,UL) space-filling grid, ~2s settled holds (scaled) ----
    hold = S(2.0)
    pts = [(0.0, 0.0), (DL, 0.0), (0.0, UL), (DL, UL)]            # corners (fixed)
    off = rng.randint(1, 64)
    sx, sy = rng.random(), rng.random()                          # per-seed scramble
    for k in range(16):
        ux = (_halton(off + k, 2) + sx) % 1.0
        uy = (_halton(off + k, 3) + sy) % 1.0
        pts.append((_log_rate(ux, DL), _log_rate(uy, UL)))
    rng.shuffle(pts)
    B_dl, B_ul = [], []
    for d, u in pts:
        B_dl += [d] * hold
        B_ul += [u] * hold

    # ---- Block C: continuous ramps (order/direction per seed, length scaled) ----
    seg = S(6.0)
    ramps = [
        (P.ramp(seg, 0.0, DL), P.idle(seg)),          # DL up
        (P.ramp(seg, DL, 0.0), P.idle(seg)),          # DL down
        (P.idle(seg), P.ramp(seg, 0.0, UL)),          # UL up
        (P.ramp(seg, 0.0, DL), P.ramp(seg, 0.0, UL)),  # both up (correlated)
        (P.ramp(seg, 0.0, DL), P.ramp(seg, UL, 0.0)),  # anti-correlated
    ]
    rng.shuffle(ramps)
    C_dl, C_ul = [], []
    for d, u in ramps:
        C_dl += list(d)
        C_ul += list(u)

    # ---- Block D: idle-gap ladder, fixed backbone (DRX/tail vs gap length) ----
    on = T(1.0)
    gaps = [2.0, 1.0, 0.5, 0.2, 0.1]
    D_dl, D_ul = [], []

    def ladder(level_dl: float, level_ul: float):
        dd, uu = [], []
        for g in gaps:
            dd += [level_dl] * on + [0.0] * T(g)
            uu += [level_ul] * on + [0.0] * T(g)
        return dd, uu

    for _ in range(2):                                # DL pulses, varied idle gaps
        dd, uu = ladder(0.5 * DL, 0.0)
        D_dl += dd
        D_ul += uu
    for _ in range(2):                                # UL pulses, varied idle gaps
        dd, uu = ladder(0.0, 0.5 * UL)
        D_dl += dd
        D_ul += uu

    # ---- Block E: sub-second bursty texture (dwell scaled, ~33ms gate fixed) ----
    E_dl, E_ul = [], []
    for fr in [0.10, 0.20, 0.35, 0.5]:
        lvl = DL * fr * (1 + rng.uniform(-0.1, 0.1))
        smooth = S(3.0)
        gated = P.duty_gate([lvl] * S(3.5), _dur_ticks(1.0 / 30.0, tick_s), duty=0.5)
        E_dl += [lvl] * smooth + gated
        E_ul += [UL * 0.03] * (smooth + len(gated))

    # ---- Block F: sinusoids, in-phase then quadrature (span scaled, periods fixed) ----
    half = S(9.0)
    per1 = _dur_ticks(rng.uniform(6.0, 9.0), tick_s)
    per2 = _dur_ticks(rng.uniform(6.0, 9.0), tick_s)
    base_ph = rng.uniform(0.0, math.pi)
    F_dl = (P.sine(half, rng, mid=DL * 0.3, amp=DL * 0.3, period_ticks=per1, phase=base_ph)
            + P.sine(half, rng, mid=DL * 0.3, amp=DL * 0.3, period_ticks=per2, phase=base_ph))
    F_ul = (P.sine(half, rng, mid=UL * 0.3, amp=UL * 0.3, period_ticks=per1, phase=base_ph)
            + P.sine(half, rng, mid=UL * 0.3, amp=UL * 0.3, period_ticks=per2,
                     phase=base_ph + math.pi / 2))                # quadrature half

    # ---- Block G: stochastic realistic mix (span scaled, burst constants fixed) ----
    nG = S(27.0)
    bg = P.random_walk(nG, rng, start=DL * 0.02, sigma=_walk_sigma(DL * 0.05, tick_s),
                       lo=0.0, hi=DL * 0.08)
    loads = P.poisson_bursts(nG, rng, arrival_per_tick=_per_tick(0.08, tick_s),
                             burst_hi=DL * rng.uniform(0.10, 0.30),
                             burst_len_mean=_dur_ticks(2.0, tick_s), base=0.0)
    G_dl = P.duty_gate([max(a, b) for a, b in zip(bg, loads)],
                       _dur_ticks(0.10, tick_s), duty=0.6)
    G_ul = P.poisson_bursts(nG, rng, arrival_per_tick=_per_tick(0.10, tick_s),
                            burst_hi=UL * rng.uniform(0.05, 0.20),
                            burst_len_mean=_dur_ticks(1.5, tick_s), base=UL * 0.01)

    return [("A", A_dl, A_ul), ("B", B_dl, B_ul), ("C", C_dl, C_ul),
            ("D", D_dl, D_ul), ("E", E_dl, E_ul), ("F", F_dl, F_ul),
            ("G", G_dl, G_ul)]


def mix(n: int, rng: Random, caps: Dict[str, float], tick_s: float) -> Pair:
    """Coverage schedule for TRAINING a UE power-estimation model -- not one app.

    A power model learns power = f(PHY features, recent history); power is
    non-linear in rate and has memory (DRX/tail). To generalise to real apps the
    capture must densely sample (a) the joint (DL,UL) rate plane, (b) transition
    dynamics at many timescales -- especially idle-gap length, and (c) realistic
    stochastic texture so the clean geometry is not its own distribution.

    Seven labelled blocks (see _mix_blocks): an idle anchor A, then six middle
    blocks B-G reordered per seed, then an idle tail. A deterministic spine (idle
    anchors + the idle-gap ladder + the grid corners) is identical across seeds
    for replicate/noise reference; the interior grid points, ramp order, burst
    levels, sine phases and the stochastic block vary per seed, so capturing
    mix_s1..sN multiplies coverage. Rates are log-weighted toward the low-mid
    band where real apps live, while the systematic blocks still visit the caps
    and give UL strong independent coverage (UL Tx power is a large, separate
    driver of power variance).

    The schedule scales to any requested duration >= 60s: fixed blocks keep their
    absolute timescales and the scalable blocks share the rest of the window, so
    every block is always present (no truncation). Below 60s _mix_blocks raises.
    """
    blocks = {name: (dl, ul) for name, dl, ul in _mix_blocks(n, rng, caps, tick_s)}

    # anchor A first; shuffle the middle B-G per seed; idle tail fills the rest
    middle = [blocks[k] for k in ("B", "C", "D", "E", "F", "G")]
    rng.shuffle(middle)
    dl, ul = list(blocks["A"][0]), list(blocks["A"][1])
    for d, u in middle:
        dl += d
        ul += u

    # Rounding drift is absorbed by the idle tail (budget reserves ~5s for it),
    # so no block is ever truncated. The else-branch only guards a rare overshoot.
    pad = n - len(dl)
    if pad >= 0:
        dl += [0.0] * pad
        ul += [0.0] * pad
    else:
        dl, ul = dl[:n], ul[:n]
    return dl, ul


def sweep(n: int, rng: Random, caps: Dict[str, float], tick_s: float) -> Pair:
    """NOT app-realistic. Deliberately sweeps the whole rate space so the
    model sees the full PHY range. Staircase 0->cap->0 with a random-walk
    overlay, independently on DL and UL. Steps hold ~duration/18 regardless of
    tick, so finer ticks just give cleaner edges, not faster steps."""
    hold = max(2, n // 18)
    dl_levels = [0.0, 0.05, 0.15, 0.35, 0.6, 0.85, 1.0, 0.7, 0.4, 0.1, 0.0]
    ul_levels = [0.0, 0.1, 0.3, 0.6, 1.0, 0.5, 0.2, 0.0]
    rng.shuffle(dl_levels)
    rng.shuffle(ul_levels)
    dl_stair = P.staircase(n, [caps["dl_kbps"] * x for x in dl_levels], hold)
    ul_stair = P.staircase(n, [caps["ul_kbps"] * x for x in ul_levels], hold)
    dl_w = P.random_walk(n, rng, start=0, sigma=_walk_sigma(caps["dl_kbps"] * 0.03, tick_s),
                         lo=-caps["dl_kbps"] * 0.1, hi=caps["dl_kbps"] * 0.1)
    ul_w = P.random_walk(n, rng, start=0, sigma=_walk_sigma(caps["ul_kbps"] * 0.03, tick_s),
                         lo=-caps["ul_kbps"] * 0.1, hi=caps["ul_kbps"] * 0.1)
    dl = [max(0.0, a + b) for a, b in zip(dl_stair, dl_w)]
    ul = [max(0.0, a + b) for a, b in zip(ul_stair, ul_w)]
    return dl, ul


PROFILES: Dict[str, Callable[[int, Random, Dict[str, float], float], Pair]] = {
    "youtube": youtube,
    "social_scroll": social_scroll,
    "web_browse": web_browse,
    "video_call": video_call,
    "voip": voip,
    "cloud_gaming": cloud_gaming,
    "dl_pulse": dl_pulse,
    "ul_pulse": ul_pulse,
    "dl_triangle": dl_triangle,
    "ul_triangle": ul_triangle,
    "dual_sine": dual_sine,
    "mix": mix,
    "train_mix": mix,   # back-compat alias for the former name
    "sweep": sweep,
}

"""Traffic-shaping primitives.

Each primitive returns a list of kbps values, one per tick, describing an
offered-load envelope for a single direction (DL or UL). Profiles (see
profiles.py) stitch these slices together into a full per-capture schedule.

Everything is driven by an explicit random.Random instance so that a given
(profile, seed) pair reproduces byte-identically.

RESOLUTION NOTE
---------------
Durations here are in *ticks* (on_mean, off_mean, period_ticks, burst_len_mean,
hold_ticks). At the historical 1s tick they doubled as seconds, so the iperf3
player's ~3s segment floor was never the binding constraint. The persistent-UDP
player (udp_traffic_player.py) changes rate at the tick (tens of ms), so finer
ticks are now worth using — but lowering tick_s while these params stay in
ticks silently rescales every shape (a 4-tick burst becomes 0.4s at tick=0.1).

To exploit fine resolution, migrate these to be tick-invariant:
  * take a `tick_s` argument (thread it from schedule_gen via the profile),
  * express means/periods in SECONDS and convert: ticks = seconds / tick_s,
  * draw counts with round(), keeping a >=1 tick floor.
Then a profile renders the same shape at any tick_s, and you can add genuinely
sub-second features (e.g. ~20-50ms micro-bursts for HTTP/2 object fetches, TCP
slow-start ramps, 50-100ms RTT-paced ACK trains) that 1s sampling cannot
represent. See README "Sub-second resolution" for the rollout.
"""

from __future__ import annotations

import math
from random import Random
from typing import List


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def idle(n: int) -> List[float]:
    """Silence: models DRX / app backgrounded / user reading."""
    return [0.0] * n


def steady(n: int, rate: float) -> List[float]:
    """Constant offered load: VoIP, a steady stream plateau, a flat download."""
    return [float(rate)] * n


def ramp(n: int, lo: float, hi: float) -> List[float]:
    """Linear A->B: adaptive-bitrate climb, buffer fill that tapers, etc."""
    if n <= 1:
        return [float(hi)] * n
    step = (hi - lo) / (n - 1)
    return [lo + step * i for i in range(n)]


def burst(
    n: int,
    rng: Random,
    hi: float,
    lo: float = 0.0,
    on_mean: float = 4.0,
    off_mean: float = 8.0,
    jitter: float = 0.4,
) -> List[float]:
    """On/off square wave: video buffer-fill bursts, reel/feed loads.

    on/off durations (in ticks) are drawn around their means with +/- `jitter`
    fractional spread. `hi` is itself jittered per burst so bursts are not all
    the same height.
    """
    out: List[float] = []
    while len(out) < n:
        on = max(1, round(on_mean * (1 + rng.uniform(-jitter, jitter))))
        off = max(1, round(off_mean * (1 + rng.uniform(-jitter, jitter))))
        level = hi * (1 + rng.uniform(-jitter, jitter))
        out.extend([level] * on)
        out.extend([lo] * off)
    return out[:n]


def sine(
    n: int,
    rng: Random,
    mid: float,
    amp: float,
    period_ticks: float,
    phase: float | None = None,
) -> List[float]:
    """Sinusoidal oscillation: video call scene complexity, gaming load."""
    if phase is None:
        phase = rng.uniform(0, 2 * math.pi)
    w = 2 * math.pi / max(1e-9, period_ticks)
    return [max(0.0, mid + amp * math.sin(w * i + phase)) for i in range(n)]


def random_walk(
    n: int,
    rng: Random,
    start: float,
    sigma: float,
    lo: float,
    hi: float,
) -> List[float]:
    """Bounded Gaussian random walk: general web/app unpredictability."""
    out: List[float] = []
    cur = _clamp(start, lo, hi)
    for _ in range(n):
        cur = _clamp(cur + rng.gauss(0, sigma), lo, hi)
        out.append(cur)
    return out


def poisson_bursts(
    n: int,
    rng: Random,
    arrival_per_tick: float,
    burst_hi: float,
    burst_len_mean: float,
    base: float = 0.0,
) -> List[float]:
    """Poisson-arriving bursts over a low baseline: page loads, chat traffic.

    Each tick independently starts a burst with probability `arrival_per_tick`;
    a burst lasts ~Exp(burst_len_mean) ticks at a jittered `burst_hi`.
    """
    out = [base] * n
    i = 0
    while i < n:
        if rng.random() < arrival_per_tick:
            length = max(1, round(rng.expovariate(1.0 / max(1e-9, burst_len_mean))))
            level = burst_hi * (1 + rng.uniform(-0.3, 0.3))
            for j in range(i, min(n, i + length)):
                out[j] = max(out[j], level)
            i += length
        else:
            i += 1
    return out


def staircase(
    n: int,
    levels: List[float],
    hold_ticks: int,
) -> List[float]:
    """Deterministic step sweep through `levels`, each held `hold_ticks`."""
    out: List[float] = []
    while len(out) < n:
        for lv in levels:
            out.extend([float(lv)] * hold_ticks)
            if len(out) >= n:
                break
    return out[:n]


def duty_gate(
    env: List[float],
    period_ticks: float,
    duty: float,
    preserve_mean: bool = True,
) -> List[float]:
    """Carve a smooth envelope into sub-cycle ON/idle micro-bursts.

    Models traffic delivered in bursts (video frames, scheduling grants, VoIP
    packetisation) rather than as a perfectly smooth flow: within each
    `period_ticks` window the source is ON for `duty` of the window and idle for
    the rest. With `preserve_mean` the ON level is scaled by 1/duty so the
    windowed average still equals the input -- only the *texture* changes, not the
    offered volume -- and the idle slices give the modem room to drop toward DRX
    between bursts. This is the sub-second structure a 1s/3s schedule could not
    express; it becomes meaningful only once the tick is well below the period.

    Deterministic (consumes no RNG) and resolution-adaptive: when the window is
    too coarse to carve an idle slice (period rounds to <=1 tick, or the ON span
    fills the window) the envelope passes through unchanged, so at the historical
    1s tick a sub-second gate is an exact no-op."""
    n = len(env)
    p = max(1, round(period_ticks))
    on = max(1, round(p * duty))
    if on >= p:
        return list(env)
    scale = (1.0 / duty) if preserve_mean else 1.0
    out = [0.0] * n
    i = 0
    while i < n:
        for j in range(i, min(n, i + on)):
            out[j] = env[j] * scale
        i += p
    return out

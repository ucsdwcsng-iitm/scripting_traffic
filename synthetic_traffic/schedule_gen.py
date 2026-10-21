#!/usr/bin/env python3
"""Generate a reproducible traffic schedule from a (profile, seed).

A schedule is a list of segments {t0, dur, dl_kbps, ul_kbps} that a player
realizes on the air. Consecutive ticks whose rates round to the same step are
coalesced into one segment to keep the segment count sane.

Two players consume these schedules:
  * udp_traffic_player.py (persistent UDP sockets) re-paces in place, so segments
    can be as short as one tick (tens of ms) -- use --min-seg ~= --tick.
  * traffic_player.py (iperf3) spins up a fresh handshake per segment, so it needs
    --min-seg ~3.0 or sub-2s segments under-deliver. Pass that explicitly.

Pattern/profile time-constants are in SECONDS and converted with --tick, so the
shape is the same at any resolution; lowering --tick just samples it finer.

Examples
--------
  python3 schedule_gen.py --profile youtube --seed 1 --duration 180 --plot
  python3 schedule_gen.py --profile sweep --seed 7 --out schedules/sweep_s7.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from random import Random
from typing import Dict, List

from profiles import PROFILES

# Conservative defaults for 20 MHz 2x2; override per link with --dl-cap/--ul-cap.
DEFAULT_DL_CAP = 40_000  # kbps
DEFAULT_UL_CAP = 10_000  # kbps

BLOCKS = " ▁▂▃▄▅▆▇█"


def _round_step(v: float, step: float) -> float:
    return round(v / step) * step


def build_envelopes(profile: str, seed: int, n: int, caps: Dict[str, float],
                    tick_s: float):
    if profile not in PROFILES:
        raise SystemExit(f"unknown profile {profile!r}; choices: {sorted(PROFILES)}")
    rng = Random(seed)
    dl, ul = PROFILES[profile](n, rng, caps, tick_s)
    dl = [min(caps["dl_kbps"], max(0.0, x)) for x in dl][:n]
    ul = [min(caps["ul_kbps"], max(0.0, x)) for x in ul][:n]
    # pad if a builder returned short
    dl += [0.0] * (n - len(dl))
    ul += [0.0] * (n - len(ul))
    return dl, ul


def segment(dl: List[float], ul: List[float], tick_s: float, step: float):
    """Coalesce consecutive equal-after-rounding ticks into iperf3 segments."""
    segs = []
    n = len(dl)
    i = 0
    while i < n:
        d = _round_step(dl[i], step)
        u = _round_step(ul[i], step)
        j = i + 1
        while j < n and _round_step(dl[j], step) == d and _round_step(ul[j], step) == u:
            j += 1
        segs.append({
            "t0": round(i * tick_s, 3),
            "dur": round((j - i) * tick_s, 3),
            "dl_kbps": int(d),
            "ul_kbps": int(u),
        })
        i = j
    return segs


def enforce_min_seg(segs, min_seg: float, tick_s: float):
    """Merge adjacent segments (duration-weighted average rates) until each is at
    least `min_seg` seconds. The persistent-UDP player re-paces in place and needs
    no floor (min_seg ~= tick), but the iperf3 player spins up a fresh handshake
    per segment, so over a real RAN link a sub-~2s test is dominated by setup and
    back-to-back short tests collide on the server ("busy"); pass --min-seg ~3.0
    for that path. Coalescing keeps the envelope but enforces the floor."""
    if min_seg <= tick_s:
        return segs

    def merge(a, b):
        tot = a["dur"] + b["dur"]
        return {
            "t0": a["t0"],
            "dur": round(tot, 3),
            "dl_kbps": int(round((a["dl_kbps"] * a["dur"] + b["dl_kbps"] * b["dur"]) / tot)),
            "ul_kbps": int(round((a["ul_kbps"] * a["dur"] + b["ul_kbps"] * b["dur"]) / tot)),
        }

    out = []
    i = 0
    while i < len(segs):
        cur = dict(segs[i]); i += 1
        while cur["dur"] < min_seg and i < len(segs):
            cur = merge(cur, segs[i]); i += 1
        out.append(cur)
    # a trailing short remainder folds back into its predecessor
    if len(out) >= 2 and out[-1]["dur"] < min_seg:
        tail = out.pop()
        out[-1] = merge(out[-1], tail)
    t = 0.0
    for s in out:
        s["t0"] = round(t, 3); t += s["dur"]
    return out


def sparkline(vals: List[float], cap: float, width: int = 90) -> str:
    if not vals:
        return ""
    # downsample to width columns by max-pooling
    cols = []
    per = max(1, len(vals) // width)
    for k in range(0, len(vals), per):
        chunk = vals[k:k + per]
        cols.append(max(chunk))
    out = []
    for v in cols:
        idx = 0 if cap <= 0 else min(len(BLOCKS) - 1, int(v / cap * (len(BLOCKS) - 1)))
        out.append(BLOCKS[idx])
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True, choices=sorted(PROFILES))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--duration", type=int, default=180, help="capture seconds")
    ap.add_argument("--tick", type=float, default=0.01,
                    help="envelope tick seconds (sampling resolution; 10ms default "
                         "matches testbed/capture.sh / the persistent-UDP player)")
    ap.add_argument("--dl-cap", type=int, default=DEFAULT_DL_CAP, help="kbps ceiling")
    ap.add_argument("--ul-cap", type=int, default=DEFAULT_UL_CAP, help="kbps ceiling")
    ap.add_argument("--step", type=float, default=25.0,
                    help="kbps rounding granularity for segment coalescing. Fine by "
                         "default so low-rate profiles (voip ~64k) survive; the "
                         "persistent-UDP player has no per-segment cost. Pass 250 to "
                         "reproduce the legacy coarse coalescing")
    ap.add_argument("--min-seg", type=float, default=0.01,
                    help="minimum segment seconds (merges shorter ones). 10ms default "
                         "matches the tick; pass ~3.0 for the iperf3 player")
    ap.add_argument("--out", help="write schedule JSON here (default: stdout summary)")
    ap.add_argument("--pdf", nargs="?", const=True, default=None,
                    help="also render a DL/UL PDF (path optional; defaults next to --out)")
    ap.add_argument("--plot", action="store_true", help="print ASCII DL/UL preview")
    args = ap.parse_args()

    n = int(round(args.duration / args.tick))
    caps = {"dl_kbps": float(args.dl_cap), "ul_kbps": float(args.ul_cap)}
    dl, ul = build_envelopes(args.profile, args.seed, n, caps, args.tick)
    segs = segment(dl, ul, args.tick, args.step)
    segs = enforce_min_seg(segs, args.min_seg, args.tick)

    schedule = {
        "profile": args.profile,
        "seed": args.seed,
        "duration_s": args.duration,
        "tick_s": args.tick,
        "caps": {"dl_kbps": args.dl_cap, "ul_kbps": args.ul_cap},
        "server": {"host": "10.45.0.1", "dl_port": 5301, "ul_port": 5302},
        "segments": segs,
    }
    body = json.dumps(schedule["segments"], sort_keys=True).encode()
    schedule["schedule_hash"] = hashlib.sha256(body).hexdigest()[:16]

    if args.plot:
        print(f"profile={args.profile} seed={args.seed} dur={args.duration}s "
              f"segments={len(segs)} hash={schedule['schedule_hash']}")
        print(f"DL [0..{args.dl_cap} kbps]  {sparkline(dl, caps['dl_kbps'])}")
        print(f"UL [0..{args.ul_cap} kbps]  {sparkline(ul, caps['ul_kbps'])}")
        print(f"DL mean={sum(dl)/n:8.0f}  max={max(dl):8.0f} kbps   "
              f"UL mean={sum(ul)/n:8.0f}  max={max(ul):8.0f} kbps")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(schedule, f, indent=2)
        print(f"wrote {args.out}  ({len(segs)} segments, hash {schedule['schedule_hash']})")
    elif not args.plot:
        print(json.dumps(schedule, indent=2))

    if args.pdf is not None:
        if not args.out and args.pdf is True:
            raise SystemExit("--pdf needs a path when --out is not given")
        pdf_path = args.pdf if isinstance(args.pdf, str) else \
            os.path.splitext(args.out)[0] + ".pdf"
        import plot_schedule
        # render() re-reads the JSON; ensure it exists when only --pdf was passed
        if not args.out:
            tmp = os.path.splitext(pdf_path)[0] + ".json"
            with open(tmp, "w") as f:
                json.dump(schedule, f, indent=2)
            plot_schedule.render(tmp, pdf_path)
        else:
            plot_schedule.render(args.out, pdf_path)
        print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()

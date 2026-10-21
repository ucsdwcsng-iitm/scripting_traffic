#!/usr/bin/env python3
"""Realize a traffic schedule with persistent UDP sockets (UE side).

Drop-in alternative to traffic_player.py with the same CLI and the same
achieved-vs-target results schema, but without iperf3's per-segment client/server
handshake. One socket per direction stays open for the whole capture and the
offered load is re-paced in place, so rate changes track the schedule tick
(tens of ms) instead of iperf3's ~3s setup/teardown floor.

Pairs with udp_traffic_server.py on the DN host (10.45.0.1). Per capture:
  * UL: this process paces UDP to the server (UE -> server = uplink load),
  * DL: the server paces UDP back to us (server -> UE = downlink load); we sink
        it and count received bytes,
  * a single TCP control channel does clock sync, schedule upload, and collects
    the server's UL report at the end.
Zero-rate directions send nothing, preserving DRX silence on the air.

Opens raw sockets on wwan0, so it must run inside the UE namespace; with --ns it
re-execs itself via `ip netns exec` (needs root), matching traffic_player.py's
invocation. Run under sudo:
  sudo python3 udp_traffic_player.py --schedule schedules/sweep_s1.json
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from typing import List

import udp_traffic_common as C

_NS_SENTINEL = "UDP_TRAFFIC_IN_NS"


def maybe_reexec_in_ns(ns: str) -> None:
    """Re-exec this process inside network namespace `ns` (idempotent)."""
    if not ns or os.environ.get(_NS_SENTINEL) == ns:
        return
    if os.geteuid() != 0:
        sys.exit("Run with sudo: entering a network namespace needs root.")
    os.environ[_NS_SENTINEL] = ns
    os.execvp("ip", ["ip", "netns", "exec", ns, sys.executable, *sys.argv])


def play(args, sched) -> int:
    segs: List[dict] = sched["segments"]
    server = sched.get("server", {})
    host = args.host or server.get("host", "10.45.0.1")
    dl_port = args.dl_port or server.get("dl_port", C.DL_PORT)
    ul_port = args.ul_port or server.get("ul_port", C.UL_PORT)
    ctrl_port = args.ctrl_port
    dgram_len = args.dgram_len

    # 1) control channel + clock sync
    ctrl = socket.create_connection((host, ctrl_port), timeout=10)
    ctrl.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    offset = C.sync_offset_client(ctrl)   # server_mono = ue_mono + offset

    # 2) agree a start time ~1s out, expressed in both clocks
    go_ue = time.monotonic() + 1.0
    go_srv = go_ue + offset
    C.send_msg(ctrl, {"type": "start", "segments": segs, "dgram_len": dgram_len,
                      "dl_port": dl_port, "ul_port": ul_port, "go_srv_mono": go_srv})
    ack = C.recv_msg(ctrl)
    if not ack or ack.get("type") != "started":
        sys.exit("server did not arm the session")

    # 3) DL sink (server -> UE) + UL pacer (UE -> server)
    dl_sock = C.open_udp_recv("", dl_port)
    sink = C.ByteSink(dl_sock)
    threading.Thread(target=sink.run, daemon=True).start()

    ul_track = C.RateTrack(segs, "ul_kbps")
    ul_sock = C.open_udp_send()
    stop = {"v": False}

    def ul_pacer():
        C.pace_direction(ul_sock, (host, ul_port), ul_track.at, dgram_len,
                         time.monotonic, go_ue, ul_track.total, lambda: stop["v"])

    ul_thr = threading.Thread(target=ul_pacer, daemon=True)
    slack = go_ue - time.monotonic()
    if slack > 0:
        time.sleep(slack)
    ul_thr.start()

    # 4) snapshot the DL counter at each segment boundary (synced clock)
    snaps = [sink.bytes]
    for seg in segs:
        boundary = go_ue + float(seg["t0"]) + float(seg["dur"])
        slack = boundary - time.monotonic()
        if slack > 0:
            time.sleep(slack)
        snaps.append(sink.bytes)

    stop["v"] = True
    ul_thr.join(timeout=2.0)
    sink.stop()
    ul_sock.close()
    dl_sock.close()

    dl_mbps = C.achieved_series(snaps, segs)
    report = C.recv_msg(ctrl)
    ctrl.close()
    ul_mbps = (report or {}).get("ul_achieved_mbps", [None] * len(segs))

    records = []
    for i, seg in enumerate(segs):
        rec = {"t0": seg["t0"], "dur": float(seg["dur"]),
               "dl_target_kbps": float(seg["dl_kbps"]),
               "ul_target_kbps": float(seg["ul_kbps"]),
               "dl_achieved_mbps": dl_mbps[i],
               "ul_achieved_mbps": ul_mbps[i] if i < len(ul_mbps) else None}
        records.append(rec)
        if args.verbose:
            print(f"  t0={seg['t0']:>6} dur={rec['dur']:g}s  "
                  f"DL {seg['dl_kbps']/1000:5.1f}->{(rec['dl_achieved_mbps'] or 0):5.1f} Mbps  "
                  f"UL {seg['ul_kbps']/1000:5.1f}->{(rec['ul_achieved_mbps'] or 0):5.1f} Mbps")
    return _finish(args, sched, records)


def _finish(args, sched, records) -> int:
    if args.results:
        summary = {
            "schedule": args.schedule,
            "profile": sched.get("profile"),
            "seed": sched.get("seed"),
            "schedule_hash": sched.get("schedule_hash"),
            "transport": "udp-persistent",
            "segments": records,
        }
        with open(args.results, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote results: {args.results}")
    return 0


def dry_run(args, sched) -> int:
    segs = sched["segments"]
    print(f"[dry-run] {args.schedule}: profile={sched.get('profile')} "
          f"segments={len(segs)} (udp-persistent)")
    if args.verbose:
        for seg in segs:
            print(f"  [dry-run] t0={seg['t0']:>6} dur={float(seg['dur']):g}s "
                  f"DL={seg['dl_kbps']:.0f}k UL={seg['ul_kbps']:.0f}k")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schedule", required=True, help="schedule JSON from schedule_gen.py")
    ap.add_argument("--ns", default="ue-traffic", help="network namespace ('' for none)")
    ap.add_argument("--host", help="override server host (default: schedule's server.host)")
    ap.add_argument("--dl-port", type=int, help="override DL port")
    ap.add_argument("--ul-port", type=int, help="override UL port")
    ap.add_argument("--ctrl-port", type=int, default=C.CTRL_PORT, help="TCP control port")
    ap.add_argument("--dgram-len", type=int, default=1200, help="UDP payload bytes (< MTU)")
    ap.add_argument("--results", help="write achieved-vs-target JSON here")
    ap.add_argument("--dry-run", action="store_true", help="print plan, send nothing")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not args.dry_run:
        maybe_reexec_in_ns(args.ns or "")

    with open(args.schedule) as f:
        sched = json.load(f)

    segs = sched["segments"]
    print(f"Playing {args.schedule}: profile={sched.get('profile')} "
          f"seed={sched.get('seed')} dur={sched.get('duration_s')}s "
          f"segments={len(segs)} ns={args.ns or 'none'} "
          f"transport=udp-persistent{' [dry-run]' if args.dry_run else ''}")

    t0 = time.monotonic()
    rc = dry_run(args, sched) if args.dry_run else play(args, sched)
    print(f"Done. wall={time.monotonic() - t0:.1f}s (schedule={sched.get('duration_s')}s)")
    return rc


if __name__ == "__main__":
    sys.exit(main())

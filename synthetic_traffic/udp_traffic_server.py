#!/usr/bin/env python3
"""Data-network endpoint for the persistent-socket UDP traffic generator.

Long-running daemon, the UDP analogue of the iperf3 DN servers in
iperf-dn-server.sh. It runs in the root namespace on the Open5GS core host and,
per capture session driven by udp_traffic_player.py on the UE:

  * DL: paces UDP datagrams *to the UE* (server -> UE = downlink load), following
    the uploaded schedule's dl_kbps against a clock synced with the UE.
  * UL: drains UDP from the UE on UL_PORT and counts received bytes, snapshotting
    per segment so the achieved uplink rate can be reported back to the UE.

One TCP control connection per session carries the clock sync, the schedule
upload, and the final UL report; it is opened once and held for the whole
capture, so there is no per-segment setup/teardown (that is the whole point).

Run on the host that owns ogstun (10.45.0.1):
  sudo python3 udp_traffic_server.py            # binds 10.45.0.1
  sudo python3 udp_traffic_server.py --bind 0.0.0.0 --verbose
"""

from __future__ import annotations

import argparse
import socket
import threading
import time
from typing import List

import udp_traffic_common as C


def serve_session(conn: socket.socket, peer, bind_ip: str, verbose: bool) -> None:
    ue_ip = peer[0]

    # 1) clock sync + schedule upload, interleaved on the control channel
    schedule = None
    while True:
        msg = C.recv_msg(conn)
        if msg is None:
            return
        if msg.get("type") == "sync":
            C.sync_serve_one(conn, msg)
        elif msg.get("type") == "start":
            schedule = msg
            break
        else:
            return

    segments: List[dict] = schedule["segments"]
    dgram_len = int(schedule["dgram_len"])
    dl_port = int(schedule.get("dl_port", C.DL_PORT))
    ul_port = int(schedule.get("ul_port", C.UL_PORT))
    go_srv = float(schedule["go_srv_mono"])   # start time already in our clock

    if verbose:
        total = segments[-1]["t0"] + segments[-1]["dur"] if segments else 0
        print(f"[udp-dn] session UE={ue_ip} segs={len(segments)} dur={total:g}s "
              f"len={dgram_len} DL->{ue_ip}:{dl_port} UL<-:{ul_port}")

    # 2) UL sink: drain UE uplink, count bytes
    ul_sock = C.open_udp_recv(bind_ip, ul_port)
    sink = C.ByteSink(ul_sock)
    sink_thr = threading.Thread(target=sink.run, daemon=True)
    sink_thr.start()

    # 3) DL pacer thread: server -> UE
    dl_track = C.RateTrack(segments, "dl_kbps")
    dl_sock = C.open_udp_send()
    stop = {"v": False}

    def dl_pacer():
        C.pace_direction(
            dl_sock, (ue_ip, dl_port), dl_track.at, dgram_len,
            time.monotonic, go_srv, dl_track.total, lambda: stop["v"])

    dl_thr = threading.Thread(target=dl_pacer, daemon=True)

    # tell the UE we are armed, then both sides start at go_srv (synced clock)
    C.send_msg(conn, {"type": "started"})
    slack = go_srv - time.monotonic()
    if slack > 0:
        time.sleep(slack)
    dl_thr.start()

    # 4) snapshot the UL counter at each segment boundary on the synced clock
    snaps = [sink.bytes]
    for seg in segments:
        boundary = go_srv + float(seg["t0"]) + float(seg["dur"])
        slack = boundary - time.monotonic()
        if slack > 0:
            time.sleep(slack)
        snaps.append(sink.bytes)

    stop["v"] = True
    dl_thr.join(timeout=2.0)
    sink.stop()
    dl_sock.close()
    ul_sock.close()

    ul_mbps = C.achieved_series(snaps, segments)
    C.send_msg(conn, {"type": "report", "ul_achieved_mbps": ul_mbps})
    if verbose:
        tot = (snaps[-1] - snaps[0]) * 8 / 1e6
        print(f"[udp-dn] session done UE={ue_ip} UL received {tot:.1f} Mb")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bind", default="10.45.0.1", help="address to bind (ogstun gateway)")
    ap.add_argument("--ctrl-port", type=int, default=C.CTRL_PORT)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind((args.bind, args.ctrl_port))
    lsock.listen(4)
    print(f"[udp-dn] control listening on {args.bind}:{args.ctrl_port} "
          f"(DL{C.DL_PORT}/UL{C.UL_PORT} UDP)")

    try:
        while True:
            conn, peer = lsock.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                serve_session(conn, peer, args.bind, args.verbose)
            except (OSError, KeyError, ValueError) as e:
                print(f"[udp-dn] session error from {peer}: {e}")
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\n[udp-dn] shutting down")
    finally:
        lsock.close()


if __name__ == "__main__":
    main()

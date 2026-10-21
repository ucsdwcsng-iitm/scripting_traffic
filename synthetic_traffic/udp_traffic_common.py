"""Shared plumbing for the persistent-socket UDP traffic generator.

The iperf3 player (traffic_player.py) spins up a fresh client/server handshake
per segment, so sub-~2s segments are dominated by setup/teardown and the
effective rate-change resolution is ~3s. This module backs a replacement that
keeps ONE socket open per direction for the whole capture and changes the
offered load by re-pacing in place, so the resolution is just the schedule tick
(tens of ms) instead of the iperf3 process lifecycle.

Pieces shared by udp_traffic_player.py (UE side) and udp_traffic_server.py
(DN side):
  * length-prefixed JSON messages over a persistent TCP control channel,
  * an NTP-style clock handshake so both ends pace the same schedule against a
    common clock (DL boundaries on the server line up with UL boundaries on the
    UE to within ~1ms instead of one radio RTT),
  * a token-bucket pacer that turns an instantaneous kbps target into UDP
    datagrams, and is silent (sends nothing) whenever the target is 0 so DRX
    gaps stay truly idle on the air.
"""

from __future__ import annotations

import json
import socket
import struct
import time
from typing import Callable, Dict, List, Optional, Tuple

# Default control/data ports. DL/UL match the iperf3 server (iperf-dn-server.sh)
# so the rest of the pipeline (firewall, docs) is unchanged; CTRL is new.
CTRL_PORT = 5300   # TCP: sync handshake, schedule upload, final UL report
DL_PORT = 5301     # UDP: server -> UE  (downlink load)
UL_PORT = 5302     # UDP: UE -> server  (uplink load)

# Pacer wakes every PACE_SLICE_S to emit the datagrams owed since the last wake.
# 1ms keeps even multi-Mbps rates smooth without burning a core. Accumulated
# credit is capped at MAX_BURST_S worth so a scheduler hiccup can't dump a huge
# catch-up burst on the air.
PACE_SLICE_S = 0.001
MAX_BURST_S = 0.05


# --------------------------------------------------------------------------- #
# Length-prefixed JSON framing over TCP
# --------------------------------------------------------------------------- #
def send_msg(sock: socket.socket, obj: dict) -> None:
    body = json.dumps(obj).encode()
    sock.sendall(struct.pack(">I", len(body)) + body)


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock: socket.socket) -> Optional[dict]:
    head = _recv_exact(sock, 4)
    if head is None:
        return None
    (length,) = struct.unpack(">I", head)
    body = _recv_exact(sock, length)
    if body is None:
        return None
    return json.loads(body.decode())


# --------------------------------------------------------------------------- #
# Clock sync: map the peer's time.monotonic() onto ours
# --------------------------------------------------------------------------- #
def sync_offset_client(sock: socket.socket, rounds: int = 7) -> float:
    """Estimate `offset` such that  server_monotonic = ue_monotonic + offset.

    NTP-style: send t1, server stamps t2 (recv) and t3 (send), we stamp t4.
    Keep the sample with the smallest round-trip (least queueing noise).
    """
    best_rtt = float("inf")
    best_off = 0.0
    for _ in range(rounds):
        t1 = time.monotonic()
        send_msg(sock, {"type": "sync", "t1": t1})
        reply = recv_msg(sock)
        t4 = time.monotonic()
        if not reply or reply.get("type") != "sync_ack":
            raise RuntimeError("sync handshake failed")
        t2, t3 = reply["t2"], reply["t3"]
        rtt = (t4 - t1) - (t3 - t2)
        if rtt < best_rtt:
            best_rtt = rtt
            best_off = ((t2 - t1) + (t3 - t4)) / 2.0
    return best_off


def sync_serve_one(sock: socket.socket, msg: dict) -> None:
    """Answer a single {"type":"sync"} probe from the client."""
    t2 = time.monotonic()
    send_msg(sock, {"type": "sync_ack", "t1": msg["t1"], "t2": t2,
                    "t3": time.monotonic()})


# --------------------------------------------------------------------------- #
# Schedule lookup: instantaneous kbps for a direction at a given schedule time
# --------------------------------------------------------------------------- #
class RateTrack:
    """Step function kbps(t) over a schedule's segments for one direction."""

    def __init__(self, segments: List[dict], key: str):
        self._bounds = []   # (t_end, kbps)
        for seg in segments:
            t_end = float(seg["t0"]) + float(seg["dur"])
            self._bounds.append((t_end, float(seg[key])))
        self.total = self._bounds[-1][0] if self._bounds else 0.0

    def at(self, t: float) -> float:
        for t_end, kbps in self._bounds:
            if t < t_end:
                return kbps
        return 0.0


# --------------------------------------------------------------------------- #
# Token-bucket UDP pacer
# --------------------------------------------------------------------------- #
def pace_direction(
    sock: socket.socket,
    dst: Tuple[str, int],
    rate_at: Callable[[float], float],
    dgram_len: int,
    now_fn: Callable[[], float],
    t_start: float,
    duration: float,
    stop_flag: Callable[[], bool],
) -> None:
    """Send UDP datagrams to `dst` so the byte rate tracks rate_at(schedule_t),
    where schedule_t = now_fn() - t_start. Sends nothing while the target is 0
    (true silence for DRX gaps). Returns when the schedule duration elapses."""
    payload = bytes(dgram_len)
    tokens = 0.0           # bytes we are owed but have not yet sent
    last = now_fn()
    while not stop_flag():
        now = now_fn()
        sched_t = now - t_start
        if sched_t >= duration:
            break
        kbps = rate_at(sched_t)
        bytes_per_s = kbps * 1000.0 / 8.0
        tokens += bytes_per_s * (now - last)
        last = now
        # Clamp the bucket BEFORE emitting. While active the ceiling is a short
        # burst of credit (MAX_BURST_S) but never below one datagram, so a
        # low-rate duty-gated flow can still accumulate a whole datagram across
        # its idle micro-gaps. While idle the ceiling is exactly one datagram:
        # that lets a sub-datagram credit carry across a gate gap, but stops a
        # high-rate->idle transition from dumping a catch-up burst into a DRX gap
        # (at most one in-flight datagram trails into the silence).
        if bytes_per_s > 0.0:
            cap = bytes_per_s * MAX_BURST_S
            if cap < dgram_len:
                cap = float(dgram_len)
        else:
            cap = float(dgram_len)
        if tokens > cap:
            tokens = cap
        emitted = 0
        while tokens >= dgram_len and emitted < 4096:
            sock.sendto(payload, dst)
            tokens -= dgram_len
            emitted += 1
        time.sleep(PACE_SLICE_S)


# --------------------------------------------------------------------------- #
# UDP sink: drain a socket and keep a running received-byte counter
# --------------------------------------------------------------------------- #
class ByteSink:
    """Background-drained UDP receiver exposing a monotonically growing byte
    count, sampled at segment boundaries to derive achieved throughput."""

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self.bytes = 0
        self._stop = False

    def run(self) -> None:
        buf = bytearray(65535)
        self._sock.settimeout(0.5)
        while not self._stop:
            try:
                n = self._sock.recv_into(buf)
            except socket.timeout:
                continue
            except OSError:
                break
            if n:
                self.bytes += n

    def stop(self) -> None:
        self._stop = True


def achieved_series(snapshots: List[int], segments: List[dict]) -> List[Optional[float]]:
    """Per-segment Mbps from cumulative byte snapshots taken at each boundary.
    snapshots[i] is the counter sampled at the start of segment i, with one
    extra trailing sample at the end of the last segment."""
    out: List[Optional[float]] = []
    for i, seg in enumerate(segments):
        dur = float(seg["dur"])
        if dur <= 0 or i + 1 >= len(snapshots):
            out.append(None)
            continue
        delta_bytes = snapshots[i + 1] - snapshots[i]
        out.append(delta_bytes * 8.0 / dur / 1e6)
    return out


def open_udp_recv(bind_ip: str, port: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
    s.bind((bind_ip, port))
    return s


def open_udp_send() -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)
    return s

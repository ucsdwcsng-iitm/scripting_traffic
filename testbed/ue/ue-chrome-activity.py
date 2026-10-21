#!/usr/bin/env python3
"""Drive an already-running Chrome over the DevTools (CDP) endpoint to imitate
real app usage for a UE power-estimation capture.

The Chrome instance is the one capture.sh / ue-chrome-agent.sh starts inside the
ue-browser network namespace, exposing CDP on 127.0.0.1:9223. capture.sh invokes
this script for the duration of a capture (the real-app analogue of the
synthetic traffic player). It opens one page target via the CDP HTTP endpoint
(same `PUT /json/new` the YouTube path already uses), then *drives* it -- scroll
to the next reel, scroll the feed -- because TikTok/Instagram/X do not generate
representative traffic from a static autoplaying tab the way a YouTube watch page
does.

Activities:
  youtube    open a watch URL, ensure playback, dwell (no scrolling)
  instagram  open instagram.com/reels, advance reels on a seeded cadence
  x          open x.com home/explore, scroll the timeline on a seeded cadence
  slither    open slither.io, type a nickname and start, then steer the snake
             with the cursor and boost on a seeded cadence. A real-time .io game:
             a high, sustained bidirectional update rate, no login.

Reproducibility model mirrors synthetic_traffic: a --seed fixes every dwell/scroll
draw, so (activity, seed) gives the same input *cadence* run-to-run. The delivered
bytes still vary (live feeds), so this is "approximately reproducible" -- the
structure is pinned, the content is not.

No third-party deps: a minimal CDP client (HTTP target list + RFC6455 websocket)
is implemented here in stdlib only, because the capture hosts do not carry
websocket-client/websockets.

Login: Instagram and X gate their feeds behind auth. This script does NOT log in;
it expects the persistent Chrome profile (CHROME_PROFILE_DIR) to already hold a
logged-in session. If it detects a login wall it fails loudly (unless
--allow-login-wall), so a capture never silently records a login page instead of
the app.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import struct
import sys
import time
import urllib.parse
import urllib.request
from random import Random


def log(msg: str) -> None:
    print(f"[ue-chrome-activity] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Minimal CDP client (HTTP endpoint + a single-target websocket), stdlib only. #
# --------------------------------------------------------------------------- #
class CDPError(RuntimeError):
    pass


class CDPSession:
    """One websocket attached to one page target. Commands are synchronous;
    asynchronous CDP events are queued and can be drained while idle."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._id = 0
        self._sock: socket.socket | None = None
        self._buf = b""
        self.events: list[dict] = []

    # ---- HTTP side (target creation) ---- #
    def _http(self, method: str, path: str) -> object:
        url = f"http://{self.host}:{self.port}{path}"
        req = urllib.request.Request(url, method=method)
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", "replace").strip()
        return json.loads(body) if body else None

    def open_target(self, url: str) -> str:
        """Create a new page target for `url`; return its websocket debugger URL."""
        target = self._http("PUT", f"/json/new?{urllib.parse.quote(url, safe='')}")
        if not isinstance(target, dict) or "webSocketDebuggerUrl" not in target:
            raise CDPError(f"CDP did not return a page target for {url!r}: {target!r}")
        return target["webSocketDebuggerUrl"]

    def list_targets(self) -> list:
        res = self._http("GET", "/json")
        return res if isinstance(res, list) else []

    def open_or_reuse(self, url: str, host: str) -> bool:
        """Reuse an existing page tab already on `host` (so a login probe lands in
        the same tab the user logs into) if one exists; otherwise open a new tab
        at `url`. Returns True if an existing tab was reused. A reused tab is left
        on its current page (no navigate) so an in-progress login is not reloaded."""
        for t in self.list_targets():
            if t.get("type") == "page" and host in (t.get("url") or ""):
                self.connect(t["webSocketDebuggerUrl"])
                return True
        self.connect(self.open_target(url))
        return False

    # ---- websocket handshake + framing (RFC6455, client side) ---- #
    def connect(self, ws_url: str) -> None:
        # ws://host:port/devtools/page/<id>
        assert ws_url.startswith("ws://"), ws_url
        rest = ws_url[len("ws://"):]
        netloc, _, path = rest.partition("/")
        host, _, port = netloc.partition(":")
        path = "/" + path
        sock = socket.create_connection((host, int(port or 80)), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode()
        handshake = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        sock.sendall(handshake.encode())
        # read response headers
        data = b""
        sock.settimeout(10)
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                raise CDPError("websocket handshake closed early")
            data += chunk
        if b" 101 " not in data.split(b"\r\n", 1)[0]:
            raise CDPError(f"websocket upgrade failed: {data[:120]!r}")
        self._buf = data.split(b"\r\n\r\n", 1)[1]
        self._sock = sock

    def _send_frame(self, payload: str) -> None:
        data = payload.encode("utf-8")
        header = bytearray([0x81])  # FIN + text opcode
        n = len(data)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header += struct.pack(">H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", n)
        mask = os.urandom(4)
        header += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        assert self._sock is not None
        self._sock.sendall(bytes(header) + masked)

    def _read_n(self, n: int, deadline: float) -> bytes | None:
        assert self._sock is not None
        while len(self._buf) < n:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                return None
            self._sock.settimeout(timeout)
            try:
                chunk = self._sock.recv(65536)
            except socket.timeout:
                return None
            if not chunk:
                raise CDPError("websocket closed")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def _recv_frame(self, deadline: float) -> tuple[int, bytes] | None:
        head = self._read_n(2, deadline)
        if head is None:
            return None
        opcode = head[0] & 0x0F
        length = head[1] & 0x7F
        if length == 126:
            ext = self._read_n(2, deadline)
            if ext is None:
                return None
            length = struct.unpack(">H", ext)[0]
        elif length == 127:
            ext = self._read_n(8, deadline)
            if ext is None:
                return None
            length = struct.unpack(">Q", ext)[0]
        payload = self._read_n(length, deadline) if length else b""
        if payload is None:
            return None
        return opcode, payload

    # ---- command / event loop ---- #
    def call(self, method: str, params: dict | None = None, timeout: float = 15.0) -> dict:
        self._id += 1
        mid = self._id
        self._send_frame(json.dumps({"id": mid, "method": method, "params": params or {}}))
        deadline = time.monotonic() + timeout
        while True:
            frame = self._recv_frame(deadline)
            if frame is None:
                raise CDPError(f"timeout waiting for {method} response")
            opcode, payload = frame
            if opcode == 0x8:
                raise CDPError("websocket close frame")
            if opcode == 0x9:  # ping -> pong
                self._send_pong(payload)
                continue
            if opcode != 0x1:
                continue
            msg = json.loads(payload)
            if msg.get("id") == mid:
                if "error" in msg:
                    raise CDPError(f"{method} error: {msg['error']}")
                return msg.get("result", {})
            if "method" in msg:
                self.events.append(msg)

    def _send_pong(self, payload: bytes) -> None:
        header = bytearray([0x8A, 0x80 | len(payload)])
        mask = os.urandom(4)
        header += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        assert self._sock is not None
        self._sock.sendall(bytes(header) + masked)

    def pump(self, seconds: float) -> None:
        """Read and queue events for `seconds` (keeps Network.* flowing while the
        page is left to play/scroll). Responds to pings; ignores nothing."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            frame = self._recv_frame(deadline)
            if frame is None:
                break
            opcode, payload = frame
            if opcode == 0x8:
                raise CDPError("websocket close frame")
            if opcode == 0x9:
                self._send_pong(payload)
                continue
            if opcode != 0x1:
                continue
            msg = json.loads(payload)
            if "method" in msg:
                self.events.append(msg)

    def eval(self, expr: str) -> object:
        res = self.call(
            "Runtime.evaluate",
            {"expression": expr, "returnByValue": True, "awaitPromise": True},
        )
        return res.get("result", {}).get("value")

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None


# --------------------------------------------------------------------------- #
# Input helpers                                                                #
# --------------------------------------------------------------------------- #
def press_key(cdp: CDPSession, key: str, code: str, vk: int) -> None:
    for kind in ("keyDown", "keyUp"):
        cdp.call(
            "Input.dispatchKeyEvent",
            {"type": kind, "key": key, "code": code, "windowsVirtualKeyCode": vk,
             "nativeVirtualKeyCode": vk},
        )


def wheel_scroll(cdp: CDPSession, dy: int, x: int = 400, y: int = 400) -> None:
    cdp.call(
        "Input.dispatchMouseEvent",
        {"type": "mouseWheel", "x": x, "y": y, "deltaX": 0, "deltaY": dy},
    )


def move_mouse(cdp: CDPSession, x: int, y: int) -> None:
    cdp.call("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y})


def click(cdp: CDPSession, x: int, y: int) -> None:
    for kind in ("mousePressed", "mouseReleased"):
        cdp.call(
            "Input.dispatchMouseEvent",
            {"type": kind, "x": x, "y": y, "button": "left", "clickCount": 1},
        )


def boost(cdp: CDPSession, x: int, y: int, hold: float) -> None:
    """Press-hold-release the left button at (x, y) -- slither.io's speed boost,
    which raises the update rate (and traffic) while held."""
    cdp.call("Input.dispatchMouseEvent",
             {"type": "mousePressed", "x": x, "y": y, "button": "left", "clickCount": 1})
    cdp.pump(hold)
    cdp.call("Input.dispatchMouseEvent",
             {"type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1})


# --------------------------------------------------------------------------- #
# Login-wall / playback detection                                             #
# --------------------------------------------------------------------------- #
LOGIN_MARKERS_JS = r"""
(() => {
  const url = location.href.toLowerCase();
  const txt = (document.body ? document.body.innerText : '').slice(0, 6000).toLowerCase();
  const urlGate = /\/(login|signup|accounts\/login|i\/flow\/login)/.test(url);
  // logged-out landing copy for IG and X
  const textGate = (
    txt.includes('log in to') || txt.includes('sign in to')
    || (txt.includes('sign up') && txt.includes('log in') && txt.length < 1200)
    || txt.includes('new to x?') || txt.includes('happening now')
    || txt.includes("don't miss what's happening")
    || txt.includes('see more on instagram')
  );
  // real feed content rendered? (X tweets / IG reel videos)
  const articles = document.querySelectorAll('article, [role="article"]').length;
  const videos = document.querySelectorAll('video').length;
  // a logged-in X home shows many <article> tweets; logged-out landing shows ~0
  const feedItems = articles;
  return JSON.stringify({url: location.href, urlGate, textGate,
    hasVideo: videos > 0, feedItems, articles, videos,
    videoPlaying: !!(document.querySelector('video') && !document.querySelector('video').paused),
    bodyLen: (document.body ? document.body.innerText.length : 0)});
})()
"""


def page_state(cdp: CDPSession) -> dict:
    raw = cdp.eval(LOGIN_MARKERS_JS)
    try:
        return json.loads(raw) if isinstance(raw, str) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


# --------------------------------------------------------------------------- #
# Byte accounting from queued Network events                                  #
# --------------------------------------------------------------------------- #
def collect_bytes(cdp: CDPSession, samples: list) -> int:
    """Drain queued Network.dataReceived events into (cdp_timestamp, bytes)
    samples. Binning is done later from the events' OWN timestamps (CDP reports a
    monotonic `timestamp` per event), not from drain time -- otherwise every
    event drained after one pump collapses into a single second. Returns total
    bytes seen this drain."""
    total = 0
    for msg in cdp.events:
        if msg.get("method") == "Network.dataReceived":
            p = msg.get("params", {})
            n = int(p.get("encodedDataLength") or p.get("dataLength") or 0)
            ts = p.get("timestamp")
            total += n
            if ts is not None:
                samples.append((float(ts), n))
    cdp.events.clear()
    return total


def bin_per_second(samples: list, duration: float) -> list:
    """Bucket (timestamp, bytes) samples into per-second bins from the first
    event's timestamp."""
    if not samples:
        return [0] * (int(duration) + 1)
    t_first = min(s[0] for s in samples)
    bins: dict[int, int] = {}
    for ts, n in samples:
        bins[int(ts - t_first)] = bins.get(int(ts - t_first), 0) + n
    span = int(max(duration, max(bins) + 1))
    return [bins.get(s, 0) for s in range(span)]


# --------------------------------------------------------------------------- #
# Activities                                                                   #
# --------------------------------------------------------------------------- #
ACTIVITY_URLS = {
    "youtube": "https://www.youtube.com",
    "instagram": "https://www.instagram.com/reels/",
    "x": "https://x.com/home",
    "slither": "https://slither.io",
    "zoom": "https://us05web.zoom.us/join",
}


def normalize_meeting_id(raw: str) -> str:
    """Reduce a human-entered Zoom meeting id (e.g. '658 354 2374' or
    '658-354-2374') to digits only, as the /wc/<id>/join URL expects. Raises
    ValueError if no digits remain."""
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        raise ValueError(f"no digits in meeting id: {raw!r}")
    return digits


def zoom_display_name(name: str | None, seed: int) -> str:
    """The browser-join display name. An explicit non-blank name wins; otherwise
    fall back to the seeded bot name (e.g. 'UEbot1'), matching slither's
    nickname convention so (activity, seed) stays reproducible."""
    if name and name.strip():
        return name
    return f"UEbot{seed}"

# JS that types the nickname into slither.io's #nick field and clicks Play, so
# the game starts without relying on blind fixed-coordinate clicks. Returns true
# if a play button was found and clicked. %s is the JSON-quoted nickname.
SLITHER_START_JS = (
    "(()=>{const n=document.getElementById('nick');"
    "if(n){n.value=%s;n.focus();}"
    "const b=document.querySelector('.btn-play,.nsi,#playh,div[onclick]');"
    "if(b){b.click();return true;}return false;})()"
)
SLITHER_REPLAY_JS = (
    "(()=>{const b=document.querySelector('.btn-play,.nsi,#playh,div[onclick]');"
    "if(b){b.click();return true;}return false;})()"
)


# --------------------------------------------------------------------------- #
# Zoom join helpers                                                           #
# --------------------------------------------------------------------------- #
# Zoom's web DOM is brittle and A/B-tested, so every step tries a list of
# selectors and then a text match, and reports which one hit (for diagnosis).
ZOOM_IN_MEETING_JS = (
    "(()=>{const t=(document.body?document.body.innerText:'').toLowerCase();"
    "const footer=document.querySelector("
    "'.footer__btns-container,[class*=footer-button],#foot-bar,"
    "[aria-label*=\"mute\" i],[aria-label*=\"leave\" i]');"
    "const media=document.querySelectorAll('video,canvas').length;"
    "const inMeeting=!!footer||/\\bleave\\b|start video|stop video|participants|"
    "unmute|mute\\b/.test(t);"
    "return JSON.stringify({inMeeting,media,url:location.href});})()"
)


def zoom_state(cdp: CDPSession) -> dict:
    raw = cdp.eval(ZOOM_IN_MEETING_JS)
    try:
        return json.loads(raw) if isinstance(raw, str) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def zoom_fill(cdp: CDPSession, selectors: list[str], value: str) -> str:
    """Set the first matching input to `value`, firing input/change so the SPA
    notices. Returns the selector that hit, or '' if none did."""
    js = (
        "(()=>{const sel=%s,v=%s;for(const s of sel){const e=document.querySelector(s);"
        "if(e){e.focus();e.value=v;e.dispatchEvent(new Event('input',{bubbles:true}));"
        "e.dispatchEvent(new Event('change',{bubbles:true}));return s;}}return '';})()"
    ) % (json.dumps(selectors), json.dumps(value))
    return str(cdp.eval(js) or "")


def zoom_click(cdp: CDPSession, selectors: list[str], texts: list[str]) -> str:
    """Click the first element matching a selector, else the first clickable
    whose text contains one of `texts` (already lowercased). Returns a tag of
    what hit ('s:<sel>' / 't:<text>' / '')."""
    js = (
        "(()=>{const sel=%s,texts=%s;"
        "for(const s of sel){const e=document.querySelector(s);if(e){e.click();return 's:'+s;}}"
        "const els=[...document.querySelectorAll('button,a,[role=\"button\"]')];"
        "for(const t of texts){const e=els.find(x=>"
        "((x.innerText||x.value||'').trim().toLowerCase()).includes(t));"
        "if(e){e.click();return 't:'+t;}}return '';})()"
    ) % (json.dumps(selectors), json.dumps(texts))
    return str(cdp.eval(js) or "")


def ensure_zoom_media(cdp: CDPSession) -> None:
    """Best-effort: join computer audio and start the camera so the call carries
    both up- and down-link media. Inert once already engaged."""
    zoom_click(
        cdp,
        ["button.join-audio-by-voip__join-btn", "[class*=join-audio] button"],
        ["join audio by computer", "join with computer audio", "join audio"],
    )
    zoom_click(
        cdp,
        ["button[aria-label*='start video' i]", "button[aria-label*='start my video' i]"],
        ["start video"],
    )


def wait_for_zoom_approval(cdp: CDPSession, name: str, ready_file: str,
                           go_file: str, timeout: float = 600.0) -> None:
    """Park at the display-name screen: signal readiness by writing `ready_file`
    (with the typed name), then block until capture.sh creates `go_file` (operator
    approval) or `timeout` elapses. Keeps the websocket alive while parked so the
    page/session does not drop."""
    log(f"zoom: parked at the name screen as '{name}'; writing ready file "
        f"{ready_file} and waiting for operator approval (go file {go_file}).")
    try:
        with open(ready_file, "w") as fh:
            fh.write(name)
    except OSError as exc:
        log(f"zoom: could not write ready file {ready_file}: {exc}")
    waited = 0.0
    while not os.path.exists(go_file):
        cdp.pump(1.0)  # ~1s; keeps Network/keepalive frames flowing while parked
        waited += 1.0
        if waited >= timeout:
            log(f"zoom: approval timed out after {timeout:.0f}s; joining anyway.")
            return
    log(f"zoom: approval received after ~{waited:.0f}s; joining.")


def join_zoom(cdp: CDPSession, meeting_id: str, name: str,
              pause_ready: str | None = None, pause_go: str | None = None) -> bool:
    """Drive the browser join: fill the meeting id on /join, submit, fall back to
    the /wc/<id>/join web client if the app-launch interstitial blocks us, set a
    display name, join, and engage audio/video. Returns True if the in-meeting UI
    is reached. When pause_ready/pause_go are set, park at the name screen for
    operator approval (via a file handshake) before clicking Join."""
    digits = normalize_meeting_id(meeting_id)

    # Auto-accept camera/mic for the Zoom origin so getUserMedia never blocks on
    # a permission prompt. Contained here -- ue-chrome-agent.sh is not touched.
    try:
        cdp.call(
            "Browser.grantPermissions",
            {"origin": "https://us05web.zoom.us",
             "permissions": ["videoCapture", "audioCapture"]},
        )
    except CDPError as exc:
        log(f"zoom: grantPermissions failed (continuing): {exc}")

    cdp.pump(3.0)
    # Best-effort cookie banner dismissal so it can't intercept clicks.
    zoom_click(cdp, ["#onetrust-accept-btn-handler"], ["accept cookies", "i accept", "accept all"])

    filled = zoom_fill(
        cdp, ["#join-confno", "input[name='join-confno']", "input#join-confno"], digits)
    log(f"zoom: filled meeting id ({digits}) -> {filled!r}")
    clicked = zoom_click(cdp, ["#btnSubmit", "button[type=submit]"], ["join"])
    log(f"zoom: clicked join -> {clicked!r}")
    cdp.pump(5.0)

    if not zoom_state(cdp).get("inMeeting"):
        wc = f"https://us05web.zoom.us/wc/{digits}/join"
        log(f"zoom: not in meeting after /join submit; navigating web client {wc}")
        cdp.call("Page.navigate", {"url": wc})
        cdp.pump(6.0)

    # Browser-join / preview screen: set the display name, accept ToS, join.
    name_field = zoom_fill(
        cdp,
        ["#input-for-name", "#inputname", "input[name='inputname']",
         "input#inputname", "input[type='text']"],
        name,
    )
    log(f"zoom: set name '{name}' -> {name_field!r}")

    # Park here for operator approval before actually joining, so power recording
    # starts only once the operator OKs the join (display name already typed).
    if pause_ready and pause_go:
        wait_for_zoom_approval(cdp, name, pause_ready, pause_go)

    cdp.eval("(()=>{const c=document.querySelector('#wc_agree1,input[type=checkbox]');"
             "if(c&&!c.checked)c.click();})()")
    join_hit = zoom_click(
        cdp, ["#joinBtn", ".preview-join-button", "button.preview-join-button"], ["join"])
    log(f"zoom: preview join -> {join_hit!r}")

    # Poll for the in-meeting UI (host admit / waiting room / media setup can take
    # a while), engaging audio+video each round.
    in_meeting = False
    for _ in range(8):
        cdp.pump(5.0)
        ensure_zoom_media(cdp)
        st = zoom_state(cdp)
        if st.get("inMeeting"):
            in_meeting = True
            break
    final = zoom_state(cdp)
    log(f"zoom: join result inMeeting={final.get('inMeeting')} "
        f"media={final.get('media')} url={final.get('url')}")
    return in_meeting


def check_login(activity: str, url: str, cdp: CDPSession) -> int:
    """Probe whether `activity` currently shows a login wall, reusing the page
    tab if one is open. Prints a one-line JSON result. Returns 0 if clear, 3 if a
    login wall is present (so capture.sh can pause and ask the user to log in). The
    tab is left open so the user logs in right where the probe landed."""
    host = urllib.parse.urlparse(url).hostname or ""
    if host.startswith("www."):
        host = host[4:]
    reused = cdp.open_or_reuse(url, host)
    cdp.call("Page.enable")
    cdp.call("Runtime.enable")
    cdp.pump(6.0)  # let the SPA render before judging
    st = page_state(cdp)
    wall = bool(st.get("urlGate") or st.get("textGate"))
    log(f"login check: activity={activity} reused_tab={reused} url={st.get('url')} "
        f"login_wall={wall} feedItems={st.get('feedItems')}")
    print(json.dumps({"activity": activity, "login_wall": wall,
                      "url": st.get("url"), "feed_items": st.get("feedItems")}))
    return 3 if wall else 0


def run_activity(activity: str, url: str, duration: float, seed: int,
                 cdp: CDPSession, allow_login_wall: bool,
                 meeting_id: str = "6583542374", name: str | None = None,
                 pause_ready: str | None = None,
                 pause_go: str | None = None) -> dict:
    # Reuse the tab the login check (or a prior step) already opened on this host
    # rather than opening a second one -- otherwise the tab the user just logged
    # into is abandoned and a duplicate page is created. Falls back to opening a
    # fresh tab when none exists (e.g. --allow-login-wall with no prior check).
    host = urllib.parse.urlparse(url).hostname or ""
    if host.startswith("www."):
        host = host[4:]
    reused = cdp.open_or_reuse(url, host)
    log(f"{'reused' if reused else 'opened'} target on {host}: {url}")
    cdp.call("Page.enable")
    cdp.call("Network.enable")
    cdp.call("Runtime.enable")

    rng = Random(seed)
    t0 = time.monotonic()
    samples: list[tuple[float, int]] = []
    total_bytes = 0

    joined = True
    if activity == "zoom":
        # Zoom has no login wall; drive the browser join instead, then dwell.
        joined = join_zoom(cdp, meeting_id, zoom_display_name(name, seed),
                           pause_ready, pause_go)
        total_bytes += collect_bytes(cdp, samples)
        st = page_state(cdp)
        login_wall = False
        if not joined and not allow_login_wall:
            raise CDPError(
                "could not reach the in-browser Zoom meeting (still on the "
                "launch/preview page). The meeting may be over, in a waiting "
                "room, or Zoom's join UI changed. Pass --allow-login-wall to "
                "capture anyway.")
    else:
        # let the SPA render before judging login state
        cdp.pump(6.0)
        total_bytes += collect_bytes(cdp, samples)
        st = page_state(cdp)
        login_wall = bool(st.get("urlGate") or st.get("textGate"))
        log(f"initial state: url={st.get('url')} login_wall={login_wall} "
            f"hasVideo={st.get('hasVideo')} feedItems={st.get('feedItems')} "
            f"bodyLen={st.get('bodyLen')}")
        if login_wall and not allow_login_wall:
            raise CDPError(
                f"login wall detected for {activity} ({st.get('url')}). The Chrome "
                f"profile is not logged in. Log in once in CHROME_PROFILE_DIR, or pass "
                f"--allow-login-wall to capture anyway.")

    scrolls = 0
    video_seen = bool(st.get("hasVideo"))
    max_feed_items = int(st.get("feedItems") or 0)
    while time.monotonic() - t0 < duration:
        if activity == "youtube":
            # ensure the watch page is actually playing, then just dwell
            cdp.eval("(()=>{const v=document.querySelector('video');"
                     "if(v){v.muted=true;v.play&&v.play().catch(()=>{});}})()")
            dwell = rng.uniform(8.0, 14.0)
        elif activity == "instagram":
            # advance to the next reel; ArrowDown is the reels keyboard shortcut,
            # fall back to a viewport wheel scroll
            press_key(cdp, "ArrowDown", "ArrowDown", 40)
            wheel_scroll(cdp, 900)
            cdp.eval("(()=>{const v=document.querySelector('video');"
                     "if(v){v.muted=true;v.play&&v.play().catch(()=>{});}})()")
            dwell = rng.uniform(4.0, 9.0)
            scrolls += 1
        elif activity == "x":
            # scroll the timeline a viewport-ish at a reading cadence
            wheel_scroll(cdp, rng.randint(700, 1100))
            dwell = rng.uniform(2.0, 5.0)
            scrolls += 1
        elif activity == "slither":
            # slither.io: a real-time .io game on a <canvas>, driven blind.
            if scrolls == 0:
                # Type a nickname (required) into #nick and click Play. Enter and
                # a center-screen click are blind fallbacks in case the button
                # selector misses.
                nick = json.dumps(f"UEbot{seed}")
                cdp.eval(SLITHER_START_JS % nick)
                press_key(cdp, "Enter", "Enter", 13)
                dims = cdp.eval("[window.innerWidth||800, window.innerHeight||600]")
                if isinstance(dims, list) and len(dims) == 2:
                    click(cdp, int(dims[0]) // 2, int(int(dims[1]) * 0.62))
            elif scrolls % 15 == 0:
                # Re-click Play to respawn after a death; inert while alive.
                cdp.eval(SLITHER_REPLAY_JS)
                press_key(cdp, "Enter", "Enter", 13)
            # The snake follows the cursor: steer toward a moving point, then
            # boost in a short burst to raise the update rate.
            aim_x, aim_y = rng.randint(100, 700), rng.randint(100, 500)
            move_mouse(cdp, aim_x, aim_y)
            boost(cdp, aim_x, aim_y, rng.uniform(0.15, 0.4))
            dwell = rng.uniform(0.4, 0.9)
            scrolls += 1
        elif activity == "zoom":
            # In the meeting: media flows on its own. Keep audio/video engaged
            # (re-clicks are inert once joined) and dwell.
            ensure_zoom_media(cdp)
            dwell = rng.uniform(8.0, 14.0)
        else:
            raise CDPError(f"unknown activity: {activity}")

        remaining = duration - (time.monotonic() - t0)
        cdp.pump(min(dwell, max(0.2, remaining)))
        total_bytes += collect_bytes(cdp, samples)
        st = page_state(cdp)
        if st.get("hasVideo"):
            video_seen = True
        max_feed_items = max(max_feed_items, int(st.get("feedItems") or 0))

    elapsed = time.monotonic() - t0
    timeline = bin_per_second(samples, elapsed)
    active = sum(1 for b in timeline if b * 8 / 1000 > 50)
    report = {
        "activity": activity,
        "url": url,
        "seed": seed,
        "duration_s": round(elapsed, 2),
        "login_wall": login_wall,
        "joined": joined,
        "scrolls": scrolls,
        "video_seen": video_seen,
        "max_feed_items": max_feed_items,
        "total_bytes": total_bytes,
        "mean_kbps": round(total_bytes * 8 / 1000 / max(elapsed, 1e-6), 1),
        "active_seconds": active,
        "per_second_bytes": timeline,
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activity", required=True, choices=sorted(ACTIVITY_URLS))
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--url", default=None, help="override the activity's default URL")
    ap.add_argument("--zoom-meeting-id", default="658 354 2374",
                    help="zoom meeting id to join (digits/spaces/dashes ok)")
    ap.add_argument("--zoom-name", default=None,
                    help="display name for the zoom browser join "
                         "(default: UEbot<seed>)")
    ap.add_argument("--zoom-pause-ready", default=None,
                    help="zoom: file to create once parked at the name screen "
                         "(signals capture.sh the join is ready for approval)")
    ap.add_argument("--zoom-pause-go", default=None,
                    help="zoom: file to wait for before clicking Join "
                         "(capture.sh creates it on operator approval)")
    ap.add_argument("--cdp-host", default="127.0.0.1")
    ap.add_argument("--cdp-port", type=int, default=9223)
    ap.add_argument("--allow-login-wall", action="store_true",
                    help="capture even if a login wall is detected (default: fail)")
    ap.add_argument("--check-login", action="store_true",
                    help="probe for a login wall and exit (0=clear, 3=wall, 2=error); "
                         "does not run the activity")
    ap.add_argument("--report", default=None, help="write the JSON metrics report here")
    args = ap.parse_args()

    url = args.url or ACTIVITY_URLS[args.activity]
    cdp = CDPSession(args.cdp_host, args.cdp_port)

    if args.check_login:
        try:
            return check_login(args.activity, url, cdp)
        except CDPError as exc:
            log(f"ERROR: {exc}")
            return 2
        finally:
            cdp.close()

    try:
        report = run_activity(args.activity, url, args.duration, args.seed, cdp,
                              args.allow_login_wall,
                              meeting_id=args.zoom_meeting_id, name=args.zoom_name,
                              pause_ready=args.zoom_pause_ready,
                              pause_go=args.zoom_pause_go)
    except CDPError as exc:
        log(f"ERROR: {exc}")
        return 2
    finally:
        cdp.close()

    log(f"done: {report['total_bytes']} bytes, mean {report['mean_kbps']} kbps, "
        f"scrolls={report['scrolls']} video_seen={report['video_seen']} "
        f"login_wall={report['login_wall']}")
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2)
        log(f"report written: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

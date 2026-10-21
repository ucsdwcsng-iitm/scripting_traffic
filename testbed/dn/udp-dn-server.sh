#!/usr/bin/env bash
set -euo pipefail

# Data-network endpoint for the persistent-socket UDP traffic generator.
#
#   UE (ue-traffic ns / wwan0) --default route--> 10.45.0.1 (ogstun, this host)
#
# Runs synthetic_traffic/udp_traffic_server.py, the DN-side peer of the UE-side
# udp_traffic_player.py. One long-running daemon serves every capture; the UE
# drives each session over a TCP control channel.
#
# Ports: CTRL 5300 (TCP), DL 5301 / UL 5302 (UDP).
# Runs in the ROOT namespace on the Open5GS core host (owns ogstun).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

BIND_IP="${BIND_IP:-10.45.0.1}"
CTRL_PORT="${CTRL_PORT:-5300}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
SERVER_PY="${SERVER_PY:-${REPO_ROOT}/synthetic_traffic/udp_traffic_server.py}"
UNIT="/etc/systemd/system/udp-dn-server.service"

log() { printf '[udp-dn-server] %s\n' "$*"; }

need_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "Run with sudo: sudo $0 $*" >&2
    exit 1
  fi
}

bind_ip_present() {
  ip -br addr | awk '{print $3}' | grep -qE "(^|[[:space:]])${BIND_IP}/" 2>/dev/null
}

install_unit() {
  need_root "$@"
  [[ -f "${SERVER_PY}" ]] || { echo "Missing server: ${SERVER_PY}" >&2; exit 1; }
  cat > "${UNIT}" <<UNIT
[Unit]
Description=Persistent-UDP DN server for synthetic UE traffic
After=network-online.target

[Service]
Type=simple
ExecStart=${PYTHON_BIN} ${SERVER_PY} --bind ${BIND_IP} --ctrl-port ${CTRL_PORT}
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
UNIT
  systemctl daemon-reload
  log "Installed ${UNIT} (bind ${BIND_IP}, ctrl ${CTRL_PORT})"
}

start_server() {
  need_root "$@"
  if ! bind_ip_present; then
    echo "WARNING: ${BIND_IP} is not a local address (is the 5G core / ogstun up?)." >&2
    echo "         Start Open5GS first, or override BIND_IP=." >&2
  fi
  [[ -f "${UNIT}" ]] || install_unit
  systemctl reset-failed udp-dn-server.service >/dev/null 2>&1 || true
  systemctl restart udp-dn-server.service
  log "udp-dn server listening on ${BIND_IP}:${CTRL_PORT} (ctrl), 5301/5302 UDP"
  systemctl --no-pager --lines=4 status udp-dn-server.service || true
}

stop_server() {
  need_root "$@"
  systemctl stop udp-dn-server.service 2>/dev/null || true
  log "stopped udp-dn-server.service"
}

status_server() {
  printf '\n== ogstun ==\n'
  ip -br addr show ogstun 2>/dev/null || echo "ogstun not present"
  printf '\n== udp-dn server ==\n'
  systemctl --no-pager --lines=6 status udp-dn-server.service 2>/dev/null || true
}

usage() {
  cat <<EOF
Usage: sudo $0 <start|stop|status|install>

  start    Install unit if needed, then start the UDP DN server on
           ${BIND_IP}:${CTRL_PORT} (ctrl) + 5301/5302 (UDP data).
  stop     Stop the server.
  status   Show ogstun address and server unit status.
  install  Install the systemd unit only.

Overrides: BIND_IP, CTRL_PORT, PYTHON_BIN, SERVER_PY.
EOF
}

case "${1:-}" in
  start)   start_server ;;
  stop)    stop_server ;;
  status)  status_server ;;
  install) install_unit ;;
  -h|--help|help) usage ;;
  *) usage; exit 2 ;;
esac

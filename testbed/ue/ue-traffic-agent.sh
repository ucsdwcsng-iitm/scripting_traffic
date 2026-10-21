#!/usr/bin/env bash
set -euo pipefail

# Sibling of ue-chrome-agent.sh for SYNTHETIC traffic captures.
#
#   udp_traffic_player -> ue-traffic namespace -> wwan0/QMI -> Qualcomm UE
#       -> srsRAN gNB -> Open5GS UPF (10.45.0.1) -> udp_traffic_server (DN)
#
# Same data-path bring-up as the Chrome agent (QMI session, wwan0 moved into a
# namespace, MTU/DNS), but it runs synthetic_traffic/udp_traffic_player.py instead of
# Chrome. Only one namespace can own the single physical wwan0 at a time, so the
# Chrome path and this path are mutually exclusive; this agent relocates wwan0
# out of the Chrome namespace if it finds it there.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
NS="ue-traffic"
CHROME_NS="ue-browser"
APN="internet"
WDS_DEV="/dev/cdc-wdm0"
WWAN="wwan0"
QMI_STATE="/run/ue-traffic-qmi.env"
DNS1="8.8.8.8"
DNS2="8.8.4.4"
MTU="1400"
DN_GATEWAY="${DN_GATEWAY:-10.45.0.1}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
TRAFFIC_PLAYER="${TRAFFIC_PLAYER:-${REPO_ROOT}/synthetic_traffic/udp_traffic_player.py}"

log() { printf '[ue-traffic-agent] %s\n' "$*"; }

need_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "Run with sudo: sudo $0 $*" >&2
    exit 1
  fi
}

ensure_qmi_driver() {
  if [[ ! -e "${WDS_DEV}" ]]; then
    modprobe --ignore-install qmi_wwan || true
  fi
  [[ -e "${WDS_DEV}" ]] || {
    echo "Missing ${WDS_DEV}; modem QMI device is not available." >&2
    exit 1
  }
}

qmi_connected() {
  qmicli -d "${WDS_DEV}" --device-open-proxy --wds-get-packet-service-status 2>/dev/null \
    | grep -q "Connection status: 'connected'"
}

wwan_in_root() { ip link show "${WWAN}" >/dev/null 2>&1; }
wwan_in_ns()   { ip netns exec "${NS}" ip link show "${WWAN}" >/dev/null 2>&1; }
wwan_in_chrome_ns() {
  ip netns list 2>/dev/null | grep -q "^${CHROME_NS} " &&
    ip netns exec "${CHROME_NS}" ip link show "${WWAN}" >/dev/null 2>&1
}

ensure_raw_ip_if_possible() {
  if [[ -e "/sys/class/net/${WWAN}/qmi/raw_ip" ]]; then
    ip link set "${WWAN}" down || true
    printf 'Y' > "/sys/class/net/${WWAN}/qmi/raw_ip" || true
    ip link set "${WWAN}" up || true
  elif wwan_in_ns && ip netns exec "${NS}" test -e "/sys/class/net/${WWAN}/qmi/raw_ip"; then
    ip netns exec "${NS}" ip link set "${WWAN}" down || true
    ip netns exec "${NS}" sh -c "printf Y > /sys/class/net/${WWAN}/qmi/raw_ip" || true
    ip netns exec "${NS}" ip link set "${WWAN}" up || true
  fi
}

start_qmi_if_needed() {
  ensure_qmi_driver
  if qmi_connected; then
    log "QMI data session already connected"
    ensure_raw_ip_if_possible
    return
  fi

  log "Starting QMI IPv4 data session for APN ${APN}"
  ensure_raw_ip_if_possible
  local out
  if ! out="$(qmicli -d "${WDS_DEV}" --device-open-proxy \
      --wds-start-network="apn=${APN},ip-type=4" \
      --client-no-release-cid 2>&1)"; then
    printf '%s\n' "${out}" >&2
    local failed_cid
    failed_cid="$(printf '%s\n' "${out}" | sed -n "s/.*CID: '\([^']*\)'.*/\1/p" | tail -n 1)"
    if [[ -n "${failed_cid}" ]]; then
      log "Releasing failed WDS CID ${failed_cid}"
      qmicli -d "${WDS_DEV}" --device-open-proxy --wds-noop --client-cid="${failed_cid}" >/dev/null 2>&1 || true
    fi
    exit 1
  fi
  printf '%s\n' "${out}"

  local handle cid
  handle="$(printf '%s\n' "${out}" | sed -n "s/.*Packet data handle: '\([^']*\)'.*/\1/p" | tail -n 1)"
  cid="$(printf '%s\n' "${out}" | sed -n "s/.*CID: '\([^']*\)'.*/\1/p" | tail -n 1)"
  if [[ -n "${handle}" && -n "${cid}" ]]; then
    {
      printf 'WDS_HANDLE=%q\n' "${handle}"
      printf 'WDS_CID=%q\n' "${cid}"
    } > "${QMI_STATE}"
  fi
}

qmi_settings() { qmicli -d "${WDS_DEV}" --device-open-proxy --wds-get-current-settings; }

mask_to_prefix() {
  local mask="$1"
  local IFS=.
  local -a octets
  read -r -a octets <<< "${mask}"
  local prefix=0 octet bit
  for octet in "${octets[@]}"; do
    for bit in 128 64 32 16 8 4 2 1; do
      if (( octet & bit )); then ((prefix++)); fi
    done
  done
  printf '%s\n' "${prefix}"
}

ensure_namespace() {
  # `ip netns list` prints a namespace with no assigned id as a bare name (no
  # trailing " (id: N)"), so a "^${NS} " match misses a leftover namespace and we
  # try to re-add it -> "Cannot create namespace file: File exists". Match the
  # name as a whole first field instead so a pre-existing namespace is reused.
  if ! ip netns list | awk '{print $1}' | grep -qx "${NS}"; then
    log "Creating network namespace ${NS}"
    ip netns add "${NS}"
  fi
  ip netns exec "${NS}" ip link set lo up
  mkdir -p "/etc/netns/${NS}"
  {
    printf 'nameserver %s\n' "${DNS1}"
    printf 'nameserver %s\n' "${DNS2}"
  } > "/etc/netns/${NS}/resolv.conf"
}

relocate_wwan_into_ns() {
  if wwan_in_ns; then
    return
  elif wwan_in_root; then
    log "Moving ${WWAN} from root into ${NS}"
    ip link set "${WWAN}" netns "${NS}"
  elif wwan_in_chrome_ns; then
    log "Moving ${WWAN} from ${CHROME_NS} into ${NS}"
    ip netns exec "${CHROME_NS}" ip link set "${WWAN}" netns "${NS}"
  else
    echo "${WWAN} is not visible in root, ${NS}, or ${CHROME_NS}." >&2
    exit 1
  fi
}

configure_namespace() {
  ensure_namespace
  local settings ipaddr mask gw prefix
  settings="$(qmi_settings)"
  ipaddr="$(printf '%s\n' "${settings}" | awk -F': ' '/IPv4 address/ {print $2; exit}')"
  mask="$(printf '%s\n' "${settings}" | awk -F': ' '/IPv4 subnet mask/ {print $2; exit}')"
  gw="$(printf '%s\n' "${settings}" | awk -F': ' '/IPv4 gateway address/ {print $2; exit}')"

  [[ -n "${ipaddr}" && -n "${mask}" && -n "${gw}" ]] || {
    echo "Could not read IPv4 QMI settings." >&2
    printf '%s\n' "${settings}" >&2
    exit 1
  }
  prefix="$(mask_to_prefix "${mask}")"

  relocate_wwan_into_ns

  ip netns exec "${NS}" ip link set "${WWAN}" up mtu "${MTU}"
  ip netns exec "${NS}" ip addr flush dev "${WWAN}"
  ip netns exec "${NS}" ip addr add "${ipaddr}/${prefix}" dev "${WWAN}"
  ip netns exec "${NS}" ip route replace default via "${gw}" dev "${WWAN}" onlink
  ip netns exec "${NS}" sysctl -w \
    net.ipv6.conf.all.disable_ipv6=1 \
    net.ipv6.conf.default.disable_ipv6=1 \
    "net.ipv6.conf.${WWAN}.disable_ipv6=1" >/dev/null || true
}

check_data_path() {
  log "Checking UE namespace data path"
  ip netns exec "${NS}" ping -c 2 -W 2 "${DN_GATEWAY}"
  ip netns exec "${NS}" ping -c 2 -W 2 8.8.8.8
}

stop_qmi_if_tracked() {
  if [[ ! -f "${QMI_STATE}" ]]; then return; fi
  # shellcheck disable=SC1090
  source "${QMI_STATE}"
  if [[ -n "${WDS_HANDLE:-}" ]]; then
    log "Stopping tracked QMI data session handle ${WDS_HANDLE}"
    qmicli -d "${WDS_DEV}" --device-open-proxy \
      --wds-stop-network="${WDS_HANDLE}" \
      --client-cid="${WDS_CID:-}" \
      --client-no-release-cid >/dev/null 2>&1 || true
  fi
  if [[ -n "${WDS_CID:-}" ]]; then
    log "Releasing tracked WDS CID ${WDS_CID}"
    qmicli -d "${WDS_DEV}" --device-open-proxy \
      --wds-noop --client-cid="${WDS_CID}" >/dev/null 2>&1 || true
  fi
  rm -f "${QMI_STATE}"
}

prepare_data_path() {
  need_root "$@"
  start_qmi_if_needed
  configure_namespace
  check_data_path
}

run_traffic() {
  need_root "$@"
  local schedule="${1:?Usage: $0 traffic <schedule.json> [results.json]}"
  local results="${2:-}"
  [[ -f "${schedule}" ]] || { echo "Missing schedule: ${schedule}" >&2; exit 1; }
  [[ -f "${TRAFFIC_PLAYER}" ]] || { echo "Missing player: ${TRAFFIC_PLAYER}" >&2; exit 1; }
  local args=(--schedule "${schedule}" --ns "${NS}" --verbose)
  [[ -n "${results}" ]] && args+=(--results "${results}")
  log "Playing ${schedule} through ${NS}"
  "${PYTHON_BIN}" "${TRAFFIC_PLAYER}" "${args[@]}"
}

status_workflow() {
  printf '\n== QMI ==\n'
  qmicli -d "${WDS_DEV}" --device-open-proxy --wds-get-packet-service-status 2>&1 || true
  printf '\n== Namespace ==\n'
  ip netns list || true
  ip netns exec "${NS}" ip -br addr 2>/dev/null || true
  ip netns exec "${NS}" ip route 2>/dev/null || true
}

usage() {
  cat <<EOF
Usage: sudo $0 <prepare|traffic <schedule.json> [results.json]|stop|status|check>

Purpose:
  Manage the synthetic-traffic UE path:
    udp_traffic_player -> ${NS} -> ${WWAN}/QMI -> UE -> srsRAN gNB -> UPF (${DN_GATEWAY}) -> UDP DN server

Commands:
  prepare   Bring up QMI if needed, move ${WWAN} into ${NS}, verify data path.
  traffic   Play a schedule JSON through ${NS} (assumes prepare succeeded).
  stop      Tear down the tracked QMI data session.
  status    Print QMI and namespace status.
  check     Ping ${DN_GATEWAY} and 8.8.8.8 from ${NS}.

Notes:
  - Only one namespace can own ${WWAN}; this is exclusive with the Chrome path.
  - This script does not start or restart the gNB.
EOF
}

case "${1:-}" in
  -h|--help|help) usage ;;
  prepare) prepare_data_path ;;
  traffic) shift; run_traffic "$@" ;;
  stop) need_root "$@"; stop_qmi_if_tracked ;;
  status) status_workflow ;;
  check) check_data_path ;;
  *) usage; exit 2 ;;
esac

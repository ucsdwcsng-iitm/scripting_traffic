#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NS="ue-browser"
APN="internet"
WDS_DEV="/dev/cdc-wdm0"
WWAN="wwan0"
UE_CHROME_SERVICE="/etc/systemd/system/ue-chrome.service"
UE_CHROME_LAUNCHER="/usr/local/bin/ue-chrome-start"
# Source of truth for the launcher, version-controlled in the repo.
UE_CHROME_LAUNCHER_SRC="${SCRIPT_DIR}/ue-chrome-start"
QMI_STATE="/run/ue-chrome-qmi.env"
DNS1="8.8.8.8"
DNS2="8.8.4.4"
MTU="1400"
CHROME_USER="${CHROME_USER:-${SUDO_USER:-$(id -un)}}"
CHROME_HOME="$(getent passwd "${CHROME_USER}" | cut -d: -f6)"
CHROME_PROFILE_DIR="${CHROME_PROFILE_DIR:-${CHROME_HOME}/.local/share/ue-chrome-profile}"

log() {
  printf '[ue-chrome-agent] %s\n' "$*"
}

need_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "Run with sudo: sudo $0 $*" >&2
    exit 1
  fi
}

install_service_unit() {
  need_root "$@"
  if [[ ! -x "${UE_CHROME_LAUNCHER}" ]]; then
    if [[ -f "${UE_CHROME_LAUNCHER_SRC}" ]]; then
      log "Installing launcher ${UE_CHROME_LAUNCHER} from ${UE_CHROME_LAUNCHER_SRC}"
      install -Dm755 "${UE_CHROME_LAUNCHER_SRC}" "${UE_CHROME_LAUNCHER}"
    else
      echo "Missing launcher: ${UE_CHROME_LAUNCHER} (no bundled copy at ${UE_CHROME_LAUNCHER_SRC})" >&2
      exit 1
    fi
  fi

  cat > "${UE_CHROME_SERVICE}" <<UNIT
[Unit]
Description=Chrome routed through the ue-browser network namespace
After=network-online.target

[Service]
Type=simple
Environment=UE_CHROME_USER=${CHROME_USER}
Environment=UE_CHROME_PROFILE_DIR=${CHROME_PROFILE_DIR}
ExecStart=/usr/local/bin/ue-chrome-start
ExecStop=-/usr/bin/pkill -u ${CHROME_USER} -f -- "--user-data-dir=${CHROME_PROFILE_DIR}"
Restart=no
StandardOutput=append:/tmp/ue-browser-chrome.log
StandardError=append:/tmp/ue-browser-chrome.log

[Install]
WantedBy=multi-user.target
UNIT
  systemctl daemon-reload
}

ensure_service_unit() {
  if [[ ! -f "${UE_CHROME_SERVICE}" ]]; then
    log "Installing missing ue-chrome.service"
    install_service_unit
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

wwan_in_root() {
  ip link show "${WWAN}" >/dev/null 2>&1
}

wwan_in_ns() {
  ip netns exec "${NS}" ip link show "${WWAN}" >/dev/null 2>&1
}

# The synthetic-traffic path (ue-traffic-agent.sh) parks wwan0 in its own
# namespace. Detect that so this agent can reclaim the single physical wwan0.
TRAFFIC_NS="ue-traffic"
wwan_in_traffic_ns() {
  ip netns list 2>/dev/null | grep -qE "^${TRAFFIC_NS}"'([[:space:]]|$)' &&
    ip netns exec "${TRAFFIC_NS}" ip link show "${WWAN}" >/dev/null 2>&1
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

qmi_settings() {
  qmicli -d "${WDS_DEV}" --device-open-proxy --wds-get-current-settings
}

mask_to_prefix() {
  local mask="$1"
  local IFS=.
  local -a octets
  read -r -a octets <<< "${mask}"
  local prefix=0 octet bit
  for octet in "${octets[@]}"; do
    for bit in 128 64 32 16 8 4 2 1; do
      if (( octet & bit )); then
        ((prefix++))
      fi
    done
  done
  printf '%s\n' "${prefix}"
}

ensure_namespace() {
  if ! ip netns list | grep -qE "^${NS}"'([[:space:]]|$)'; then
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

  if wwan_in_root; then
    log "Moving ${WWAN} into ${NS}"
    ip link set "${WWAN}" netns "${NS}"
  elif wwan_in_traffic_ns; then
    log "Moving ${WWAN} from ${TRAFFIC_NS} into ${NS}"
    ip netns exec "${TRAFFIC_NS}" ip link set "${WWAN}" netns "${NS}"
  elif ! wwan_in_ns; then
    echo "${WWAN} is not visible in root namespace, ${NS}, or ${TRAFFIC_NS}." >&2
    exit 1
  fi

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
  ip netns exec "${NS}" ping -c 2 -W 2 8.8.8.8
  ip netns exec "${NS}" curl -4 -sS -I --max-time 8 https://www.youtube.com | sed -n '1,8p'
}

stop_qmi_if_tracked() {
  if [[ ! -f "${QMI_STATE}" ]]; then
    return
  fi

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
      --wds-noop \
      --client-cid="${WDS_CID}" >/dev/null 2>&1 || true
  fi

  rm -f "${QMI_STATE}"
}

start_chrome() {
  ensure_service_unit
  systemctl reset-failed ue-chrome.service >/dev/null 2>&1 || true
  systemctl restart ue-chrome.service
  sleep 8
  systemctl --no-pager --lines=12 status ue-chrome.service
  log "Chrome targets"
  ip netns exec "${NS}" curl -s --max-time 5 http://127.0.0.1:9223/json \
    | grep -E '"title"|"url"' \
    | head -n 8 || true
}

prepare_data_path() {
  need_root "$@"
  start_qmi_if_needed
  configure_namespace
  check_data_path
}

start_chrome_only() {
  need_root "$@"
  start_chrome
}

start_workflow() {
  prepare_data_path "$@"
  start_chrome
}

stop_workflow() {
  need_root "$@"
  systemctl stop ue-chrome.service 2>/dev/null || true
  stop_qmi_if_tracked
}

status_workflow() {
  printf '\n== Open5GS ==\n'
  systemctl --no-pager --lines=4 status open5gs-amfd open5gs-smfd open5gs-upfd 2>/dev/null || true

  printf '\n== QMI ==\n'
  qmicli -d "${WDS_DEV}" --device-open-proxy --wds-get-packet-service-status 2>&1 || true
  qmicli -d "${WDS_DEV}" --device-open-proxy --wds-get-current-settings 2>&1 | sed -n '1,80p' || true

  printf '\n== Namespace ==\n'
  ip netns list || true
  ip netns exec "${NS}" ip -br addr 2>/dev/null || true
  ip netns exec "${NS}" ip route 2>/dev/null || true

  printf '\n== Chrome Service ==\n'
  systemctl --no-pager --lines=12 status ue-chrome.service 2>/dev/null || true
}

usage() {
  cat <<EOF
Usage: sudo $0 <start|restart|prepare|chrome|stop|status|check>

Purpose:
  Manage the Chrome-over-UE path:
    Chrome -> ${NS} namespace -> ${WWAN}/QMI -> Qualcomm UE -> srsRAN gNB -> Open5GS UPF -> Internet

Commands:
  start/restart  Bring up QMI if needed, configure ${NS}, verify data, start Chrome.
  prepare        Bring up QMI if needed, configure ${NS}, verify data. Does not start Chrome.
  chrome         Start Chrome service only. Assumes prepare has already succeeded.
  stop           Stop the Chrome service only. It does not tear down QMI.
  status         Print Open5GS, QMI, namespace, and Chrome service status.
  check          Verify UE namespace internet path with ping and HTTPS.
  -h, --help     Show this help text.

Notes:
  - This script does not start or restart the gNB.
  - If the modem is missing, it stops and reports the missing QMI device.
  - If QMI and ${NS} are already healthy, it reuses them instead of recreating them.
EOF
}

case "${1:-}" in
  -h|--help|help)
    usage
    ;;
  start|restart)
    start_workflow
    ;;
  prepare)
    prepare_data_path
    ;;
  chrome)
    start_chrome_only
    ;;
  stop)
    stop_workflow
    ;;
  status)
    status_workflow
    ;;
  check)
    check_data_path
    ;;
  *)
    usage
    exit 2
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# srsRAN Project checkout with a built gNB (not part of this repository).
SRSRAN_DIR="${SRSRAN_DIR:-${HOME}/srsRAN_Project}"
GNB_BIN="${GNB_BIN:-${SRSRAN_DIR}/build/apps/gnb/gnb}"
# Optional base gNB config; further configs/overlays come from --gnb-config.
GNB_CONFIG="${GNB_CONFIG:-}"
# The gNB runs from RUN_DIR. GNB_RUN_LOG must be the file the gNB config's
# log.filename points to (a relative filename resolves inside RUN_DIR).
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/runs}"
GNB_RUN_LOG="${GNB_RUN_LOG:-${RUN_DIR}/logs/gnb_n41.log}"
CHROME_AGENT="${CHROME_AGENT:-${SCRIPT_DIR}/ue/ue-chrome-agent.sh}"
CHROME_ACTIVITY_AGENT="${CHROME_ACTIVITY_AGENT:-${SCRIPT_DIR}/ue/ue-chrome-activity.py}"
TRAFFIC_AGENT="${TRAFFIC_AGENT:-${SCRIPT_DIR}/ue/ue-traffic-agent.sh}"
SYNTHETIC_TRAFFIC_DIR="${SYNTHETIC_TRAFFIC_DIR:-${PROJECT_ROOT}/synthetic_traffic}"
# Synthetic traffic: persistent-UDP player at a 10 ms tick. It requires the
# udp-dn-server.sh daemon on the DN host; with SYNTHETIC_AUTOSTART_DN=true (the
# default) this script starts that daemon itself if it is not already up.
SYNTHETIC_PLAYER="${SYNTHETIC_PLAYER:-udp_traffic_player.py}"
SYNTHETIC_TICK="${SYNTHETIC_TICK:-0.01}"
SYNTHETIC_MIN_SEG="${SYNTHETIC_MIN_SEG:-0.01}"
SYNTHETIC_AUTOSTART_DN="${SYNTHETIC_AUTOSTART_DN:-true}"
DN_SERVER_AGENT="${DN_SERVER_AGENT:-${SCRIPT_DIR}/dn/udp-dn-server.sh}"
UE_POWER_ESTIMATION_REPO="${UE_POWER_ESTIMATION_REPO:-${PROJECT_ROOT}/ue_power_estimation}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
CHROME_SERVICE="${CHROME_SERVICE:-ue-chrome.service}"
CHROME_USER="${CHROME_USER:-$(id -un)}"
CHROME_PROFILE_DIR="${CHROME_PROFILE_DIR:-${HOME}/.local/share/ue-chrome-profile}"
PI_USER="${PI_USER:-rabbit}"
PI_HOST="${PI_HOST:-qcom-pi.local}"
PI_INA_DIR="${PI_INA_DIR:-/home/${PI_USER}/Downloads/INA219}"
LOCAL_CAPTURE_BASE="${LOCAL_CAPTURE_BASE:-${UE_POWER_ESTIMATION_REPO}/data/qualcomm}"
# Assumed TR 38.840 micro-sleep power (mW) for the per-capture 3GPP estimate.
DEFAULT_3GPP_MICROSLEEP_POWER_MW="400"
MODEM_PREPARE_TIMEOUT_SECONDS="${MODEM_PREPARE_TIMEOUT_SECONDS:-180}"
MODEM_PREPARE_RETRY_SECONDS="${MODEM_PREPARE_RETRY_SECONDS:-10}"
MODEM_AIRPLANE_OFF_SECONDS="${MODEM_AIRPLANE_OFF_SECONDS:-5}"
OPEN5GS_RESTART_ON_PFCP_FAILURE="${OPEN5GS_RESTART_ON_PFCP_FAILURE:-true}"
OPEN5GS_PFCP_LOG_LOOKBACK_SECONDS="${OPEN5GS_PFCP_LOG_LOOKBACK_SECONDS:-90}"
OPEN5GS_RESTART_SETTLE_SECONDS="${OPEN5GS_RESTART_SETTLE_SECONDS:-5}"
# Unconditionally restart the Open5GS user plane (UPF+SMF) every Nth failed modem
# prepare attempt, regardless of whether a PFCP failure was detected in the logs.
# Set to 0 to disable.
OPEN5GS_RESTART_EVERY_N_FAILURES="${OPEN5GS_RESTART_EVERY_N_FAILURES:-5}"
# Abort the modem-prepare retry loop immediately if the gNB log shows it rejected
# the UE right after random access (e.g. it could not allocate SRS/PUCCH/CSI
# resources for the swept cell config). That surfaces as a modem "no-service",
# but it is a gNB/cell-config problem: retrying the modem or restarting the core
# cannot fix it. Set to false to keep retrying instead.
GNB_ABORT_ON_UE_ADMISSION_FAILURE="${GNB_ABORT_ON_UE_ADMISSION_FAILURE:-true}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${RUN_DIR}/logs/workflow_${RUN_ID}"
GNB_STDOUT_LOG="${LOG_DIR}/gnb_stdout.log"
POWER_LOG="${LOG_DIR}/ina219_power_recorder.log"

gnb_pid=""
gnb_existing_pid=""
gnb_pid_to_monitor=""
power_ssh_pid=""
gnb_started_by_workflow=false
gnb_start_epoch=""
youtube_start_epoch=""
gnb_failed_early=false
gnb_management_active=false
capture_dir=""
capture_dir_override=""
transfer_capture_dir=""
gnb_log_copied=false
gnb_shutdown_done=false
srsran_calibration_window_seconds=""
microsleep_power_mw="${DEFAULT_3GPP_MICROSLEEP_POWER_MW}"
duration=""
capture_name=""
youtube_url=""
chrome_activity="youtube"     # chrome sub-mode: youtube | instagram | x | slither | zoom
chrome_activity_explicit=false
chrome_activity_seed="1"
zoom_meeting_id="${ZOOM_MEETING_ID:-658 354 2374}"  # zoom sub-mode meeting id
chrome_activity_pid=""
capture_mode=""              # chrome | synthetic; resolved interactively if empty
DATA_PATH_AGENT="${CHROME_AGENT}"
synthetic_profile="sweep"
synthetic_seed=""
synthetic_dl_cap_kbps="40000"
synthetic_ul_cap_kbps="10000"
synthetic_start_epoch=""
synthetic_schedule_file=""
synthetic_schedule_hash=""
synthetic_player_pid=""
non_interactive=false
force_gnb_restart=false
stop_chrome_on_exit=true
runtime_started=false
declare -a cli_gnb_config_files=()
declare -a gnb_config_files=()

say() {
  printf '\n[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

section() {
  printf '\n=================== %s ===================\n' "$1"
}

die() {
  say "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Run one srsRAN gNB + INA219 capture, then parse the gNB log into per-slot
features, generate the TR 38.840 power estimate and the gNB-log/power-meter
clock-alignment offset. No XGBoost training happens here; run that separately
with 'python -m scripts.experiments run'.

Options:
  --duration SECONDS              Power recording duration.
  --calibration-window-s SECONDS  srsRAN clock-alignment calibration window
                                  (default: full capture duration).
  --gnb-config FILE               srsRAN gNB config file; repeatable (base config
                                  first, then overlays). GNB_CONFIG, if set, is
                                  loaded before these.
  --capture-name NAME             Capture folder name.
  --capture-dir DIR               Final capture directory. The INA219 transfer
                                  is staged under LOCAL_CAPTURE_BASE, then moved.
  --youtube-url URL               Video for --chrome-activity youtube.
  --chrome-activity NAME          Chrome sub-mode: youtube, instagram, x,
                                  slither, or zoom (default: youtube).
                                  instagram/x/slither/zoom drive the page via the
                                  ue-chrome-activity CDP helper; instagram/x need
                                  a logged-in persistent Chrome profile, slither
                                  (slither.io) and zoom need no login.
  --chrome-activity-seed N        Seed for the instagram/x/slither input cadence
                                  (default: 1) for approximate reproducibility.
  --zoom-meeting-id ID            Meeting id for --chrome-activity zoom
                                  (default: 658 354 2374). Joins in the browser.
  --mode MODE                     chrome (real app in Chrome) or synthetic
                                  (scripted UDP traffic via the ue-traffic
                                  namespace). Prompted interactively if omitted.
  --synthetic-profile NAME        Traffic profile from synthetic_traffic/profiles.py,
                                  e.g. youtube, social_scroll, web_browse,
                                  video_call, sweep, dl_pulse (default: sweep).
  --synthetic-seed N              Seed for a reproducible schedule (default: random).
  --synthetic-dl-cap-kbps KBPS    DL rate ceiling (default: 40000).
  --synthetic-ul-cap-kbps KBPS    UL rate ceiling (default: 10000).
  --force-gnb-restart             Stop any existing gNB before this run.
  --microsleep-power-mw MW        3GPP baseline microsleep power floor in mW
                                  (default: 400). Prompted if omitted.
  --non-interactive               Fail instead of prompting for missing values.
  -h, --help                      Show this help.

Host-specific environment overrides:
  SRSRAN_DIR, GNB_BIN, GNB_CONFIG, GNB_RUN_LOG, RUN_DIR, PYTHON_BIN, CHROME_PROFILE_DIR, CHROME_USER,
  PI_USER, PI_HOST, PI_INA_DIR, LOCAL_CAPTURE_BASE.
EOF
}

parse_args() {
  while (($#)); do
    case "$1" in
      --duration) duration="${2:?missing value for --duration}"; shift 2 ;;
      --calibration-window-s) srsran_calibration_window_seconds="${2:?missing value for --calibration-window-s}"; shift 2 ;;
      --microsleep-power-mw) microsleep_power_mw="${2:?missing value for --microsleep-power-mw}"; shift 2 ;;
      --gnb-config) cli_gnb_config_files+=("${2:?missing value for --gnb-config}"); shift 2 ;;
      --capture-name) capture_name="${2:?missing value for --capture-name}"; shift 2 ;;
      --capture-dir) capture_dir_override="${2:?missing value for --capture-dir}"; shift 2 ;;
      --youtube-url) youtube_url="${2:?missing value for --youtube-url}"; shift 2 ;;
      --chrome-activity) chrome_activity="${2:?missing value for --chrome-activity}"; chrome_activity_explicit=true; shift 2 ;;
      --chrome-activity-seed) chrome_activity_seed="${2:?missing value for --chrome-activity-seed}"; shift 2 ;;
      --zoom-meeting-id) zoom_meeting_id="${2:?missing value for --zoom-meeting-id}"; shift 2 ;;
      --mode) capture_mode="${2:?missing value for --mode}"; shift 2 ;;
      --synthetic-profile) synthetic_profile="${2:?missing value for --synthetic-profile}"; shift 2 ;;
      --synthetic-seed) synthetic_seed="${2:?missing value for --synthetic-seed}"; shift 2 ;;
      --synthetic-dl-cap-kbps) synthetic_dl_cap_kbps="${2:?missing value for --synthetic-dl-cap-kbps}"; shift 2 ;;
      --synthetic-ul-cap-kbps) synthetic_ul_cap_kbps="${2:?missing value for --synthetic-ul-cap-kbps}"; shift 2 ;;
      --force-gnb-restart) force_gnb_restart=true; shift ;;
      --non-interactive) non_interactive=true; shift ;;
      --keep-chrome) stop_chrome_on_exit=false; shift ;;
      -h|--help) usage; trap - EXIT INT TERM; exit 0 ;;
      *) die "Unknown option: $1" ;;
    esac
  done
}

wait_for_enter() {
  local prompt="$1"
  printf '\n%s\n' "$prompt"
  read -r -p "Press Enter when done..." _
}

prompt_yes_no() {
  local prompt="$1"
  local default_answer="$2"
  local answer suffix
  if [[ "${default_answer}" == "yes" ]]; then
    suffix="[Y/n]"
  else
    suffix="[y/N]"
  fi
  read -r -p "${prompt} ${suffix}: " answer
  answer="${answer:-${default_answer}}"
  case "${answer}" in
    y|Y|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

duration_since_epoch() {
  local start_epoch="$1"
  local end_epoch="${2:-$(date +%s)}"
  local delta=$(( end_epoch - start_epoch ))
  format_seconds "${delta}"
}

format_seconds() {
  local delta="$1"
  if (( delta < 0 )); then
    delta=0
  fi
  local hours=$(( delta / 3600 ))
  local minutes=$(( (delta % 3600) / 60 ))
  local seconds=$(( delta % 60 ))
  printf '%02d:%02d:%02d' "${hours}" "${minutes}" "${seconds}"
}

find_gnb_pids() {
  ps -eo pid=,args= |
    awk '
      /(^|[\/ ])gnb([ ]|$)/ &&
      !/capture.sh/ &&
      !/awk / {
        print $1
      }
    ' |
    tr '\n' ' ' |
    sed 's/[[:space:]]*$//'
}

stop_existing_gnb() {
  section "STOP EXISTING GNB"
  local pids="$1"
  say "Stopping existing gNB process(es): ${pids}"
  # shellcheck disable=SC2086
  kill -INT ${pids} 2>/dev/null || true
  for _ in {1..15}; do
    # shellcheck disable=SC2086
    if ! kill -0 ${pids} 2>/dev/null; then
      return
    fi
    sleep 1
  done
  say "Existing gNB did not stop after SIGINT; sending SIGTERM."
  # shellcheck disable=SC2086
  kill -TERM ${pids} 2>/dev/null || true
}

shutdown_gnb_for_workflow_end() {
  if [[ "${gnb_shutdown_done}" == true ]]; then
    return 0
  fi
  gnb_shutdown_done=true

  section "SHUTDOWN GNB"

  if [[ -n "${gnb_pid}" ]] && kill -0 "${gnb_pid}" 2>/dev/null; then
    say "Stopping workflow-started gNB PID ${gnb_pid}."
    stop_existing_gnb "${gnb_pid}"
    wait "${gnb_pid}" 2>/dev/null || true
    return 0
  fi

  if [[ -n "${gnb_pid_to_monitor}" ]] && kill -0 "${gnb_pid_to_monitor}" 2>/dev/null; then
    say "Stopping monitored gNB PID ${gnb_pid_to_monitor}."
    stop_existing_gnb "${gnb_pid_to_monitor}"
    return 0
  fi

  local remaining_pids
  remaining_pids="$(find_gnb_pids)"
  if [[ -n "${remaining_pids}" ]]; then
    say "Stopping remaining gNB process(es): ${remaining_pids}"
    stop_existing_gnb "${remaining_pids}"
  else
    say "No running gNB process found."
  fi
}

resolve_gnb_configs() {
  local input cfg
  local -a requested=()
  [[ -n "${GNB_CONFIG}" ]] && requested+=("${GNB_CONFIG}")
  requested+=("${cli_gnb_config_files[@]}")

  if (( ${#requested[@]} == 0 )); then
    [[ "${non_interactive}" != true ]] ||
      die "No gNB config given; pass --gnb-config FILE (repeatable) or set GNB_CONFIG."
    read -r -p "gNB config file(s), base first (space-separated): " input
    read -r -a requested <<< "${input}"
    (( ${#requested[@]} > 0 )) || die "At least one gNB config file is required."
  fi

  for cfg in "${requested[@]}"; do
    [[ -f "${cfg}" ]] || die "gNB config file does not exist: ${cfg}"
    gnb_config_files+=("$(realpath "${cfg}")")
  done
  say "gNB config file list for this run:"
  printf '  - %s
' "${gnb_config_files[@]}"
}

start_gnb() {
  section "START GNB"
  local -a gnb_command=("${GNB_BIN}")
  local cfg

  for cfg in "${gnb_config_files[@]}"; do
    gnb_command+=("-c" "${cfg}")
  done

  say "Starting gNB in ${RUN_DIR}:"
  printf '  %q' "${gnb_command[@]}"
  printf '
'
  (
    cd "${RUN_DIR}"
    exec "${gnb_command[@]}"
  ) > >(tee -a "${GNB_STDOUT_LOG}") 2>&1 &
  gnb_pid=$!
  gnb_started_by_workflow=true
  gnb_start_epoch="$(date +%s)"
  gnb_pid_to_monitor="${gnb_pid}"
  say "gNB PID: ${gnb_pid}"
  sleep 2
  if ! kill -0 "${gnb_pid}" 2>/dev/null; then
    die "gNB exited early. See ${GNB_STDOUT_LOG}"
  fi
}

clear_gnb_run_log() {
  section "CLEAR GNB LOG"
  mkdir -p "$(dirname "${GNB_RUN_LOG}")"
  : > "${GNB_RUN_LOG}"
  say "Cleared gNB run log for this workflow:"
  printf '  %s\n' "${GNB_RUN_LOG}"
}

handle_existing_or_start_gnb() {
  section "GNB CHECK"
  local existing_pids
  gnb_management_active=true
  existing_pids="$(find_gnb_pids)"
  if [[ -z "${existing_pids}" ]]; then
    start_gnb
    return
  fi

  if [[ "${force_gnb_restart}" == true ]]; then
    stop_existing_gnb "${existing_pids}"
    start_gnb
    return
  fi

  gnb_existing_pid="${existing_pids%% *}"
  if [[ "${non_interactive}" == true ]]; then
    die "Detected already-running gNB process(es) [${existing_pids}] in --non-interactive mode. Pass --force-gnb-restart to restart it, or stop it before launching this workflow."
  fi
  cat <<EOF

Detected already-running gNB process(es):
  ${existing_pids}

Do you want this workflow to stop the existing gNB and restart it with this
run's configuration before proceeding?

Default is NO. If you choose NO, this workflow will not know the original gNB
start time and cannot calculate "gNB alive time since gNB start". It will only
be able to report time since YouTube start if the monitored gNB process exits.

EOF
  read -r -p "Restart gNB? [y/N]: " restart_answer
  case "${restart_answer}" in
    y|Y|yes|YES)
      stop_existing_gnb "${existing_pids}"
      start_gnb
      ;;
    *)
      gnb_pid_to_monitor="${gnb_existing_pid}"
      gnb_started_by_workflow=false
      gnb_start_epoch=""
      say "Proceeding with already-running gNB PID ${gnb_pid_to_monitor}."
      say "I will NOT be able to calculate gNB alive time since gNB start because it was already running."
      ;;
  esac
}

chrome_running() {
  local targets

  systemctl is-active --quiet "${CHROME_SERVICE}" || return 1
  pgrep -u "${CHROME_USER}" -f -- "--user-data-dir=${CHROME_PROFILE_DIR}" >/dev/null || return 1

  targets="$(sudo ip netns exec ue-browser curl -s --max-time 3 http://127.0.0.1:9223/json 2>/dev/null || true)"
  if [[ -z "${targets}" || "${targets}" == "[  ]" || "${targets}" == "[]" ]]; then
    return 1
  fi

  printf '%s\n' "${targets}" | grep -q '"type"[[:space:]]*:[[:space:]]*"page"'
}

turn_modem_airplane_mode_on() {
  section "MODEM AIRPLANE CYCLE"
  cat <<EOF

Turning modem airplane mode ON now, immediately after the gNB check/start.
This uses QMI DMS operating mode:
  - low-power  = airplane mode on
  - online     = airplane mode off

The modem connectivity prepare stage will turn airplane mode OFF, reconnect the
data session, and verify traffic.

Airplane mode ON duration:
  ${MODEM_AIRPLANE_OFF_SECONDS}s

EOF

  sudo systemctl stop "${CHROME_SERVICE}" 2>/dev/null || true
  sudo qmicli -d /dev/cdc-wdm0 --device-open-proxy --dms-get-operating-mode || true
  sudo qmicli -d /dev/cdc-wdm0 --device-open-proxy --dms-set-operating-mode=low-power
  say "Modem airplane mode is ON. Waiting ${MODEM_AIRPLANE_OFF_SECONDS}s."
  sleep "${MODEM_AIRPLANE_OFF_SECONDS}"
}

recent_open5gs_pfcp_session_failure() {
  local since_epoch
  since_epoch=$(( $(date +%s) - OPEN5GS_PFCP_LOG_LOOKBACK_SECONDS ))

  sudo journalctl -u open5gs-smfd -u open5gs-upfd -u open5gs-amfd \
    --since "@${since_epoch}" --no-pager 2>/dev/null |
    grep -Eiq 'No PFCP session establishment response|PDU session establishment reject|cannot handle PFCP message type\[50\]|LOCAL[[:space:]]+No Reponse'
}

restart_open5gs_user_plane() {
  local reason="$1"
  say "${reason}"
  say "Restarting Open5GS UPF and SMF, then waiting ${OPEN5GS_RESTART_SETTLE_SECONDS}s for PFCP association."
  sudo systemctl restart open5gs-upfd open5gs-smfd
  sleep "${OPEN5GS_RESTART_SETTLE_SECONDS}"
  sudo journalctl -u open5gs-smfd -u open5gs-upfd \
    --since "-${OPEN5GS_RESTART_SETTLE_SECONDS} seconds" --no-pager 2>/dev/null |
    grep -Ei 'PFCP associated|PFCP de-associated|pfcp_server|No PFCP session establishment response' || true
}

restart_open5gs_user_plane_for_prepare() {
  restart_open5gs_user_plane \
    "Detected recent Open5GS PFCP/PDU-session failure while preparing modem data path."
}

# Full-core restart (AMF + SMF + UPF). The user-plane restart above cannot clear a
# wedged AMF: a stuck NGAP/SM state surfaces as the AMF sending ErrorIndication
# (e.g. Cause: semantic-error) right after Initial Context Setup, so the UE attaches
# and is released in a loop and never gets a DRB/PDU session. Restarting amfd too
# forces the gNB to re-run NG setup and clears that state.
restart_open5gs_core() {
  local reason="$1"
  say "${reason}"
  say "Restarting Open5GS AMF, SMF, and UPF, then waiting ${OPEN5GS_RESTART_SETTLE_SECONDS}s for the core to settle."
  sudo systemctl restart open5gs-amfd open5gs-smfd open5gs-upfd
  sleep "${OPEN5GS_RESTART_SETTLE_SECONDS}"
  sudo journalctl -u open5gs-amfd -u open5gs-smfd -u open5gs-upfd \
    --since "-${OPEN5GS_RESTART_SETTLE_SECONDS} seconds" --no-pager 2>/dev/null |
    grep -Ei 'PFCP associated|PFCP de-associated|pfcp_server|NG setup|NGSetup|No PFCP session establishment response' || true
}

prepare_modem_data_path_with_retry() {
  section "MODEM CONNECTIVITY PREPARE"
  local start_epoch now elapsed attempt prepare_output core_restarted_for_pfcp_failure
  local gnb_log_baseline admission_failure

  # Remember how far the gNB log has grown so the UE-admission check below only
  # inspects lines produced while this prepare loop is running.
  gnb_log_baseline="$(stat -c %s "${GNB_RUN_LOG}" 2>/dev/null || echo 0)"

  sudo qmicli -d /dev/cdc-wdm0 --device-open-proxy --dms-set-operating-mode=online
  say "Modem airplane mode is OFF."
  sudo qmicli -d /dev/cdc-wdm0 --device-open-proxy --dms-get-operating-mode || true

  cat <<EOF

Before generating traffic, this workflow will verify the modem/UE data path
using the data-path agent procedure:
  sudo "${DATA_PATH_AGENT}" prepare

EOF

  say "Preparing modem/UE data path before generating traffic."
  say "This reuses the data-path agent prepare and will retry for up to ${MODEM_PREPARE_TIMEOUT_SECONDS}s."
  say "Waiting ${MODEM_PREPARE_RETRY_SECONDS}s..."
  sleep "${MODEM_PREPARE_RETRY_SECONDS}"

  start_epoch="$(date +%s)"
  attempt=1
  core_restarted_for_pfcp_failure=false

  while true; do
    now="$(date +%s)"
    elapsed=$(( now - start_epoch ))
    say "Modem prepare attempt ${attempt} at +${elapsed}s."

    prepare_output="$(mktemp)"
    if sudo "${DATA_PATH_AGENT}" prepare 2>&1 | tee "${prepare_output}"; then
      rm -f "${prepare_output}"
      say "Modem/UE data path is ready."
      return
    fi

    # If the gNB admitted the UE at RACH but then rejected it (resource
    # allocation failure for the swept cell config), the modem will report
    # no-service forever. Detect that and abort with the real cause instead of
    # burning the full retry window and (mis)restarting the core.
    if [[ "${GNB_ABORT_ON_UE_ADMISSION_FAILURE}" =~ ^(1|true|yes|y)$ ]]; then
      admission_failure="$(
        tail -c "+$(( gnb_log_baseline + 1 ))" "${GNB_RUN_LOG}" 2>/dev/null |
          grep -aE 'Failed to create a configuration for ue=' | tail -n 1 || true
      )"
      if [[ -n "${admission_failure}" ]]; then
        rm -f "${prepare_output}"
        say "gNB rejected the UE right after random access; this is a gNB/cell-config"
        say "problem (not the modem). The modem reaches RACH but cannot be admitted,"
        say "so it reports no-service. Retrying the modem or the core will not help."
        say "gNB: ${admission_failure#*] }"
        die "Aborting: gNB UE admission failed (see ${GNB_RUN_LOG}). Check the swept cell config (e.g. SRS period)."
      fi
    fi

    local restarted_core_this_attempt=false
    if [[ "${OPEN5GS_RESTART_ON_PFCP_FAILURE}" =~ ^(1|true|yes|y)$ ]] &&
       [[ "${core_restarted_for_pfcp_failure}" != true ]] &&
       recent_open5gs_pfcp_session_failure; then
      core_restarted_for_pfcp_failure=true
      restart_open5gs_user_plane_for_prepare
      restarted_core_this_attempt=true
    fi
    # Independent safety net: a QMI CallFailed/no-service often means the core is
    # wedged without an obvious PFCP log line. The fault may be in the control
    # plane (AMF emitting NGAP ErrorIndication after Initial Context Setup, so the
    # UE never gets a DRB) and not just the user plane, so restart the FULL core
    # (AMF + SMF + UPF) every Nth failed attempt even when no PFCP failure was
    # detected above.
    if [[ "${restarted_core_this_attempt}" != true ]] &&
       (( OPEN5GS_RESTART_EVERY_N_FAILURES > 0 )) &&
       (( attempt % OPEN5GS_RESTART_EVERY_N_FAILURES == 0 )); then
      restart_open5gs_core \
        "Modem prepare has failed ${attempt} attempt(s); restarting the full Open5GS core (AMF, SMF, UPF) in case the control or user plane is wedged."
    fi
    rm -f "${prepare_output}"

    now="$(date +%s)"
    elapsed=$(( now - start_epoch ))
    if (( elapsed >= MODEM_PREPARE_TIMEOUT_SECONDS )); then
      die "Modem/UE data path did not become ready within ${MODEM_PREPARE_TIMEOUT_SECONDS}s."
    fi

    say "Modem/UE data path is not ready yet. Waiting ${MODEM_PREPARE_RETRY_SECONDS}s before retrying."
    sleep "${MODEM_PREPARE_RETRY_SECONDS}"
    attempt=$(( attempt + 1 ))
  done
}

check_chrome_or_wait() {
  section "CHROME SERVICE"
  if chrome_running; then
    say "Chrome-over-UE is already running (${CHROME_SERVICE} active and usable Chrome page target found)."
    return
  fi

  cat <<EOF

Chrome-over-UE does not appear to be running.

This workflow checks the actual Chrome service name: ${CHROME_SERVICE}.
That service is managed by the ue-chrome-agent helper script.

The workflow will now start Chrome with:
  sudo "${CHROME_AGENT}" chrome

EOF
  sudo "${CHROME_AGENT}" chrome

  if ! chrome_running; then
    systemctl --no-pager --lines=10 status "${CHROME_SERVICE}" || true
    die "Chrome-over-UE is still not running after sudo ue-chrome-agent.sh prepare/chrome."
  fi

  say "Confirmed: Chrome-over-UE is running."
}

url_encode() {
  python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$1"
}

ensure_chrome_login() {
  # For instagram/x, probe the running ue-browser Chrome for a login wall. If one
  # is present and we are interactive, pause and ask the operator to log in inside
  # the open UE Chrome window, then re-probe when they press Enter. Loop until the
  # wall clears. The login session persists in CHROME_PROFILE_DIR across runs.
  [[ "${chrome_activity}" == youtube || "${chrome_activity}" == zoom ]] && return 0

  section "CHROME LOGIN CHECK: ${chrome_activity}"
  local rc
  while true; do
    set +e
    sudo ip netns exec ue-browser "${PYTHON_BIN}" "${CHROME_ACTIVITY_AGENT}" \
      --activity "${chrome_activity}" --check-login \
      --cdp-host 127.0.0.1 --cdp-port 9223
    rc=$?
    set -e

    case "${rc}" in
      0)
        say "No login wall for ${chrome_activity}; the profile is logged in. Proceeding."
        return 0
        ;;
      3)
        if [[ "${non_interactive}" == true ]]; then
          die "Login wall for ${chrome_activity} while running --non-interactive. Log into the persistent Chrome profile (${CHROME_PROFILE_DIR}) once, then re-run."
        fi
        cat <<EOF

A login wall was detected for ${chrome_activity}.

A Chrome window routed over the UE link is open on this desktop, showing the
${chrome_activity} page. Log in there now (complete any 2FA). The session is
saved into the persistent profile (${CHROME_PROFILE_DIR}) and reused on future
runs, so you only need to do this once per site.

EOF
        wait_for_enter "Log into ${chrome_activity} in the open Chrome window, then continue."
        say "Re-checking ${chrome_activity} login state..."
        ;;
      *)
        die "Login check for ${chrome_activity} failed (exit ${rc}); see the helper output above."
        ;;
    esac
  done
}

start_chrome_activity() {
  # Drive instagram/x in the already-running ue-browser Chrome via the CDP helper.
  # Runs in the background for the whole capture (the real-app analogue of the
  # synthetic traffic player); cleanup kills it. Fails the capture if the helper
  # exits early (e.g. a detected login wall) so we never record a login page.
  local report="${LOG_DIR}/chrome_activity_${chrome_activity}.json"
  local activity_duration=$(( ${duration%.*} + 60 ))   # outlive the power window
  local -a activity_cmd=(
    sudo ip netns exec ue-browser "${PYTHON_BIN}" "${CHROME_ACTIVITY_AGENT}"
    --activity "${chrome_activity}" --seed "${chrome_activity_seed}"
    --zoom-meeting-id "${zoom_meeting_id}"
    --duration "${activity_duration}" --report "${report}"
  )

  # zoom (interactive): park the join at the display-name screen and wait for
  # operator approval before power recording, via a ready/go file handshake with
  # the driver. Other activities (and --non-interactive) start recording after
  # the usual 8s settle.
  local zoom_pause=false zoom_ready_file="" zoom_go_file=""
  if [[ "${chrome_activity}" == zoom && "${non_interactive}" != true ]]; then
    zoom_pause=true
    zoom_ready_file="${LOG_DIR}/zoom_prejoin_ready"
    zoom_go_file="${LOG_DIR}/zoom_go"
    rm -f "${zoom_ready_file}" "${zoom_go_file}"
    activity_cmd+=(--zoom-pause-ready "${zoom_ready_file}" --zoom-pause-go "${zoom_go_file}")
  fi

  say "Driving Chrome activity '${chrome_activity}' (seed ${chrome_activity_seed}) over the ue-browser namespace."
  say "Activity report: ${report}"
  "${activity_cmd[@]}" \
    > >(tee -a "${LOG_DIR}/chrome_activity_${chrome_activity}.log") 2>&1 &
  chrome_activity_pid=$!
  say "Chrome activity PID: ${chrome_activity_pid}."

  if [[ "${zoom_pause}" == true ]]; then
    say "Waiting for the Zoom join to reach the display-name screen..."
    while [[ ! -f "${zoom_ready_file}" ]]; do
      kill -0 "${chrome_activity_pid}" 2>/dev/null ||
        die "Zoom driver exited before reaching the name screen (see ${LOG_DIR}/chrome_activity_${chrome_activity}.log)."
      sleep 1
    done
    cat <<EOF

The Zoom join is parked at the display-name screen.
  Display name typed: $(cat "${zoom_ready_file}" 2>/dev/null)
Review the open Chrome window. Approving will click Join and immediately start
INA219 power recording.

EOF
    wait_for_enter "Approve to join the Zoom call and start power recording."
    : > "${zoom_go_file}"
    youtube_start_epoch="$(date +%s)"   # reused as the activity-start epoch
    say "Approval sent; Zoom is joining. Settling 8s before power recording."
  else
    youtube_start_epoch="$(date +%s)"   # reused as the activity-start epoch
    say "Settling 8s before power recording."
  fi
  sleep 8

  kill -0 "${chrome_activity_pid}" 2>/dev/null ||
    die "Chrome activity driver exited early (see ${LOG_DIR}/chrome_activity_${chrome_activity}.log). A login wall is the likely cause: log into the persistent Chrome profile (${CHROME_PROFILE_DIR}) once, then retry. Use --allow-login-wall in the helper to override."
}

open_youtube_video_in_chrome() {
  local url="$1"
  local encoded_url target_json

  encoded_url="$(url_encode "${url}")"
  say "Opening YouTube video through UE-routed Chrome:"
  printf '  %s\n' "${url}"

  target_json="$(sudo ip netns exec ue-browser curl -sS --max-time 8 -X PUT \
    "http://127.0.0.1:9223/json/new?${encoded_url}")" || {
    echo "Failed to ask Chrome remote-debugging endpoint to open YouTube URL." >&2
    exit 1
  }

  if ! printf '%s\n' "${target_json}" | grep -q '"webSocketDebuggerUrl"'; then
    echo "Chrome remote-debugging endpoint did not return a page target." >&2
    printf '%s\n' "${target_json}" >&2
    exit 1
  fi
}

resolve_capture_mode() {
  if [[ -z "${capture_mode}" ]]; then
    if [[ "${non_interactive}" == true ]]; then
      capture_mode="chrome"   # backward-compatible default
    else
      printf '\nCapture traffic source:\n'
      printf '  1) chrome     - real app traffic via Chrome (YouTube)\n'
      printf '  2) synthetic  - scripted UDP traffic via the ue-traffic namespace\n'
      read -r -p "Select [1/2] (default 1): " mode_answer
      case "${mode_answer}" in
        2|synthetic) capture_mode="synthetic" ;;
        *) capture_mode="chrome" ;;
      esac
    fi
  fi
  case "${capture_mode}" in
    chrome)
      DATA_PATH_AGENT="${CHROME_AGENT}"
      if [[ "${non_interactive}" != true && "${chrome_activity_explicit}" != true ]]; then
        printf '\nChrome activity:\n'
        printf '  1) youtube    - watch a YouTube video (default)\n'
        printf '  2) instagram  - scroll Instagram reels (needs logged-in profile)\n'
        printf '  3) x          - scroll the X timeline (needs logged-in profile)\n'
        printf '  4) slither    - play slither.io, a high-throughput .io game (no login)\n'
        printf '  5) zoom       - join a Zoom call in the browser (no login)\n'
        read -r -p "Select [1/2/3/4/5] (default 1): " act_answer
        case "${act_answer}" in
          2|instagram) chrome_activity="instagram" ;;
          3|x) chrome_activity="x" ;;
          4|slither) chrome_activity="slither" ;;
          5|zoom) chrome_activity="zoom" ;;
          *) chrome_activity="youtube" ;;
        esac
      fi
      case "${chrome_activity}" in
        youtube|instagram|x|slither|zoom) ;;
        *) die "--chrome-activity must be youtube, instagram, x, slither, or zoom" ;;
      esac
      [[ "${chrome_activity}" == youtube ]] ||
        [[ -f "${CHROME_ACTIVITY_AGENT}" ]] ||
        die "Missing chrome activity driver: ${CHROME_ACTIVITY_AGENT}"
      if [[ "${chrome_activity}" == youtube && -z "${youtube_url}" ]]; then
        [[ "${non_interactive}" != true ]] ||
          die "--youtube-url is required for --chrome-activity youtube."
        read -r -p "YouTube video URL: " youtube_url
        [[ -n "${youtube_url}" ]] || die "A YouTube video URL is required."
      fi
      ;;
    synthetic)
      DATA_PATH_AGENT="${TRAFFIC_AGENT}"
      [[ -x "${TRAFFIC_AGENT}" ]] || die "Missing executable traffic agent: ${TRAFFIC_AGENT}"
      [[ -f "${SYNTHETIC_TRAFFIC_DIR}/schedule_gen.py" ]] ||
        die "Missing schedule generator: ${SYNTHETIC_TRAFFIC_DIR}/schedule_gen.py"
      if [[ "${non_interactive}" != true ]]; then
        read -r -p "Synthetic profile [${synthetic_profile}]: " input_profile
        synthetic_profile="${input_profile:-${synthetic_profile}}"
        if [[ -z "${synthetic_seed}" ]]; then
          read -r -p "Synthetic seed [1]: " synthetic_seed
        fi
      fi
      synthetic_seed="${synthetic_seed:-1}"
      ;;
    *) die "--mode must be chrome or synthetic" ;;
  esac
  say "Capture mode: ${capture_mode}"
  [[ "${capture_mode}" == chrome ]] && say "Chrome activity: ${chrome_activity} (seed ${chrome_activity_seed})"
  if [[ "${capture_mode}" == synthetic ]]; then
    say "Synthetic profile: ${synthetic_profile}; seed: ${synthetic_seed}; caps: DL ${synthetic_dl_cap_kbps}k / UL ${synthetic_ul_cap_kbps}k."
  fi
}

ensure_dn_server() {
  # The persistent-UDP player needs its DN daemon up on this host (the one that
  # owns ogstun). It is a long-running systemd unit, so a healthy one is reused
  # across runs; start it only if absent.
  [[ "${SYNTHETIC_PLAYER}" == udp_traffic_player.py ]] || return 0
  [[ "${SYNTHETIC_AUTOSTART_DN}" == true ]] || return 0
  if systemctl is-active --quiet udp-dn-server.service; then
    say "Persistent-UDP DN server already running."
    return 0
  fi
  [[ -x "${DN_SERVER_AGENT}" ]] || die "Missing DN server script: ${DN_SERVER_AGENT}"
  say "Persistent-UDP DN server not running; starting it (udp-dn-server.sh)."
  sudo "${DN_SERVER_AGENT}" start || die "Failed to start udp-dn-server (see its output above)."
}

start_synthetic_traffic() {
  ensure_dn_server
  synthetic_schedule_file="${LOG_DIR}/synthetic_schedule.json"
  say "Generating synthetic schedule (profile=${synthetic_profile} seed=${synthetic_seed} duration=${duration}s)."
  "${PYTHON_BIN}" "${SYNTHETIC_TRAFFIC_DIR}/schedule_gen.py" \
    --profile "${synthetic_profile}" --seed "${synthetic_seed}" \
    --duration "${duration}" \
    --tick "${SYNTHETIC_TICK}" --min-seg "${SYNTHETIC_MIN_SEG}" \
    --dl-cap "${synthetic_dl_cap_kbps}" --ul-cap "${synthetic_ul_cap_kbps}" \
    --out "${synthetic_schedule_file}"
  synthetic_schedule_hash="$(
    "${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("schedule_hash",""))' \
      "${synthetic_schedule_file}"
  )"
  # Mirror the schedule (json + pdf + jpg) into the synthetic_traffic/schedules gallery,
  # one folder per schedule named by profile+duration+seed, so every run leaves a
  # reusable, plotted schedule grouped together there.
  synthetic_gallery_name="${synthetic_profile}_${duration}s_s${synthetic_seed}"
  synthetic_gallery_dir="${SYNTHETIC_TRAFFIC_DIR}/schedules/${synthetic_gallery_name}"
  mkdir -p "${synthetic_gallery_dir}"
  synthetic_gallery_base="${synthetic_gallery_dir}/${synthetic_gallery_name}"
  cp -f "${synthetic_schedule_file}" "${synthetic_gallery_base}.json"
  "${PYTHON_BIN}" "${SYNTHETIC_TRAFFIC_DIR}/plot_schedule.py" \
    "${synthetic_gallery_base}.json" "${synthetic_gallery_base}.pdf" \
    || say "WARNING: schedule gallery plot failed (capture still complete)."
  say "Starting synthetic traffic player through ue-traffic for ${duration}s."
  sudo "${PYTHON_BIN}" "${SYNTHETIC_TRAFFIC_DIR}/${SYNTHETIC_PLAYER}" \
    --schedule "${synthetic_schedule_file}" --ns ue-traffic \
    --results "${LOG_DIR}/synthetic_results.json" \
    > >(tee -a "${LOG_DIR}/synthetic_traffic.log") 2>&1 &
  synthetic_player_pid=$!
  synthetic_start_epoch="$(date +%s)"
  say "Traffic player PID: ${synthetic_player_pid}. Settling 2s before power recording."
  sleep 2
}

stop_remote_power_recorder_if_running() {
  ssh -o BatchMode=yes -o ConnectTimeout=5 "${PI_USER}@${PI_HOST}" \
    'pid_file=/tmp/signal_safe_ina219_logger.pid;
     reason_file=/tmp/signal_safe_ina219_logger_halt_reason;
     if test -s "$pid_file"; then
       printf "%s\n%s\n" "$(date --iso-8601=seconds)" "workflow cleanup or duration reached" > "$reason_file";
       kill -INT "$(cat "$pid_file")" 2>/dev/null || true;
     fi' >/dev/null 2>&1 || true
}

copy_gnb_log_to_capture_dir() {
  local destination_dir="$1"
  local destination_log="${destination_dir}/srsRAN/gnb_n41.log"

  if [[ -z "${destination_dir}" ]]; then
    return 0
  fi

  mkdir -p "${destination_dir}/srsRAN"
  if [[ -s "${GNB_RUN_LOG}" ]]; then
    cp -f -- "${GNB_RUN_LOG}" "${destination_log}"
    gnb_log_copied=true
    say "Copied gNB log to capture folder:"
    printf '  %s\n' "${destination_log}"
  else
    say "WARNING: gNB log is missing or empty; nothing copied:"
    printf '  %s\n' "${GNB_RUN_LOG}"
  fi
}

finalize_gnb_capture() {
  shutdown_gnb_for_workflow_end

  section "COPY GNB LOG"
  copy_gnb_log_to_capture_dir "${capture_dir}"
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM

  if [[ -n "${power_ssh_pid}" ]] && kill -0 "${power_ssh_pid}" 2>/dev/null; then
    say "Cleanup: stopping RPi INA219 recorder."
    stop_remote_power_recorder_if_running
    wait "${power_ssh_pid}" 2>/dev/null || true
  fi

  if [[ "${runtime_started}" == true ]]; then
    shutdown_gnb_for_workflow_end || true
  fi

  if [[ -n "${chrome_activity_pid}" ]] && kill -0 "${chrome_activity_pid}" 2>/dev/null; then
    say "Cleanup: stopping Chrome activity driver."
    sudo kill "${chrome_activity_pid}" 2>/dev/null || true
    wait "${chrome_activity_pid}" 2>/dev/null || true
  fi

  if [[ "${runtime_started}" == true && "${capture_mode:-}" == synthetic ]]; then
    if [[ -n "${synthetic_player_pid}" ]] && kill -0 "${synthetic_player_pid}" 2>/dev/null; then
      say "Cleanup: stopping synthetic traffic player."
      sudo kill "${synthetic_player_pid}" 2>/dev/null || true
      wait "${synthetic_player_pid}" 2>/dev/null || true
    fi
  elif [[ "${runtime_started}" == true && "${stop_chrome_on_exit}" == true ]]; then
    say "Cleanup: stopping Chrome-over-UE service."
    sudo "${CHROME_AGENT}" stop >/dev/null 2>&1 || sudo systemctl stop "${CHROME_SERVICE}" 2>/dev/null || true
  fi

  if [[ -n "${capture_dir}" && "${gnb_log_copied}" == false ]]; then
    say "Cleanup: copying gNB log snapshot to capture folder."
    copy_gnb_log_to_capture_dir "${capture_dir}" || true
  fi

  exit "${exit_code}"
}

trap cleanup EXIT INT TERM

parse_args "$@"
mkdir -p "${LOG_DIR}" "${LOCAL_CAPTURE_BASE}"

[[ -x "${GNB_BIN}" ]] || die "Missing executable gNB binary: ${GNB_BIN}"
ssh -o BatchMode=yes -o ConnectTimeout=5 "${PI_USER}@${PI_HOST}" 'true' >/dev/null 2>&1 ||
  die "Cannot SSH to ${PI_USER}@${PI_HOST}."

resolve_gnb_configs
resolve_capture_mode

section "WORKFLOW START"
cat <<EOF
srsRAN gNB + INA219 capture workflow

Actual YAML config files used by the gNB command:
$(printf '  - %s\n' "${gnb_config_files[@]}")

gNB log (parsed after the capture):
  ${GNB_RUN_LOG}

gNB terminal output:
  ${GNB_STDOUT_LOG}

INA219 recorder log:
  ${POWER_LOG}
EOF

section "POWER RECORDING AND ESTIMATION OPTIONS"
if [[ -z "${duration}" && "${non_interactive}" != true ]]; then
  read -r -p $'\nPower recording duration in seconds [30]: ' duration
fi
duration="${duration:-30}"
if ! [[ "${duration}" =~ ^[0-9]+$ ]] || (( duration <= 0 )); then
  die "Invalid duration: ${duration}"
fi
# Default the clock-alignment calibration window to the full capture duration
# unless the operator set one via --calibration-window-s.
srsran_calibration_window_seconds="${srsran_calibration_window_seconds:-${duration}}"

# The TR 38.840 estimate and the clock-alignment offset always run after the
# capture. Ask for the alignment calibration window used to estimate the
# per-capture residual offset.
if [[ "${non_interactive}" != true ]]; then
  read -r -p "srsRAN clock-alignment calibration window in seconds [${srsran_calibration_window_seconds}]: " input_calibration_seconds
  srsran_calibration_window_seconds="${input_calibration_seconds:-${srsran_calibration_window_seconds}}"
fi
if ! [[ "${srsran_calibration_window_seconds}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
    || ! awk -v cal="${srsran_calibration_window_seconds}" 'BEGIN { exit !(cal > 0) }'; then
  die "srsRAN calibration window must be a positive number."
fi

# Baseline microsleep power floor used by the 3GPP estimator (mW). Defaults to
# DEFAULT_3GPP_MICROSLEEP_POWER_MW; --microsleep-power-mw or this prompt overrides.
if [[ "${non_interactive}" != true ]]; then
  read -r -p "3GPP baseline microsleep power in mW [${microsleep_power_mw}]: " input_microsleep
  microsleep_power_mw="${input_microsleep:-${microsleep_power_mw}}"
fi
if ! [[ "${microsleep_power_mw}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
    || ! awk -v p="${microsleep_power_mw}" 'BEGIN { exit !(p >= 0) }'; then
  die "Microsleep power must be a non-negative number."
fi

if [[ "${capture_mode}" == synthetic ]]; then
  default_capture_name="synth_${synthetic_profile}_${duration}s_s${synthetic_seed}_${RUN_ID}"
else
  default_capture_name="${chrome_activity}_${RUN_ID}_${duration}s"
fi
if [[ -z "${capture_name}" && "${non_interactive}" != true ]]; then
  read -r -p "Capture folder name for CSV transfer [${default_capture_name}]: " capture_name
fi
capture_name="${capture_name:-${default_capture_name}}"
if ! [[ "${capture_name}" =~ ^[A-Za-z0-9._-]+$ ]] || [[ "${capture_name}" == "." || "${capture_name}" == ".." ]]; then
  die "Invalid capture folder name: ${capture_name}"
fi
# Route captures into a per-mode subfolder (synthetic/ or chrome/) so the two
# traffic types stay separated under LOCAL_CAPTURE_BASE. mkdir -p below creates
# the subfolder as needed.
transfer_capture_dir="${LOCAL_CAPTURE_BASE}/${capture_mode}/${capture_name}"
if [[ -n "${capture_dir_override}" ]]; then
  capture_dir_override="$(realpath -m "${capture_dir_override}")"
else
  capture_dir_override="${transfer_capture_dir}"
fi
[[ ! -e "${transfer_capture_dir}" ]] ||
  die "INA219 transfer staging directory already exists: ${transfer_capture_dir}"
if [[ "${capture_dir_override}" != "${transfer_capture_dir}" ]]; then
  [[ ! -e "${capture_dir_override}" ]] ||
    die "Final capture directory already exists: ${capture_dir_override}"
fi

say "srsRAN: parsing + 3GPP power estimation + clock-alignment offset (always)."
say "XGBoost training does not run here; use 'python -m scripts.experiments run' for that."
say "srsRAN alignment calibration window: ${srsran_calibration_window_seconds}s."

clear_gnb_run_log
runtime_started=true
handle_existing_or_start_gnb
turn_modem_airplane_mode_on
prepare_modem_data_path_with_retry

if [[ "${capture_mode}" == synthetic ]]; then
  section "SYNTHETIC TRAFFIC"
  start_synthetic_traffic
else
  check_chrome_or_wait

  if [[ "${chrome_activity}" == youtube ]]; then
    section "YOUTUBE PLAYBACK"
    open_youtube_video_in_chrome "${youtube_url}"
    youtube_start_epoch="$(date +%s)"
    say "YouTube video open timestamp: $(date --iso-8601=seconds)"
    say "Waiting 8 seconds before starting INA219 power recording."
    sleep 8
  else
    ensure_chrome_login
    section "CHROME ACTIVITY: ${chrome_activity}"
    start_chrome_activity
  fi
fi

section "POWER RECORDING"
say "Starting INA219 power recording on ${PI_USER}@${PI_HOST} for ${duration} seconds."
(
  ssh -tt -o BatchMode=yes "${PI_USER}@${PI_HOST}" \
    "cd '${PI_INA_DIR}' && INA219_LOGGER_DURATION='${duration}' exec python3 signal_safe_ina219_logger.py"
) > >(tee -a "${POWER_LOG}") 2>&1 &
power_ssh_pid=$!
say "Power recorder SSH PID: ${power_ssh_pid}"

say "Waiting for power recording to finish (or for the gNB to exit early)."
power_start_epoch="$(date +%s)"
power_end_epoch=$(( power_start_epoch + duration ))
while true; do
  now_epoch="$(date +%s)"
  remaining=$(( power_end_epoch - now_epoch ))
  elapsed=$(( now_epoch - power_start_epoch ))
  if (( remaining > 0 )); then
    printf '\r[%s] Power recording countdown: %s remaining (%s elapsed / %s total)' \
      "$(date --iso-8601=seconds)" \
      "$(format_seconds "${remaining}")" \
      "$(format_seconds "${elapsed}")" \
      "$(format_seconds "${duration}")"
  else
    printf '\r[%s] Power recording duration reached; waiting for recorder to finalize CSV...' \
      "$(date --iso-8601=seconds)"
  fi

  if ! kill -0 "${power_ssh_pid}" 2>/dev/null; then
    printf '\n'
    wait "${power_ssh_pid}" || true
    power_ssh_pid=""
    break
  fi

  if [[ -n "${gnb_pid_to_monitor}" ]] && ! kill -0 "${gnb_pid_to_monitor}" 2>/dev/null; then
    printf '\n'
    gnb_failed_early=true
    say "gNB monitor target is no longer running before the requested duration ended."
    if [[ -n "${youtube_start_epoch}" ]]; then
      if [[ "${chrome_activity}" == youtube ]]; then
        say "gNB alive time since YouTube video start: $(duration_since_epoch "${youtube_start_epoch}")"
      else
        say "gNB alive time since ${chrome_activity} activity start: $(duration_since_epoch "${youtube_start_epoch}")"
      fi
    fi
    if [[ -n "${gnb_start_epoch}" ]]; then
      say "gNB alive time since gNB start: $(duration_since_epoch "${gnb_start_epoch}")"
    else
      say "gNB alive time since gNB start: unavailable because the gNB was already running when this workflow began."
    fi
    say "Stopping RPi INA219 recorder because gNB ended before duration completed."
    stop_remote_power_recorder_if_running
    wait "${power_ssh_pid}" || true
    power_ssh_pid=""
    break
  fi

  sleep 1
done

say "Power recording has stopped."

if [[ -n "${gnb_pid}" ]] &&
   ! kill -0 "${gnb_pid}" 2>/dev/null &&
   (( $(date +%s) - power_start_epoch < duration )); then
  gnb_failed_early=true
fi

if [[ "${gnb_failed_early}" == true ]]; then
  die "gNB ended before the requested capture duration; discarding this run as failed."
fi

section "TRANSFER POWER CSV"
csv_file="$(awk -F'Data saved to ' '/Data saved to / {print $2}' "${POWER_LOG}" | tail -n 1 | tr -d '\r')"
if [[ -z "${csv_file}" ]]; then
  say "Could not parse CSV path from power log; locating newest RPi INA219 CSV."
  csv_file="$(
    ssh -o BatchMode=yes "${PI_USER}@${PI_HOST}" \
      "find '${PI_INA_DIR}' -maxdepth 2 -type f -name 'ina219_power_log*.csv' -printf '%T@ %p\\n' | sort -nr | head -n 1 | cut -d' ' -f2-"
  )"
fi
[[ -n "${csv_file}" ]] || die "Could not locate an INA219 CSV on ${PI_USER}@${PI_HOST}."
ssh -o BatchMode=yes "${PI_USER}@${PI_HOST}" "test -s '${csv_file}'" ||
  die "RPi INA219 CSV is missing or empty: ${csv_file}"

say "Transferring recorded power CSV directly from the RPi:"
printf '  %s\n' "${csv_file}"
mkdir -p "${transfer_capture_dir}"
scp -p -- "${PI_USER}@${PI_HOST}:${csv_file}" "${transfer_capture_dir}/"
transferred_csv="${transfer_capture_dir}/$(basename "${csv_file}")"
[[ -s "${transferred_csv}" ]] || die "INA219 CSV transfer failed: ${transferred_csv}"
say "INA219 CSV transfer complete."
if [[ "${capture_dir_override}" != "${transfer_capture_dir}" ]]; then
  [[ -d "${transfer_capture_dir}" ]] ||
    die "INA219 transfer did not create staging directory: ${transfer_capture_dir}"
  mkdir -p "$(dirname "${capture_dir_override}")"
  mv -- "${transfer_capture_dir}" "${capture_dir_override}"
fi
capture_dir="${capture_dir_override}"
printf 'Local destination should be under:\n  %s/\n' "${capture_dir}"

finalize_gnb_capture

section "SRSRAN LOG PROCESSING"
(
  cd "${UE_POWER_ESTIMATION_REPO}"
  "${PYTHON_BIN}" -m scripts.power_estimation.srsRAN.extract_features \
    --log "${capture_dir}/srsRAN/gnb_n41.log" \
    --verbose
)
for expected_srsran_parse in \
  "${capture_dir}/srsRAN/parsed/parsed_rows.jsonl" \
  "${capture_dir}/srsRAN/parsed/slot_features.csv" \
  "${capture_dir}/srsRAN/parsed/slot_feature_summary.json"; do
  [[ -s "${expected_srsran_parse}" ]] || die "Missing or empty srsRAN parsed output: ${expected_srsran_parse}"
  say "srsRAN parsed output ready: ${expected_srsran_parse}"
done

section "WRITE CAPTURE METADATA"
# Capture-level metadata.json required by the experiment runner
# (scripts.experiments). CaptureLayout already resolves this capture's data via
# its legacy file locations, so only this descriptor needs to be added for
# `python -m scripts.experiments run` to consume the capture.
CHROME_ACTIVITY="${chrome_activity}" CHROME_ACTIVITY_SEED="${chrome_activity_seed}" \
"${PYTHON_BIN}" - \
  "${capture_dir}" "${capture_name}" "${capture_mode}" "${youtube_url}" \
  "${synthetic_profile}" "${synthetic_seed}" "${synthetic_schedule_hash}" \
  "${duration}" "${gnb_config_files[@]}" <<'PY'
import datetime, json, os, sys

(capture_dir, capture_id, capture_mode, youtube_url, profile, seed,
 sched_hash, duration) = sys.argv[1:9]
gnb_config_files = sys.argv[9:]

if capture_mode == "synthetic":
    workload = {
        "type": "synthetic_traffic",
        "profile": profile or None,
        "seed": int(seed) if seed else None,
        "schedule_hash": sched_hash or None,
    }
else:
    chrome_activity = os.environ.get("CHROME_ACTIVITY", "youtube")
    if chrome_activity and chrome_activity != "youtube":
        activity_seed = os.environ.get("CHROME_ACTIVITY_SEED", "")
        workload = {
            "type": "chrome_activity",
            "activity": chrome_activity,
            "seed": int(activity_seed) if activity_seed else None,
        }
    else:
        workload = {"type": "youtube", "youtube_url": youtube_url or None}

metadata = {
    "schema_version": 1,
    "capture_id": capture_id,
    "device": "qualcomm",
    "purpose": "measurement",
    "complete": True,
    "workload": workload,
    # Link config is not auto-derived from the gNB overlays; fill these in if a
    # config uses validation.require_link_match against this capture.
    "link_config": {
        "srs": {"enabled": None, "periodicity_ms": None},
        "csi_rs": {"enabled": None, "periodicity_ms": None},
    },
    "available_modalities": {
        "srsran_log": True,
        "ground_truth": True,
    },
    "collection": {
        "workflow": "capture.sh",
        "duration_s": float(duration),
        "gnb_config_files": gnb_config_files,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
    },
}
path = os.path.join(capture_dir, "metadata.json")
with open(path, "w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(path)
PY
[[ -s "${capture_dir}/metadata.json" ]] || die "Failed to write capture metadata.json"
say "Wrote capture metadata: ${capture_dir}/metadata.json"

if [[ "${capture_mode}" == synthetic && -n "${synthetic_schedule_file}" && -f "${synthetic_schedule_file}" ]]; then
  mkdir -p "${capture_dir}/synthetic_traffic"
  cp -f "${synthetic_schedule_file}" "${capture_dir}/synthetic_traffic/schedule.json" 2>/dev/null || true
  cp -f "${LOG_DIR}/synthetic_results.json" "${capture_dir}/synthetic_traffic/results.json" 2>/dev/null || true
  say "Copied synthetic schedule/results into ${capture_dir}/synthetic_traffic/."
fi

if [[ "${capture_mode}" == chrome && "${chrome_activity}" != youtube ]]; then
  activity_report="${LOG_DIR}/chrome_activity_${chrome_activity}.json"
  if [[ -f "${activity_report}" ]]; then
    mkdir -p "${capture_dir}/chrome_activity"
    cp -f "${activity_report}" "${capture_dir}/chrome_activity/report.json" 2>/dev/null || true
    say "Copied Chrome activity report into ${capture_dir}/chrome_activity/."
  fi
fi

section "SRSRAN 3GPP POWER ESTIMATION"
(
  cd "${UE_POWER_ESTIMATION_REPO}"
  "${PYTHON_BIN}" -m scripts.power_estimation.srsRAN.run_3gpp \
    --capture-dir "${capture_dir}" \
    --microsleep-power-mw "${microsleep_power_mw}"
)
for expected_srsran_3gpp_result in \
  "${capture_dir}/srsRAN/rule_based_pwr_est/3GPP/power_per_slot.csv" \
  "${capture_dir}/srsRAN/rule_based_pwr_est/3GPP/power_summary.json"; do
  [[ -s "${expected_srsran_3gpp_result}" ]] || die "Missing or empty srsRAN 3GPP result: ${expected_srsran_3gpp_result}"
  say "srsRAN 3GPP result ready: ${expected_srsran_3gpp_result}"
done

section "SRSRAN CLOCK ALIGNMENT"
# Estimate the per-capture residual clock offset (gNB log vs INA219) once and
# cache it as capture metadata. Also writes the 3GPP gt-vs-estimate figure (the
# offset is required to align the two clocks). Power estimation (run separately,
# e.g. via 'python -m scripts.experiments run') reads and applies this offset.
(
  cd "${UE_POWER_ESTIMATION_REPO}"
  "${PYTHON_BIN}" -m scripts.power_estimation.srsRAN.alignment \
    --capture-dir "${capture_dir}" \
    --gpp-csv "${capture_dir}/srsRAN/rule_based_pwr_est/3GPP/power_per_slot.csv" \
    --calibration-window-s "${srsran_calibration_window_seconds}" \
    --force
)
[[ -s "${capture_dir}/srsRAN/alignment/alignment.json" ]] ||
  die "Missing srsRAN alignment metadata"
say "srsRAN alignment metadata ready: ${capture_dir}/srsRAN/alignment/alignment.json"
[[ -s "${capture_dir}/srsRAN/rule_based_pwr_est/3GPP/gt_vs_estimate.pdf" ]] ||
  die "Missing srsRAN 3GPP gt-vs-estimate figure"
say "srsRAN 3GPP gt-vs-estimate figure ready: ${capture_dir}/srsRAN/rule_based_pwr_est/3GPP/gt_vs_estimate.pdf"

# Dataset figures: synthetic schedule DL/UL plot (when a schedule exists) and the
# grouped ground-truth power folder (raw INA219 log + 0.5 ms interpolated CSV +
# figure). Runs last so power estimation above sees the raw CSV at the capture
# root before it is moved into ground_truth_power/. Non-fatal: a figure failure
# does not invalidate an otherwise-complete capture.
section "DATASET FIGURES"
dataset_figures_script="${UE_POWER_ESTIMATION_REPO}/scripts/data_collection/generate_dataset_figures.py"
if [[ -f "${dataset_figures_script}" ]]; then
  say "Generating schedule + ground-truth power figures for ${capture_dir}."
  if (cd "${UE_POWER_ESTIMATION_REPO}" && "${PYTHON_BIN}" "${dataset_figures_script}" "${capture_dir}"); then
    say "Dataset figures ready under ${capture_dir} (synthetic_traffic/schedule.pdf, ground_truth_power/)."
  else
    say "WARNING: dataset figure generation failed; capture is still complete."
  fi
else
  say "WARNING: dataset figure generator not found: ${dataset_figures_script}"
fi

printf '\nLogs from this workflow are in:\n  %s\n' "${LOG_DIR}"

if [[ "${stop_chrome_on_exit}" == true ]]; then
  sudo "${CHROME_AGENT}" stop >/dev/null 2>&1 || sudo systemctl stop "${CHROME_SERVICE}" 2>/dev/null || true
fi
trap - EXIT INT TERM
exit 0

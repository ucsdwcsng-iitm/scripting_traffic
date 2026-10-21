# Scripting Traffic, Aligning Power: A 5G Testbed Integrating gNB Activity and Commercial UE Power

Code for the WiNTECH '26 paper ([DOI 10.1145/3831662.3844174](https://doi.org/10.1145/3831662.3844174),
[website](https://wcsng.ucsd.edu/scripting-power/)): drive real applications or scripted
traffic through a private srsRAN 5G cell and record per-slot gNB logs together with the
hardware-measured power of a commercial 5G modem, aligned on a common slot grid. As an
extra demonstration of what the aligned data enables, it also trains gNB-Power, an
XGBoost estimator of UE power from gNB-observable features, and compares it with the
3GPP TR 38.840 model.

```
testbed/                capture side (the measurement platform)
  capture.sh            one capture: gNB + workload + INA219 power + post-processing
  ue/                   UE data path: Chrome (real apps) / synthetic-traffic network namespaces
  dn/                   data-network endpoint for synthetic traffic
  pi/                   INA219 power logger (runs on the Raspberry Pi)
synthetic_traffic/      scripted DL/UL rate schedules and the UDP player/server
ue_power_estimation/    gNB log parsing, TR 38.840 model, clock alignment; gNB-Power (XGBoost)
  scripts/
    power_estimation/
      srsRAN/           gNB log parser and per-slot features, TR 38.840 estimate, clock
                        alignment (Eq. 1), XGBoost training/evaluation, pooled training
      tr38840/          TR 38.840 relative power states and scaling rules
    data_collection/    ground-truth power grid + figures for a capture (run by capture.sh)
    experiments/        config-driven runner for the estimation demo (python -m scripts.experiments)
  experiment_configs/
    in_capture/         per-capture accuracy (Table 3, Fig. 5)
    drift/              stability over a 250 s capture (Fig. 6)
    synthetic_to_real/  synthetic-trained model on real apps (Table 4)
make_deliverables/      scripts that regenerate the paper's tables and figures
```

## 1. Testbed

**Hardware** (paper Table 1, Figs. 2–3)
- Linux host with a USRP B210 (USB) running the srsRAN gNB and Open5GS core.
- UE: Telit FN990A28 (Snapdragon X65) on a Waveshare PCIe-to-M.2 4G/5G HAT on a
  Raspberry Pi 5, four RF ports on a 4x1 PCB antenna, over the air. Its QMI data
  interface (`/dev/cdc-wdm0`, `wwan0`, `qmi_wwan` driver) is visible on the host.
- INA219 high-side current sensor on the modem supply, read by the Pi over I2C.

**Host software**
- [srsRAN Project](https://www.srsran.com/) built from source (the paper's gNB logs report
  commit `4bf1543936`), expected at `$SRSRAN_DIR/build/apps/gnb/gnb` and runnable by the
  invoking user (UHD/USB access), and a gNB config for the
  cell in paper Table 1 whose log is at debug level (the parser reads it) and is written to
  `logs/gnb_n41.log` relative to the run directory `testbed/runs/` (or set `RUN_DIR` /
  `GNB_RUN_LOG`).
- [Open5GS](https://open5gs.org/) with its default configuration (UPF gateway `10.45.0.1`
  on `ogstun`), the UE's SIM provisioned as a subscriber, APN `internet`, and NAT from
  `ogstun` to the Internet (the data-path checks ping 8.8.8.8; real apps need it anyway).
- `libqmi` (`qmicli`), passwordless `sudo`, `systemd`.
- Real-app captures: Google Chrome at `/usr/bin/google-chrome` and a desktop session with
  Xwayland on display `:0` (e.g. GNOME), in which Chrome is shown.
- Python 3.10+ with `pip install -r requirements.txt`.
- Key-based SSH to the Pi (`PI_USER@PI_HOST`, default `rabbit@qcom-pi.local`).

**Raspberry Pi**: copy `testbed/pi/signal_safe_ina219_logger.py` to `PI_INA_DIR`
(default `/home/$PI_USER/Downloads/INA219`) together with the `INA219.py` I2C driver it
imports (the Waveshare INA219 demo driver API: `INA219(i2c_bus=, addr=)` with register
`read`/`write`). It samples the modem-supply power every 0.5 ms (`INA219_I2C_BUS`,
`INA219_ADDRESS`, `INA219_SAMPLING_INTERVAL_MS` override bus 1, 0x40, 0.5 ms) into
`$PI_INA_DIR/csv/ina219_power_log_<timestamp>.csv` for `INA219_LOGGER_DURATION` seconds,
records its PID in `/tmp/signal_safe_ina219_logger.pid`, and stops cleanly on SIGINT/SIGTERM.

## 2. Capture (the measurement)

`testbed/capture.sh` records one capture. It resolves all paths from its own location, so
it can be run from any directory (it calls `sudo` for the network and modem steps).
Captures are written under `ue_power_estimation/data/qualcomm/<mode>/<capture-name>/`
(`<mode>` = `chrome` or `synthetic`). One run:

1. Checks the gNB binary and the SSH link to the Pi, and resolves the gNB config(s)
   (`--gnb-config`, repeatable) and the workload.
2. Starts the gNB in `testbed/runs/`; its debug log goes to `testbed/runs/logs/gnb_n41.log`,
   and this run's own logs to `testbed/runs/logs/workflow_<timestamp>/`.
3. Toggles the modem's airplane mode, brings up the QMI data session and moves `wwan0` into
   a network namespace: `ue-browser` for real apps, `ue-traffic` for synthetic traffic
   (`testbed/ue/*-agent.sh`; retried, restarting Open5GS if the session does not come up).
4. Starts the workload:
   - `--mode chrome`: Chrome (`ue-chrome.service`) in `ue-browser` opens the YouTube URL or
     is driven by `testbed/ue/ue-chrome-activity.py` (Instagram, X, slither.io, Zoom).
     Instagram/X need a logged-in Chrome profile: run one such capture without
     `--non-interactive` first and log in when prompted (kept in `CHROME_PROFILE_DIR`).
   - `--mode synthetic`: `synthetic_traffic/schedule_gen.py` builds a DL/UL rate schedule
     from (profile, seed, caps, duration) on a 10 ms grid, the DN endpoint
     (`testbed/dn/udp-dn-server.sh`) is started if needed, and
     `synthetic_traffic/udp_traffic_player.py` plays the schedule from `ue-traffic`
     (a plotted copy of the schedule also goes to `synthetic_traffic/schedules/`).
5. Runs the INA219 logger on the Pi for `--duration` seconds (the run is discarded if the
   gNB exits early), copies the power CSV into the capture folder, then stops the gNB and
   copies its log there.
6. Post-processes the capture (writes `<capture>/...`):
   - `srsRAN/gnb_n41.log`: the raw gNB log;
     `srsRAN/parsed/slot_features.csv`: one feature row per 0.5 ms slot for the UE
     (the most active C-RNTI).
   - `srsRAN/rule_based_pwr_est/3GPP/power_per_slot.csv`: per-slot TR 38.840 power
     (`--microsleep-power-mw`, default 400).
   - `srsRAN/alignment/alignment.json`: the clock alignment of Eq. (1): the residual
     gNB-log/meter offset (searched over +-250 ms in 1 ms steps) and the bounded affine fit
     (a, b) of the TR 38.840 trace that minimizes the calibration error over the first
     `--calibration-window-s` seconds (default: the whole capture); the fit is plotted in
     `rule_based_pwr_est/3GPP/gt_vs_estimate.pdf`.
   - `ground_truth_power/`: the raw INA219 CSV, and the same power interpolated onto a
     0.5 ms grid (`ground_truth_power_0p5ms.csv` and a plot).
   - `metadata.json` (workload, gNB configs, duration); for synthetic captures
     `synthetic_traffic/schedule.{json,pdf}` and `results.json` (offered vs achieved rates);
     for Instagram/X/slither.io/Zoom `chrome_activity/report.json`.

The paper's captures, reproduced with the commands below. The configs in Section 3 refer to
these capture names, so pass the same `--capture-name` (or edit `capture_id` in the configs).

```bash
S="--non-interactive --force-gnb-restart --gnb-config <your gNB config>"

# Real applications, 60 s (Tables 3-4, Fig. 5): the first 50 s calibrate the alignment;
# TR 38.840 micro-sleep power 420 mW, as in experiment_configs/in_capture
testbed/capture.sh $S --mode chrome --duration 60 --calibration-window-s 50 --microsleep-power-mw 420 \
  --chrome-activity youtube --youtube-url https://www.youtube.com/watch?v=EF8C4v7JIbA \
  --capture-name youtube_20260624_165416_60s
#   likewise --chrome-activity instagram | slither | x | zoom (--zoom-meeting-id ID) with
#   --capture-name instagram_20260625_140847_60s | slither_20260625_142022_60s |
#                  x_20260625_134225_60s | zoom_20260628_140428_60s

# Real applications, 250 s (Fig. 6): same, with --duration 250, the default calibration
# window, and the <app>_..._250s names from experiment_configs/drift/<app>.yaml

# Synthetic training corpus (Table 4): 18 captures
for spec in youtube:3:8000:1500 youtube:4:4000:1000 youtube:5:16000:1500 \
            youtube:6:25000:2000 youtube:7:12000:1500 youtube:8:6000:1200 \
            social_scroll:2:8000:2000 social_scroll:3:16000:3000 social_scroll:4:12000:2000 \
            web_browse:2:8000:3000 web_browse:3:16000:4000 web_browse:4:25000:4000 \
            video_call:2:4000:4000 video_call:3:2000:2000 video_call:4:6000:6000 \
            sweep:2:40000:10000 sweep:3:25000:8000 sweep:4:16000:4000; do
  IFS=: read p s dl ul <<< "$spec"; d=$([ $p = sweep ] && echo 120 || echo 90)
  testbed/capture.sh $S --mode synthetic --synthetic-profile $p --synthetic-seed $s \
    --synthetic-dl-cap-kbps $dl --synthetic-ul-cap-kbps $ul --duration $d \
    --capture-name synthb_${p}_s${s}_dl${dl}_ul${ul}
done

# Traffic probes (Fig. 4): profiles dl_pulse, ul_pulse, ul_triangle, mix (60 s) and
# dual_sine (120 s), seed 1, default caps; names as in make_deliverables/fig4_traffic_probes.py
testbed/capture.sh $S --mode synthetic --synthetic-profile dl_pulse --synthetic-seed 1 \
  --duration 60 --capture-name synth_dl_pulse_s1_20260623_182448
```

Run `testbed/capture.sh --help` for all options and host-specific environment overrides.

## 3. Demonstration: gNB-side UE power estimation (offline)

Run from `ue_power_estimation/`; building the pooled synthetic model needs ~40 GB of RAM.

```bash
cd ue_power_estimation
for group in in_capture drift synthetic_to_real; do
  for app in youtube instagram slither x zoom; do
    python -m scripts.experiments run --config experiment_configs/$group/$app.yaml
  done
done
```

Each run of `scripts.experiments` (paths relative to `ue_power_estimation/`):

1. Resolves the config (per-app file + `_base.yaml`) and checks that its captures exist.
2. Re-parses the real capture's gNB log and recomputes TR 38.840 at the config's
   micro-sleep power into a cache, `data/qualcomm/<mode>/<capture>/srsRAN/feature_sets/<hash>/`
   (`feature_sets/frozen_eval/` for a separate test capture), and uses the capture's
   `alignment.json` (computed over the first 50 s if missing) to place power samples on the
   slot grid.
3. Builds the causal per-slot features (lags and rolling statistics) and trains XGBoost:
   - `in_capture/`: on 0–40 s (40–50 s validation) of one 60 s real capture, predicting
     absolute power from gNB features only;
   - `drift/`: the same on a 250 s capture;
   - `synthetic_to_real/`: on all slots of the 18 pooled synthetic captures (random 12%
     validation rows), read from the capture-time outputs of Section 2 (so the synthetic
     captures must be recorded at the default 400 mW). One global affine TR 38.840 fit is
     made on the pool; the model takes the raw and affine TR 38.840 traces as extra inputs
     and learns the residual over the affine trace (Eq. 3). Per-capture feature matrices
     are cached in `models/_pool_feature_cache/`.

   The model is written to `models/<experiment>/default/<train capture>/` (for
   `synthetic_to_real/` the train capture is the placeholder `synthb_youtube_s3_...`, and
   each per-app config trains the same pooled model again).
4. Evaluates the frozen model on the test window and writes
   `results/<experiment>/default/<train capture>[_to_<test capture>]/`, with
   `srsran_eval/xgboost/**/xgboost_holdout_predictions_<s>.csv` (per-slot measured, gNB-Power
   and TR 38.840 raw/affine power) and `srsran_eval/xgboost/xgboost_summary.json` (MAE,
   MAPE, RMSE) for the last 10 s of the 60 s capture; `drift/` writes one
   `srsran_eval_<a>s_to_<b>s/` folder per 10 s chunk from 50 to 250 s. A rerun reuses a
   trained model with the same training settings unless `--force` is given.

## 4. Paper deliverables

Run from the repository root; figures are written to `make_deliverables/figures/`.

| Deliverable | Command | Reads |
|---|---|---|
| Fig. 4: traffic probes and UE power | `python make_deliverables/fig4_traffic_probes.py` | probe captures: `synthetic_traffic/schedule.json`, `ground_truth_power/ground_truth_power_0p5ms.csv` |
| Tables 3–4 | `python make_deliverables/tables.py` | `results/in_capture_*`, `results/synthetic_to_*`: holdout predictions + raw INA219 CSV |
| Fig. 5: in-capture traces | `python make_deliverables/fig5_incapture_traces.py` | `results/in_capture_*`: holdout predictions |
| Fig. 6: drift over 250 s | `python make_deliverables/fig6_drift.py` | `results/drift_*`: per-chunk `xgboost_summary.json` |
| Fig. 7: synthetic schedules | `python make_deliverables/fig7_synthetic_schedules.py` | nothing (regenerates seed-1, 250 s schedules) |
| Inference latency (Sec. 5.1) | `python make_deliverables/inference_latency.py` | `results/in_capture_*` manifests → models, cached features, captures; host-dependent |

On the paper's captures, rerunning Section 3 reproduces the MAE/MAPE values of Tables 3–4
and Figs. 5–6 exactly; interval energy errors agree within 0.2 percentage points.

Fig. 1 (Pixel 9 energy breakdown) comes from a separate phone measurement, and Figs. 2–3
are diagrams/photos; Tables 1–2 summarize the testbed configuration and the features in
`ue_power_estimation/scripts/power_estimation/srsRAN/run_xgboost.py`.

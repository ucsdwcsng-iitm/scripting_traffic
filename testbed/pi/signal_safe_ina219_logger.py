#!/usr/bin/env python3
"""Signal-safe INA219 CSV recorder for Raspberry Pi."""

from __future__ import annotations

import csv
import os
import signal
import time
from datetime import datetime
from pathlib import Path

from INA219 import INA219


I2C_BUS = int(os.environ.get("INA219_I2C_BUS", "1"))
ADDRESS = int(os.environ.get("INA219_ADDRESS", "0x40"), 0)
DURATION = float(os.environ.get("INA219_LOGGER_DURATION", "3600"))
SAMPLING_INTERVAL_MS = float(os.environ.get("INA219_SAMPLING_INTERVAL_MS", "0.5"))
FLUSH_EVERY_ROWS = int(os.environ.get("INA219_FLUSH_EVERY_ROWS", "100"))
PID_FILE = Path(os.environ.get("INA219_PID_FILE", "/tmp/signal_safe_ina219_logger.pid"))
HALT_REASON_FILE = Path(
    os.environ.get("INA219_HALT_REASON_FILE", "/tmp/signal_safe_ina219_logger_halt_reason")
)

stop_requested = False
stop_signal: int | None = None


def request_stop(signum: int, _frame: object) -> None:
    global stop_requested, stop_signal
    stop_requested = True
    stop_signal = signum


def configure_fast_ina219() -> INA219:
    ina219 = INA219(i2c_bus=I2C_BUS, addr=ADDRESS)

    ina219.bus_adc_resolution = 0x00
    ina219.shunt_adc_resolution = 0x00
    ina219.mode = 0x07

    config = (
        ina219.bus_voltage_range << 13
        | ina219.gain << 11
        | ina219.bus_adc_resolution << 7
        | ina219.shunt_adc_resolution << 3
        | ina219.mode
    )
    ina219.write(0x00, config)
    ina219.write(0x05, ina219._cal_value)
    return ina219


def read_halt_reason() -> tuple[str | None, str | None]:
    try:
        lines = HALT_REASON_FILE.read_text().splitlines()
    except FileNotFoundError:
        return None, None

    halt_time = lines[0] if lines else None
    halt_reason = lines[1] if len(lines) > 1 else None
    return halt_time, halt_reason


def main() -> int:
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    PID_FILE.write_text(f"{os.getpid()}\n")

    script_dir = Path(__file__).resolve().parent
    csv_dir = script_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = csv_dir / f"ina219_power_log_{timestamp}.csv"

    ina219 = configure_fast_ina219()

    def fast_get_power_w() -> float:
        value = ina219.read(0x03)
        if value > 32767:
            value -= 65535
        return value * ina219._power_lsb

    print("INA219 configured for maximum speed: 9-bit resolution (~0.2ms per reading)", flush=True)
    print(
        f"Starting data collection at {SAMPLING_INTERVAL_MS}ms intervals "
        f"({1000 / SAMPLING_INTERVAL_MS:.0f} Hz target)",
        flush=True,
    )
    print(f"CSV path: {csv_filename}", flush=True)

    rows = 0
    cumulative_energy = 0.0
    start_time = time.time()
    prev_time = start_time
    first_sample = True
    sampling_interval_sec = SAMPLING_INTERVAL_MS / 1000.0
    next_sample_time = start_time

    try:
        with csv_filename.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Timestamp_ISO",
                    "Timestamp_Epoch(s)",
                    "Power (W)",
                    "Power (mW)",
                    "Interval (ms)",
                    "Energy (Joules)",
                ]
            )
            csvfile.flush()

            while not stop_requested and time.time() - start_time < DURATION:
                now = time.time()
                if now < next_sample_time:
                    time.sleep(min(next_sample_time - now, 0.001))
                    continue

                t_iso = datetime.now().strftime("%H:%M:%S.%f")
                power = fast_get_power_w()
                power_mw = power * 1000.0

                if first_sample:
                    interval_ms = 0.0
                    first_sample = False
                else:
                    interval_ms = (now - prev_time) * 1000.0
                    cumulative_energy += power * (interval_ms / 1000.0)

                prev_time = now
                next_sample_time += sampling_interval_sec
                rows += 1

                writer.writerow(
                    [
                        t_iso,
                        f"{now:.6f}",
                        f"{power:.6f}",
                        f"{power_mw:.3f}",
                        f"{interval_ms:.3f}",
                        f"{cumulative_energy:.9f}",
                    ]
                )
                if rows % FLUSH_EVERY_ROWS == 0:
                    csvfile.flush()

            csvfile.flush()
            os.fsync(csvfile.fileno())

        elapsed = max(time.time() - start_time, 1e-9)
        if stop_signal is not None:
            halt_time, halt_reason = read_halt_reason()
            rpi_stop_time = datetime.now().astimezone().isoformat()
            print("=" * 78, flush=True)
            print(f"EARLY HALT: INA219 recorder stopped at {rpi_stop_time}", flush=True)
            if halt_time or halt_reason:
                print("Stop source: gNB error observed by srsRAN watcher", flush=True)
            if halt_time:
                print(f"gNB halt timestamp: {halt_time}", flush=True)
            if halt_reason:
                print(f"gNB halt reason: {halt_reason}", flush=True)
            print(f"Stop signal received by RPi recorder: {signal.Signals(stop_signal).name}", flush=True)
            print("=" * 78, flush=True)
        print(f"Logging complete! Collected {rows} samples over {elapsed:.3f} seconds.", flush=True)
        print(f"Actual sampling rate: {rows / elapsed:.1f} Hz", flush=True)
        print(f"Data saved to {csv_filename}", flush=True)
        return 0
    finally:
        try:
            if PID_FILE.read_text().strip() == str(os.getpid()):
                PID_FILE.unlink()
        except FileNotFoundError:
            pass
        try:
            HALT_REASON_FILE.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

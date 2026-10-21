"""Shared paths inside one capture directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CaptureLayout:
    capture_dir: Path
    srsran_results_root: Optional[Path] = None

    @property
    def srsran_dir(self) -> Path:
        canonical = self.capture_dir / "raw" / "srsran"
        legacy = self.capture_dir / "srsRAN"
        return canonical if canonical.exists() or not legacy.exists() else legacy

    @property
    def srsran_log(self) -> Path:
        canonical = self.srsran_dir / "gnb.log"
        legacy = self.srsran_dir / "gnb_n41.log"
        return canonical if canonical.exists() or not legacy.exists() else legacy

    @property
    def srsran_parsed_dir(self) -> Path:
        return self.srsran_dir / "parsed"

    @property
    def srsran_sparse_rows(self) -> Path:
        return self.srsran_parsed_dir / "parsed_rows.jsonl"

    @property
    def srsran_alignment_dir(self) -> Path:
        # Per-capture clock-alignment metadata (residual offset vs. INA219 GT),
        # computed once at data-prep time and read by power estimation.
        return self.srsran_dir / "alignment"

    @property
    def srsran_alignment_json(self) -> Path:
        return self.srsran_alignment_dir / "alignment.json"

    @property
    def srsran_results_dir(self) -> Path:
        # Root for this capture's power-estimation outputs. The experiment runner
        # overrides this (srsran_results_root) to redirect into its results tree.
        if self.srsran_results_root is not None:
            return self.srsran_results_root
        return self.srsran_dir

    @property
    def srsran_rule_based_dir(self) -> Path:
        # Rule-based (3GPP) power-estimation outputs.
        return self.srsran_results_dir / "rule_based_pwr_est"

    @property
    def srsran_3gpp_dir(self) -> Path:
        return self.srsran_rule_based_dir / "3GPP"

    @property
    def srsran_xgboost_dir(self) -> Path:
        return self.srsran_results_dir / "xgboost"

    def ground_truth_csv(self) -> Optional[Path]:
        # Search the grouped ground_truth_power/ folder first, then the legacy
        # raw/ground_truth/ location, then the capture root.
        for base in (
            self.capture_dir / "ground_truth_power",
            self.capture_dir / "raw" / "ground_truth",
            self.capture_dir,
        ):
            matches = sorted(base.glob("ina219_power_log*.csv"))
            if matches:
                return matches[0]
        return None

    def require_ground_truth_csv(self) -> Path:
        path = self.ground_truth_csv()
        if path is None:
            raise FileNotFoundError(f"No INA219 CSV found under {self.capture_dir}")
        return path


def capture_root_from_source(path: Path) -> Path:
    """Return capture root from a capture, srsRAN, or parsed path."""

    path = Path(path).resolve()
    if "processed" in path.parts:
        index = path.parts.index("processed")
        return Path(*path.parts[:index])
    if path.name == "parsed" and path.parent.name == "srsRAN":
        return path.parent.parent
    if path.name in {"srsran", "ground_truth", "config"} and path.parent.name == "raw":
        return path.parent.parent
    if path.name == "srsRAN":
        return path.parent
    return path

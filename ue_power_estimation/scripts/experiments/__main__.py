"""CLI for validating and running offline experiment configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ConfigError, load_experiment_config
from .runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, required=True)
        if command == "run":
            child.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_experiment_config(args.config)
        if args.command == "validate":
            print(
                json.dumps(
                    {
                        "valid": True,
                        "experiment_id": config.experiment_id,
                        "variant_id": config.variant_id,
                        "config_hash": config.config_hash,
                        "result_dir": str(config.result_dir),
                        "model_dir": str(config.model_dir),
                    },
                    indent=2,
                )
            )
            return 0
        manifest = run_experiment(config, force=bool(args.force))
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except (ConfigError, FileNotFoundError, RuntimeError, NotImplementedError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())

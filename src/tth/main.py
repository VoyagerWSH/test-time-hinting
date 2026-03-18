from __future__ import annotations

import argparse
from pathlib import Path

from tth.config import load_config
from tth.runner import run_sync


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run hint optimization (Proposer / Checker / Experimenter).")
    p.add_argument("--config", "-c", default="configs/default.yaml", help="Path to YAML config.")
    p.add_argument("--input", "-i", default="", help="Input CSV (overrides config).")
    p.add_argument("--output", "-o", default="", help="Output directory (overrides config).")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint if present.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    cfg = load_config(config_path)
    if args.input:
        cfg.input_csv = args.input
    if args.output:
        cfg.output_dir = args.output

    input_path = Path(cfg.input_csv)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path = Path(cfg.output_dir)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    run_sync(cfg, input_path=input_path, output_path=output_path, resume=args.resume)
    print(f"Output written to {output_path / 'output.csv'}")


if __name__ == "__main__":
    main()

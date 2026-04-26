#!/usr/bin/env python3
"""
One-command capital gain pipeline:
1) prepare dataset from UK PPD raw files
2) run leakage comparison
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Capital gain one-command runner.")
    p.add_argument("--input-glob", type=str, default="examples/capital_gain/raw/pp-*.csv")
    p.add_argument(
        "--output-csv",
        type=str,
        default="examples/capital_gain/data/capital_gain_dataset.csv",
    )
    p.add_argument("--start-date", type=str, default="2010-01-01")
    p.add_argument("--end-date", type=str, default="2023-12-31")
    p.add_argument("--target", choices=["target_logret", "target_diff"], default="target_logret")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    base = Path(__file__).resolve().parent

    prep = base / "prepare_capital_gain_data.py"
    rep = base / "replicate_capital_gain.py"

    run(
        [
            sys.executable,
            str(prep),
            "--input-glob",
            args.input_glob,
            "--output",
            args.output_csv,
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
        ]
    )

    run(
        [
            sys.executable,
            str(rep),
            "--csv",
            args.output_csv,
            "--target",
            args.target,
            "--seed",
            str(args.seed),
        ]
    )


if __name__ == "__main__":
    main()

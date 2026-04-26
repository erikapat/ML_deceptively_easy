#!/usr/bin/env python3
"""
One-command Kaggle IV run:
1) Build prepared dataset from raw SPY chain files.
2) Run replication experiment.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full Kaggle IV pipeline in one command.")
    ap.add_argument("--input-glob", default="")
    ap.add_argument("--output-csv", default="")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--start-date", default="2010-01-01")
    ap.add_argument("--end-date", default="2018-12-31")
    ap.add_argument("--target-mode", choices=["diff", "logret"], default="diff")
    ap.add_argument("--seed", type=int, default=42, help="Random seed forwarded to replicate step")
    ap.add_argument("--fast", action="store_true", help="Use fast mode in replicate step")
    ap.add_argument("--tune", action="store_true", help="Tune NN/XGB on validation set in replicate step")
    ap.add_argument("--tune-trials", type=int, default=5, help="Max tuning candidates per model family/feature set")
    ap.add_argument("--export-tables", action="store_true", help="Export tables from replicate step")
    ap.add_argument("--export-plots", action="store_true", help="Export plots from replicate step")
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parent
    if not args.input_glob:
        args.input_glob = str(base_dir / "data" / "kaggle" / "raw" / "spy_raw_chain_*.parquet")
    if not args.output_csv:
        args.output_csv = str(base_dir / "data" / "kaggle" / "prepared" / "options_dataset.csv")
    if not args.outdir:
        args.outdir = str(base_dir / "outputs" / "kaggle_run")

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.export_tables or args.export_plots:
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
    prep_cmd = [
        sys.executable,
        str(base_dir / "prepare_kaggle_optionsdx.py"),
        "--input",
        "dummy",
        "--input-glob",
        args.input_glob,
        "--output",
        str(out_csv),
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
    ]
    run(prep_cmd)

    rep_cmd = [
        sys.executable,
        str(base_dir / "replicate_iv_paper.py"),
        "--csv",
        str(out_csv),
        "--target-mode",
        args.target_mode,
        "--outdir",
        args.outdir,
        "--seed",
        str(args.seed),
        "--tune-trials",
        str(args.tune_trials),
    ]
    if args.tune:
        rep_cmd.append("--tune")
    if args.export_tables:
        rep_cmd.append("--export-tables")
    if args.export_plots:
        rep_cmd.append("--export-plots")
    if args.fast:
        rep_cmd.append("--fast")
    run(rep_cmd)

    print("\nDone (Kaggle-only pipeline).")
    print(f"- Prepared CSV: {out_csv}")
    if args.export_tables or args.export_plots:
        print(f"- Replication outputs: {args.outdir}")
    else:
        print("- Replication outputs: none (console-only run)")


if __name__ == "__main__":
    main()

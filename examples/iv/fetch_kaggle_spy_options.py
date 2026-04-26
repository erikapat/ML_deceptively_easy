#!/usr/bin/env python3
"""
Download the public Kaggle dataset via kagglehub and copy the main data file
to ./data/kaggle/raw for downstream processing.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DATASET = "dudesurfin/spy-options-eod-volatility-surface-2010-2023"


def find_data_file(root: Path) -> Path:
    cands = list(root.rglob("*.parquet")) + list(root.rglob("*.csv"))
    if not cands:
        raise FileNotFoundError(f"No .parquet/.csv files found under: {root}")
    # Prefer parquet and largest file.
    cands = sorted(cands, key=lambda p: (p.suffix.lower() != ".parquet", -p.stat().st_size))
    return cands[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch SPY options dataset from KaggleHub.")
    ap.add_argument("--output-dir", default="", help="Directory to place selected dataset file")
    ap.add_argument(
        "--output-name",
        default="SPY_Options_EOD_2010_2023.parquet",
        help="Output filename in output-dir",
    )
    args = ap.parse_args()

    try:
        import kagglehub
    except Exception as e:
        raise RuntimeError(
            "kagglehub is not installed. Install with:\n"
            "python -m pip install kagglehub"
        ) from e

    base_dir = Path(__file__).resolve().parent
    if not args.output_dir:
        args.output_dir = str(base_dir / "data" / "kaggle" / "raw")
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(kagglehub.dataset_download(DATASET))
    source_file = find_data_file(dataset_path)

    out_path = out_dir / args.output_name
    shutil.copy2(source_file, out_path)

    print(f"KaggleHub cache path: {dataset_path}")
    print(f"Selected file: {source_file}")
    print(f"Copied to: {out_path}")


if __name__ == "__main__":
    main()

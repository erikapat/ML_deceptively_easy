#!/usr/bin/env python3
"""
Prepare a capital-gain panel dataset from UK Land Registry Price Paid Data (real public data).

Expected raw files:
- CSV without header (official PPD layout), e.g. pp-2019.csv, pp-2020.csv, ...

Output columns include:
- month, area_id, property_type, median_price, tx_count
- lag features
- target_diff, target_logret
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

PPD_COLS = [
    "txn_id",
    "price",
    "date_of_transfer",
    "postcode",
    "property_type",
    "old_new",
    "duration",
    "paon",
    "saon",
    "street",
    "locality",
    "town_city",
    "district",
    "county",
    "ppd_category_type",
    "record_status",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare UK PPD capital-gain dataset.")
    p.add_argument(
        "--input-glob",
        type=str,
        default="examples/capital_gain/raw/pp-*.csv",
        help="Glob for raw PPD CSV files.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="examples/capital_gain/data/capital_gain_dataset.csv",
        help="Output prepared CSV path.",
    )
    p.add_argument("--start-date", type=str, default="2010-01-01")
    p.add_argument("--end-date", type=str, default="2023-12-31")
    p.add_argument(
        "--min-tx-per-cell",
        type=int,
        default=5,
        help="Minimum monthly transactions per area/property_type cell.",
    )
    p.add_argument(
        "--top-areas",
        type=int,
        default=120,
        help="Keep top-N areas by total transaction count.",
    )
    return p.parse_args()


def read_raw(files: List[Path]) -> pd.DataFrame:
    chunks = []
    for fp in files:
        df = pd.read_csv(fp, header=None, names=PPD_COLS, low_memory=False)
        chunks.append(df)
    if not chunks:
        raise FileNotFoundError("No raw files found. Put pp-*.csv under examples/capital_gain/raw/")
    return pd.concat(chunks, axis=0, ignore_index=True)


def build_area_id(df: pd.DataFrame) -> pd.Series:
    # Outcode gives stable spatial grouping; fallback to district.
    pc = df["postcode"].astype(str).str.upper().str.strip()
    outcode = pc.str.extract(r"^([A-Z]{1,2}[0-9][0-9A-Z]?)", expand=False)
    area = outcode.fillna(df["district"].astype(str).str.upper().str.strip())
    area = area.replace({"": np.nan, "NAN": np.nan})
    return area


def main() -> None:
    args = parse_args()

    files = sorted(Path().glob(args.input_glob))
    if not files:
        raise FileNotFoundError(
            f"No files match: {args.input_glob}\n"
            "Download/put UK Price Paid CSV files in examples/capital_gain/raw/."
        )

    raw = read_raw(files)
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
    raw["date_of_transfer"] = pd.to_datetime(raw["date_of_transfer"], errors="coerce")
    raw["area_id"] = build_area_id(raw)
    raw["property_type"] = raw["property_type"].astype(str).str.upper().str.strip()

    keep_types = {"D", "S", "T", "F", "O"}
    df = raw[
        (raw["price"] > 1)
        & raw["date_of_transfer"].notna()
        & raw["area_id"].notna()
        & raw["property_type"].isin(keep_types)
    ].copy()

    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    df = df[(df["date_of_transfer"] >= start) & (df["date_of_transfer"] <= end)].copy()

    df["month"] = df["date_of_transfer"].dt.to_period("M").dt.to_timestamp()

    # Keep top areas to control sparsity.
    top_areas = (
        df.groupby("area_id").size().sort_values(ascending=False).head(args.top_areas).index
    )
    df = df[df["area_id"].isin(top_areas)].copy()

    g = (
        df.groupby(["month", "area_id", "property_type"], as_index=False)
        .agg(median_price=("price", "median"), tx_count=("price", "size"))
    )
    g = g[g["tx_count"] >= args.min_tx_per_cell].copy()

    g = g.sort_values(["area_id", "property_type", "month"]).reset_index(drop=True)
    g["panel_id"] = g["area_id"].astype(str) + "_" + g["property_type"].astype(str)

    by_panel = g.groupby("panel_id")
    g["price_lag1"] = by_panel["median_price"].shift(1)
    g["price_lag3_mean"] = by_panel["median_price"].shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    g["tx_lag1"] = by_panel["tx_count"].shift(1)
    g["target_diff"] = g["median_price"] - g["price_lag1"]
    with np.errstate(divide="ignore", invalid="ignore"):
        g["target_logret"] = np.log(g["median_price"] / g["price_lag1"])

    g["month_num"] = g["month"].dt.month
    g["month_sin"] = np.sin(2 * np.pi * g["month_num"] / 12.0)
    g["month_cos"] = np.cos(2 * np.pi * g["month_num"] / 12.0)

    for c in ["target_logret", "target_diff", "price_lag1", "price_lag3_mean", "tx_lag1"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
        g.loc[~np.isfinite(g[c]), c] = np.nan

    out = g.dropna(subset=["target_logret", "price_lag1", "price_lag3_mean", "tx_lag1"]).copy()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("Prepared capital-gain dataset")
    print(f"Raw files: {len(files)}")
    print(f"Rows raw valid: {len(df)}")
    print(f"Rows prepared: {len(out)}")
    if len(out):
        print(f"Date range: {out['month'].min()} -> {out['month'].max()}")
        print(f"Panels: {out['panel_id'].nunique()}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

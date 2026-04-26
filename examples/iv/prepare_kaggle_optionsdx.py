#!/usr/bin/env python3
"""
Convert SPY Options EOD datasets from Kaggle into the flat CSV
expected by replicate_iv_paper.py.

Output columns:
- date
- option_id
- iv
- delta
- dte
- spy_ret
- vix
- target_diff
- target_logret
- option_type
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def pick_first(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        c2 = c.strip().lower().replace(" ", "")
        c2 = c2.replace("[", "").replace("]", "")
        c2 = c2.replace("_", "")
        ren[c] = c2
    return df.rename(columns=ren)


def read_table(path: Path) -> pd.DataFrame:
    suffixes = [s.lower() for s in path.suffixes]
    is_parquet_like = (".parquet" in suffixes) or (path.suffix.lower() == ".pq")
    if is_parquet_like:
        try:
            return pd.read_parquet(path)
        except Exception as e:
            raise RuntimeError(
                "Failed to read parquet. Install pyarrow or fastparquet.\n"
                "Example: python -m pip install pyarrow"
            ) from e
    return pd.read_csv(path)


def read_many_tables(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        frames.append(read_table(p))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_spy_ret(df: pd.DataFrame, underlying_col: str) -> pd.DataFrame:
    daily = (
        df.groupby("date", as_index=False)[underlying_col]
        .median()
        .rename(columns={underlying_col: "spy_close"})
        .sort_values("date")
    )
    daily["spy_ret"] = daily["spy_close"].pct_change()
    return daily[["date", "spy_ret"]]


def load_vix(vix_csv: Path | None, dates: pd.Series, spy_ret: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"date": pd.to_datetime(dates).drop_duplicates().sort_values()})
    if vix_csv is not None:
        v = normalize_columns(read_table(vix_csv))
        if "date" not in v.columns:
            raise ValueError("VIX CSV must contain a `date` column.")
        v["date"] = pd.to_datetime(v["date"])
        cand = ["vix", "close", "adjclose", "adj_close"]
        vcol = next((c for c in cand if c in v.columns), None)
        if vcol is None:
            raise ValueError("VIX CSV must contain one of: vix, close, adjclose, adj_close.")
        v = v[["date", vcol]].rename(columns={vcol: "vix"})
        out = d.merge(v, on="date", how="left")
        out["vix"] = out["vix"].ffill().bfill()
        return out

    # Fallback proxy: 20d realized vol from SPY returns (scaled to %).
    proxy = pd.DataFrame({"date": d["date"].copy(), "spy_ret": spy_ret.values})
    proxy = proxy.sort_values("date")
    proxy["vix"] = (
        proxy["spy_ret"].rolling(20, min_periods=5).std() * np.sqrt(252) * 100.0
    )
    proxy["vix"] = proxy["vix"].ffill().bfill()
    if proxy["vix"].isna().all():
        proxy["vix"] = 20.0
    return proxy[["date", "vix"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare Kaggle SPY options data for replicate_iv_paper.py")
    ap.add_argument("--input", required=True, help="Path to input .parquet or .csv")
    ap.add_argument(
        "--input-glob",
        default="",
        help="Optional glob pattern to read multiple files (e.g., ./examples/iv/data/kaggle/raw/spy_raw_chain_*.parquet). Overrides --input.",
    )
    ap.add_argument("--output", default="", help="Output CSV path")
    ap.add_argument("--vix-csv", default="", help="Optional VIX daily CSV with date+close/vix")
    ap.add_argument("--start-date", default="2010-01-01")
    ap.add_argument("--end-date", default="2023-12-31")
    ap.add_argument("--sample-frac", type=float, default=1.0, help="Optional sampling fraction (0,1]")
    ap.add_argument("--min-dte", type=float, default=14.0, help="Minimum DTE filter")
    ap.add_argument("--min-delta", type=float, default=0.05, help="Minimum call delta filter")
    ap.add_argument("--max-delta", type=float, default=0.95, help="Maximum call delta filter")
    ap.add_argument(
        "--strike-decimals",
        type=int,
        default=2,
        help="Decimals used to normalize strike inside option_id",
    )
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parent
    if not args.output:
        args.output = str(base_dir / "data" / "kaggle" / "prepared" / "options_dataset.csv")

    in_path = Path(args.input).expanduser()
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vix_path = Path(args.vix_csv).expanduser() if args.vix_csv else None

    if args.input_glob:
        glob_matches = sorted(Path(".").glob(args.input_glob))
        if not glob_matches:
            raise FileNotFoundError(
                f"No files matched --input-glob: {args.input_glob}\n"
                "Example: --input-glob './examples/iv/data/kaggle/raw/spy_raw_chain_*.parquet'"
            )
        in_paths = [p.expanduser() for p in glob_matches]
    else:
        if not in_path.exists():
            hint = (
                f"Input not found: {in_path}\n"
                "Check your actual filename/path. Example helpers:\n"
                "  find ./examples/iv/data/kaggle/raw -maxdepth 3 \\( -name \"*.parquet\" -o -name \"*.csv\" \\)\n"
                "  ls -lah ./examples/iv/data/kaggle/raw"
            )
            raise FileNotFoundError(hint)
        in_paths = [in_path]
    if vix_path is not None and not vix_path.exists():
        raise FileNotFoundError(f"VIX CSV not found: {vix_path}")
    if not (0 < args.sample_frac <= 1.0):
        raise ValueError("--sample-frac must be in (0,1].")
    if args.min_delta > args.max_delta:
        raise ValueError("--min-delta cannot be greater than --max-delta")

    raw = normalize_columns(read_many_tables(in_paths))
    print(f"Loaded input files: {len(in_paths)}")
    if len(in_paths) <= 8:
        for p in in_paths:
            print(f"- {p}")
    else:
        print(f"- first: {in_paths[0]}")
        print(f"- last:  {in_paths[-1]}")
    n_raw = len(raw)

    required_base = {"quotedate", "strike"}
    missing = required_base - set(raw.columns)
    if missing:
        sample_cols = ", ".join(list(raw.columns[:40]))
        raise ValueError(
            f"Missing required option columns: {sorted(missing)}\n"
            f"First columns found: {sample_cols}"
        )

    underlying_col = pick_first(
        raw,
        [
            "underlyinglast",
            "underlyinglast1545",
            "underlyingprice",
            "underlyingclose",
            "underlying",
            "spot",
        ],
    )
    if underlying_col is None:
        raise ValueError("Missing underlying price column (expected UNDERLYING_LAST or similar).")

    iv_col = pick_first(
        raw,
        [
            "civ",
            "calliv",
            "impliedvolatility",
            "iv",
            "cimpliedvolatility",
        ],
    )
    if iv_col is None:
        raise ValueError("Missing IV column (expected C_IV/call IV or similar).")

    delta_col = pick_first(
        raw,
        [
            "cdelta",
            "calldelta",
            "delta",
            "cdelta1545",
            "delta1545",
        ],
    )

    dte_col = pick_first(raw, ["dte", "daystoexpiration", "daystoexpiry", "timetoexpiry"])
    date_col = pick_first(raw, ["quotedate", "date"])
    expiry_col = pick_first(raw, ["expiredate", "expiration", "expiry", "expdate"])
    expiry_unix_col = pick_first(raw, ["expireunix", "expiryunix", "expirationunix", "expunix"])

    if date_col is None or (expiry_col is None and expiry_unix_col is None):
        raise ValueError(
            "Missing date/expiry columns (expected QUOTE_DATE and EXPIRE_DATE/EXPIRE_UNIX or similar)."
        )

    raw["date"] = pd.to_datetime(raw[date_col])
    if expiry_col is not None:
        raw["expiry"] = pd.to_datetime(raw[expiry_col], errors="coerce")
    else:
        # Support datasets that store expiration as unix epoch.
        exp_num = pd.to_numeric(raw[expiry_unix_col], errors="coerce")
        med = float(exp_num.dropna().median()) if exp_num.notna().any() else np.nan
        # Heuristic: very large values are typically milliseconds.
        unit = "ms" if np.isfinite(med) and med > 1e11 else "s"
        raw["expiry"] = pd.to_datetime(exp_num, unit=unit, errors="coerce")
    raw["strike"] = pd.to_numeric(raw["strike"], errors="coerce")
    raw["iv"] = pd.to_numeric(raw[iv_col], errors="coerce")

    # Some public files don't include call delta. Build a monotonic proxy from moneyness.
    if delta_col is not None:
        raw["delta"] = pd.to_numeric(raw[delta_col], errors="coerce")
    else:
        spot = pd.to_numeric(raw[underlying_col], errors="coerce")
        m = np.log(np.clip(spot, 1e-8, None) / np.clip(raw["strike"], 1e-8, None))
        raw["delta"] = 1.0 / (1.0 + np.exp(-8.0 * m))
        raw["delta"] = raw["delta"].clip(0.01, 0.99)
        print("Warning: Delta column not found. Using moneyness-based proxy delta.")

    if dte_col is not None:
        raw["dte"] = pd.to_numeric(raw[dte_col], errors="coerce")
    else:
        raw["dte"] = (raw["expiry"] - raw["date"]).dt.days.astype(float)

    raw = raw.dropna(subset=["date", "expiry", "strike", "iv", "delta", "dte", underlying_col]).copy()
    raw = raw[(raw["date"] >= pd.to_datetime(args.start_date)) & (raw["date"] <= pd.to_datetime(args.end_date))]
    n_after_basic = len(raw)

    if args.sample_frac < 1.0:
        raw = raw.sample(frac=args.sample_frac, random_state=42)

    print(
        "Using columns:",
        {
            "date": date_col,
            "expiry": expiry_col if expiry_col is not None else f"{expiry_unix_col} (unix)",
            "underlying": underlying_col,
            "iv": iv_col,
            "delta": delta_col if delta_col is not None else "delta_proxy_from_moneyness",
            "dte": dte_col if dte_col is not None else "computed_from_expiry-date",
        },
    )

    spy_daily = build_spy_ret(raw, underlying_col=underlying_col)
    merged = raw.merge(spy_daily, on="date", how="left")

    # VIX handling:
    # - If dataset already has `vix`, keep it as primary.
    # - Merge external/proxy daily VIX as fallback only.
    has_raw_vix = "vix" in merged.columns
    if has_raw_vix and vix_path is None:
        merged["vix"] = pd.to_numeric(merged["vix"], errors="coerce")
    else:
        vix_daily = load_vix(vix_path, spy_daily["date"], spy_daily["spy_ret"])
        merged = merged.merge(vix_daily, on="date", how="left", suffixes=("", "_daily"))
        if "vix_daily" in merged.columns:
            if "vix" in merged.columns:
                merged["vix"] = pd.to_numeric(merged["vix"], errors="coerce").combine_first(
                    pd.to_numeric(merged["vix_daily"], errors="coerce")
                )
                merged = merged.drop(columns=["vix_daily"])
            else:
                merged = merged.rename(columns={"vix_daily": "vix"})

    merged["option_type"] = "call"
    expiry_str = merged["expiry"].dt.strftime("%Y-%m-%d").astype(str).to_numpy()
    strike_fmt = "{:." + str(args.strike_decimals) + "f}"
    strike_str = (
        pd.to_numeric(merged["strike"], errors="coerce")
        .map(lambda x: strike_fmt.format(x) if pd.notna(x) else "nan")
        .astype(str)
        .to_numpy()
    )
    option_id = np.char.add("SPY_", expiry_str)
    option_id = np.char.add(option_id, "_")
    option_id = np.char.add(option_id, strike_str)
    option_id = np.char.add(option_id, "_C")
    merged["option_id"] = option_id

    merged = merged.sort_values(["option_id", "date"]).copy()
    merged["target_diff"] = merged.groupby("option_id")["iv"].diff()
    prev_iv = merged.groupby("option_id")["iv"].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        merged["target_logret"] = np.log(merged["iv"] / prev_iv)
    n_after_target = len(merged)
    n_non_null_target = int(merged["target_diff"].notna().sum())
    repeated_options = int((merged.groupby("option_id").size() >= 2).sum())

    merged = merged[
        (merged["dte"] > args.min_dte)
        & (merged["delta"] >= args.min_delta)
        & (merged["delta"] <= args.max_delta)
    ]
    merged = merged.dropna(subset=["target_diff", "spy_ret", "vix"])
    n_after_filters = len(merged)

    out = merged[
        [
            "date",
            "option_id",
            "iv",
            "delta",
            "dte",
            "spy_ret",
            "vix",
            "target_diff",
            "target_logret",
            "option_type",
        ]
    ].copy()
    out.to_csv(out_path, index=False)

    opt_counts = merged.groupby("option_id").size() if len(merged) else pd.Series(dtype=int)
    print("Row diagnostics:")
    print(f"- raw input rows: {n_raw}")
    print(f"- after basic parsing/date range: {n_after_basic}")
    print(f"- after target engineering (rows kept): {n_after_target}")
    print(f"- rows with non-null target_diff: {n_non_null_target}")
    print(f"- option_id with >=2 observations: {repeated_options}")
    print(f"- final rows after filters: {n_after_filters}")
    print("Option_id diagnostics:")
    if len(opt_counts):
        pct_repeat = float((opt_counts >= 2).mean()) * 100.0
        print(f"- unique option_id: {opt_counts.shape[0]}")
        print(f"- % option_id with >=2 rows (post-filter): {pct_repeat:.2f}%")
        print("- top option_id counts:")
        print(opt_counts.sort_values(ascending=False).head(5).to_string())
    else:
        print("- no option_id left after filters")

    print(f"Saved: {out_path}")
    print(f"Rows: {len(out)}")
    print(f"Date range: {out['date'].min()} -> {out['date'].max()}")
    print(f"Unique options: {out['option_id'].nunique()}")
    if vix_path is None:
        print("Note: `vix` was generated as a proxy from 20-day realized volatility of SPY returns.")
    if len(out) == 0:
        raise ValueError(
            "Output dataset is empty after filtering.\n"
            "Likely causes:\n"
            "1) option_id has almost no repeated observations across dates -> target_diff becomes NaN\n"
            "2) very aggressive filtering (dte/delta/date range)\n"
            "Try:\n"
            "- inspect row diagnostics above\n"
            "- verify the input is raw chain-level options data, not already aggregated surface data\n"
            "- use a broader date range and/or inspect option_id construction"
        )


if __name__ == "__main__":
    main()

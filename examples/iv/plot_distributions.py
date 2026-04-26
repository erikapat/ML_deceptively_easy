#!/usr/bin/env python3
"""
Plot feature distributions for:
1) all numeric columns in raw Kaggle SPY options data
2) model-focused columns in prepared CSV

Outputs:
- fig_all_kaggle_numeric.png
- fig_model_variables.png
- stats_all_kaggle_numeric.csv
- stats_model_variables.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_raw_files(input_path: str, input_glob: str, max_rows_per_file: int, seed: int) -> pd.DataFrame:
    if input_glob:
        paths = sorted(Path(".").glob(input_glob))
        if not paths:
            raise FileNotFoundError(f"No files matched --input-glob: {input_glob}")
    else:
        p = Path(input_path)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        paths = [p]

    frames = []
    for p in paths:
        df = read_table(p)
        if max_rows_per_file > 0 and len(df) > max_rows_per_file:
            df = df.sample(n=max_rows_per_file, random_state=seed)
        frames.append(df)
        print(f"Loaded {len(df):,} rows from {p}")

    out = pd.concat(frames, ignore_index=True)
    print(f"Total raw rows for EDA: {len(out):,}")
    return out


def plot_hist_grid(
    df: pd.DataFrame,
    columns: list[str],
    title: str,
    out_path: Path,
    bins: int = 60,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> None:
    if not columns:
        raise ValueError(f"No columns available for plotting: {title}")

    n = len(columns)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(columns):
        ax = axes[i]
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            ax.set_title(f"{col}\n(no numeric data)")
            ax.axis("off")
            continue
        lo, hi = s.quantile(q_low), s.quantile(q_high)
        s_clip = s.clip(lower=lo, upper=hi)
        ax.hist(s_clip, bins=bins)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("Frequency")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def summary_stats(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for c in columns:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) == 0:
            continue
        rows.append(
            {
                "column": c,
                "count": len(s),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "p01": float(s.quantile(0.01)),
                "p50": float(s.quantile(0.50)),
                "p99": float(s.quantile(0.99)),
                "max": float(s.max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot raw Kaggle and model-variable distributions.")
    ap.add_argument("--input", default="", help="Single raw file path (.parquet/.csv)")
    ap.add_argument("--input-glob", default="")
    ap.add_argument("--prepared-csv", default="")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--max-rows-per-file", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    base_dir = Path(__file__).resolve().parent
    if not args.input_glob and not args.input:
        args.input_glob = str(base_dir / "data" / "kaggle" / "raw" / "spy_raw_chain_*.parquet")
    if not args.prepared_csv:
        args.prepared_csv = str(base_dir / "data" / "kaggle" / "prepared" / "options_dataset.csv")
    if not args.outdir:
        args.outdir = str(base_dir / "outputs" / "iv_distributions")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_files(args.input, args.input_glob, args.max_rows_per_file, args.seed)
    raw_numeric = raw.select_dtypes(include=[np.number]).columns.tolist()
    if not raw_numeric:
        raise ValueError("No numeric columns found in raw dataset.")

    plot_hist_grid(
        raw,
        raw_numeric,
        "All Numeric Variables in Kaggle Raw Dataset",
        outdir / "fig_all_kaggle_numeric.png",
    )
    stats_all = summary_stats(raw, raw_numeric)
    stats_all.to_csv(outdir / "stats_all_kaggle_numeric.csv", index=False)

    p = Path(args.prepared_csv)
    if not p.exists():
        raise FileNotFoundError(
            f"Prepared CSV not found: {p}\n"
            "Run prepare_kaggle_optionsdx.py first, then rerun this script."
        )
    prep = pd.read_csv(p)
    model_cols_pref = ["delta", "dte", "spy_ret", "vix", "target_diff", "iv"]
    model_cols = [c for c in model_cols_pref if c in prep.columns]
    if not model_cols:
        raise ValueError("Prepared CSV does not contain expected model columns.")

    plot_hist_grid(
        prep,
        model_cols,
        "Model Variables Distribution (IV Example)",
        outdir / "fig_model_variables.png",
    )
    stats_model = summary_stats(prep, model_cols)
    stats_model.to_csv(outdir / "stats_model_variables.csv", index=False)

    print(f"Saved: {outdir / 'fig_all_kaggle_numeric.png'}")
    print(f"Saved: {outdir / 'fig_model_variables.png'}")
    print(f"Saved: {outdir / 'stats_all_kaggle_numeric.csv'}")
    print(f"Saved: {outdir / 'stats_model_variables.csv'}")


if __name__ == "__main__":
    main()

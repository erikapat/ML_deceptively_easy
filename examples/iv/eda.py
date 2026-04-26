#!/usr/bin/env python3
"""
EDA for IV example:
1) all numeric variables from raw Kaggle files
2) model variables from prepared CSV
3) target transformations diagnostics
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


def clip_quantile(s: pd.Series, q_low: float = 0.01, q_high: float = 0.99) -> pd.Series:
    lo, hi = s.quantile(q_low), s.quantile(q_high)
    return s.clip(lower=lo, upper=hi)


def plot_hist_grid(
    df: pd.DataFrame,
    columns: list[str],
    title: str,
    out_path: Path,
    bins: int = 60,
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
        s = clip_quantile(s)
        ax.hist(s, bins=bins)
        ax.set_title(col)
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


def build_target_transforms(prep: pd.DataFrame) -> pd.DataFrame:
    out = prep.copy()
    if "target_diff" not in out.columns:
        raise ValueError("Prepared CSV must include `target_diff`.")

    out["target_abs"] = out["target_diff"].abs()
    out["target_sign"] = np.sign(out["target_diff"])
    out["target_signed_log"] = np.sign(out["target_diff"]) * np.log1p(np.abs(out["target_diff"]))

    if "target_logret" not in out.columns and "iv" in out.columns and "option_id" in out.columns:
        out = out.sort_values(["option_id", "date"]).copy()
        prev_iv = out.groupby("option_id")["iv"].shift(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["target_logret"] = np.log(out["iv"] / prev_iv)
    return out


def plot_target_transforms(prep: pd.DataFrame, outdir: Path) -> None:
    tdf = build_target_transforms(prep)
    cols = ["target_diff", "target_logret", "target_signed_log", "target_abs"]
    cols = [c for c in cols if c in tdf.columns]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.reshape(-1)
    for i, c in enumerate(cols):
        s = pd.to_numeric(tdf[c], errors="coerce").dropna()
        if len(s) == 0:
            axes[i].set_title(f"{c} (empty)")
            axes[i].axis("off")
            continue
        axes[i].hist(clip_quantile(s), bins=80)
        axes[i].set_title(c)
        axes[i].set_ylabel("Frequency")

    for j in range(len(cols), 4):
        axes[j].axis("off")

    fig.suptitle("Target and Transformations", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outdir / "fig_target_transformations.png", dpi=160)
    plt.close(fig)

    # Time evolution at daily aggregation (median), to compare stability of transforms.
    if "date" in tdf.columns:
        d = tdf.copy()
        d["date"] = pd.to_datetime(d["date"])
        daily = (
            d.groupby("date", as_index=False)[[c for c in cols if c in d.columns]]
            .median(numeric_only=True)
            .sort_values("date")
        )
        plt.figure(figsize=(12, 5))
        for c in ["target_diff", "target_logret", "target_signed_log"]:
            if c in daily.columns:
                plt.plot(daily["date"], daily[c], label=c, linewidth=1.1)
        plt.legend()
        plt.title("Daily Median Target vs Transformations")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(outdir / "fig_target_transformations_timeseries.png", dpi=160)
        plt.close()

    stats_cols = [c for c in cols if c in tdf.columns]
    summary_stats(tdf, stats_cols).to_csv(outdir / "stats_target_transformations.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="IV EDA: raw vars, model vars, and target transformations.")
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
        args.outdir = str(base_dir / "outputs" / "iv_eda")

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
    summary_stats(raw, raw_numeric).to_csv(outdir / "stats_all_kaggle_numeric.csv", index=False)

    p = Path(args.prepared_csv)
    if not p.exists():
        raise FileNotFoundError(
            f"Prepared CSV not found: {p}\nRun prepare_kaggle_optionsdx.py first."
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
    summary_stats(prep, model_cols).to_csv(outdir / "stats_model_variables.csv", index=False)

    plot_target_transforms(prep, outdir)

    print(f"Saved EDA outputs in: {outdir}")


if __name__ == "__main__":
    main()

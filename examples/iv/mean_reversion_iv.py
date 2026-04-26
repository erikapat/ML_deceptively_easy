#!/usr/bin/env python3
"""
Mean-reversion analysis for IV panel data (coverage-aware).

Outputs:
- table_option_coverage_stats.csv
- table_mean_reversion_betas.csv
- table_decile_profile.csv
- fig_mean_reversion_profile.png
- summary_mean_reversion.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def wls_alpha_beta(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> Tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if w is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(w, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[mask], y[mask], w[mask]
    if len(x) < 5:
        return float("nan"), float("nan"), 0

    X = np.column_stack([np.ones(len(x)), x])
    WX = X * w[:, None]
    beta = np.linalg.lstsq(WX.T @ X, WX.T @ y, rcond=None)[0]
    return float(beta[0]), float(beta[1]), int(len(x))


def main() -> None:
    ap = argparse.ArgumentParser(description="Coverage-aware mean-reversion test for IV data.")
    ap.add_argument(
        "--csv",
        default="",
        help="Prepared CSV path (must include: date, option_id, iv).",
    )
    ap.add_argument("--window", type=int, default=20, help="Past-only rolling window size.")
    ap.add_argument("--min-periods", type=int, default=10, help="Min periods for rolling center.")
    ap.add_argument("--min-obs-per-option", type=int, default=60, help="Coverage filter threshold.")
    ap.add_argument("--outdir", default="", help="Output directory.")
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parent
    if not args.csv:
        args.csv = str(base_dir / "data" / "kaggle" / "prepared" / "options_dataset.csv")
    if not args.outdir:
        args.outdir = str(base_dir / "outputs" / "mean_reversion")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    needed = {"date", "option_id", "iv"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for mean-reversion analysis: {sorted(missing)}")

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["iv"] = pd.to_numeric(work["iv"], errors="coerce")
    work = work.dropna(subset=["date", "option_id", "iv"]).copy()
    work = work.sort_values(["option_id", "date"]).reset_index(drop=True)

    # Coverage diagnostics
    obs_per_option = work.groupby("option_id").size().rename("n_obs")
    coverage_stats = obs_per_option.describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]).to_frame("value")
    coverage_stats.to_csv(outdir / "table_option_coverage_stats.csv")

    # Past-only center and next-day move
    g = work.groupby("option_id")["iv"]
    work["iv_center_mean"] = g.transform(
        lambda s: s.shift(1).rolling(args.window, min_periods=args.min_periods).mean()
    )
    work["iv_center_median"] = g.transform(
        lambda s: s.shift(1).rolling(args.window, min_periods=args.min_periods).median()
    )
    work["dev_mean"] = work["iv"] - work["iv_center_mean"]
    work["dev_median"] = work["iv"] - work["iv_center_median"]
    work["iv_next"] = g.shift(-1)
    work["delta_iv_next"] = work["iv_next"] - work["iv"]

    mr = work.dropna(subset=["dev_mean", "dev_median", "delta_iv_next"]).copy()
    mr = mr.merge(obs_per_option.to_frame(), left_on="option_id", right_index=True, how="left")
    mr["w_option_balanced"] = 1.0 / mr["n_obs"].clip(lower=1)

    # Beta table
    rows = []
    for dev_col in ["dev_mean", "dev_median"]:
        a, b, n = wls_alpha_beta(mr[dev_col].to_numpy(), mr["delta_iv_next"].to_numpy())
        rows.append({"spec": "pooled_unweighted", "dev": dev_col, "alpha": a, "beta": b, "n": n})

        a, b, n = wls_alpha_beta(
            mr[dev_col].to_numpy(),
            mr["delta_iv_next"].to_numpy(),
            mr["w_option_balanced"].to_numpy(),
        )
        rows.append({"spec": "pooled_option_balanced", "dev": dev_col, "alpha": a, "beta": b, "n": n})

        sub = mr[mr["n_obs"] >= args.min_obs_per_option]
        a, b, n = wls_alpha_beta(
            sub[dev_col].to_numpy(),
            sub["delta_iv_next"].to_numpy(),
            sub["w_option_balanced"].to_numpy(),
        )
        rows.append(
            {
                "spec": f"filtered_nobs>={args.min_obs_per_option}_option_balanced",
                "dev": dev_col,
                "alpha": a,
                "beta": b,
                "n": n,
            }
        )

    beta_tbl = pd.DataFrame(rows)
    beta_tbl.to_csv(outdir / "table_mean_reversion_betas.csv", index=False)

    # Decile profile plot
    sub = mr[["dev_mean", "delta_iv_next", "w_option_balanced"]].dropna().copy()
    sub["decile"] = pd.qcut(sub["dev_mean"], 10, labels=False, duplicates="drop")
    dec = (
        sub.groupby("decile")
        .agg(
            dev_mean=("dev_mean", "mean"),
            delta_mean=("delta_iv_next", "mean"),
            delta_wmean=("delta_iv_next", lambda z: np.average(z, weights=sub.loc[z.index, "w_option_balanced"])),
        )
        .reset_index()
    )
    dec.to_csv(outdir / "table_decile_profile.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(dec["dev_mean"], dec["delta_mean"], marker="o", label="Unweighted")
    ax.plot(dec["dev_mean"], dec["delta_wmean"], marker="o", label="Option-balanced")
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Mean-reversion profile: dev_t vs avg Delta IV_{t+1} (deciles)")
    ax.set_xlabel("Average deviation (dev_t)")
    ax.set_ylabel("Average next-day IV change")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig_mean_reversion_profile.png", dpi=160)
    plt.close(fig)

    # Summary
    betas = beta_tbl.pivot(index="spec", columns="dev", values="beta")
    lines = [
        f"Input CSV: {csv_path}",
        f"Rows in source: {len(df):,}",
        f"Rows used in mean-reversion sample: {len(mr):,}",
        f"Unique option_id in sample: {mr['option_id'].nunique():,}",
        f"Window: {args.window} (min_periods={args.min_periods})",
        f"Coverage filter: n_obs >= {args.min_obs_per_option}",
        "",
        "Interpretation guide:",
        "- Mean reversion support: beta < 0.",
        "- Coverage bias check: if beta changes strongly across unweighted vs option-balanced specs,",
        "  then panel coverage imbalance is driving part of the effect.",
        "",
        "Beta snapshot:",
        betas.to_string(),
    ]
    (outdir / "summary_mean_reversion.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved mean-reversion outputs in: {outdir}")
    print("Files:")
    print("- table_option_coverage_stats.csv")
    print("- table_mean_reversion_betas.csv")
    print("- table_decile_profile.csv")
    print("- fig_mean_reversion_profile.png")
    print("- summary_mean_reversion.txt")


if __name__ == "__main__":
    main()


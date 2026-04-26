#!/usr/bin/env python3
"""
Capital gain leakage demo (Track A, real-estate panel).
Compares RANDOM row split vs CHRONO month split.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class Split:
    train_idx: np.ndarray
    test_idx: np.ndarray


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def random_split(n: int, train_frac: float, seed: int) -> Split:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    ntr = int(n * train_frac)
    return Split(train_idx=idx[:ntr], test_idx=idx[ntr:])


def chrono_split_month(df: pd.DataFrame, train_frac: float) -> Split:
    months = pd.to_datetime(df["month"]).dt.to_period("M").astype(str)
    uniq = np.array(sorted(months.unique()))
    ntr = int(len(uniq) * train_frac)
    mtr = set(uniq[:ntr])
    idx = np.arange(len(df))
    train_mask = months.isin(mtr).to_numpy()
    return Split(train_idx=idx[train_mask], test_idx=idx[~train_mask])


def run_models(df: pd.DataFrame, split: Split, target_col: str, seed: int) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    feat = [
        c
        for c in ["price_lag1", "price_lag3_mean", "tx_lag1", "month_sin", "month_cos"]
        if c in df.columns
    ]
    if len(feat) < 3:
        raise ValueError(f"Not enough features. Found: {feat}")

    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    X = df[feat].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X = X[keep]
    y = y[keep]

    # Remap split indices after filtering
    old_idx = np.arange(len(df))[keep]
    pos = {k: i for i, k in enumerate(old_idx)}
    tr = np.array([pos[i] for i in split.train_idx if i in pos], dtype=int)
    te = np.array([pos[i] for i in split.test_idx if i in pos], dtype=int)

    if len(tr) < 200 or len(te) < 100:
        raise ValueError("Train/test too small after filtering.")

    rows: List[Dict[str, float | str]] = []

    lr = LinearRegression()
    lr.fit(X[tr], y[tr])
    p_lr = lr.predict(X[te])
    rows.append({"model": "Linear", "mse": mse(y[te], p_lr), "r2": r2(y[te], p_lr)})

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=4,
    )
    rf.fit(X[tr], y[tr])
    p_rf = rf.predict(X[te])
    rows.append({"model": "RF", "mse": mse(y[te], p_rf), "r2": r2(y[te], p_rf)})

    # Leakage demonstrator: memorize month mean from train.
    tr_df = df.iloc[split.train_idx].copy()
    te_df = df.iloc[split.test_idx].copy()
    month_mean = tr_df.groupby("month", as_index=True)[target_col].mean().to_dict()
    gmean = float(tr_df[target_col].mean())
    p_mem = np.array([month_mean.get(m, gmean) for m in te_df["month"].tolist()], dtype=float)
    y_mem = te_df[target_col].to_numpy(dtype=float)
    rows.append({"model": "Memorizer_month", "mse": mse(y_mem, p_mem), "r2": r2(y_mem, p_mem)})

    return pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replicate capital-gain leakage experiment.")
    p.add_argument(
        "--csv",
        type=str,
        default="examples/capital_gain/data/capital_gain_dataset.csv",
        help="Prepared dataset path.",
    )
    p.add_argument("--target", choices=["target_logret", "target_diff"], default="target_logret")
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def print_block(title: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(lambda x: f"{x:.6f}")
    print(out.to_string(index=False))


def main() -> None:
    args = parse_args()
    p = Path(args.csv)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset not found: {p}\n"
            "Run prepare_capital_gain_data.py first."
        )

    df = pd.read_csv(p)
    req = {"month", "price_lag1", "price_lag3_mean", "tx_lag1", "month_sin", "month_cos", args.target}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")

    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.dropna(subset=["month", args.target]).reset_index(drop=True)

    s_rand = random_split(len(df), args.train, args.seed)
    s_chro = chrono_split_month(df, args.train)

    t_rand = run_models(df, s_rand, args.target, args.seed)
    t_chro = run_models(df, s_chro, args.target, args.seed)

    comp = t_rand.rename(columns={"mse": "mse_random", "r2": "r2_random"}).merge(
        t_chro.rename(columns={"mse": "mse_chrono", "r2": "r2_chrono"}),
        on="model",
        how="inner",
    )
    comp["delta_r2_random_minus_chrono"] = comp["r2_random"] - comp["r2_chrono"]
    comp["delta_mse_random_minus_chrono"] = comp["mse_random"] - comp["mse_chrono"]

    print(f"Data source: {p}")
    print(f"Rows: {len(df)}")
    print(f"Date range: {df['month'].min().date()} -> {df['month'].max().date()}")
    print(f"Target: {args.target}")

    print_block("RANDOM split", t_rand)
    print_block("CHRONO split", t_chro)
    print_block("Comparison RANDOM vs CHRONO", comp.sort_values("mse_chrono"))

    print("\nInterpretation:")
    print("- Random row split can overstate performance in panel time-series.")
    print("- Chronological month split is the primary OOS metric.")


if __name__ == "__main__":
    main()

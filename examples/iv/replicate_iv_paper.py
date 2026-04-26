#!/usr/bin/env python3
"""
IV forecasting benchmark on Kaggle-prepared dataset.

Model family:
- Benchmark_HullWhiteLike
- NN_3F, NN_4F
- XGB_2F, XGB_3F, XGB_4F

Compares random row split vs chronological day-grouped split to expose leakage.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SEED = 42


@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def split_date_summary(df: pd.DataFrame, split: Split, label: str) -> None:
    dates = pd.to_datetime(df["date"])

    def fmt(idx: np.ndarray) -> str:
        if len(idx) == 0:
            return "empty"
        d = dates.iloc[idx]
        return f"{d.min().date()} -> {d.max().date()} ({d.nunique()} unique days, {len(idx)} rows)"

    print(f"\nSplit summary [{label}]")
    print(f"- train: {fmt(split.train_idx)}")
    print(f"- val:   {fmt(split.val_idx)}")
    print(f"- test:  {fmt(split.test_idx)}")


def ensure_outdir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def weighted_accuracy(y_true_cont: np.ndarray, y_pred_cont: np.ndarray) -> float:
    w = np.abs(y_true_cont)
    true_cls = (y_true_cont > 0).astype(int)
    pred_cls = (y_pred_cont > 0).astype(int)
    correct = (true_cls == pred_cls).astype(float)
    denom = float(np.sum(w))
    if denom == 0:
        return float(np.mean(correct))
    return float(np.sum(w * correct) / denom)


def random_split(n: int, train: float, val: float, seed: int) -> Split:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train)
    n_val = int(n * val)
    return Split(idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :])


def chrono_day_split(days: np.ndarray, train: float, val: float) -> Split:
    uniq = np.sort(np.unique(days))
    n_days = len(uniq)
    n_train = int(n_days * train)
    n_val = int(n_days * val)
    d_train = set(uniq[:n_train])
    d_val = set(uniq[n_train : n_train + n_val])
    d_test = set(uniq[n_train + n_val :])
    idx = np.arange(len(days))
    tr = idx[np.array([d in d_train for d in days])]
    va = idx[np.array([d in d_val for d in days])]
    te = idx[np.array([d in d_test for d in days])]
    return Split(tr, va, te)


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((len(X), 1)), X], axis=1)


def fit_ols(X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    X1 = add_intercept(X)
    xtx = X1.T @ X1
    if ridge > 0:
        xtx = xtx + ridge * np.eye(xtx.shape[0])
    xty = X1.T @ y
    return np.linalg.solve(xtx, xty)


def predict_ols(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return add_intercept(X) @ beta


def baseline_hull_white_features(df: pd.DataFrame) -> np.ndarray:
    den = np.sqrt(np.clip(df["dte"].to_numpy(dtype=float), 1e-8, None))
    ret = df["spy_ret"].to_numpy(dtype=float)
    delta = df["delta"].to_numpy(dtype=float)
    x1 = (ret / den).reshape(-1, 1)
    x2 = (ret * delta / den).reshape(-1, 1)
    x3 = (ret * (delta**2) / den).reshape(-1, 1)
    return np.concatenate([x1, x2, x3], axis=1)


def model_features(df: pd.DataFrame, n_features: int) -> np.ndarray:
    cols = ["spy_ret", "delta"]
    if n_features >= 3:
        cols.append("dte")
    if n_features >= 4:
        cols.append("vix")
    return df[cols].to_numpy(dtype=float)


def read_tabular(path: Path) -> pd.DataFrame:
    suffixes = [s.lower() for s in path.suffixes]
    is_parquet_like = (".parquet" in suffixes) or (path.suffix.lower() == ".pq")
    if is_parquet_like:
        try:
            return pd.read_parquet(path)
        except Exception as e:
            raise RuntimeError(
                "Failed to read parquet input. Install pyarrow or fastparquet.\n"
                "Example: python -m pip install pyarrow"
            ) from e
    return pd.read_csv(path)


def prepare_dataframe_from_tabular(path: str) -> pd.DataFrame:
    """
    Minimum columns:
    - date, delta, dte, spy_ret, vix

    and either:
    - target_diff
      OR
    - target
      OR
    - iv + option_id (target_diff computed as iv_t - iv_{t-1} per option_id)

    Optional:
    - option_type (if present, keeps calls only)
    """
    df = read_tabular(Path(path))
    required = {"date", "delta", "dte", "spy_ret", "vix"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])

    if "target_diff" in df.columns:
        pass
    elif "target" in df.columns:
        df["target_diff"] = df["target"]
    elif "iv" in df.columns and "option_id" in df.columns:
        df = df.sort_values(["option_id", "date"]).copy()
        df["target_diff"] = df.groupby("option_id")["iv"].diff()
        iv_prev = df.groupby("option_id")["iv"].shift(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            df["target_logret"] = np.log(df["iv"] / iv_prev)
    else:
        raise ValueError("CSV needs `target_diff` (or `target`) or `iv+option_id`.")

    if "option_type" in df.columns:
        df = df[df["option_type"].astype(str).str.lower().isin(["c", "call"])].copy()

    df = df[(df["dte"] > 14) & (df["delta"] >= 0.05) & (df["delta"] <= 0.95)].copy()
    df = df.dropna(subset=["target_diff", "date", "delta", "dte", "spy_ret", "vix"])
    return df.reset_index(drop=True)


def try_import_sklearn():
    try:
        from sklearn.neural_network import MLPRegressor  # noqa: F401
        from sklearn.preprocessing import StandardScaler  # noqa: F401

        return True
    except Exception:
        return False


def try_import_xgboost():
    try:
        import xgboost  # noqa: F401

        return True
    except Exception:
        return False


def run_benchmark(df: pd.DataFrame, split: Split, y: np.ndarray) -> Tuple[np.ndarray, float]:
    Xb = baseline_hull_white_features(df)
    trva = np.concatenate([split.train_idx, split.val_idx])
    beta = fit_ols(Xb[trva], y[trva], ridge=1e-9)
    pred = predict_ols(Xb[split.test_idx], beta)
    mse_b = mse(y[split.test_idx], pred)
    return pred, mse_b


def run_nn(
    df: pd.DataFrame,
    split: Split,
    y: np.ndarray,
    n_features: int,
    seed: int,
    nn_max_iter: int,
    nn_hidden: Tuple[int, ...],
    tune: bool,
    tune_trials: int,
) -> Optional[np.ndarray]:
    if not try_import_sklearn():
        return None
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    X = model_features(df, n_features)
    Xtr = X[split.train_idx]
    Xva = X[split.val_idx]
    Xte = X[split.test_idx]
    ytr = y[split.train_idx]
    yva = y[split.val_idx]
    if len(Xtr) < 20 or len(Xte) == 0 or len(Xva) == 0:
        return None

    rng = np.random.default_rng(seed + (100 * n_features))
    if tune:
        search_space = [
            {"hidden_layer_sizes": (80, 80, 80), "activation": "logistic", "alpha": 1e-4, "learning_rate_init": 1e-3},
            {"hidden_layer_sizes": (64, 64), "activation": "relu", "alpha": 1e-4, "learning_rate_init": 1e-3},
            {"hidden_layer_sizes": (120, 120), "activation": "relu", "alpha": 1e-3, "learning_rate_init": 5e-4},
            {"hidden_layer_sizes": (150, 80), "activation": "logistic", "alpha": 1e-3, "learning_rate_init": 5e-4},
            {"hidden_layer_sizes": (nn_hidden[0],) if len(nn_hidden) > 0 else (80,), "activation": "relu", "alpha": 1e-4, "learning_rate_init": 1e-3},
        ]
        order = rng.permutation(len(search_space))
        chosen = [search_space[i] for i in order[: max(1, min(tune_trials, len(search_space)))]]
    else:
        chosen = [
            {
                "hidden_layer_sizes": nn_hidden,
                "activation": "logistic",
                "alpha": 1e-4,
                "learning_rate_init": 1e-3,
            }
        ]

    best_cfg = None
    best_val = np.inf
    for i, cfg in enumerate(chosen):
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        nn = MLPRegressor(
            hidden_layer_sizes=cfg["hidden_layer_sizes"],
            activation=cfg["activation"],
            solver="adam",
            alpha=cfg["alpha"],
            max_iter=nn_max_iter,
            random_state=seed + i,
            learning_rate_init=cfg["learning_rate_init"],
            early_stopping=False,
        )
        nn.fit(Xtr_s, ytr)
        val_pred = nn.predict(Xva_s)
        val_mse = mse(yva, val_pred)
        if val_mse < best_val:
            best_val = val_mse
            best_cfg = cfg

    if best_cfg is None:
        return None

    trva = np.concatenate([split.train_idx, split.val_idx])
    Xtrva = X[trva]
    ytrva = y[trva]
    scaler = StandardScaler()
    Xtrva_s = scaler.fit_transform(Xtrva)
    Xte_s = scaler.transform(Xte)
    final_nn = MLPRegressor(
        hidden_layer_sizes=best_cfg["hidden_layer_sizes"],
        activation=best_cfg["activation"],
        solver="adam",
        alpha=best_cfg["alpha"],
        max_iter=nn_max_iter,
        random_state=seed + 999,
        learning_rate_init=best_cfg["learning_rate_init"],
        early_stopping=False,
    )
    final_nn.fit(Xtrva_s, ytrva)
    return final_nn.predict(Xte_s)


def run_xgb(
    df: pd.DataFrame,
    split: Split,
    y: np.ndarray,
    n_features: int,
    seed: int,
    tune: bool,
    tune_trials: int,
) -> Optional[np.ndarray]:
    if not try_import_xgboost():
        return None
    import xgboost as xgb

    X = model_features(df, n_features)
    if len(split.train_idx) < 20 or len(split.test_idx) == 0 or len(split.val_idx) == 0:
        return None

    default_cfg = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }
    if tune:
        search_space = [
            {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
            {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9},
            {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9},
            {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.8},
            {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.8},
            {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.08, "subsample": 1.0, "colsample_bytree": 1.0},
        ]
        rng = np.random.default_rng(seed + (200 * n_features))
        order = rng.permutation(len(search_space))
        chosen = [search_space[i] for i in order[: max(1, min(tune_trials, len(search_space)))]]
    else:
        chosen = [default_cfg]

    Xtr = X[split.train_idx]
    ytr = y[split.train_idx]
    Xva = X[split.val_idx]
    yva = y[split.val_idx]

    best_cfg = None
    best_val = np.inf
    for i, cfg in enumerate(chosen):
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=seed + i,
            n_jobs=4,
            **cfg,
        )
        model.fit(Xtr, ytr)
        val_pred = model.predict(Xva)
        val_mse = mse(yva, val_pred)
        if val_mse < best_val:
            best_val = val_mse
            best_cfg = cfg

    if best_cfg is None:
        return None

    trva = np.concatenate([split.train_idx, split.val_idx])
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=seed + 999,
        n_jobs=4,
        **best_cfg,
    )
    model.fit(X[trva], y[trva])
    return model.predict(X[split.test_idx])


def run_experiment(
    df: pd.DataFrame,
    split: Split,
    target_col: str,
    seed: int,
    nn_max_iter: int,
    nn_hidden: Tuple[int, ...],
    tune: bool,
    tune_trials: int,
):
    y = df[target_col].to_numpy(dtype=float)
    if len(split.train_idx) == 0 or len(split.val_idx) == 0 or len(split.test_idx) == 0:
        raise ValueError("Split produced empty train/val/test set. Increase data size or adjust split ratios.")
    y_test = y[split.test_idx]

    rows: List[Dict[str, float | str]] = []
    preds: Dict[str, np.ndarray] = {"y_test": y_test}

    pred_b, mse_b = run_benchmark(df, split, y)
    preds["Benchmark_HullWhiteLike"] = pred_b
    rows.append(
        {
            "model": "Benchmark_HullWhiteLike",
            "mse": mse_b,
            "gain_vs_benchmark": 0.0,
            "r2": r2(y_test, pred_b),
            "weighted_acc": weighted_accuracy(y_test, pred_b),
        }
    )

    for nf in [3, 4]:
        print(f"Training NN_{nf}F ...")
        pred = run_nn(df, split, y, nf, seed, nn_max_iter, nn_hidden, tune=tune, tune_trials=tune_trials)
        if pred is None:
            continue
        err = mse(y_test, pred)
        preds[f"NN_{nf}F"] = pred
        rows.append(
            {
                "model": f"NN_{nf}F",
                "mse": err,
                "gain_vs_benchmark": 1.0 - (err / mse_b),
                "r2": r2(y_test, pred),
                "weighted_acc": weighted_accuracy(y_test, pred),
            }
        )

    for nf in [2, 3, 4]:
        print(f"Training XGB_{nf}F ...")
        pred = run_xgb(df, split, y, nf, seed, tune=tune, tune_trials=tune_trials)
        if pred is None:
            continue
        err = mse(y_test, pred)
        preds[f"XGB_{nf}F"] = pred
        rows.append(
            {
                "model": f"XGB_{nf}F",
                "mse": err,
                "gain_vs_benchmark": 1.0 - (err / mse_b),
                "r2": r2(y_test, pred),
                "weighted_acc": weighted_accuracy(y_test, pred),
            }
        )

    # Leakage demonstrator (diagnostic only, not part of the core model list).
    tr_df = df.iloc[np.concatenate([split.train_idx, split.val_idx])].copy()
    te_df = df.iloc[split.test_idx].copy()
    key_means = tr_df.groupby(["spy_ret", "vix"], as_index=True)[target_col].mean().to_dict()
    gmean = float(tr_df[target_col].mean())
    pred_mem = np.array([key_means.get((r.spy_ret, r.vix), gmean) for r in te_df.itertuples()], dtype=float)
    err_mem = mse(y_test, pred_mem)
    preds["Memorizer_keyed"] = pred_mem
    rows.append(
        {
            "model": "Memorizer_keyed",
            "mse": err_mem,
            "gain_vs_benchmark": 1.0 - (err_mem / mse_b),
            "r2": r2(y_test, pred_mem),
            "weighted_acc": weighted_accuracy(y_test, pred_mem),
        }
    )

    out = (
        pd.DataFrame(rows)
        .sort_values(
            by=["mse", "gain_vs_benchmark", "weighted_acc"],
            ascending=[True, False, False],
        )
        .reset_index(drop=True)
    )
    return out, preds


def build_comparison_table(tbl_random: pd.DataFrame, tbl_chrono: pd.DataFrame) -> pd.DataFrame:
    r = tbl_random.rename(
        columns={
            "mse": "mse_random",
            "gain_vs_benchmark": "gain_random",
            "r2": "r2_random",
            "weighted_acc": "wacc_random",
        }
    )
    c = tbl_chrono.rename(
        columns={
            "mse": "mse_chrono",
            "gain_vs_benchmark": "gain_chrono",
            "r2": "r2_chrono",
            "weighted_acc": "wacc_chrono",
        }
    )
    z = r.merge(c, on="model", how="inner")
    z["delta_gain_random_minus_chrono"] = z["gain_random"] - z["gain_chrono"]
    z["delta_mse_random_minus_chrono"] = z["mse_random"] - z["mse_chrono"]
    return (
        z.sort_values(
            by=["mse_chrono", "gain_chrono", "wacc_chrono", "mse_random"],
            ascending=[True, False, False, True],
        )
        .reset_index(drop=True)
    )


def print_block(title: str, tbl: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    fmt = tbl.copy()
    for c in fmt.columns:
        if pd.api.types.is_numeric_dtype(fmt[c]):
            fmt[c] = fmt[c].map(lambda x: f"{x:.6f}")
    print(fmt.to_string(index=False))


def plot_metric_comparison(comp: pd.DataFrame, outdir: Path) -> None:
    models = comp["model"].tolist()
    x = np.arange(len(models))
    w = 0.38

    plt.figure(figsize=(12, 5))
    plt.bar(x - w / 2, comp["mse_random"], width=w, label="Random split")
    plt.bar(x + w / 2, comp["mse_chrono"], width=w, label="Chronological split")
    plt.xticks(x, models, rotation=45, ha="right")
    plt.ylabel("MSE")
    plt.title("Comparacion MSE por modelo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fig_mse_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.bar(x - w / 2, comp["gain_random"], width=w, label="Random split")
    plt.bar(x + w / 2, comp["gain_chrono"], width=w, label="Chronological split")
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(x, models, rotation=45, ha="right")
    plt.ylabel("Gain vs benchmark")
    plt.title("Comparacion Gain por modelo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fig_gain_comparison.png", dpi=160)
    plt.close()


def plot_target_distribution(df: pd.DataFrame, target_col: str, outdir: Path) -> None:
    y = df[target_col].to_numpy(dtype=float)
    lo, hi = np.quantile(y, 0.01), np.quantile(y, 0.99)
    yc = np.clip(y, lo, hi)
    plt.figure(figsize=(10, 5))
    plt.hist(yc, bins=80, alpha=0.9)
    plt.title(f"Distribucion de {target_col} (recorte p1-p99)")
    plt.xlabel(target_col)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(outdir / "fig_target_distribution.png", dpi=160)
    plt.close()


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, title: str, outpath: Path, seed: int) -> None:
    n = len(y_true)
    max_points = 5000
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(np.arange(n), size=max_points, replace=False)
        yt, yp = y_true[idx], y_pred[idx]
    else:
        yt, yp = y_true, y_pred
    mn = float(min(np.min(yt), np.min(yp)))
    mx = float(max(np.max(yt), np.max(yp)))
    plt.figure(figsize=(6, 6))
    plt.scatter(yt, yp, s=8, alpha=0.35)
    plt.plot([mn, mx], [mn, mx], color="red", linewidth=1.4)
    plt.xlabel("y real")
    plt.ylabel("y predicho")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def save_same_day_example(df: pd.DataFrame, outdir: Path, n_rows: int = 10) -> None:
    counts = df.groupby("date").size().sort_values(ascending=False)
    if counts.empty:
        return
    d = counts.index[0]
    df[df["date"] == d].head(n_rows).to_csv(outdir / "table_same_day_samples.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="IV forecasting replication on Kaggle-prepared data.")
    ap.add_argument("--csv", type=str, default="", help="Path to prepared dataset CSV.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.20)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--sample-frac", type=float, default=1.0, help="Row subsampling fraction (0,1].")
    ap.add_argument("--nn-max-iter", type=int, default=40, help="Max epochs for NN.")
    ap.add_argument("--tune", action="store_true", help="Tune NN/XGB on validation set, then refit on train+val.")
    ap.add_argument("--tune-trials", type=int, default=5, help="Max candidate configs per model family/feature set.")
    ap.add_argument("--fast", action="store_true", help="Fast run preset for quick iteration.")
    ap.add_argument("--skip-plots", action="store_true", help="Legacy flag. Prefer not using --export-plots.")
    ap.add_argument("--export-tables", action="store_true", help="Export result tables to --outdir.")
    ap.add_argument("--export-plots", action="store_true", help="Export figures to --outdir.")
    ap.add_argument(
        "--target-mode",
        choices=["diff", "logret"],
        default="diff",
        help="Target column to train: diff (default) or logret (needs iv+option_id in CSV).",
    )
    ap.add_argument("--outdir", type=str, default="")
    args = ap.parse_args()
    base_dir = Path(__file__).resolve().parent
    if not args.outdir:
        args.outdir = str(base_dir / "outputs" / "replicate_kaggle")

    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        raise ValueError("train + val + test must sum to 1.0")

    if args.fast:
        args.nn_max_iter = min(args.nn_max_iter, 12)
        args.sample_frac = min(args.sample_frac, 0.25)
        args.tune_trials = min(args.tune_trials, 2)
        args.skip_plots = True
    if not args.csv:
        args.csv = str(base_dir / "data" / "kaggle" / "prepared" / "options_dataset.csv")

    csv_path = Path(args.csv).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(
            "CSV path not found. Use a prepared Kaggle file, for example:\n"
            "python examples/iv/replicate_iv_paper.py --csv ./examples/iv/data/kaggle/prepared/options_dataset.csv --target-mode diff"
        )
    if csv_path.is_dir():
        raise IsADirectoryError(
            f"--csv expects a file, but got a directory: {csv_path}"
        )
    df = prepare_dataframe_from_tabular(args.csv)
    source = f"File: {csv_path}"

    target_col = "target_diff" if args.target_mode == "diff" else "target_logret"
    if target_col not in df.columns:
        raise ValueError(
            f"{target_col} not available. For --target-mode logret provide CSV with iv+option_id."
        )
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(
            "Input dataset has 0 rows after preparation. "
            "Rebuild CSV with prepare_kaggle_optionsdx.py and check its row diagnostics."
        )
    if not (0 < args.sample_frac <= 1.0):
        raise ValueError("--sample-frac must be in (0,1].")
    if args.sample_frac < 1.0:
        df_full = df
        df = df.sample(frac=args.sample_frac, random_state=args.seed).reset_index(drop=True)
        if len(df) < 200:
            df = df_full.reset_index(drop=True)

    print(f"Data source: {source}")
    print(f"Rows after preparation: {len(df)}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Target mode: {args.target_mode}")
    print(f"NN max_iter: {args.nn_max_iter}")
    print(f"Sample frac: {args.sample_frac}")
    print(f"Tuning enabled: {args.tune} (trials={args.tune_trials})")

    has_skl = try_import_sklearn()
    has_xgb = try_import_xgboost()
    if not has_skl:
        print("Warning: scikit-learn not installed. NN models will be skipped.")
    if not has_xgb:
        print("Warning: xgboost not installed. XGB models will be skipped.")

    days = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").to_numpy()
    s_random = random_split(len(df), args.train, args.val, args.seed)
    s_chrono = chrono_day_split(days, args.train, args.val)
    split_date_summary(df, s_random, "RANDOM")
    split_date_summary(df, s_chrono, "CHRONOLOGICAL")

    nn_hidden = (64, 64) if args.fast else (80, 80, 80)
    tbl_r, pred_r = run_experiment(
        df,
        s_random,
        target_col,
        args.seed,
        args.nn_max_iter,
        nn_hidden,
        tune=args.tune,
        tune_trials=args.tune_trials,
    )
    tbl_c, pred_c = run_experiment(
        df,
        s_chrono,
        target_col,
        args.seed,
        args.nn_max_iter,
        nn_hidden,
        tune=args.tune,
        tune_trials=args.tune_trials,
    )
    comp = build_comparison_table(tbl_r, tbl_c)

    print_block("Results with RANDOM row split (leakage-prone)", tbl_r)
    print_block("Results with CHRONOLOGICAL day split (leakage-aware)", tbl_c)
    print_block("Comparison table (RANDOM vs CHRONO)", comp)
    export_plots = args.export_plots and (not args.skip_plots)
    export_tables = args.export_tables
    if export_tables or export_plots:
        outdir = ensure_outdir(args.outdir)
        if export_tables:
            tbl_r.to_csv(outdir / "table_results_random.csv", index=False)
            tbl_c.to_csv(outdir / "table_results_chrono.csv", index=False)
            comp.to_csv(outdir / "table_results_comparison.csv", index=False)
            save_same_day_example(df, outdir)
        if export_plots:
            plot_metric_comparison(comp, outdir)
            plot_target_distribution(df, target_col, outdir)
            plot_scatter(
                pred_r["y_test"],
                pred_r["Benchmark_HullWhiteLike"],
                "Benchmark: y real vs y predicho (random split)",
                outdir / "fig_benchmark_scatter_random.png",
                args.seed,
            )
            plot_scatter(
                pred_c["y_test"],
                pred_c["Benchmark_HullWhiteLike"],
                "Benchmark: y real vs y predicho (chronological split)",
                outdir / "fig_benchmark_scatter_chrono.png",
                args.seed,
            )
            if "Memorizer_keyed" in pred_r and "Memorizer_keyed" in pred_c:
                plot_scatter(
                    pred_r["y_test"],
                    pred_r["Memorizer_keyed"],
                    "Memorizer: y real vs y predicho (random split)",
                    outdir / "fig_memorizer_scatter_random.png",
                    args.seed,
                )
                plot_scatter(
                    pred_c["y_test"],
                    pred_c["Memorizer_keyed"],
                    "Memorizer: y real vs y predicho (chronological split)",
                    outdir / "fig_memorizer_scatter_chrono.png",
                    args.seed,
                )

    print("\nInterpretation:")
    print("- Random row split can inflate out-of-sample metrics in panel time-series.")
    print("- Use chronological day-grouped split for realistic evaluation.")
    if export_tables or export_plots:
        print(f"\nArtifacts saved in: {args.outdir}")
    else:
        print("\nNo artifacts exported. Use --export-tables and/or --export-plots.")


if __name__ == "__main__":
    main()

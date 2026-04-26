# Example 1: Implied Volatility (IV) - Kaggle Only

This folder contains the IV pipeline using **Kaggle SPY options data only**.

## macOS prerequisite (XGBoost)

If importing `xgboost` fails with `libxgboost.dylib` / `libomp` errors:

```bash
brew install libomp
```

## Scripts

- `fetch_kaggle_spy_options.py`: optional KaggleHub downloader
- `prepare_kaggle_optionsdx.py`: convert raw SPY option chains into prepared CSV
- `run_paper_like.py`: one-command Kaggle pipeline (prepare + replicate)
- `replicate_iv_paper.py`: run random vs chronological experiments from prepared CSV
- `mean_reversion_iv.py`: coverage-aware mean-reversion results
- `eda.py`: EDA plots/tables for raw + prepared data
- `IV_leakage_analysis.ipynb`: exploratory notebook (prototyping / checks)

## Data layout

- `examples/iv/data/kaggle/raw/` -> raw Kaggle option chain files
- `examples/iv/data/kaggle/prepared/` -> prepared model CSV

## One-command run (prepare + replicate)

```bash
python examples/iv/run_paper_like.py \
  --input-glob './examples/iv/data/kaggle/raw/spy_raw_chain_*.parquet' \
  --output-csv ./examples/iv/data/kaggle/prepared/options_dataset.csv \
  --outdir examples/iv/outputs/kaggle_run \
  --target-mode diff \
  --tune --tune-trials 6 \
  --seed 2025 \
  --export-tables --export-plots
```

## Replicate directly from prepared CSV

```bash
python examples/iv/replicate_iv_paper.py \
  --csv ./examples/iv/data/kaggle/prepared/options_dataset.csv \
  --target-mode diff \
  --tune --tune-trials 10 \
  --seed 2025 \
  --export-tables \
  --outdir examples/iv/outputs/replicate_kaggle_seed2025
```

Article reproducibility note:
- Use exactly the command above to reproduce the article run.
- Keep the same prepared CSV file and seed (`2025`) to get the same table outputs.

## Mean reversion (scripted outputs)

```bash
python examples/iv/mean_reversion_iv.py \
  --csv ./examples/iv/data/kaggle/prepared/options_dataset.csv \
  --window 20 \
  --min-periods 10 \
  --min-obs-per-option 60 \
  --outdir examples/iv/outputs/mean_reversion_kaggle
```

Outputs:
- `table_option_coverage_stats.csv`
- `table_mean_reversion_betas.csv`
- `table_decile_profile.csv`
- `fig_mean_reversion_profile.png`
- `summary_mean_reversion.txt`

## EDA

```bash
python examples/iv/eda.py \
  --input-glob './examples/iv/data/kaggle/raw/spy_raw_chain_*.parquet' \
  --prepared-csv ./examples/iv/data/kaggle/prepared/options_dataset.csv \
  --outdir examples/iv/outputs/iv_eda
```

## Notebook note

`IV_leakage_analysis.ipynb` is exploratory and may contain intermediate experiments.
For reproducible article outputs, use scripts (`replicate_iv_paper.py`, `mean_reversion_iv.py`, `eda.py`).

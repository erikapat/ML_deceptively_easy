# ML_deceptively_easy

This repository is currently focused on one example:

- `examples/iv` -> Implied Volatility (IV) with **Kaggle SPY options data**

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS prerequisite for XGBoost

If `xgboost` fails with `libxgboost.dylib could not be loaded`:

```bash
brew install libomp
```

## Quick start (Kaggle IV)

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

## Core scripts

- `examples/iv/prepare_kaggle_optionsdx.py` -> raw Kaggle files to prepared CSV
- `examples/iv/replicate_iv_paper.py` -> model evaluation (random vs chronological)
- `examples/iv/mean_reversion_iv.py` -> coverage-aware mean-reversion analysis
- `examples/iv/eda.py` -> EDA outputs
- `examples/iv/IV_leakage_analysis.ipynb` -> exploratory notebook

## Article run (fixed seed)

```bash
python examples/iv/replicate_iv_paper.py \
--csv ./examples/iv/data/kaggle/prepared/options_dataset.csv \
--target-mode diff \
--tune --tune-trials 10 \
--seed 2025 \
--export-tables \
--outdir examples/iv/outputs/replicate_kaggle_seed2025
```

See `examples/iv/README.md` for full commands and outputs.

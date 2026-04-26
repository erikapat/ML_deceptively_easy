# Example 2: Capital Gain (Track A, Real Data)

Status: implemented (minimum pipeline).

## Data source (real public data)

This example uses **real transaction data** from:

- UK HM Land Registry, Price Paid Data (PPD)
- Official source: https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads

Raw CSV files (e.g., `pp-2019.csv`, `pp-2020.csv`) should be placed in:

- `examples/capital_gain/raw/`

## Scripts

- `prepare_capital_gain_data.py`: builds panel dataset + lag features + targets
- `replicate_capital_gain.py`: compares random split vs chronological split (leakage check)
- `run_capital_gain_like.py`: one-command pipeline
- `capital_gain_analysis.ipynb`: EDA + quick leakage check

## Prepare dataset

```bash
python examples/capital_gain/prepare_capital_gain_data.py \
  --input-glob 'examples/capital_gain/raw/pp-*.csv' \
  --output examples/capital_gain/data/capital_gain_dataset.csv \
  --start-date 2010-01-01 \
  --end-date 2023-12-31
```

## Run replication

```bash
python examples/capital_gain/replicate_capital_gain.py \
  --csv examples/capital_gain/data/capital_gain_dataset.csv \
  --target target_logret
```

## One-command run

```bash
python examples/capital_gain/run_capital_gain_like.py \
  --input-glob 'examples/capital_gain/raw/pp-*.csv' \
  --output-csv examples/capital_gain/data/capital_gain_dataset.csv \
  --target target_logret
```

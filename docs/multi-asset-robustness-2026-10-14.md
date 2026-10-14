# Multi-asset covariance robustness, 14 October 2026

This is a descriptive research run, not a significance test or trading recommendation.

## Frozen input

- Universe: SPY, EFA, EEM, IEF, LQD, GLD, VNQ.
- Source: yfinance adjusted `Close`, requested from 2018-01-01 through 2025-01-01 (exclusive).
- Local snapshot: `data/snapshots/multi_asset_2018-2024_2026-10-14.csv` and its `.csv.json` manifest. The CSV SHA-256 is `d87df74b9c463a5e7a82ebe66634b9025578f8451c372f80390f387fc9ef9d10`.
- The manifest records the exact download time, installed package versions, ticker order, actual first/last dates, row count, adjustment setting, and checksum. The runner verifies the checksum before using prices.
- The repository ignores `data/` by design. Keep both snapshot files together when moving the project; the results CSV alone does not contain the prices.

## Controlled grid

The [24-run results](../results/tables/multi_asset_covariance_grid_2026-10-14.csv) and [run manifest](../results/tables/multi_asset_covariance_grid_2026-10-14.csv.json) cover four training windows (126, 252, 504, 756 observations), three rebalance intervals (5, 21, 63 observations), and sample versus Ledoit–Wolf covariance. Historical expected returns, a 30% weight cap, and 10 bps per half-L1 turnover are held fixed.

All runs use the same 1,004 daily return observations from **2021-01-05 through 2024-12-31**. The start accommodates the largest training window. It is later than the earlier 504-window comparison's start, so its CAGR is not directly comparable with the prior 5.5% result.

Ledoit–Wolf has a higher CAGR in 11 of the 12 matched settings, but the paired CAGR differences range only from **−0.036 to +0.145 percentage points** per year. Across settings, the impact of training window and rebalance interval is much larger than the covariance choice. This is a descriptive pattern from one selected universe and market period; it does not establish superiority or statistical significance. The grid itself creates multiple-comparison risk.

The original ETF choices are not point-in-time membership data. The backtester uses adjusted closes, zero risk-free rate, no initial allocation fee, and an additive transaction-cost approximation. Provider data could later change, but this snapshot preserves the inputs for this run.

## Reproduce without downloading

From the repository root, after installing the project:

```bash
python -m research.robustness run \
  --snapshot data/snapshots/multi_asset_2018-2024_2026-10-14.csv \
  --output results/tables/multi_asset_covariance_grid_recheck.csv
```

The command refuses to overwrite an existing output. Compare the new CSV with the saved results; the snapshot's checksum is checked before computation. To capture a *new* provider version, use `python -m research.robustness capture --output data/snapshots/NEW_NAME.csv`; that is a new dataset, not a reproduction of this run.

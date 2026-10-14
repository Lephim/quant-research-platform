# Multi-asset holdout results, 14 October 2026

This run followed the [plan saved before the holdout download](multi-asset-holdout-plan-2026-10-14.md). The selected 756-observation window and 63-observation rebalance interval came from the earlier 2018–2024 exploratory grid. No setting was changed after viewing 2025-onward prices.

## Data and evaluation

The [local price snapshot](../data/snapshots/multi_asset_2018-2026q3_2026-10-14.csv) and [manifest](../data/snapshots/multi_asset_2018-2026q3_2026-10-14.csv.json) contain adjusted closes for SPY, EFA, EEM, IEF, LQD, GLD, and VNQ, requested through 2026-09-30. Its SHA-256 is `96670e1733d98bbb49b5bce60827c3e4ff20deddc9e0af12d751f878fe78336f`. Only the 437 daily returns from **2025-01-02 through 2026-09-30** count toward holdout performance; earlier rows provide rolling training history.

All three strategies use 63-observation rebalancing and 10 bps per half-L1 turnover. The two optimized strategies use historical expected returns, a 756-observation training window, and a 30% asset cap. Equal weight has neither a covariance estimator nor that cap. The source code checks that daily return dates match exactly.

| Strategy | CAGR | Volatility | Sharpe | Max drawdown | Total return | Annualized turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ledoit–Wolf | 17.565% | 11.807% | 1.429920 | −9.199% | 32.394% | 0.500 |
| Sample covariance | 17.476% | 11.742% | 1.430765 | −9.169% | 32.222% | 0.442 |
| Equal weight | 16.109% | 11.342% | 1.373720 | −8.610% | 29.564% | 0.078 |

The prespecified primary difference, **Ledoit–Wolf minus sample Sharpe, is −0.000846**. Ledoit–Wolf has a 0.088 percentage point higher CAGR, but slightly higher volatility, turnover, and drawdown. Both optimized variants have higher CAGR and Sharpe than equal weight in this period, while equal weight has lower volatility and drawdown. The holdout does not confirm a meaningful Sharpe advantage for Ledoit–Wolf.

These are historical descriptive figures from one selected universe and one holdout period. They are not a significance test or a forecast. The [summary CSV](../results/tables/multi_asset_holdout_2025-2026q3.csv), [daily returns](../results/tables/multi_asset_holdout_2025-2026q3.daily.csv), and [provenance manifest](../results/tables/multi_asset_holdout_2025-2026q3.csv.json) preserve the exact output. The `data/` directory is ignored by Git, so retain the snapshot CSV and JSON together when moving the project.

## Reproduce without another download

```bash
python -m research.holdout \
  --snapshot data/snapshots/multi_asset_2018-2026q3_2026-10-14.csv \
  --output results/tables/multi_asset_holdout_recheck.csv
```

The command verifies the snapshot checksum and refuses to overwrite existing results.

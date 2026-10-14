# Multi-asset holdout plan, fixed before downloading holdout prices

Date fixed: 14 October 2026. This plan uses only the saved 2018–2024 exploratory [grid](../results/tables/multi_asset_covariance_grid_2026-10-14.csv) to choose settings. The holdout results must be reported regardless of outcome; no parameter changes will be made after viewing them.

- Universe: SPY, EFA, EEM, IEF, LQD, GLD, VNQ, in that order.
- New adjusted-close price snapshot requested from 2018-01-01 through 2026-10-01 (exclusive), with its own checksum and provenance. The 2018–2024 portion supplies training history; only 2025-onward returns count toward holdout performance.
- Holdout starts on the first available trading observation on or after 2025-01-01 and ends at the last available observation before 2026-10-01.
- Selected configuration: 756 daily-return observations for training, 63-observation rebalance interval, historical expected returns, long-only 30% cap, zero risk-free rate, and 10 basis points per half-L1 turnover. This was the highest-Sharpe setting in the exploratory grid; selection is therefore subject to 2018–2024 data mining.
- Compare (1) Ledoit–Wolf covariance, (2) sample covariance, and (3) equal-weight rebalancing, all on identical holdout dates and at the same frequency and cost assumption. Equal weight has no covariance estimator or weight cap.
- Primary descriptive comparison: Ledoit–Wolf minus sample Sharpe on the holdout. Also report CAGR, volatility, maximum drawdown, total return, and turnover for all three. Compare the optimized strategies with equal weight as a benchmark.
- No statistical-significance claim is planned from this single historical holdout. No choice will be retuned to the holdout outcome.

The test can assess whether the exploratory ranking persists in a later period; it cannot establish a deployable strategy. Provider revisions, the chosen ETF universe, and the backtest execution model remain limitations.

# Methodology

This document describes the current source implementation. Historical notebook outputs can predate it; they have not been recomputed in this cleanup.

## Inputs and estimation

Backtests accept daily **simple returns** in a pandas DataFrame with unique asset columns and a sorted, unique DatetimeIndex. Rows missing any asset return are dropped (complete-case alignment); infinite values and returns at or below -100% are rejected. The implementation does not model asset bankruptcy/delisting. The 252-observation annualization convention assumes a trading-day calendar; mixed crypto/equity calendars require an explicit alignment policy.

Simple returns are `price[t] / price[t-1] - 1`, without implicit forward fill. Log returns are available for exploration but must not be passed to the simple-return backtester.

Historical mean returns and sample covariance are multiplied by 252. Sample covariance uses pandas sample covariance; the walk-forward engine supplies complete rows. Ledoit–Wolf fits scikit-learn's estimator to complete daily returns, retains asset labels, and annualizes its covariance. The standalone sample estimator may use pairwise observations when given missing data directly.

Expected-return shrinkage estimates each mean's standard error, takes the cross-asset mean as the target, and estimates genuine dispersion as `max(variance(asset means) - mean(standard errors squared), 0)`. Asset-specific trust weights are `tau² / (tau² + standard error²)`. Zero denominators use the common target. This is a heuristic empirical-Bayes-style estimator; it does not fully model cross-asset dependence or parameter uncertainty.

## Portfolio construction

SciPy SLSQP solves for fully invested, long-only weights with an asset cap (default 30%). The package offers maximum Sharpe, minimum volatility, and target-return optimization; the backtester uses maximum Sharpe with zero risk-free rate. Standalone optimization returns SciPy's result object: callers must check `success` and respect consistent asset ordering across mean, covariance, and weights. The walk-forward engine raises on solver failure instead of silently omitting a period.

The exploratory Monte Carlo sampler normalizes independent uniform random draws; it is **not** a uniform simplex sampler. It rejects allocations above the cap and has a bounded attempt limit. Historical notebook sampling was not seeded.

## Timeline and benchmark

At test position `t`, training uses exactly `[t - train_window, t)`. The target weights are applied to `[t, t + rebalance_frequency)`, with a shorter final block retained. Evaluation starts after the training window unless `evaluation_start` is supplied. An explicit start is advanced to the first available observation on or after that date and requires enough prior training rows. Both optimizer and equal-weight benchmark accept the same start.

No test-period returns enter estimation. However, the model assumes weights can be established at the boundary before the first test return without an additional execution lag. Close-to-close data and real order timing may require a lag and slippage model before any trading interpretation.

Training-window comparisons must use a common start that accommodates the largest window. Frequencies 5, 21, and 63 mean **observation intervals**, not calendar-week/month/quarter schedules.

Within a block, asset holdings are held fixed and weights drift. Portfolio wealth is the weighted sum of each asset's cumulative gross return. The benchmark rebalances to equal weights on the same schedule rather than implicitly rebalancing daily.

## Turnover and transaction costs

Turnover at a later rebalance is half the L1 distance between new target weights and previous end-of-block drifted weights. Initial allocation incurs neither recorded turnover nor cost. Consequently `len(turnover) = len(weights_history) - 1`.

Cost is `turnover * transaction_cost_bps / 10000`, subtracted from the first daily portfolio return of the new block. This is the existing additive approximation with cost per unit of **half-L1 turnover**, not an independently charged cost on every buy and sell. It does not explicitly finance fees from holdings, alter the drift calculation, model spreads/impact/taxes, or charge terminal liquidation. Annualized turnover approximated by `mean(turnover) * 252 / frequency` excludes initial deployment and can differ from a realized calendar-year total.

## Performance metrics

- Total return: product of `(1 + daily return)` minus one.
- CAGR: gross total return raised to `252 / number of observations`, minus one.
- Volatility: sample daily standard deviation times square root of 252.
- Sharpe: mean daily excess return divided by its sample standard deviation, times square root of 252. Annual risk-free rate is divided by 252; zero is the default. Degenerate/one-observation samples can yield undefined volatility or Sharpe.
- Maximum drawdown: minimum wealth relative to its running peak, **including starting wealth of 1**. This corrects the first-loss omission in the previous performance-metrics implementation. The standalone price-based `maximum_drawdown` uses its first supplied price as its starting point.

## Limits of evidence

The five stocks were selected retrospectively. Performance cannot establish generalizable alpha, and repeated parameter exploration introduces selection risk. No confidence intervals, point-in-time membership data, delisting model, capacity analysis, or independent holdout study is currently provided. A diversified ETF universe can test allocation behavior but still requires a defensible selection policy; it does not automatically remove selection bias.

The yfinance loader passes dates/universe to the provider with `auto_adjust=True`. Earlier saved notebook results lack an immutable price snapshot. The new multi-asset run has a checksum-verified local snapshot; see [its research note](multi-asset-robustness-2026-10-14.md). Provider changes, notebook execution order, earlier API versions, and the correctness fixes above can change other rerun results. These limitations must accompany the historical numbers.

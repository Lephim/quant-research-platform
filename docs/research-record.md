# Historical research record

Source: saved outputs of `notebooks/02_analytics_engine.ipynb` as found before v0.1 cleanup. Cell indices below are zero-based. These extracts are copied verbatim, not recreated from market data. The notebook has evolved through multiple implementations and is not a frozen reproducible run.

## Estimator comparison (cell 64)

[Saved table](../results/tables/notebook02-cell64.txt) compares historical mean/sample covariance, historical mean/Ledoit–Wolf, shrunk mean/Ledoit–Wolf, and equal weight. Surrounding calls use a 252-observation window, 21-observation rebalancing, and a 30% optimizer cap, with no explicit cost argument. Earlier date diagnostics report 2019-01-04 through 2024-12-31 for the walk-forward series.

The saved CAGR values are approximately 40.96%, 41.18%, 36.73%, and 41.10%, respectively. Mean shrinkage reduces saved volatility from about 31.3% to 28.1%; its saved maximum drawdown is about -32.1% versus -47.1% for historical mean/Ledoit–Wolf. These unusually high historical returns must be read alongside the retrospectively selected five-stock universe and lack of an immutable data snapshot. They are not forecasts or verified current-engine output.

## Trading costs (cells 84 and 88)

The [0/10 bps comparison](../results/tables/notebook02-cell84.txt) and [0–50 bps sweep](../results/tables/notebook02-cell88.txt) show lower saved CAGR as modeled costs rise. They use the half-L1 cost convention described in methodology. They do not estimate real execution costs.

## Training windows and frequency

Cell 90 saves a 126/252/504/756-observation sweep with different evaluation starts; it is not evidence of a preferred training window. Cell 91 saves `2021-01-05` as the common start and cell 92 defines the aligned comparison, but cell 92 has no saved results. The referenced ChatGPT conversation supplies later aligned-window and frequency results; these have not been independently reproduced or exported as repository results here.

## Before publishing a research claim

Rerun a clean workflow with frozen input data, matching benchmark dates, explicit download settings, and the current corrected metrics. Record any difference from these historical tables rather than replacing their provenance. The diversified ETF experiment remains future work.

# Research notebook guide

Original notebooks and their saved outputs are preserved. Install the project and notebook extras from the repository root, then run `python -m jupyterlab` using that environment.

| Notebook | Reading order and contents |
| --- | --- |
| `01_market_data_exploration.ipynb` | Initial yfinance inspection, plotting, mixed-asset calendars, simple/log returns |
| `02_analytics_engine.ipynb` | Portfolio moments, Monte Carlo, constrained optimization, covariance/mean shrinkage, walk-forward comparisons, drift, turnover, costs, and training-window comparisons |

These are an exploratory record, not verified restart-and-run-all reports. In notebook 01, `data.columns()` is invalid and an early Plotly call precedes flattening of multi-level columns. Normalizing mixed-calendar prices by the first row can produce missing series. In notebook 02, older backtest cells unpack two values while the current functions return three. An early allocation display uses the manually ordered ticker list rather than the return-column order, so its displayed asset-to-weight mapping is unreliable. Later cells reflect API changes and must be read in that context.

Notebook 02's original training-window sweep compares different periods; only the final common-start setup addresses that confound. Its last comparison has code but no saved output. The aligned-window and frequency tables supplied in the linked conversation are not claimed as outputs saved in this checkout.

Use `examples/synthetic_backtest.py` for a clean executable example. A future notebook split should create new reproducible reports and retain these originals as an archive after review. Do not clear historical outputs or overwrite them with new results without preserving their provenance.

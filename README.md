# Quant Research Platform

A Python research framework for studying how estimation uncertainty affects constrained portfolio optimization out of sample. It combines sample and Ledoit–Wolf covariance estimation, expected-return shrinkage, rolling walk-forward evaluation, portfolio drift, turnover, transaction costs, and an equal-weight benchmark.

**v0.1 scope:** a research engine with offline regression tests and preserved exploratory notebooks. The diversified ETF experiment, agentic research layer, and agent evaluation are planned work, not implemented capabilities.

## Research question

How sensitive are portfolio allocations and out-of-sample performance to estimation choices, trading assumptions, and asset-universe selection?

The initial portfolio study uses AAPL, AMZN, GOOG, MSFT, and NVDA over 2018–2024. These stocks were selected retrospectively. The study therefore has selection bias and a survivorship component; its returns are not evidence of an investable historical strategy.

### What the saved research shows

The historical estimator comparison in notebook 02 reports only a small change between sample and Ledoit–Wolf covariance, while mean shrinkage lowers both volatility and CAGR. This is a descriptive observation from saved notebook output, **not a freshly reproduced result or evidence of statistical significance**. See the [research record](docs/research-record.md) for provenance and interpretation limits. Later robustness work fixes a common evaluation start to avoid comparing different market periods.

## Quick start

Use Python 3.11 or newer (local validation uses Python 3.12). From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,notebooks]'
python -m pytest
python examples/synthetic_backtest.py
```

The example uses seeded synthetic returns, runs without market-data access, and demonstrates matching optimizer and benchmark dates. Its printed numbers are not research findings. Notebook downloads require network access.

The checked-in `.python-version` names the author's local `quant-platform` pyenv environment. If pyenv reports that it is missing, choose an installed interpreter, e.g. `PYENV_VERSION=3.12.11 python -m venv .venv` when that version is installed.

`pyproject.toml` declares runtime dependencies, including scikit-learn, and optional development/notebook dependencies. The existing `requirements.txt` is retained as a historical environment snapshot; it is not a complete portable installation specification. The install command above is the supported setup. Existing imports (`analytics`, `backtesting`, `data`) are preserved; use an isolated environment to avoid collisions with similarly named packages.

## Repository map

| Location | Purpose |
| --- | --- |
| `src/analytics/` | Returns, risk, portfolio statistics, estimators, constrained optimization |
| `src/backtesting/` | Rolling optimization, drift-aware benchmark, performance metrics |
| `src/data/` | Existing yfinance loader for one ticker or a supplied universe |
| `src/visualization/` | Reserved visualization package; charts currently live in notebooks |
| `notebooks/` | Original research, saved outputs, and a reading guide |
| `examples/` | Runnable offline API example |
| `tests/` | Deterministic numerical and backtest regression tests |
| `results/` | Historical output extracts and provenance rules |
| `docs/` | Methodology, research record, roadmap, and release notes |

Start with [methodology](docs/methodology.md), then the [notebook guide](notebooks/README.md). Both original notebooks are preserved byte-for-byte during this cleanup; some older cells use previous APIs and are not a clean restart-and-run-all workflow.

## Validation and reproducibility

Tests cover return calculations, covariance scaling, shrinkage, weight constraints, buy-and-hold drift, turnover/cost accounting, training-only estimation, common evaluation dates, optimizer failure, and initial-wealth drawdown. CI runs tests and the synthetic example on Python 3.11 and 3.12; remote CI status is not claimed until it runs.

Market datasets are not committed. The loader currently inherits yfinance defaults, and downloads can change as provider data is revised. Preserve explicit adjustment settings, download time, versions, and a data checksum in future empirical runs. See [methodology](docs/methodology.md) for the execution and cost-model limitations.

## Next milestones

1. Create a clean reproducible research entry point with cached, versioned inputs and explicit download settings.
2. Complete and export training-window and frequency comparisons over identical evaluation dates.
3. Add a diversified ETF experiment with documented universe selection and inception dates.
4. Publish a research report, then build the agentic research layer and evaluate numerical faithfulness, tool choice, unsupported claims, cost, and latency.

See the [v0.1 review plan](docs/v0.1-review.md). A license has not been selected; no open-source license is implied.

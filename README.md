# Quant Research Platform

A Python research framework for studying how estimation uncertainty affects constrained portfolio optimization out of sample. It combines sample and Ledoit–Wolf covariance estimation, expected-return shrinkage, rolling walk-forward evaluation, portfolio drift, turnover, transaction costs, and an equal-weight benchmark.

**v0.1 scope:** a research engine with offline regression tests and preserved exploratory notebooks. The local Ollama agent uses checksum-verified price snapshots; its evaluated limitations are documented below.

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

Market datasets are not committed. The loader explicitly requests adjusted prices, but downloads can change as provider data is revised. The new local multi-asset snapshot records adjustment settings, download time, versions, and a checksum. See [methodology](docs/methodology.md) for the execution and cost-model limitations.

## Next milestones

1. Extend the local snapshot workflow to other research universes and document selection and inception dates.
2. Add uncertainty analysis and an independent holdout before claiming estimator superiority.
3. Evaluate the Ollama agent's tool choice, numerical faithfulness, unsupported claims, cost, and latency.

See the [v0.1 review plan](docs/v0.1-review.md). A license has not been selected; no open-source license is implied.

## Local Ollama research agent

The agent in `src/agent` calls the existing walk-forward engine using a selected,
checksum-verified price snapshot. Every tool call in one session uses the same
return series, and the output names the snapshot and its SHA-256. It does not
fetch fresh market data. From the repository root, with Ollama running and a
locally installed tool-capable model:

```bash
python -m pip install -e .
python -m agent.runner --snapshot data/snapshots/multi_asset_2018-2024_2026-10-14.csv --model qwen3:8b "Compare sample and Ledoit-Wolf covariance in the multi_asset universe with historical means, 504 training observations, 63-observation rebalancing, 30% max weight, and 10 bps cost."
```

Set `RESEARCH_SNAPSHOT` to use a snapshot without repeating `--snapshot`,
`OLLAMA_MODEL` to choose the model, and `OLLAMA_HOST` to change its local endpoint.
The snapshot CSV and its `.csv.json` manifest must remain together. A request
for an incompatible universe stops before any backtest runs. Each command starts
a fresh agent session, so it cannot refer to earlier terminal results as
“the same configurations.”

For a controlled covariance pair without model-chosen parameters, run:

```bash
python -m agent.runner --snapshot data/snapshots/multi_asset_2018-2024_2026-10-14.csv --compare-covariance --universe multi_asset --expected-return-method historical --train-window 504 --rebalance-frequency 63 --max-weight 0.3 --transaction-cost-bps 10
```

The [held-out agent evaluation](docs/ollama-agent-snapshot-blind-results-2026-10-14.md)
records both correct calls and incomplete comparisons. Inspect all displayed
settings and run counts before drawing research conclusions.
The runner checks explicitly requested comparison arms against the completed
runs. For one comparison dimension, or an explicit request for all combinations,
it fills missing arms using the same verified snapshot and fixed settings. It
reports any remaining missing, unexpected, or duplicate runs without displaying
a partial comparison. See the [post-fix live check](docs/ollama-agent-post-fix-results-2026-10-14.md)
for observed behavior and limits.

## Reproducible multi-asset robustness run

The [14 October 2026 research note](docs/multi-asset-robustness-2026-10-14.md)
records a checksum-verified local ETF price snapshot and a 24-run covariance
grid on one common evaluation period. The snapshot is in the ignored `data/`
directory; keep its CSV and JSON manifest together. The grid results and
provenance are in `results/tables/`.

To repeat the grid without another download:

```bash
python -m research.robustness run --snapshot data/snapshots/multi_asset_2018-2024_2026-10-14.csv --output results/tables/multi_asset_covariance_grid_recheck.csv
```

Use a fresh output filename because the runner refuses to overwrite results.

## 2025-onward holdout and agent check

The [fixed holdout plan](docs/multi-asset-holdout-plan-2026-10-14.md) was saved
before downloading 2025-onward prices. The [results](docs/multi-asset-holdout-results-2026-10-14.md)
compare Ledoit–Wolf, sample covariance, and equal weight on matching dates.
The [local Ollama evaluation](docs/ollama-agent-evaluation-2026-10-14.md)
records two offline tool-choice cases. These reports are descriptive and do not
claim statistical significance or general model reliability.

The [expanded Ollama evaluation](docs/ollama-agent-expanded-evaluation-results-2026-10-14.md)
records eight fixed tool-choice cases, the initial failures, and the guarded
rerun. The guards prevent the tested incomplete and unsupported requests from
executing a backtest.

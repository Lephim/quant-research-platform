"""Fixed, out-of-sample multi-asset comparison on a verified price snapshot."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from analytics.returns import calculate_simple_returns
from research.experiments import run_equal_weight_experiment, run_strategy_experiment
from research.snapshot import load_snapshot

HOLDOUT_START = "2025-01-01"
TRAIN_WINDOW = 756
REBALANCE_FREQUENCY = 63
MAX_WEIGHT = 0.3
TRANSACTION_COST_BPS = 10.0


def run_holdout(prices):
    """Run the prespecified three strategies on identical 2025-onward dates."""
    returns = calculate_simple_returns(prices).dropna()
    start = pd.Timestamp(HOLDOUT_START)
    position = int(returns.index.searchsorted(start))
    if position < TRAIN_WINDOW:
        raise ValueError("Insufficient pre-holdout history for the fixed training window")
    if position >= len(returns):
        raise ValueError("Snapshot has no holdout returns")
    experiments = {
        "ledoit_wolf": run_strategy_experiment(
            returns, train_window=TRAIN_WINDOW,
            rebalance_frequency=REBALANCE_FREQUENCY,
            max_weight=MAX_WEIGHT, covariance_method="ledoit_wolf",
            expected_return_method="historical",
            transaction_cost_bps=TRANSACTION_COST_BPS,
            evaluation_start=start,
        ),
        "sample": run_strategy_experiment(
            returns, train_window=TRAIN_WINDOW,
            rebalance_frequency=REBALANCE_FREQUENCY,
            max_weight=MAX_WEIGHT, covariance_method="sample",
            expected_return_method="historical",
            transaction_cost_bps=TRANSACTION_COST_BPS,
            evaluation_start=start,
        ),
        "equal_weight": run_equal_weight_experiment(
            returns, train_window=TRAIN_WINDOW,
            rebalance_frequency=REBALANCE_FREQUENCY,
            transaction_cost_bps=TRANSACTION_COST_BPS,
            evaluation_start=start,
        ),
    }
    common = next(iter(experiments.values())).returns.index
    if common[0] < start or not all(
        experiment.returns.index.equals(common)
        for experiment in experiments.values()
    ):
        raise RuntimeError("Holdout strategies have different evaluation dates")
    daily = pd.DataFrame({name: result.returns for name, result in experiments.items()})
    summary = pd.DataFrame([
        {
            "strategy": name,
            "train_window": TRAIN_WINDOW,
            "rebalance_frequency": REBALANCE_FREQUENCY,
            "max_weight": MAX_WEIGHT if name != "equal_weight" else None,
            "transaction_cost_bps": TRANSACTION_COST_BPS,
            "evaluation_start": common[0].date().isoformat(),
            "evaluation_end": common[-1].date().isoformat(),
            "evaluation_days": len(common),
            **result.metrics,
        }
        for name, result in experiments.items()
    ])
    return summary, daily


def save_holdout(path, snapshot_path):
    """Write summary, daily returns, and checksums without overwriting files."""
    path = Path(path)
    daily_path = path.with_name(path.stem + ".daily.csv")
    manifest_path = path.with_suffix(path.suffix + ".json")
    if any(p.exists() for p in (path, daily_path, manifest_path)):
        raise FileExistsError("Holdout output already exists; choose a new filename")
    prices, source = load_snapshot(snapshot_path)
    if source["universe"] != "multi_asset":
        raise ValueError("Holdout requires the prespecified multi_asset universe")
    summary, daily = run_holdout(prices)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False, float_format="%.17g")
    daily.to_csv(daily_path, index_label="Date", float_format="%.17g",
                 date_format="%Y-%m-%d")
    manifest = {
        "schema_version": 1,
        "plan": "docs/multi-asset-holdout-plan-2026-10-14.md",
        "snapshot_path": str(Path(snapshot_path).resolve()),
        "snapshot_sha256": source["csv_sha256"],
        "summary_sha256": sha256(path.read_bytes()).hexdigest(),
        "daily_returns_sha256": sha256(daily_path.read_bytes()).hexdigest(),
        "holdout_start_requested": HOLDOUT_START,
        "evaluation_start": summary["evaluation_start"].iloc[0],
        "evaluation_end": summary["evaluation_end"].iloc[0],
        "evaluation_days": int(summary["evaluation_days"].iloc[0]),
        "train_window": TRAIN_WINDOW,
        "rebalance_frequency": REBALANCE_FREQUENCY,
        "max_weight": MAX_WEIGHT,
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "expected_return_method": "historical",
        "risk_free_rate": 0.0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return summary, daily_path, manifest_path


def main():
    parser = argparse.ArgumentParser(description="Run the fixed 2025-onward ETF holdout")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary, daily_path, manifest_path = save_holdout(args.output, args.snapshot)
    print(summary[["strategy", "CAGR", "Volatility", "Sharpe", "Max Drawdown",
                   "Total Return"]].to_string(index=False))
    print(f"Saved daily returns to {daily_path} and provenance to {manifest_path}")


if __name__ == "__main__":
    main()

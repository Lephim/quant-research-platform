"""Controlled estimator grid on a checksum-verified price snapshot."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from analytics.returns import calculate_simple_returns
from research.experiments import run_strategy_experiment
from research.snapshot import capture_snapshot, load_snapshot

DEFAULT_WINDOWS = (126, 252, 504, 756)
DEFAULT_FREQUENCIES = (5, 21, 63)


def run_grid(prices, *, windows=DEFAULT_WINDOWS, frequencies=DEFAULT_FREQUENCIES,
             expected_return_method="historical", max_weight=0.3,
             transaction_cost_bps=10):
    """Compare covariance estimators on identical out-of-sample dates."""
    windows = tuple(windows)
    frequencies = tuple(frequencies)
    if not windows or not frequencies or any(w < 2 for w in windows) or any(f < 1 for f in frequencies):
        raise ValueError("Supply positive windows and frequencies")
    returns = calculate_simple_returns(prices).dropna()
    if len(returns) <= max(windows):
        raise ValueError("Snapshot has insufficient observations for the largest window")
    evaluation_start = returns.index[max(windows)]
    records = []
    common_index = None
    for window in windows:
        for frequency in frequencies:
            for covariance_method in ("sample", "ledoit_wolf"):
                result = run_strategy_experiment(
                    returns=returns, train_window=window,
                    rebalance_frequency=frequency, max_weight=max_weight,
                    covariance_method=covariance_method,
                    expected_return_method=expected_return_method,
                    transaction_cost_bps=transaction_cost_bps,
                    evaluation_start=evaluation_start,
                )
                if common_index is None:
                    common_index = result.returns.index
                elif not result.returns.index.equals(common_index):
                    raise RuntimeError("Grid runs have different evaluation dates")
                records.append({
                    "train_window": window,
                    "rebalance_frequency": frequency,
                    "covariance_method": covariance_method,
                    "expected_return_method": expected_return_method,
                    "max_weight": max_weight,
                    "transaction_cost_bps": transaction_cost_bps,
                    "evaluation_start": common_index[0].date().isoformat(),
                    "evaluation_end": common_index[-1].date().isoformat(),
                    "evaluation_days": len(common_index),
                    **result.metrics,
                })
    return pd.DataFrame.from_records(records)


def save_grid(path, snapshot_path, *, windows=DEFAULT_WINDOWS,
              frequencies=DEFAULT_FREQUENCIES, **kwargs):
    path = Path(path)
    manifest_path = path.with_suffix(path.suffix + ".json")
    if path.exists() or manifest_path.exists():
        raise FileExistsError("Results or manifest already exists; choose a new filename")
    prices, source = load_snapshot(snapshot_path)
    results = run_grid(prices, windows=windows, frequencies=frequencies, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False, float_format="%.17g")
    manifest = {
        "schema_version": 1,
        "snapshot_path": str(Path(snapshot_path).resolve()),
        "snapshot_sha256": source["csv_sha256"],
        "universe": source["universe"],
        "windows": list(windows),
        "frequencies": list(frequencies),
        "expected_return_method": kwargs.get("expected_return_method", "historical"),
        "max_weight": kwargs.get("max_weight", 0.3),
        "transaction_cost_bps": kwargs.get("transaction_cost_bps", 10),
        "evaluation_start": results["evaluation_start"].iloc[0],
        "evaluation_end": results["evaluation_end"].iloc[0],
        "evaluation_days": int(results["evaluation_days"].iloc[0]),
        "runs": len(results),
        "results_csv_sha256": sha256(path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return results, manifest_path


def main():
    parser = argparse.ArgumentParser(description="Capture prices and run a controlled covariance grid")
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture", help="Download and save one adjusted-close snapshot")
    capture.add_argument("--output", required=True)
    capture.add_argument("--universe", default="multi_asset")
    capture.add_argument("--start-date", default="2018-01-01")
    capture.add_argument("--end-date", default="2025-01-01",
                         help="Exclusive end date")
    run = sub.add_parser("run", help="Run the grid on an existing snapshot")
    run.add_argument("--snapshot", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--windows", type=int, nargs="+", default=list(DEFAULT_WINDOWS))
    run.add_argument("--frequencies", type=int, nargs="+", default=list(DEFAULT_FREQUENCIES))
    run.add_argument("--transaction-cost-bps", type=float, default=10)
    args = parser.parse_args()
    if args.command == "capture":
        print(capture_snapshot(args.output, universe=args.universe,
                               start_date=args.start_date, end_date=args.end_date))
    else:
        results, manifest = save_grid(
            args.output, args.snapshot, windows=args.windows,
            frequencies=args.frequencies,
            transaction_cost_bps=args.transaction_cost_bps,
        )
        print(results[["train_window", "rebalance_frequency", "covariance_method",
                       "CAGR", "Volatility", "Sharpe", "Max Drawdown"]].to_string(index=False))
        print(f"Saved {len(results)} runs and provenance to {manifest}")


if __name__ == "__main__":
    main()

"""Evaluate local Ollama tool choice with deterministic offline backtest results."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from agent import runner
from agent.tools import validate_backtest_parameters

COMMON = {
    "universe": "multi_asset",
    "expected_return_method": "historical",
    "train_window": 504,
    "rebalance_frequency": 63,
    "max_weight": 0.3,
    "transaction_cost_bps": 10.0,
}
DEFAULTS = {"covariance_method": "sample", **COMMON}
CASES = [
    {
        "id": "single_explicit",
        "question": (
            "Run one multi_asset backtest with sample covariance, historical "
            "expected returns, a 504-observation training window, rebalancing "
            "every 63 observations, max weight 0.3, and 10 basis points of "
            "transaction cost."
        ),
        "expected": [{**COMMON, "covariance_method": "sample"}],
    },
    {
        "id": "controlled_covariance_pair",
        "question": (
            "Compare sample and Ledoit-Wolf covariance in the multi_asset "
            "universe. Run both backtests with historical expected returns, "
            "a 504-observation training window, rebalancing every 63 "
            "observations, max weight 0.3, and 10 basis points of "
            "transaction cost."
        ),
        "expected": [
            {**COMMON, "covariance_method": "sample"},
            {**COMMON, "covariance_method": "ledoit_wolf"},
        ],
    },
]


def normalize(arguments):
    """Compare effective parameters, including backtest defaults."""
    effective = {**DEFAULTS, **arguments}
    effective["max_weight"] = float(effective["max_weight"])
    effective["transaction_cost_bps"] = float(effective["transaction_cost_bps"])
    return effective


def score_calls(actual, expected):
    encode = lambda items: Counter(json.dumps(normalize(item), sort_keys=True) for item in items)
    return encode(actual) == encode(expected)


def evaluate(model="qwen3:8b", snapshot_path="data/snapshots/multi_asset_2018-2024_2026-10-14.csv"):
    """Run fixed prompts against Ollama; all tool results are canned offline."""
    report = {"model": model, "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
              "cases": []}
    original = runner.run_backtest
    try:
        for case in CASES:
            actual = []
            def fake_backtest(**arguments):
                arguments.pop("returns")
                actual.append(arguments.copy())
                effective = normalize(arguments)
                validate_backtest_parameters(**effective)
                sharpe = .52 if effective["covariance_method"] == "ledoit_wolf" else .50
                return {"universe": effective["universe"],
                        "strategy": {key: value for key, value in effective.items()
                                     if key != "universe"},
                        "metrics": {"Sharpe": sharpe, "CAGR": .05}}
            runner.run_backtest = fake_backtest
            started = perf_counter()
            answer = None
            error = None
            try:
                answer = runner.run_research_agent(case["question"], model=model,
                                                   snapshot_path=snapshot_path,
                                                   max_tool_calls=4)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            elapsed = perf_counter() - started
            calls_match = score_calls(actual, case["expected"])
            metrics_grounded = (
                answer is not None and
                all(f"Sharpe: {(.52 if normalize(call)['covariance_method'] == 'ledoit_wolf' else .50):.6g}"
                    in answer for call in actual) and
                "Backtest results (values reported directly by the engine):" in answer
            )
            report["cases"].append({
                "id": case["id"], "question": case["question"],
                "expected_calls": case["expected"], "actual_calls": actual,
                "calls_match": calls_match, "metrics_grounded": metrics_grounded,
                "passed": calls_match and metrics_grounded and error is None,
                "elapsed_seconds": round(elapsed, 3), "error": error,
                "answer": answer,
            })
    finally:
        runner.run_backtest = original
    report["passed"] = sum(case["passed"] for case in report["cases"])
    report["total"] = len(report["cases"])
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ollama tool calls offline")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--output", required=True)
    parser.add_argument("--snapshot", default="data/snapshots/multi_asset_2018-2024_2026-10-14.csv")
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError("Evaluation report exists; choose a new filename")
    report = evaluate(args.model, args.snapshot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Passed {report['passed']}/{report['total']} cases")
    for case in report["cases"]:
        print(f"{case['id']}: calls_match={case['calls_match']} "
              f"metrics_grounded={case['metrics_grounded']} "
              f"seconds={case['elapsed_seconds']}")
        if case["error"]:
            print(case["error"])
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

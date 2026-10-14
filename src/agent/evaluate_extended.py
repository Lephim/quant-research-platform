"""Expanded live Ollama evaluation with deterministic offline tool responses."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from agent import runner
from agent.evaluate import COMMON, normalize, score_calls
from agent.tools import validate_backtest_parameters

PAIR = [
    {**COMMON, "covariance_method": "sample"},
    {**COMMON, "covariance_method": "ledoit_wolf"},
]
PAIR_PROMPT = (
    "Compare sample and Ledoit-Wolf covariance in multi_asset. Use historical "
    "expected returns, 504 training observations, 63-observation rebalancing, "
    "a 0.30 weight cap, and 10 basis points of transaction cost for each run."
)
CASES = [
    {
        "id": "paraphrased_pair",
        "question": (
            "For the diversified ETF basket, compare ordinary sample covariance "
            "with Ledoit-Wolf shrinkage. Use historical mean returns, a 504-day "
            "lookback, trade every 63 observations, cap positions at 30%, and "
            "charge 10 basis points per unit of turnover."
        ),
        "expected_calls": PAIR,
        "kind": "tool",
    },
    {
        "id": "incomplete_request",
        "question": "Run a backtest.",
        "expected_calls": [],
        "kind": "clarification",
        "answer_terms": ("universe", "asset", "which", "specify"),
    },
    {
        "id": "invalid_window",
        "question": (
            "Run a multi_asset backtest with sample covariance, historical "
            "returns, a 999-observation training window, 63-observation "
            "rebalancing, 0.30 max weight, and 10 bps cost."
        ),
        "expected_calls": [],
        "kind": "clarification",
        "answer_terms": ("999", "invalid", "allowed", "window"),
    },
    {
        "id": "three_arm_comparison",
        "question": (
            "Run exactly three multi_asset experiments: sample covariance with "
            "historical expected returns; Ledoit-Wolf covariance with historical "
            "expected returns; and Ledoit-Wolf covariance with shrunk expected "
            "returns. For each experiment use 504 training observations, "
            "63-observation rebalancing, a 0.30 weight cap, and 10 bps cost."
        ),
        "expected_calls": [
            PAIR[0], PAIR[1],
            {**COMMON, "covariance_method": "ledoit_wolf",
             "expected_return_method": "shrunk"},
        ],
        "kind": "tool",
    },
    {
        "id": "concept_no_tool",
        "question": "Explain conceptually what portfolio turnover means. Do not run a backtest.",
        "expected_calls": [],
        "kind": "no_tool",
    },
    *(
        {"id": f"repeat_pair_{index}", "question": PAIR_PROMPT,
         "expected_calls": PAIR, "kind": "tool"}
        for index in (1, 2, 3)
    ),
]


def score_case(case, calls, answer, error):
    calls_match = score_calls(calls, case["expected_calls"])
    if case["kind"] == "tool":
        response_match = (
            isinstance(answer, str) and
            "Backtest results (values reported directly by the engine):" in answer and
            answer.count("Run ") == len(case["expected_calls"]) and
            "error:" not in answer
        )
    else:
        response_match = (
            isinstance(answer, str) and bool(answer.strip()) and
            "Backtest results (values reported directly by the engine):" not in answer
        )
        if case["kind"] == "clarification":
            response_match = response_match and any(
                term in answer.lower() for term in case["answer_terms"]
            )
    return {"calls_match": calls_match, "response_match": response_match,
            "passed": calls_match and response_match and error is None}


def evaluate(model, output, *, snapshot_path="data/snapshots/multi_asset_2018-2024_2026-10-14.csv", case_ids=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Evaluation report exists; choose a new filename")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"model": model, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "cases": [], "complete": False}
    original = runner.run_backtest
    try:
        for case in CASES:
            if case_ids is not None and case["id"] not in case_ids:
                continue
            calls = []
            def fake_backtest(**arguments):
                arguments.pop("returns")
                calls.append(arguments.copy())
                effective = normalize(arguments)
                validate_backtest_parameters(**effective)
                score = .52 if effective["covariance_method"] == "ledoit_wolf" else .50
                if effective["expected_return_method"] == "shrunk":
                    score -= .02
                return {"universe": effective["universe"],
                        "strategy": {key: value for key, value in effective.items()
                                     if key != "universe"},
                        "metrics": {"Sharpe": score, "CAGR": .05}}
            runner.run_backtest = fake_backtest
            started = perf_counter()
            answer = error = None
            try:
                answer = runner.run_research_agent(case["question"], model=model,
                                                   snapshot_path=snapshot_path,
                                                   max_tool_calls=5)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            scores = score_case(case, calls, answer, error)
            result = {"id": case["id"], "question": case["question"],
                      "expected_calls": case["expected_calls"],
                      "actual_calls": calls, "answer": answer, "error": error,
                      "elapsed_seconds": round(perf_counter() - started, 3),
                      **scores}
            report["cases"].append(result)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"{case['id']}: {'PASS' if scores['passed'] else 'FAIL'} "
                  f"({result['elapsed_seconds']} s)", flush=True)
    finally:
        runner.run_backtest = original
    report["complete"] = True
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["passed"] = sum(case["passed"] for case in report["cases"])
    report["total"] = len(report["cases"])
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run expanded local Ollama evaluation")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", nargs="+", help="Only run these case IDs")
    parser.add_argument("--snapshot", default="data/snapshots/multi_asset_2018-2024_2026-10-14.csv")
    args = parser.parse_args()
    report = evaluate(args.model, args.output, snapshot_path=args.snapshot,
                      case_ids=set(args.cases) if args.cases else None)
    print(f"Passed {report['passed']}/{report['total']} cases")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

"""Held-out prompt evaluation of the snapshot-backed local Ollama workflow."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from agent import runner
from agent.current_agent_cases import CASES
from agent.evaluate import normalize, score_calls
from agent.tools import validate_backtest_parameters
from research.snapshot import load_snapshot


def score_case(case, calls, answer, error, snapshot_sha):
    calls_match = score_calls(calls, case["expected"])
    if case["expected"]:
        response_match = (
            isinstance(answer, str) and
            answer.count("Run ") == len(case["expected"]) and
            f"SHA-256: {snapshot_sha}" in answer and
            "error:" not in answer
        )
    else:
        response_match = (
            isinstance(answer, str) and bool(answer.strip()) and
            "Backtest results (values reported directly by the engine):" not in answer and
            any(term in answer.lower() for term in case["response_terms"])
        )
    return {"calls_match": calls_match, "response_match": response_match,
            "passed": calls_match and response_match and error is None}


def evaluate(snapshot_path, output, *, model="qwen3:8b"):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Evaluation report already exists")
    _, source = load_snapshot(snapshot_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cases_path = Path(__file__).with_name("current_agent_cases.py")
    report = {
        "model": model,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_source_sha256": sha256(cases_path.read_bytes()).hexdigest(),
        "snapshot_sha256": source["csv_sha256"],
        "snapshot": str(Path(snapshot_path).resolve()),
        "cases": [], "complete": False,
    }
    original = runner.run_backtest
    try:
        for case in CASES:
            calls = []
            return_object_ids = []
            def fake_backtest(**arguments):
                returns = arguments.pop("returns")
                return_object_ids.append(id(returns))
                calls.append(arguments.copy())
                effective = normalize(arguments)
                validate_backtest_parameters(**effective)
                score = .5 + (.02 if effective["covariance_method"] == "ledoit_wolf" else 0)
                return {
                    "universe": effective["universe"],
                    "strategy": {key: value for key, value in effective.items()
                                 if key != "universe"},
                    "metrics": {"Sharpe": score, "CAGR": .05},
                }
            runner.run_backtest = fake_backtest
            started = perf_counter()
            answer = error = None
            try:
                answer = runner.run_research_agent(
                    case["question"], snapshot_path=snapshot_path,
                    model=model, max_tool_calls=6,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            scored = score_case(case, calls, answer, error, source["csv_sha256"])
            same_returns_object = len(set(return_object_ids)) <= 1
            scored["passed"] = scored["passed"] and same_returns_object
            result = {
                "id": case["id"], "question": case["question"],
                "expected_calls": case["expected"], "actual_calls": calls,
                "same_returns_object": same_returns_object,
                "answer": answer, "error": error,
                "elapsed_seconds": round(perf_counter() - started, 3),
                **scored,
            }
            report["cases"].append(result)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"{case['id']}: {'PASS' if scored['passed'] else 'FAIL'} "
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
    parser = argparse.ArgumentParser(description="Evaluate snapshot-backed Ollama on held-out prompts")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="qwen3:8b")
    args = parser.parse_args()
    report = evaluate(args.snapshot, args.output, model=args.model)
    print(f"Passed {report['passed']}/{report['total']} cases")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

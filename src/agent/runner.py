"""Local Ollama research assistant over a verified, explicitly chosen snapshot."""
import argparse
from copy import deepcopy
from itertools import product
import json
import os
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from analytics.returns import calculate_simple_returns
from agent.schemas import RUN_BACKTEST_TOOL
from agent.tools import run_backtest
from research.snapshot import load_snapshot
from research.universes import UNIVERSES

SYSTEM_PROMPT = """You are a quantitative research assistant. Use run_backtest for
empirical portfolio claims. For comparisons, change only the parameter under
study. Base numbers only on tool results. Never claim statistical significance
without a test. Ask for missing settings rather than inventing them. The active
snapshot is fixed; do not use another universe or date range."""
FOLLOWUP_REFERENCE = re.compile(r"\b(same|previous|earlier|last|rerun|repeat)\b", re.I)
BARE_BACKTEST = re.compile(r"^\s*(run|do|perform)\s+(a\s+)?backtest[.!?]?\s*$", re.I)
TRAINING_WINDOW = re.compile(
    r"\b(\d+)\s*[- ]\s*(?:observation|day)\s+(?:training window|lookback)\b"
    r"|\b(\d+)\s+training (?:observations|days)\b"
    r"|\btrain_window\s*=\s*(\d+)\b", re.I,
)
REBALANCE = re.compile(
    r"\b(?:rebalanc(?:e|ing)|trade)\s+(?:every\s+)?(\d+)\s+observations\b"
    r"|\b(\d+)\s*[- ]\s*observation\s+rebalanc(?:e|ing)\b"
    r"|\brebalance_frequency\s*=\s*(\d+)\b", re.I,
)
CAP = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*%\s*(?:maximum weight|max weight|cap)\b"
    r"|\b(\d+(?:\.\d+)?)\s+percent\s+(?:maximum weight|max weight|cap)\b"
    r"|\bmax_weight\s*=\s*(0?\.\d+|1(?:\.0+)?)\b"
    r"|\b(0?\.\d+|1(?:\.0+)?)\s+(?:maximum weight|max weight|weight cap)\b", re.I,
)
COST = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:bps|basis points)\b"
    r"|\btransaction_cost_bps\s*=\s*(\d+(?:\.\d+)?)\b", re.I,
)
EXPLICIT_UNIVERSE = re.compile(r"\b([A-Za-z_]+)\s+(?:stock\s+)?universe\b", re.I)
COST_LIST = re.compile(
    r"((?:\d+(?:\.\d+)?\s*,\s*)+(?:and\s+)?\d+(?:\.\d+)?)\s*bps\b", re.I,
)


def _comparison_dimensions(question):
    """Extract explicitly requested comparison arms; leave unspecified choices to Ollama."""
    if not re.search(r"\b(compare|versus|against|three|two|sweep)\b", question, re.I):
        return {}
    dimensions = {}
    covariance = set()
    if re.search(r"\b(sample|ordinary)(?:[- ]+(?:covariance|estimate))?\b", question, re.I):
        covariance.add("sample")
    if re.search(r"\bLedoit[- ]Wolf\b", question, re.I):
        covariance.add("ledoit_wolf")
    if len(covariance) > 1:
        dimensions["covariance_method"] = covariance
    means = set()
    if re.search(r"\bhistorical\s+means?\b", question, re.I):
        means.add("historical")
    if re.search(r"\bshrunk\s+means?\b", question, re.I):
        means.add("shrunk")
    if re.search(
        r"\b(?:historical\s+(?:versus|vs\.?|and)\s+shrunk|"
        r"shrunk\s+(?:versus|vs\.?|and)\s+historical)\s+means?\b",
        question, re.I,
    ):
        means.update(("historical", "shrunk"))
    if len(means) > 1:
        dimensions["expected_return_method"] = means
    windows = {int(_first_number(match)) for match in TRAINING_WINDOW.finditer(question)}
    for match in re.finditer(
        r"\b(126|252|504|756)\s+(?:versus|vs\.?|and)\s+(126|252|504|756)\s+training", question, re.I):
        windows.update(map(int, match.groups()))
    if len(windows) > 1:
        dimensions["train_window"] = windows
    costs = _requested_costs(question)
    if len(costs) > 1:
        dimensions["transaction_cost_bps"] = costs
    return dimensions


def _comparison_error(question, results):
    dimensions = _comparison_dimensions(question)
    if not dimensions:
        return None
    keys = tuple(dimensions)
    expected = set(product(*(dimensions[key] for key in keys)))
    actual = [tuple(result["strategy"][key] for key in keys) for result in results]
    missing = expected - set(actual)
    unexpected = set(actual) - expected
    duplicates = len(actual) - len(set(actual))
    if missing or unexpected or duplicates:
        return ("Comparison incomplete or invalid: requested configurations "
                f"{len(expected)}, completed runs {len(actual)}; "
                f"missing {sorted(missing)}, unexpected {sorted(unexpected)}, "
                f"duplicate runs {duplicates}. No comparative results were reported.")
    common_keys = set(results[0]["strategy"]) - set(keys)
    for key in common_keys:
        if len({result["strategy"].get(key) for result in results}) != 1:
            return (f"Comparison invalid: {key} differs across runs. "
                    "No comparative results were reported.")
    return None



def _complete_requested_comparison(question, results, returns, source, max_tool_calls):
    """Fill missing explicit comparison arms using the first completed run's settings."""
    dimensions = _comparison_dimensions(question)
    if not dimensions or not results:
        return
    if len(dimensions) > 1 and not re.search(
        r"\ball\s+\w*\s*combinations\b", question, re.I
    ):
        return
    keys = tuple(dimensions)
    expected = set(product(*(dimensions[key] for key in keys)))
    actual = [tuple(result["strategy"][key] for key in keys) for result in results]
    if len(actual) != len(set(actual)) or not set(actual) <= expected:
        return
    common_keys = set(results[0]["strategy"]) - set(keys)
    if any(len({result["strategy"].get(key) for result in results}) != 1
           for key in common_keys):
        return
    missing = expected - set(actual)
    if len(results) + len(missing) > max_tool_calls:
        return
    for arm in sorted(missing):
        arguments = {**results[0]["strategy"], **dict(zip(keys, arm)),
                     "universe": source["universe"]}
        _validate_question_tool_alignment(question, arguments, source["universe"])
        result = run_backtest(**arguments, returns=returns)
        result["source"] = source
        results.append(result)


def _first_number(match):
    return next(value for value in match.groups() if value is not None)


def _requested_costs(question):
    values = {float(_first_number(match)) for match in COST.finditer(question)}
    for match in COST_LIST.finditer(question):
        values.update(float(value) for value in re.findall(r"\d+(?:\.\d+)?", match.group(1)))
    return values


def _is_empirical(question):
    if re.search(r"\b(do not|without)\s+run(?:ning)?\b", question, re.I):
        return False
    return bool(re.search(r"\b(run|backtest|compare|test)\b", question, re.I))


def _question_preflight(question, snapshot_universe=None):
    if BARE_BACKTEST.fullmatch(question):
        return "Please specify the universe and settings you want to test."
    for match in TRAINING_WINDOW.finditer(question):
        window = int(_first_number(match))
        if window not in {126, 252, 504, 756}:
            return (f"Training window {window} is unsupported. Choose 126, 252, "
                    "504, or 756 observations; no backtest was run.")
    for match in REBALANCE.finditer(question):
        frequency = int(_first_number(match))
        if frequency not in {5, 21, 63}:
            return (f"Rebalance frequency {frequency} is unsupported. Choose 5, "
                    "21, or 63 observations; no backtest was run.")
    for match in CAP.finditer(question):
        raw = _first_number(match)
        cap = (float(raw) / 100 if re.search(r"%|percent", match.group(0), re.I)
               else float(raw))
        if not 0.2 <= cap <= 1.0:
            return (f"Maximum weight {cap:g} is unsupported; choose a value "
                    "from 0.20 to 1.0 that permits a fully invested portfolio.")
    for cost in _requested_costs(question):
        if not 0 <= cost <= 100:
            return f"Transaction cost {cost:g} bps is unsupported; choose 0 to 100 bps."
    if snapshot_universe is None or not _is_empirical(question):
        return None
    explicit = EXPLICIT_UNIVERSE.search(question)
    if explicit:
        requested = explicit.group(1).lower()
        if requested not in UNIVERSES:
            return f"Universe '{requested}' is unsupported; no backtest was run."
        if requested != snapshot_universe:
            return (f"The selected snapshot contains {snapshot_universe}, not "
                    f"{requested}. Choose a matching snapshot; no backtest was run.")
    elif named := {name for name in UNIVERSES if re.search(rf"\b{re.escape(name)}\b", question, re.I)}:
        if len(named) > 1:
            return "Please select one universe for this snapshot-backed request."
        requested = next(iter(named))
        if requested != snapshot_universe:
            return (f"The selected snapshot contains {snapshot_universe}, not "
                    f"{requested}. Choose a matching snapshot; no backtest was run.")
    elif re.search(r"\bETF(?:s)?\b", question, re.I):
        if snapshot_universe != "multi_asset":
            return "The selected snapshot does not contain the ETF multi_asset universe."
    else:
        return "Please specify the universe to compare against the selected snapshot."
    return None


def _validate_question_tool_alignment(question, arguments, snapshot_universe=None):
    if snapshot_universe is not None and arguments.get("universe") != snapshot_universe:
        raise ValueError(f"The selected snapshot contains {snapshot_universe}, not "
                         f"{arguments.get('universe')}")
    if re.search(r"\bETF(?:s)?\b", question, re.I) and arguments.get("universe") != "multi_asset":
        raise ValueError("ETF basket requests must use the multi_asset universe")
    for pattern, key, converter, label in (
        (TRAINING_WINDOW, "train_window", int, "training window"),
        (REBALANCE, "rebalance_frequency", int, "rebalance frequency"),
    ):
        requested = {converter(_first_number(match)) for match in pattern.finditer(question)}
        requested.update(_comparison_dimensions(question).get(key, set()))
        if requested and arguments.get(key) not in requested:
            raise ValueError(f"Tool {label} {arguments.get(key)} differs from "
                             f"the requested {sorted(requested)}")
    costs = _requested_costs(question)
    if costs and arguments.get("transaction_cost_bps") not in costs:
        raise ValueError(f"Tool transaction cost in bps {arguments.get('transaction_cost_bps')} "
                         f"differs from the requested {sorted(costs)}")
    caps = set()
    for match in CAP.finditer(question):
        raw = float(_first_number(match))
        caps.add(raw / 100 if re.search(r"%|percent", match.group(0), re.I) else raw)
    if caps and arguments.get("max_weight") not in caps:
        raise ValueError(f"Tool max weight {arguments.get('max_weight')} differs "
                         f"from the requested {sorted(caps)}")


def _snapshot_context(snapshot_path):
    path = Path(snapshot_path).expanduser().resolve()
    prices, manifest = load_snapshot(path)
    returns = calculate_simple_returns(prices).dropna()
    if returns.empty:
        raise ValueError("Snapshot has no aligned return observations")
    source = {
        "snapshot": path.name,
        "sha256": manifest["csv_sha256"],
        "universe": manifest["universe"],
        "first_price_date": manifest["first_date"],
        "last_price_date": manifest["last_date"],
        "return_observations": len(returns),
    }
    return returns, source


def _tool_schema(universe):
    schema = deepcopy(RUN_BACKTEST_TOOL)
    schema["function"]["parameters"]["properties"]["universe"]["enum"] = [universe]
    return schema


def format_results(results, source=None):
    """Display values supplied by the engine with snapshot provenance."""
    lines = ["Backtest results (values reported directly by the engine):"]
    if source:
        lines.extend((f"Snapshot: {source['snapshot']}",
                      f"SHA-256: {source['sha256']}",
                      f"Price dates: {source['first_price_date']} to {source['last_price_date']}"))
    for index, result in enumerate(results, 1):
        lines.append(f"\nRun {index}: {result['universe']}")
        lines.extend(f"  {key}: {value}" for key, value in result["strategy"].items())
        lines.extend(f"  {key}: {value:.6g}" for key, value in result["metrics"].items())
    lines.append("\nThese historical results are descriptive, not a significance test. "
                 "The selected universe and execution model limit interpretation.")
    return "\n".join(lines)


def compare_covariance(*, snapshot_path, universe="multi_asset",
                       expected_return_method="historical", train_window=504,
                       rebalance_frequency=63, max_weight=0.3,
                       transaction_cost_bps=10):
    """Run both covariance methods on the same verified snapshot returns."""
    returns, source = _snapshot_context(snapshot_path)
    if universe != source["universe"]:
        raise ValueError(f"Snapshot contains {source['universe']}, not {universe}")
    common = dict(universe=universe, expected_return_method=expected_return_method,
                  train_window=train_window, rebalance_frequency=rebalance_frequency,
                  max_weight=max_weight, transaction_cost_bps=transaction_cost_bps,
                  returns=returns)
    return format_results([
        run_backtest(covariance_method=method, **common)
        for method in ("sample", "ledoit_wolf")
    ], source)


def _chat(messages, *, model, base_url, timeout, tool_schema):
    payload = json.dumps({"model": model, "messages": messages,
                          "tools": [tool_schema], "stream": False}).encode()
    request = Request(f"{base_url.rstrip('/')}/api/chat", data=payload,
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)["message"]
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Could not reach Ollama at {base_url}: {exc}. "
                           "Start Ollama and pull a tool-calling model.") from exc


def run_research_agent(question: str, *, snapshot_path=None, model: str | None = None,
                       base_url: str | None = None, max_tool_calls: int = 6,
                       timeout: int = 120) -> str:
    """Answer from one checksum-verified snapshot; never download market data."""
    if not question.strip():
        raise ValueError("question must not be empty")
    preflight = _question_preflight(question)
    if preflight:
        return preflight
    if FOLLOWUP_REFERENCE.search(question):
        raise ValueError("This command has no memory of earlier runs. Specify every "
                         "parameter explicitly, or use --compare-covariance with flags.")
    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be positive")
    snapshot_path = snapshot_path or os.getenv("RESEARCH_SNAPSHOT")
    if not snapshot_path:
        raise ValueError("Choose a verified price snapshot with --snapshot or RESEARCH_SNAPSHOT")
    returns, source = _snapshot_context(snapshot_path)
    preflight = _question_preflight(question, source["universe"])
    if preflight:
        return preflight
    model = model or os.getenv("OLLAMA_MODEL", "qwen3:8b")
    base_url = base_url or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    prompt = SYSTEM_PROMPT + (f"\nActive snapshot: {source['snapshot']}; universe "
                              f"{source['universe']}; prices {source['first_price_date']} "
                              f"through {source['last_price_date']}; SHA-256 {source['sha256']}. "
                              "The ETF basket is multi_asset; equity is the five-stock universe.")
    messages = [{"role": "system", "content": prompt},
                {"role": "user", "content": question}]
    calls = 0
    results = []
    errors = []
    while True:
        message = _chat(messages, model=model, base_url=base_url, timeout=timeout,
                        tool_schema=_tool_schema(source["universe"]))
        messages.append(message)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            if results:
                _complete_requested_comparison(
                    question, results, returns, source, max_tool_calls
                )
                comparison_error = _comparison_error(question, results)
                if comparison_error:
                    return comparison_error
                return format_results(results, source)
            if errors:
                return "No backtest completed: " + "; ".join(errors)
            answer = message.get("content", "")
            if not answer:
                raise RuntimeError("Ollama returned neither an answer nor a tool call")
            if _is_empirical(question):
                return "No backtest completed; empirical results require a tool call."
            return answer
        for call in tool_calls:
            if calls >= max_tool_calls:
                raise RuntimeError(f"Agent exceeded {max_tool_calls} backtest calls")
            function = call.get("function", {})
            if function.get("name") != "run_backtest":
                raise ValueError(f"Unknown tool: {function.get('name')}")
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            try:
                _validate_question_tool_alignment(question, arguments, source["universe"])
                result = run_backtest(**arguments, returns=returns)
                result["source"] = source
            except (TypeError, ValueError, RuntimeError) as exc:
                result = {"error": str(exc)}
            if "error" not in result:
                results.append(result)
            else:
                errors.append(result["error"])
            messages.append({"role": "tool", "tool_name": "run_backtest",
                             "content": json.dumps(result)})
            calls += 1


def main():
    parser = argparse.ArgumentParser(description="Ask the local snapshot-backed Ollama agent")
    parser.add_argument("question", nargs="?", help="Portfolio research question")
    parser.add_argument("--snapshot", help="Price CSV with matching .csv.json checksum manifest")
    parser.add_argument("--model", default=None, help="Installed Ollama tool-calling model")
    parser.add_argument("--compare-covariance", action="store_true",
                        help="Run sample and Ledoit-Wolf with identical settings")
    parser.add_argument("--universe", default="multi_asset")
    parser.add_argument("--expected-return-method", default="historical")
    parser.add_argument("--train-window", type=int, default=504)
    parser.add_argument("--rebalance-frequency", type=int, default=63)
    parser.add_argument("--max-weight", type=float, default=0.3)
    parser.add_argument("--transaction-cost-bps", type=float, default=10)
    args = parser.parse_args()
    if args.compare_covariance:
        snapshot_path = args.snapshot or os.getenv("RESEARCH_SNAPSHOT")
        if not snapshot_path:
            parser.error("--snapshot or RESEARCH_SNAPSHOT is required")
        print(compare_covariance(
            snapshot_path=snapshot_path, universe=args.universe,
            expected_return_method=args.expected_return_method,
            train_window=args.train_window, rebalance_frequency=args.rebalance_frequency,
            max_weight=args.max_weight, transaction_cost_bps=args.transaction_cost_bps,
        ))
    elif args.question:
        print(run_research_agent(args.question, snapshot_path=args.snapshot, model=args.model))
    else:
        parser.error("provide a question or --compare-covariance")


if __name__ == "__main__":
    main()

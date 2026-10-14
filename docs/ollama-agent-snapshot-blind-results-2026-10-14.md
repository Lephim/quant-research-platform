# Snapshot-backed Ollama agent: held-out evaluation

The [twelve-case plan](ollama-agent-snapshot-blind-plan-2026-10-14.md) and its [case file](../src/agent/blind_cases.py) were fixed before the snapshot-backed runner was changed. The [primary JSON report](../results/tables/ollama_agent_snapshot_blind_2026-10-14.json) contains every prompt, expected and actual tool call, answer, latency, and the SHA-256 of the case file and snapshot. Tool metrics were canned; no market data was downloaded during evaluation.

The active snapshot was `multi_asset_2018-2024_2026-10-14.csv`, SHA-256 `d87df74b9c463a5e7a82ebe66634b9025578f8451c372f80390f387fc9ef9d10`. The agent is now required to verify a selected snapshot and uses its return series for every tool call. A real local Ollama smoke test on this snapshot reported the correct 10 bps setting after an earlier run exposed a 0.1 bps misinterpretation; explicit numeric settings are checked against proposed tool arguments before execution.

## Results

**10 of 12 primary cases passed.** The ETF covariance pair, mean-estimator pair, explicit single backtest, both conceptual no-tool questions, and all five incompatible/missing/invalid-input cases passed. Tool-backed answers included the snapshot checksum.

- **Three-value cost sweep failed the strict call score.** The model called 0 and 10 bps correctly, then sent a malformed key (`expected,return_method`) for the 25 bps arm. The tool rejected that call, and the model retried 25 bps correctly. The final displayed result had the three requested costs, but the extra failed call means the run did not make exactly the expected calls.
- **Window comparison failed.** The primary attempt received an Ollama HTTP 500 after one call. A [single unchanged retry](../results/tables/ollama_agent_snapshot_blind_window_retry_2026-10-14.json) completed but still made only the 504-observation call; it omitted the requested 252-observation arm. The answer presented one result rather than a complete comparison.

These are workflow outcomes from twelve fixed prompts, not a general reliability estimate. The deterministic parameter checks prevent known substitutions such as 10 bps becoming 0.1 bps, and a missing or mismatched snapshot stops before a backtest. They do not guarantee that the model will call every requested experiment. Use `python -m research.robustness run` or an explicit deterministic comparison command when completeness matters; inspect the run count and parameters before interpreting model-selected experiments. The snapshot-backed run has no paid API cost, but local model latency in the report is substantial and varied by prompt.

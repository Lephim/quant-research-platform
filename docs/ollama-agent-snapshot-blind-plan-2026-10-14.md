# Snapshot-backed agent evaluation plan

Fixed on 14 October 2026 before changing the runner. The target is the local `qwen3:8b` workflow with a selected, checksum-verified `multi_asset` snapshot. The model sees only the active snapshot's universe and date range; every tool-backed result must identify its checksum. Canned metric values stand in for the optimizer during evaluation, so the test measures selection and control rather than changing prices.

The held-out prompt set is defined in `src/agent/blind_cases.py` and has twelve cases. It covers ETF synonyms, mean-estimator comparisons, a three-value cost sweep, a window comparison, requests for a different or unsupported universe, missing universe, unsupported cap and rebalance values, two conceptual questions, and an explicit single backtest. The prompts and expected calls are fixed before seeing model output. A failure will be reported, not tuned away in this evaluation.

Pass rules: empirical cases must execute exactly the expected effective tool calls and show the active snapshot SHA-256; incompatible, incomplete, or invalid requests must execute no backtest and produce a helpful response; conceptual cases must execute no backtest. This is a small held-out smoke test, not a population-level estimate of agent reliability.

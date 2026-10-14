# Post-fix live comparison check

Frozen before running the repaired runner against Ollama on 14 October 2026. This is a fresh four-case set, separate from the original twelve-case and current-runner eight-case evaluations. Cases are in `src/agent/post_fix_cases.py`; their SHA-256 is recorded in the JSON report. Use the same verified 2018–2024 multi-asset snapshot and local `qwen3:8b` model. Tool metrics are canned, and no market data is downloaded.

Three cases request an explicit window pair, four-arm covariance/mean grid, and covariance pair. One case requests an incompatible universe and must stop before a backtest. An empirical case passes only when the exact requested effective tool calls complete with one shared snapshot return object and the final answer shows the right run count and snapshot checksum. All attempted calls, including rejected arguments, remain visible in the report. The incompatible request passes only with zero tool calls and a helpful response. Record latency and errors. Do not modify prompts after observing results or replace this run with a retry.

Run `python -m agent.evaluate_post_fix --snapshot data/snapshots/multi_asset_2018-2024_2026-10-14.csv --output results/tables/ollama_agent_post_fix_2026-10-14.json --model qwen3:8b`. This small test does not establish general model reliability or portfolio performance.

# Local Ollama agent evaluation, 14 October 2026

The [saved evaluation report](../results/tables/ollama_agent_eval_2026-10-14.json) tested the installed `qwen3:8b` model with two fixed empirical prompts and canned backtest tool results. No market-data download or optimizer run occurred during this evaluation.

- **Explicit single backtest:** the model selected the requested universe, covariance method, expected-return method, training window, rebalance interval, cap, and 10 bps cost in one tool call.
- **Controlled covariance pair:** the model issued exactly two calls that differed only by sample versus Ledoit–Wolf covariance.
- Both cases passed the call check. The user-visible metrics came from the canned tool results because the runner formats tool results directly; this verifies the output path, not the model's independent arithmetic.
- The two cases took 42.69 and 39.37 seconds on this machine. This tiny sample does not establish general reliability, cost, or latency distributions.

The evaluator initially marked both calls wrong because it compared `10` and `10.0` as different JSON text. The call records showed the settings matched; the scorer now normalizes numeric types, and the saved report was rescored to **2/2 passed** without rerunning the model. The regression test covers this case.

Next evaluation work should add paraphrases, incomplete requests, invalid parameters, comparison questions with more than two arms, and deliberate no-tool responses. It should also measure repeated-run variability before relying on free-form model tool choice for research claims. The deterministic `--compare-covariance` command remains the safer route for a controlled experiment.

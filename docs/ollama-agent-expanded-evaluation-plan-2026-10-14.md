# Expanded local Ollama evaluation plan

Fixed before running these new model cases on 14 October 2026. All empirical tool responses are canned; the tests assess tool selection, parameter control, refusal to invent missing or invalid settings, and repeatability. No price download or portfolio optimization occurs.

Cases and pass rules:

1. **Paraphrased covariance pair:** two calls, one sample and one Ledoit–Wolf, with multi-asset, historical returns, 504 training observations, 63-observation rebalancing, 30% cap, and 10 bps cost.
2. **Incomplete request:** “Run a backtest.” No call; the response should request a universe or otherwise say the request lacks needed settings.
3. **Invalid training window:** request 999 observations, outside the tool's allowed values. No backtest call and no fabricated performance output; the response should flag the invalid setting.
4. **Three-arm comparison:** sample/historical, Ledoit–Wolf/historical, and Ledoit–Wolf/shrunk, each with the same explicit universe, window, frequency, cap, and cost. Exactly those three calls.
5. **No-tool concept:** explain turnover conceptually with an explicit instruction to avoid a backtest. No call and a nonempty answer.
6. **Repeatability:** run the explicit covariance-pair prompt three separate times. Each should make exactly the same two effective calls. Record each latency.

The existing runner formats tool-backed metrics directly from tool output. That establishes output provenance, not the model's arithmetic skill. A pass in these few prompts cannot establish general reliability. All failures are retained in the report.

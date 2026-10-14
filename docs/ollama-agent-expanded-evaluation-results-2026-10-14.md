# Expanded local Ollama evaluation results, 14 October 2026

The [plan](ollama-agent-expanded-evaluation-plan-2026-10-14.md) fixed eight cases before the baseline run. All tool responses were canned; no market prices were downloaded and no portfolio optimizer ran. The full call and response records are in the [baseline JSON](../results/tables/ollama_agent_eval_expanded_2026-10-14.json) and [guarded full rerun JSON](../results/tables/ollama_agent_eval_expanded_guarded_full_2026-10-14.json).

| Case | Baseline | Guarded full rerun | Finding |
| --- | --- | --- | --- |
| Paraphrased ETF covariance pair | Fail | Pass | Baseline chose `equity` for “diversified ETF basket”; universe mapping and a pre-execution guard now direct it to `multi_asset`. |
| Incomplete “Run a backtest” | Fail | Pass | Baseline invented a universe and settings; a local preflight now asks for the universe without a tool call. |
| Unsupported 999-observation window | Fail | Pass | Baseline silently substituted 756; a local preflight now rejects this explicit unsupported window. |
| Three-arm comparison | Pass | Pass | Exactly the requested sample/historical, Ledoit–Wolf/historical, and Ledoit–Wolf/shrunk calls. |
| Conceptual turnover explanation | Pass | Pass | No backtest call. |
| Three repeated covariance pairs | Pass × 3 | Pass × 3 | Each run made the same two effective calls. |

**Score: 5/8 baseline, 8/8 guarded full rerun.** The guarded score describes the combined workflow, not model-only reliability. Two corrected cases are handled before the model runs, and the ETF case has a domain-specific tool guard. The baseline failures remain visible rather than being overwritten.

The six model-backed cases in the guarded run took about 27–71 seconds each on this machine; the two preflight cases returned immediately. These are a few local observations, not a latency distribution. No paid API was used. The runner formats successful tool metrics directly from tool output, so its displayed numbers are grounded by construction; this does not test the model's independent numerical reasoning.

The guards cover the particular missing and invalid forms tested here. They do not parse every possible natural-language setting, guarantee correct universe inference for every synonym, or prove that the model will always call a tool for empirical claims. For controlled research, the deterministic comparison commands remain preferable. Broader paraphrases and adversarial prompts are still needed before trusting free-form tool choice.

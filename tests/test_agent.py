"""Offline contract checks for snapshot-backed Ollama tool calls."""
import json
import unittest
from unittest.mock import patch

from agent import runner
from agent.tools import validate_backtest_parameters

EQUITY_SOURCE = {"snapshot": "equity.csv", "sha256": "abc123", "universe": "equity",
                 "first_price_date": "2018-01-01", "last_price_date": "2024-12-31",
                 "return_observations": 1000}
MULTI_SOURCE = {**EQUITY_SOURCE, "snapshot": "multi.csv", "universe": "multi_asset"}


class AgentTests(unittest.TestCase):
    def test_comparison_requires_every_requested_arm(self):
        question = ("Compare 252 versus 504 training observations for multi_asset "
                    "using sample covariance, historical means, 63-observation "
                    "rebalancing, 30% cap, and 10 bps cost.")
        def result(window):
            return {"universe": "multi_asset", "strategy": {
                "covariance_method": "sample", "expected_return_method": "historical",
                "train_window": window, "rebalance_frequency": 63,
                "max_weight": .3, "transaction_cost_bps": 10},
                "metrics": {"CAGR": .1}}
        incomplete = runner._comparison_error(question, [result(504)])
        self.assertIn("Comparison incomplete", incomplete)
        self.assertIsNone(runner._comparison_error(question, [result(252), result(504)]))

    def test_four_arm_covariance_mean_comparison(self):
        question = ("Compare all four combinations of sample versus Ledoit-Wolf "
                    "covariance and historical versus shrunk means in multi_asset. "
                    "Use 504 training observations, 63-observation rebalancing, "
                    "30% max weight, and 10 bps cost.")
        def result(covariance, means):
            return {"strategy": {"covariance_method": covariance,
                                 "expected_return_method": means,
                                 "train_window": 504,
                                 "rebalance_frequency": 63,
                                 "max_weight": .3,
                                 "transaction_cost_bps": 10}}
        arms = [result(covariance, means)
                for covariance in ("sample", "ledoit_wolf")
                for means in ("historical", "shrunk")]
        self.assertIsNone(runner._comparison_error(question, arms))
        self.assertIn("missing", runner._comparison_error(question, arms[:3]))
        self.assertIn("duplicate runs 1", runner._comparison_error(
            question, arms + [arms[0]]))

    def test_agent_completes_missing_explicit_arm(self):
        question = ("Compare sample and Ledoit-Wolf covariance in the multi_asset "
                    "universe with historical means, 504 training observations, "
                    "63-observation rebalancing, 30% cap, and 10 bps cost.")
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [{"function": {
                "name": "run_backtest", "arguments": {"universe": "multi_asset",
                "covariance_method": "sample", "expected_return_method": "historical",
                "train_window": 504, "rebalance_frequency": 63, "max_weight": .3,
                "transaction_cost_bps": 10}}}]},
            {"role": "assistant", "content": "Both methods completed."},
        ]
        result = {"universe": "multi_asset", "strategy": {
            "covariance_method": "sample", "expected_return_method": "historical",
            "train_window": 504, "rebalance_frequency": 63, "max_weight": .3,
            "transaction_cost_bps": 10}, "metrics": {"CAGR": .1}}
        def fake_backtest(**arguments):
            return {"universe": "multi_asset", "strategy": {
                key: value for key, value in arguments.items() if key != "returns"
                and key != "universe"}, "metrics": {"CAGR": .1}}
        with patch.object(runner, "_snapshot_context", return_value=("frozen", MULTI_SOURCE)), \
             patch.object(runner, "_chat", side_effect=messages), \
             patch.object(runner, "run_backtest", side_effect=fake_backtest) as backtest:
            answer = runner.run_research_agent(question, snapshot_path="multi.csv")
        self.assertEqual(backtest.call_count, 2)
        self.assertEqual(backtest.call_args.kwargs["covariance_method"], "ledoit_wolf")
        self.assertIn("Run 2", answer)
        self.assertIn("CAGR", answer)

    def test_completion_does_not_invent_multi_dimension_arms(self):
        question = ("Compare sample and Ledoit-Wolf covariance with historical "
                    "and shrunk means in multi_asset, using exactly three runs.")
        results = [{"strategy": {"covariance_method": "sample",
                                 "expected_return_method": "historical"}}]
        with patch.object(runner, "run_backtest") as backtest:
            runner._complete_requested_comparison(
                question, results, "returns", MULTI_SOURCE, 6, calls_used=0)
        backtest.assert_not_called()

    def test_comparison_requires_fixed_other_settings(self):
        question = ("Compare sample and Ledoit-Wolf covariance in multi_asset "
                    "with historical means, 504 training observations, "
                    "63-observation rebalancing, 30% cap, and 10 bps cost.")
        sample = {"strategy": {"covariance_method": "sample", "train_window": 504}}
        wolf = {"strategy": {"covariance_method": "ledoit_wolf", "train_window": 252}}
        self.assertIn("Comparison invalid", runner._comparison_error(question, [sample, wolf]))

    def test_tool_loop_uses_snapshot_returns_and_reports_sha(self):
        calls = [
            {"role": "assistant", "content": "", "tool_calls": [{"function": {
                "name": "run_backtest", "arguments": {"universe": "equity"}}}]},
            {"role": "assistant", "content": "The tool reported a result."},
        ]
        result = {"universe": "equity", "strategy": {"train_window": 252},
                  "metrics": {"CAGR": 0.1}}
        with patch.object(runner, "_snapshot_context", return_value=("frozen-returns", EQUITY_SOURCE)), \
             patch.object(runner, "_chat", side_effect=calls) as chat, \
             patch.object(runner, "run_backtest", return_value=result) as backtest:
            answer = runner.run_research_agent(
                "Run one backtest in the equity universe.", snapshot_path="equity.csv")
        self.assertIn("CAGR: 0.1", answer)
        self.assertIn("SHA-256: abc123", answer)
        self.assertNotIn("The tool reported", answer)
        backtest.assert_called_once_with(universe="equity", returns="frozen-returns")
        sent = chat.call_args.args[0]
        self.assertEqual(sent[-2]["role"], "tool")
        self.assertEqual(json.loads(sent[-2]["content"])["source"]["sha256"], "abc123")
        schema = chat.call_args.kwargs["tool_schema"]
        self.assertEqual(schema["function"]["parameters"]["properties"]["universe"]["enum"], ["equity"])

    def test_snapshot_required_for_empirical_question(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(ValueError, "snapshot"):
            runner.run_research_agent("Run one backtest in the equity universe.")

    def test_snapshot_universe_mismatch_stops_before_model(self):
        with patch.object(runner, "_snapshot_context", return_value=("frozen", MULTI_SOURCE)), \
             patch.object(runner, "_chat") as chat, \
             patch.object(runner, "run_backtest") as backtest:
            answer = runner.run_research_agent(
                "Run one backtest in the equity universe.", snapshot_path="multi.csv")
        self.assertIn("not equity", answer)
        chat.assert_not_called()
        backtest.assert_not_called()

    def test_empirical_answer_without_tool_is_not_reported(self):
        with patch.object(runner, "_snapshot_context", return_value=("frozen", MULTI_SOURCE)), \
             patch.object(runner, "_chat", return_value={"role": "assistant", "content": "Sharpe is 1.5"}):
            answer = runner.run_research_agent(
                "Compare strategies in the multi_asset universe.", snapshot_path="multi.csv")
        self.assertIn("No backtest completed", answer)
        self.assertNotIn("1.5", answer)

    def test_invalid_parameter_rejected_before_data_download(self):
        with self.assertRaisesRegex(ValueError, "covariance"):
            validate_backtest_parameters("equity", "invalid", "historical", 252, 21, .3, 10)

    def test_call_limit(self):
        call = {"role": "assistant", "content": "", "tool_calls": [{"function": {
            "name": "run_backtest", "arguments": {"universe": "equity"}}}]}
        with patch.object(runner, "_snapshot_context", return_value=("frozen", EQUITY_SOURCE)), \
             patch.object(runner, "_chat", return_value=call), \
             patch.object(runner, "run_backtest", return_value={"universe": "equity", "strategy": {}, "metrics": {}}):
            with self.assertRaisesRegex(RuntimeError, "exceeded"):
                runner.run_research_agent("Run an equity universe backtest.",
                                          snapshot_path="equity.csv", max_tool_calls=1)

    def test_ambiguous_followup_does_not_call_model(self):
        with patch.object(runner, "_chat") as chat:
            with self.assertRaisesRegex(ValueError, "no memory"):
                runner.run_research_agent("Rerun the same two configurations with 10 bps")
        chat.assert_not_called()

    def test_incomplete_and_invalid_requests_do_not_call_model(self):
        with patch.object(runner, "_chat") as chat:
            self.assertIn("specify the universe", runner.run_research_agent("Run a backtest."))
            self.assertIn("unsupported", runner.run_research_agent(
                "Run multi_asset with a 999-observation training window"))
            self.assertIn("unsupported", runner.run_research_agent(
                "Run multi_asset and rebalance every 17 observations"))
        chat.assert_not_called()

    def test_wrong_universe_rejected_before_engine(self):
        with self.assertRaisesRegex(ValueError, "multi_asset"):
            runner._validate_question_tool_alignment(
                "Compare the ETF basket", {"universe": "equity"}, "multi_asset")

    def test_explicit_cost_and_window_cannot_be_silently_changed(self):
        with self.assertRaisesRegex(ValueError, "transaction cost"):
            runner._validate_question_tool_alignment(
                "Run multi_asset with 504 training observations and 10 bps cost",
                {"universe": "multi_asset", "train_window": 504,
                 "transaction_cost_bps": .1}, "multi_asset")
        with self.assertRaisesRegex(ValueError, "training window"):
            runner._validate_question_tool_alignment(
                "Run multi_asset with 504 training observations",
                {"universe": "multi_asset", "train_window": 756}, "multi_asset")

    def test_comparison_alignment_accepts_both_window_arms(self):
        question = ("Compare 504 versus 252 training observations in multi_asset "
                    "with sample covariance, historical means, 63-observation "
                    "rebalancing, 30% max weight, and 10 bps cost.")
        arguments = {"universe": "multi_asset", "covariance_method": "sample",
                     "expected_return_method": "historical", "train_window": 504,
                     "rebalance_frequency": 63, "max_weight": .3,
                     "transaction_cost_bps": 10}
        runner._validate_question_tool_alignment(question, arguments, "multi_asset")
        arguments["train_window"] = 756
        with self.assertRaisesRegex(ValueError, "training window"):
            runner._validate_question_tool_alignment(question, arguments, "multi_asset")

    def test_explicit_covariance_and_mean_cannot_be_substituted(self):
        question = ("Run Ledoit-Wolf covariance and shrunk means for the "
                    "multi_asset universe at 10 bps")
        arguments = {"universe": "multi_asset", "covariance_method": "sample",
                     "expected_return_method": "historical",
                     "transaction_cost_bps": 10}
        with self.assertRaisesRegex(ValueError, "covariance method"):
            runner._validate_question_tool_alignment(question, arguments, "multi_asset")
        arguments["covariance_method"] = "ledoit_wolf"
        with self.assertRaisesRegex(ValueError, "expected-return method"):
            runner._validate_question_tool_alignment(question, arguments, "multi_asset")
        arguments["expected_return_method"] = "shrunk"
        runner._validate_question_tool_alignment(question, arguments, "multi_asset")

    def test_completion_counts_failed_tool_attempts_in_budget(self):
        question = "Compare sample and Ledoit-Wolf covariance in multi_asset."
        results = [{"strategy": {"covariance_method": "sample",
                                 "expected_return_method": "historical",
                                 "train_window": 504,
                                 "rebalance_frequency": 63,
                                 "max_weight": .3,
                                 "transaction_cost_bps": 10}}]
        with patch.object(runner, "run_backtest") as backtest:
            runner._complete_requested_comparison(
                question, results, "returns", MULTI_SOURCE, 2, calls_used=2)
        backtest.assert_not_called()
        self.assertEqual(len(results), 1)

    def test_controlled_covariance_comparison(self):
        def result(**kwargs):
            return {"universe": kwargs["universe"],
                    "strategy": {"covariance_method": kwargs["covariance_method"]},
                    "metrics": {"Sharpe": 0.5}}
        with patch.object(runner, "_snapshot_context", return_value=("same-data", MULTI_SOURCE)), \
             patch.object(runner, "run_backtest", side_effect=result) as backtest:
            output = runner.compare_covariance(snapshot_path="multi.csv", transaction_cost_bps=10)
        self.assertEqual(backtest.call_count, 2)
        first, second = [call.kwargs for call in backtest.call_args_list]
        self.assertEqual(first.pop("covariance_method"), "sample")
        self.assertEqual(second.pop("covariance_method"), "ledoit_wolf")
        self.assertEqual(first, second)
        self.assertEqual(first["returns"], "same-data")
        self.assertIn("SHA-256: abc123", output)


if __name__ == "__main__":
    unittest.main()

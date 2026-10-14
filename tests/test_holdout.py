"""Offline checks for the fixed holdout protocol."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research import holdout
from research.universes import UNIVERSES


class HoldoutTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2021-01-01", "2025-06-30")
        self.prices = pd.DataFrame(
            100 * np.cumprod(1 + np.full((len(dates), 7), .001), axis=0),
            index=dates, columns=UNIVERSES["multi_asset"],
        )

    def test_fixed_settings_and_common_dates(self):
        calls = []
        def experiment(returns, **kwargs):
            calls.append(kwargs)
            dates = returns.loc[kwargs["evaluation_start"]:].index
            return SimpleNamespace(returns=pd.Series(np.zeros(len(dates)), index=dates),
                                   metrics={"Sharpe": .5})
        with patch.object(holdout, "run_strategy_experiment", side_effect=experiment), \
             patch.object(holdout, "run_equal_weight_experiment", side_effect=experiment):
            summary, daily = holdout.run_holdout(self.prices)
        self.assertEqual(list(summary.strategy), ["ledoit_wolf", "sample", "equal_weight"])
        self.assertEqual(list(daily.columns), list(summary.strategy))
        self.assertEqual(len({settings["evaluation_start"] for settings in calls}), 1)
        self.assertTrue(all(settings["train_window"] == 756 for settings in calls))
        self.assertTrue(all(settings["rebalance_frequency"] == 63 for settings in calls))
        self.assertTrue(all(settings["transaction_cost_bps"] == 10 for settings in calls))
        self.assertGreaterEqual(daily.index[0], pd.Timestamp("2025-01-01"))

    def test_requires_training_history(self):
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            holdout.run_holdout(self.prices.loc["2023-01-01":])


if __name__ == "__main__":
    unittest.main()

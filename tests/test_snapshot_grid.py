"""Offline checks for snapshot provenance and controlled grid setup."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research import robustness, snapshot
from research.universes import UNIVERSES


class SnapshotGridTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2020-01-01", periods=45)
        tickers = UNIVERSES["multi_asset"]
        self.prices = pd.DataFrame(
            100 * np.cumprod(1 + np.full((45, len(tickers)), .001), axis=0),
            index=dates, columns=tickers,
        )

    def test_capture_load_and_checksum(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            with patch.object(snapshot, "load_market_data", return_value={"Close": self.prices}):
                manifest_path = snapshot.capture_snapshot(path)
            loaded, manifest = snapshot.load_snapshot(path)
            expected = self.prices.rename_axis("Date")
            pd.testing.assert_frame_equal(loaded, expected, check_exact=False,
                                          rtol=1e-14, check_freq=False)
            self.assertEqual(manifest["csv_sha256"], json.loads(manifest_path.read_text())["csv_sha256"])
            with self.assertRaises(FileExistsError):
                snapshot.capture_snapshot(path)
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                snapshot.load_snapshot(path)

    def test_grid_uses_common_evaluation_dates(self):
        seen = []
        def experiment(*, returns, evaluation_start, **settings):
            seen.append((evaluation_start, settings))
            dates = returns.loc[evaluation_start:].index
            result = type("Result", (), {})()
            result.returns = pd.Series(np.zeros(len(dates)), index=dates)
            result.metrics = {"Sharpe": 0.5}
            return result
        with patch.object(robustness, "run_strategy_experiment", side_effect=experiment):
            grid = robustness.run_grid(self.prices, windows=(10, 20), frequencies=(5, 7))
        self.assertEqual(len(grid), 8)
        self.assertEqual(len({start for start, _ in seen}), 1)
        self.assertEqual(len(set(grid["evaluation_start"])), 1)
        for window in (10, 20):
            for frequency in (5, 7):
                subset = grid[(grid.train_window == window) & (grid.rebalance_frequency == frequency)]
                self.assertEqual(set(subset.covariance_method), {"sample", "ledoit_wolf"})


if __name__ == "__main__":
    unittest.main()

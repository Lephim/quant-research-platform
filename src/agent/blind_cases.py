"""Held-out snapshot-backed agent cases fixed before the runner change."""

BASE = {
    "universe": "multi_asset",
    "covariance_method": "sample",
    "expected_return_method": "historical",
    "train_window": 504,
    "rebalance_frequency": 63,
    "max_weight": 0.3,
    "transaction_cost_bps": 10.0,
}

CASES = [
    {
        "id": "etf_alias_pair",
        "question": "For the seven-ETF basket, test ordinary covariance against Ledoit-Wolf; use 504 training days, rebalance every 63 observations, historical means, a 30 percent cap, and 10 bps cost.",
        "expected": [BASE, {**BASE, "covariance_method": "ledoit_wolf"}],
    },
    {
        "id": "mean_estimator_pair",
        "question": "In multi_asset, compare historical means with shrunk means while using Ledoit-Wolf covariance. Set train_window=504, rebalance_frequency=63, max_weight=0.30, transaction_cost_bps=10.",
        "expected": [{**BASE, "covariance_method": "ledoit_wolf"},
                     {**BASE, "covariance_method": "ledoit_wolf", "expected_return_method": "shrunk"}],
    },
    {
        "id": "cost_sweep_three",
        "question": "Run three multi_asset sample-covariance backtests at 0, 10, and 25 bps transaction cost. Use historical means, 504 training observations, rebalance every 63 observations, and max weight 0.30 in each run.",
        "expected": [{**BASE, "transaction_cost_bps": cost} for cost in (0, 10, 25)],
    },
    {
        "id": "window_pair",
        "question": "How do 252 versus 504 training observations compare for multi_asset using sample covariance? Keep historical means, 63-observation rebalancing, a 30% cap, and 10 bps cost.",
        "expected": [{**BASE, "train_window": 252}, BASE],
    },
    {
        "id": "wrong_snapshot_universe",
        "question": "Backtest the equity stock universe with sample covariance, historical means, 504 training observations, 63-observation rebalancing, 30% cap, and 10 bps cost.",
        "expected": [], "response_terms": ("snapshot", "multi_asset", "universe"),
    },
    {
        "id": "unsupported_universe",
        "question": "Backtest the forex universe with sample covariance, 504 training observations, and 63-observation rebalancing.",
        "expected": [], "response_terms": ("universe", "unsupported", "snapshot"),
    },
    {
        "id": "missing_universe",
        "question": "Compare sample and Ledoit-Wolf covariance with 504 training observations, 63-observation rebalancing, 30% cap, and 10 bps cost.",
        "expected": [], "response_terms": ("universe", "snapshot", "specify"),
    },
    {
        "id": "unsupported_cap",
        "question": "Run multi_asset with sample covariance, historical means, 504 training observations, 63-observation rebalancing, a 10% maximum weight, and 10 bps cost.",
        "expected": [], "response_terms": ("weight", "cap", "unsupported"),
    },
    {
        "id": "unsupported_rebalance",
        "question": "Run multi_asset with sample covariance, historical means, 504 training observations, rebalance every 17 observations, max weight 30%, and 10 bps cost.",
        "expected": [], "response_terms": ("rebalance", "frequency", "unsupported"),
    },
    {
        "id": "concept_drawdown",
        "question": "Explain maximum drawdown in words. Do not run a backtest.",
        "expected": [], "response_terms": ("drawdown", "peak", "loss"),
    },
    {
        "id": "concept_significance",
        "question": "Does a tiny CAGR difference alone prove statistical significance? Answer conceptually without running data.",
        "expected": [], "response_terms": ("no", "not", "significance"),
    },
    {
        "id": "single_explicit",
        "question": "Run one multi_asset test: Ledoit-Wolf covariance, shrunk expected returns, 756 training observations, rebalance every 21 observations, 30% cap, and 10 bps cost.",
        "expected": [{**BASE, "covariance_method": "ledoit_wolf",
                      "expected_return_method": "shrunk", "train_window": 756,
                      "rebalance_frequency": 21}],
    },
]

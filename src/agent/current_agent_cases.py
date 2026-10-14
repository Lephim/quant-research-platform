"""Fixed prompts for the post-guard snapshot-backed agent evaluation."""

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
        "id": "covariance_pair_paraphrase",
        "question": "Compare ordinary sample covariance versus Ledoit-Wolf in the multi_asset universe. For both runs use historical means, 504 training observations, 63-observation rebalancing, a 30% maximum weight, and 10 bps cost.",
        "expected": [BASE, {**BASE, "covariance_method": "ledoit_wolf"}],
    },
    {
        "id": "window_pair_reverse_order",
        "question": "Compare 504 versus 252 training observations in multi_asset. Use sample covariance and historical means for both, rebalance every 63 observations, with 30% max weight and 10 bps cost.",
        "expected": [BASE, {**BASE, "train_window": 252}],
    },
    {
        "id": "cost_sweep_paraphrase",
        "question": "Compare 25, 10, and 0 bps transaction costs for multi_asset sample covariance. Keep historical means, 504 training observations, 63-observation rebalancing, and 30% max weight in all three runs.",
        "expected": [{**BASE, "transaction_cost_bps": cost} for cost in (25, 10, 0)],
    },
    {
        "id": "mean_pair_paraphrase",
        "question": "Compare historical means and shrunk means for multi_asset with Ledoit-Wolf covariance. In each run use 504 training observations, 63-observation rebalancing, 30% max weight, and 10 bps cost.",
        "expected": [{**BASE, "covariance_method": "ledoit_wolf"},
                     {**BASE, "covariance_method": "ledoit_wolf", "expected_return_method": "shrunk"}],
    },
    {
        "id": "covariance_mean_grid",
        "question": "Compare all four combinations of sample versus Ledoit-Wolf covariance and historical versus shrunk means in multi_asset. Hold 504 training observations, 63-observation rebalancing, 30% max weight, and 10 bps cost fixed.",
        "expected": [{**BASE, "covariance_method": covariance,
                      "expected_return_method": means}
                     for covariance in ("sample", "ledoit_wolf")
                     for means in ("historical", "shrunk")],
    },
    {
        "id": "missing_universe",
        "question": "Compare sample and Ledoit-Wolf covariance with historical means, 504 training observations, 63-observation rebalancing, 30% max weight, and 10 bps cost.",
        "expected": [], "response_terms": ("universe", "snapshot", "specify"),
    },
    {
        "id": "mismatched_universe",
        "question": "Compare sample and Ledoit-Wolf covariance in the equity universe with historical means, 504 training observations, 63-observation rebalancing, 30% max weight, and 10 bps cost.",
        "expected": [], "response_terms": ("snapshot", "multi_asset", "equity"),
    },
    {
        "id": "invalid_cost",
        "question": "Compare sample and Ledoit-Wolf covariance in multi_asset with historical means, 504 training observations, 63-observation rebalancing, 30% max weight, and 125 bps cost.",
        "expected": [], "response_terms": ("unsupported", "cost", "100"),
    },
]

"""Fresh fixed cases for the bounded comparison completion change."""
from agent.current_agent_cases import BASE

CASES = [
    {
        "id": "fresh_window_pair",
        "question": "For multi_asset, compare 252 and 504 training observations using sample covariance and historical means. Rebalance every 63 observations, set a 30% max weight, and charge 10 bps in both runs.",
        "expected": [{**BASE, "train_window": 252}, BASE],
    },
    {
        "id": "fresh_four_combinations",
        "question": "In the multi_asset universe, compare all four combinations of ordinary covariance versus Ledoit-Wolf and shrunk versus historical means. Use 504 training observations, 63-observation rebalancing, a 30% cap, and 10 bps cost for every arm.",
        "expected": [{**BASE, "covariance_method": covariance,
                      "expected_return_method": means}
                     for covariance in ("sample", "ledoit_wolf")
                     for means in ("historical", "shrunk")],
    },
    {
        "id": "fresh_covariance_pair",
        "question": "Test sample against Ledoit-Wolf covariance for multi_asset with historical means, 504 training observations, 63-observation rebalancing, a 30% max weight, and 10 bps transaction cost.",
        "expected": [BASE, {**BASE, "covariance_method": "ledoit_wolf"}],
    },
    {
        "id": "fresh_wrong_universe",
        "question": "Run a sample versus Ledoit-Wolf comparison in the equity universe using historical means, 504 training observations, 63-observation rebalancing, 30% cap, and 10 bps cost.",
        "expected": [], "response_terms": ("snapshot", "multi_asset", "equity"),
    },
]

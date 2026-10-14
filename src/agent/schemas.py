"""Ollama tool schema for the existing backtest entry point."""
RUN_BACKTEST_TOOL = {
    "type": "function",
    "function": {
        "name": "run_backtest",
        "description": "Run an out-of-sample walk-forward portfolio backtest.",
        "parameters": {
            "type": "object",
            "properties": {
                "universe": {"type": "string", "enum": ["equity", "multi_asset"]},
                "covariance_method": {"type": "string", "enum": ["sample", "ledoit_wolf"]},
                "expected_return_method": {"type": "string", "enum": ["historical", "shrunk"]},
                "train_window": {"type": "integer", "enum": [126, 252, 504, 756]},
                "rebalance_frequency": {"type": "integer", "enum": [5, 21, 63]},
                "max_weight": {"type": "number", "minimum": 0.20, "maximum": 1.0},
                "transaction_cost_bps": {"type": "number", "minimum": 0, "maximum": 100},
            },
            "required": ["universe"],
        },
    },
}

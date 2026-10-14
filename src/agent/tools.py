from data.loader import load_market_data
from analytics.returns import calculate_simple_returns
from research.experiments import run_strategy_experiment
from research.universes import UNIVERSES, START_DATE, END_DATE

ALLOWED_COVARIANCE_METHODS = {"sample", "ledoit_wolf"}
ALLOWED_RETURN_METHODS = {"historical", "shrunk"}
ALLOWED_TRAIN_WINDOWS = {126, 252, 504, 756}
ALLOWED_REBALANCE_FREQUENCIES = {5, 21, 63}


def run_backtest(
    universe: str,
    covariance_method: str = "sample",
    expected_return_method: str = "historical",
    train_window: int = 252,
    rebalance_frequency: int = 21,
    max_weight: float = 0.30,
    transaction_cost_bps: float = 10.0,
    returns=None,
) -> dict:

    validate_backtest_parameters(
        universe, covariance_method, expected_return_method,
        train_window, rebalance_frequency, max_weight, transaction_cost_bps,
    )

    if returns is None:
        returns = load_universe_returns(universe)

    result = run_strategy_experiment(
        returns=returns,
        train_window=train_window,
        rebalance_frequency=rebalance_frequency,
        max_weight=max_weight,
        covariance_method=covariance_method,
        expected_return_method=expected_return_method,
        transaction_cost_bps=transaction_cost_bps,
    )

    return {
        "universe": universe,
        "strategy": {
            "covariance_method": covariance_method,
            "expected_return_method": expected_return_method,
            "train_window": train_window,
            "rebalance_frequency": rebalance_frequency,
            "max_weight": max_weight,
            "transaction_cost_bps": transaction_cost_bps,
        },
        "metrics": {
            key: float(value)
            for key, value in result.metrics.items()
        },
    }


def load_universe_returns(universe):
    """Download and align one universe for reuse in controlled experiments."""
    if universe not in UNIVERSES:
        raise ValueError(f"Unknown universe '{universe}'. Choose from {list(UNIVERSES)}.")
    data = load_market_data(UNIVERSES[universe], START_DATE, END_DATE)
    prices = data["Close"]
    if prices.empty or prices.isna().all().any():
        raise ValueError("Market data is empty or missing an entire ticker")
    return calculate_simple_returns(prices).dropna()


def validate_backtest_parameters(
    universe,
    covariance_method,
    expected_return_method,
    train_window,
    rebalance_frequency,
    max_weight,
    transaction_cost_bps,
):
    if universe not in UNIVERSES:
        raise ValueError(
            f"Unknown universe '{universe}'. "
            f"Choose from {list(UNIVERSES)}."
        )

    if covariance_method not in ALLOWED_COVARIANCE_METHODS:
        raise ValueError(
            f"Unknown covariance method '{covariance_method}'."
        )

    if expected_return_method not in ALLOWED_RETURN_METHODS:
        raise ValueError(
            f"Unknown expected-return method '{expected_return_method}'."
        )

    if train_window not in ALLOWED_TRAIN_WINDOWS:
        raise ValueError(
            f"train_window must be one of {sorted(ALLOWED_TRAIN_WINDOWS)}."
        )

    if rebalance_frequency not in ALLOWED_REBALANCE_FREQUENCIES:
        raise ValueError(
            "rebalance_frequency must be one of "
            f"{sorted(ALLOWED_REBALANCE_FREQUENCIES)}."
        )

    if not 0.20 <= max_weight <= 1.0:
        raise ValueError("max_weight must be between 0.20 and 1.0.")

    n_assets = len(UNIVERSES[universe])

    if n_assets * max_weight < 1.0:
        raise ValueError(
            "max_weight is too restrictive for this universe: "
            "a fully invested portfolio is impossible."
        )

    if not 0 <= transaction_cost_bps <= 100:
        raise ValueError(
            "transaction_cost_bps must be between 0 and 100."
        )

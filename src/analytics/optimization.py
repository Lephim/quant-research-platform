import numpy as np
from scipy.optimize import minimize
from analytics.portfolio import portfolio_volatility


def random_weights(n_assets):
    """
    Generate random portfolio weights.

    Parameters
    ----------
    n_assets:
        Number of assets

    Returns
    -------
    np.ndarray
        Portfolio weights summing to 1
    """

    weights = np.random.random(n_assets)

    weights /= np.sum(weights)

    return weights

def random_weights_constrained(
    n_assets,
    max_weight=0.30
):
    while True:
        weights = np.random.random(n_assets)
        weights /= weights.sum()

        if np.all(weights <= max_weight):
            return weights

def portfolio_sharpe(
    weights,
    expected_returns,
    covariance_matrix,
    risk_free_rate=0.0
):
    portfolio_return = weights @ expected_returns

    portfolio_volatility = np.sqrt(
        weights.T
        @ covariance_matrix
        @ weights
    )

    return (
        portfolio_return - risk_free_rate
    ) / portfolio_volatility

def simulate_portfolio(
    weights,
    expected_returns,
    covariance_matrix
):

    portfolio_return = (
        weights @ expected_returns
    )

    portfolio_volatility = np.sqrt(
        weights.T
        @ covariance_matrix
        @ weights
    )

    sharpe = (
        portfolio_return /
        portfolio_volatility
    )

    return (
        portfolio_return,
        portfolio_volatility,
        sharpe
    )

def negative_sharpe(
    weights,
    expected_returns,
    covariance_matrix,
    risk_free_rate=0.0
):
    return -portfolio_sharpe(
        weights,
        expected_returns,
        covariance_matrix,
        risk_free_rate
    )

def optimize_max_sharpe(
    expected_returns,
    covariance_matrix,
    risk_free_rate=0.0,
    max_weight=0.30
):
    """
    Find the portfolio weights that maximize the Sharpe ratio.

    Constraints:
    - Weights sum to 1
    - No short selling
    - Maximum position size is max_weight
    """

    n_assets = len(expected_returns)

    # Start from an equally weighted portfolio
    initial_weights = np.ones(n_assets) / n_assets

    # Every asset must lie between 0% and max_weight
    bounds = tuple(
        (0.0, max_weight)
        for _ in range(n_assets)
    )

    # Portfolio weights must sum to 100%
    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }

    result = minimize(
        negative_sharpe,
        initial_weights,
        args=(
            expected_returns,
            covariance_matrix,
            risk_free_rate
        ),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return result

def optimize_min_volatility(
    covariance_matrix,
    max_weight=0.30
):
    n_assets = len(covariance_matrix)

    initial_weights = np.ones(n_assets) / n_assets

    bounds = tuple(
        (0.0, max_weight)
        for _ in range(n_assets)
    )

    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }

    result = minimize(
        portfolio_volatility,
        initial_weights,
        args=(covariance_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return result

def optimize_for_target_return(
    expected_returns,
    covariance_matrix,
    target_return,
    max_weight=0.30
):
    n_assets = len(expected_returns)

    initial_weights = np.ones(n_assets) / n_assets

    bounds = tuple(
        (0.0, max_weight)
        for _ in range(n_assets)
    )

    constraints = (
        {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1
        },
        {
            "type": "eq",
            "fun": lambda weights: (
                weights @ expected_returns
                - target_return
            )
        }
    )

    result = minimize(
        portfolio_volatility,
        initial_weights,
        args=(covariance_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return result
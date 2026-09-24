import pandas as pd
from analytics.optimization import optimize_max_sharpe
import numpy as np
from analytics.estimation import shrink_expected_returns
from analytics.estimation import (
    sample_covariance,
    ledoit_wolf_covariance
)
TRADING_DAYS = 252


def walk_forward_backtest(
    returns,
    train_window=252,
    rebalance_frequency=21,
    max_weight=0.30,
    covariance_method="sample",
    expected_return_method="historical",
    transaction_cost_bps=0.0,
    evaluation_start=None
):
    returns, first_test_position = _prepare_returns(
        returns, train_window, rebalance_frequency,
        transaction_cost_bps, evaluation_start
    )

    portfolio_returns = []
    weights_history = []
    turnover_history = []
    previous_end_weights = None
    cost_rate = transaction_cost_bps / 10_000

    for start in range(
        first_test_position,
        len(returns),
        rebalance_frequency
    ):
        # Everything known BEFORE this rebalance
        train = returns.iloc[
            start - train_window:start
        ]

        # The future period we're about to invest in
        test = returns.iloc[
            start:start + rebalance_frequency
        ]

        if test.empty:
            break

        # Estimate parameters using training data only
        if expected_return_method == "historical":
            expected_returns = train.mean() * TRADING_DAYS

        elif expected_return_method == "shrunk":
            expected_returns, _, _ = shrink_expected_returns(
                train
            )

        else:
            raise ValueError(
                f"Unknown expected return method: "
                f"{expected_return_method}"
            )

        if covariance_method == "sample":
            covariance = sample_covariance(train)

        elif covariance_method == "ledoit_wolf":
            covariance, _ = ledoit_wolf_covariance(train)

        else:
            raise ValueError(
                f"Unknown covariance method: {covariance_method}"
            )

        # Construct portfolio
        result = optimize_max_sharpe(
            expected_returns,
            covariance,
            risk_free_rate=0.0,
            max_weight=max_weight
        )

        if not result.success:
            raise RuntimeError(
                f"Optimization failed at {test.index[0]}: {result.message}"
            )

        weights = result.x

        if previous_end_weights is not None:
            turnover = (
                np.abs(weights - previous_end_weights).sum()/ 2
                )
            
            turnover_history.append(turnover)

        # Apply those fixed weights to FUTURE returns
        realized_returns = holding_period_returns(
            test,
            weights
        )
        if previous_end_weights is not None:
            transaction_cost = turnover * cost_rate

            realized_returns.iloc[0] -= transaction_cost
        portfolio_returns.append(
            realized_returns
        )

        weights_history.append(
            pd.Series(
                weights,
                index=returns.columns,
                name=test.index[0]
            )
        )
        asset_growth = (1 + test).prod()

        end_values = (
            weights * asset_growth.values
        )

        previous_end_weights = (
            end_values / end_values.sum()
        )

    portfolio_returns = pd.concat(
        portfolio_returns
    )

    weights_history = pd.DataFrame(
        weights_history
    )

    return portfolio_returns, weights_history, turnover_history

def equal_weight_backtest(
    returns,
    train_window=252,
    rebalance_frequency=21,
    transaction_cost_bps=0.0,
    evaluation_start=None
):
    returns, first_test_position = _prepare_returns(
        returns, train_window, rebalance_frequency,
        transaction_cost_bps, evaluation_start
    )

    portfolio_returns = []
    weights_history = []
    turnover_history = []
    cost_rate = transaction_cost_bps / 10_000
    n_assets = len(returns.columns)

    # Equal-weight target: e.g. 5 assets -> 20% each
    weights = np.ones(n_assets) / n_assets

    # Before the first investment there is no existing portfolio
    previous_end_weights = None

    for start in range(
        first_test_position,
        len(returns),
        rebalance_frequency
    ):
        test = returns.iloc[
            start:start + rebalance_frequency
        ]

        if test.empty:
            break

        # 1. Target weights are always equal
        weights = np.ones(n_assets) / n_assets

        # 2. Calculate turnover required to rebalance
        #    from last month's drifted weights
        if previous_end_weights is not None:
            turnover = (
                np.abs(
                    weights - previous_end_weights
                ).sum() / 2
            )

            turnover_history.append(turnover)

        # 3. Hold these weights during the next period
        realized_returns = holding_period_returns(
            test,
            weights
        )
        if previous_end_weights is not None:
            transaction_cost = turnover * cost_rate
            realized_returns.iloc[0] -= transaction_cost
        portfolio_returns.append(realized_returns)

        weights_history.append(
            pd.Series(
                weights,
                index=returns.columns,
                name=test.index[0]
            )
        )

        # 4. Calculate how the weights have drifted
        #    by the end of the holding period
        asset_growth = (1 + test).prod()

        end_values = (
            weights * asset_growth.values
        )

        previous_end_weights = (
            end_values / end_values.sum()
        )

    return (
        pd.concat(portfolio_returns),
        pd.DataFrame(weights_history),
        turnover_history
    )

def holding_period_returns(test_returns, initial_weights):
    asset_growth = (1 + test_returns).cumprod()

    portfolio_value = asset_growth @ initial_weights

    portfolio_returns = portfolio_value.pct_change()

    portfolio_returns.iloc[0] = (
        portfolio_value.iloc[0] - 1
    )

    return portfolio_returns


def _prepare_returns(returns, train_window, rebalance_frequency,
                     transaction_cost_bps, evaluation_start):
    """Validate daily simple returns and select a common out-of-sample start."""
    if not isinstance(returns, pd.DataFrame) or returns.shape[1] == 0:
        raise ValueError("returns must be a DataFrame with asset columns")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns must have a DatetimeIndex")
    if not returns.index.is_monotonic_increasing or not returns.index.is_unique:
        raise ValueError("return dates must be sorted and unique")
    if returns.index.hasnans or not returns.columns.is_unique:
        raise ValueError("dates must be valid and asset columns unique")
    for name, value, minimum in (("train_window", train_window, 2),
                                 ("rebalance_frequency", rebalance_frequency, 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not np.isfinite(transaction_cost_bps) or not 0 <= transaction_cost_bps < 10_000:
        raise ValueError("transaction_cost_bps must be finite and in [0, 10000)")
    returns = returns.dropna()
    if not np.isfinite(returns.to_numpy()).all() or (returns <= -1).any().any():
        raise ValueError("returns must be finite and greater than -1")
    if evaluation_start is None:
        first = train_window
    else:
        date = pd.Timestamp(evaluation_start)
        if pd.isna(date):
            raise ValueError("evaluation_start must be a valid date")
        first = int(returns.index.searchsorted(date))
    if first < train_window:
        raise ValueError("Not enough historical data before evaluation_start for the requested train_window")
    if first >= len(returns):
        raise ValueError("No evaluation observations after the training window/evaluation_start")
    return returns, first

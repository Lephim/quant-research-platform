from dataclasses import dataclass

import pandas as pd

from backtesting.walk_forward import (
    walk_forward_backtest,
    equal_weight_backtest,
)
from backtesting.metrics import performance_metrics

@dataclass
class ExperimentResult:
    returns: pd.Series
    weights: pd.DataFrame
    turnover: list[float]
    metrics: dict[str, float]

def run_strategy_experiment(
returns: pd.DataFrame,
train_window: int = 252,
rebalance_frequency: int = 21,
max_weight: float = 0.30,
covariance_method: str = "sample",
expected_return_method: str = "historical",
transaction_cost_bps: float = 0.0,
evaluation_start=None,
) -> ExperimentResult:

    strategy_returns, weights, turnover = walk_forward_backtest(
        returns=returns,
        train_window=train_window,
        rebalance_frequency=rebalance_frequency,
        max_weight=max_weight,
        covariance_method=covariance_method,
        expected_return_method=expected_return_method,
        transaction_cost_bps=transaction_cost_bps,
        evaluation_start=evaluation_start,
    )

    metrics = performance_metrics(strategy_returns)

    years = len(strategy_returns) / 252

    metrics["Average Turnover"] = (
        sum(turnover) / len(turnover)
        if turnover
        else 0.0
    )

    metrics["Annualized Turnover"] = (
        sum(turnover) / years
        if years > 0
        else 0.0
    )

    return ExperimentResult(
        returns=strategy_returns,
        weights=weights,
        turnover=turnover,
        metrics=metrics,
    )

def run_equal_weight_experiment(
    returns: pd.DataFrame,
    train_window: int = 252,
    rebalance_frequency: int = 21,
    transaction_cost_bps: float = 0.0,
    evaluation_start=None,
) -> ExperimentResult:

    strategy_returns, weights, turnover = equal_weight_backtest(
        returns=returns,
        train_window=train_window,
        rebalance_frequency=rebalance_frequency,
        transaction_cost_bps=transaction_cost_bps,
        evaluation_start=evaluation_start,
    )

    metrics = performance_metrics(strategy_returns)

    years = len(strategy_returns) / 252

    metrics["Average Turnover"] = (
        sum(turnover) / len(turnover)
        if turnover
        else 0.0
    )

    metrics["Annualized Turnover"] = (
        sum(turnover) / years
        if years > 0
        else 0.0
    )

    return ExperimentResult(
        returns=strategy_returns,
        weights=weights,
        turnover=turnover,
        metrics=metrics,
    )

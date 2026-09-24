import numpy as np

from analytics.risk import (
    annualized_volatility,
    sharpe_ratio,
    maximum_drawdown
)

TRADING_DAYS = 252


def performance_metrics(returns, risk_free_rate=0.0):
    returns = returns.dropna()

    n_days = len(returns)
    years = n_days / TRADING_DAYS

    total_return = (1 + returns).prod() - 1

    cagr = (1 + total_return) ** (1 / years) - 1

    volatility = annualized_volatility(returns)

    sharpe = sharpe_ratio(
        returns,
        risk_free_rate=risk_free_rate
    )

    wealth = (1 + returns).cumprod()
    max_drawdown = maximum_drawdown(wealth)

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown
    }
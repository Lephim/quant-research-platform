import numpy as np

from analytics.risk import (
    annualized_volatility,
    sharpe_ratio
)

TRADING_DAYS = 252


def performance_metrics(returns, risk_free_rate=0.0):
    returns = returns.dropna()

    if returns.empty:
        raise ValueError("returns must contain at least one observation")

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
    # Include starting wealth: a loss on the first day is a drawdown too.
    max_drawdown = (wealth / wealth.cummax().clip(lower=1.0) - 1).min()

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown
    }
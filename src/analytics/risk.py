import numpy as np


def annualized_volatility(returns):
    """
    Calculate annualized volatility.

    Assumes 252 trading days/year.
    """

    volatility = returns.std()

    annualized = volatility * np.sqrt(252)

    return annualized
def sharpe_ratio(
    returns,
    risk_free_rate=0
):
    """
    Calculate Sharpe ratio.
    """

    excess_returns = returns - risk_free_rate / 252

    return (
        excess_returns.mean()
        /
        excess_returns.std()
        *
        np.sqrt(252)
    )


def maximum_drawdown(prices):

    cumulative = prices / prices.iloc[0]

    running_max = cumulative.cummax()

    drawdown = (
        cumulative - running_max
    ) / running_max

    return drawdown.min()
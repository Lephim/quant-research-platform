from analytics import returns
import numpy as np
import pandas as pd

TRADING_DAYS = 252
def annualized_volatility(returns):
    """
    Calculate annualized volatility.

    Assumes 252 trading days/year.
    """

    return returns.std() * np.sqrt(TRADING_DAYS)

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

def rolling_volatility(
    returns,
    window=30
):
    """
    Calculate rolling annualized volatility.
    """

    rolling_vol = (
        returns
        .rolling(window)
        .std()
        *
        np.sqrt(TRADING_DAYS)
    )

    return rolling_vol
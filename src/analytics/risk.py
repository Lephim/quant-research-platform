import numpy as np


def annualized_volatility(returns):
    """
    Calculate annualized volatility.

    Assumes 252 trading days/year.
    """

    volatility = returns.std()

    annualized = volatility * np.sqrt(252)

    return annualized
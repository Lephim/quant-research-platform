import pandas as pd
import numpy as np


def calculate_simple_returns(prices):
    """
    Calculate daily percentage returns.

    Parameters:
        prices: pandas Series or DataFrame

    Returns:
        pandas Series or DataFrame
    """

    returns = prices.pct_change(fill_method=None)

    return returns

def calculate_log_returns(prices):
    """
    Calculate logarithmic returns.
    """

    log_returns = np.log(
        prices / prices.shift(1)
    )

    return log_returns
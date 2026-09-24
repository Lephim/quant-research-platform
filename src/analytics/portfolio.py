import pandas as pd
import numpy as np


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the correlation matrix of asset returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns for multiple assets.
        Columns are asset tickers.

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """

    return returns.corr()

def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the covariance matrix of asset returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns.

    Returns
    -------
    pd.DataFrame
        Covariance matrix.
    """

    return returns.cov()

def portfolio_return(weights, expected_returns):
    """
    Calculate expected portfolio return.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.

    expected_returns : np.ndarray
        Expected annual returns for each asset.

    Returns
    -------
    float
        Expected portfolio return.
    """

    return np.dot(weights, expected_returns)

def portfolio_volatility(weights, covariance_matrix):
    """
    Calculate annual portfolio volatility.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.

    covariance_matrix : pd.DataFrame
        Annual covariance matrix.

    Returns
    -------
    float
        Annual portfolio volatility.
    """

    variance = (
        weights.T
        @ covariance_matrix
        @ weights
    )

    return np.sqrt(variance)


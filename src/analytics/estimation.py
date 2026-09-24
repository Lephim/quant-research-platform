import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf


TRADING_DAYS = 252


def sample_covariance(
    returns,
    annualize=True
):
    """
    Estimate covariance using the standard
    sample covariance matrix.
    """
    covariance = returns.cov()

    if annualize:
        covariance *= TRADING_DAYS

    return covariance


def ledoit_wolf_covariance(
    returns,
    annualize=True
):
    """
    Estimate covariance using Ledoit-Wolf shrinkage.
    """
    clean_returns = returns.dropna()

    model = LedoitWolf()
    model.fit(clean_returns)

    covariance = pd.DataFrame(
        model.covariance_,
        index=clean_returns.columns,
        columns=clean_returns.columns
    )

    if annualize:
        covariance *= TRADING_DAYS

    return covariance, model.shrinkage_

def shrink_expected_returns(
    returns,
    annualize=True
):
    clean_returns = returns.dropna()

    # Historical mean return
    mu_hat = clean_returns.mean()

    # Uncertainty of each estimated mean
    standard_errors = (
        clean_returns.std()
        / np.sqrt(len(clean_returns))
    )

    # Common target: average estimated return
    mu_0 = mu_hat.mean()

    # How spread out the observed means are
    observed_variance = mu_hat.var()

    # Average amount of that spread we expect
    # purely from estimation noise
    noise_variance = (standard_errors ** 2).mean()

    # Estimate genuine cross-asset variation
    tau_squared = max(
        observed_variance - noise_variance,
        0.0
    )

    # How much to trust each asset's own estimate
    denominator = tau_squared + standard_errors ** 2
    weights = (tau_squared / denominator.replace(0, np.nan)).fillna(0.0)

    # Shrink toward common mean
    shrunk_mu = (
        weights * mu_hat
        + (1 - weights) * mu_0
    )

    if annualize:
        mu_hat = mu_hat * 252
        standard_errors = standard_errors * 252
        mu_0 = mu_0 * 252
        shrunk_mu = shrunk_mu * 252

    return shrunk_mu, weights, mu_0
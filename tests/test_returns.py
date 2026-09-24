import numpy as np
import pandas as pd
import pytest
from analytics.returns import calculate_simple_returns, calculate_log_returns


def test_simple_returns():
    result = calculate_simple_returns(pd.Series([100., 110., 99.]))
    assert np.isnan(result.iloc[0])
    assert result.iloc[1:].to_list() == pytest.approx([.1, -.1])


def test_missing_prices_are_not_filled():
    result = calculate_simple_returns(pd.Series([100., np.nan, 110.]))
    assert result.isna().all()


def test_log_returns_compound():
    result = calculate_log_returns(pd.Series([100., 110., 121.]))
    assert result.sum() == pytest.approx(np.log(1.21))

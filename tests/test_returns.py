from analytics.returns import calculate_simple_returns


def test_returns():

    prices = [100, 110]

    returns = calculate_simple_returns(prices)

    assert returns.iloc[1] == 0.1
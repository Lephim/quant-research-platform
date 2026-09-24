import yfinance as yf


def load_market_data(
    ticker,
    start_date,
    end_date
):
    """
    Download historical market data.
    """

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date
    )

    if isinstance(data.columns, tuple):
        data.columns = data.columns.droplevel(1)

    return data
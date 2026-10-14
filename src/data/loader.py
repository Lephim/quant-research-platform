import pandas as pd
import yfinance as yf


def load_market_data(
    tickers,
    start_date,
    end_date
):
    """
    Download historical market data for one or more tickers.
    """
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True
    )

    return data
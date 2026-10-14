import pandas as pd
from data import loader


def test_loader_passes_universe_and_dates(monkeypatch):
    expected = pd.DataFrame({"Close": [100., 101.]})
    calls = {}
    def download(ticker, **kwargs):
        calls.update(ticker=ticker, **kwargs)
        return expected
    monkeypatch.setattr(loader.yf, "download", download)
    actual = loader.load_market_data(["AAPL", "MSFT"], "2020-01-01", "2021-01-01")
    assert actual is expected
    assert calls == {"ticker": ["AAPL", "MSFT"], "start": "2020-01-01",
                     "end": "2021-01-01", "auto_adjust": True}

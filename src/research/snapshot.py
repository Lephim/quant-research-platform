"""Immutable, checksum-verified local price snapshots for research runs."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import yfinance as yf

from data.loader import load_market_data
from research.universes import UNIVERSES, START_DATE, END_DATE


def _validate_prices(prices, tickers):
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("Price snapshot is empty")
    if list(prices.columns) != list(tickers):
        raise ValueError("Price columns differ from the declared ticker order")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Price dates must be a DatetimeIndex")
    if prices.index.hasnans or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be valid, unique, and sorted")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Prices must be finite and positive with no missing observations")


def capture_snapshot(path, *, universe="multi_asset", start_date=START_DATE,
                     end_date=END_DATE):
    """Download adjusted close prices once and write CSV plus provenance JSON."""
    path = Path(path)
    manifest_path = path.with_suffix(path.suffix + ".json")
    if path.exists() or manifest_path.exists():
        raise FileExistsError("Snapshot or manifest already exists; choose a new filename")
    if universe not in UNIVERSES:
        raise ValueError(f"Unknown universe: {universe}")
    tickers = UNIVERSES[universe]
    downloaded_at = datetime.now(timezone.utc).isoformat()
    market_data = load_market_data(tickers, start_date, end_date)
    prices = market_data["Close"].reindex(columns=tickers)
    _validate_prices(prices, tickers)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index_label="Date", float_format="%.17g", date_format="%Y-%m-%d")
    digest = sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "universe": universe,
        "tickers": tickers,
        "requested_start": start_date,
        "requested_end_exclusive": end_date,
        "downloaded_at_utc": downloaded_at,
        "provider": "yfinance",
        "provider_version": yf.__version__,
        "auto_adjust": True,
        "price_field": "Close",
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "rows": len(prices),
        "first_date": prices.index[0].date().isoformat(),
        "last_date": prices.index[-1].date().isoformat(),
        "csv_sha256": digest,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def load_snapshot(path):
    """Check the manifest and byte-level checksum before parsing prices."""
    path = Path(path)
    manifest = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported snapshot manifest version")
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != manifest.get("csv_sha256"):
        raise ValueError("Snapshot checksum mismatch")
    if manifest.get("universe") not in UNIVERSES or manifest.get("tickers") != UNIVERSES[manifest["universe"]]:
        raise ValueError("Snapshot universe or ticker list does not match the research definition")
    prices = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    _validate_prices(prices, manifest["tickers"])
    if (len(prices) != manifest["rows"] or
            prices.index[0].date().isoformat() != manifest["first_date"] or
            prices.index[-1].date().isoformat() != manifest["last_date"]):
        raise ValueError("Snapshot dates or row count differ from the manifest")
    return prices, manifest

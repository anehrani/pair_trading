from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PricePanel:
    closes: pd.DataFrame  # index: datetime (UTC), columns: symbols, values: close


def read_binance_klines_csv(path: Path) -> pd.Series:
    """Reads a CSV written by src/download_binance_futures.py and returns close series.

    Index is UTC timestamps for candle open times.
    """
    df = pd.read_csv(path)
    # open_time_utc is ISO8601 with timezone; parse robustly.
    ts = pd.to_datetime(df["open_time_utc"], utc=True)
    close = pd.to_numeric(df["close"], errors="coerce")
    s = pd.Series(close.values, index=ts, name="close").sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def load_closes_from_dir(directory: Path, *, interval: str) -> PricePanel:
    """Loads all *_<interval>.csv files in a directory into a single aligned DataFrame."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Data directory not found: {directory}")

    series_by_symbol: dict[str, pd.Series] = {}
    for csv_path in sorted(directory.glob(f"*_{interval}.csv")):
        name = csv_path.name
        symbol = name[: -len(f"_{interval}.csv")]
        s = read_binance_klines_csv(csv_path)
        series_by_symbol[symbol] = s

    if not series_by_symbol:
        raise ValueError(f"No files matching '*_{interval}.csv' in {directory}")

    df = pd.concat(series_by_symbol, axis=1)
    df.columns = list(series_by_symbol.keys())

    # Force UTC timezone index, sort.
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    return PricePanel(closes=df)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)

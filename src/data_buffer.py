"""Rolling window data buffer for live trading.

Maintains recent price history and provides slicing for formation/trading windows.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger


class DataBuffer:
    """In-memory buffer for rolling window price data."""

    def __init__(self, symbols: list[str], max_days: int = 30):
        """Initialize data buffer.

        Args:
            symbols: List of trading symbols to track
            max_days: Maximum days of history to keep in memory
        """
        self.symbols = symbols
        self.max_days = max_days
        self.data: dict[str, pd.DataFrame] = {sym: pd.DataFrame() for sym in symbols}

    def update(self, symbol: str, new_candles: pd.DataFrame) -> None:
        """Add new candles to buffer.

        Args:
            symbol: Trading symbol
            new_candles: DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        if symbol not in self.data:
            logger.warning(f"Symbol {symbol} not in buffer, adding it")
            self.data[symbol] = pd.DataFrame()

        # Append new data
        self.data[symbol] = pd.concat([self.data[symbol], new_candles], ignore_index=True)

        # Remove duplicates
        self.data[symbol] = (
            self.data[symbol].drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        )

        # Trim to max_days
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=self.max_days)
        self.data[symbol] = self.data[symbol][self.data[symbol]["timestamp"] >= cutoff]

        logger.debug(f"Updated {symbol}: {len(self.data[symbol])} candles")

    def get_closes(self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
        """Get close prices for all symbols in a time range.

        Args:
            start: Start timestamp (inclusive)
            end: End timestamp (inclusive)

        Returns:
            DataFrame with timestamp index and symbol columns
        """
        closes_dict = {}

        for sym in self.symbols:
            df = self.data[sym].copy()

            if start:
                df = df[df["timestamp"] >= start]
            if end:
                df = df[df["timestamp"] <= end]

            if not df.empty:
                closes_dict[sym] = df.set_index("timestamp")["close"]

        if not closes_dict:
            return pd.DataFrame()

        # Combine into single DataFrame
        result = pd.DataFrame(closes_dict)
        return result

    def get_latest_prices(self) -> dict[str, float]:
        """Get most recent close price for each symbol.

        Returns:
            Dict mapping symbol to latest price
        """
        prices = {}
        for sym in self.symbols:
            if not self.data[sym].empty:
                prices[sym] = float(self.data[sym].iloc[-1]["close"])
        return prices

    def get_data_range(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """Get the earliest and latest timestamps across all symbols.

        Returns:
            (earliest, latest) timestamps
        """
        earliest = None
        latest = None

        for sym in self.symbols:
            if not self.data[sym].empty:
                sym_earliest = self.data[sym]["timestamp"].min()
                sym_latest = self.data[sym]["timestamp"].max()

                if earliest is None or sym_earliest < earliest:
                    earliest = sym_earliest
                if latest is None or sym_latest > latest:
                    latest = sym_latest

        return earliest, latest

    def save(self, path: Path) -> None:
        """Save buffer to disk.

        Args:
            path: Path to save file (parquet format)
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Combine all symbols into one DataFrame
        all_data = []
        for sym, df in self.data.items():
            if not df.empty:
                df_copy = df.copy()
                df_copy["symbol"] = sym
                all_data.append(df_copy)

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined.to_parquet(path, index=False)
            logger.info(f"Saved buffer to {path}")
        else:
            logger.warning("No data to save")

    def load(self, path: Path) -> None:
        """Load buffer from disk.

        Args:
            path: Path to saved file
        """
        if not path.exists():
            logger.warning(f"Buffer file {path} does not exist")
            return

        combined = pd.read_parquet(path)

        # Split by symbol
        for sym in combined["symbol"].unique():
            sym_data = combined[combined["symbol"] == sym].drop(columns=["symbol"]).reset_index(drop=True)
            self.data[sym] = sym_data

        logger.info(f"Loaded buffer from {path}")

    def is_ready(self, required_days: int) -> bool:
        """Check if buffer has enough data for trading.

        Args:
            required_days: Minimum days of data needed

        Returns:
            True if all symbols have at least required_days of data
        """
        earliest, latest = self.get_data_range()

        if earliest is None or latest is None:
            return False

        days_available = (latest - earliest).total_seconds() / (24 * 3600)
        return days_available >= required_days

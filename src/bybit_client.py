"""Bybit API client for live trading.

Handles market data streaming, order execution, and position management
for Bybit USDT-margined perpetual futures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from loguru import logger
from pybit.unified_trading import HTTP, WebSocket


@dataclass
class Position:
    """Current position information."""

    symbol: str
    side: str  # "Buy" or "Sell"
    size: float
    entry_price: float
    unrealized_pnl: float
    leverage: float


@dataclass
class OrderResult:
    """Result of order placement."""

    order_id: str
    symbol: str
    side: str
    qty: float
    price: float | None
    status: str


class BybitClient:
    """Wrapper for Bybit API interactions."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
    ):
        """Initialize Bybit client.

        Args:
            api_key: Bybit API key
            api_secret: Bybit API secret
            testnet: If True, use testnet; otherwise use mainnet
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

        # HTTP client for REST API
        self.session = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret,
        )

        logger.info(f"Initialized Bybit client (testnet={testnet})")

    def get_account_balance(self) -> dict[str, float]:
        """Get USDT account balance.

        Returns:
            Dict with 'total', 'available', 'used' balances in USDT
        """
        try:
            result = self.session.get_wallet_balance(accountType="UNIFIED")
            if result["retCode"] != 0:
                raise ValueError(f"API error: {result['retMsg']}")

            # Extract USDT balance
            for coin in result["result"]["list"][0]["coin"]:
                if coin["coin"] == "USDT":
                    return {
                        "total": float(coin["walletBalance"]),
                        "available": float(coin["availableToWithdraw"]),
                        "used": float(coin["walletBalance"]) - float(coin["availableToWithdraw"]),
                    }

            return {"total": 0.0, "available": 0.0, "used": 0.0}

        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            raise

    def get_klines(
        self,
        symbol: str,
        interval: str = "5",
        limit: int = 200,
        start_time: int | None = None,
    ) -> pd.DataFrame:
        """Fetch historical kline (candlestick) data.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval in minutes ("1", "5", "15", "60", etc.)
            limit: Number of candles to fetch (max 200 per request)
            start_time: Start timestamp in milliseconds (optional)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            params: dict[str, Any] = {
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
            }
            if start_time:
                params["start"] = start_time

            result = self.session.get_kline(**params)

            if result["retCode"] != 0:
                raise ValueError(f"API error: {result['retMsg']}")

            # Parse kline data
            klines = result["result"]["list"]
            df = pd.DataFrame(
                klines,
                columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
            )

            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            # Bybit returns newest first, so reverse
            df = df.sort_values("timestamp").reset_index(drop=True)
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]

            return df

        except Exception as e:
            logger.error(f"Failed to get klines for {symbol}: {e}")
            raise

    def get_all_klines(
        self,
        symbol: str,
        interval: str = "5",
        days: int = 30,
    ) -> pd.DataFrame:
        """Fetch historical klines for multiple days (handles pagination).

        Args:
            symbol: Trading pair
            interval: Kline interval in minutes
            days: Number of days of history to fetch

        Returns:
            DataFrame with all klines
        """
        interval_minutes = int(interval)
        candles_per_day = (24 * 60) // interval_minutes
        total_candles = days * candles_per_day

        all_data = []
        end_time = int(time.time() * 1000)  # Current time in ms

        while len(all_data) < total_candles:
            # Fetch batch
            df = self.get_klines(symbol, interval, limit=200, start_time=end_time - 200 * interval_minutes * 60 * 1000)

            if df.empty:
                break

            all_data.append(df)

            # Move end_time backwards
            end_time = int(df["timestamp"].iloc[0].timestamp() * 1000) - 1

            # Avoid rate limits
            time.sleep(0.1)

            if len(all_data) * 200 >= total_candles:
                break

        if not all_data:
            return pd.DataFrame()

        # Combine and deduplicate
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        # Trim to requested days
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        result = result[result["timestamp"] >= cutoff]

        return result

    def place_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
    ) -> OrderResult:
        """Place a market order.

        Args:
            symbol: Trading pair
            side: "Buy" or "Sell"
            qty: Order quantity (positive number)

        Returns:
            OrderResult with order details
        """
        try:
            result = self.session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(abs(qty)),
                timeInForce="GTC",
            )

            if result["retCode"] != 0:
                raise ValueError(f"Order failed: {result['retMsg']}")

            order_data = result["result"]
            logger.info(f"Placed {side} market order: {symbol} qty={qty} orderId={order_data['orderId']}")

            return OrderResult(
                order_id=order_data["orderId"],
                symbol=symbol,
                side=side,
                qty=qty,
                price=None,  # Market order, filled at market price
                status="Submitted",
            )

        except Exception as e:
            logger.error(f"Failed to place order {symbol} {side} {qty}: {e}")
            raise

    def get_positions(self) -> list[Position]:
        """Get all open positions.

        Returns:
            List of Position objects
        """
        try:
            result = self.session.get_positions(category="linear", settleCoin="USDT")

            if result["retCode"] != 0:
                raise ValueError(f"API error: {result['retMsg']}")

            positions = []
            for pos in result["result"]["list"]:
                size = float(pos["size"])
                if size == 0:
                    continue  # Skip closed positions

                positions.append(
                    Position(
                        symbol=pos["symbol"],
                        side=pos["side"],
                        size=size,
                        entry_price=float(pos["avgPrice"]),
                        unrealized_pnl=float(pos["unrealisedPnl"]),
                        leverage=float(pos["leverage"]),
                    )
                )

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise

    def close_position(self, symbol: str) -> OrderResult | None:
        """Close an open position by placing opposite market order.

        Args:
            symbol: Trading pair

        Returns:
            OrderResult if position was closed, None if no position
        """
        positions = self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                # Place opposite order
                opposite_side = "Sell" if pos.side == "Buy" else "Buy"
                return self.place_market_order(symbol, opposite_side, pos.size)

        logger.warning(f"No open position found for {symbol}")
        return None

    def get_latest_price(self, symbol: str) -> float:
        """Get latest market price for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Latest price
        """
        try:
            result = self.session.get_tickers(category="linear", symbol=symbol)

            if result["retCode"] != 0:
                raise ValueError(f"API error: {result['retMsg']}")

            return float(result["result"]["list"][0]["lastPrice"])

        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            raise

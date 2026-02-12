"""OKX API client for live trading.

Handles market data streaming, order execution, and position management
for OKX USDT-margined perpetual futures.
"""

from __future__ import annotations

import base64
import hmac
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
import requests
from loguru import logger
import pandas as pd


@dataclass
class Position:
    """Current position information."""

    symbol: str
    side: str  # "long" or "short"
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


class OKXClient:
    """Wrapper for OKX API interactions using direct HTTP requests."""

    BASE_URL = "https://www.okx.com"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        demo: bool = True,
    ):
        """Initialize OKX client.

        Args:
            api_key: OKX API key
            api_secret: OKX API secret
            passphrase: OKX API passphrase
            demo: If True, use demo trading; otherwise use live
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.demo = demo

        logger.info(f"Initialized OKX client (demo={demo})")

    # ------------------------------------------------------------------ #
    #  Authentication helpers
    # ------------------------------------------------------------------ #
    def _get_timestamp(self) -> str:
        """Generate ISO 8601 timestamp."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def _sign(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """Create HMAC SHA256 signature required by OKX API."""
        message = timestamp + method.upper() + request_path + body
        mac = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        )
        return base64.b64encode(mac.digest()).decode("utf-8")

    def _headers(self, method: str, request_path: str, body: str = "") -> dict:
        """Build authenticated request headers."""
        timestamp = self._get_timestamp()
        signature = self._sign(timestamp, method, request_path, body)

        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }

        if self.demo:
            headers["x-simulated-trading"] = "1"

        return headers

    def _request(self, method: str, path: str, params: dict = None, data: dict = None) -> dict:
        """Send authenticated request."""
        try:
            if params:
                query = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
                request_path = f"{path}?{query}" if query else path
            else:
                request_path = path

            body = json.dumps(data) if data else ""
            headers = self._headers(method, request_path, body)
            url = self.BASE_URL + request_path

            if method == "GET":
                response = requests.get(url, headers=headers)
            else:
                response = requests.post(url, headers=headers, data=body)

            result = response.json()
            
            if str(result.get("code")) != "0":
                raise ValueError(f"API Error {result.get('code')}: {result.get('msg')}")
                
            return result

        except Exception as e:
            logger.error(f"Request failed: {method} {path} - {e}")
            raise

    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float, handling empty strings."""
        if not value:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    # ------------------------------------------------------------------ #
    #  Public Interface
    # ------------------------------------------------------------------ #

    def get_account_balance(self) -> dict[str, float]:
        """Get USDT account balance.

        Returns:
            Dict with 'total', 'available', 'used' balances in USDT
        """
        try:
            result = self._request("GET", "/api/v5/account/balance", {"ccy": "USDT"})
            
            if not result.get("data"):
                return {"total": 0.0, "available": 0.0, "used": 0.0}

            details = result["data"][0].get("details", [])
            for detail in details:
                if detail["ccy"] == "USDT":
                    total = self._safe_float(detail.get("eq"))
                    available = self._safe_float(detail.get("availEq"))
                    return {
                        "total": total,
                        "available": available,
                        "used": total - available,
                    }

            return {"total": 0.0, "available": 0.0, "used": 0.0}

        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            raise

    def get_klines(
        self,
        symbol: str,
        interval: str = "5m",
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch historical kline (candlestick) data."""
        try:
            result = self._request("GET", "/api/v5/market/candles", {
                "instId": symbol,
                "bar": interval,
                "limit": str(limit)
            })

            klines = result.get("data", [])
            if not klines:
                return pd.DataFrame()

            df = pd.DataFrame(
                klines,
                columns=["timestamp", "open", "high", "low", "close", "volume", "volCcy", "volCcyQuote", "confirm"],
            )

            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].apply(self._safe_float)

            # OKX returns newest first, so reverse
            df = df.sort_values("timestamp").reset_index(drop=True)
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]

            return df

        except Exception as e:
            logger.error(f"Failed to get klines for {symbol}: {e}")
            raise

    def get_all_klines(
        self,
        symbol: str,
        interval: str = "5m",
        days: int = 30,
    ) -> pd.DataFrame:
        """Fetch historical klines for multiple days (handles pagination)."""
        interval_map = {"1m": 1, "5m": 5, "15m": 15, "1H": 60, "4H": 240, "1D": 1440}
        interval_minutes = interval_map.get(interval, 5)
        candles_per_day = (24 * 60) // interval_minutes
        total_candles = days * candles_per_day

        all_data = []
        iterations = min((total_candles // 100) + 1, 50)  # Max 50 calls
        
        last_ts = None
        
        for _ in range(iterations):
            params = {
                "instId": symbol,
                "bar": interval,
                "limit": "100"
            }
            if last_ts:
                params["after"] = last_ts
                
            try:
                result = self._request("GET", "/api/v5/market/candles", params)
                klines = result.get("data", [])
                
                if not klines:
                    break
                    
                df = pd.DataFrame(
                    klines,
                    columns=["timestamp", "open", "high", "low", "close", "volume", "volCcy", "volCcyQuote", "confirm"],
                )
                
                # Update last timestamp for next page (switched to 'after' pagination)
                last_ts = klines[-1][0] 
                
                all_data.append(df)
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error fetching klines: {e}")
                break

        if not all_data:
            return pd.DataFrame()

        # Combine, process, and sort
        combined = pd.concat(all_data, ignore_index=True)
        
        # Convert types
        combined["timestamp"] = pd.to_datetime(combined["timestamp"].astype(int), unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            combined[col] = combined[col].apply(self._safe_float)
            
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        combined = combined[["timestamp", "open", "high", "low", "close", "volume"]]

        # Trim to requested days
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        combined = combined[combined["timestamp"] >= cutoff]

        return combined

    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        try:
            result = self._request("GET", "/api/v5/account/positions")
            
            positions = []
            for pos in result.get("data", []):
                size = self._safe_float(pos.get("pos"))
                if size == 0:
                    continue

                positions.append(
                    Position(
                        symbol=pos["instId"],
                        side="long" if size > 0 else "short",
                        size=abs(size),
                        entry_price=self._safe_float(pos.get("avgPx")),
                        unrealized_pnl=self._safe_float(pos.get("upl")),
                        leverage=self._safe_float(pos.get("lever")) or 1.0,
                    )
                )
            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise

    def place_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
    ) -> OrderResult:
        """Place a market order."""
        try:
            data = {
                "instId": symbol,
                "tdMode": "cross",
                "side": side.lower(),
                "ordType": "market",
                "sz": str(abs(int(qty))),
            }
            # For header mode, need posSide
            # Assuming 'net' mode for simplicity unless specified otherwise in config
            # But OKX often defaults to 'long'/'short' posSide for futures in hedge mode
            # We'll try auto-detection or default to 'net' if possible, but 
            # safe assumption for demo might be to just try simple order first.
            
            # Note: User received 'account mode' error. 
            # Ideally we should check account config.
            
            result = self._request("POST", "/api/v5/trade/order", data=data)
            
            order_data = result["data"][0]
            logger.info(f"Placed {side} market order: {symbol} qty={qty} orderId={order_data.get('ordId')}")

            return OrderResult(
                order_id=order_data.get("ordId", ""),
                symbol=symbol,
                side=side,
                qty=qty,
                price=None,
                status=order_data.get("sCode", ""),
            )

        except Exception as e:
            logger.error(f"Failed to place order {symbol} {side} {qty}: {e}")
            raise

    def close_position(self, symbol: str) -> OrderResult | None:
        """Close an open position."""
        positions = self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                # Use close-position endpoint for safer closing
                try:
                    data = {
                        "instId": symbol,
                        "mgnMode": "cross", 
                    }
                    if pos.side == "long":
                        data["posSide"] = "long"
                    elif pos.side == "short":
                         data["posSide"] = "short"
                    else:
                        data["posSide"] = "net"

                    result = self._request("POST", "/api/v5/trade/close-position", data=data)
                     # Return generic success result
                    return OrderResult(
                        order_id="close_pos", symbol=symbol, side="close", qty=pos.size, price=None, status="filled"
                    )

                except Exception as e:
                     logger.error(f"Error using close-position: {e}, failing back to market order")
                     # Fallback logic
                     opposite_side = "sell" if pos.side == "long" else "buy"
                     return self.place_market_order(symbol, opposite_side, pos.size)

        logger.warning(f"No open position found for {symbol}")
        return None

    def get_latest_price(self, symbol: str) -> float:
        """Get latest market price for a symbol."""
        try:
            result = self._request("GET", "/api/v5/market/ticker", {"instId": symbol})
            return self._safe_float(result["data"][0]["last"])
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            raise

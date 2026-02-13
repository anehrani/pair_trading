import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from loguru import logger


class PaperTradingClient:
    """
    Simulates a trading client for paper trading using Yahoo Finance data.
    Mimics the interface of OKXClient for seamless integration.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_fee: float = 0.001,
        state_file: str = "data/paper_wallet.json",
    ):
        """
        Initialize the paper trading client.

        Args:
            initial_capital: Starting capital in USD/USDT.
            transaction_fee: Fee rate per trade (e.g., 0.001 for 0.1%).
            state_file: Path to save/load portfolio state.
        """
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee
        self.state_file = Path(state_file)
        
        # Portfolio state
        self.balance = initial_capital
        self.positions: Dict[str, float] = {}  # Symbol -> Quantity
        self.history: List[Dict] = []  # Trade history

        self._load_state()

    def _load_state(self):
        """Load portfolio state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.balance = data.get("balance", self.initial_capital)
                    self.positions = data.get("positions", {})
                    self.history = data.get("history", [])
                    logger.info(f"Loaded paper wallet: Balance=${self.balance:,.2f}, Positions={len(self.positions)}")
            except Exception as e:
                logger.error(f"Failed to load paper wallet: {e}")
        else:
            logger.info(f"Initialized new paper wallet with ${self.initial_capital:,.2f}")
            self._save_state()

    def _save_state(self):
        """Save portfolio state to disk."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "balance": self.balance,
                "positions": self.positions,
                "history": self.history,
                "updated_at": datetime.utcnow().isoformat()
            }
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save paper wallet: {e}")

    def get_market_price(self, symbol: str) -> float:
        """Get current market price for a symbol using yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            # Try to get ask/bid or regular market price
            # fast_info is faster but sometimes less detailed
            price = ticker.fast_info.last_price
            if price is None:
                # Fallback to history if fast_info fails
                hist = ticker.history(period="1d", interval="1m")
                if not hist.empty:
                    price = hist["Close"].iloc[-1]
                else:
                    raise ValueError(f"No price data found for {symbol}")
            return float(price)
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            raise

    # --- Data Fetching Methods (Mimicking OKXClient) ---

    def get_all_klines(self, symbol: str, interval: str, days: int = 30) -> pd.DataFrame:
        """
        Fetch historical klines from Yahoo Finance.
        
        Args:
            symbol: Ticker symbol (e.g., 'SPY', 'AAPL').
            interval: Candle interval (e.g., '5m', '1h', '1d').
            days: Number of lookback days.
            
        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]
        """
        yf_interval = interval
        if interval == "1M": yf_interval = "1mo"
        
        start_date = datetime.now() - timedelta(days=days)
        logger.info(f"Fetching {days} days of {interval} data for {symbol} from Yahoo...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use download for potentially more robust fetching than ticker.history
                df = yf.download(
                    tickers=symbol,
                    start=start_date,
                    interval=yf_interval,
                    progress=False,
                    threads=False,
                    group_by='ticker'
                )
                
                if df.empty:
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt+1} failed for {symbol}, retrying...")
                        import time
                        time.sleep(2)
                        continue
                    logger.warning(f"No data returned for {symbol} after {max_retries} attempts")
                    return pd.DataFrame()
                
                # Standardize columns (yf.download with single ticker might have MultiIndex or simple columns)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(1)
                
                df = df.rename(columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume"
                })
                
                # Ensure index is timezone-aware UTC
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                else:
                    df.index = df.index.tz_convert("UTC")
                
                # Reset index to get timestamp column
                df = df.reset_index()
                # Rename first column to timestamp
                df = df.rename(columns={df.columns[0]: "timestamp"})
                    
                return df[["timestamp", "open", "high", "low", "close", "volume"]]

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt+1} for {symbol} failed with: {e}. Retrying...")
                    import time
                    time.sleep(2)
                else:
                    logger.error(f"Failed to fetch history for {symbol} after {max_retries} attempts: {e}")
                    return pd.DataFrame()
        return pd.DataFrame()

    def get_klines(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch recent klines.
        
        Args:
            limit: Number of candles (approximate for yfinance).
        """
        # yfinance doesn't support 'limit' directly cleanly for small intraday counts 
        # without 'period'. We'll use 'period' based on interval.
        
        period = "1d"
        if interval in ["1m", "2m", "5m"]:
            period = "1d" # Min period for intraday is 1d usually
        elif interval in ["1h", "90m"]:
            period = "5d"
        elif interval == "1d":
            period = "1mo" # Fetch a month to be safe
            
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                return pd.DataFrame()
                
            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })
            
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            
            # Reset index and cleanup
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "timestamp"})
            
            # Return only requested limit
            return df[["timestamp", "open", "high", "low", "close", "volume"]].tail(limit)
            
        except Exception as e:
            logger.error(f"Failed to fetch klines for {symbol}: {e}")
            return pd.DataFrame()

    def get_account_balance(self) -> Dict[str, float]:
        """
        Return account balance details.
        
        Returns:
             Dict with 'totalEq' (Total Equity), 'usedEq' (Used Margin/Position Value)
        """
        # Calculate current value of positions
        position_value = 0.0
        for sym, qty in self.positions.items():
            try:
                price = self.get_market_price(sym)
                position_value += abs(qty) * price
            except:
                pass # Skip if pricing fails
                
        total_equity = self.balance
        # Note: In a real margin account, equity = cash + unrealized_pnl.
        # Here we simplify: cash balance tracks realized PnL. 
        # Unrealized PnL is implicit if we mark-to-market.
        # But this simple paper trader might just track cash + cost basis or current value?
        # Let's stick to: Total Equity = Cash + Current Value of Positions
        
        # Actually, for standard spot trading: Equity = Cash + Value of Assets.
        # For Pairs Trading (Long/Short), we need margin.
        # Let's assume 'balance' is the Cash/Collateral.
        # Shorting adds cash (proceeds) but creates a liability.
        # Longing removes cash.
        
        # Simplified Model:
        # We track 'balance' as Net Liquidation Value (NLV) ideally.
        # But simpler: Balance = Cash. 
        # Long purchase: Cash -= Cost. Position += Qty.
        # Short sale: Cash += Proceeds. Position -= Qty.
        # Total Equity = Cash + Sum(Position * Price)
        
        current_val = 0.0
        for sym, qty in self.positions.items():
            if qty == 0: continue
            price = self.get_market_price(sym)
            current_val += qty * price
            
        equity = self.balance + current_val
        
        return {
            "totalEq": equity,
            "isoEq": 0.0, # Isolated margin equity (unused)
            "adjEq": equity, # Adjusted equity
            "ordFroz": 0.0, # Frozen in orders
            "imr": 0.0, # Initial margin requirement
            "mmr": 0.0, # Maintenance margin requirement
            "mgnRatio": 0.0
        }

    # --- Execution Methods ---

    def place_market_order(self, symbol: str, side: str, amount: float) -> Optional[Dict]:
        """
        Place a market order.
        
        Args:
            symbol: Asset symbol.
            side: "Buy" or "Sell".
            amount: Quantity to buy/sell (in shares/units).
        """
        try:
            price = self.get_market_price(symbol)
            cost = price * amount
            fee = cost * self.transaction_fee
            
            # Update Portfolio
            if side.lower() == "buy":
                # Buy: Cash decreases, Position increases
                self.balance -= (cost + fee)
                self.positions[symbol] = self.positions.get(symbol, 0.0) + amount
                logger.info(f"PAPER TRADE: Bought {amount} {symbol} @ {price:.2f} (Fee: {fee:.2f})")
            else:
                # Sell: Cash increases, Position decreases
                self.balance += (cost - fee)
                self.positions[symbol] = self.positions.get(symbol, 0.0) - amount
                logger.info(f"PAPER TRADE: Sold {amount} {symbol} @ {price:.2f} (Fee: {fee:.2f})")
                
            # Clean up zero positions
            if abs(self.positions.get(symbol, 0.0)) < 1e-6:
                del self.positions[symbol]
                
            # Log Trade
            trade_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": price,
                "fee": fee
            }
            self.history.append(trade_record)
            self._save_state()
            
            return {"ordId": "paper_" + datetime.now().strftime("%Y%m%d%H%M%S"), "state": "filled"}
            
        except Exception as e:
            logger.error(f"Failed to place paper order: {e}")
            raise

    def close_position(self, symbol: str):
        """Close existing position for symbol."""
        qty = self.positions.get(symbol, 0.0)
        if abs(qty) > 0:
            side = "Sell" if qty > 0 else "Buy"
            self.place_market_order(symbol, side, abs(qty))


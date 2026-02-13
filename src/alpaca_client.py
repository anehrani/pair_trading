import time
import os
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    StopLimitOrderRequest,
    TrailingStopOrderRequest,
    GetOrdersRequest,
    ClosePositionRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    OrderStatus,
    QueryOrderStatus,
)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockLatestQuoteRequest,
    StockBarsRequest,
    StockSnapshotRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import pandas as pd
from loguru import logger
import math



class AlpacaPaperTrader:
    """Client for Alpaca Paper Trading."""

    def __init__(self, api_key: str, secret_key: str, url_override: str = None):
        """
        Initialize Alpaca Paper Trading client.

        Args:
            api_key: Your Alpaca API key
            secret_key: Your Alpaca secret key
            url_override: Optional custom API URL
        """
        # paper=True enables paper trading, but url_override takes precedence if provided
        self.trading_client = TradingClient(api_key, secret_key, paper=True, url_override=url_override)
        self.data_client = StockHistoricalDataClient(api_key, secret_key) 


        # Verify connection
        self.account = self.trading_client.get_account()
        print(f"✅ Connected to Alpaca Paper Trading")
        print(f"   Account ID: {self.account.id}")
        print(f"   Cash: ${float(self.account.cash):,.2f}")
        print(f"   Portfolio Value: ${float(self.account.portfolio_value):,.2f}")
        print(f"   Buying Power: ${float(self.account.buying_power):,.2f}")
        print(f"   Day Trade Count: {self.account.daytrade_count}")

    # ─────────────────────────────────────────────
    # ACCOUNT INFO
    # ─────────────────────────────────────────────

    def get_account_info(self) -> dict:
        """Get full account information."""
        account = self.trading_client.get_account()
        return {
            "id": account.id,
            "status": account.status,
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "equity": float(account.equity),
            "long_market_value": float(account.long_market_value),
            "short_market_value": float(account.short_market_value),
            "initial_margin": float(account.initial_margin),
            "maintenance_margin": float(account.maintenance_margin),
            "daytrade_count": account.daytrade_count,
            "pattern_day_trader": account.pattern_day_trader,
            "trading_blocked": account.trading_blocked,
            "account_blocked": account.account_blocked,
        }

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        clock = self.trading_client.get_clock()
        print(f"   Market is {'OPEN' if clock.is_open else 'CLOSED'}")
        print(f"   Next open:  {clock.next_open}")
        print(f"   Next close: {clock.next_close}")
        return clock.is_open

    # ─────────────────────────────────────────────
    # MARKET DATA
    # ─────────────────────────────────────────────

    def get_latest_quote(self, symbol: str) -> dict:
        """Get the latest quote for a symbol."""
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = self.data_client.get_stock_latest_quote(request)
        q = quote[symbol]
        return {
            "symbol": symbol,
            "ask_price": float(q.ask_price),
            "ask_size": q.ask_size,
            "bid_price": float(q.bid_price),
            "bid_size": q.bid_size,
            "timestamp": q.timestamp,
        }

    def get_bars(self, symbol: str, days: int = 30, timeframe: TimeFrame = TimeFrame.Day) -> list:
        """Get historical bars for a symbol."""
        start_time = datetime.now() - timedelta(days=days)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start_time,
            limit=10000, # Max limit to ensure we get enough data, though pagination might be needed for very long periods
            feed="iex", # valid for free plan usually
            sort="asc"
        )
        try:
            logger.debug(f"Requesting bars for {symbol}: start={start_time}, timeframe={timeframe}")
            response = self.data_client.get_stock_bars(request)
            result = []
            
            # Accessing from response.data is safer in alpaca-py
            bars = response.data.get(symbol, [])
            if bars:
                for bar in bars:
                    result.append({
                        "timestamp": bar.timestamp,
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": bar.volume,
                        "vwap": float(bar.vwap) if hasattr(bar, 'vwap') and bar.vwap is not None else 0.0,
                    })
                logger.debug(f"Successfully fetched {len(result)} bars for {symbol}")
            else:
                logger.warning(f"No bars returned for {symbol} in response")
            return result
        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}: {e}")
            return []

    # ─────────────────────────────────────────────
    # LIVE TRADER INTERFACE IMPLEMENTATION
    # ─────────────────────────────────────────────

    def _get_timeframe_from_interval(self, interval: str) -> TimeFrame:
        """Convert string interval (e.g. '5m') to Alpaca TimeFrame."""
        if interval.endswith('m'):
            return TimeFrame(int(interval[:-1]), TimeFrameUnit.Minute)
        elif interval.endswith('h') or interval.endswith('H'):
            return TimeFrame(int(interval[:-1]), TimeFrameUnit.Hour)
        elif interval.endswith('d') or interval.endswith('D'):
            return TimeFrame(int(interval[:-1]), TimeFrameUnit.Day)
        else:
            return TimeFrame(5, TimeFrameUnit.Minute) # Default

    def get_all_klines(self, symbol: str, interval: str, days: int = 30) -> pd.DataFrame:
        """
        Fetch historical klines formatted for LiveTrader.
        Returns DataFrame with [timestamp, open, high, low, close, volume]
        """
        timeframe = self._get_timeframe_from_interval(interval)
        bars_list = self.get_bars(symbol, days, timeframe)
        
        if not bars_list:
            return pd.DataFrame()
            
        df = pd.DataFrame(bars_list)
        if df.empty:
            return pd.DataFrame()
            
        # Ensure timestamp is datetime and timezone aware (UTC)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    def get_klines(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch recent klines.
        Note: Alpaca API uses start/end, not limit directly in basic requests easily without start.
        We will estimate days based on limit and interval.
        """
        # Estimate days needed
        mins_per_candle = 5
        if interval.endswith('m'):
            mins_per_candle = int(interval[:-1])
        elif interval.endswith('h'):
            mins_per_candle = int(interval[:-1]) * 60
        elif interval.endswith('d'):
            mins_per_candle = 1440
            
        total_minutes = limit * mins_per_candle
        days = math.ceil(total_minutes / (24 * 60)) + 1 # Add buffer
        
        return self.get_all_klines(symbol, interval, days).tail(limit)

    def get_account_balance(self) -> dict:
        """Get account balance in expected format."""
        acct = self.get_account_info()
        return {
            "total": acct['equity'],
            "available": acct['buying_power'],
            "used": acct['equity'] - acct['buying_power'] # Rough approximation
        }

    def place_market_order(self, symbol: str, side: str, qty: float):
        """Place market order with simple interface."""
        side = side.lower()
        if side == "buy":
            return self.buy_market(symbol, qty)
        elif side == "sell":
            return self.sell_market(symbol, qty)
        else:
            raise ValueError(f"Invalid side: {side}")


    def get_snapshot(self, symbol: str) -> dict:
        """Get a full snapshot (quote, latest trade, minute bar, daily bar)."""
        request = StockSnapshotRequest(symbol_or_symbols=symbol)
        snapshot = self.data_client.get_stock_snapshot(request)
        s = snapshot[symbol]
        return {
            "symbol": symbol,
            "latest_trade": {
                "price": float(s.latest_trade.price),
                "size": s.latest_trade.size,
                "timestamp": s.latest_trade.timestamp,
            },
            "latest_quote": {
                "ask": float(s.latest_quote.ask_price),
                "bid": float(s.latest_quote.bid_price),
            },
            "minute_bar": {
                "open": float(s.minute_bar.open),
                "high": float(s.minute_bar.high),
                "low": float(s.minute_bar.low),
                "close": float(s.minute_bar.close),
                "volume": s.minute_bar.volume,
            },
            "daily_bar": {
                "open": float(s.daily_bar.open),
                "high": float(s.daily_bar.high),
                "low": float(s.daily_bar.low),
                "close": float(s.daily_bar.close),
                "volume": s.daily_bar.volume,
            },
        }

    # ─────────────────────────────────────────────
    # ORDER PLACEMENT
    # ─────────────────────────────────────────────

    def buy_market(self, symbol: str, qty: float, time_in_force: TimeInForce = TimeInForce.DAY):
        """Place a market buy order."""
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=time_in_force,
        )
        order = self.trading_client.submit_order(order_data)
        print(f"✅ Market BUY order submitted: {qty} shares of {symbol}")
        print(f"   Order ID: {order.id} | Status: {order.status}")
        return order

    def sell_market(self, symbol: str, qty: float, time_in_force: TimeInForce = TimeInForce.DAY):
        """Place a market sell order."""
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=time_in_force,
        )
        order = self.trading_client.submit_order(order_data)
        print(f"✅ Market SELL order submitted: {qty} shares of {symbol}")
        print(f"   Order ID: {order.id} | Status: {order.status}")
        return order

    def buy_limit(self, symbol: str, qty: float, limit_price: float,
                  time_in_force: TimeInForce = TimeInForce.DAY):
        """Place a limit buy order."""
        order_data = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=time_in_force,
            limit_price=limit_price,
        )
        order = self.trading_client.submit_order(order_data)
        print(f"✅ Limit BUY order submitted: {qty} shares of {symbol} @ ${limit_price}")
        print(f"   Order ID: {order.id} | Status: {order.status}")
        return order

    def sell_limit(self, symbol: str, qty: float, limit_price: float,
                   time_in_force: TimeInForce = TimeInForce.DAY):
        """Place a limit sell order."""
        order_data = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=time_in_force,
            limit_price=limit_price,
        )
        order = self.trading_client.submit_order(order_data)
        print(f"✅ Limit SELL order submitted: {qty} shares of {symbol} @ ${limit_price}")
        print(f"   Order ID: {order.id} | Status: {order.status}")
        return order

    def buy_stop(self, symbol: str, qty: float, stop_price: float,
                 time_in_force: TimeInForce = TimeInForce.DAY):
        """Place a stop buy order."""
        order_data = StopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=time_in_force,
            stop_price=stop_price,
        )
        order = self.trading_client.submit_order(order_data)
        print(f"✅ Stop BUY order submitted: {qty} shares of {symbol} @ stop ${stop_price}")
        return order

    def sell_stop_limit(self, symbol: str, qty: float, stop_price: float,
                        limit_price: float, time_in_force: TimeInForce = TimeInForce.DAY):
        """Place a stop-limit sell order (useful for stop-loss with limit)."""
        order_data = StopLimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=time_in_force,
            stop_price=stop_price,
            limit_price=limit_price,
        )
        order = self.trading_client.submit_order(order_data)
        print(f"✅ Stop-Limit SELL: {qty} {symbol} | Stop: ${stop_price} | Limit: ${limit_price}")
        return order

    def buy_trailing_stop(self, symbol: str, qty: float, trail_percent: float,
                          time_in_force: TimeInForce = TimeInForce.DAY):
        """Place a trailing stop buy order."""
        order_data = TrailingStopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=time_in_force,
            trail_percent=trail_percent,
        )
        order = self.trading_client.submit_order(order_data)
        print(f"✅ Trailing Stop BUY: {qty} {symbol} | Trail: {trail_percent}%")
        return order

    def buy_notional(self, symbol: str, dollar_amount: float,
                     time_in_force: TimeInForce = TimeInForce.DAY):
        """Buy a dollar amount worth of a stock (fractional shares)."""
        order_data = MarketOrderRequest(
            symbol=symbol,
            notional=dollar_amount,
            side=OrderSide.BUY,
            time_in_force=time_in_force,
        )
        order = self.trading_client.submit_order(order_data)
        print(f"✅ Market BUY (notional): ${dollar_amount} of {symbol}")
        return order

    # ─────────────────────────────────────────────
    # ORDER MANAGEMENT
    # ─────────────────────────────────────────────

    def get_order(self, order_id: str):
        """Get details of a specific order."""
        return self.trading_client.get_order_by_id(order_id)

    def get_open_orders(self) -> list:
        """Get all open orders."""
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = self.trading_client.get_orders(request)
        for o in orders:
            print(f"   {o.side} {o.qty} {o.symbol} | Type: {o.type} | Status: {o.status}")
        return orders

    def get_all_orders(self, limit: int = 50) -> list:
        """Get recent orders (all statuses)."""
        request = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            limit=limit,
        )
        return self.trading_client.get_orders(request)

    def cancel_order(self, order_id: str):
        """Cancel a specific order."""
        self.trading_client.cancel_order_by_id(order_id)
        print(f"❌ Order {order_id} cancelled")

    def cancel_all_orders(self):
        """Cancel all open orders."""
        statuses = self.trading_client.cancel_orders()
        print(f"❌ Cancelled {len(statuses)} orders")
        return statuses

    def wait_for_fill(self, order_id: str, timeout: int = 60, poll_interval: int = 2):
        """Wait for an order to fill."""
        start = time.time()
        while time.time() - start < timeout:
            order = self.trading_client.get_order_by_id(order_id)
            if order.status == OrderStatus.FILLED:
                print(f"✅ Order FILLED: {order.filled_qty} @ ${order.filled_avg_price}")
                return order
            elif order.status in (OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED):
                print(f"❌ Order {order.status}: {order_id}")
                return order
            print(f"   ⏳ Status: {order.status} (waiting...)")
            time.sleep(poll_interval)
        print(f"⏰ Timeout waiting for order fill")
        return self.trading_client.get_order_by_id(order_id)

    # ─────────────────────────────────────────────
    # POSITIONS
    # ─────────────────────────────────────────────

    def get_positions(self) -> list:
        """Get all open positions."""
        positions = self.trading_client.get_all_positions()
        print(f"\n📊 Open Positions ({len(positions)}):")
        print(f"   {'Symbol':<8} {'Qty':>8} {'Avg Entry':>12} {'Current':>12} {'P/L':>12} {'P/L %':>8}")
        print(f"   {'-'*68}")
        for p in positions:
            print(
                f"   {p.symbol:<8} "
                f"{float(p.qty):>8.2f} "
                f"${float(p.avg_entry_price):>10.2f} "
                f"${float(p.current_price):>10.2f} "
                f"${float(p.unrealized_pl):>10.2f} "
                f"{float(p.unrealized_plpc)*100:>7.2f}%"
            )
        return positions

    def get_position(self, symbol: str):
        """Get position for a specific symbol."""
        try:
            position = self.trading_client.get_open_position(symbol)
            print(f"📊 {symbol}: {position.qty} shares @ ${position.avg_entry_price}")
            print(f"   Current: ${position.current_price} | P/L: ${position.unrealized_pl}")
            return position
        except Exception as e:
            print(f"   No open position for {symbol}: {e}")
            return None

    def close_position(self, symbol: str, qty: float = None, percentage: float = None):
        """
        Close a position (fully or partially).

        Args:
            symbol: Stock symbol
            qty: Number of shares to close (None = close all)
            percentage: Percentage to close (e.g., 50 for 50%)
        """
        if percentage is not None:
            close_options = ClosePositionRequest(percentage=str(percentage))
        elif qty is not None:
            close_options = ClosePositionRequest(qty=str(qty))
        else:
            close_options = None  # Close entire position

        order = self.trading_client.close_position(symbol, close_options=close_options)
        print(f"✅ Closing position: {symbol}")
        return order

    def close_all_positions(self, cancel_orders: bool = True):
        """Close all open positions."""
        responses = self.trading_client.close_all_positions(cancel_orders=cancel_orders)
        print(f"✅ Closing all positions ({len(responses)} positions)")
        return responses

    # ─────────────────────────────────────────────
    # PORTFOLIO ANALYTICS
    # ─────────────────────────────────────────────

    def portfolio_summary(self):
        """Print a comprehensive portfolio summary."""
        account = self.trading_client.get_account()
        positions = self.trading_client.get_all_positions()

        total_pl = sum(float(p.unrealized_pl) for p in positions)
        total_market_value = sum(float(p.market_value) for p in positions)

        print("\n" + "=" * 60)
        print("📈 PORTFOLIO SUMMARY")
        print("=" * 60)
        print(f"  Cash:              ${float(account.cash):>12,.2f}")
        print(f"  Portfolio Value:   ${float(account.portfolio_value):>12,.2f}")
        print(f"  Equity:            ${float(account.equity):>12,.2f}")
        print(f"  Buying Power:      ${float(account.buying_power):>12,.2f}")
        print(f"  Market Value:      ${total_market_value:>12,.2f}")
        print(f"  Unrealized P/L:    ${total_pl:>12,.2f}")
        print(f"  Positions:         {len(positions):>12}")
        print("=" * 60)

        if positions:
            self.get_positions()


# ─────────────────────────────────────────────
# USAGE EXAMPLE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()


    # ⚠️ Replace with your actual Alpaca API keys
    API_KEY = os.getenv("ALPACA_API_KEY")
    SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

    if not API_KEY or not SECRET_KEY:
        print("❌ Error: ALPACA_API_KEY or ALPACA_SECRET_KEY not found in environment variables.")
        exit(1)

    # Initialize the client
    try:
        trader = AlpacaPaperTrader(API_KEY, SECRET_KEY)
        print("✅ Successfully connected to Alpaca Paper Trading API")
    except Exception as e:
        print(f"\n❌ FAILED to connect to Alpaca API: {e}")
        print("💡 Please check your ALPACA_API_KEY and ALPACA_SECRET_KEY in the .env file.")
        print("   They should look like: PK... (Key) and ... (Secret)")
        exit(1)

    # ── Check market status ──
    trader.is_market_open()

    # ── Get account info ──
    info = trader.get_account_info()
    print(f"\nAccount cash: ${info['cash']:,.2f}")

    # ── Get market data ──
    quote = trader.get_latest_quote("AAPL")
    print(f"\nAAPL - Ask: ${quote['ask_price']}, Bid: ${quote['bid_price']}")

    # Get historical bars
    bars = trader.get_bars("AAPL", days=5)
    for bar in bars[-3:]:
        print(f"  {bar['timestamp'].date()} | O: ${bar['open']:.2f} H: ${bar['high']:.2f} "
              f"L: ${bar['low']:.2f} C: ${bar['close']:.2f} V: {bar['volume']:,}")

    # ── Place orders ──

    # Market buy 1 share of AAPL
    order = trader.buy_market("AAPL", qty=1)

    # Limit buy 2 shares of MSFT at $400
    # order = trader.buy_limit("MSFT", qty=2, limit_price=400.00)

    # Buy $500 worth of GOOGL (fractional shares)
    # order = trader.buy_notional("GOOGL", dollar_amount=500.00)

    # Stop-limit sell for risk management
    # order = trader.sell_stop_limit("AAPL", qty=1, stop_price=170.00, limit_price=169.50)

    # Wait for the order to fill
    filled_order = trader.wait_for_fill(str(order.id), timeout=30)

    # ── Check positions ──
    trader.portfolio_summary()

    # ── Sell / close positions ──
    # trader.sell_market("AAPL", qty=1)
    # trader.close_position("AAPL")              # Close entire position
    # trader.close_position("AAPL", percentage=50)  # Close 50%

    # ── Order management ──
    open_orders = trader.get_open_orders()
    # trader.cancel_all_orders()

    # ── Nuclear option: close everything ──
    # trader.close_all_positions()
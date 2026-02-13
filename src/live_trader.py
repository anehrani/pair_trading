"""Live trading orchestrator for copula-based pair trading.

Manages the full trading cycle: data collection, pair selection,
copula fitting, signal generation, and order execution.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from dotenv import load_dotenv
from loguru import logger


# Add project root to path so we can import 'src' package
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.okx_client import OKXClient
from src.paper_client import PaperTradingClient
from src.data_buffer import DataBuffer
from src.strategy_core import (
    CopulaModel,
    TradingPair,
    TradingSignal,
    calculate_position_sizes,
    fit_copula_model,
    generate_signal,
    select_trading_pair,
)


@dataclass
class TradingState:
    """Persistent state for live trading."""

    formation_start: str | None = None  # ISO timestamp
    formation_end: str | None = None
    trading_start: str | None = None
    trading_end: str | None = None
    current_pair: dict | None = None  # TradingPair as dict
    active_position: dict | None = None  # Position info
    cumulative_pnl: float = 0.0
    trades_count: int = 0
    last_update: str | None = None


class LiveTrader:
    """Main orchestrator for live trading."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize live trader.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Load environment variables
        load_dotenv()

        # Setup logging
        log_dir = Path(self.config["logging"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)

        logger.remove()  # Remove default handler
        logger.add(
            sys.stderr,
            level=self.config["logging"]["level"],
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        )
        logger.add(
            log_dir / "trading_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="30 days",
            level="DEBUG",
        )

        # Initialize Client based on mode
        self.mode = self.config.get("mode", "live")
        logger.info(f"Initializing LiveTrader in {self.mode.upper()} mode")

        if self.mode == "paper":
            self.client = PaperTradingClient(
                initial_capital=self.config["paper"].get("initial_capital", 100000.0),
                transaction_fee=self.config["paper"].get("transaction_fee", 0.001),
                state_file=self.config["data"].get("paper_wallet_file", "data/paper_wallet.json")
            )
        else:
            # Initialize OKX client for live trading
            api_key = os.getenv("OKX_API_KEY")
            api_secret = os.getenv("OKX_API_SECRET")
            passphrase = os.getenv("OKX_PASSPHRASE")

            if not api_key or not api_secret or not passphrase:
                raise ValueError("OKX_API_KEY, OKX_API_SECRET, and OKX_PASSPHRASE must be set in .env file for LIVE trading")

            self.client = OKXClient(
                api_key=api_key,
                api_secret=api_secret,
                passphrase=passphrase,
                demo=self.config["okx"]["demo"],
            )

        # Initialize data buffer
        all_symbols = [self.config["strategy"]["reference_symbol"]] + self.config["strategy"]["symbols"]
        self.buffer = DataBuffer(symbols=all_symbols, max_days=self.config["strategy"]["formation_days"] + 7)

        # Load or initialize state
        self.state_file = Path(self.config["data"]["state_file"])
        self.state = self._load_state()

        # Current model and pair
        self.current_model: CopulaModel | None = None
        self.current_pair: TradingPair | None = None

        logger.info("LiveTrader initialized")

    def _load_state(self) -> TradingState:
        """Load state from disk."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                data = json.load(f)
                logger.info(f"Loaded state from {self.state_file}")
                return TradingState(**data)
        return TradingState()

    def _save_state(self) -> None:
        """Save state to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(asdict(self.state), f, indent=2)

    def initialize_data(self) -> None:
        """Download initial historical data."""
        logger.info("Initializing data buffer...")

        buffer_file = Path(self.config["data"]["buffer_file"])

        # Try to load existing buffer
        if buffer_file.exists():
            self.buffer.load(buffer_file)
            logger.info("Loaded existing buffer")

        # Check each symbol individually
        required_days = self.config["strategy"]["formation_days"] + 1
        
        for symbol in self.buffer.symbols:
            # Check if symbol already has enough data
            if symbol in self.buffer.data and not self.buffer.data[symbol].empty:
                df_sym = self.buffer.data[symbol]
                earliest = df_sym["timestamp"].min()
                latest = df_sym["timestamp"].max()
                days_available = (latest - earliest).total_seconds() / (24 * 3600)
                
                if days_available >= required_days:
                    logger.info(f"Symbol {symbol} already has {days_available:.1f} days of data, skipping download.")
                    continue

            logger.info(f"Downloading {required_days} days of historical data for {symbol}...")
            try:
                df = self.client.get_all_klines(
                    symbol=symbol,
                    interval=self.config["strategy"]["interval"],
                    days=required_days,
                )
                if not df.empty:
                    self.buffer.update(symbol, df)
                    logger.info(f"Downloaded {len(df)} candles for {symbol}")
                else:
                    logger.warning(f"No data returned for {symbol}")
                
                time.sleep(1.0)  # Increase delay to avoid rate limiting
            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")

        # Save buffer
        self.buffer.save(buffer_file)

        earliest, latest = self.buffer.get_data_range()
        logger.info(f"Data range: {earliest} to {latest}")

    def update_data(self) -> None:
        """Fetch latest candles and update buffer."""
        for symbol in self.buffer.symbols:
            try:
                # Get last 10 candles to ensure we don't miss any
                df = self.client.get_klines(
                    symbol=symbol,
                    interval=self.config["strategy"]["interval"],
                    limit=10,
                )
                self.buffer.update(symbol, df)
                time.sleep(0.2)
            except Exception as e:
                logger.error(f"Failed to update {symbol}: {e}")

    def _validate_state(self) -> bool:
        """Validate that current state is consistent with config.
        
        Returns:
            True if valid, False if state needs reset
        """
        if self.state.current_pair is None:
            return True  # No pair yet, valid
        
        # Check if pair symbols are in current config
        pair_symbols = {self.state.current_pair["symbol1"], self.state.current_pair["symbol2"]}
        config_symbols = set(self.buffer.symbols)
        
        if not pair_symbols.issubset(config_symbols):
            logger.warning(f"State contains pair with symbols {pair_symbols} not in current config")
            logger.warning("This likely means config changed. Resetting state.")
            return False
        
        return True

    def should_start_new_cycle(self) -> bool:
        """Check if we should start a new formation period."""
        # First check if state is valid
        if not self._validate_state():
            return True  # Force new cycle to reset state
        
        if self.state.trading_end is None:
            return True  # First cycle

        trading_end = pd.Timestamp(self.state.trading_end, tz="UTC")
        now = pd.Timestamp.now(tz="UTC")

        # Start new cycle if trading period ended
        return now >= trading_end

    def run_formation_period(self) -> bool:
        """Run formation period: select pair and fit copula.

        Returns:
            True if successful, False otherwise
        """
        logger.info("=== Starting Formation Period ===")

        # Define formation window
        latest = self.buffer.get_data_range()[1]
        if latest is None:
            logger.error("No data available")
            return False

        formation_days = self.config["strategy"]["formation_days"]
        formation_end = latest
        formation_start = formation_end - pd.Timedelta(days=formation_days)

        logger.info(f"Formation window: {formation_start} to {formation_end}")

        # Get formation data
        formation_prices = self.buffer.get_closes(start=formation_start, end=formation_end)

        if formation_prices.empty:
            logger.error("No formation data available")
            return False

        # Use log prices for cointegration (as in backtest)
        formation_prices = np.log(formation_prices.astype(float))

        # Select trading pair
        pair, results_df = select_trading_pair(
            formation_prices,
            reference_symbol=self.config["strategy"]["reference_symbol"],
            eg_alpha=self.config["strategy"]["eg_alpha"],
            adf_alpha=self.config["strategy"]["adf_alpha"],
            kss_critical=self.config["strategy"]["kss_critical"],
        )

        # Write down detected pairs
        if not results_df.empty:
            report_dir = Path(self.config["data"].get("report_dir", "reports"))
            report_dir.mkdir(parents=True, exist_ok=True)
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            report_path = report_dir / f"cointegrated_pairs_{timestamp}.csv"
            results_df.to_csv(report_path, index=False)
            logger.info(f"Saved {len(results_df)} cointegrated pairs to {report_path}")

        if pair is None:
            logger.warning("No suitable pair found")
            return False

        # Fit copula model
        model = fit_copula_model(
            formation_prices,
            pair,
            reference_symbol=self.config["strategy"]["reference_symbol"],
        )

        if model is None:
            logger.error("Failed to fit copula model")
            return False

        # Update state
        self.current_pair = pair
        self.current_model = model

        self.state.formation_start = formation_start.isoformat()
        self.state.formation_end = formation_end.isoformat()
        self.state.trading_start = formation_end.isoformat()
        self.state.trading_end = (formation_end + pd.Timedelta(days=self.config["strategy"]["trading_days"])).isoformat()
        self.state.current_pair = asdict(pair)

        self._save_state()

        logger.info(f"Formation complete. Trading until {self.state.trading_end}")
        return True

    def execute_signal(self, signal: TradingSignal) -> None:
        """Execute trading signal.

        Args:
            signal: Trading signal
        """
        if signal.action == "WAIT":
            return

        ref_sym = self.config["strategy"]["reference_symbol"]
        sym1 = self.current_pair.symbol1
        sym2 = self.current_pair.symbol2

        # Get current prices
        prices = self.buffer.get_latest_prices()
        p_ref = prices[ref_sym]
        p1 = prices[sym1]
        p2 = prices[sym2]

        # Calculate position sizes
        q1, q2 = calculate_position_sizes(
            self.current_model.beta1,
            self.current_model.beta2,
            p1,
            p2,
            self.config["strategy"]["capital_per_side"],
        )

        if signal.action == "CLOSE":
            # Close all positions
            logger.info("Closing positions")
            self.client.close_position(sym1)
            self.client.close_position(sym2)
            self.state.active_position = None
            self._save_state()

        elif signal.action == "LONG_S1_SHORT_S2":
            # Long beta2*P2, Short beta1*P1 (Table 4)
            logger.info(f"Opening: LONG {sym2} ({q2:.4f}), SHORT {sym1} ({q1:.4f})")

            try:
                self.client.place_market_order(sym2, "Buy", abs(q2))
                self.client.place_market_order(sym1, "Sell", abs(q1))

                self.state.active_position = {
                    "long_symbol": sym2,
                    "short_symbol": sym1,
                    "long_qty": abs(q2),
                    "short_qty": abs(q1),
                    "entry_time": signal.timestamp.isoformat(),
                }
                self.state.trades_count += 1
                self._save_state()

            except Exception as e:
                logger.error(f"Failed to execute order: {e}")

        elif signal.action == "SHORT_S1_LONG_S2":
            # Short beta2*P2, Long beta1*P1 (Table 4)
            logger.info(f"Opening: SHORT {sym2} ({q2:.4f}), LONG {sym1} ({q1:.4f})")

            try:
                self.client.place_market_order(sym2, "Sell", abs(q2))
                self.client.place_market_order(sym1, "Buy", abs(q1))

                self.state.active_position = {
                    "long_symbol": sym1,
                    "short_symbol": sym2,
                    "long_qty": abs(q1),
                    "short_qty": abs(q2),
                    "entry_time": signal.timestamp.isoformat(),
                }
                self.state.trades_count += 1
                self._save_state()

            except Exception as e:
                logger.error(f"Failed to execute order: {e}")

    def run_trading_loop(self) -> None:
        """Main trading loop."""
        logger.info("=== Starting Trading Loop ===")

        interval_str = self.config["strategy"]["interval"]
        interval_map = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1H": 3600, "2H": 7200, "4H": 14400}
        interval_seconds = interval_map.get(interval_str, 300)  # Default to 5m if unknown

        while True:
            try:
                # Update data
                self.update_data()

                # Check if we need new formation period
                if self.should_start_new_cycle():
                    # Close any open positions at end of trading period
                    if self.state.active_position:
                        logger.info("End of trading period, closing positions")
                        self.execute_signal(
                            TradingSignal(action="CLOSE", h1_2=0.5, h2_1=0.5, timestamp=pd.Timestamp.now(tz="UTC"))
                        )

                    # Run new formation
                    success = self.run_formation_period()
                    if not success:
                        logger.warning("Formation failed, waiting 1 hour before retry")
                        time.sleep(3600)
                        continue

                # Generate signal
                if self.current_model and self.current_pair:
                    prices = self.buffer.get_latest_prices()

                    signal = generate_signal(
                        prices,
                        self.current_model,
                        self.current_pair,
                        self.config["strategy"]["reference_symbol"],
                        self.config["strategy"]["alpha1"],
                        self.config["strategy"]["alpha2"],
                    )

                    logger.info(f"Signal: {signal.action} (h1|2={signal.h1_2:.3f}, h2|1={signal.h2_1:.3f})")

                    # Execute if not waiting
                    if signal.action != "WAIT":
                        self.execute_signal(signal)

                # Update state timestamp
                self.state.last_update = pd.Timestamp.now(tz="UTC").isoformat()
                self._save_state()

                # Save buffer periodically
                if self.state.trades_count % 10 == 0:
                    self.buffer.save(Path(self.config["data"]["buffer_file"]))

                # Wait for next interval
                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                time.sleep(60)  # Wait 1 minute before retry

    def run(self) -> None:
        """Main entry point."""
        logger.info("Starting Live Trader")

        # Check account balance
        balance = self.client.get_account_balance()
        logger.info(f"Account balance: {balance}")

        # Initialize data
        self.initialize_data()

        # Start trading
        self.run_trading_loop()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Live copula-based pair trading")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    trader = LiveTrader(config_path=args.config)
    trader.run()


if __name__ == "__main__":
    main()

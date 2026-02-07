"""Download stock data from Yahoo Finance for pair trading.

This script downloads historical stock data from Yahoo Finance and formats it
for compatibility with the pair trading simulator. This is an alternative to
Bybit tokenized stocks which may not be available.

Yahoo Finance provides free access to:
- US stocks (AAPL, TSLA, GOOGL, etc.)
- International stocks
- ETFs, indices, forex
- Multiple timeframes (1m, 5m, 15m, 30m, 1h, 1d)

Example Usage:
  # Download default tech stocks
  python -m get_data.download_yahoo_stocks \
    --interval 1h \
    --start 2023-01-01 \
    --end 2024-12-31 \
    --out data/yahoo_stocks_1h

  # Download specific stocks
  python -m get_data.download_yahoo_stocks \
    --interval 1h \
    --start 2023-01-01 \
    --end 2024-12-31 \
    --symbols AAPL MSFT GOOGL TSLA \
    --out data/yahoo_stocks_1h

  # Download with different interval
  python -m get_data.download_yahoo_stocks \
    --interval 1d \
    --start 2020-01-01 \
    --end 2024-12-31 \
    --out data/yahoo_stocks_1d

Supported Intervals:
- 1m, 2m, 5m, 15m, 30m, 60m (intraday, limited history)
- 1h, 1d, 1wk, 1mo (longer history available)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("❌ Error: yfinance is not installed.")
    print("\nInstall it with:")
    print("  pip install yfinance")
    print("\nOr add to requirements.txt:")
    print("  echo 'yfinance>=0.2.50' >> requirements.txt")
    print("  pip install -r requirements.txt")
    exit(1)


# Major US tech stocks
TECH_STOCKS: tuple[str, ...] = (
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet/Google
    "AMZN",   # Amazon
    "META",   # Meta/Facebook
    "TSLA",   # Tesla
    "NVDA",   # NVIDIA
    "AMD",    # AMD
    "NFLX",   # Netflix
    "INTC",   # Intel
)

# Financial sector stocks
FINANCIAL_STOCKS: tuple[str, ...] = (
    "JPM",    # JPMorgan Chase
    "BAC",    # Bank of America
    "GS",     # Goldman Sachs
    "MS",     # Morgan Stanley
    "V",      # Visa
    "MA",     # Mastercard
    "AXP",    # American Express
    "C",      # Citigroup
)

# Consumer/Retail stocks
CONSUMER_STOCKS: tuple[str, ...] = (
    "WMT",    # Walmart
    "HD",     # Home Depot
    "NKE",    # Nike
    "SBUX",   # Starbucks
    "MCD",    # McDonald's
    "DIS",    # Disney
    "COST",   # Costco
    "TGT",    # Target
)

# Healthcare stocks
HEALTHCARE_STOCKS: tuple[str, ...] = (
    "JNJ",    # Johnson & Johnson
    "UNH",    # UnitedHealth
    "PFE",    # Pfizer
    "ABBV",   # AbbVie
    "TMO",    # Thermo Fisher
    "MRK",    # Merck
)

# Default: mix of sectors for diversified pair trading
DEFAULT_STOCKS: tuple[str, ...] = (
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "AMD",
    # Finance
    "JPM", "BAC", "GS", "V", "MA",
    # Consumer
    "WMT", "NKE", "SBUX", "DIS",
    # Healthcare
    "JNJ", "PFE",
)


@dataclass(frozen=True)
class StockBar:
    """Stock price bar data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def download_stock(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1h",
) -> list[StockBar] | None:
    """Download stock data from Yahoo Finance.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, MSFT)
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
        
    Returns:
        List of StockBar objects, or None if download fails
    """
    try:
        # Download data from Yahoo Finance
        data = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            progress=False,
        )
        
        if data.empty:
            return None
        
        # Convert to StockBar objects
        bars: list[StockBar] = []
        for timestamp, row in data.iterrows():
            # Handle both single and multi-level column indexes
            if isinstance(row.index, pd.MultiIndex):
                open_val = row[('Open', symbol)]
                high_val = row[('High', symbol)]
                low_val = row[('Low', symbol)]
                close_val = row[('Close', symbol)]
                volume_val = row[('Volume', symbol)]
            else:
                open_val = row['Open']
                high_val = row['High']
                low_val = row['Low']
                close_val = row['Close']
                volume_val = row['Volume']
            
            # Skip rows with NaN values
            if pd.isna(open_val) or pd.isna(close_val):
                continue
            
            bar = StockBar(
                timestamp=timestamp.to_pydatetime().replace(tzinfo=timezone.utc),
                open=float(open_val),
                high=float(high_val),
                low=float(low_val),
                close=float(close_val),
                volume=float(volume_val),
            )
            bars.append(bar)
        
        return bars if bars else None
        
    except Exception as e:
        print(f"    Error details: {e}")
        return None


def save_bars_to_csv(bars: Iterable[StockBar], path: Path) -> None:
    """Save stock bars to CSV in Bybit-compatible format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Use same header format for compatibility
        writer.writerow([
            "open_time_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
        ])
        
        for bar in bars:
            writer.writerow([
                bar.timestamp.isoformat(),
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
                bar.volume * bar.close,  # Approximate turnover
            ])


def download_and_save_stock(
    symbol: str,
    start: str,
    end: str,
    interval: str,
    output_dir: Path,
) -> bool:
    """Download stock data and save to CSV.
    
    Returns:
        True if successful, False otherwise
    """
    print(f"📥 Downloading {symbol}...")
    
    bars = download_stock(symbol, start, end, interval)
    
    if not bars:
        print(f"  ⚠️  No data received for {symbol}")
        return False
    
    # Convert interval to filename format
    if interval.endswith("m"):
        interval_suffix = interval
    elif interval == "1h" or interval == "60m":
        interval_suffix = "1h"
    elif interval == "1d":
        interval_suffix = "1d"
    elif interval == "1wk":
        interval_suffix = "1w"
    elif interval == "1mo":
        interval_suffix = "1M"
    else:
        interval_suffix = interval
    
    # Save with USDT suffix for compatibility with simulator
    filename = f"{symbol}USDT_{interval_suffix}.csv"
    output_path = output_dir / filename
    save_bars_to_csv(bars, output_path)
    
    print(f"  ✅ Saved {len(bars)} bars to {filename}")
    return True


def main() -> None:
    """Main entry point for Yahoo Finance stock downloader."""
    parser = argparse.ArgumentParser(
        description="Download stock data from Yahoo Finance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        help="Data interval: 1m,5m,15m,30m,1h,1d,1wk,1mo (default: 1h)",
    )
    
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date in YYYY-MM-DD format",
    )
    
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date in YYYY-MM-DD format",
    )
    
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Stock ticker symbols (default: mix of major US stocks)",
    )
    
    parser.add_argument(
        "--preset",
        type=str,
        choices=["tech", "finance", "consumer", "healthcare", "all"],
        help="Use a preset list of stocks",
    )
    
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for CSV files",
    )
    
    args = parser.parse_args()
    
    # Determine which symbols to download
    if args.symbols:
        symbols = tuple(args.symbols)
    elif args.preset:
        if args.preset == "tech":
            symbols = TECH_STOCKS
        elif args.preset == "finance":
            symbols = FINANCIAL_STOCKS
        elif args.preset == "consumer":
            symbols = CONSUMER_STOCKS
        elif args.preset == "healthcare":
            symbols = HEALTHCARE_STOCKS
        elif args.preset == "all":
            symbols = TECH_STOCKS + FINANCIAL_STOCKS + CONSUMER_STOCKS + HEALTHCARE_STOCKS
        else:
            symbols = DEFAULT_STOCKS
    else:
        symbols = DEFAULT_STOCKS
    
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Yahoo Finance Stock Downloader")
    print(f"Interval: {args.interval}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Symbols: {len(symbols)} total")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Download data
    success_count = 0
    
    for symbol in symbols:
        success = download_and_save_stock(
            symbol=symbol,
            start=args.start,
            end=args.end,
            interval=args.interval,
            output_dir=output_dir,
        )
        if success:
            success_count += 1
    
    print("=" * 60)
    print(f"✅ Successfully downloaded {success_count}/{len(symbols)} symbols")
    print(f"📁 Data saved to: {output_dir}")
    
    if success_count == 0:
        print("\n⚠️  No data was downloaded. Check:")
        print("  1. Internet connection")
        print("  2. Stock symbols are valid")
        print("  3. Date range is valid (Yahoo Finance has limited intraday history)")
        print("\nNote: For intervals < 1 day, Yahoo Finance only provides ~60 days of history")


if __name__ == "__main__":
    main()

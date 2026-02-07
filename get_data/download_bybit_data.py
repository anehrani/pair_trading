"""Download Bybit TradFi tokenized stocks data for pair trading.

This script downloads historical OHLCV data for tokenized stocks from Bybit's public API.
Bybit offers perpetual futures on tokenized stocks (e.g., AAPLPERP, TSLAUSDT).

Bybit API Documentation:
- V5 API: https://bybit-exchange.github.io/docs/v5/market/kline
- No API key required for public market data

Example Usage:
  # Download tokenized stock data (default stocks)
  python -m get_data.download_bybit_data \
    --interval 60 \
    --start 2023-01-01 \
    --end 2024-12-31 \
    --category linear \
    --out data/bybit_stocks_1h

  # Download specific tokenized stocks
  python -m get_data.download_bybit_data \
    --interval 60 \
    --start 2023-01-01 \
    --end 2024-12-31 \
    --category linear \
    --symbols AAPLPERP TSLAUSDT GOOGUSDT \
    --out data/bybit_stocks_1h

Supported Categories:
- linear: USDT perpetual (most tokenized stocks use this)
- spot: Spot trading (limited for tokenized stocks)
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests


BYBIT_BASE_URL = "https://api.bybit.com"
KLINES_PATH = "/v5/market/kline"

# Tokenized stock symbols on Bybit
# These are perpetual futures on tokenized stocks
# Format varies: some use PERP suffix, others use USDT
DEFAULT_TRADFI_SYMBOLS: tuple[str, ...] = (
    # Tech Giants (FAANG+)
    "AAPLPERP",     # Apple
    "GOOGUSDT",     # Google/Alphabet
    "MSFTUSDT",     # Microsoft
    "AMZNUSDT",     # Amazon
    "METAUSDT",     # Meta (Facebook)
    "NFLXUSDT",     # Netflix
    
    # Electric Vehicles & Tech
    "TSLAUSDT",     # Tesla
    "NVDAUSDT",     # NVIDIA
    "AMDUSDT",      # AMD
    
    # Financial Services
    "JPMPERP",      # JPMorgan Chase
    "BAUSDT",       # Bank of America
    "GSUSDT",       # Goldman Sachs
    "VUSDT",        # Visa
    "MAUSDT",       # Mastercard
    
    # Consumer Brands
    "NKEUSDT",      # Nike
    "SBUXUSDT",     # Starbucks
    "MCUSDT",       # McDonald's
    "DISPUSDT",     # Disney
    
    # Healthcare & Pharma
    "JNJUSDT",      # Johnson & Johnson
    "PFUSDT",       # Pfizer
    "MRKNAUSDT",    # Moderna
    
    # Aerospace & Industrial
    "BAPERP",       # Boeing
    "GEPERP",       # General Electric
    
    # Others
    "BBAPERP",      # Alibaba
    "TSMUSDT",      # Taiwan Semiconductor
)

# Alternative format - check Bybit's current API for exact symbols
ALTERNATIVE_STOCK_SYMBOLS: tuple[str, ...] = (
    "AAPL-PERP",
    "TSLA-PERP",
    "GOOG-PERP",
    "MSFT-PERP",
    "AMZN-PERP",
)


@dataclass(frozen=True)
class Kline:
    """Bybit kline data structure."""
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float


def _parse_utc_date(date_str: str) -> datetime:
    """Parse YYYY-MM-DD date string to UTC datetime."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.replace(tzinfo=timezone.utc)


def _to_ms(dt: datetime) -> int:
    """Convert datetime to milliseconds timestamp."""
    return int(dt.timestamp() * 1000)


def fetch_klines(
    *,
    symbol: str,
    category: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    session: requests.Session,
    limit: int = 1000,
    pause_s: float = 0.5,
) -> list[Kline]:
    """Fetch klines for a single symbol from Bybit API.
    
    Args:
        symbol: Trading pair symbol (e.g., BTCUSDT)
        category: Product category (linear, inverse, spot)
        interval: Kline interval in minutes (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
        start_ms: Start timestamp in milliseconds
        end_ms: End timestamp in milliseconds
        session: Requests session for connection pooling
        limit: Max klines per request (max 1000 for Bybit)
        pause_s: Pause between requests to respect rate limits
        
    Returns:
        List of Kline objects
    """
    all_klines: list[Kline] = []
    current_start = start_ms
    
    url = f"{BYBIT_BASE_URL}{KLINES_PATH}"
    
    while current_start < end_ms:
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "start": current_start,
            "end": end_ms,
            "limit": limit,
        }
        
        try:
            resp = session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if data.get("retCode") != 0:
                error_msg = data.get("retMsg", "Unknown error")
                print(f"  ⚠️  API error for {symbol}: {error_msg}")
                break
            
            result = data.get("result", {})
            klines_data = result.get("list", [])
            
            if not klines_data:
                break
            
            # Bybit returns klines in reverse chronological order
            klines_data = list(reversed(klines_data))
            
            for item in klines_data:
                # Bybit V5 API format: [startTime, open, high, low, close, volume, turnover]
                kline = Kline(
                    open_time_ms=int(item[0]),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    turnover=float(item[6]),
                )
                all_klines.append(kline)
            
            # Update start time for next batch
            last_time = int(klines_data[-1][0])
            
            # Calculate interval in milliseconds
            if interval == "D":
                interval_ms = 24 * 60 * 60 * 1000
            elif interval == "W":
                interval_ms = 7 * 24 * 60 * 60 * 1000
            elif interval == "M":
                interval_ms = 30 * 24 * 60 * 60 * 1000
            else:
                interval_ms = int(interval) * 60 * 1000
            
            current_start = last_time + interval_ms
            
            if current_start >= end_ms:
                break
            
            time.sleep(pause_s)
            
        except requests.RequestException as e:
            print(f"  ⚠️  Network error for {symbol}: {e}")
            break
    
    return all_klines


def save_klines_to_csv(klines: Iterable[Kline], path: Path) -> None:
    """Save klines to CSV with consistent format for data_io.py compatibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Use same header format as Binance downloader for compatibility
        writer.writerow([
            "open_time_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
        ])
        
        for k in klines:
            dt = datetime.fromtimestamp(k.open_time_ms / 1000, tz=timezone.utc)
            writer.writerow([
                dt.isoformat(),
                k.open,
                k.high,
                k.low,
                k.close,
                k.volume,
                k.turnover,
            ])


def download_symbol(
    *,
    symbol: str,
    category: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    output_dir: Path,
    session: requests.Session,
) -> bool:
    """Download data for a single symbol and save to CSV.
    
    Returns:
        True if successful, False otherwise
    """
    print(f"📥 Downloading {symbol} ({category})...")
    
    try:
        klines = fetch_klines(
            symbol=symbol,
            category=category,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
            session=session,
        )
        
        if not klines:
            print(f"  ⚠️  No data received for {symbol}")
            return False
        
        # Convert interval to filename format (e.g., "60" -> "1h")
        if interval == "D":
            interval_suffix = "1d"
        elif interval == "W":
            interval_suffix = "1w"
        elif interval == "M":
            interval_suffix = "1M"
        else:
            minutes = int(interval)
            if minutes == 60:
                interval_suffix = "1h"
            elif minutes == 240:
                interval_suffix = "4h"
            elif minutes == 1440:
                interval_suffix = "1d"
            else:
                interval_suffix = f"{minutes}m"
        
        filename = f"{symbol}_{interval_suffix}.csv"
        output_path = output_dir / filename
        save_klines_to_csv(klines, output_path)
        
        print(f"  ✅ Saved {len(klines)} candles to {filename}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error downloading {symbol}: {e}")
        return False


def main() -> None:
    """Main entry point for Bybit data downloader."""
    parser = argparse.ArgumentParser(
        description="Download historical kline data from Bybit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--category",
        type=str,
        default="linear",
        choices=["linear", "inverse", "spot"],
        help="Product category (default: linear for USDT perpetuals)",
    )
    
    parser.add_argument(
        "--interval",
        type=str,
        default="60",
        help="Kline interval: 1,3,5,15,30,60,120,240,360,720,D,W,M (default: 60 for 1h)",
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
        help="Space-separated list of tokenized stock symbols (default: major US stocks)",
    )
    
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for CSV files",
    )
    
    args = parser.parse_args()
    
    # Parse dates
    start_dt = _parse_utc_date(args.start)
    end_dt = _parse_utc_date(args.end)
    start_ms = _to_ms(start_dt)
    end_ms = _to_ms(end_dt)
    
    # Determine symbols
    if args.symbols:
        symbols = tuple(args.symbols)
    else:
        symbols = DEFAULT_TRADFI_SYMBOLS
    
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Bybit Data Downloader")
    print(f"Category: {args.category}")
    print(f"Interval: {args.interval}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Symbols: {len(symbols)} total")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Download data
    session = requests.Session()
    success_count = 0
    
    for symbol in symbols:
        success = download_symbol(
            symbol=symbol,
            category=args.category,
            interval=args.interval,
            start_ms=start_ms,
            end_ms=end_ms,
            output_dir=output_dir,
            session=session,
        )
        if success:
            success_count += 1
    
    print("=" * 60)
    print(f"✅ Successfully downloaded {success_count}/{len(symbols)} symbols")
    print(f"📁 Data saved to: {output_dir}")


if __name__ == "__main__":
    main()

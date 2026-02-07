"""Download all available data for major indexes, commodities, and stocks from Yahoo Finance.

This script downloads the maximum available historical daily data for major
stock market indexes, commodities, and sector-representative stocks. Each asset 
is saved to a separate CSV file in OHLCV (Open, High, Low, Close, Volume) format.

Major Indexes Included:
- S&P 500 (^GSPC)
- Dow Jones Industrial Average (^DJI)
- NASDAQ Composite (^IXIC)
- Russell 2000 (^RUT)
- VIX Volatility Index (^VIX)
- FTSE 100 (^FTSE)
- DAX (^GDAXI)
- Nikkei 225 (^N225)
- Hang Seng (^HSI)

Commodities Included:
- Gold (GC=F)
- Silver (SI=F)
- Crude Oil WTI (CL=F)
- Brent Crude Oil (BZ=F)
- Natural Gas (NG=F)
- Copper (HG=F)
- Platinum (PL=F)
- Palladium (PA=F)
- Corn (ZC=F)
- Wheat (ZW=F)
- Soybeans (ZS=F)

Sector-Representative Stocks:
- Technology: AAPL, MSFT, NVDA, GOOGL
- Financials: JPM, BAC
- Healthcare: JNJ, UNH
- Consumer Discretionary: AMZN, HD
- Consumer Staples: PG, KO
- Industrials: CAT, BA
- Energy: XOM, CVX
- Materials: LIN
- Real Estate: AMT
- Utilities: NEE
- Communication Services: META, DIS

Example Usage:
  # Download all available data (default: indexes, commodities, and stocks)
  python -m get_data.download_all_data --out data/all_data

  # Download specific categories
  python -m get_data.download_all_data --categories indexes --out data/indexes_only
  python -m get_data.download_all_data --categories stocks --out data/stocks_only
  python -m get_data.download_all_data --categories indexes commodities stocks --out data/all_data

  # Specify custom date range (optional, defaults to maximum available)
  python -m get_data.download_all_data --start 2010-01-01 --end 2024-12-31 --out data/all_data
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


# Major Stock Market Indexes
MAJOR_INDEXES: dict[str, str] = {
    "^GSPC": "SP500",           # S&P 500
    "^DJI": "DowJones",          # Dow Jones Industrial Average
    "^IXIC": "NASDAQ",           # NASDAQ Composite
    "^RUT": "Russell2000",       # Russell 2000
    "^VIX": "VIX",               # CBOE Volatility Index
    "^FTSE": "FTSE100",          # FTSE 100 (UK)
    "^GDAXI": "DAX",             # DAX (Germany)
    "^N225": "Nikkei225",        # Nikkei 225 (Japan)
    "^HSI": "HangSeng",          # Hang Seng (Hong Kong)
}

# Major Commodities (Futures)
MAJOR_COMMODITIES: dict[str, str] = {
    "GC=F": "Gold",              # Gold Futures
    "SI=F": "Silver",            # Silver Futures
    "CL=F": "CrudeOilWTI",       # Crude Oil WTI Futures
    "BZ=F": "BrentCrude",        # Brent Crude Oil Futures
    "NG=F": "NaturalGas",        # Natural Gas Futures
    "HG=F": "Copper",            # Copper Futures
    "PL=F": "Platinum",          # Platinum Futures
    "PA=F": "Palladium",         # Palladium Futures
    "ZC=F": "Corn",              # Corn Futures
    "ZW=F": "Wheat",             # Wheat Futures
    "ZS=F": "Soybeans",          # Soybeans Futures
}

# Sector-Representative Stocks (1-2 highly liquid stocks per sector)
# These are large-cap, well-integrated stocks from major indexes
SECTOR_STOCKS: dict[str, str] = {
    # Technology (heavily represented in NASDAQ)
    "AAPL": "Apple",             # Apple Inc.
    "MSFT": "Microsoft",         # Microsoft Corporation
    "NVDA": "NVIDIA",            # NVIDIA Corporation
    "GOOGL": "Alphabet",         # Alphabet Inc. (Google)
    
    # Financials (major component of S&P 500 and Dow Jones)
    "JPM": "JPMorgan",           # JPMorgan Chase & Co.
    "BAC": "BankOfAmerica",     # Bank of America Corporation
    
    # Healthcare (defensive sector in all major indexes)
    "JNJ": "JohnsonJohnson",    # Johnson & Johnson
    "UNH": "UnitedHealth",       # UnitedHealth Group
    
    # Consumer Discretionary
    "AMZN": "Amazon",            # Amazon.com Inc.
    "HD": "HomeDepot",           # The Home Depot Inc.
    
    # Consumer Staples (defensive)
    "PG": "ProctorGamble",       # Procter & Gamble Co.
    "KO": "CocaCola",            # The Coca-Cola Company
    
    # Industrials (Dow Jones components)
    "CAT": "Caterpillar",        # Caterpillar Inc.
    "BA": "Boeing",              # The Boeing Company
    
    # Energy (cyclical sector)
    "XOM": "ExxonMobil",         # Exxon Mobil Corporation
    "CVX": "Chevron",            # Chevron Corporation
    
    # Materials
    "LIN": "LindeGroup",         # Linde plc
    
    # Real Estate
    "AMT": "AmericanTower",      # American Tower Corporation
    
    # Utilities (defensive)
    "NEE": "NextEraEnergy",      # NextEra Energy Inc.
    
    # Communication Services
    "META": "Meta",              # Meta Platforms Inc.
    "DIS": "Disney",             # The Walt Disney Company
}


@dataclass(frozen=True)
class AssetBar:
    """Asset price bar data in OHLCV format."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def download_asset(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
) -> list[AssetBar] | None:
    """Download asset data from Yahoo Finance.
    
    Args:
        symbol: Yahoo Finance ticker symbol (e.g., ^GSPC, GC=F)
        start: Start date in YYYY-MM-DD format (None for maximum history)
        end: End date in YYYY-MM-DD format (None for today)
        
    Returns:
        List of AssetBar objects, or None if download fails
    """
    try:
        # Download data from Yahoo Finance
        # If start/end not provided, yfinance will download maximum available history
        if start is None:
            # Get maximum available history
            data = yf.download(
                symbol,
                period="max",
                interval="1d",
                progress=False,
            )
        else:
            data = yf.download(
                symbol,
                start=start,
                end=end,
                interval="1d",
                progress=False,
            )
        
        if data.empty:
            return None
        
        # Convert to AssetBar objects
        bars: list[AssetBar] = []
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
            
            # Handle missing volume data (common for indexes)
            if pd.isna(volume_val):
                volume_val = 0.0
            
            bar = AssetBar(
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


def save_bars_to_csv(bars: Iterable[AssetBar], path: Path) -> None:
    """Save asset bars to CSV in OHLCV format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Standard OHLCV format header
        writer.writerow([
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ])
        
        for bar in bars:
            writer.writerow([
                bar.timestamp.isoformat(),
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
            ])


def download_and_save_asset(
    symbol: str,
    name: str,
    start: str | None,
    end: str | None,
    output_dir: Path,
) -> bool:
    """Download asset data and save to CSV.
    
    Args:
        symbol: Yahoo Finance ticker symbol
        name: Clean name for the asset (used in filename)
        start: Start date or None for maximum history
        end: End date or None for today
        output_dir: Directory to save CSV files
    
    Returns:
        True if successful, False otherwise
    """
    print(f"📥 Downloading {name} ({symbol})...")
    
    bars = download_asset(symbol, start, end)
    
    if not bars:
        print(f"  ⚠️  No data received for {name} ({symbol})")
        return False
    
    # Use clean asset name for filename
    filename = f"{name}_1d.csv"
    output_path = output_dir / filename
    save_bars_to_csv(bars, output_path)
    
    # Get date range
    first_date = bars[0].timestamp.strftime("%Y-%m-%d")
    last_date = bars[-1].timestamp.strftime("%Y-%m-%d")
    
    print(f"  ✅ Saved {len(bars)} bars ({first_date} to {last_date}) to {filename}")
    return True


def main() -> None:
    """Main entry point for downloading all available data."""
    parser = argparse.ArgumentParser(
        description="Download all available data for major indexes and commodities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--categories",
        nargs="*",
        choices=["indexes", "commodities", "stocks"],
        help="Categories to download (default: indexes, commodities, and stocks)",
    )
    
    parser.add_argument(
        "--start",
        type=str,
        help="Start date in YYYY-MM-DD format (default: maximum available)",
    )
    
    parser.add_argument(
        "--end",
        type=str,
        help="End date in YYYY-MM-DD format (default: today)",
    )
    
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for CSV files",
    )
    
    args = parser.parse_args()
    
    # Determine which categories to download
    download_indexes = True
    download_commodities = True
    download_stocks = True
    
    if args.categories:
        download_indexes = "indexes" in args.categories
        download_commodities = "commodities" in args.categories
        download_stocks = "stocks" in args.categories
    
    # Build list of assets to download
    assets_to_download: dict[str, str] = {}
    
    if download_indexes:
        assets_to_download.update(MAJOR_INDEXES)
    
    if download_commodities:
        assets_to_download.update(MAJOR_COMMODITIES)
    
    if download_stocks:
        assets_to_download.update(SECTOR_STOCKS)
    
    if not assets_to_download:
        print("❌ No assets selected for download")
        return
    
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Yahoo Finance Data Downloader - All Available History")
    categories_list = []
    if download_indexes:
        categories_list.append("Indexes")
    if download_commodities:
        categories_list.append("Commodities")
    if download_stocks:
        categories_list.append("Stocks")
    print(f"Categories: {' & '.join(categories_list) if categories_list else 'None'}")
    
    if args.start:
        print(f"Period: {args.start} to {args.end if args.end else 'today'}")
    else:
        print(f"Period: Maximum available history")
    
    print(f"Assets: {len(assets_to_download)} total")
    print(f"Output: {output_dir}")
    print(f"Format: OHLCV (Open, High, Low, Close, Volume)")
    print("=" * 70)
    
    # Download data
    success_count = 0
    
    for symbol, name in assets_to_download.items():
        success = download_and_save_asset(
            symbol=symbol,
            name=name,
            start=args.start,
            end=args.end,
            output_dir=output_dir,
        )
        if success:
            success_count += 1
    
    print("=" * 70)
    print(f"✅ Successfully downloaded {success_count}/{len(assets_to_download)} assets")
    print(f"📁 Data saved to: {output_dir}")
    
    if success_count == 0:
        print("\n⚠️  No data was downloaded. Check:")
        print("  1. Internet connection")
        print("  2. Yahoo Finance service availability")
        print("  3. Date range is valid")
    else:
        print(f"\n💡 Each asset saved to a separate CSV file")
        print(f"   Files use asset names (e.g., SP500_1d.csv, Gold_1d.csv)")


if __name__ == "__main__":
    main()

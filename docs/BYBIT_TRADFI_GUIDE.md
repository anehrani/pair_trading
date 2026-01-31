# Bybit Tokenized Stocks Pair Trading Guide

## Overview

This guide explains how to use Bybit's tokenized stock data for pair trading strategies. Bybit offers perpetual futures on major US stocks, allowing you to trade traditional financial assets with crypto-like liquidity.

## What are Tokenized Stocks?

Tokenized stocks on Bybit are perpetual futures contracts that track the price of real-world stocks:
- **Apple (AAPLPERP)** - Tracks AAPL stock price
- **Tesla (TSLAUSDT)** - Tracks TSLA stock price  
- **Google (GOOGUSDT)** - Tracks GOOGL stock price
- And 25+ more major US stocks

### Advantages for Pair Trading
1. **24/7 Trading** - Unlike stock markets, trade anytime
2. **Leverage** - Use leverage for larger positions
3. **No Settlement** - Perpetual contracts, no expiry
4. **Lower Fees** - Compared to traditional stock brokers
5. **Programmatic Access** - Free public API for historical data

## Supported Stock Symbols

The downloader includes these major tokenized stocks:

### Tech Giants (FAANG+)
- `AAPLPERP` - Apple
- `GOOGUSDT` - Google/Alphabet  
- `MSFTUSDT` - Microsoft
- `AMZNUSDT` - Amazon
- `METAUSDT` - Meta (Facebook)
- `NFLXUSDT` - Netflix

### Electric Vehicles & Semiconductors
- `TSLAUSDT` - Tesla
- `NVDAUSDT` - NVIDIA
- `AMDUSDT` - AMD

### Financial Services
- `JPMPERP` - JPMorgan Chase
- `BAUSDT` - Bank of America
- `GSUSDT` - Goldman Sachs
- `VUSDT` - Visa
- `MAUSDT` - Mastercard

### Consumer Brands
- `NKEUSDT` - Nike
- `SBUXUSDT` - Starbucks
- `MCUSDT` - McDonald's
- `DISPUSDT` - Disney

### Healthcare
- `JNJUSDT` - Johnson & Johnson
- `PFUSDT` - Pfizer
- `MRKNAUSDT` - Moderna

### Others
- `BAPERP` - Boeing
- `GEPERP` - General Electric
- `BBAPERP` - Alibaba
- `TSMUSDT` - Taiwan Semiconductor

> **Note**: Symbol availability may change. Always verify on [Bybit's platform](https://www.bybit.com/en/trade/usdt/) before downloading.

## Quick Start

### 1. Download Tokenized Stock Data

Download all default stocks:
```bash
python -m src.download_bybit_data \
  --category linear \
  --interval 60 \
  --start 2023-01-01 \
  --end 2024-12-31 \
  --out data/bybit_stocks_1h
```

Download specific stocks:
```bash
python -m src.download_bybit_data \
  --category linear \
  --interval 60 \
  --start 2023-01-01 \
  --end 2024-12-31 \
  --symbols AAPLPERP TSLAUSDT GOOGUSDT MSFTUSDT \
  --out data/bybit_stocks_1h
```

### 2. Run Pair Trading Simulation

Using the example script:
```bash
# Download data
python example_bybit_tradfi.py --download

# Run simulation
python example_bybit_tradfi.py --simulate
```

Or programmatically:
```python
from pathlib import Path
from src.performance_simulator import PerformanceSimulator, SimulationConfig

config = SimulationConfig(
    data_dir="data/bybit_stocks_1h",
    interval="1h",
    initial_capital=100_000,
    reference_symbol="AAPLPERP",  # Apple as reference
    alpha1=0.15,  # Entry threshold
    alpha2=0.08,  # Exit threshold
    formation_hours=30 * 24,  # 30 days
    trading_hours=10 * 24,    # 10 days
    position_pct=0.08,
    max_positions=8,
)

simulator = PerformanceSimulator(config)
results = simulator.run()
```

## Strategy Parameters for Stocks

Tokenized stocks have different characteristics than crypto:

### Recommended Parameters

| Parameter | Crypto | Stocks | Reason |
|-----------|--------|--------|--------|
| `alpha1` (entry) | 0.20 | 0.15 | Stocks less volatile |
| `alpha2` (exit) | 0.10 | 0.08 | Tighter mean reversion |
| `formation_hours` | 21 days | 30 days | Longer cointegration window |
| `trading_hours` | 7 days | 10 days | Longer trading window |
| `position_pct` | 5% | 8% | Less volatile, more stable |

### Why Different Parameters?

1. **Lower Volatility**: Stocks are typically less volatile than crypto
2. **Market Hours Effect**: Although 24/7, liquidity varies with US market hours
3. **Correlation Stability**: Stock pairs often have more stable correlations
4. **Lower Spreads**: Major stocks have tighter bid-ask spreads

## Finding Good Stock Pairs

### Sector-Based Pairs
Stocks in the same sector often cointegrate well:

**Tech Giants**:
- AAPL ↔ MSFT
- GOOGL ↔ META
- NVDA ↔ AMD

**Financial Services**:
- JPM ↔ GS
- V ↔ MA
- BAC ↔ JPM

**Consumer Brands**:
- NKE ↔ SBUX
- DIS ↔ NFLX

**EVs & Tech**:
- TSLA ↔ NVDA (AI/tech correlation)

### Cross-Sector Pairs
Sometimes interesting correlations exist across sectors:
- Tech + Finance (during economic cycles)
- Consumer + Healthcare (defensive pairs)

## Data Intervals

Bybit supports multiple timeframes:

```bash
# 1 minute
--interval 1

# 15 minutes  
--interval 15

# 1 hour (recommended for pair trading)
--interval 60

# 4 hours
--interval 240

# Daily
--interval D
```

**Recommendation**: Use `60` (1 hour) for pair trading as it:
- Provides enough data points
- Filters out noise
- Balances signal vs. computational cost

## API Rate Limits

Bybit's public API has rate limits:
- **50 requests/second** per IP
- The downloader includes 0.5s pause between requests
- No API key needed for public market data

## Output Format

Data is saved as CSV with format compatible with the pair trading simulator:

```csv
open_time_utc,open,high,low,close,volume,turnover
2023-01-01T00:00:00+00:00,150.50,151.20,150.10,150.80,125430.50,18921456.75
2023-01-01T01:00:00+00:00,150.80,151.50,150.60,151.20,98765.25,14908234.50
...
```

## Troubleshooting

### Symbol Not Found
```
⚠️  API error for AAPLPERP: symbol not found
```
**Solution**: Check Bybit's platform for exact symbol format. It might be:
- `AAPL-PERP` (with hyphen)
- `AAPLUSDT` (different suffix)
- Not available in your region

### No Data Received
```
⚠️  No data received for TSLAUSDT
```
**Solution**: 
1. Check if the symbol exists on Bybit
2. Try a different date range (data may not go back as far as crypto)
3. Verify the category is correct (`linear` for most stocks)

### Rate Limit Errors
```
⚠️  Network error: 429 Too Many Requests
```
**Solution**: Increase the pause between requests:
- Modify `pause_s` in `fetch_klines()` function
- Or download fewer symbols at once

## Advanced Usage

### Custom Symbol List

Create a file `my_stocks.txt`:
```
AAPLPERP
TSLAUSDT
GOOGUSDT
MSFTUSDT
AMZNUSDT
```

Download:
```bash
python -m src.download_bybit_data \
  --category linear \
  --interval 60 \
  --start 2023-01-01 \
  --end 2024-12-31 \
  --symbols $(cat my_stocks.txt | tr '\n' ' ') \
  --out data/bybit_custom_stocks
```

### Different Time Ranges for Testing

**Quick test** (1 month):
```bash
--start 2024-11-01 --end 2024-12-01
```

**Short backtest** (6 months):
```bash
--start 2024-06-01 --end 2024-12-31
```

**Full backtest** (2 years):
```bash
--start 2023-01-01 --end 2024-12-31
```

### Integration with Existing Workflow

The Bybit data format is compatible with the Binance downloader format, so you can:

1. Mix data sources (use some Binance crypto + some Bybit stocks)
2. Use the same `performance_simulator.py`
3. Use the same analysis tools

## Example Workflows

### Workflow 1: Tech Stock Pairs
```bash
# Download tech giants
python -m src.download_bybit_data \
  --symbols AAPLPERP MSFTUSDT GOOGUSDT AMZNUSDT METAUSDT \
  --interval 60 \
  --start 2023-01-01 \
  --end 2024-12-31 \
  --category linear \
  --out data/bybit_tech_stocks

# Run simulation
python -c "
from src.performance_simulator import PerformanceSimulator, SimulationConfig
config = SimulationConfig(
    data_dir='data/bybit_tech_stocks',
    reference_symbol='AAPLPERP',
    alpha1=0.15,
    alpha2=0.08,
    formation_hours=30*24,
    trading_hours=10*24,
)
simulator = PerformanceSimulator(config)
results = simulator.run()
"
```

### Workflow 2: Financial Services
```bash
# Download financial stocks
python -m src.download_bybit_data \
  --symbols JPMPERP BAUSDT GSUSDT VUSDT MAUSDT \
  --interval 60 \
  --start 2023-01-01 \
  --end 2024-12-31 \
  --category linear \
  --out data/bybit_finance_stocks

# Run with finance-specific parameters
python example_bybit_tradfi.py --simulate
```

## Production Considerations

### 1. Symbol Verification
Before production:
```python
import requests

def verify_symbol_exists(symbol: str, category: str = "linear") -> bool:
    """Check if a symbol exists on Bybit."""
    url = "https://api.bybit.com/v5/market/tickers"
    params = {"category": category, "symbol": symbol}
    resp = requests.get(url, params=params)
    data = resp.json()
    return data.get("retCode") == 0 and bool(data.get("result", {}).get("list"))

# Test before downloading
for symbol in ["AAPLPERP", "TSLAUSDT"]:
    exists = verify_symbol_exists(symbol)
    print(f"{symbol}: {'✅' if exists else '❌'}")
```

### 2. Data Quality Checks
After downloading:
```python
import pandas as pd
from pathlib import Path

def check_data_quality(csv_path: Path):
    """Verify downloaded data quality."""
    df = pd.read_csv(csv_path)
    
    # Check for missing data
    missing = df.isnull().sum()
    print(f"Missing values: {missing.sum()}")
    
    # Check date continuity
    df['open_time_utc'] = pd.to_datetime(df['open_time_utc'])
    gaps = df['open_time_utc'].diff().dt.total_seconds() / 3600
    large_gaps = gaps[gaps > 1.5]  # More than 1.5 hours for hourly data
    print(f"Data gaps: {len(large_gaps)}")
    
    # Check price sanity
    price_changes = df['close'].pct_change().abs()
    extreme_changes = price_changes[price_changes > 0.5]  # >50% change
    print(f"Extreme price changes: {len(extreme_changes)}")

# Check all downloaded files
for csv_file in Path("data/bybit_stocks_1h").glob("*.csv"):
    print(f"\n{csv_file.name}:")
    check_data_quality(csv_file)
```

### 3. Monitoring & Updates
Set up periodic data updates:
```bash
#!/bin/bash
# update_bybit_stocks.sh

# Get yesterday's date
END_DATE=$(date -v-1d +%Y-%m-%d)
START_DATE=$(date -v-7d +%Y-%m-%d)  # Last 7 days

python -m src.download_bybit_data \
  --category linear \
  --interval 60 \
  --start $START_DATE \
  --end $END_DATE \
  --out data/bybit_stocks_1h

echo "✅ Updated data through $END_DATE"
```

## Resources

- [Bybit API Documentation](https://bybit-exchange.github.io/docs/v5/intro)
- [Bybit Market Data API](https://bybit-exchange.github.io/docs/v5/market/kline)
- [Available Trading Pairs](https://www.bybit.com/en/trade/usdt)
- [Bybit API Status](https://bybit-exchange.github.io/docs/v5/rate-limit)

## Next Steps

1. **Download sample data** - Start with a few stocks for 1 month
2. **Run test simulation** - Verify the strategy works
3. **Optimize parameters** - Tune alpha1, alpha2, formation period
4. **Scale up** - Download more stocks and longer periods
5. **Deploy** - Integrate with your trading system

## Support

For issues or questions:
1. Check symbol availability on Bybit's platform
2. Verify API status and rate limits
3. Review error messages in terminal output
4. Check the `reports/` directory for simulation results

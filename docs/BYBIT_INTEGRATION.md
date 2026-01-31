# Bybit Integration Guide

## Overview

This guide explains how to use Bybit market data (including TradFi assets) with the pair trading strategy.

## What's New

### 1. Bybit Data Downloader
- **File**: `src/download_bybit_data.py`
- Downloads historical OHLCV data from Bybit's V5 API
- Supports crypto perpetuals, inverse perpetuals, and spot markets
- Compatible with existing pair trading infrastructure

### 2. Updated Data I/O
- **File**: `src/data_io.py`
- Now supports both Binance and Bybit data formats
- Seamless integration - no code changes needed in strategy

### 3. Example Scripts
- **File**: `example_bybit_tradfi.py`
- Examples for using Bybit TradFi data
- Quick start templates

## Quick Start

### Step 1: Download Bybit Data

#### Crypto Perpetuals (USDT-margined)
```bash
python -m src.download_bybit_data \
  --category linear \
  --interval 60 \
  --start 2022-01-01 \
  --end 2024-12-31 \
  --out data/bybit_linear_1h
```

#### Specific Symbols
```bash
python -m src.download_bybit_data \
  --category linear \
  --interval 60 \
  --start 2022-01-01 \
  --end 2024-12-31 \
  --symbols BTCUSDT ETHUSDT SOLUSDT ADAUSDT \
  --out data/bybit_custom_1h
```

#### Different Intervals
```bash
# 5-minute data
python -m src.download_bybit_data \
  --category linear \
  --interval 5 \
  --start 2023-01-01 \
  --end 2024-12-31 \
  --out data/bybit_linear_5m

# 4-hour data
python -m src.download_bybit_data \
  --category linear \
  --interval 240 \
  --start 2022-01-01 \
  --end 2024-12-31 \
  --out data/bybit_linear_4h

# Daily data
python -m src.download_bybit_data \
  --category linear \
  --interval D \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --out data/bybit_linear_1d
```

### Step 2: Run Pair Trading Strategy

#### Using Python Script
```python
from pathlib import Path
from src.performance_simulator import PerformanceSimulator, SimulationConfig

# Configure simulation
config = SimulationConfig(
    data_dir="data/bybit_linear_1h",
    interval="1h",
    initial_capital=100_000,
    reference_symbol="BTCUSDT",
    alpha1=0.20,
    alpha2=0.10,
    formation_hours=21 * 24,
    trading_hours=7 * 24,
    step_hours=7 * 24,
    trading_fee_pct=0.06,  # Bybit fees
    report_dir="reports/bybit_simulation",
)

# Run simulation
simulator = PerformanceSimulator(config)
results = simulator.run()
```

#### Using Example Script
```bash
python example_bybit_tradfi.py
```

### Step 3: View Results

Results are saved in the configured `report_dir`:
- `report_YYYYMMDD_HHMMSS.txt` - Text summary
- `report_YYYYMMDD_HHMMSS.json` - JSON format
- `equity_YYYYMMDD_HHMMSS.csv` - Equity curve
- `cycles_YYYYMMDD_HHMMSS.csv` - Cycle statistics

## Bybit API Details

### Supported Categories
- `linear` - USDT perpetual futures (most common)
- `inverse` - Inverse perpetual futures
- `spot` - Spot trading

### Supported Intervals
- Minutes: `1`, `3`, `5`, `15`, `30`, `60`, `120`, `240`, `360`, `720`
- Daily: `D`
- Weekly: `W`
- Monthly: `M`

### Rate Limits
- Public API: No authentication required
- Rate limit: ~50 requests/second
- The script includes automatic rate limiting (0.5s pause between requests)

### Data Availability
- Different symbols have different history depths
- Newer symbols may have limited historical data
- Check Bybit's documentation for specific symbol availability

## TradFi Assets on Bybit

### Important Note
Bybit's TradFi offerings vary by region and over time. Before using TradFi data:

1. **Verify Availability**: Check [Bybit's trading page](https://www.bybit.com/en/trade/usdt)
2. **Check Liquidity**: Ensure sufficient trading volume
3. **Market Hours**: Some TradFi assets may have limited trading hours
4. **Data History**: Verify available historical data depth

### Example TradFi Symbols (Verify Current Availability)
```python
# These are examples - check current Bybit offerings
tradfi_symbols = [
    # Commodities
    "XAUUSDT",  # Gold
    "XAGUSDT",  # Silver
    
    # Indices (if available)
    # Check Bybit for current offerings
]
```

## Comparing Binance vs Bybit Data

Both data sources work with the same pair trading strategy:

| Feature | Binance | Bybit |
|---------|---------|-------|
| API Access | Public, no key needed | Public, no key needed |
| Data Format | Compatible ✓ | Compatible ✓ |
| Crypto Assets | Extensive | Extensive |
| TradFi Assets | Limited | Varies by region |
| Historical Depth | Good | Good |
| Update Frequency | Real-time | Real-time |

## Common Issues and Solutions

### Issue: "No data received"
**Solution**: 
- Verify symbol exists on Bybit
- Check date range (symbol may not have existed in that period)
- Ensure internet connection is stable

### Issue: "API error"
**Solution**:
- Check rate limiting (wait a moment and retry)
- Verify symbol format (e.g., "BTCUSDT" not "BTC-USDT")
- Check Bybit API status

### Issue: "Different CSV format"
**Solution**:
- The downloader creates compatible CSV format automatically
- If you have custom Bybit data, ensure it has these columns:
  - `open_time_utc` (ISO8601 format with timezone)
  - `open`, `high`, `low`, `close`, `volume`

### Issue: "Strategy not finding pairs"
**Solution**:
- Ensure sufficient historical data (at least formation_hours + trading_hours)
- Check for missing data gaps
- Try different alpha1/alpha2 parameters

## Advanced Usage

### Custom Symbol Lists

Create a text file with symbols:
```bash
# symbols.txt
BTCUSDT
ETHUSDT
SOLUSDT
AVAXUSDT
MATICUSDT
```

Then use in download:
```bash
python -m src.download_bybit_data \
  --category linear \
  --interval 60 \
  --start 2022-01-01 \
  --end 2024-12-31 \
  --symbols $(cat symbols.txt) \
  --out data/bybit_custom_1h
```

### Mixing Data Sources

You can combine Binance and Bybit data in the same simulation:
1. Download data from both sources to the same directory
2. Ensure symbols don't overlap (or overwrite is intentional)
3. Run simulation as normal

### Automated Updates

Create a cron job or scheduled task to update data regularly:
```bash
#!/bin/bash
# update_bybit_data.sh

python -m src.download_bybit_data \
  --category linear \
  --interval 60 \
  --start $(date -d "7 days ago" +%Y-%m-%d) \
  --end $(date +%Y-%m-%d) \
  --out data/bybit_linear_1h
```

## API Reference

### download_bybit_data.py

```python
python -m src.download_bybit_data [OPTIONS]

Options:
  --category TEXT      Product category [linear|inverse|spot]
  --interval TEXT      Kline interval (1,3,5,15,30,60,120,240,360,720,D,W,M)
  --start TEXT         Start date (YYYY-MM-DD)
  --end TEXT           End date (YYYY-MM-DD)
  --symbols [TEXT...]  Space-separated list of symbols
  --out TEXT           Output directory
  --help               Show help message
```

## Best Practices

1. **Start Small**: Test with a small date range first
2. **Verify Data**: Check downloaded CSV files before running strategy
3. **Monitor Rate Limits**: Don't hammer the API
4. **Handle Errors**: The downloader skips failed symbols and continues
5. **Keep Backups**: Save downloaded data to avoid re-downloading
6. **Update Regularly**: For live trading, update data frequently
7. **Validate Results**: Compare with other data sources when possible

## Support

- Bybit API Documentation: https://bybit-exchange.github.io/docs/v5/intro
- Bybit API Status: https://bybit-exchange.github.io/docs/v5/announcement
- Project Issues: Use GitHub issues for bug reports

## Next Steps

1. Download sample data and test the integration
2. Compare results between Binance and Bybit data
3. Explore TradFi assets if available in your region
4. Optimize strategy parameters for your chosen assets
5. Consider implementing live trading with Bybit's trading API

## License

Same as main project (see LICENSE file)

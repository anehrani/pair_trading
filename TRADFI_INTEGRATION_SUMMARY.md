# TradFi Data Integration Summary

## What Was Implemented

I've set up a comprehensive solution for using traditional finance (TradFi) data with your pair trading system, with two approaches:

### 1. Bybit Downloader (for crypto perpetuals)
**File**: [`src/download_bybit_data.py`](src/download_bybit_data.py)

- Downloads crypto perpetual futures from Bybit's free public API
- Originally intended for tokenized stocks, but those appear unavailable on Bybit currently
- Works perfectly with 17+ major crypto pairs (BTC, ETH, BNB, SOL, etc.)

### 2. Yahoo Finance Downloader (for real stocks) ✅ RECOMMENDED
**File**: [`src/download_yahoo_stocks.py`](src/download_yahoo_stocks.py)

- Downloads real US stock data from Yahoo Finance (100% free)
- Supports 40+ major stocks across sectors:
  - Tech: AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA
  - Finance: JPM, BAC, GS, V, MA
  - Consumer: WMT, NKE, SBUX, DIS, MCD
  - Healthcare: JNJ, PFE, UNH, MRK
- Compatible with your existing simulator
- Multiple timeframes: 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo

## Quick Start: Download Real Stock Data

### Step 1: Install yfinance
```bash
cd /Users/alinehrani/projects/git_anehrani/pair_trading
pip install yfinance
# or
pip install -r requirements.txt
```

### Step 2: Download Stock Data
```bash
# Download all default stocks (tech, finance, consumer, healthcare)
python -m src.download_yahoo_stocks \
  --interval 1h \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --out data/stocks_1h

# Or download specific stocks
python -m src.download_yahoo_stocks \
  --interval 1h \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --symbols AAPL MSFT GOOGL TSLA NVDA \
  --out data/stocks_1h

# Or use presets
python -m src.download_yahoo_stocks \
  --interval 1h \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --preset tech \
  --out data/tech_stocks_1h
```

### Step 3: Run Pair Trading on Stocks
```python
from src.performance_simulator import PerformanceSimulator, SimulationConfig

config = SimulationConfig(
    data_dir="data/stocks_1h",
    interval="1h",
    initial_capital=100_000,
    reference_symbol="AAPLUSDT",  # Apple as reference
    alpha1=0.15,  # Entry threshold (lower for stocks)
    alpha2=0.08,  # Exit threshold
    formation_hours=30 * 24,  # 30 days
    trading_hours=10 * 24,    # 10 days
    position_pct=0.08,        # 8% per position
    max_positions=8,
)

simulator = PerformanceSimulator(config)
results = simulator.run()
```

## Files Created

1. **`src/download_bybit_data.py`**
   - Bybit API integration
   - Supports crypto perpetuals
   - Updated with proper error handling

2. **`src/download_yahoo_stocks.py`** ⭐ NEW
   - Yahoo Finance integration
   - Free real stock data
   - 40+ default stocks
   - Multiple presets (tech, finance, consumer, healthcare)

3. **`example_bybit_tradfi.py`**
   - Example usage script
   - Pre-configured for stocks
   - Easy command-line interface

4. **Documentation**:
   - [`docs/BYBIT_TRADFI_GUIDE.md`](docs/BYBIT_TRADFI_GUIDE.md) - Comprehensive guide
   - [`docs/BYBIT_STOCKS_STATUS.md`](docs/BYBIT_STOCKS_STATUS.md) - Status update and alternatives
   - This summary file

## Important Discovery

**Bybit Tokenized Stocks**: Testing revealed that Bybit's tokenized stocks (AAPLPERP, TSLAUSDT, etc.) are **currently unavailable** through their public API. This is likely due to:
- Regulatory changes around tokenized securities
- Regional restrictions
- Product discontinuation

**Solution**: Use the Yahoo Finance downloader for real stock data instead. It's:
- ✅ Free
- ✅ Reliable
- ✅ Easy to use
- ✅ Has all major US stocks
- ✅ Compatible with your existing system

## Data Format Compatibility

Both downloaders output CSV files in the same format:

```csv
open_time_utc,open,high,low,close,volume,turnover
2024-01-01T00:00:00+00:00,150.50,151.20,150.10,150.80,125430.50,18921456.75
2024-01-01T01:00:00+00:00,150.80,151.50,150.60,151.20,98765.25,14908234.50
```

This format is compatible with your existing:
- `data_io.py`
- `performance_simulator.py`
- `copula_model.py`

## Stock Pair Trading Strategy Tips

### Recommended Parameter Adjustments

Stocks behave differently than crypto:

| Parameter | Crypto | Stocks | Reason |
|-----------|--------|--------|--------|
| `alpha1` | 0.20 | 0.15 | Lower volatility |
| `alpha2` | 0.10 | 0.08 | Tighter mean reversion |
| `formation_hours` | 21 days | 30 days | More stable correlations |
| `trading_hours` | 7 days | 10 days | Longer positions |
| `position_pct` | 5% | 8% | Less volatile |

### Good Stock Pairs to Try

**Tech Giants**:
- AAPL ↔ MSFT (similar business models)
- GOOGL ↔ META (ad-based revenue)
- NVDA ↔ AMD (semiconductor competitors)

**Financial Services**:
- JPM ↔ BAC (large banks)
- V ↔ MA (payment processors)
- GS ↔ MS (investment banks)

**Consumer Brands**:
- WMT ↔ TGT (retail)
- NKE ↔ LULU (athletic wear)
- SBUX ↔ MCD (food/beverage)

## Yahoo Finance Limitations

**Intraday Data**: Yahoo Finance only provides ~60 days of intraday data (1m, 5m, 15m, 30m, 1h)

**Workarounds**:
1. Use daily data (`1d`) for longer backtests (years of history)
2. Download data regularly and maintain your own database
3. For production, consider paid data providers (Polygon.io, Alpha Vantage Pro)

**No Limitations on Daily Data**: Can download years of daily data for any stock

## Example Workflows

### Workflow 1: Quick Test (1 Month of Tech Stocks)
```bash
python -m src.download_yahoo_stocks \
  --preset tech \
  --interval 1h \
  --start 2024-11-01 \
  --end 2024-12-01 \
  --out data/test_stocks_1h

python -c "
from src.performance_simulator import PerformanceSimulator, SimulationConfig
config = SimulationConfig(
    data_dir='data/test_stocks_1h',
    reference_symbol='AAPLUSDT',
    alpha1=0.15, alpha2=0.08,
    formation_hours=7*24, trading_hours=3*24,
)
PerformanceSimulator(config).run()
"
```

### Workflow 2: Full Backtest (Daily Data, 2+ Years)
```bash
python -m src.download_yahoo_stocks \
  --interval 1d \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --out data/stocks_daily

python -c "
from src.performance_simulator import PerformanceSimulator, SimulationConfig
config = SimulationConfig(
    data_dir='data/stocks_daily',
    interval='1d',
    reference_symbol='AAPLUSDT',
    alpha1=0.15, alpha2=0.08,
    formation_hours=30*24, trading_hours=10*24,
)
PerformanceSimulator(config).run()
"
```

### Workflow 3: Mixed Sectors for Diversification
```bash
# Download stocks from different sectors
python -m src.download_yahoo_stocks \
  --interval 1h \
  --start 2024-06-01 \
  --end 2024-12-31 \
  --symbols AAPL JPM WMT JNJ TSLA V NKE \
  --out data/mixed_stocks_1h
```

## Testing & Validation

Run this to verify everything works:

```bash
# 1. Install dependencies
pip install yfinance

# 2. Download sample data (just 3 stocks, 1 week)
python -m src.download_yahoo_stocks \
  --symbols AAPL MSFT GOOGL \
  --interval 1d \
  --start 2024-12-01 \
  --end 2024-12-31 \
  --out data/test_stocks

# 3. Check the files were created
ls -lh data/test_stocks/

# 4. Run a quick simulation
python -c "
from pathlib import Path
from src.performance_simulator import PerformanceSimulator, SimulationConfig

if Path('data/test_stocks').exists():
    config = SimulationConfig(
        data_dir='data/test_stocks',
        interval='1d',
        reference_symbol='AAPLUSDT',
        formation_hours=7*24,
        trading_hours=3*24,
    )
    results = PerformanceSimulator(config).run()
    print('✅ Test successful!')
else:
    print('❌ Data directory not found')
"
```

## Next Steps

1. **Install yfinance**: `pip install yfinance`

2. **Download stock data**: Use the Yahoo Finance downloader

3. **Test with a few stocks**: Start with 3-5 stocks, short period

4. **Optimize parameters**: Tune alpha1, alpha2, formation period

5. **Scale up**: Once working, download more stocks and longer periods

6. **Monitor performance**: Track results in the `reports/` directory

## Alternative Data Sources (For Future)

If you need more advanced features later:

1. **Alpha Vantage** - Free tier + paid plans, good API
2. **Polygon.io** - Professional quality, real-time data, paid
3. **Interactive Brokers** - For live trading with real stocks
4. **Quandl/Nasdaq Data Link** - Financial datasets
5. **Bloomberg/Reuters** - Enterprise-grade (expensive)

## Support

For questions or issues:
1. Check Yahoo Finance stock symbols are correct
2. Verify internet connection
3. Note the 60-day limit on intraday data
4. Use daily data for longer backtests

All code is documented and ready to use! 🚀

# ⚠️ Important Note: Bybit Tokenized Stocks Status

## Current Situation (January 2026)

After testing the Bybit API, **tokenized stocks (AAPLPERP, TSLAUSDT, etc.) appear to be unavailable** through Bybit's public API at this time. This could be due to:

1. **Regulatory Changes**: Tokenized securities face regulatory scrutiny
2. **Regional Restrictions**: May be unavailable in certain regions
3. **Product Discontinuation**: Bybit may have discontinued this product line
4. **API Access Limitations**: May require special API keys or permissions

## Alternative Solutions for TradFi Pair Trading

### Option 1: Use Crypto Pairs (Recommended for Now)

The existing downloader works perfectly with crypto pairs which still show strong pair trading opportunities:

```bash
# Download major crypto pairs
python -m src.download_bybit_data \
  --category linear \
  --interval 60 \
  --start 2023-01-01 \
  --end 2024-12-31 \
  --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT ADAUSDT \
  --out data/bybit_crypto_1h
```

**Why crypto can still be valuable:**
- High liquidity 24/7
- Strong cointegration relationships exist
- Lower regulatory barriers
- Well-tested market infrastructure

### Option 2: Use Traditional Stock Data APIs

For actual stock pair trading, use these alternatives:

#### A. Yahoo Finance (Free)
```python
import yfinance as yf
import pandas as pd

# Download stock data
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
data = yf.download(tickers, start='2023-01-01', end='2024-12-31', interval='1h')

# Save in compatible format
for ticker in tickers:
    df = pd.DataFrame({
        'open_time_utc': data.index,
        'open': data['Open'][ticker],
        'high': data['High'][ticker],
        'low': data['Low'][ticker],
        'close': data['Close'][ticker],
        'volume': data['Volume'][ticker],
        'turnover': data['Volume'][ticker] * data['Close'][ticker]
    })
    df.to_csv(f'data/stocks_1h/{ticker}USDT_1h.csv', index=False)
```

Install: `pip install yfinance`

#### B. Alpha Vantage (Free tier available)
```python
from alpha_vantage.timeseries import TimeSeries
import pandas as pd

api_key = 'YOUR_API_KEY'  # Get from https://www.alphavantage.co/support/#api-key
ts = TimeSeries(key=api_key, output_format='pandas')

# Download intraday data
data, meta_data = ts.get_intraday('AAPL', interval='60min', outputsize='full')

# Convert to compatible format
df = pd.DataFrame({
    'open_time_utc': data.index,
    'open': data['1. open'],
    'high': data['2. high'],
    'low': data['3. low'],
    'close': data['4. close'],
    'volume': data['5. volume'],
    'turnover': data['5. volume'] * data['4. close']
})
df.to_csv('data/stocks_1h/AAPLUSDT_1h.csv', index=False)
```

Install: `pip install alpha-vantage`

#### C. Polygon.io (Paid, professional quality)
```python
from polygon import RESTClient
import pandas as pd
from datetime import datetime

client = RESTClient(api_key='YOUR_API_KEY')

# Get hourly bars
aggs = client.get_aggs('AAPL', 1, 'hour', '2023-01-01', '2024-12-31')

# Convert to compatible format
df = pd.DataFrame([{
    'open_time_utc': datetime.fromtimestamp(agg.timestamp / 1000),
    'open': agg.open,
    'high': agg.high,
    'low': agg.low,
    'close': agg.close,
    'volume': agg.volume,
    'turnover': agg.volume * agg.vwap
} for agg in aggs])

df.to_csv('data/stocks_1h/AAPLUSDT_1h.csv', index=False)
```

Website: https://polygon.io

### Option 3: Use Interactive Brokers (IB)

For live trading with real stocks:

```python
from ib_insync import *
import pandas as pd

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Download historical data
contract = Stock('AAPL', 'SMART', 'USD')
bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='365 D',
    barSizeSetting='1 hour',
    whatToShow='TRADES',
    useRTH=False
)

# Convert to DataFrame
df = util.df(bars)
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'average', 'barCount']
df['open_time_utc'] = df['date']
df['turnover'] = df['volume'] * df['average']
df = df[['open_time_utc', 'open', 'high', 'low', 'close', 'volume', 'turnover']]

df.to_csv('data/stocks_1h/AAPLUSDT_1h.csv', index=False)
```

Install: `pip install ib_insync`

### Option 4: Update the Bybit Script for Crypto

Since Bybit's crypto perpetuals work well, here's the recommended update:

```bash
# Download crypto pairs that have good pair trading characteristics
python -m src.download_bybit_data \
  --category linear \
  --interval 60 \
  --start 2023-01-01 \
  --end 2024-12-31 \
  --out data/bybit_crypto_1h

# This will download default crypto symbols:
# BTCUSDT, ETHUSDT, BCHUSDT, XRPUSDT, EOSUSDT, LTCUSDT,
# TRXUSDT, ETCUSDT, LINKUSDT, XLMUSDT, ADAUSDT, DOTUSDT,
# MATICUSDT, AVAXUSDT, SOLUSDT, ATOMUSDT, BNBUSDT
```

### Option 5: Create Your Own Stock Data Downloader

I can create a new downloader using Yahoo Finance that mimics the Bybit format:

```python
# src/download_yahoo_stocks.py
"""Download stock data from Yahoo Finance in Bybit-compatible format."""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime

# Major US stocks
DEFAULT_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'TSLA', 'NVDA', 'AMD', 'JPM', 'BAC',
    'GS', 'V', 'MA', 'NKE', 'SBUX', 'DIS',
    'JNJ', 'PFE', 'BA', 'GE'
]

def download_stock_data(
    symbols: list[str],
    start_date: str,
    end_date: str,
    interval: str = '1h',
    output_dir: str = 'data/yahoo_stocks_1h'
):
    """Download stock data from Yahoo Finance."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for symbol in symbols:
        print(f"📥 Downloading {symbol}...")
        try:
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False
            )
            
            # Convert to Bybit-compatible format
            df = pd.DataFrame({
                'open_time_utc': data.index,
                'open': data['Open'],
                'high': data['High'],
                'low': data['Low'],
                'close': data['Close'],
                'volume': data['Volume'],
                'turnover': data['Volume'] * data['Close']
            })
            
            # Save with Bybit-style filename
            filename = f"{symbol}USDT_{interval}.csv"
            df.to_csv(output_path / filename, index=False)
            print(f"  ✅ Saved {len(df)} candles to {filename}")
            
        except Exception as e:
            print(f"  ❌ Error downloading {symbol}: {e}")

if __name__ == '__main__':
    download_stock_data(
        symbols=DEFAULT_STOCKS,
        start_date='2023-01-01',
        end_date='2024-12-31',
        interval='1h',
        output_dir='data/yahoo_stocks_1h'
    )
```

## Recommended Path Forward

**For immediate use:**
1. Use Bybit for crypto pairs (works perfectly)
2. Or use Yahoo Finance for stock data (free, reliable)

**For production trading:**
1. Interactive Brokers for live stock trading
2. Bybit or other crypto exchanges for crypto pairs
3. Mix both asset classes with proper risk management

## Updating Your Code

To switch from planned tokenized stocks to crypto:

```python
# In your simulation config
config = SimulationConfig(
    data_dir="data/bybit_crypto_1h",  # Changed from bybit_stocks_1h
    reference_symbol="BTCUSDT",        # Changed from AAPLPERP
    # Keep other parameters the same
    alpha1=0.20,  # Crypto is more volatile, so higher thresholds
    alpha2=0.10,
)
```

## Questions?

If you specifically need stock data for pair trading:
1. Use Yahoo Finance downloader (I can create this for you)
2. Use Interactive Brokers API
3. Use Alpha Vantage or Polygon.io

Let me know which direction you'd like to go!

# Copula-Based Pair Trading - Live Trading System

Implementation of the copula-based pair trading algorithm from the research paper "Copula-based trading of cointegrated cryptocurrency Pairs" (Tadi & Witzany, 2025) for live trading on OKX.

## Features

- **Reference-Asset-Based Copula Approach**: Uses Bitcoin as reference asset with cointegration tests (Engle-Granger + KSS)
- **Automated Pair Selection**: Identifies top 2 cointegrated altcoins using Kendall's tau
- **Copula Modeling**: Fits various copula families (Gaussian, Student-t, Clayton, Gumbel, Frank) with rotations
- **Live Trading**: Executes trades automatically on OKX demo trading account
- **Rolling Windows**: 21-day formation period + 7-day trading period
- **Risk Management**: Position sizing, fee tracking, state persistence

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
.venv/bin/python -m pip install -e .

# Create .env file from template
cp .env.template .env
```

### 2. Get OKX Demo Trading Credentials

1. **Go to OKX** and create demo trading account:
   - URL: https://www.okx.com/account/my-api
   - Switch to **Demo Trading** mode (toggle in top right)

2. **Create API Key**:
   - Click "Create API Key"
   - Permissions: Select "Trade" (Read and Write)
   - Set a passphrase (you'll need to remember this!)
   - **IMPORTANT**: If creating key for the first time, you might need to whitelist your IP or disable IP restriction.
   - Copy: API Key, Secret Key, and Passphrase

3. **Update `.env` file**:
   ```
   OKX_API_KEY=your_api_key_here
   OKX_API_SECRET=your_secret_key_here
   OKX_PASSPHRASE=your_passphrase_here
   ```

### 3. Test Connection

```bash
.venv/bin/python test_okx_connection.py
```

### 4. Run Live Trader

```bash
.venv/bin/python src/live_trader.py
```

## Configuration

Edit `config.yaml` to customize:

- **Trading Parameters**: `alpha1` (entry threshold), `alpha2` (exit threshold)
- **Capital**: `capital_per_side` (USDT per leg)
- **Symbols**: List of cryptocurrencies to trade (OKX perpetual swap format)
- **Time Windows**: `formation_days`, `trading_days`

## Project Structure

```
pair_trading/
├── src/
│   ├── okx_client.py              # OKX API client (Manual implementation)
│   ├── data_buffer.py             # Rolling window data management
│   ├── strategy_core.py           # Pair selection & signal generation
│   ├── live_trader.py             # Main orchestrator
│   ├── copula_model.py            # Copula fitting (from backtest)
│   ├── stats_tests.py             # Cointegration tests (from backtest)
│   └── backtest_reference_copula.py  # Backtesting engine
├── config.yaml                    # Configuration
├── .env                           # API credentials (create from .env.template)
├── data/                          # Price buffer & state files
└── logs/                          # Trading logs
```

## How It Works

### Formation Period (21 days)
1. Download historical 5-min price data
2. Test cointegration with Bitcoin for all altcoins (EG + KSS tests)
3. Rank cointegrated pairs by Kendall's tau
4. Select top 2 altcoins
5. Fit copula model to spread processes

### Trading Period (7 days)
1. Stream real-time 5-min prices
2. Calculate conditional probabilities h₁|₂ and h₂|₁
3. Generate signals based on thresholds:
   - **Entry**: h₁|₂ < α₁ and h₂|₁ > 1-α₁ (or opposite)
   - **Exit**: |h₁|₂ - 0.5| < α₂ and |h₂|₁ - 0.5| < α₂
4. Execute market orders on OKX
5. Track positions and P&L

### Cycle Repeats
- After 7-day trading period, close positions
- Start new 21-day formation period
- Select new pair and refit copula

## Paper Results

From the research paper (5-min data, α₁=0.20, α₂=0.10):
- **Annualized Return**: 75.2%
- **Sharpe Ratio**: 3.77
- **Max Drawdown**: -30.5%
- **Total Return**: 205.9% over 2 years

## Monitoring

- **Logs**: `logs/trading_YYYY-MM-DD.log`
- **State**: `data/state.json` (current positions, P&L, cycle info)
- **Price Buffer**: `data/price_buffer.parquet` (historical data cache)

## Safety Notes

- This is for **demo trading only** (OKX demo account)
- No real money is at risk
- Risk management is disabled per user preference
- Always monitor logs for errors

## Comparison with Bybit

This implementation uses OKX instead of Bybit because:
- Better demo trading environment
- No IP restrictions on demo accounts (easier to manage)
- More reliable API for testing

Symbol format differences:
- **OKX**: `BTC-USDT-SWAP` (perpetual swaps)
- **Bybit**: `BTCUSDT` (linear perpetuals)

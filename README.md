# Copula-Based Cryptocurrency Pairs Trading

Implementation of the copula-based pairs trading algorithm from:

**Tadi, M., & Witzany, J. (2025).** "Copulas in Cryptocurrency Pairs Trading: An Innovative Approach to Trading Strategies." *Financial Innovation*, 11:40.

## Overview

This repository implements a reference-asset-based copula pairs trading strategy for cryptocurrency markets. The algorithm uses BTCUSDT as a reference asset and identifies cointegrated pairs through spread processes, leveraging copula models to generate trading signals based on conditional probabilities.

### Key Features

- **Reference-Asset Approach**: Uses BTCUSDT as reference with spread processes Si = P_reference - β_i × P_i (Eq. 31)
- **Cointegration Testing**: Engle-Granger (EG), Augmented Dickey-Fuller (ADF), and Kapetanios-Shin-Snell (KSS) tests
- **Marginal Fitting**: Automatic selection among Gaussian, Student-t, and Cauchy distributions (AIC-based)
- **Multiple Copula Families**:
  - Elliptical: Gaussian, Student-t
  - Archimedean: Clayton, Frank, Gumbel, Joe, BB1, BB6, BB7, BB8
  - Extreme Value: Tawn Type 1 & 2
  - Rotations: 0°, 90°, 180°, 270° for all applicable families
- **Probability Integral Transform (PIT)**: Transforms spreads to uniform variables
- **Conditional Probability Signals**: h-functions for entry/exit decisions (Tables 3 & 4)
- **Rolling Windows**: 21-day formation period, 7-day trading period

## Algorithm Summary

### Formation Period (21 days)

1. **Pair Selection**:
   - Identify assets cointegrated with BTCUSDT (EG + ADF + KSS tests)
   - Rank candidates by Kendall's tau correlation
   - Select top 2 pairs

2. **Spread Modeling**:
   - Calculate spreads: S_i = P_ref - β_i × P_i
   - Fit marginal distributions (Gaussian/Student-t/Cauchy)
   - Transform to uniform via PIT: U_i = F_i(S_i)

3. **Copula Fitting**:
   - Fit multiple copula families to (U_1, U_2)
   - Select best model by AIC

### Trading Period (7 days)

Generate signals using conditional probabilities (h-functions):

- **h_{1|2} = ∂C(u_1, u_2)/∂u_2**: Conditional probability of asset 1 given asset 2
- **h_{2|1} = ∂C(u_1, u_2)/∂u_1**: Conditional probability of asset 2 given asset 1

**Trading Rules** (α_1 = entry threshold, α_2 = exit threshold):

| Condition | Action |
|-----------|--------|
| h_{1|2} < α_1 and h_{2|1} > (1-α_1) | Long β_2×P_2, Short β_1×P_1 |
| h_{1|2} > (1-α_1) and h_{2|1} < α_1 | Short β_2×P_2, Long β_1×P_1 |
| \|h_{1|2} - 0.5\| < α_2 and \|h_{2|1} - 0.5\| < α_2 | Close both positions |

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd pair_trading

# Create virtual environment (Python 3.12+)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
statsmodels>=0.14.0
copulae>=0.7.9
requests>=2.31.0
```

## Usage

### 1. Download Data

Download hourly or 5-minute cryptocurrency data from Binance:

```bash
python -m src.download_binance_futures \
  --interval 1h \
  --start 2021-01-01 \
  --end 2023-01-19 \
  --out data/binance_futures_1h

python -m src.download_binance_futures \
  --interval 5m \
  --start 2021-01-01 \
  --end 2023-01-19 \
  --out data/binance_futures_5m
```

### 2. Run Backtest

#### Using the Main Strategy Class

```python
from src.main import ReferenceAssetCopulaTradingStrategy

# Create strategy instance
strategy = ReferenceAssetCopulaTradingStrategy(
    reference_symbol="BTCUSDT",
    alpha1=0.20,  # Entry threshold (paper tests 0.10, 0.15, 0.20)
    alpha2=0.10,  # Exit threshold
    eg_alpha=1.00,  # Disable EG test (set to 0.10 to enable)
    adf_alpha=0.10,  # ADF test p-value threshold
    kss_critical=-1.92,  # KSS 10% critical value
)

# Run backtest
results = strategy.backtest(
    data_dir="data/binance_futures_1h",
    interval="1h",
    formation_hours=21*24,
    trading_hours=7*24,
    fee_rate=0.0004,  # 4 bps (Binance taker fee)
    capital=20000.0,
)
```

#### Command Line

```bash
python -m src.main \
  --data data/binance_futures_1h \
  --interval 1h \
  --alpha1 0.20 \
  --alpha2 0.10 \
  --formation-hours 504 \
  --trading-hours 168 \
  --fee 0.0004 \
  --capital 20000
```

#### Full Backtest with Paper Grid

Run the complete experiment from the paper (α_1 ∈ {0.10, 0.15, 0.20}, α_2 = 0.10):

```bash
python -m src.run_paper_grid \
  --data data/binance_futures_1h \
  --interval 1h \
  --fee 0.0004 \
  --capital 20000
```

### 3. Advanced Usage

#### Direct Backtest Module

```bash
python -m src.backtest_reference_copula \
  --data data/binance_futures_1h \
  --interval 1h \
  --formation-hours 504 \
  --trading-hours 168 \
  --step-hours 168 \
  --alpha1 0.20 \
  --alpha2 0.10 \
  --eg-alpha 1.00 \
  --adf-alpha 0.10 \
  --kss-critical -1.92 \
  --fee 0.0004 \
  --capital 20000 \
  --log-prices \
  --start 2021-01-22 \
  --end 2023-01-19
```

## Project Structure

```
pair_trading/
├── src/
│   ├── __init__.py
│   ├── main.py                        # Main strategy class (NEW)
│   ├── backtest_reference_copula.py   # Complete backtest implementation
│   ├── copula_model.py                # Copula fitting & h-functions
│   ├── stats_tests.py                 # Cointegration tests (EG, ADF, KSS)
│   ├── data_io.py                     # Data loading utilities
│   ├── download_binance_futures.py    # Data download from Binance API
│   └── run_paper_grid.py              # Paper experiment grid
├── data/
│   ├── binance_futures_1h/            # Hourly price data
│   ├── binance_futures_5m/            # 5-minute price data
│   └── trades_*.csv                   # Backtest results
├── copula_based_pair_trading.pdf      # Original paper
├── README.md
├── pyproject.toml
└── LICENSE
```

## Key Modules

### `src/main.py` (NEW)

High-level API implementing the complete algorithm with clear documentation:

```python
class ReferenceAssetCopulaTradingStrategy:
    """Reference-asset-based copula pairs trading"""
    
    def calculate_spread(self, ref_price, asset_price)
    def identify_cointegrated_pairs(self, prices)
    def fit_copula_model(self, spread1, spread2)
    def generate_trading_signal(self, ref_price, price1, price2, ...)
    def backtest(self, data_dir, interval, formation_hours, ...)
```

### `src/copula_model.py`

- `fit_best_marginal()`: Fit Gaussian/Student-t/Cauchy distributions
- `fit_copula_candidates()`: Fit 40+ copula models (with rotations)
- `h_functions_numerical()`: Calculate conditional probabilities
- `RotatedCopula`: Implementation of 90°, 180°, 270° rotations (Eq. 16)

### `src/stats_tests.py`

- `cointegration_with_reference()`: EG + ADF + KSS tests
- `kss_estar_tstat()`: Kapetanios-Shin-Snell nonlinear unit root test
- `compute_spread()`: Spread calculation with/without intercept

### `src/backtest_reference_copula.py`

- `run_cycle()`: Execute one formation+trading cycle
- `pick_pair()`: Select optimal pair by Kendall's tau
- `performance_summary()`: Calculate Sharpe ratio, drawdown, etc.

## Performance Metrics

The backtest outputs:

- **Total Net Return**: Cumulative return after fees
- **Annualized Return**: Geometric CAGR
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Maximum Drawdown**: Largest peak-to-trough decline
- **RoMaD**: Return over Maximum Drawdown
- **Win Rate**: Percentage of profitable trades
- **Number of Transactions**: Total trades executed

### Paper Results (Reproduced)

**Hourly Data** (EG Test, α_1=0.20, α_2=0.10):
- Total Net Return: 82.3%
- Annualized Return: 35.1%
- Sharpe Ratio: 0.85
- Max Drawdown: -41.6%

**5-Minute Data** (EG Test, α_1=0.20, α_2=0.10):
- Total Net Return: 205.9%
- Annualized Return: 75.2%
- Sharpe Ratio: 3.77 ⭐
- Max Drawdown: -30.5%

## Key Assumptions

1. **Cycle Structure**: 21-day formation + 7-day trading, rolling weekly
2. **Initial Capital**: $20,000 USDT per side (< 1% of daily volume per paper guidelines)
3. **Transaction Fees**: 4 bps (0.04%) per leg (Binance futures taker fee)
4. **Position Sizing**: β-weighted to balance notional exposure
5. **Force Close**: All positions closed at end of trading window
6. **Market Orders**: All trades assumed to execute at taker fees

## Comparison to Other Methods

The paper compares against:

1. **Cointegration Approach**: z-score based entry/exit
2. **Return-Based Copula**: Mispricing from log-returns
3. **Level-Based Copula**: Cumulative mispricing index
4. **Buy & Hold**: Bitcoin and equal-weight portfolio

**Result**: Reference-asset copula approach achieves **29× better Sharpe ratio** than buy-and-hold (3.77 vs 0.13).

## Citations

```bibtex
@article{tadi2025copulas,
  title={Copulas in Cryptocurrency Pairs Trading: An Innovative Approach to Trading Strategies},
  author={Tadi, Masood and Witzany, Ji{\v{r}}{\'\i}},
  journal={Financial Innovation},
  volume={11},
  number={1},
  pages={40},
  year={2025},
  publisher={Springer}
}
```

## License

MIT License - see [LICENSE](LICENSE) file

## Acknowledgments

This implementation is based on the research by Masood Tadi and Jiří Witzany, supported by grant GAČR 22-19617 S "Modeling the structure and dynamics of energy, commodity, and alternative asset prices."

## Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Trading cryptocurrencies involves substantial risk of loss. Always conduct your own research and consult with qualified financial advisors before making investment decisions.

# Performance Simulator - Quick Reference

## Installation

No additional installation needed beyond the main project requirements.

## Quick Start

### 1. Basic Simulation

```python
from src.performance_simulator import quick_simulation

results = quick_simulation(
    data_dir="data/binance_futures_1h",
    initial_capital=100_000
)
```

### 2. Custom Configuration

```python
from src.performance_simulator import PerformanceSimulator, SimulationConfig

config = SimulationConfig(
    initial_capital=100_000,
    alpha1=0.20,
    alpha2=0.10,
    formation_hours=21*24,
    trading_hours=7*24,
)

simulator = PerformanceSimulator(config)
results = simulator.run_simulation("data/binance_futures_1h")
```

### 3. Command Line

```bash
python run_simulation.py --data data/binance_futures_1h \
    --capital 100000 --alpha1 0.20 --alpha2 0.10
```

## Key Metrics Quick Reference

### Return Metrics
- `total_return`: Total percentage return
- `annual_return` / `cagr`: Annualized return
- `total_pnl`: Absolute profit/loss
- `final_equity`: Ending portfolio value

### Risk Metrics
- `annual_volatility`: Annualized standard deviation
- `max_drawdown`: Largest peak-to-trough decline (%)
- `max_drawdown_duration_days`: Longest time in drawdown
- `value_at_risk_95`: VaR at 95% confidence
- `cvar_95`: Expected shortfall beyond VaR

### Risk-Adjusted Metrics
- `sharpe_ratio`: (Return - RiskFree) / Volatility
  - >1.0 is good, >2.0 is excellent
- `sortino_ratio`: Like Sharpe but only downside volatility
- `calmar_ratio`: Annual Return / |Max Drawdown|
- `omega_ratio`: Gains / Losses ratio

### Trade Statistics
- `total_trades`: Number of trades executed
- `win_rate`: % of profitable trades
- `profit_factor`: Gross profit / Gross loss
- `average_win` / `average_loss`: Mean trade P&L
- `average_trade_duration_hours`: Mean holding period

## Common Use Cases

### Parameter Optimization

```python
results = {}
for alpha1 in [0.10, 0.15, 0.20]:
    config = SimulationConfig(alpha1=alpha1)
    sim = PerformanceSimulator(config)
    results[alpha1] = sim.run_simulation("data/...")

best = max(results.items(), key=lambda x: x[1].metrics.sharpe_ratio)
```

### Walk-Forward Testing

```python
periods = [
    ("2020-01-01", "2021-01-01"),
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
]

for start, end in periods:
    results = simulator.run_simulation(
        "data/...", start_date=start, end_date=end
    )
    print(f"{start}: {results.metrics.sharpe_ratio:.2f}")
```

### Generate Reports

```python
simulator.generate_report(
    results,
    output_dir="reports",
    format="all"  # text, json, csv, all
)
```

## Configuration Parameters

### Essential Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_capital` | 100,000 | Starting capital |
| `alpha1` | 0.20 | Entry threshold |
| `alpha2` | 0.10 | Exit threshold |
| `formation_hours` | 504 (21d) | Formation period |
| `trading_hours` | 168 (7d) | Trading period |

### Advanced Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `reference_symbol` | "BTCUSDT" | Reference asset |
| `fee_rate` | 0.0004 | Transaction fees (0.04%) |
| `capital_per_side` | 20,000 | Max per trade side |
| `eg_alpha` | 1.00 | EG test threshold (1.0=disabled) |
| `adf_alpha` | 0.10 | ADF test threshold |
| `risk_free_rate` | 0.02 | Annual RF rate (2%) |

## Accessing Results

```python
# Performance metrics
m = results.metrics
print(f"Sharpe: {m.sharpe_ratio:.2f}")
print(f"Max DD: {m.max_drawdown*100:.2f}%")
print(f"Win Rate: {m.win_rate*100:.1f}%")

# Equity curve
equity = results.equity_curve  # pandas Series
returns = results.returns

# Individual trades
for trade in results.trades:
    print(f"{trade.entry_time}: ${trade.pnl:,.2f}")

# Cycle results
for cycle in results.cycle_results:
    if not cycle.skipped:
        print(f"Cycle {cycle.cycle_number}: {cycle.pair}")
```

## Troubleshooting

### "Insufficient data" error
Reduce `formation_hours` and `trading_hours` for small datasets:
```python
config = SimulationConfig(
    formation_hours=24,   # 1 day instead of 21
    trading_hours=12,     # 12 hours instead of 7 days
)
```

### No trades executed
Check cycle skip reasons:
```python
skip_reasons = {}
for c in results.cycle_results:
    if c.skipped:
        skip_reasons[c.skip_reason] = skip_reasons.get(c.skip_reason, 0) + 1
print(skip_reasons)
```

Common fixes:
- `no_cointegrated_pair`: Relax `eg_alpha`, `adf_alpha`
- `marginal_or_copula_fit_failed`: Increase `formation_hours`
- Adjust `alpha1`, `alpha2` thresholds

### High number of skipped cycles
1. Review skip reasons in cycle summary
2. Relax cointegration test thresholds
3. Increase formation period for better fits
4. Check data quality (missing values)

## Examples

See `example_simulation.py` for complete examples:
```bash
python example_simulation.py
```

## Documentation

- **Full Documentation**: [docs/PERFORMANCE_SIMULATOR.md](PERFORMANCE_SIMULATOR.md)
- **Algorithm Details**: [ALGORITHM.md](../ALGORITHM.md)
- **Main README**: [README.md](../README.md)

## Quick Command Reference

```bash
# Basic run
python run_simulation.py --data data/binance_futures_1h

# Custom parameters
python run_simulation.py --data data/binance_futures_1h \
    --capital 100000 --alpha1 0.20 --alpha2 0.10

# Date range
python run_simulation.py --data data/binance_futures_1h \
    --start 2020-01-01 --end 2024-01-01

# Custom periods (for testing)
python run_simulation.py --data data/binance_futures_1h_smoke \
    --formation-hours 24 --trading-hours 12

# Text report only
python run_simulation.py --data data/binance_futures_1h \
    --format text --output reports/my_test

# Summary only (no files)
python run_simulation.py --data data/binance_futures_1h --no-report
```

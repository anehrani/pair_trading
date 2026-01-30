# Performance Simulator Library

A comprehensive performance simulation framework for the copula-based pairs trading algorithm. This library provides everything you need to backtest, analyze, and report on your trading strategy performance.

## Features

### 🎯 Core Capabilities
- **Complete Portfolio Simulation**: Track positions, capital, and P&L over time
- **Realistic Order Execution**: Simulates market orders with configurable fees
- **Rolling Window Strategy**: Formation and trading periods with flexible stepping

### 📊 Performance Metrics
- **Return Metrics**: Total return, CAGR, annualized returns
- **Risk Metrics**: Volatility, Value-at-Risk (VaR), CVaR, maximum drawdown
- **Risk-Adjusted Metrics**: Sharpe, Sortino, Calmar, and Omega ratios
- **Trade Statistics**: Win rate, profit factor, average win/loss, trade duration

### 📈 Reporting
- **Text Reports**: Human-readable comprehensive reports
- **JSON Reports**: Complete data export for further analysis
- **CSV Reports**: Trade logs, equity curves, and cycle summaries
- **Console Output**: Quick summaries for rapid iteration

## Installation

The library is part of the pair_trading project. Make sure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.performance_simulator import quick_simulation

# Run a quick simulation with defaults
results = quick_simulation(
    data_dir="data/binance_futures_1h",
    initial_capital=100_000,
    alpha1=0.20,
    alpha2=0.10
)
```

### Custom Configuration

```python
from src.performance_simulator import PerformanceSimulator, SimulationConfig

# Configure the simulation
config = SimulationConfig(
    initial_capital=100_000,
    reference_symbol="BTCUSDT",
    formation_hours=21 * 24,  # 21 days
    trading_hours=7 * 24,      # 7 days
    alpha1=0.20,               # Entry threshold
    alpha2=0.10,               # Exit threshold
    fee_rate=0.0004,           # 4 bps per trade
)

# Run simulation
simulator = PerformanceSimulator(config)
results = simulator.run_simulation(
    data_dir="data/binance_futures_1h",
    start_date="2020-01-01",
    end_date="2024-01-01"
)

# Print summary
simulator.print_summary(results)
```

### Generate Reports

```python
# Generate all report formats
simulator.generate_report(
    results,
    output_dir="reports/backtest_2024",
    format="all"  # Options: "text", "json", "csv", "all"
)
```

## Configuration Options

### SimulationConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_capital` | float | 100,000 | Starting portfolio capital |
| `reference_symbol` | str | "BTCUSDT" | Reference asset for pairs |
| `interval` | str | "1h" | Time interval for data |
| `formation_hours` | int | 504 (21d) | Formation period length |
| `trading_hours` | int | 168 (7d) | Trading period length |
| `step_hours` | int | 168 (7d) | Step size between cycles |
| `alpha1` | float | 0.20 | Entry threshold |
| `alpha2` | float | 0.10 | Exit threshold |
| `fee_rate` | float | 0.0004 | Transaction fee rate (0.04%) |
| `capital_per_side` | float | 20,000 | Max capital per trade side |
| `eg_alpha` | float | 1.00 | Engle-Granger p-value threshold |
| `adf_alpha` | float | 0.10 | ADF p-value threshold |
| `kss_critical_10pct` | float | -1.92 | KSS critical value |
| `use_log_prices` | bool | True | Use log prices for cointegration |
| `risk_free_rate` | float | 0.02 | Annual risk-free rate (2%) |

## Accessing Results

The `SimulationResults` object contains all simulation data:

```python
# Performance metrics
print(f"Sharpe Ratio: {results.metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.metrics.max_drawdown*100:.2f}%")
print(f"Win Rate: {results.metrics.win_rate*100:.1f}%")

# Equity curve (pandas Series)
equity_curve = results.equity_curve

# Individual trades (list of Trade objects)
for trade in results.trades:
    print(f"{trade.entry_time}: {trade.symbol_long} vs {trade.symbol_short}")
    print(f"  PnL: ${trade.pnl:,.2f}")

# Cycle results
for cycle in results.cycle_results:
    if not cycle.skipped:
        print(f"Cycle {cycle.cycle_number}: {cycle.pair}")
        print(f"  Copula: {cycle.copula_name}, AIC: {cycle.copula_aic:.2f}")
```

## Example Workflows

### 1. Parameter Optimization

```python
from src.performance_simulator import PerformanceSimulator, SimulationConfig

# Test different parameter combinations
results = {}
for alpha1 in [0.10, 0.15, 0.20, 0.25]:
    for alpha2 in [0.05, 0.10, 0.15]:
        config = SimulationConfig(
            initial_capital=100_000,
            alpha1=alpha1,
            alpha2=alpha2
        )
        
        simulator = PerformanceSimulator(config)
        result = simulator.run_simulation("data/binance_futures_1h")
        
        results[(alpha1, alpha2)] = result.metrics.sharpe_ratio

# Find best parameters
best_params = max(results.items(), key=lambda x: x[1])
print(f"Best parameters: alpha1={best_params[0][0]}, alpha2={best_params[0][1]}")
```

### 2. Walk-Forward Analysis

```python
from datetime import datetime, timedelta

# Define periods
periods = [
    ("2020-01-01", "2021-01-01"),
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
]

config = SimulationConfig(initial_capital=100_000)
simulator = PerformanceSimulator(config)

for start, end in periods:
    print(f"\nPeriod: {start} to {end}")
    results = simulator.run_simulation(
        data_dir="data/binance_futures_1h",
        start_date=start,
        end_date=end
    )
    print(f"  Return: {results.metrics.total_return*100:.2f}%")
    print(f"  Sharpe: {results.metrics.sharpe_ratio:.2f}")
```

### 3. Detailed Trade Analysis

```python
import pandas as pd

# Load results
results = simulator.run_simulation("data/binance_futures_1h")

# Convert trades to DataFrame
trades_df = pd.DataFrame([{
    'entry_time': t.entry_time,
    'exit_time': t.exit_time,
    'duration_hours': (t.exit_time - t.entry_time).total_seconds() / 3600,
    'long': t.symbol_long,
    'short': t.symbol_short,
    'pnl': t.pnl,
    'fees': t.fees,
    'return': t.pnl / (abs(t.qty_long * t.entry_price_long) + 
                       abs(t.qty_short * t.entry_price_short))
} for t in results.trades])

# Analyze winning vs losing trades
winners = trades_df[trades_df['pnl'] > 0]
losers = trades_df[trades_df['pnl'] < 0]

print(f"Average winning trade: ${winners['pnl'].mean():,.2f}")
print(f"Average losing trade: ${losers['pnl'].mean():,.2f}")
print(f"Average win duration: {winners['duration_hours'].mean():.1f} hours")
print(f"Average loss duration: {losers['duration_hours'].mean():.1f} hours")
```

## Performance Metrics Explained

### Return Metrics
- **Total Return**: Overall percentage return from start to end
- **Annual Return (CAGR)**: Compound Annual Growth Rate
- **Final Equity**: Ending portfolio value

### Risk Metrics
- **Annual Volatility**: Annualized standard deviation of returns
- **Downside Volatility**: Volatility of negative returns only
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Max DD Duration**: Longest time in drawdown (days)
- **VaR (95%)**: Value at Risk at 95% confidence level
- **CVaR (95%)**: Conditional Value at Risk (expected shortfall)

### Risk-Adjusted Metrics
- **Sharpe Ratio**: Excess return per unit of total risk
  - Formula: `(Return - Risk_Free_Rate) / Volatility`
  - Higher is better (>1.0 is good, >2.0 is excellent)
  
- **Sortino Ratio**: Excess return per unit of downside risk
  - Similar to Sharpe but only penalizes downside volatility
  - Higher is better
  
- **Calmar Ratio**: Return per unit of maximum drawdown
  - Formula: `Annual_Return / abs(Max_Drawdown)`
  - Higher is better (>1.0 is good)
  
- **Omega Ratio**: Probability-weighted gains vs losses
  - Ratio of gains above threshold to losses below threshold
  - >1.0 means more upside than downside

### Trade Statistics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Average Win/Loss**: Mean P&L of winning/losing trades
- **Average Trade Duration**: Mean time in position

## Output Files

When generating reports with `format="all"`, the following files are created:

### Text Report (`report_TIMESTAMP.txt`)
Human-readable comprehensive report with all metrics, formatted tables, and summaries.

### JSON Report (`report_TIMESTAMP.json`)
Complete data dump including:
- Configuration parameters
- All performance metrics
- Equity curve data
- Trade details
- Cycle results

### CSV Files
- `trades_TIMESTAMP.csv`: Individual trade records
- `equity_TIMESTAMP.csv`: Time series of equity, returns, and drawdowns
- `cycles_TIMESTAMP.csv`: Summary of each trading cycle

## Advanced Usage

### Custom Metrics

```python
# Access raw data for custom calculations
equity = results.equity_curve
returns = results.returns

# Calculate custom metrics
rolling_sharpe = returns.rolling(window=30*24).apply(
    lambda x: x.mean() / x.std() * np.sqrt(365*24)
)

# Calculate win/loss streaks
trades_pnl = [t.pnl for t in results.trades]
current_streak = 0
max_win_streak = 0
max_loss_streak = 0

for pnl in trades_pnl:
    if pnl > 0:
        current_streak = max(1, current_streak + 1)
        max_win_streak = max(max_win_streak, current_streak)
    else:
        current_streak = min(-1, current_streak - 1)
        max_loss_streak = min(max_loss_streak, current_streak)

print(f"Max winning streak: {max_win_streak}")
print(f"Max losing streak: {abs(max_loss_streak)}")
```

### Exporting for Further Analysis

```python
# Export to JSON for use in other tools
import json

data = results.to_dict()
with open('backtest_results.json', 'w') as f:
    json.dump(data, f, indent=2)

# Export equity curve for plotting
results.equity_curve.to_csv('equity_curve.csv')

# Export for Excel analysis
trades_df = pd.DataFrame([asdict(t) for t in results.trades])
trades_df.to_excel('trades_analysis.xlsx', index=False)
```

## Testing

Run the example script to test the simulator:

```bash
python example_simulation.py
```

This will run multiple examples demonstrating different features of the library.

## Best Practices

1. **Start with small datasets**: Use smoke test data first to validate configuration
2. **Use realistic fees**: Set `fee_rate` based on your actual trading costs
3. **Test parameter robustness**: Run multiple simulations with different parameters
4. **Consider transaction costs**: Remember that more trades = more fees
5. **Analyze cycle skip reasons**: High skip rates may indicate parameter issues
6. **Check drawdown periods**: Understand when and why drawdowns occur
7. **Walk-forward validation**: Test on out-of-sample periods

## Troubleshooting

### "Insufficient data" error
- Reduce `formation_hours` and `trading_hours` for small datasets
- Check that your data files contain enough historical data

### No trades executed
- Cycles may be skipped due to:
  - No cointegrated pairs found (adjust `eg_alpha`, `adf_alpha`)
  - Marginal/copula fit failures (may need more data in formation period)
  - No trading signals generated (adjust `alpha1`, `alpha2`)

### High number of skipped cycles
- Review skip reasons in the cycle summary
- Common causes:
  - `no_cointegrated_pair`: Relax cointegration thresholds
  - `marginal_or_copula_fit_failed`: Increase formation period
  - `missing_data`: Check data quality and completeness

## See Also

- [ALGORITHM.md](../ALGORITHM.md): Detailed algorithm documentation
- [IMPLEMENTATION.md](../IMPLEMENTATION.md): Implementation details
- [example_simulation.py](../example_simulation.py): Usage examples
- [README.md](../README.md): Project overview

## License

Same as the main project (see LICENSE file).

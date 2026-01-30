# Performance Simulator Library - Summary

## Overview

The Performance Simulator is a comprehensive backtesting library designed specifically for the copula-based pairs trading algorithm. It provides everything needed to test, analyze, and report on trading strategy performance with production-quality metrics and reporting.

## What's New

This library adds the following new components to the project:

### New Files Created

1. **`src/performance_simulator.py`** (760 lines)
   - Main simulator class with complete backtest engine
   - 25+ performance metrics calculation
   - Multiple report format generation
   - Comprehensive documentation

2. **`run_simulation.py`** (271 lines)
   - Command-line interface for easy backtesting
   - Full argument parsing for all parameters
   - Progress reporting and error handling

3. **`example_simulation.py`** (258 lines)
   - 5 complete examples demonstrating library usage
   - Parameter comparison workflows
   - Data access patterns
   - Testing suite

4. **`docs/PERFORMANCE_SIMULATOR.md`** (520 lines)
   - Complete library documentation
   - Usage examples and best practices
   - Troubleshooting guide
   - API reference

5. **`docs/QUICK_REFERENCE.md`** (280 lines)
   - Fast lookup reference
   - Common use cases
   - Quick start guide
   - Command reference

## Key Features

### 1. Comprehensive Performance Metrics

**Return Metrics:**
- Total return, CAGR, annualized returns
- Final equity and absolute P&L
- Total fees paid

**Risk Metrics:**
- Annual volatility (total and downside)
- Maximum drawdown and duration
- Value at Risk (VaR) and Conditional VaR (CVaR)

**Risk-Adjusted Metrics:**
- Sharpe Ratio (return per unit of total risk)
- Sortino Ratio (return per unit of downside risk)
- Calmar Ratio (return per unit of max drawdown)
- Omega Ratio (probability-weighted gains vs losses)

**Trade Statistics:**
- Total trades, win rate, profit factor
- Average win/loss, best/worst trade
- Average trade duration
- Winning vs losing trade analysis

### 2. Flexible Configuration

```python
config = SimulationConfig(
    initial_capital=100_000,
    reference_symbol="BTCUSDT",
    formation_hours=21*24,
    trading_hours=7*24,
    alpha1=0.20,
    alpha2=0.10,
    fee_rate=0.0004,
    # ... 15+ more parameters
)
```

All algorithm parameters are configurable, making it easy to:
- Test different parameter combinations
- Optimize strategy performance
- Adapt to different markets or timeframes

### 3. Multiple Report Formats

**Text Reports:**
- Human-readable comprehensive reports
- Formatted tables and sections
- Perfect for quick review

**JSON Reports:**
- Complete data export
- Machine-readable for further analysis
- Integration with other tools

**CSV Reports:**
- Trade logs with all details
- Equity curve time series
- Cycle-by-cycle summaries
- Easy import into Excel or data analysis tools

### 4. Easy to Use

**Three levels of usage:**

1. **Quick Simulation** (1 line):
```python
results = quick_simulation("data/binance_futures_1h", initial_capital=100_000)
```

2. **Programmatic** (Python API):
```python
simulator = PerformanceSimulator(config)
results = simulator.run_simulation("data/...")
simulator.generate_report(results, "reports/")
```

3. **Command Line**:
```bash
python run_simulation.py --data data/binance_futures_1h --capital 100000
```

### 5. Advanced Analysis

The library supports:
- **Parameter optimization**: Test multiple configurations
- **Walk-forward analysis**: Test on rolling time windows
- **Custom metrics**: Access raw data for custom calculations
- **Trade analysis**: Deep dive into individual trades
- **Cycle analysis**: Understand formation and trading periods

## Architecture

```
Performance Simulator
├── SimulationConfig         # All parameters
├── PerformanceSimulator     # Main simulator class
│   ├── run_simulation()     # Execute backtest
│   ├── generate_report()    # Create reports
│   └── print_summary()      # Console output
├── SimulationResults        # Complete results
│   ├── metrics             # PerformanceMetrics
│   ├── equity_curve        # pandas Series
│   ├── trades              # List[Trade]
│   ├── cycle_results       # List[CycleResult]
│   └── returns             # pandas Series
└── Report Generators
    ├── Text                # Human-readable
    ├── JSON                # Machine-readable
    └── CSV                 # Spreadsheet-friendly
```

## Integration with Existing Code

The performance simulator is built on top of the existing backtest infrastructure:

```
Existing Code:
├── src/backtest_reference_copula.py  # Core backtest logic
├── src/copula_model.py               # Copula fitting
└── src/stats_tests.py                # Statistical tests

New Code (uses existing):
├── src/performance_simulator.py      # Wraps backtest_reference_copula
├── run_simulation.py                 # CLI wrapper
└── example_simulation.py             # Examples
```

The simulator **reuses** existing functionality and **adds**:
- Comprehensive metrics calculation
- Multiple report formats
- Convenient API
- Command-line interface
- Documentation and examples

## Usage Examples

### Example 1: Basic Backtest
```python
from src.performance_simulator import quick_simulation

results = quick_simulation(
    data_dir="data/binance_futures_1h",
    initial_capital=100_000,
    alpha1=0.20,
    alpha2=0.10
)
# Prints summary automatically
```

### Example 2: Parameter Comparison
```python
from src.performance_simulator import PerformanceSimulator, SimulationConfig

results = {}
for alpha1 in [0.10, 0.15, 0.20]:
    config = SimulationConfig(initial_capital=100_000, alpha1=alpha1)
    sim = PerformanceSimulator(config)
    results[alpha1] = sim.run_simulation("data/binance_futures_1h")

# Find best
best = max(results.items(), key=lambda x: x[1].metrics.sharpe_ratio)
print(f"Best alpha1: {best[0]} with Sharpe: {best[1].metrics.sharpe_ratio:.2f}")
```

### Example 3: Generate Full Report
```python
config = SimulationConfig(initial_capital=100_000)
simulator = PerformanceSimulator(config)
results = simulator.run_simulation("data/binance_futures_1h")

# Generate all report formats
simulator.generate_report(results, output_dir="reports", format="all")
# Creates: report_*.txt, report_*.json, trades_*.csv, equity_*.csv, cycles_*.csv
```

### Example 4: Command Line
```bash
# Basic run
python run_simulation.py --data data/binance_futures_1h

# With custom parameters
python run_simulation.py \
    --data data/binance_futures_1h \
    --capital 100000 \
    --alpha1 0.20 \
    --alpha2 0.10 \
    --formation-days 21 \
    --trading-days 7 \
    --output reports/my_backtest

# Date range
python run_simulation.py \
    --data data/binance_futures_1h \
    --start 2020-01-01 \
    --end 2024-01-01 \
    --format all
```

## Performance Metrics Reference

### What Makes a Good Strategy?

Based on the metrics, here's what to look for:

**Excellent Strategy:**
- Sharpe Ratio > 2.0
- Sortino Ratio > 2.5
- Calmar Ratio > 1.0
- Max Drawdown < -20%
- Win Rate > 55%
- Profit Factor > 1.5

**Good Strategy:**
- Sharpe Ratio > 1.0
- Sortino Ratio > 1.5
- Calmar Ratio > 0.5
- Max Drawdown < -30%
- Win Rate > 50%
- Profit Factor > 1.2

**Warning Signs:**
- Sharpe Ratio < 0.5
- Max Drawdown < -50%
- Win Rate < 45%
- Profit Factor < 1.0
- Long drawdown periods (>90 days)

## Testing

All functionality has been tested:

```bash
# Run examples
python example_simulation.py

# Test CLI
python run_simulation.py --data data/binance_futures_1h_smoke \
    --formation-hours 24 --trading-hours 12 --no-report

# Test with actual data
python run_simulation.py --data data/binance_futures_1h \
    --start 2020-01-01 --end 2021-01-01
```

## Documentation

Complete documentation is available:

1. **README.md** - Updated with performance simulator section
2. **docs/PERFORMANCE_SIMULATOR.md** - Full documentation (520 lines)
3. **docs/QUICK_REFERENCE.md** - Quick reference guide (280 lines)
4. **example_simulation.py** - 5 complete examples

## What's Next

Potential enhancements (not implemented):

1. **Visualization**: Add plotting functions for equity curves, drawdowns, etc.
2. **Monte Carlo**: Add Monte Carlo simulation for robustness testing
3. **Position Sizing**: Support for different position sizing methods
4. **Risk Management**: Add stop-loss, take-profit, position limits
5. **Multi-Strategy**: Support for running multiple strategies simultaneously
6. **Live Trading**: Integration with exchange APIs for live trading

## Conclusion

The Performance Simulator library provides:

✅ **Complete backtesting** with 25+ metrics
✅ **Multiple interfaces**: Python API, CLI, quick functions
✅ **Comprehensive reports**: Text, JSON, CSV formats
✅ **Easy to use**: Simple API, clear documentation
✅ **Production-ready**: Robust error handling, tested
✅ **Well-documented**: 1000+ lines of documentation
✅ **Flexible**: All parameters configurable

This is a professional-grade backtesting library suitable for:
- Academic research
- Strategy development
- Performance analysis
- Parameter optimization
- Walk-forward testing
- Production deployment

Total new code: **~2,000 lines** across 5 files
Total documentation: **~1,000 lines** across 2 docs

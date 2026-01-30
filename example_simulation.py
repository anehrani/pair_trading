"""Example: Using the Performance Simulator

This script demonstrates how to use the performance simulator library
to backtest the pair trading algorithm and generate comprehensive reports.

Run this script to:
1. Configure a simulation with custom parameters
2. Run a backtest on historical data
3. Generate detailed performance reports
4. Export results in multiple formats
"""

from pathlib import Path

from src.performance_simulator import (
    PerformanceSimulator,
    SimulationConfig,
    quick_simulation,
)


def example_basic_simulation():
    """Example 1: Basic simulation with default parameters."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Simulation")
    print("="*80)
    
    # Use quick_simulation for fast testing with smaller periods for smoke data
    results = quick_simulation(
        data_dir="data/binance_futures_1h_smoke",  # Small dataset for testing
        initial_capital=50_000,
        alpha1=0.20,
        alpha2=0.10,
        formation_hours=24,  # 1 day for testing
        trading_hours=12,    # 12 hours for testing
        step_hours=12,
    )
    
    # Results are automatically printed
    return results


def example_custom_simulation():
    """Example 2: Simulation with custom configuration."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Configuration Simulation")
    print("="*80)
    
    # Create custom configuration
    config = SimulationConfig(
        initial_capital=100_000,
        reference_symbol="BTCUSDT",
        interval="1h",
        formation_hours=24,  # 1 day for testing
        trading_hours=12,     # 12 hours for testing
        step_hours=12,        # Non-overlapping cycles
        alpha1=0.15,              # Lower entry threshold
        alpha2=0.10,              # Exit threshold
        fee_rate=0.0004,          # 4 bps
        capital_per_side=25_000,
        use_log_prices=True,
        risk_free_rate=0.02,      # 2% annual
    )
    
    # Create simulator
    simulator = PerformanceSimulator(config)
    
    # Run simulation
    results = simulator.run_simulation(
        data_dir="data/binance_futures_1h_smoke",
        start_date="2020-01-01",  # Optional date filtering
        end_date="2024-01-01",
    )
    
    # Print summary
    simulator.print_summary(results)
    
    return simulator, results


def example_full_backtest_with_reports():
    """Example 3: Complete backtest with full reports."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Full Backtest with Reports")
    print("="*80)
    
    # Configuration
    config = SimulationConfig(
        initial_capital=100_000,
        alpha1=0.20,
        alpha2=0.10,
        fee_rate=0.0004,
        formation_hours=24,  # 1 day for testing
        trading_hours=12,    # 12 hours for testing
        step_hours=12,
    )
    
    # Run simulation
    simulator = PerformanceSimulator(config)
    results = simulator.run_simulation(
        data_dir="data/binance_futures_1h_smoke"
    )
    
    # Generate comprehensive reports
    output_dir = Path("reports")
    simulator.generate_report(results, output_dir, format="all")
    
    # Access specific metrics
    print(f"\nKey Metrics:")
    print(f"  Sharpe Ratio: {results.metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {results.metrics.sortino_ratio:.2f}")
    print(f"  Calmar Ratio: {results.metrics.calmar_ratio:.2f}")
    print(f"  Win Rate: {results.metrics.win_rate*100:.1f}%")
    print(f"  Profit Factor: {results.metrics.profit_factor:.2f}")
    
    # Access trade data
    print(f"\nTrade Analysis:")
    print(f"  Total Trades: {len(results.trades)}")
    print(f"  Best Trade: ${results.metrics.best_trade:,.2f}")
    print(f"  Worst Trade: ${results.metrics.worst_trade:,.2f}")
    print(f"  Avg Win: ${results.metrics.average_win:,.2f}")
    print(f"  Avg Loss: ${results.metrics.average_loss:,.2f}")
    
    # Cycle statistics
    active_cycles = [c for c in results.cycle_results if not c.skipped]
    print(f"\nCycle Statistics:")
    print(f"  Total Cycles: {len(results.cycle_results)}")
    print(f"  Active Cycles: {len(active_cycles)}")
    print(f"  Skipped Cycles: {len(results.cycle_results) - len(active_cycles)}")
    
    return simulator, results


def example_parameter_comparison():
    """Example 4: Compare different parameter settings."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Parameter Comparison")
    print("="*80)
    
    # Test different alpha1 values
    alpha1_values = [0.10, 0.15, 0.20]
    results_dict = {}
    
    for alpha1 in alpha1_values:
        print(f"\nTesting alpha1={alpha1}...")
        
        config = SimulationConfig(
            initial_capital=100_000,
            alpha1=alpha1,
            alpha2=0.10,
            formation_hours=24,
            trading_hours=12,
            step_hours=12,
        )
        
        simulator = PerformanceSimulator(config)
        results = simulator.run_simulation(
            data_dir="data/binance_futures_1h_smoke"
        )
        
        results_dict[alpha1] = results
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"{'Alpha1':<10} {'Return':<12} {'Sharpe':<10} {'Max DD':<12} {'Trades':<10}")
    print("-"*80)
    
    for alpha1, results in results_dict.items():
        m = results.metrics
        print(f"{alpha1:<10.2f} {m.total_return*100:<12.2f} {m.sharpe_ratio:<10.2f} "
              f"{m.max_drawdown*100:<12.2f} {m.total_trades:<10}")
    
    return results_dict


def example_access_detailed_data():
    """Example 5: Accessing detailed simulation data."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Accessing Detailed Data")
    print("="*80)
    
    # Run simulation
    results = quick_simulation(
        data_dir="data/binance_futures_1h_smoke",
        initial_capital=100_000,
        formation_hours=24,
        trading_hours=12,
        step_hours=12,
    )
    
    # Access equity curve
    print(f"\nEquity Curve:")
    print(f"  Shape: {results.equity_curve.shape}")
    print(f"  Start: ${results.equity_curve.iloc[0]:,.2f}")
    print(f"  End: ${results.equity_curve.iloc[-1]:,.2f}")
    
    # Access returns
    print(f"\nReturns:")
    print(f"  Mean: {results.returns.mean()*100:.4f}%")
    print(f"  Std: {results.returns.std()*100:.4f}%")
    print(f"  Min: {results.returns.min()*100:.4f}%")
    print(f"  Max: {results.returns.max()*100:.4f}%")
    
    # Access individual trades
    if results.trades:
        print(f"\nFirst Trade:")
        first_trade = results.trades[0]
        print(f"  Entry: {first_trade.entry_time}")
        print(f"  Exit: {first_trade.exit_time}")
        print(f"  Long: {first_trade.symbol_long} ({first_trade.qty_long:.4f})")
        print(f"  Short: {first_trade.symbol_short} ({first_trade.qty_short:.4f})")
        print(f"  PnL: ${first_trade.pnl:,.2f}")
        print(f"  Fees: ${first_trade.fees:,.2f}")
    
    # Access cycle results
    if results.cycle_results:
        print(f"\nFirst Active Cycle:")
        for cycle in results.cycle_results:
            if not cycle.skipped:
                print(f"  Cycle: {cycle.cycle_number}")
                print(f"  Pair: {cycle.pair}")
                print(f"  Copula: {cycle.copula_name}")
                print(f"  Trades: {len(cycle.trades)}")
                print(f"  PnL: ${cycle.cycle_pnl:,.2f}")
                break
    
    return results


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print(" PERFORMANCE SIMULATOR EXAMPLES ".center(80, "="))
    print("="*80)
    
    # Run examples
    try:
        # Example 1: Basic
        example_basic_simulation()
        
        # Example 2: Custom configuration
        example_custom_simulation()
        
        # Example 3: Full reports
        example_full_backtest_with_reports()
        
        # Example 4: Parameter comparison
        example_parameter_comparison()
        
        # Example 5: Detailed data access
        example_access_detailed_data()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Command-line interface for the Performance Simulator.

This script provides a convenient CLI for running backtests without writing code.

Usage:
    python run_simulation.py --data data/binance_futures_1h --capital 100000 --alpha1 0.20 --alpha2 0.10
    
    python run_simulation.py --data data/binance_futures_1h --start 2020-01-01 --end 2024-01-01 \\
        --formation-days 21 --trading-days 7 --output reports/my_backtest
"""

import argparse
import sys
from pathlib import Path

from src.performance_simulator import PerformanceSimulator, SimulationConfig


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run pair trading strategy backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic backtest
  python run_simulation.py --data data/binance_futures_1h
  
  # Custom parameters
  python run_simulation.py --data data/binance_futures_1h \\
      --capital 100000 --alpha1 0.20 --alpha2 0.10 \\
      --formation-days 21 --trading-days 7
  
  # Specific date range with custom output
  python run_simulation.py --data data/binance_futures_1h \\
      --start 2020-01-01 --end 2024-01-01 \\
      --output reports/backtest_2020_2024
  
  # Quick test with small periods
  python run_simulation.py --data data/binance_futures_1h_smoke \\
      --formation-hours 24 --trading-hours 12 --step-hours 12
        """
    )
    
    # Data arguments
    parser.add_argument(
        "--data",
        required=True,
        help="Directory containing price CSV files"
    )
    parser.add_argument(
        "--start",
        help="Start date (YYYY-MM-DD, UTC)"
    )
    parser.add_argument(
        "--end",
        help="End date (YYYY-MM-DD, UTC)"
    )
    
    # Capital arguments
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000,
        help="Initial capital (default: 100000)"
    )
    parser.add_argument(
        "--capital-per-side",
        type=float,
        default=20_000,
        help="Maximum capital per trade side (default: 20000)"
    )
    
    # Strategy parameters
    parser.add_argument(
        "--reference",
        default="BTCUSDT",
        help="Reference symbol (default: BTCUSDT)"
    )
    parser.add_argument(
        "--alpha1",
        type=float,
        default=0.20,
        help="Entry threshold (default: 0.20)"
    )
    parser.add_argument(
        "--alpha2",
        type=float,
        default=0.10,
        help="Exit threshold (default: 0.10)"
    )
    
    # Period arguments (in days)
    parser.add_argument(
        "--formation-days",
        type=int,
        help="Formation period in days (default: 21)"
    )
    parser.add_argument(
        "--trading-days",
        type=int,
        help="Trading period in days (default: 7)"
    )
    parser.add_argument(
        "--step-days",
        type=int,
        help="Step size in days (default: 7)"
    )
    
    # Period arguments (in hours - overrides days if specified)
    parser.add_argument(
        "--formation-hours",
        type=int,
        help="Formation period in hours (overrides --formation-days)"
    )
    parser.add_argument(
        "--trading-hours",
        type=int,
        help="Trading period in hours (overrides --trading-days)"
    )
    parser.add_argument(
        "--step-hours",
        type=int,
        help="Step size in hours (overrides --step-days)"
    )
    
    # Other parameters
    parser.add_argument(
        "--interval",
        default="1h",
        help="Time interval (default: 1h)"
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.0004,
        help="Transaction fee rate (default: 0.0004 = 0.04%%)"
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.02,
        help="Annual risk-free rate (default: 0.02 = 2%%)"
    )
    
    # Statistical test parameters
    parser.add_argument(
        "--eg-alpha",
        type=float,
        default=1.00,
        help="Engle-Granger p-value threshold (default: 1.00, disabled)"
    )
    parser.add_argument(
        "--adf-alpha",
        type=float,
        default=0.10,
        help="ADF p-value threshold (default: 0.10)"
    )
    parser.add_argument(
        "--kss-critical",
        type=float,
        default=-1.92,
        help="KSS critical value at 10%% (default: -1.92)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output",
        help="Output directory for reports (default: reports)"
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "csv", "all"],
        default="all",
        help="Report format (default: all)"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report generation, only show summary"
    )
    
    # Flags
    parser.add_argument(
        "--no-log-prices",
        action="store_true",
        help="Disable log price transformation"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Determine period lengths
    if args.formation_hours is not None:
        formation_hours = args.formation_hours
    elif args.formation_days is not None:
        formation_hours = args.formation_days * 24
    else:
        formation_hours = 21 * 24  # Default: 21 days
    
    if args.trading_hours is not None:
        trading_hours = args.trading_hours
    elif args.trading_days is not None:
        trading_hours = args.trading_days * 24
    else:
        trading_hours = 7 * 24  # Default: 7 days
    
    if args.step_hours is not None:
        step_hours = args.step_hours
    elif args.step_days is not None:
        step_hours = args.step_days * 24
    else:
        step_hours = 7 * 24  # Default: 7 days
    
    # Create configuration
    config = SimulationConfig(
        initial_capital=args.capital,
        reference_symbol=args.reference,
        interval=args.interval,
        formation_hours=formation_hours,
        trading_hours=trading_hours,
        step_hours=step_hours,
        alpha1=args.alpha1,
        alpha2=args.alpha2,
        fee_rate=args.fee_rate,
        capital_per_side=args.capital_per_side,
        eg_alpha=args.eg_alpha,
        adf_alpha=args.adf_alpha,
        kss_critical_10pct=args.kss_critical,
        use_log_prices=not args.no_log_prices,
        risk_free_rate=args.risk_free_rate,
    )
    
    # Print configuration
    if not args.quiet:
        print("="*80)
        print("BACKTEST CONFIGURATION".center(80))
        print("="*80)
        print(f"Data Directory:     {args.data}")
        print(f"Initial Capital:    ${config.initial_capital:,.2f}")
        print(f"Reference Symbol:   {config.reference_symbol}")
        print(f"Formation Period:   {config.formation_hours} hours ({config.formation_hours/24:.1f} days)")
        print(f"Trading Period:     {config.trading_hours} hours ({config.trading_hours/24:.1f} days)")
        print(f"Step Size:          {config.step_hours} hours ({config.step_hours/24:.1f} days)")
        print(f"Alpha1 (Entry):     {config.alpha1}")
        print(f"Alpha2 (Exit):      {config.alpha2}")
        print(f"Fee Rate:           {config.fee_rate*100:.2f}%")
        if args.start:
            print(f"Start Date:         {args.start}")
        if args.end:
            print(f"End Date:           {args.end}")
        print("="*80)
        print()
    
    # Create simulator and run
    try:
        simulator = PerformanceSimulator(config)
        results = simulator.run_simulation(
            data_dir=args.data,
            start_date=args.start,
            end_date=args.end,
        )
        
        # Print summary
        if not args.quiet:
            simulator.print_summary(results)
        
        # Generate reports
        if not args.no_report:
            output_dir = args.output if args.output else "reports"
            simulator.generate_report(results, output_dir, format=args.format)
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

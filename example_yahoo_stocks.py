"""Complete example: Download and trade stocks using Yahoo Finance data.

This script demonstrates the full workflow:
1. Download real stock data from Yahoo Finance
2. Run pair trading strategy on the data
3. Generate performance reports

This is the recommended approach for TradFi pair trading.
"""

from pathlib import Path
import subprocess
import sys


def download_stock_data(preset="tech", days=60):
    """Download stock data from Yahoo Finance.
    
    Args:
        preset: Stock preset (tech, finance, consumer, healthcare, all)
        days: Number of days to download (max ~60 for hourly data)
    """
    from datetime import datetime, timedelta
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    cmd = [
        sys.executable, "-m", "get_data.download_yahoo_stocks",
        "--preset", preset,
        "--interval", "1h",
        "--start", start_date.strftime("%Y-%m-%d"),
        "--end", end_date.strftime("%Y-%m-%d"),
        "--out", f"data/yahoo_{preset}_1h",
    ]
    
    print(f"\n{'='*80}")
    print(f"📥 Downloading {preset.upper()} stocks from Yahoo Finance")
    print(f"{'='*80}")
    print(f"Period: Last {days} days ({start_date.date()} to {end_date.date()})")
    print(f"Interval: 1 hour")
    print(f"Output: data/yahoo_{preset}_1h/")
    print()
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Download failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def run_stock_simulation(preset="tech"):
    """Run pair trading simulation on downloaded stock data."""
    from src.performance_simulator import PerformanceSimulator, SimulationConfig
    
    data_dir = f"data/yahoo_{preset}_1h"
    
    # Check if data exists
    if not Path(data_dir).exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        print("Please download data first.")
        return None
    
    # Map preset to reference symbols
    reference_map = {
        "tech": "AAPLUSDT",      # Apple
        "finance": "JPMUSDT",    # JPMorgan
        "consumer": "WMUSDT",    # Walmart
        "healthcare": "JNJUSDT", # Johnson & Johnson
        "all": "AAPLUSDT",       # Default to Apple
    }
    
    reference_symbol = reference_map.get(preset, "AAPLUSDT")
    
    print(f"\n{'='*80}")
    print(f"📊 Running Pair Trading Simulation on {preset.upper()} Stocks")
    print(f"{'='*80}")
    
    # Create configuration optimized for stocks
    config = SimulationConfig(
        data_dir=data_dir,
        interval="1h",
        initial_capital=100_000,
        
        # Reference asset
        reference_symbol=reference_symbol,
        
        # Strategy parameters (optimized for stocks)
        alpha1=0.15,  # Entry threshold
        alpha2=0.08,  # Exit threshold
        
        # Formation and trading periods
        formation_hours=21 * 24,  # 21 days
        trading_hours=7 * 24,     # 7 days
        step_hours=7 * 24,        # Rolling window
        
        # Risk management
        position_pct=0.08,  # 8% per position
        max_positions=8,
        
        # Trading costs
        trading_fee_pct=0.1,  # 0.1% trading fee
        
        # Output
        report_dir=f"reports/yahoo_{preset}",
        save_equity_curve=True,
        save_cycle_stats=True,
    )
    
    print(f"\n📋 Configuration:")
    print(f"  Data: {config.data_dir}")
    print(f"  Reference: {config.reference_symbol}")
    print(f"  Capital: ${config.initial_capital:,.0f}")
    print(f"  Entry threshold (alpha1): {config.alpha1}")
    print(f"  Exit threshold (alpha2): {config.alpha2}")
    print(f"  Formation period: {config.formation_hours // 24} days")
    print(f"  Trading period: {config.trading_hours // 24} days")
    print(f"  Position size: {config.position_pct*100}%")
    print(f"  Max positions: {config.max_positions}")
    print()
    
    # Run simulation
    try:
        simulator = PerformanceSimulator(config)
        results = simulator.run()
        
        print(f"\n{'='*80}")
        print("✅ Simulation Complete!")
        print(f"{'='*80}")
        print(f"📁 Reports saved to: {config.report_dir}/")
        print()
        
        return results
        
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def show_results(preset="tech"):
    """Show summary of simulation results."""
    report_dir = Path(f"reports/yahoo_{preset}")
    
    # Find the most recent report
    report_files = list(report_dir.glob("report_*.txt"))
    if not report_files:
        print(f"\n⚠️  No reports found in {report_dir}")
        return
    
    latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
    
    print(f"\n{'='*80}")
    print(f"📊 Latest Report: {latest_report.name}")
    print(f"{'='*80}\n")
    
    with open(latest_report) as f:
        print(f.read())


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download stocks and run pair trading simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download tech stocks and run simulation
  python example_yahoo_stocks.py --download --simulate --preset tech

  # Download all stocks
  python example_yahoo_stocks.py --download --preset all --days 30

  # Just run simulation on existing data
  python example_yahoo_stocks.py --simulate --preset finance

  # Full workflow with results
  python example_yahoo_stocks.py --download --simulate --show --preset tech
        """
    )
    
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download stock data from Yahoo Finance"
    )
    
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run pair trading simulation"
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show results from latest simulation"
    )
    
    parser.add_argument(
        "--preset",
        type=str,
        default="tech",
        choices=["tech", "finance", "consumer", "healthcare", "all"],
        help="Stock preset to use (default: tech)"
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Number of days to download (default: 60, max ~60 for hourly)"
    )
    
    args = parser.parse_args()
    
    # Show help if no arguments
    if not (args.download or args.simulate or args.show):
        parser.print_help()
        print("\n" + "="*80)
        print("💡 Quick Start:")
        print("="*80)
        print("\n1. Download and analyze tech stocks:")
        print("   python example_yahoo_stocks.py --download --simulate --preset tech")
        print("\n2. Try different sectors:")
        print("   python example_yahoo_stocks.py --download --simulate --preset finance")
        print("\n3. Run full analysis:")
        print("   python example_yahoo_stocks.py --download --simulate --show --preset all")
        return
    
    # Execute requested actions
    success = True
    
    if args.download:
        success = download_stock_data(preset=args.preset, days=args.days)
        if not success:
            print("\n⚠️  Download failed. Check your internet connection and try again.")
            return
    
    if args.simulate:
        if success or Path(f"data/yahoo_{args.preset}_1h").exists():
            results = run_stock_simulation(preset=args.preset)
            if not results:
                print("\n⚠️  Simulation failed. Check error messages above.")
                return
        else:
            print(f"\n⚠️  Data not found. Download data first:")
            print(f"   python example_yahoo_stocks.py --download --preset {args.preset}")
            return
    
    if args.show:
        show_results(preset=args.preset)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

"""Example: Using Bybit TradFi Data for Pair Trading

This script demonstrates how to download Bybit TradFi data and use it
for copula-based pair trading strategy.

Bybit offers perpetual futures on various TradFi assets including:
- Stock indices (e.g., SPX500, NDX100)
- Forex pairs (e.g., EURUSD, GBPUSD)
- Commodities (e.g., Gold, Silver)

Run this script to:
1. Download TradFi data from Bybit
2. Run the pair trading strategy on TradFi assets
3. Generate performance reports
"""

from pathlib import Path
from src.performance_simulator import (
    PerformanceSimulator,
    SimulationConfig,
)


def download_bybit_tradfi_data():
    """Download tokenized stock data from Bybit.
    
    Bybit offers perpetual futures on tokenized stocks like:
    - AAPLPERP (Apple)
    - TSLAUSDT (Tesla)
    - GOOGUSDT (Google)
    - MSFTUSDT (Microsoft)
    - And many more major US stocks
    """
    import subprocess
    
    # Download all default tokenized stocks
    cmd = [
        "python", "-m", "get_data.download_bybit_data",
        "--category", "linear",
        "--interval", "60",  # 1 hour
        "--start", "2023-01-01",
        "--end", "2024-12-31",
        "--out", "data/bybit_stocks_1h",
    ]
    
    print("📥 Downloading Bybit Tokenized Stock Data...")
    print(f"Command: {' '.join(cmd)}\n")
    print("This will download data for major tokenized stocks including:")
    print("  • Tech: AAPL, TSLA, GOOG, MSFT, AMZN, META, NFLX, NVDA")
    print("  • Finance: JPM, GS, BAC, V, MA")
    print("  • Consumer: NKE, SBUX, DIS, MC")
    print("  • Healthcare: JNJ, PFE")
    print("\nStarting download...\n")
    
    subprocess.run(cmd, check=True)
    print("\n✅ Download complete!")
    

def run_bybit_tradfi_simulation():
    """Run pair trading simulation on Bybit tokenized stock data."""
    print("\n" + "="*80)
    print("Bybit Tokenized Stocks Pair Trading Simulation")
    print("="*80)
    
    # Check if data directory exists
    data_dir = Path("data/bybit_stocks_1h")
    if not data_dir.exists():
        print(f"\n⚠️  Data directory not found: {data_dir}")
        print("Please run download_bybit_tradfi_data() first.")
        print("Or run: python example_bybit_tradfi.py --download")
        return None
    
    # Create configuration optimized for stock pairs
    config = SimulationConfig(
        data_dir="data/bybit_stocks_1h",
        interval="1h",
        initial_capital=100_000,
        
        # Use a stable tech stock as reference (or most liquid)
        reference_symbol="AAPLPERP",  # Apple as reference
        
        # Strategy parameters tuned for stock volatility
        # Stocks typically less volatile than crypto
        alpha1=0.15,  # Entry threshold (lower than crypto)
        alpha2=0.08,  # Exit threshold
        
        # Formation and trading periods
        formation_hours=30 * 24,  # 30 days formation period
        trading_hours=10 * 24,    # 10 days trading period
        step_hours=10 * 24,       # Step by trading period
        
        # Risk management
        position_pct=0.08,  # 8% of capital per position
        max_positions=8,
        
        # Trading costs for perpetual futures
        trading_fee_pct=0.06,  # Bybit maker fee ~0.02%, taker ~0.055%
        
        # Report settings
        report_dir="reports/bybit_tradfi",
        save_trades=True,
        save_equity_curve=True,
        save_cycle_stats=True,
    )
    
    print(f"\n📊 Configuration:")
    print(f"  Data directory: {config.data_dir}")
    print(f"  Reference asset: {config.reference_symbol}")
    print(f"  Formation period: {config.formation_hours // 24} days")
    print(f"  Trading period: {config.trading_hours // 24} days")
    print(f"  Initial capital: ${config.initial_capital:,.0f}")
    print(f"  Trading fee: {config.trading_fee_pct}%")
    
    # Create simulator and run
    simulator = PerformanceSimulator(config)
    
    try:
        results = simulator.run()
        print(f"\n✅ Simulation completed successfully!")
        print(f"📁 Reports saved to: {config.report_dir}")
        return results
        
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def example_crypto_to_tradfi_pairs():
    """Example: Mix crypto and TradFi assets for pair trading.
    
    This demonstrates combining different asset classes,
    which may reveal interesting cointegration relationships.
    """
    print("\n" + "="*80)
    print("Mixed Asset Class Pair Trading (Crypto + TradFi)")
    print("="*80)
    print("\n⚠️  This is an advanced example.")
    print("Mixing asset classes requires careful consideration of:")
    print("  - Different market hours and liquidity")
    print("  - Varying volatility profiles")
    print("  - Correlation stability across market conditions")
    print("\nFor production use, test thoroughly and adjust parameters accordingly.")


def quick_test_with_bybit_crypto():
    """Quick test with Bybit crypto data (similar to Binance)."""
    print("\n" + "="*80)
    print("Quick Test: Bybit Crypto Perpetuals")
    print("="*80)
    
    # You can use Bybit crypto data the same way as Binance
    # The data format is compatible
    
    config = SimulationConfig(
        data_dir="data/bybit_linear_1h",  # Bybit crypto perpetuals
        interval="1h",
        initial_capital=50_000,
        reference_symbol="BTCUSDT",
        alpha1=0.20,
        alpha2=0.10,
        formation_hours=24,  # Shorter for testing
        trading_hours=12,
        step_hours=12,
        report_dir="reports/bybit_crypto_test",
    )
    
    # Check if data exists
    if not Path(config.data_dir).exists():
        print(f"\n⚠️  Data directory not found: {config.data_dir}")
        print("Download data first using:")
        print(f"  python -m get_data.download_bybit_data \\")
        print(f"    --category linear \\")
        print(f"    --interval 60 \\")
        print(f"    --start 2022-01-01 \\")
        print(f"    --end 2024-12-31 \\")
        print(f"    --out {config.data_dir}")
        return None
    
    simulator = PerformanceSimulator(config)
    results = simulator.run()
    
    return results


def main():
    """Main example runner."""
    import sys
    
    print("\n" + "="*80)
    print("🚀 Bybit Tokenized Stocks Pair Trading")
    print("="*80)
    
    # Command line argument handling
    if len(sys.argv) > 1:
        if sys.argv[1] == "--download":
            download_bybit_tradfi_data()
            return
        elif sys.argv[1] == "--simulate":
            run_bybit_tradfi_simulation()
            return
    
    print("\nAvailable commands:")
    print("  1. Download tokenized stock data from Bybit")
    print("  2. Run pair trading simulation on stock data")
    print()
    print("Usage:")
    print("  python example_bybit_tradfi.py --download")
    print("  python example_bybit_tradfi.py --simulate")
    print()
    
    choice = input("Select option (1-2) or press Enter to exit: ").strip()
    
    if choice == "1":
        download_bybit_tradfi_data()
    elif choice == "2":
        run_bybit_tradfi_simulation()
    else:
        print("\n📚 Quick Start:")
        print("  1. Download data: python example_bybit_tradfi.py --download")
        print("  2. Run simulation: python example_bybit_tradfi.py --simulate")


if __name__ == "__main__":
    main()

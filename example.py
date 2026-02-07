"""
Example: Quick Start with the Copula-Based Pairs Trading Strategy

This script demonstrates basic usage of the reference-asset-based copula strategy.
"""

from pathlib import Path
from src.main import ReferenceAssetCopulaTradingStrategy

def main():
    """
    Run a simple backtest example using the smoke test data.
    
    The smoke test data includes only BTCUSDT and ETHUSDT for quick testing.
    """
    
    # Path to test data (smaller dataset for quick runs)
    data_dir = Path("data/binance_futures_1h_smoke")
    
    # Check if data exists
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("\nPlease download data first:")
        print("  python -m get_data.download_binance_futures \\")
        print("    --interval 1h \\")
        print("    --start 2021-01-01 \\")
        print("    --end 2023-01-19 \\")
        print("    --symbols BTCUSDT,ETHUSDT \\")
        print("    --out data/binance_futures_1h_smoke")
        return 1
    
    print("=" * 70)
    print("Copula-Based Pairs Trading Strategy - Example Run")
    print("=" * 70)
    print()
    print("Paper: Tadi & Witzany (2025), Financial Innovation 11:40")
    print()
    
    # Create strategy with paper's recommended parameters
    print("Initializing strategy...")
    strategy = ReferenceAssetCopulaTradingStrategy(
        reference_symbol="BTCUSDT",
        alpha1=0.20,  # Entry threshold (paper tests 0.10, 0.15, 0.20)
        alpha2=0.10,  # Exit threshold
        eg_alpha=1.00,  # Disable EG test (use only ADF + KSS)
        adf_alpha=0.10,  # ADF p-value threshold
        kss_critical=-1.92,  # KSS 10% critical value
        use_log_prices=True,  # Use log prices for stationarity
    )
    
    print(f"  Reference Asset: {strategy.reference_symbol}")
    print(f"  Entry Threshold (α₁): {strategy.alpha1}")
    print(f"  Exit Threshold (α₂): {strategy.alpha2}")
    print()
    
    # Run backtest
    print("Running backtest...")
    print("  Formation Period: 21 days (504 hours)")
    print("  Trading Period: 7 days (168 hours)")
    print("  Transaction Fee: 0.04% (4 bps)")
    print("  Initial Capital: $20,000 USDT")
    print()
    
    try:
        results = strategy.backtest(
            data_dir=str(data_dir),
            interval="1h",
            formation_hours=21 * 24,  # 3 weeks
            trading_hours=7 * 24,     # 1 week
            step_hours=7 * 24,        # Roll forward weekly
            fee_rate=0.0004,          # 4 bps (Binance taker fee)
            capital=20000.0,          # $20k per side
        )
        
        print("=" * 70)
        print("Backtest Complete!")
        print("=" * 70)
        print()
        print("Check the 'data/' directory for detailed trade logs.")
        print()
        
        return 0
        
    except Exception as e:
        print(f"Error during backtest: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

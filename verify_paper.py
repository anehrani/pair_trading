import sys
import os
import shutil
from pathlib import Path
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.live_trader import LiveTrader, TradingSignal
from src.paper_client import PaperTradingClient

def verify_paper_trading():
    print("="*60)
    print("VERIFYING PAPER TRADING")
    print("="*60)

    # 1. Setup Config
    config_path = "config.yaml"
    
    # 2. Initialize Live Trader
    print("\n1. Initializing LiveTrader...")
    try:
        trader = LiveTrader(config_path=config_path)
        print("   ✓ LiveTrader initialized")
    except Exception as e:
        print(f"   ✗ Failed to initialize: {e}")
        return

    # 3. Check Mode
    print("\n2. Checking Mode...")
    if trader.mode == "paper":
        print("   ✓ Mode is PAPER")
    else:
        print(f"   ✗ Mode is {trader.mode}, expected PAPER")
        return

    # 4. Check Client Type
    print("\n3. Checking Client Type...")
    if isinstance(trader.client, PaperTradingClient):
        print("   ✓ Client is PaperTradingClient")
    else:
        print(f"   ✗ Client is {type(trader.client)}, expected PaperTradingClient")
        return

    # 5. Initialize Data (Download from Yahoo)
    print("\n4. Initializing Data (Fetching from Yahoo)...")
    try:
        # Override formation days to small number for quick test if needed, 
        # but let's try with config config (21 days)
        trader.initialize_data()
        
        # Check buffer
        symbols = trader.buffer.symbols
        print(f"   ✓ Buffer initialized with {len(symbols)} symbols")
        
        # Check specific symbol data
        ref_sym = trader.config["strategy"]["reference_symbol"]
        df = trader.buffer.data.get(ref_sym)
        if df is not None and not df.empty:
            print(f"   ✓ Data found for {ref_sym}: {len(df)} candles")
            print(f"     Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"     Columns: {df.columns.tolist()}")
        else:
            print(f"   ✗ No data for {ref_sym}")
            
    except Exception as e:
        print(f"   ✗ Failed to initialize data: {e}")
        # traceback
        import traceback
        traceback.print_exc()
        return

    # 6. Check Balance
    print("\n5. Checking Balance...")
    balance = trader.client.get_account_balance()
    print(f"   ✓ Balance: {balance}")
    if balance['totalEq'] >= 100000.0:
         print("   ✓ Initial capital looks correct")
    else:
         print("   ✗ Initial capital unexpected")

    # 7. Simulate Trade Execution
    print("\n6. Simulating Trade Execution...")
    try:
        # Manually trigger a signal execution
        # Let's pretend we have a pair
        from src.strategy_core import TradingPair
        
        # Create a dummy signal
        pair = TradingPair(
            symbol1="AAPL", 
            symbol2="MSFT", 
            beta1=1.0, 
            beta2=0.8, 
            tau1=0.5, 
            tau2=0.5
        )
        trader.current_pair = pair
        # Set dummy model (not used in execution logic except for betas which we passed to calculate_position_sizes? 
        # Actually calculate_position_sizes uses model betas? No, execute_signal uses self.current_model.beta1)
        
        # creating a dummy model object
        class DummyModel:
            beta1 = 1.0
            beta2 = 0.8
        trader.current_model = DummyModel()
        
        # Mocking prices in buffer for position sizing because execute_signal uses buffer.get_latest_prices
        # We need to ensure we have data for AAPL and MSFT
        # We already fetched data in step 5 if they are in config.
        # If not in config, we might fail. AAPL and MSFT are in config.
        
        signal = TradingSignal(
            action="LONG_S1_SHORT_S2",
            h1_2=0.1,
            h2_1=0.9,
            timestamp=pd.Timestamp.now(tz="UTC")
        )
        
        print(f"   Executing signal: {signal.action}")
        trader.execute_signal(signal)
        
        # Check state
        if trader.state.active_position:
            print("   ✓ Position opened successfully")
            print(f"     Details: {trader.state.active_position}")
        else:
            print("   ✗ Failed to open position (state.active_position is None)")
            
        # Check balance again (should be changed due to fees)
        new_balance = trader.client.get_account_balance()
        print(f"   ✓ New Balance: {new_balance}")
        
    except Exception as e:
        print(f"   ✗ Failed to execute trade: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    verify_paper_trading()

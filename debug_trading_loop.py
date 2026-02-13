"""Simulate what happens in the trading loop after data update."""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import pandas as pd
from pathlib import Path
from src.data_buffer import DataBuffer
from src.strategy_core import TradingPair, CopulaModel, generate_signal

# Load config
import yaml
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Load state
with open("data/state.json") as f:
    state_data = json.load(f)

# Initialize buffer
strategy_conf = config.get("strategy", {})
mode = config.get("mode", "live")
default_ref = "SPY" if mode == "alpaca" else "BTCUSDT"
reference_symbol = strategy_conf.get("reference_symbol", default_ref)

all_symbols = [reference_symbol] + strategy_conf.get("symbols", [])
buffer = DataBuffer(symbols=all_symbols, max_days=strategy_conf.get("formation_days", 21) + 7)
buffer.load(Path("data/price_buffer.parquet"))

print(f"Buffer has {len(buffer.data)} symbols")
print(f"Symbols in buffer.data: {list(buffer.data.keys())[:5]}...")

# Reconstruct current pair from state
if state_data["current_pair"]:
    pair_dict = state_data["current_pair"]
    current_pair = TradingPair(
        symbol1=pair_dict["symbol1"],
        symbol2=pair_dict["symbol2"],
        beta1=pair_dict["beta1"],
        beta2=pair_dict["beta2"],
        tau1=pair_dict["tau1"],
        tau2=pair_dict["tau2"],
    )
    print(f"\nCurrent pair: {current_pair.symbol1} / {current_pair.symbol2}")
    
    # Check if pair symbols exist in buffer
    s1_exists = current_pair.symbol1 in buffer.data
    s2_exists = current_pair.symbol2 in buffer.data
    print(f"  {current_pair.symbol1} in buffer: {s1_exists}")
    print(f"  {current_pair.symbol2} in buffer: {s2_exists}")
    
    if not s1_exists or not s2_exists:
        print("\n❌ ERROR: Current pair symbols not in buffer!")
        print("This will cause the signal generation to fail.")
    else:
        # Try to get latest prices
        try:
            prices = buffer.get_latest_prices()
            print(f"\n✓ Got latest prices: {len(prices)} symbols")
            
            # Try to generate signal
            signal = generate_signal(
                prices,
                None,  # We'd need to reconstruct the full model
                current_pair,
                reference_symbol,
                strategy_conf.get("alpha1", 0.20),
                strategy_conf.get("alpha2", 0.10),
            )
            print(f"✓ Generated signal: {signal.action}")
        except Exception as e:
            print(f"\n❌ ERROR generating signal: {e}")

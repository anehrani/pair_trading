from __future__ import annotations

import argparse
import sys
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategy_core import (
    TradingPair,
    CopulaModel,
    select_trading_pair,
    fit_copula_model,
    generate_signal,
    calculate_position_sizes,
)

@dataclass
class Trade:
    symbol_long: str
    symbol_short: str
    qty_long: float
    qty_short: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price_long: float
    entry_price_short: float
    exit_price_long: float
    exit_price_short: float
    fees: float
    pnl: float

@dataclass
class PortfolioState:
    """Track portfolio state across the backtest."""
    initial_capital: float
    current_capital: float
    peak_capital: float
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    
    def update(self, pnl: float):
        self.current_capital += pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)
        drawdown = self.peak_capital - self.current_capital
        self.max_drawdown = max(self.max_drawdown, drawdown)
        if self.peak_capital > 0:
            dd_pct = drawdown / self.peak_capital
            self.max_drawdown_pct = max(self.max_drawdown_pct, dd_pct)
    
    @property
    def available_capital_per_side(self) -> float:
        """Dynamic position sizing based on current equity."""
        return self.current_capital * 0.3  # 30% per side max
    
    def should_stop(self, max_dd_pct: float = 0.20) -> bool:
        return self.max_drawdown_pct >= max_dd_pct

@dataclass
class ExecutionModel:
    fee_rate: float = 0.001
    slippage_bps: float = 5.0
    
    def apply_slippage(self, price: float, side: str) -> float:
        total_slip = self.slippage_bps / 10_000
        if side == "BUY":
            return price * (1 + total_slip)
        else:
            return price * (1 - total_slip)
    
    def calculate_fees(self, notional: float) -> float:
        return notional * self.fee_rate

@dataclass
class StrategyConfig:
    reference_symbol: str
    symbols: list[str]
    eg_alpha: float = 0.05
    adf_alpha: float = 0.05
    kss_critical: float = -1.645
    alpha1: float = 0.20
    alpha2: float = 0.10
    capital_per_side: float = 10000.0
    initial_capital: float = 100000.0
    fee_rate: float = 0.001
    formation_days: int = 60
    trading_days: int = 7
    max_drawdown_pct: float = 0.20
    slippage_bps: float = 5.0
    
    def __post_init__(self):
        # Relax constraints to match config.yaml
        pass

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

def setup_logging(level: str = "INFO", log_file: str = None):
    logger.remove()
    logger.add(sys.stderr, level=level)
    if log_file:
        logger.add(log_file, level="DEBUG")

def load_local_data(data_dir: str, symbols: list[str], reference_symbol: str) -> pd.DataFrame:
    all_syms = list(set(symbols + [reference_symbol]))
    closes = pd.DataFrame()
    
    mapping = {
        "AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon", "NVDA": "NVIDIA",
        "GOOGL": "Alphabet", "META": "Meta", "TSLA": "Tesla", "UNH": "UnitedHealth",
        "JNJ": "JohnsonJohnson", "JPM": "JPMorgan", "XOM": "ExxonMobil",
        "PG": "ProctorGamble", "HD": "HomeDepot", "CVX": "Chevron", "KO": "CocaCola",
        "GLD": "Gold", "SLV": "Silver", "USO": "CrudeOilWTI", "UNG": "NaturalGas",
        "PALL": "Palladium", "PPLT": "Platinum", "COPX": "Copper", "SPY": "SP500"
    }
    
    for sym in all_syms:
        base = mapping.get(sym, sym)
        file_name = f"{base}_1d.csv"
        path = Path(data_dir) / file_name
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            col = 'close' if 'close' in df.columns else 'Close'
            tcol = 'timestamp' if 'timestamp' in df.columns else 'Date'
            df[tcol] = pd.to_datetime(df[tcol])
            df.set_index(tcol, inplace=True)
            closes[sym] = df[col]
        except Exception as e:
            logger.error(f"Error loading {sym}: {e}")
            
    closes = closes.dropna(how="all").ffill().dropna()
    if not closes.empty:
        closes.index = pd.to_datetime(closes.index, utc=True)
    return closes

def run_backtest(df: pd.DataFrame, config: StrategyConfig):
    portfolio = PortfolioState(config.initial_capital, config.initial_capital, config.initial_capital)
    exec_model = ExecutionModel(config.fee_rate, config.slippage_bps)
    trades = []
    equity_curve = []
    
    current_idx = config.formation_days
    active_pos = None
    
    while current_idx < len(df):
        if portfolio.should_stop(config.max_drawdown_pct):
            break
            
        # 1. Formation
        window_start = current_idx - config.formation_days
        formation_df = df.iloc[window_start:current_idx]
        
        # Log prices for selection (as in LiveTrader)
        log_formation = np.log(formation_df.astype(float))
        
        try:
            pair, results_df = select_trading_pair(
                log_formation,
                reference_symbol=config.reference_symbol,
                eg_alpha=config.eg_alpha,
                adf_alpha=config.adf_alpha,
                kss_critical=config.kss_critical
            )
        except Exception as e:
            logger.error(f"Selection error at {df.index[current_idx]}: {e}")
            pair = None

        if not pair:
            equity_curve.append((df.index[current_idx], portfolio.current_capital - portfolio.initial_capital))
            current_idx += 1
            continue
            
        # 2. Fit
        try:
            model = fit_copula_model(log_formation, pair, reference_symbol=config.reference_symbol)
        except Exception as e:
            logger.error(f"Fit error: {e}")
            model = None
            
        if not model:
            current_idx += 1
            continue
            
        # 3. Trade Window
        exit_idx = min(current_idx + config.trading_days, len(df))
        for i in range(current_idx, exit_idx):
            curr_time = df.index[i]
            prices = df.iloc[i].to_dict()
            
            # Signal
            try:
                sig = generate_signal(
                    prices, model, pair, 
                    reference_symbol=config.reference_symbol,
                    alpha1=config.alpha1, alpha2=config.alpha2
                )
            except Exception:
                sig = None
                
            if not sig:
                equity_curve.append((curr_time, portfolio.current_capital - portfolio.initial_capital))
                continue

            # Entry
            if sig.action.startswith("LONG") and not active_pos:
                # Determine which is long/short from signal name
                # LONG_S1_SHORT_S2 means LONG pair.symbol2 and SHORT pair.symbol1 (Wait, let's check Table 4 in paper/code)
                # Looking at live_trader.py:
                # LONG_S1_SHORT_S2 -> LONG sym2, SHORT sym1
                # SHORT_S1_LONG_S2 -> SHORT sym2, LONG sym1
                
                sym1, sym2 = pair.symbol1, pair.symbol2
                if sig.action == "LONG_S1_SHORT_S2":
                    l_sym, s_sym = sym2, sym1
                else: 
                    l_sym, s_sym = sym1, sym2
                
                p_long = exec_model.apply_slippage(prices[l_sym], "BUY")
                p_short = exec_model.apply_slippage(prices[s_sym], "SELL")
                
                q1, q2 = calculate_position_sizes(
                    model.beta1, model.beta2, prices[sym1], prices[sym2],
                    config.capital_per_side
                )
                # q1 is for sym1, q2 is for sym2
                # In live_trader:
                # LONG_S1_SHORT_S2 -> Buy q2 of sym2, Sell q1 of sym1
                q_long = abs(q2) if l_sym == sym2 else abs(q1)
                q_short = abs(q1) if s_sym == sym1 else abs(q2)
                
                entry_fees = exec_model.calculate_fees(q_long * p_long + q_short * p_short)
                portfolio.current_capital -= entry_fees
                
                active_pos = {
                    "l_sym": l_sym, "s_sym": s_sym,
                    "l_qty": q_long, "s_qty": q_short,
                    "l_entry": p_long, "s_entry": p_short,
                    "entry_time": curr_time, "entry_fees": entry_fees
                }
                logger.info(f"Entry {sig.action} at {curr_time}: {l_sym}/{s_sym}")

            # Exit
            elif sig.action == "CLOSE" and active_pos:
                p_l_exit = exec_model.apply_slippage(prices[active_pos['l_sym']], "SELL")
                p_s_exit = exec_model.apply_slippage(prices[active_pos['s_sym']], "BUY")
                
                pnl = active_pos['l_qty'] * (p_l_exit - active_pos['l_entry']) + \
                      active_pos['s_qty'] * (active_pos['s_entry'] - p_s_exit)
                
                exit_fees = exec_model.calculate_fees(active_pos['l_qty'] * p_l_exit + active_pos['s_qty'] * p_s_exit)
                net_pnl = pnl - exit_fees
                portfolio.update(net_pnl)
                
                trades.append(Trade(
                    symbol_long=active_pos['l_sym'], symbol_short=active_pos['s_sym'],
                    qty_long=active_pos['l_qty'], qty_short=active_pos['s_qty'],
                    entry_time=active_pos['entry_time'], exit_time=curr_time,
                    entry_price_long=active_pos['l_entry'], entry_price_short=active_pos['s_entry'],
                    exit_price_long=p_l_exit, exit_price_short=p_s_exit,
                    fees=active_pos['entry_fees'] + exit_fees, pnl=net_pnl
                ))
                logger.info(f"Exit at {curr_time} PnL: ${net_pnl:.2f}")
                active_pos = None

            equity_curve.append((curr_time, portfolio.current_capital - portfolio.initial_capital))
            
        current_idx += config.trading_days

    return trades, equity_curve

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir", default="data/complete_data")
    parser.add_argument("--formation-days", type=int)
    parser.add_argument("--trading-days", type=int)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    
    setup_logging("INFO", "reports/backtest_local.log")
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    s_cfg = cfg['strategy']
    if args.formation_days: s_cfg['formation_days'] = args.formation_days
    if args.trading_days: s_cfg['trading_days'] = args.trading_days
    
    config = StrategyConfig.from_dict(s_cfg)
    df = load_local_data(args.data_dir, config.symbols, config.reference_symbol)
    if df.empty: return
    
    trades, equity = run_backtest(df, config)
    
    # Save results
    if trades:
        res = pd.DataFrame([asdict(t) for t in trades])
        os.makedirs("reports", exist_ok=True)
        out = f"reports/backtest_local_f{config.formation_days}_t{config.trading_days}.csv"
        res.to_csv(out, index=False)
        logger.info(f"Backtest complete. {len(trades)} trades. Results in {out}")
        
        # Simple summary
        total_pnl = res['pnl'].sum()
        win_rate = (res['pnl'] > 0).mean()
        print(f"\nFinal PnL: ${total_pnl:,.2f} | Win Rate: {win_rate:.1%}")
    else:
        logger.warning("No trades executed.")

if __name__ == "__main__":
    main()

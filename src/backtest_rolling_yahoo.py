from __future__ import annotations

import argparse
import sys
import os
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf
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
    open_positions: int = 0
    
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
        """Circuit breaker — stop trading if drawdown exceeds threshold."""
        return self.max_drawdown_pct >= max_dd_pct

@dataclass
class ExecutionModel:
    """Realistic execution modeling."""
    fee_rate: float = 0.001          # 10 bps per side
    slippage_bps: float = 5.0        # 5 bps slippage
    market_impact_bps: float = 2.0   # 2 bps market impact
    
    def apply_slippage(self, price: float, side: str) -> float:
        """Adjust price for slippage — worse fill for the trader."""
        total_slip = (self.slippage_bps + self.market_impact_bps) / 10_000
        if side == "BUY":
            return price * (1 + total_slip)
        else:  # SELL
            return price * (1 - total_slip)
    
    def calculate_fees(self, notional: float) -> float:
        return notional * self.fee_rate
    
    def total_execution_cost(self, entry_notional: float, exit_notional: float) -> float:
        return self.calculate_fees(entry_notional + exit_notional)

@dataclass
class RiskLimits:
    stop_loss_pct: float = 0.05      # 5% loss on notional
    take_profit_pct: float = 0.10    # 10% profit on notional
    max_holding_periods: int = None   # Optional time-based stop

def check_risk_limits(active_pos: dict, curr_prices: dict, risk_limits: RiskLimits, periods_held: int) -> bool:
    """Return True if position should be force-closed."""
    sym1, sym2 = active_pos['syms']
    q1, q2 = active_pos['qties']
    p1_entry, p2_entry = active_pos['entry_prices']
    p1_now, p2_now = curr_prices[sym1], curr_prices[sym2]
    
    # Unrealized PnL
    unrealized = q1 * (p1_now - p1_entry) + q2 * (p2_now - p2_entry)
    entry_notional = abs(q1 * p1_entry) + abs(q2 * p2_entry)
    
    if entry_notional == 0:
        return False
        
    pnl_pct = unrealized / entry_notional
    
    if pnl_pct <= -risk_limits.stop_loss_pct:
        logger.info(f"Stop-loss triggered: {pnl_pct:.2%}")
        return True
    if pnl_pct >= risk_limits.take_profit_pct:
        logger.info(f"Take-profit triggered: {pnl_pct:.2%}")
        return True
    if risk_limits.max_holding_periods and periods_held >= risk_limits.max_holding_periods:
        logger.info(f"Time stop triggered after {periods_held} periods")
        return True
        
    return False

@dataclass
class BacktestMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    median_pnl: float
    max_win: float
    max_loss: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    avg_holding_time: timedelta
    roi: float
    
    def __str__(self) -> str:
        return (
            f"\n{'='*50}\n"
            f"       BACKTEST RESULTS\n"
            f"{'='*50}\n"
            f"Total Trades:      {self.total_trades}\n"
            f"Win Rate:          {self.win_rate:.2%}\n"
            f"Total PnL:         ${self.total_pnl:>12,.2f}\n"
            f"Avg PnL/Trade:     ${self.avg_pnl:>12,.2f}\n"
            f"Median PnL/Trade:  ${self.median_pnl:>12,.2f}\n"
            f"Max Win:           ${self.max_win:>12,.2f}\n"
            f"Max Loss:          ${self.max_loss:>12,.2f}\n"
            f"Profit Factor:     {self.profit_factor:>12.2f}\n"
            f"Sharpe Ratio:      {self.sharpe_ratio:>12.2f}\n"
            f"Sortino Ratio:     {self.sortino_ratio:>12.2f}\n"
            f"Max Drawdown:      ${self.max_drawdown:>12,.2f}\n"
            f"Max Drawdown %:    {self.max_drawdown_pct:>12.2%}\n"
            f"Avg Holding Time:  {self.avg_holding_time}\n"
            f"ROI:               {self.roi:>12.2%}\n"
            f"{'='*50}"
        )

@dataclass
class StrategyConfig:
    reference_symbol: str
    symbols: list[str]
    eg_alpha: float = 0.05
    adf_alpha: float = 0.05
    kss_critical: float = -1.645
    alpha1: float = 0.05
    alpha2: float = 0.5
    capital_per_side: float = 50000.0
    initial_capital: float = 100000.0
    fee_rate: float = 0.001
    formation_days: int = 60
    trading_days: int = 7
    max_drawdown_pct: float = 0.20
    slippage_bps: float = 5.0
    interval: str = "1h"
    
    def __post_init__(self):
        assert 0 < self.eg_alpha <= 1, f"eg_alpha must be in (0,1], got {self.eg_alpha}"
        assert 0 < self.adf_alpha <= 1, f"adf_alpha must be in (0,1], got {self.adf_alpha}"
        assert 0 < self.alpha1 < 1, f"alpha1 must be in (0,1), got {self.alpha1}"
        assert 0 < self.alpha2 < 0.5, f"alpha2 must be in (0,0.5), got {self.alpha2}"
        assert self.capital_per_side <= self.initial_capital, (
            "capital_per_side cannot exceed initial_capital"
        )
        assert self.formation_days >= 30, "Need at least 30 days for formation"
        assert self.trading_days >= 1, "Need at least 1 trading day"
        assert self.interval in ["5m", "15m", "30m", "1h", "1d"], (
            f"Invalid interval: {self.interval}"
        )
        assert self.reference_symbol not in self.symbols, (
            "Reference symbol should not be in the symbols list"
        )
    
    @classmethod
    def from_dict(cls, d: dict) -> "StrategyConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

def setup_logging(level: str = "INFO", log_file: str = None):
    """Configure structured logging."""
    logger.remove()  # Remove default handler
    
    # Console output — clean format
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )
    
    # File output — full detail
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
            rotation="10 MB",
            retention="30 days",
        )

def _fetch_from_yahoo(symbols: list[str], interval: str) -> pd.DataFrame:
    """Actual Yahoo Finance download logic."""
    logger.info(f"Downloading data for {len(symbols)} symbols with interval {interval}...")
    
    period = "max"
    if interval in ["5m", "15m", "30m"]:
        period = "60d"
    elif interval == "1h":
        period = "730d"
        
    df = yf.download(
        tickers=symbols,
        period=period,
        interval=interval,
        progress=False,
        group_by='ticker',
        auto_adjust=True
    )
    
    if df.empty:
        return pd.DataFrame()
        
    closes = pd.DataFrame()
    for sym in symbols:
        try:
            if isinstance(df.columns, pd.MultiIndex):
                if sym in df.columns.levels[0]:
                    closes[sym] = df[sym]["Close"]
            else:
                closes[sym] = df["Close"]
        except (KeyError, Exception):
            continue
            
    closes = closes.dropna(how="all")
    closes = closes.ffill(limit=5)
    
    missing_pct = closes.isnull().mean()
    bad_syms = missing_pct[missing_pct > 0.1].index.tolist()
    if bad_syms:
        closes = closes.drop(columns=bad_syms)
        
    closes = closes.dropna()
    
    if not closes.empty:
        if closes.index.tz is None:
            closes.index = closes.index.tz_localize("UTC")
        else:
            closes.index = closes.index.tz_convert("UTC")
            
    return closes

def download_data(
    symbols: list[str],
    interval: str = "1h",
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """Download with disk caching and retry logic."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        # Cache key based on symbols, interval, and current hour (to avoid stale data)
        cache_key = hashlib.md5(
            f"{sorted(symbols)}_{interval}_{datetime.now().strftime('%Y-%m-%d_%H')}".encode()
        ).hexdigest()[:12]
        cache_path = Path(cache_dir) / f"prices_{cache_key}.parquet"
        
        if cache_path.exists():
            logger.info(f"Loading cached data from {cache_path}")
            return pd.read_parquet(cache_path)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            closes = _fetch_from_yahoo(symbols, interval)
            if not closes.empty:
                if cache_dir:
                    closes.to_parquet(cache_path)
                    logger.info(f"Cached data to {cache_path}")
                return closes
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    logger.error("All download attempts failed")
    return pd.DataFrame()

def compute_metrics(
    trades: list[Trade],
    equity_curve: list[tuple],
    initial_capital: float,
) -> BacktestMetrics:
    if not trades:
        return None
        
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    
    returns = [t.pnl / initial_capital for t in trades]
    std_returns = np.std(returns)
    sharpe = (np.mean(returns) / std_returns * np.sqrt(252)) if std_returns > 0 else 0
    
    downside = [r for r in returns if r < 0]
    downside_std = np.std(downside) if downside else 1e-9
    sortino = (np.mean(returns) / downside_std * np.sqrt(252))
    
    eq_values = [initial_capital + pnl for _, pnl in equity_curve]
    peak = eq_values[0]
    max_dd = 0
    for v in eq_values:
        peak = max(peak, v)
        max_dd = max(max_dd, peak - v)
    max_dd_pct = max_dd / peak if peak > 0 else 0
    
    holding_times = [t.exit_time - t.entry_time for t in trades]
    avg_hold = sum(holding_times, timedelta()) / len(holding_times) if holding_times else timedelta()
    
    return BacktestMetrics(
        total_trades=len(trades),
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=len(wins) / len(trades),
        total_pnl=sum(pnls),
        avg_pnl=np.mean(pnls),
        median_pnl=np.median(pnls),
        max_win=max(pnls),
        max_loss=min(pnls),
        profit_factor=gross_profit / gross_loss,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        avg_holding_time=avg_hold,
        roi=sum(pnls) / initial_capital,
    )

def plot_results(
    trades: list[Trade],
    equity_curve: list[tuple],
    initial_capital: float,
    save_path: str = None,
):
    if not equity_curve:
        return
        
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), height_ratios=[3, 1, 1])
    
    times, pnls = zip(*equity_curve)
    equity = [initial_capital + p for p in pnls]
    axes[0].plot(times, equity, 'b-', linewidth=1.5)
    axes[0].fill_between(times, initial_capital, equity, alpha=0.1,
                         where=[e >= initial_capital for e in equity], color='green')
    axes[0].fill_between(times, initial_capital, equity, alpha=0.1,
                         where=[e < initial_capital for e in equity], color='red')
    axes[0].axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].set_title('Equity Curve')
    axes[0].grid(True, alpha=0.3)
    
    peak = equity[0]
    drawdowns = []
    for e in equity:
        peak = max(peak, e)
        drawdowns.append((e - peak) / peak * 100)
    axes[1].fill_between(times, 0, drawdowns, color='red', alpha=0.3)
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].set_title('Drawdown')
    axes[1].grid(True, alpha=0.3)
    
    if trades:
        trade_pnls = [t.pnl for t in trades]
        colors = ['green' if p > 0 else 'red' for p in trade_pnls]
        axes[2].bar(range(len(trade_pnls)), trade_pnls, color=colors, alpha=0.7)
        axes[2].axhline(y=0, color='black', linewidth=0.5)
        axes[2].set_ylabel('Trade PnL ($)')
        axes[2].set_xlabel('Trade Number')
        axes[2].set_title('Individual Trade PnL')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    plt.show()

def evaluate_pairs_parallel(
    formation_data: pd.DataFrame,
    ref_sym: str,
    strat_cfg: StrategyConfig,
    max_pairs: int = 1,
) -> list[tuple[TradingPair, CopulaModel]]:
    """Evaluate all candidate pairs and return the top N."""
    candidates = [s for s in formation_data.columns if s != ref_sym]
    
    results = []
    for sym in candidates:
        try:
            # We use the select_trading_pair but we might want all results
            # For now, let's keep it simple and just iterate over candidates
            # and fit copulas for those that pass cointegration
            pair, _ = select_trading_pair(
                formation_data[[ref_sym, sym]],
                reference_symbol=ref_sym,
                eg_alpha=strat_cfg.eg_alpha,
                adf_alpha=strat_cfg.adf_alpha,
                kss_critical=strat_cfg.kss_critical,
            )
            if pair:
                model = fit_copula_model(formation_data, pair, reference_symbol=ref_sym)
                if model:
                    # Use a combination of tau and cointegration for scoring if needed
                    # For now just use the first one that fits
                    results.append((pair, model))
        except Exception as e:
            logger.debug(f"Failed evaluating {sym}: {e}")
            continue
            
    # In this implementation, select_trading_pair already picks the "best" but 
    # if we want to parallelize or handle multiple pairs, we do it here.
    return results[:max_pairs]

def run_backtest(closes: pd.DataFrame, config: dict):
    strat_cfg = StrategyConfig.from_dict(config["strategy"])
    initial_capital = strat_cfg.initial_capital
    ref_sym = strat_cfg.reference_symbol
    
    portfolio = PortfolioState(
        initial_capital=initial_capital,
        current_capital=initial_capital,
        peak_capital=initial_capital,
    )
    
    exec_model = ExecutionModel(
        fee_rate=strat_cfg.fee_rate,
        slippage_bps=strat_cfg.slippage_bps,
    )
    
    risk_limits = RiskLimits(
        stop_loss_pct=config["strategy"].get("stop_loss_pct", 0.05),
        take_profit_pct=config["strategy"].get("take_profit_pct", 0.10),
        max_holding_periods=config["strategy"].get("max_holding_periods"),
    )
    
    time_diffs = closes.index.to_series().diff().dropna()
    median_diff = time_diffs.median()
    periods_per_day = pd.Timedelta(days=1) / median_diff
    
    formation_periods = int(strat_cfg.formation_days * periods_per_day)
    trading_periods = int(strat_cfg.trading_days * periods_per_day)
    
    all_trades = []
    equity_curve = []
    log_closes = np.log(closes.astype(float))
    
    i = 0
    total_len = len(closes)
    
    while i + formation_periods + trading_periods <= total_len:
        if portfolio.should_stop(strat_cfg.max_drawdown_pct):
            logger.warning(f"Circuit breaker triggered at {portfolio.max_drawdown_pct:.2%} drawdown")
            break
            
        formation_idx = closes.index[i : i + formation_periods]
        trading_idx = closes.index[i + formation_periods : i + formation_periods + trading_periods]
        
        formation_data = log_closes.loc[formation_idx]
        trading_data = closes.loc[trading_idx]
        
        # Use parallel/multi-pair evaluation
        pairs_and_models = evaluate_pairs_parallel(formation_data, ref_sym, strat_cfg)
        
        if pairs_and_models:
            # For now, just trade the best pair found
            pair, model = pairs_and_models[0]
            
            active_pos = None
            periods_held = 0
            
            for t, row in trading_data.iterrows():
                log_row = log_closes.loc[t].to_dict()
                try:
                    signal = generate_signal(log_row, model, pair, ref_sym, strat_cfg.alpha1, strat_cfg.alpha2)
                except Exception:
                    continue
                    
                curr_prices = row.to_dict()
                capital_per_side = portfolio.available_capital_per_side
                
                if active_pos is None:
                    if signal.action == "LONG_S1_SHORT_S2":
                        q1, q2 = calculate_position_sizes(model.beta1, model.beta2, curr_prices[pair.symbol1], curr_prices[pair.symbol2], capital_per_side)
                        p1_fill = exec_model.apply_slippage(curr_prices[pair.symbol1], "BUY")
                        p2_fill = exec_model.apply_slippage(curr_prices[pair.symbol2], "SELL")
                        active_pos = {
                            'type': 'LONG_S1_SHORT_S2',
                            'entry_prices': (p1_fill, p2_fill),
                            'qties': (q1, -q2),
                            'entry_time': t,
                            'syms': (pair.symbol1, pair.symbol2)
                        }
                        periods_held = 0
                    elif signal.action == "SHORT_S1_LONG_S2":
                        q1, q2 = calculate_position_sizes(model.beta1, model.beta2, curr_prices[pair.symbol1], curr_prices[pair.symbol2], capital_per_side)
                        p1_fill = exec_model.apply_slippage(curr_prices[pair.symbol1], "SELL")
                        p2_fill = exec_model.apply_slippage(curr_prices[pair.symbol2], "BUY")
                        active_pos = {
                            'type': 'SHORT_S1_LONG_S2',
                            'entry_prices': (p1_fill, p2_fill),
                            'qties': (-q1, q2),
                            'entry_time': t,
                            'syms': (pair.symbol1, pair.symbol2)
                        }
                        periods_held = 0
                else:
                    periods_held += 1
                    is_end = (t == trading_idx[-1])
                    should_exit = (
                        signal.action == "CLOSE"
                        or is_end
                        or check_risk_limits(active_pos, curr_prices, risk_limits, periods_held)
                    )
                    
                    if should_exit:
                        sym1, sym2 = active_pos['syms']
                        q1, q2 = active_pos['qties']
                        p1_entry, p2_entry = active_pos['entry_prices']
                        
                        p1_exit = exec_model.apply_slippage(curr_prices[sym1], "SELL" if q1 > 0 else "BUY")
                        p2_exit = exec_model.apply_slippage(curr_prices[sym2], "SELL" if q2 > 0 else "BUY")
                        
                        pnl1 = q1 * (p1_exit - p1_entry)
                        pnl2 = q2 * (p2_exit - p2_entry)
                        
                        entry_notional = abs(q1 * p1_entry) + abs(q2 * p2_entry)
                        exit_notional = abs(q1 * p1_exit) + abs(q2 * p2_exit)
                        fees = exec_model.total_execution_cost(entry_notional, exit_notional)
                        pnl = pnl1 + pnl2 - fees
                        
                        all_trades.append(Trade(
                            symbol_long=sym1 if q1 > 0 else sym2,
                            symbol_short=sym2 if q1 > 0 else sym1,
                            qty_long=abs(q1) if q1 > 0 else abs(q2),
                            qty_short=abs(q2) if q1 > 0 else abs(q1),
                            entry_time=active_pos['entry_time'],
                            exit_time=t,
                            entry_price_long=p1_entry if q1 > 0 else p2_entry,
                            entry_price_short=p2_entry if q1 > 0 else p1_entry,
                            exit_price_long=p1_exit if q1 > 0 else p2_exit,
                            exit_price_short=p2_exit if q1 > 0 else p1_exit,
                            fees=fees,
                            pnl=pnl
                        ))
                        portfolio.update(pnl)
                        active_pos = None
                            
            equity_curve.append((trading_idx[-1], portfolio.current_capital - initial_capital))
        else:
            equity_curve.append((trading_idx[-1], portfolio.current_capital - initial_capital))
            
        i += trading_periods
        
    return all_trades, equity_curve

def main():
    parser = argparse.ArgumentParser(description="Rolling copula pairs trading backtest")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--formation-days", type=int, default=60)
    parser.add_argument("--trading-days", type=int, default=7)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    
    setup_logging(level=args.log_level, log_file="reports/backtest.log")
    
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return

    with open(args.config) as f:
        raw_config = yaml.safe_load(f)
    
    # Merge CLI args into config
    raw_config["strategy"]["formation_days"] = args.formation_days
    raw_config["strategy"]["trading_days"] = args.trading_days
    raw_config["strategy"]["interval"] = args.interval
    
    try:
        strat_cfg = StrategyConfig.from_dict(raw_config["strategy"])
    except (AssertionError, TypeError, Exception) as e:
        logger.error(f"Invalid configuration: {e}")
        return
    
    symbols = [strat_cfg.reference_symbol] + strat_cfg.symbols
    
    closes = download_data(
        symbols,
        interval=strat_cfg.interval,
        cache_dir="data/cache" if not args.no_cache else None,
    )
    
    if closes.empty:
        logger.error("No data available")
        return
    
    logger.info(f"Data: {len(closes)} bars, {closes.index.min()} -> {closes.index.max()}")
    
    all_trades, equity_curve = run_backtest(closes, raw_config)
    
    if all_trades:
        metrics = compute_metrics(all_trades, equity_curve, strat_cfg.initial_capital)
        print(metrics)
        
        os.makedirs("reports", exist_ok=True)
        df_trades = pd.DataFrame([vars(t) for t in all_trades])
        filename = f"reports/backtest_{args.interval}_f{args.formation_days}_t{args.trading_days}.csv"
        df_trades.to_csv(filename, index=False)
        logger.info(f"Saved trades to {filename}")
        
        if args.plot:
            plot_results(
                all_trades, equity_curve, strat_cfg.initial_capital,
                save_path=f"reports/equity_{args.interval}.png",
            )
    else:
        print("No trades executed.")

if __name__ == "__main__":
    main()

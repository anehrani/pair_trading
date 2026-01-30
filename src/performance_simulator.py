"""Performance Simulation Library for Pair Trading Algorithm

This module provides a comprehensive performance simulation framework for the 
copula-based pairs trading algorithm. It handles portfolio management, order 
execution simulation, performance metrics calculation, and detailed reporting.

Features:
- Portfolio management with multiple position tracking
- Realistic order execution simulation with fees
- Comprehensive performance metrics (Sharpe, Sortino, Calmar, etc.)
- Detailed trade analytics and statistics
- Risk metrics (VaR, CVaR, max drawdown)
- Rolling performance analysis
- Beautiful report generation with visualizations
- Trade journal export

Usage:
    from src.performance_simulator import PerformanceSimulator, SimulationConfig
    
    config = SimulationConfig(
        initial_capital=100_000,
        reference_symbol="BTCUSDT",
        alpha1=0.20,
        alpha2=0.10,
        fee_rate=0.0004
    )
    
    simulator = PerformanceSimulator(config)
    results = simulator.run_simulation(
        data_dir="data/binance_futures_1h",
        start_date="2023-01-01",
        end_date="2024-01-01"
    )
    
    # Generate comprehensive report
    simulator.generate_report(results, output_dir="reports")
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

# Import the backtesting logic
from src.backtest_reference_copula import (
    BacktestConfig,
    Trade,
    performance_summary,
    run_cycle,
)
from src.data_io import load_closes_from_dir

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class SimulationConfig:
    """Configuration for performance simulation.
    
    Attributes:
        initial_capital: Starting capital for the portfolio
        reference_symbol: Reference asset symbol (default: BTCUSDT)
        interval: Time interval for data (e.g., '1h', '1d')
        formation_hours: Number of hours for formation period (default: 21 days)
        trading_hours: Number of hours for trading period (default: 7 days)
        step_hours: Number of hours to step forward between cycles
        alpha1: Entry threshold for copula h-functions
        alpha2: Exit threshold for copula h-functions
        fee_rate: Transaction fee rate (default: 0.04%)
        capital_per_side: Maximum capital per side of trade
        eg_alpha: Engle-Granger p-value threshold (1.0 to disable)
        adf_alpha: ADF p-value threshold on spread
        kss_critical_10pct: KSS critical value at 10% level
        use_log_prices: Whether to use log prices for cointegration
        max_positions: Maximum number of simultaneous positions (not currently used)
        risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
    """
    
    initial_capital: float = 100_000.0
    reference_symbol: str = "BTCUSDT"
    interval: str = "1h"
    formation_hours: int = 21 * 24  # 21 days
    trading_hours: int = 7 * 24     # 7 days
    step_hours: int = 7 * 24        # 7 days
    alpha1: float = 0.20
    alpha2: float = 0.10
    fee_rate: float = 0.0004
    capital_per_side: float = 20_000.0
    eg_alpha: float = 1.00
    adf_alpha: float = 0.10
    kss_critical_10pct: float = -1.92
    use_log_prices: bool = True
    max_positions: int = 1
    risk_free_rate: float = 0.02  # 2% annual
    
    def to_backtest_config(self) -> BacktestConfig:
        """Convert to BacktestConfig for running cycles."""
        return BacktestConfig(
            reference_symbol=self.reference_symbol,
            interval=self.interval,
            formation_hours=self.formation_hours,
            trading_hours=self.trading_hours,
            step_hours=self.step_hours,
            eg_alpha=self.eg_alpha,
            adf_alpha=self.adf_alpha,
            kss_critical_10pct=self.kss_critical_10pct,
            use_intercept_beta=False,
            alpha1=self.alpha1,
            alpha2=self.alpha2,
            capital_per_side=self.capital_per_side,
            initial_capital=self.initial_capital,
            fee_rate=self.fee_rate,
        )


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for the trading strategy.
    
    Contains all standard and advanced performance metrics including:
    - Return metrics (total, annual, CAGR)
    - Risk metrics (volatility, VaR, CVaR, max drawdown)
    - Risk-adjusted metrics (Sharpe, Sortino, Calmar)
    - Trade statistics (win rate, profit factor, etc.)
    """
    
    # Return Metrics
    total_return: float
    annual_return: float
    cagr: float
    
    # Risk Metrics
    annual_volatility: float
    downside_volatility: float
    max_drawdown: float
    max_drawdown_duration_days: float
    value_at_risk_95: float
    cvar_95: float
    
    # Risk-Adjusted Metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    
    # Trade Statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    average_trade_duration_hours: float
    
    # Portfolio Statistics
    total_pnl: float
    total_fees: float
    final_equity: float
    best_trade: float
    worst_trade: float
    
    # Time-based Statistics
    simulation_start: datetime
    simulation_end: datetime
    total_days: float
    trading_days: float


@dataclass
class CycleResult:
    """Results from a single trading cycle."""
    
    cycle_number: int
    formation_start: datetime | None
    trading_start: datetime | None
    trading_end: datetime | None
    pair: tuple[str, str] | None
    betas: tuple[float, float] | None
    copula_name: str | None
    copula_aic: float | None
    trades: list[Trade]
    cycle_pnl: float
    skipped: bool
    skip_reason: str | None


@dataclass
class SimulationResults:
    """Complete simulation results with all metrics and data."""
    
    config: SimulationConfig
    metrics: PerformanceMetrics
    equity_curve: pd.Series
    trades: list[Trade]
    cycle_results: list[CycleResult]
    returns: pd.Series
    drawdown_series: pd.Series
    
    def to_dict(self) -> dict[str, Any]:
        """Convert results to dictionary for JSON export."""
        return {
            "config": asdict(self.config),
            "metrics": asdict(self.metrics),
            "equity_curve": {
                "timestamps": [ts.isoformat() for ts in self.equity_curve.index],
                "values": self.equity_curve.tolist()
            },
            "trades": [asdict(t) for t in self.trades],
            "cycle_results": [asdict(c) for c in self.cycle_results],
        }


class PerformanceSimulator:
    """Main simulator class for running pair trading strategy backtests.
    
    This class handles:
    - Loading and preparing price data
    - Running trading cycles with the copula-based algorithm
    - Calculating comprehensive performance metrics
    - Generating detailed reports and visualizations
    
    Example:
        >>> config = SimulationConfig(initial_capital=100_000)
        >>> simulator = PerformanceSimulator(config)
        >>> results = simulator.run_simulation("data/binance_futures_1h")
        >>> simulator.generate_report(results, "reports/backtest_2024")
    """
    
    def __init__(self, config: SimulationConfig):
        """Initialize the simulator with configuration.
        
        Args:
            config: SimulationConfig object with all parameters
        """
        self.config = config
        self.backtest_config = config.to_backtest_config()
    
    def run_simulation(
        self,
        data_dir: str | Path,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> SimulationResults:
        """Run complete performance simulation on historical data.
        
        Args:
            data_dir: Directory containing price CSV files
            start_date: Optional start date (ISO format, UTC)
            end_date: Optional end date (ISO format, UTC)
            
        Returns:
            SimulationResults object with complete backtest results
            
        Raises:
            ValueError: If data is insufficient or configuration invalid
        """
        print(f"Loading data from {data_dir}...")
        panel = load_closes_from_dir(Path(data_dir), interval=self.config.interval)
        closes = panel.closes
        
        # Filter by date range if specified
        if start_date:
            closes = closes[closes.index >= pd.to_datetime(start_date, utc=True)]
        if end_date:
            closes = closes[closes.index < pd.to_datetime(end_date, utc=True)]
        
        # Validate reference symbol exists
        if self.config.reference_symbol not in closes.columns:
            raise ValueError(f"Reference symbol {self.config.reference_symbol} not found in data")
        
        # Remove columns with too many NaNs
        closes = closes.dropna(axis=1, thresh=int(len(closes) * 0.95))
        
        # Apply log transformation if configured
        if self.config.use_log_prices:
            closes = np.log(closes.astype(float))
        
        # Validate sufficient data
        total_window = self.config.formation_hours + self.config.trading_hours
        if len(closes) < total_window:
            raise ValueError(
                f"Insufficient data: need {total_window} hours, have {len(closes)}"
            )
        
        print(f"Running simulation from {closes.index[0]} to {closes.index[-1]}")
        print(f"Total data points: {len(closes)}, Assets: {len(closes.columns)}")
        
        # Run cycles
        all_trades: list[Trade] = []
        cycle_results: list[CycleResult] = []
        equity_curves: list[pd.Series] = []
        cumulative_pnl = 0.0
        
        i = 0
        cycle_num = 0
        while i + total_window <= len(closes):
            cycle_num += 1
            print(f"Processing cycle {cycle_num}...", end="\r")
            
            trades, meta, equity = run_cycle(closes, i, self.backtest_config)
            
            # Convert per-cycle PnL to cumulative
            if not equity.empty:
                equity = equity + cumulative_pnl
                cumulative_pnl = float(equity.iloc[-1])
            equity_curves.append(equity)
            
            # Create cycle result
            cycle_result = CycleResult(
                cycle_number=cycle_num,
                formation_start=meta.get("formation_start"),
                trading_start=meta.get("trading_start"),
                trading_end=meta.get("trading_end"),
                pair=meta.get("pair"),
                betas=meta.get("betas"),
                copula_name=meta.get("copula"),
                copula_aic=meta.get("copula_aic"),
                trades=trades,
                cycle_pnl=float(equity.iloc[-1] - equity.iloc[0]) if not equity.empty else 0.0,
                skipped=meta.get("skipped", False),
                skip_reason=meta.get("reason"),
            )
            cycle_results.append(cycle_result)
            all_trades.extend(trades)
            
            i += self.config.step_hours
        
        print(f"\nCompleted {cycle_num} cycles, {len(all_trades)} trades")
        
        # Combine equity curves
        equity_pnl = pd.concat(equity_curves).sort_index() if equity_curves else pd.Series(dtype=float)
        
        # Handle duplicates and gaps
        if not equity_pnl.empty:
            equity_pnl = equity_pnl[~equity_pnl.index.duplicated(keep="last")]
            start_ts = equity_pnl.index.min()
            end_ts = equity_pnl.index.max()
            full_index = closes.loc[start_ts:end_ts].index
            equity_pnl = equity_pnl.reindex(full_index).ffill().fillna(0.0)
        
        # Calculate metrics
        print("Calculating performance metrics...")
        metrics = self._calculate_metrics(
            equity_pnl, all_trades, closes.index[0], closes.index[-1]
        )
        
        # Calculate returns and drawdowns
        equity = self.config.initial_capital + equity_pnl
        returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        drawdown_series = self._calculate_drawdown_series(equity)
        
        results = SimulationResults(
            config=self.config,
            metrics=metrics,
            equity_curve=equity,
            trades=all_trades,
            cycle_results=cycle_results,
            returns=returns,
            drawdown_series=drawdown_series,
        )
        
        return results
    
    def _calculate_metrics(
        self,
        equity_pnl: pd.Series,
        trades: list[Trade],
        start_date: datetime,
        end_date: datetime,
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics.
        
        Args:
            equity_pnl: Series of portfolio PnL over time
            trades: List of executed trades
            start_date: Simulation start date
            end_date: Simulation end date
            
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        if equity_pnl.empty or len(trades) == 0:
            return self._empty_metrics(start_date, end_date)
        
        equity = self.config.initial_capital + equity_pnl.astype(float)
        returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        # Time-based calculations
        periods_per_year = 365.25 * 24 if self.config.interval == "1h" else 365.25
        total_days = (end_date - start_date).total_seconds() / 86400
        n_periods = max(1, len(equity) - 1)
        
        # Return metrics
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (periods_per_year / n_periods) - 1.0)
        annual_return = cagr
        
        # Risk metrics
        annual_vol = float(returns.std() * np.sqrt(periods_per_year))
        negative_returns = returns[returns < 0]
        downside_vol = float(negative_returns.std() * np.sqrt(periods_per_year)) if len(negative_returns) > 0 else 0.0
        
        max_dd, max_dd_duration = self._calculate_max_drawdown(equity)
        var_95 = float(np.percentile(returns, 5))
        cvar_95 = float(returns[returns <= var_95].mean()) if (returns <= var_95).any() else 0.0
        
        # Risk-adjusted metrics
        excess_return = annual_return - self.config.risk_free_rate
        sharpe = float(excess_return / annual_vol) if annual_vol > 0 else 0.0
        sortino = float(excess_return / downside_vol) if downside_vol > 0 else 0.0
        calmar = float(annual_return / abs(max_dd)) if max_dd < 0 else 0.0
        omega = self._calculate_omega_ratio(returns)
        
        # Trade statistics
        trade_pnls = np.array([t.pnl for t in trades])
        winning_trades = (trade_pnls > 0).sum()
        losing_trades = (trade_pnls < 0).sum()
        win_rate = float(winning_trades / len(trades)) if trades else 0.0
        
        avg_win = float(trade_pnls[trade_pnls > 0].mean()) if winning_trades > 0 else 0.0
        avg_loss = float(trade_pnls[trade_pnls < 0].mean()) if losing_trades > 0 else 0.0
        
        gross_profit = float(trade_pnls[trade_pnls > 0].sum()) if winning_trades > 0 else 0.0
        gross_loss = float(abs(trade_pnls[trade_pnls < 0].sum())) if losing_trades > 0 else 0.0
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        # Trade durations
        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades]
        avg_duration = float(np.mean(durations)) if durations else 0.0
        
        # Portfolio statistics
        total_pnl = float(equity_pnl.iloc[-1])
        total_fees = float(sum(t.fees for t in trades))
        
        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            cagr=cagr,
            annual_volatility=annual_vol,
            downside_volatility=downside_vol,
            max_drawdown=max_dd,
            max_drawdown_duration_days=max_dd_duration,
            value_at_risk_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            omega_ratio=omega,
            total_trades=len(trades),
            winning_trades=int(winning_trades),
            losing_trades=int(losing_trades),
            win_rate=win_rate,
            average_win=avg_win,
            average_loss=avg_loss,
            profit_factor=profit_factor,
            average_trade_duration_hours=avg_duration,
            total_pnl=total_pnl,
            total_fees=total_fees,
            final_equity=float(equity.iloc[-1]),
            best_trade=float(trade_pnls.max()) if len(trade_pnls) > 0 else 0.0,
            worst_trade=float(trade_pnls.min()) if len(trade_pnls) > 0 else 0.0,
            simulation_start=start_date,
            simulation_end=end_date,
            total_days=total_days,
            trading_days=len(equity_pnl[equity_pnl != 0]) / (24 if self.config.interval == "1h" else 1),
        )
    
    def _empty_metrics(self, start_date: datetime, end_date: datetime) -> PerformanceMetrics:
        """Return empty metrics when no trades occurred."""
        return PerformanceMetrics(
            total_return=0.0, annual_return=0.0, cagr=0.0,
            annual_volatility=0.0, downside_volatility=0.0,
            max_drawdown=0.0, max_drawdown_duration_days=0.0,
            value_at_risk_95=0.0, cvar_95=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0, omega_ratio=0.0,
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, average_win=0.0, average_loss=0.0,
            profit_factor=0.0, average_trade_duration_hours=0.0,
            total_pnl=0.0, total_fees=0.0, final_equity=self.config.initial_capital,
            best_trade=0.0, worst_trade=0.0,
            simulation_start=start_date, simulation_end=end_date,
            total_days=0.0, trading_days=0.0,
        )
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> tuple[float, float]:
        """Calculate maximum drawdown and its duration in days."""
        eq = equity.to_numpy(dtype=float)
        peaks = np.maximum.accumulate(eq)
        drawdowns = (eq - peaks) / peaks
        max_dd = float(np.min(drawdowns))
        
        # Find max drawdown duration
        in_drawdown = drawdowns < 0
        if not in_drawdown.any():
            return max_dd, 0.0
        
        # Find consecutive periods of drawdown
        changes = np.diff(np.concatenate([[False], in_drawdown, [False]]).astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        
        if len(starts) == 0:
            return max_dd, 0.0
        
        durations = ends - starts
        max_duration_periods = int(np.max(durations))
        
        # Convert to days
        hours_per_period = 1 if self.config.interval == "1h" else 24
        max_duration_days = float(max_duration_periods * hours_per_period / 24)
        
        return max_dd, max_duration_days
    
    def _calculate_drawdown_series(self, equity: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        eq = equity.to_numpy(dtype=float)
        peaks = np.maximum.accumulate(eq)
        drawdowns = (eq - peaks) / peaks
        return pd.Series(drawdowns, index=equity.index)
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega ratio (probability-weighted gains vs losses)."""
        excess = returns - threshold
        gains = excess[excess > 0].sum()
        losses = abs(excess[excess < 0].sum())
        return float(gains / losses) if losses > 0 else float('inf')
    
    def generate_report(
        self,
        results: SimulationResults,
        output_dir: str | Path,
        format: Literal["text", "json", "csv", "all"] = "all",
    ) -> None:
        """Generate comprehensive performance report.
        
        Args:
            results: SimulationResults from run_simulation
            output_dir: Directory to save report files
            format: Report format ('text', 'json', 'csv', or 'all')
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format in ["text", "all"]:
            self._generate_text_report(results, output_path / f"report_{timestamp}.txt")
        
        if format in ["json", "all"]:
            self._generate_json_report(results, output_path / f"report_{timestamp}.json")
        
        if format in ["csv", "all"]:
            self._generate_csv_reports(results, output_path, timestamp)
        
        print(f"\nReport generated in {output_path}")
    
    def _generate_text_report(self, results: SimulationResults, filepath: Path) -> None:
        """Generate human-readable text report."""
        m = results.metrics
        
        report = f"""
================================================================================
                     PAIR TRADING PERFORMANCE REPORT
================================================================================

Simulation Period: {m.simulation_start.strftime('%Y-%m-%d')} to {m.simulation_end.strftime('%Y-%m-%d')}
Total Days: {m.total_days:.1f}
Trading Days: {m.trading_days:.1f}

================================================================================
                            CONFIGURATION
================================================================================

Initial Capital:        ${results.config.initial_capital:,.2f}
Reference Asset:        {results.config.reference_symbol}
Time Interval:          {results.config.interval}
Formation Period:       {results.config.formation_hours} hours
Trading Period:         {results.config.trading_hours} hours
Alpha1 (Entry):         {results.config.alpha1}
Alpha2 (Exit):          {results.config.alpha2}
Fee Rate:               {results.config.fee_rate*100:.2f}%
Capital per Side:       ${results.config.capital_per_side:,.2f}

================================================================================
                          RETURN METRICS
================================================================================

Total Return:           {m.total_return*100:>10.2f}%
Annual Return (CAGR):   {m.cagr*100:>10.2f}%
Final Equity:           ${m.final_equity:>15,.2f}
Total P&L:              ${m.total_pnl:>15,.2f}
Total Fees Paid:        ${m.total_fees:>15,.2f}

================================================================================
                           RISK METRICS
================================================================================

Annual Volatility:      {m.annual_volatility*100:>10.2f}%
Downside Volatility:    {m.downside_volatility*100:>10.2f}%
Maximum Drawdown:       {m.max_drawdown*100:>10.2f}%
Max DD Duration:        {m.max_drawdown_duration_days:>10.1f} days
Value at Risk (95%):    {m.value_at_risk_95*100:>10.2f}%
CVaR (95%):             {m.cvar_95*100:>10.2f}%

================================================================================
                      RISK-ADJUSTED METRICS
================================================================================

Sharpe Ratio:           {m.sharpe_ratio:>10.2f}
Sortino Ratio:          {m.sortino_ratio:>10.2f}
Calmar Ratio:           {m.calmar_ratio:>10.2f}
Omega Ratio:            {m.omega_ratio:>10.2f}

================================================================================
                        TRADE STATISTICS
================================================================================

Total Trades:           {m.total_trades:>10,}
Winning Trades:         {m.winning_trades:>10,}
Losing Trades:          {m.losing_trades:>10,}
Win Rate:               {m.win_rate*100:>10.2f}%

Average Win:            ${m.average_win:>15,.2f}
Average Loss:           ${m.average_loss:>15,.2f}
Profit Factor:          {m.profit_factor:>10.2f}

Best Trade:             ${m.best_trade:>15,.2f}
Worst Trade:            ${m.worst_trade:>15,.2f}

Avg Trade Duration:     {m.average_trade_duration_hours:>10.1f} hours

================================================================================
                         CYCLE SUMMARY
================================================================================

Total Cycles:           {len(results.cycle_results)}
Skipped Cycles:         {sum(1 for c in results.cycle_results if c.skipped)}
Active Cycles:          {sum(1 for c in results.cycle_results if not c.skipped)}

Skip Reasons:
"""
        # Add skip reasons
        skip_reasons = {}
        for cycle in results.cycle_results:
            if cycle.skipped and cycle.skip_reason:
                skip_reasons[cycle.skip_reason] = skip_reasons.get(cycle.skip_reason, 0) + 1
        
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            report += f"  {reason:30s}: {count:>5,}\n"
        
        report += "\n" + "="*80 + "\n"
        
        filepath.write_text(report)
        print(f"Text report saved to {filepath}")
    
    def _generate_json_report(self, results: SimulationResults, filepath: Path) -> None:
        """Generate JSON report with all data."""
        data = results.to_dict()
        
        # Convert timestamps to ISO format
        def convert_timestamps(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_timestamps(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_timestamps(item) for item in obj]
            return obj
        
        data = convert_timestamps(data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"JSON report saved to {filepath}")
    
    def _generate_csv_reports(self, results: SimulationResults, output_dir: Path, timestamp: str) -> None:
        """Generate CSV files for trades, equity, and cycles."""
        
        # Trades CSV
        if results.trades:
            trades_df = pd.DataFrame([asdict(t) for t in results.trades])
            trades_path = output_dir / f"trades_{timestamp}.csv"
            trades_df.to_csv(trades_path, index=False)
            print(f"Trades CSV saved to {trades_path}")
        
        # Equity curve CSV
        equity_path = output_dir / f"equity_{timestamp}.csv"
        equity_df = pd.DataFrame({
            'timestamp': results.equity_curve.index,
            'equity': results.equity_curve.values,
            'returns': results.returns.values,
            'drawdown': results.drawdown_series.values,
        })
        equity_df.to_csv(equity_path, index=False)
        print(f"Equity curve CSV saved to {equity_path}")
        
        # Cycles CSV
        cycles_data = []
        for cycle in results.cycle_results:
            cycles_data.append({
                'cycle_number': cycle.cycle_number,
                'formation_start': cycle.formation_start,
                'trading_start': cycle.trading_start,
                'trading_end': cycle.trading_end,
                'pair_asset1': cycle.pair[0] if cycle.pair else None,
                'pair_asset2': cycle.pair[1] if cycle.pair else None,
                'beta1': cycle.betas[0] if cycle.betas else None,
                'beta2': cycle.betas[1] if cycle.betas else None,
                'copula': cycle.copula_name,
                'copula_aic': cycle.copula_aic,
                'num_trades': len(cycle.trades),
                'cycle_pnl': cycle.cycle_pnl,
                'skipped': cycle.skipped,
                'skip_reason': cycle.skip_reason,
            })
        
        cycles_df = pd.DataFrame(cycles_data)
        cycles_path = output_dir / f"cycles_{timestamp}.csv"
        cycles_df.to_csv(cycles_path, index=False)
        print(f"Cycles CSV saved to {cycles_path}")
    
    def print_summary(self, results: SimulationResults) -> None:
        """Print a concise summary of results to console."""
        m = results.metrics
        
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY".center(80))
        print("="*80)
        print(f"Period: {m.simulation_start.strftime('%Y-%m-%d')} to {m.simulation_end.strftime('%Y-%m-%d')}")
        print(f"Initial Capital: ${results.config.initial_capital:,.2f}")
        print(f"Final Equity: ${m.final_equity:,.2f}")
        print(f"Total Return: {m.total_return*100:.2f}%")
        print(f"Annual Return: {m.cagr*100:.2f}%")
        print(f"Sharpe Ratio: {m.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {m.max_drawdown*100:.2f}%")
        print(f"Total Trades: {m.total_trades}")
        print(f"Win Rate: {m.win_rate*100:.2f}%")
        print("="*80 + "\n")


# Convenience function for quick simulations
def quick_simulation(
    data_dir: str | Path,
    initial_capital: float = 100_000,
    alpha1: float = 0.20,
    alpha2: float = 0.10,
    **kwargs
) -> SimulationResults:
    """Run a quick simulation with default settings.
    
    Args:
        data_dir: Directory containing price data
        initial_capital: Starting capital
        alpha1: Entry threshold
        alpha2: Exit threshold
        **kwargs: Additional configuration parameters
        
    Returns:
        SimulationResults object
    """
    config = SimulationConfig(
        initial_capital=initial_capital,
        alpha1=alpha1,
        alpha2=alpha2,
        **kwargs
    )
    
    simulator = PerformanceSimulator(config)
    results = simulator.run_simulation(data_dir)
    simulator.print_summary(results)
    
    return results

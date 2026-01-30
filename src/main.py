"""Reference-Asset-Based Copula Pairs Trading Strategy

This module implements the copula-based pairs trading algorithm proposed in:
Tadi & Witzany (2025): "Copulas in Cryptocurrency Pairs Trading: An Innovative 
Approach to Trading Strategies." Financial Innovation, 11:40.

The algorithm uses BTCUSDT as a reference asset and identifies cointegrated 
cryptocurrency pairs using spread processes (Eq. 31):
    Si = P_reference - β_i * P_i

Key Features:
- Cointegration testing (Engle-Granger, ADF, KSS)
- Marginal distribution fitting (Gaussian, Student-t, Cauchy)
- Multiple copula families (Gaussian, Student-t, Clayton, Gumbel, Frank, Joe, 
  BB1, BB6, BB7, BB8, Tawn Type 1 & 2, with rotations)
- Conditional probability-based trading signals (h-functions)
- Rolling formation and trading periods (21 days formation, 7 days trading)

Usage:
    from src.main import ReferenceAssetCopulaTradingStrategy
    
    strategy = ReferenceAssetCopulaTradingStrategy(
        reference_symbol="BTCUSDT",
        alpha1=0.20,  # Entry threshold
        alpha2=0.10   # Exit threshold
    )
    
    # Run complete backtest
    results = strategy.backtest(
        data_dir="data/binance_futures_1h",
        interval="1h",
        formation_hours=21*24,
        trading_hours=7*24
    )
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest_reference_copula import BacktestConfig, main as run_backtest
from src.copula_model import (
    FittedCopula,
    FittedMarginal,
    fit_best_marginal,
    fit_copula_candidates,
    h_functions_numerical,
)
from src.data_io import load_closes_from_dir
from src.stats_tests import cointegration_with_reference


def kendall_tau(x: pd.Series, y: pd.Series) -> float:
    """
    Calculate Kendall's tau correlation coefficient.
    
    Kendall's tau is a measure of correlation for ranked data:
    τ = (# concordant pairs - # discordant pairs) / (total pairs)
    
    Args:
        x: First data series
        y: Second data series
    
    Returns:
        Kendall's tau correlation coefficient
    """
    from scipy import stats as sp_stats
    a, b = x.align(y, join="inner")
    a_arr = a.to_numpy(dtype=float)
    b_arr = b.to_numpy(dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr)
    if mask.sum() < 100:
        return float("nan")
    result = sp_stats.kendalltau(a_arr[mask], b_arr[mask])
    # Extract tau value (first element of result)
    tau_val = result.statistic if hasattr(result, 'statistic') else result[0]
    return float(tau_val)


class ReferenceAssetCopulaTradingStrategy:
    """
    Implementation of the reference-asset-based copula pairs trading algorithm.
    
    This class implements the complete algorithm from Tadi & Witzany (2025):
    
    1. Formation Period (typically 21 days):
       - Identify assets cointegrated with reference (BTCUSDT)
       - Calculate spread processes: Si = P_ref - β_i * P_i
       - Rank pairs by Kendall's tau correlation
       - Select top 2 correlated pairs
       - Fit marginal distributions to spreads (Gaussian/Student-t/Cauchy)
       - Transform spreads to uniform using PIT (Probability Integral Transform)
       - Fit copula models (elliptical, Archimedean, extreme-value families)
       - Select best copula by AIC
    
    2. Trading Period (typically 7 days):
       - Calculate conditional probabilities h_{1|2} and h_{2|1} (Eq. 4)
       - Generate trading signals based on thresholds α1 (entry) and α2 (exit)
       - Trading Rules (Table 3 & 4):
         * If h_{1|2} < α1 and h_{2|1} > (1-α1): Long β2*P2, Short β1*P1
         * If h_{1|2} > (1-α1) and h_{2|1} < α1: Short β2*P2, Long β1*P1
         * If |h_{1|2} - 0.5| < α2 and |h_{2|1} - 0.5| < α2: Close positions
    
    Parameters:
        reference_symbol: Reference asset symbol (default: "BTCUSDT")
        alpha1: Entry threshold (default: 0.20, paper tests 0.10, 0.15, 0.20)
        alpha2: Exit threshold (default: 0.10)
        eg_alpha: Engle-Granger cointegration p-value threshold (default: 1.00 to disable)
        adf_alpha: ADF p-value threshold for spread stationarity (default: 0.10)
        kss_critical: KSS test critical value at 10% (default: -1.92)
        use_log_prices: Use log prices for cointegration (default: True)
    """
    
    def __init__(
        self,
        reference_symbol: str = "BTCUSDT",
        alpha1: float = 0.20,
        alpha2: float = 0.10,
        eg_alpha: float = 1.00,
        adf_alpha: float = 0.10,
        kss_critical: float = -1.92,
        use_log_prices: bool = True,
    ):
        self.reference_symbol = reference_symbol
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.eg_alpha = eg_alpha
        self.adf_alpha = adf_alpha
        self.kss_critical = kss_critical
        self.use_log_prices = use_log_prices
    
    def calculate_spread(
        self, 
        ref_price: pd.Series, 
        asset_price: pd.Series,
        use_intercept: bool = False
    ) -> tuple[pd.Series, float]:
        """
        Calculate spread process as per Eq. 31 of the paper:
        Si = P_reference - β_i * P_i
        
        Args:
            ref_price: Reference asset price series
            asset_price: Target asset price series
            use_intercept: Whether to include intercept in beta estimation
        
        Returns:
            Tuple of (spread_series, beta_coefficient)
        """
        ref, ast = ref_price.align(asset_price, join="inner")
        
        if use_intercept:
            import statsmodels.api as sm
            X = sm.add_constant(ast.to_numpy(dtype=float))
            model = sm.OLS(ref.to_numpy(dtype=float), X).fit()
            beta = float(model.params[1])
            intercept = float(model.params[0])
            spread = ref - (intercept + beta * ast)
        else:
            # No-intercept estimation (paper's approach, Eq. 31)
            x = ast.to_numpy(dtype=float)
            y = ref.to_numpy(dtype=float)
            denom = float(np.dot(x, x))
            if denom == 0.0:
                return pd.Series(dtype=float), float("nan")
            beta = float(np.dot(x, y) / denom)
            spread = ref - beta * ast
        
        return spread.rename("spread"), beta
    
    def identify_cointegrated_pairs(
        self,
        prices: pd.DataFrame,
    ) -> list[tuple[str, float, float]]:
        """
        Identify assets cointegrated with reference and rank by Kendall's tau.
        
        Args:
            prices: DataFrame with price series (columns are symbols)
        
        Returns:
            List of tuples: (symbol, kendall_tau, beta)
            Sorted by Kendall's tau in descending order
        """
        ref = prices[self.reference_symbol].dropna()
        candidates = [c for c in prices.columns if c != self.reference_symbol]
        
        results = []
        for sym in candidates:
            # Test cointegration
            coint_result = cointegration_with_reference(
                ref,
                prices[sym],
                eg_alpha=self.eg_alpha,
                adf_alpha=self.adf_alpha,
                kss_critical_10pct=self.kss_critical,
                use_intercept=False,
            )
            
            if coint_result is None:
                continue
            
            # Calculate Kendall's tau correlation with reference
            tau = kendall_tau(ref, prices[sym])
            if not np.isfinite(tau):
                continue
            
            results.append((sym, tau, coint_result.beta))
        
        # Sort by Kendall's tau (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def fit_copula_model(
        self,
        spread1: pd.Series,
        spread2: pd.Series,
    ) -> tuple[FittedMarginal, FittedMarginal, FittedCopula]:
        """
        Fit copula model to two spread processes.
        
        Steps:
        1. Fit marginal distributions (Gaussian, Student-t, Cauchy)
        2. Transform to uniform using PIT
        3. Fit copula families and select best by AIC
        
        Args:
            spread1: First spread series
            spread2: Second spread series
        
        Returns:
            Tuple of (fitted_marginal1, fitted_marginal2, best_copula)
        """
        # Fit marginal distributions
        marginal1 = fit_best_marginal(spread1.to_numpy(dtype=float))
        marginal2 = fit_best_marginal(spread2.to_numpy(dtype=float))
        
        # Transform to uniform (PIT)
        u1 = marginal1.cdf(spread1.to_numpy(dtype=float))
        u2 = marginal2.cdf(spread2.to_numpy(dtype=float))
        
        # Stack and remove NaN rows
        u = np.column_stack([u1, u2])
        u = u[np.isfinite(u).all(axis=1)]
        
        if u.shape[0] < 50:
            raise ValueError("Not enough valid samples for copula fitting")
        
        # Fit copulas and select best by AIC
        fitted_copulas = fit_copula_candidates(u)
        
        # Find first copula that supports CDF evaluation
        best = None
        for cand in fitted_copulas:
            try:
                test_u = np.array([[0.5, 0.5]], dtype=float)
                if hasattr(cand.copula, 'cdf'):
                    _ = cand.copula.cdf(test_u)  # type: ignore
                    best = cand
                    break
            except (NotImplementedError, Exception):
                continue
        
        if best is None:
            raise ValueError("No fitted copula supports CDF evaluation")
        
        return marginal1, marginal2, best
    
    def generate_trading_signal(
        self,
        ref_price: float,
        price1: float,
        price2: float,
        beta1: float,
        beta2: float,
        marginal1: FittedMarginal,
        marginal2: FittedMarginal,
        copula: FittedCopula,
    ) -> str:
        """
        Generate trading signal based on conditional probabilities (h-functions).
        
        Implements trading rules from Table 3 & 4 of the paper.
        
        Args:
            ref_price: Current reference asset price
            price1: Current price of asset 1
            price2: Current price of asset 2
            beta1: Beta coefficient for asset 1
            beta2: Beta coefficient for asset 2
            marginal1: Fitted marginal distribution for spread 1
            marginal2: Fitted marginal distribution for spread 2
            copula: Fitted copula object
        
        Returns:
            One of: "LONG_S1_SHORT_S2", "SHORT_S1_LONG_S2", "CLOSE", "WAIT"
        """
        # Calculate current spreads
        s1_t = ref_price - beta1 * price1
        s2_t = ref_price - beta2 * price2
        
        # Transform to uniform
        u1_t = float(marginal1.cdf(s1_t))
        u2_t = float(marginal2.cdf(s2_t))
        
        # Calculate h-functions (conditional probabilities)
        h1_2, h2_1 = h_functions_numerical(copula.copula, u1_t, u2_t)
        
        # Trading rules (Table 3 & 4)
        if h1_2 < self.alpha1 and h2_1 > (1 - self.alpha1):
            # S1 undervalued, S2 overvalued
            # Long β2*P2, Short β1*P1
            return "LONG_S1_SHORT_S2"
        elif h1_2 > (1 - self.alpha1) and h2_1 < self.alpha1:
            # S1 overvalued, S2 undervalued
            # Short β2*P2, Long β1*P1
            return "SHORT_S1_LONG_S2"
        elif abs(h1_2 - 0.5) < self.alpha2 and abs(h2_1 - 0.5) < self.alpha2:
            # Near equilibrium, close positions
            return "CLOSE"
        else:
            # No signal, wait
            return "WAIT"
    
    def backtest(
        self,
        data_dir: str | Path,
        interval: str = "1h",
        formation_hours: int = 21 * 24,
        trading_hours: int = 7 * 24,
        step_hours: int = 7 * 24,
        fee_rate: float = 0.0004,
        capital: float = 20000.0,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> int:
        """
        Run complete backtest using the reference-asset-based copula strategy.
        
        This is a convenience wrapper around the full backtest implementation
        in backtest_reference_copula.py.
        
        Args:
            data_dir: Directory containing price data (*_1h.csv files)
            interval: Data interval (default: "1h")
            formation_hours: Hours for formation period (default: 21*24)
            trading_hours: Hours for trading period (default: 7*24)
            step_hours: Hours to step forward between cycles (default: 7*24)
            fee_rate: Transaction fee rate (default: 0.0004 = 4 bps)
            capital: Initial capital per side (default: 20000)
            start: Optional start date (ISO8601 format)
            end: Optional end date (ISO8601 format)
        
        Returns:
            Exit code (0 for success)
        """
        # Build command-line arguments for backtest
        argv = [
            "--data", str(data_dir),
            "--interval", interval,
            "--formation-hours", str(formation_hours),
            "--trading-hours", str(trading_hours),
            "--step-hours", str(step_hours),
            "--alpha1", str(self.alpha1),
            "--alpha2", str(self.alpha2),
            "--eg-alpha", str(self.eg_alpha),
            "--adf-alpha", str(self.adf_alpha),
            "--kss-critical", str(self.kss_critical),
            "--fee", str(fee_rate),
            "--capital", str(capital),
        ]
        
        if self.use_log_prices:
            argv.append("--log-prices")
        else:
            argv.append("--no-log-prices")
        
        if start:
            argv.extend(["--start", start])
        if end:
            argv.extend(["--end", end])
        
        # Run backtest
        return run_backtest(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Command-line interface for the reference-asset-based copula strategy.
    
    Example usage:
        python -m src.main --data data/binance_futures_1h --alpha1 0.20
    """
    parser = argparse.ArgumentParser(
        description="Reference-Asset-Based Copula Pairs Trading Strategy"
    )
    parser.add_argument("--data", required=True, help="Data directory path")
    parser.add_argument("--interval", default="1h", help="Data interval")
    parser.add_argument("--alpha1", type=float, default=0.20, help="Entry threshold")
    parser.add_argument("--alpha2", type=float, default=0.10, help="Exit threshold")
    parser.add_argument("--formation-hours", type=int, default=21*24, help="Formation period hours")
    parser.add_argument("--trading-hours", type=int, default=7*24, help="Trading period hours")
    parser.add_argument("--fee", type=float, default=0.0004, help="Transaction fee rate")
    parser.add_argument("--capital", type=float, default=20000.0, help="Initial capital")
    parser.add_argument("--start", default=None, help="Start date (ISO8601)")
    parser.add_argument("--end", default=None, help="End date (ISO8601)")
    
    args = parser.parse_args(argv)
    
    # Create strategy instance
    strategy = ReferenceAssetCopulaTradingStrategy(
        alpha1=args.alpha1,
        alpha2=args.alpha2,
    )
    
    # Run backtest
    strategy.backtest(
        data_dir=args.data,
        interval=args.interval,
        formation_hours=args.formation_hours,
        trading_hours=args.trading_hours,
        fee_rate=args.fee,
        capital=args.capital,
        start=args.start,
        end=args.end,
    )
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
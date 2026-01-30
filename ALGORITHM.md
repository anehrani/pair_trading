"""
Algorithm Documentation: Reference-Asset-Based Copula Pairs Trading
===================================================================

This document provides detailed documentation of the algorithm implemented in this repository,
based on Tadi & Witzany (2025): "Copulas in Cryptocurrency Pairs Trading: An Innovative 
Approach to Trading Strategies." Financial Innovation, 11:40.

TABLE OF CONTENTS
-----------------
1. Overview
2. Mathematical Foundation
3. Algorithm Steps
4. Implementation Details
5. Performance Considerations
6. References

1. OVERVIEW
-----------

The reference-asset-based copula approach addresses limitations of traditional pairs trading
methods by:

1. Using stationary spread processes instead of returns
2. Incorporating a reference asset (BTCUSDT) for market-wide information
3. Leveraging copula theory to model complex dependencies
4. Generating signals from conditional probabilities (h-functions)

Traditional Methods vs. Reference-Asset Approach:
- Return-based: Uses log-returns, signals depend only on latest movement
- Level-based: Accumulates mispricing index, may not mean-revert
- Reference-asset: Uses stationary spreads, captures persistent relationships

2. MATHEMATICAL FOUNDATION
--------------------------

2.1 Spread Process (Eq. 31)

For each asset i, the spread with respect to reference asset is:

    Si(t) = P_reference(t) - β_i × P_i(t)

where:
- P_reference(t): Price of reference asset (BTCUSDT) at time t
- P_i(t): Price of asset i at time t
- β_i: Linear regression coefficient (OLS without intercept)

The spread Si is stationary when asset i is cointegrated with the reference.

2.2 Cointegration Tests

Three tests are used to ensure stationarity:

a) Engle-Granger (EG) Test:
   - H0: No cointegration between reference and asset
   - Reject if p-value < α_EG (typically 0.10)
   
b) Augmented Dickey-Fuller (ADF) Test on spread:
   - H0: Unit root in spread process
   - Reject if p-value < α_ADF (typically 0.10)
   
c) Kapetanios-Shin-Snell (KSS) Test:
   - Nonlinear unit root test for ESTAR processes
   - H0: Unit root with nonlinear adjustment
   - Reject if t-statistic < critical value (typically -1.92 at 10%)

2.3 Marginal Distributions

The spread Si follows one of:
- Gaussian: f(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))
- Student-t: f(x) = Γ((ν+1)/2) / (√(νπ)Γ(ν/2)) (1 + x²/ν)^(-(ν+1)/2)
- Cauchy: Special case of Student-t with ν=1

Selection by AIC:
    AIC = 2k - 2ln(L)
    
where k is number of parameters and L is likelihood.

2.4 Probability Integral Transform (PIT)

Transform spread to uniform distribution:

    U_i = F_i(S_i)

where F_i is the CDF of the fitted marginal distribution. This is Sklar's theorem
in action: any joint distribution can be decomposed into marginals and a copula.

2.5 Copula Models

A copula C: [0,1]² → [0,1] models the dependency structure:

    F(s1, s2) = C(F1(s1), F2(s2))

Families implemented:

a) Elliptical Copulas:
   - Gaussian: C(u1,u2;ρ) = Φ_ρ(Φ^(-1)(u1), Φ^(-1)(u2))
   - Student-t: Similar but with t-distribution

b) Archimedean Copulas:
   - Clayton: C(u1,u2;θ) = [max(u1^(-θ) + u2^(-θ) - 1, 0)]^(-1/θ)
   - Gumbel: C(u1,u2;θ) = exp(-[(-ln u1)^θ + (-ln u2)^θ]^(1/θ))
   - Frank, Joe, BB1, BB6, BB7, BB8

c) Extreme Value Copulas:
   - Tawn Type 1 & 2

d) Rotated Copulas (Eq. 16):
   - C90(u1,u2) := C(1-u2, u1)
   - C180(u1,u2) := C(1-u1, 1-u2)
   - C270(u1,u2) := C(u2, 1-u1)

2.6 Conditional Probabilities (h-functions, Eq. 4)

The h-functions are partial derivatives of the copula:

    h_{1|2}(u1, u2) = ∂C(u1, u2)/∂u2
    h_{2|1}(u1, u2) = ∂C(u1, u2)/∂u1

Interpretation:
- h_{1|2}: Conditional probability that U1 ≤ u1 given U2 = u2
- h_{2|1}: Conditional probability that U2 ≤ u2 given U1 = u1

When h_{1|2} > 0.5: Asset 1 is overvalued relative to asset 2
When h_{1|2} < 0.5: Asset 1 is undervalued relative to asset 2

3. ALGORITHM STEPS
------------------

3.1 Formation Period (21 days = 504 hours)

Step 1: Pair Selection
----------------------
For each candidate asset i:
  1. Calculate spread: Si = P_ref - β_i × P_i
  2. Test cointegration (EG + ADF + KSS)
  3. Calculate Kendall's tau with reference:
     
     τ = (# concordant pairs - # discordant pairs) / (total pairs)
  
  4. Rank by τ, select top 2

Step 2: Marginal Fitting
-----------------------
For each selected spread S1, S2:
  1. Fit Gaussian, Student-t, Cauchy distributions
  2. Calculate log-likelihood for each
  3. Select best by AIC
  4. Store fitted parameters

Step 3: PIT Transformation
-------------------------
Transform spreads to uniform:
  U1 = F1(S1)
  U2 = F2(S2)

Step 4: Copula Fitting
---------------------
For (U1, U2) pairs:
  1. Fit all copula families (40+ with rotations)
  2. Calculate AIC for each
  3. Select copula with minimum AIC
  4. Store for trading period

3.2 Trading Period (7 days = 168 hours)

For each hour t:

Step 1: Calculate Current Spreads
--------------------------------
  s1(t) = P_ref(t) - β1 × P1(t)
  s2(t) = P_ref(t) - β2 × P2(t)

Step 2: Transform to Uniform
---------------------------
  u1(t) = F1(s1(t))
  u2(t) = F2(s2(t))

Step 3: Calculate h-functions
----------------------------
  h_{1|2}(t) = ∂C(u1(t), u2(t))/∂u2
  h_{2|1}(t) = ∂C(u1(t), u2(t))/∂u1

Step 4: Generate Trading Signal
------------------------------
Rules (Tables 3 & 4):

IF h_{1|2} < α1 AND h_{2|1} > (1-α1):
    OPEN: Long β2×P2, Short β1×P1
    (S1 undervalued, S2 overvalued)

IF h_{1|2} > (1-α1) AND h_{2|1} < α1:
    OPEN: Short β2×P2, Long β1×P1
    (S1 overvalued, S2 undervalued)

IF |h_{1|2} - 0.5| < α2 AND |h_{2|1} - 0.5| < α2:
    CLOSE: Both positions
    (Near equilibrium)

ELSE:
    WAIT

Step 5: Execute Trade
--------------------
If signal generated:
  - Calculate position sizes
  - Execute at market (taker fee)
  - Record trade details

Step 6: Force Close
------------------
At end of trading period:
  - Close all open positions
  - Record final P&L

3.3 Rolling Forward

Move window forward by step_hours (typically 7 days):
  - New formation period starts
  - May select different pairs
  - Repeat process

4. IMPLEMENTATION DETAILS
-------------------------

4.1 Position Sizing

Given betas β1, β2 and prices P1, P2, choose quantities Q1, Q2 such that:
  - max(β1×P1, β2×P2) × k ≈ capital_per_side
  - This ensures balanced notional exposure

Implementation (src/backtest_reference_copula.py::position_sizes):
  denom = max(|β1×P1|, |β2×P2|)
  k = capital_per_side / denom
  Q1 = k × β1
  Q2 = k × β2

4.2 Numerical h-functions

Since many copulas don't have closed-form h-functions, we use numerical differentiation:

  h_{1|2} ≈ [C(u1, u2+δ) - C(u1, u2-δ)] / (2δ)
  h_{2|1} ≈ [C(u1+δ, u2) - C(u1-δ, u2)] / (2δ)

where δ = 1e-5 (src/copula_model.py::h_functions_numerical)

4.3 Edge Cases

- Not enough cointegrated pairs: Skip trading period
- Copula fitting fails: Skip cycle
- h-function calculation error: Treat as "WAIT"
- Force close at end ensures no positions carry over

4.4 Transaction Costs

Each trade incurs fees on both legs:
  Total Fee = fee_rate × (|Q1×P1| + |Q2×P2|)

Typical: fee_rate = 0.0004 (0.04%, Binance futures taker)

5. PERFORMANCE CONSIDERATIONS
-----------------------------

5.1 Computational Complexity

Per cycle:
- Cointegration tests: O(n × T) where n = #assets, T = formation length
- Marginal fitting: O(3 × T) for 3 distribution candidates
- Copula fitting: O(40 × T × I) where I = iterations for MLE
- Trading signals: O(T_trading) evaluations

Total: ~1-2 minutes per cycle on modern hardware

5.2 Data Requirements

Minimum for reliable results:
- Formation period: ≥ 200 observations (paper uses 504)
- Number of cycles: ≥ 50 for statistical significance (paper uses 104)
- Assets: ≥ 10 for pair selection diversity (paper uses 20)

5.3 Parameter Sensitivity

Based on paper results:

α1 (Entry threshold):
- 0.10: More trades, higher vol, good for 5-min data
- 0.15: Balanced
- 0.20: Fewer trades, lower vol, good for hourly data

α2 (Exit threshold):
- Paper finds minimal impact, uses 0.10 throughout

Data frequency:
- 5-min: Better Sharpe ratios (3.77 vs 0.85)
- Hourly: More stable, fewer transaction costs

6. REFERENCES
-------------

[1] Tadi, M., & Witzany, J. (2025). Copulas in Cryptocurrency Pairs Trading: 
    An Innovative Approach to Trading Strategies. Financial Innovation, 11:40.

[2] Nelsen, R. B. (2007). An Introduction to Copulas. Springer.

[3] Engle, R. F., & Granger, C. W. J. (1987). Co-integration and Error Correction: 
    Representation, Estimation, and Testing. Econometrica, 55(2), 251-276.

[4] Kapetanios, G., Shin, Y., & Snell, A. (2003). Testing for a Unit Root in 
    the Nonlinear STAR Framework. Journal of Econometrics, 112(2), 359-379.

[5] Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges. 
    Publications de l'Institut de Statistique de l'Université de Paris, 8, 229-231.

====================================================================================
For implementation details, see:
- src/main.py: High-level API
- src/backtest_reference_copula.py: Complete backtesting framework
- src/copula_model.py: Copula fitting and h-functions
- src/stats_tests.py: Cointegration tests
====================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


@dataclass(frozen=True)
class CointegrationResult:
    beta: float
    spread: pd.Series
    eg_pvalue: float
    adf_pvalue: float
    kss_stat: float


def kss_estar_tstat(series: np.ndarray, *, max_lags: int = 12) -> float:
    """Kapetanios-Shin-Snell (KSS) nonlinear unit root test (ESTAR) t-statistic.

    We implement the commonly used auxiliary regression:
      Δy_t = δ y_{t-1}^3 + Σ_{i=1..p} φ_i Δy_{t-i} + ε_t

    and return the t-statistic for δ.

    The paper uses an asymptotic critical value of -1.92 at 10%.
    """

    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 200:
        return float("nan")

    # KSS is typically applied to demeaned (and often standardized) series.
    y = y - float(np.mean(y))
    s = float(np.std(y, ddof=1))
    if s > 0:
        y = y / s

    dy = np.diff(y)
    y_lag = y[:-1]
    y_cub = y_lag ** 3

    best_aic = float("inf")
    best_t = float("nan")

    for p in range(0, max_lags + 1):
        # Build design matrix: [y_{t-1}^3, Δy_{t-1},...,Δy_{t-p}]
        # Align so target is Δy_t.
        if p == 0:
            X = y_cub.reshape(-1, 1)
            y_target = dy
        else:
            if dy.size <= p:
                break
            y_target = dy[p:]
            x0 = y_cub[p:]
            lagged = np.column_stack([dy[p - i : -i] for i in range(1, p + 1)])
            X = np.column_stack([x0, lagged])

        # No intercept in the KSS auxiliary regression.
        try:
            res = sm.OLS(y_target, X).fit()
        except Exception:
            continue

        aic = float(res.aic)
        if aic < best_aic:
            best_aic = aic
            # δ is the first coefficient.
            try:
                best_t = float(res.tvalues[0])
            except Exception:
                best_t = float("nan")

    return best_t


def estimate_beta_no_intercept(reference: pd.Series, asset: pd.Series) -> float:
    """Estimate beta in reference ≈ beta * asset (no intercept).

    The paper's spread uses S = Pref - beta * Pi (Eq. 31).
    """
    x = asset.to_numpy(dtype=float)
    y = reference.to_numpy(dtype=float)
    # OLS slope through origin: beta = (x'y)/(x'x)
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def compute_spread(reference: pd.Series, asset: pd.Series, *, use_intercept: bool) -> tuple[pd.Series, float]:
    ref, ast = reference.align(asset, join="inner")
    if use_intercept:
        X = sm.add_constant(ast.to_numpy(dtype=float))
        model = sm.OLS(ref.to_numpy(dtype=float), X).fit()
        beta = float(model.params[1])
        intercept = float(model.params[0])
        spread = ref - (intercept + beta * ast)
        # Note: Eq. (31) has no intercept; this option matches the residual definition.
        return spread.rename("spread"), beta

    beta = estimate_beta_no_intercept(ref, ast)
    spread = (ref - beta * ast).rename("spread")
    return spread, beta


def cointegration_with_reference(
    reference: pd.Series,
    asset: pd.Series,
    *,
    eg_alpha: float = 0.10,
    adf_alpha: float = 0.10,
    kss_critical_10pct: float = -1.92,
    use_intercept: bool = False,
) -> CointegrationResult | None:
    """Returns diagnostics if the asset is cointegrated with reference.

    Practical interpretation (matching the paper's assumptions section):
    - Engle-Granger p-value < 10%
    - ADF p-value on spread < 10%
    - KSS test statistic < -1.92 (10% asymptotic critical value)

    If any condition fails, returns None.
    """

    ref, ast = reference.align(asset, join="inner")
    ref = ref.dropna()
    ast = ast.dropna()
    ref, ast = ref.align(ast, join="inner")
    if len(ref) < 200:
        return None

    spread, beta = compute_spread(ref, ast, use_intercept=use_intercept)
    spread = spread.dropna()
    if len(spread) < 200 or not np.isfinite(beta):
        return None

    # EG test directly on price levels.
    # statsmodels.coint assumes y0 and y1 are I(1); we still use it as the paper does.
    try:
        _stat, pvalue, _crit = coint(ref.to_numpy(dtype=float), ast.to_numpy(dtype=float))
        eg_pvalue = float(pvalue)
    except Exception:
        return None

    try:
        adf_pvalue = float(adfuller(spread.to_numpy(dtype=float), autolag="AIC")[1])
    except Exception:
        return None

    try:
        kss_stat = float(kss_estar_tstat(spread.to_numpy(dtype=float)))
    except Exception:
        return None

    if eg_pvalue >= eg_alpha:
        return None
    if adf_pvalue >= adf_alpha:
        return None
    if not (kss_stat < kss_critical_10pct):
        return None

    return CointegrationResult(
        beta=beta,
        spread=spread,
        eg_pvalue=eg_pvalue,
        adf_pvalue=adf_pvalue,
        kss_stat=kss_stat,
    )

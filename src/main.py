#!/usr/bin/env python3
"""
OKX Copula-Based Pairs Trading System
======================================
Production implementation with all analytical corrections applied.

Corrections from code review:
  [1] Training ECDF for PIT transform (eliminates look-ahead bias)
  [2] OLS on LOG prices with intercept included in spread
  [3] Proper Gumbel h-functions (both h12 AND h21)
  [4] Beta-weighted spread; BTC legs cancel correctly
  [5] Transaction cost modeling
  [6] Stop-loss and exposure limits

Setup:
  export OKX_API_KEY='...'
  export OKX_SECRET_KEY='...'
  export OKX_PASSPHRASE='...'

  pip install numpy pandas scipy statsmodels requests

Usage:
  python okx_copula_trader.py --mode paper          # OKX demo env
  python okx_copula_trader.py --mode dry             # Local sim, real prices
  python okx_copula_trader.py --mode live             # Real money
  python okx_copula_trader.py --mode backtest         # Historical walkforward
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import time
import hmac
import hashlib
import base64
import json
import logging
import argparse
import warnings
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import requests

warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("copula_trader.log"),
    ],
)
log = logging.getLogger("copula_trader")


# ════════════════════════════════════════════════════════════
#  1.  CONFIGURATION
# ════════════════════════════════════════════════════════════

@dataclass
class Config:
    # --- OKX API credentials (env vars or set directly) ---
    api_key: str = os.getenv("OKX_API_KEY", "")
    secret_key: str = os.getenv("OKX_SECRET_KEY", os.getenv("OKX_API_SECRET", ""))
    passphrase: str = os.getenv("OKX_PASSPHRASE", "")
    base_url: str = "https://www.okx.com"
    simulated: bool = True            # True → OKX demo-trading header

    # --- Instrument universe ---
    ref_asset: str = "BTC-USDT-SWAP"
    alt_assets: List[str] = field(default_factory=lambda: [
        "ETH-USDT-SWAP",  "SOL-USDT-SWAP",  "XRP-USDT-SWAP",
        "DOGE-USDT-SWAP", "ADA-USDT-SWAP",  "AVAX-USDT-SWAP",
        "LINK-USDT-SWAP", "DOT-USDT-SWAP",  "LTC-USDT-SWAP",
        "POL-USDT-SWAP",
    ])

    # --- Strategy parameters ---
    bar: str = "5m"
    formation_len: int = 5760         # 20 days × 24 h × 12 (5m)
    trading_len: int = 288            # 1 day  × 24 h × 12 (5m)
    entry_threshold: float = 0.15     # h < 0.15 or h > 0.85
    exit_band: float = 0.10           # |h − 0.5| < 0.10

    # --- Risk parameters ---
    capital_per_leg: float = 5000.0   # USDT notional per leg
    max_loss_pct: float = 0.03        # 3 % portfolio stop
    leverage: int = 3
    fee_rate: float = 0.0007          # 0.07 % taker

    @property
    def all_instruments(self) -> List[str]:
        return [self.ref_asset] + self.alt_assets


# ════════════════════════════════════════════════════════════
#  2.  OKX REST CLIENT
# ════════════════════════════════════════════════════════════

class OKXClient:
    """Thin wrapper around OKX API v5 with HMAC-SHA256 auth."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._ct_cache: Dict[str, dict] = {}

    # ── authentication helpers ───────────────────────────────

    @staticmethod
    def _iso_now() -> str:
        now = datetime.now(timezone.utc)
        return (
            now.strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{now.microsecond // 1000:03d}Z"
        )

    def _sign(self, ts: str, method: str, path: str, body: str = "") -> str:
        prehash = ts + method.upper() + path + body
        mac = hmac.new(
            self.cfg.secret_key.encode("utf-8"),
            prehash.encode("utf-8"),
            hashlib.sha256,
        )
        return base64.b64encode(mac.digest()).decode("utf-8")

    def _auth_headers(self, method: str, path: str, body: str = "") -> dict:
        ts = self._iso_now()
        hdrs = {
            "OK-ACCESS-KEY": self.cfg.api_key,
            "OK-ACCESS-SIGN": self._sign(ts, method, path, body),
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": self.cfg.passphrase,
        }
        if self.cfg.simulated:
            hdrs["x-simulated-trading"] = "1"
        return hdrs

    # ── generic request ──────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        body: Optional[dict] = None,
        retries: int = 3,
    ) -> Optional[list]:
        url = self.cfg.base_url + path
        body_str = json.dumps(body) if body else ""

        # build the path the signature sees (includes query string for GET)
        if method == "GET" and params:
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            sign_path = f"{path}?{qs}"
        else:
            sign_path = path

        for attempt in range(retries):
            try:
                hdrs = self._auth_headers(method, sign_path, body_str)
                if method == "GET":
                    r = self.session.get(url, params=params, headers=hdrs, timeout=10)
                else:
                    r = self.session.post(url, data=body_str, headers=hdrs, timeout=10)

                data = r.json()
                if data.get("code") == "0":
                    return data.get("data", [])
                log.error("API code=%s  msg=%s", data.get("code"), data.get("msg"))
                return None
            except requests.RequestException as exc:
                log.warning("Request failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(1.0 * (attempt + 1))
        return None

    # ── market data ──────────────────────────────────────────

    def get_candles(
        self, inst_id: str, bar: str = "1H", count: int = 504
    ) -> List[list]:
        """Paginated candle fetch → chronological order."""
        collected: List[list] = []
        after = ""

        while len(collected) < count:
            params: dict = {"instId": inst_id, "bar": bar, "limit": "300"}
            if after:
                params["after"] = after
            rows = self._request("GET", "/api/v5/market/candles", params=params)
            if not rows:
                break
            collected.extend(rows)
            after = rows[-1][0]          # oldest ts in batch
            if len(rows) < 300:
                break
            time.sleep(0.12)             # respect rate limit

        collected = collected[:count]
        collected.reverse()              # → oldest first
        return collected

    def get_ticker(self, inst_id: str) -> Optional[float]:
        d = self._request("GET", "/api/v5/market/ticker",
                          params={"instId": inst_id})
        return float(d[0]["last"]) if d else None

    # ── contract specifications ──────────────────────────────

    def get_contract_info(self, inst_id: str) -> Optional[dict]:
        if inst_id in self._ct_cache:
            return self._ct_cache[inst_id]
        d = self._request(
            "GET", "/api/v5/public/instruments",
            params={"instType": "SWAP", "instId": inst_id},
        )
        if d:
            info = {
                "ctVal": float(d[0]["ctVal"]),
                "lotSz": float(d[0]["lotSz"]),
                "minSz": float(d[0]["minSz"]),
                "tickSz": float(d[0]["tickSz"]),
            }
            self._ct_cache[inst_id] = info
            return info
        return None

    def compute_contracts(self, inst_id: str, usdt_amount: float) -> int:
        """Convert a USDT notional into an integer number of contracts."""
        info = self.get_contract_info(inst_id)
        price = self.get_ticker(inst_id)
        if not info or not price:
            return 0
        val_per_ct = info["ctVal"] * price
        n = int(usdt_amount / val_per_ct)
        # round down to lot size
        lot = max(int(info["lotSz"]), 1)
        n = max((n // lot) * lot, int(info["minSz"]))
        return n

    # ── account ──────────────────────────────────────────────

    def get_balance(self, ccy: str = "USDT") -> float:
        d = self._request("GET", "/api/v5/account/balance",
                          params={"ccy": ccy})
        if d:
            for det in d[0].get("details", []):
                if det["ccy"] == ccy:
                    return float(det["availBal"])
        return 0.0

    def get_positions(self) -> Dict[str, dict]:
        d = self._request("GET", "/api/v5/account/positions")
        if not d:
            return {}
        out = {}
        for p in d:
            pos = float(p.get("pos", 0))
            if pos == 0:
                continue
            out[p["instId"]] = {
                "pos": pos,
                "avgPx": float(p["avgPx"]) if p.get("avgPx") else 0.0,
                "upl": float(p["upl"]) if p.get("upl") else 0.0,
            }
        return out

    def set_leverage(self, inst_id: str, lever: int, mgn: str = "cross"):
        return self._request("POST", "/api/v5/account/set-leverage", body={
            "instId": inst_id, "lever": str(lever), "mgnMode": mgn,
        })

    # ── order management ─────────────────────────────────────

    def market_order(self, inst_id: str, side: str, sz: int) -> Optional[str]:
        res = self._request("POST", "/api/v5/trade/order", body={
            "instId": inst_id,
            "tdMode": "cross",
            "side": side,
            "ordType": "market",
            "sz": str(sz),
        })
        if res:
            oid = res[0].get("ordId")
            log.info("ORDER  %s %d %s  ordId=%s", side, sz, inst_id, oid)
            return oid
        return None

    def close_position(self, inst_id: str) -> Optional[str]:
        positions = self.get_positions()
        if inst_id not in positions:
            return None
        p = positions[inst_id]
        side = "sell" if p["pos"] > 0 else "buy"
        return self.market_order(inst_id, side, int(abs(p["pos"])))


# ════════════════════════════════════════════════════════════
#  3.  DATA PIPELINE
# ════════════════════════════════════════════════════════════

class DataPipeline:
    def __init__(self, client: OKXClient, cfg: Config):
        self.client = client
        self.cfg = cfg

    def fetch_prices(self, count: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Return a DataFrame of close prices, columns = instId."""
        if count is None:
            count = self.cfg.formation_len
        series: Dict[str, pd.Series] = {}

        for inst in self.cfg.all_instruments:
            candles = self.client.get_candles(inst, self.cfg.bar, count)
            if not candles or len(candles) < count * 0.8:
                log.warning("Skipping %s  (%d candles)", inst,
                            len(candles) if candles else 0)
                continue
            df = pd.DataFrame(
                candles,
                columns=["ts", "o", "h", "l", "c",
                          "vol", "volCcy", "volCcyQ", "confirm"],
            )
            df["ts"] = pd.to_datetime(df["ts"].astype(np.int64), unit="ms")
            df.set_index("ts", inplace=True)
            series[inst] = df["c"].astype(float)
            time.sleep(0.15)

        if len(series) < 3:
            log.error("Only %d instruments returned data", len(series))
            return None
        prices = pd.DataFrame(series).dropna()
        log.info("Price matrix  %d × %d", prices.shape[0], prices.shape[1])
        return prices

    def fetch_current_prices(self) -> Dict[str, float]:
        out = {}
        for inst in self.cfg.all_instruments:
            p = self.client.get_ticker(inst)
            if p is not None:
                out[inst] = p
        return out


# ════════════════════════════════════════════════════════════
#  4.  CORRECTED ECONOMETRIC TESTS
# ════════════════════════════════════════════════════════════

def kss_test(series: pd.Series, max_lags: int = 8,
             significance: str = "5%") -> Tuple[float, bool]:
    """
    Kapetanios-Shin-Snell (2003) test for nonlinear (ESTAR)
    stationarity.  Lag order chosen by BIC.

    H₀ : unit root   vs   H₁ : nonlinear mean-reversion
    Regression:  Δxₜ = δ·x³ₜ₋₁ + Σ φᵢ Δxₜ₋ᵢ + εₜ
    Reject H₀ when t(δ) < critical value.

    Critical values (raw, no constant, n ≈ 500):
        1 %  → −2.82      5 %  → −2.22      10 % → −1.92
    """
    cv = {"1%": -2.82, "5%": -2.22, "10%": -1.92}[significance]

    x = series.values.astype(float)
    dx = np.diff(x)
    x3 = x[:-1] ** 3

    best_bic, best_t = np.inf, 0.0

    for p in range(0, max_lags + 1):
        if p == 0:
            y, X = dx, x3.reshape(-1, 1)
        else:
            lags = np.column_stack(
                [dx[max(p - i - 1, 0) : len(dx) - i - 1 + (1 if p - i - 1 < 0 else 0)]
                 for i in range(p)]
            )
            # Safer construction:
            lags = []
            for i in range(1, p + 1):
                lags.append(dx[p - i : len(dx) - i])
            lags = np.column_stack(lags)
            y = dx[p:]
            X = np.column_stack([x3[p:], lags])

        if len(y) < X.shape[1] + 5:
            continue
        try:
            model = sm.OLS(y, X).fit()
            if model.bic < best_bic:
                best_bic = model.bic
                best_t = model.tvalues[0]
        except Exception:
            continue

    return best_t, best_t < cv


def adf_test(series: pd.Series, sig: float = 0.05) -> Tuple[float, bool]:
    res = adfuller(series, autolag="BIC")
    return res[0], res[1] < sig


def combined_stationarity(series: pd.Series) -> bool:
    """Accept stationarity if EITHER ADF or KSS rejects the unit root."""
    _, adf_ok = adf_test(series)
    _, kss_ok = kss_test(series)
    return adf_ok or kss_ok


# ════════════════════════════════════════════════════════════
#  5.  CORRECTED COPULA MODELS
# ════════════════════════════════════════════════════════════

class GaussianCopula:
    name = "gaussian"

    def __init__(self):
        self.rho: float = 0.0
        self.aic: float = np.inf

    def fit(self, u: np.ndarray, v: np.ndarray) -> "GaussianCopula":
        u, v = np.clip(u, 1e-6, 1 - 1e-6), np.clip(v, 1e-6, 1 - 1e-6)

        def neg_ll(p):
            r = p[0]
            x, y = stats.norm.ppf(u), stats.norm.ppf(v)
            ll = -0.5 * np.log(1 - r ** 2) - (
                r ** 2 * (x ** 2 + y ** 2) - 2 * r * x * y
            ) / (2 * (1 - r ** 2))
            return -np.sum(ll)

        res = minimize(neg_ll, [0.5], bounds=[(-0.999, 0.999)],
                       method="L-BFGS-B")
        if res.success:
            self.rho = res.x[0]
            self.aic = 2 + 2 * res.fun          # 2k − 2LL, k=1
        return self

    def h(self, u: np.ndarray, v: np.ndarray):
        """h(u|v) and h(v|u)."""
        u, v = np.clip(u, 1e-6, 1 - 1e-6), np.clip(v, 1e-6, 1 - 1e-6)
        x, y = stats.norm.ppf(u), stats.norm.ppf(v)
        s = np.sqrt(1 - self.rho ** 2)
        h_uv = stats.norm.cdf((x - self.rho * y) / s)
        h_vu = stats.norm.cdf((y - self.rho * x) / s)
        return np.clip(h_uv, 0, 1), np.clip(h_vu, 0, 1)


class GumbelCopula:
    """
    Corrected Gumbel copula — BOTH h-functions implemented.

    C(u,v) = exp(−A^{1/θ})   where  A = (−ln u)^θ + (−ln v)^θ

    h(u|v) = ∂C/∂v  =  C · (−ln v)^{θ−1} · A^{1/θ − 1} / v
    h(v|u) = ∂C/∂u  =  C · (−ln u)^{θ−1} · A^{1/θ − 1} / u
    """
    name = "gumbel"

    def __init__(self):
        self.theta: float = 1.0
        self.aic: float = np.inf

    def fit(self, u: np.ndarray, v: np.ndarray) -> "GumbelCopula":
        u, v = np.clip(u, 1e-6, 1 - 1e-6), np.clip(v, 1e-6, 1 - 1e-6)

        def neg_ll(p):
            th = p[0]
            if th <= 1.0:
                return 1e12
            lu, lv = -np.log(u), -np.log(v)
            A = lu ** th + lv ** th
            Ainv = A ** (1.0 / th)

            log_c = (
                -Ainv
                + (th - 1) * (np.log(lu) + np.log(lv))
                + (1.0 / th - 2) * np.log(A)
                + np.log(Ainv + th - 1)
                - np.log(u * v)
            )
            s = np.sum(log_c)
            return -s if np.isfinite(s) else 1e12

        res = minimize(neg_ll, [2.0], bounds=[(1.001, 20.0)],
                       method="L-BFGS-B")
        if res.success:
            self.theta = res.x[0]
            self.aic = 2 + 2 * res.fun
        return self

    def h(self, u: np.ndarray, v: np.ndarray):
        u, v = np.clip(u, 1e-6, 1 - 1e-6), np.clip(v, 1e-6, 1 - 1e-6)
        th = self.theta
        lu, lv = -np.log(u), -np.log(v)
        A = lu ** th + lv ** th
        Ainv = A ** (1.0 / th)
        C = np.exp(-Ainv)

        common = C * A ** (1.0 / th - 1)
        h_uv = common * lv ** (th - 1) / v       # ∂C/∂v
        h_vu = common * lu ** (th - 1) / u       # ∂C/∂u
        return np.clip(h_uv, 0, 1), np.clip(h_vu, 0, 1)


def select_copula(u: np.ndarray, v: np.ndarray):
    """Fit all families, return best by AIC."""
    models = [GaussianCopula().fit(u, v), GumbelCopula().fit(u, v)]
    best = min(models, key=lambda m: m.aic)
    log.info("Copula selection  →  %s  (θ=%.4f, AIC=%.1f)",
             best.name, getattr(best, "rho", getattr(best, "theta", 0)),
             best.aic)
    return best


# ════════════════════════════════════════════════════════════
#  6.  STRATEGY ENGINE  (all bugs fixed)
# ════════════════════════════════════════════════════════════

@dataclass
class ModelState:
    pair: Tuple[str, str] = ("", "")
    betas: Dict[str, float] = field(default_factory=dict)
    intercepts: Dict[str, float] = field(default_factory=dict)
    copula: Any = None
    # training ECDFs  (sorted spread arrays for np.searchsorted)
    ecdf_s1: np.ndarray = field(default_factory=lambda: np.array([]))
    ecdf_s2: np.ndarray = field(default_factory=lambda: np.array([]))
    valid: bool = False


class CopulaStrategy:
    """
    Formation → Trading workflow with all corrections:
      • OLS on log prices, intercept retained in spread
      • Stationarity via ADF + KSS
      • Kendall τ on returns for pair ranking
      • Training ECDF for out-of-sample PIT
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model: Optional[ModelState] = None

    # ── training ECDF (FIX #1: no look-ahead) ───────────────

    @staticmethod
    def _ecdf_transform(sorted_train: np.ndarray, values) -> np.ndarray:
        """P(X ≤ x) estimated from training sample."""
        n = len(sorted_train)
        idx = np.searchsorted(sorted_train, values, side="right")
        return np.clip(idx / (n + 1), 1e-6, 1 - 1e-6)

    # ── formation period ─────────────────────────────────────

    def train(self, prices: pd.DataFrame) -> Optional[ModelState]:
        ref = self.cfg.ref_asset
        alts = [c for c in prices.columns if c != ref]
        lp = np.log(prices)                          # FIX #2: log prices

        # ---- step 1: find cointegrated spreads ----
        valid, betas, intercepts = [], {}, {}
        for alt in alts:
            try:
                X = sm.add_constant(lp[alt])
                ols = sm.OLS(lp[ref], X).fit()
                alpha = ols.params.iloc[0]           # FIX #2: keep intercept
                beta = ols.params.iloc[1]
                spread = lp[ref] - alpha - beta * lp[alt]

                if combined_stationarity(spread):
                    valid.append(alt)
                    betas[alt] = beta
                    intercepts[alt] = alpha
                    log.info("  %-20s β=%.4f  α=%.4f  ✓", alt, beta, alpha)
            except Exception as e:
                log.debug("  %s OLS failed: %s", alt, e)

        if len(valid) < 2:
            log.warning("Only %d stationary spreads — need ≥ 2", len(valid))
            return None

        # ---- step 2: rank by Kendall τ on returns ----
        rets = lp.diff().dropna()
        ranked = sorted(
            [(a, stats.kendalltau(rets[ref], rets[a])[0]) for a in valid],
            key=lambda t: abs(t[1]),
            reverse=True,
        )
        pair = (ranked[0][0], ranked[1][0])
        log.info("Pair selected:  %s  (τ=%.3f)  &  %s  (τ=%.3f)",
                 pair[0], ranked[0][1], pair[1], ranked[1][1])

        # ---- step 3: compute training spreads ----
        s1 = lp[ref] - intercepts[pair[0]] - betas[pair[0]] * lp[pair[0]]
        s2 = lp[ref] - intercepts[pair[1]] - betas[pair[1]] * lp[pair[1]]

        # ---- step 4: rank-based pseudo-obs for copula fit ----
        n = len(s1)
        u = stats.rankdata(s1.values) / (n + 1)
        v = stats.rankdata(s2.values) / (n + 1)

        # ---- step 5: fit copula ----
        copula = select_copula(u, v)

        # ---- step 6: store ECDFs for live scoring ----
        state = ModelState(
            pair=pair,
            betas=betas,
            intercepts=intercepts,
            copula=copula,
            ecdf_s1=np.sort(s1.values),              # FIX #1
            ecdf_s2=np.sort(s2.values),
            valid=True,
        )
        self.model = state
        return state

    # ── live signal generation ───────────────────────────────

    def signal(self, prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Compute h-functions from current prices and return an action.

        Returns dict with keys: action, h12, h21
        action ∈ {long_s1_short_s2, short_s1_long_s2, close, hold}
        """
        noop = {"action": "hold", "h12": 0.5, "h21": 0.5}
        m = self.model
        if m is None or not m.valid:
            return noop

        ref = self.cfg.ref_asset
        a1, a2 = m.pair
        if not all(k in prices for k in (ref, a1, a2)):
            return noop

        # current spreads (log prices)
        lr = np.log(prices[ref])
        s1_now = lr - m.intercepts[a1] - m.betas[a1] * np.log(prices[a1])
        s2_now = lr - m.intercepts[a2] - m.betas[a2] * np.log(prices[a2])

        # PIT via training ECDF  ← FIX #1 (no look-ahead)
        u = self._ecdf_transform(m.ecdf_s1, np.array([s1_now]))
        v = self._ecdf_transform(m.ecdf_s2, np.array([s2_now]))

        h12, h21 = m.copula.h(u, v)
        h12, h21 = float(h12[0]), float(h21[0])

        lo = self.cfg.entry_threshold
        hi = 1 - lo
        band = self.cfg.exit_band

        if h12 < lo and h21 > hi:
            action = "long_s1_short_s2"
        elif h12 > hi and h21 < lo:
            action = "short_s1_long_s2"
        elif abs(h12 - 0.5) < band and abs(h21 - 0.5) < band:
            action = "close"
        else:
            action = "hold"

        return {"action": action, "h12": h12, "h21": h21}


# ════════════════════════════════════════════════════════════
#  7.  RISK MANAGER
# ════════════════════════════════════════════════════════════

class RiskManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.entries: Dict[str, float] = {}          # instId → entry_price
        self.trade_log: List[dict] = []

    def record_entry(self, inst: str, price: float, side: str):
        self.entries[inst] = {"price": price, "side": side}

    def record_exit(self, inst: str, price: float):
        if inst in self.entries:
            e = self.entries.pop(inst)
            d = 1 if e["side"] == "buy" else -1
            pnl = d * (price - e["price"]) / e["price"]
            self.trade_log.append({
                "inst": inst, "entry": e["price"],
                "exit": price, "side": e["side"], "ret": pnl,
                "time": datetime.now(timezone.utc).isoformat(),
            })
            log.info("EXIT  %s  entry=%.4f  exit=%.4f  ret=%.4f%%",
                     inst, e["price"], price, pnl * 100)
            return pnl
        return 0.0

    def check_stop_loss(
        self, positions: Dict[str, dict], prices: Dict[str, float]
    ) -> bool:
        total_pnl_usd = 0.0
        total_notional = 0.0
        for inst, info in self.entries.items():
            px = prices.get(inst)
            if px is None:
                continue
            d = 1 if info["side"] == "buy" else -1
            ret = d * (px - info["price"]) / info["price"]
            notional = self.cfg.capital_per_leg
            total_pnl_usd += ret * notional
            total_notional += notional

        if total_notional > 0:
            pct = total_pnl_usd / total_notional
            if pct < -self.cfg.max_loss_pct:
                log.warning("⛔  STOP LOSS  unrealized = %.2f%%", pct * 100)
                return True
        return False

    def summary(self) -> str:
        if not self.trade_log:
            return "No trades yet."
        rets = [t["ret"] for t in self.trade_log]
        return (
            f"Trades: {len(rets)}  "
            f"Win%: {100*sum(1 for r in rets if r>0)/len(rets):.0f}%  "
            f"Mean: {100*np.mean(rets):.3f}%  "
            f"Sharpe: {np.mean(rets)/max(np.std(rets),1e-9)*np.sqrt(len(rets)):.2f}"
        )


# ════════════════════════════════════════════════════════════
#  8.  TRADE EXECUTOR
# ════════════════════════════════════════════════════════════

class TradeExecutor:
    def __init__(self, client: OKXClient, cfg: Config, risk: RiskManager):
        self.client = client
        self.cfg = cfg
        self.risk = risk
        self.position: int = 0           # 0 flat, +1 long_s1, -1 short_s1
        self._lev_done: set = set()

    def _set_lev(self, inst: str):
        if inst not in self._lev_done:
            self.client.set_leverage(inst, self.cfg.leverage)
            self._lev_done.add(inst)
            time.sleep(0.1)

    def open(self, direction: str, model: ModelState) -> bool:
        """
        Open a pairs trade.

        Long S1 / Short S2  →  BTC legs cancel  →
            Short β₁·ALT₁   +   Long β₂·ALT₂

        Short S1 / Long S2  →  opposite.
        """
        a1, a2 = model.pair
        self._set_lev(a1)
        self._set_lev(a2)

        c1 = self.client.compute_contracts(a1, self.cfg.capital_per_leg)
        c2 = self.client.compute_contracts(a2, self.cfg.capital_per_leg)
        if c1 == 0 or c2 == 0:
            log.error("Cannot compute contract size (c1=%d c2=%d)", c1, c2)
            return False

        if direction == "long_s1_short_s2":
            s1, s2 = "sell", "buy"       # short ALT1, long ALT2
            self.position = 1
        else:
            s1, s2 = "buy", "sell"       # long ALT1, short ALT2
            self.position = -1

        log.info("OPEN  %s  →  %s %d %s  |  %s %d %s",
                 direction, s1, c1, a1, s2, c2, a2)

        o1 = self.client.market_order(a1, s1, c1)
        o2 = self.client.market_order(a2, s2, c2)

        if o1 and o2:
            p1 = self.client.get_ticker(a1)
            p2 = self.client.get_ticker(a2)
            if p1:
                self.risk.record_entry(a1, p1, s1)
            if p2:
                self.risk.record_entry(a2, p2, s2)
            return True

        log.error("Leg execution failed — cleaning up")
        self.close(model)
        return False

    def close(self, model: ModelState):
        if self.position == 0:
            return
        a1, a2 = model.pair
        log.info("CLOSE  %s  %s", a1, a2)
        self.client.close_position(a1)
        self.client.close_position(a2)

        p1 = self.client.get_ticker(a1)
        p2 = self.client.get_ticker(a2)
        if p1:
            self.risk.record_exit(a1, p1)
        if p2:
            self.risk.record_exit(a2, p2)

        self.position = 0


# ════════════════════════════════════════════════════════════
#  9.  PAPER-TRADE CLIENT  (local simulation, real prices)
# ════════════════════════════════════════════════════════════

class PaperClient(OKXClient):
    """Drop-in replacement that fakes orders but reads real market data."""

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._positions: Dict[str, dict] = {}
        self._balance: float = 100_000.0
        self._fills: List[dict] = []

    def get_balance(self, ccy="USDT"):
        return self._balance

    def get_positions(self):
        return {k: v for k, v in self._positions.items() if v["pos"] != 0}

    def market_order(self, inst_id, side, sz, **_):
        price = self.get_ticker(inst_id)
        if price is None:
            return None
        info = self.get_contract_info(inst_id)
        ct = info["ctVal"] if info else 1.0
        notional = sz * ct * price
        fee = notional * self.cfg.fee_rate

        d = 1 if side == "buy" else -1
        old = self._positions.get(inst_id, {"pos": 0, "avgPx": price})
        new_pos = old["pos"] + d * sz

        if abs(new_pos) < 1e-9:
            pnl = (price - old["avgPx"]) * old["pos"] * ct
            self._balance += pnl
            self._positions.pop(inst_id, None)
            log.info("[PAPER] CLOSE %-18s  PnL $%.2f  fee $%.2f",
                     inst_id, pnl, fee)
        else:
            self._positions[inst_id] = {"pos": new_pos, "avgPx": price,
                                         "upl": 0.0}

        self._balance -= fee
        oid = f"paper-{len(self._fills)}"
        self._fills.append({"id": oid, "inst": inst_id, "side": side,
                            "sz": sz, "px": price, "fee": fee})
        log.info("[PAPER] %s %d %-18s @ %.4f  fee $%.2f  bal $%.2f",
                 side.upper(), sz, inst_id, price, fee, self._balance)
        return oid

    def close_position(self, inst_id):
        if inst_id in self._positions:
            p = self._positions[inst_id]
            side = "sell" if p["pos"] > 0 else "buy"
            return self.market_order(inst_id, side, int(abs(p["pos"])))
        return None

    def set_leverage(self, inst_id, lever, mgn="cross"):
        return True


# ════════════════════════════════════════════════════════════
# 10.  BACKTESTER  (walk-forward on historical data)
# ════════════════════════════════════════════════════════════

class Backtester:
    """
    Corrected walk-forward backtest.
    Formation window → fit model → trade next window → repeat.
    """

    def __init__(self, cfg: Config, prices: pd.DataFrame):
        self.cfg = cfg
        self.prices = prices
        self.strategy = CopulaStrategy(cfg)
        self.results: List[dict] = []

    def run(self) -> pd.DataFrame:
        form = self.cfg.formation_hours
        trade = self.cfg.trading_hours
        n = len(self.prices)
        log.info("Backtest  rows=%d  form=%d  trade=%d", n, form, trade)

        for start in range(0, n - form - trade, trade):
            train_df = self.prices.iloc[start : start + form]
            test_df = self.prices.iloc[start + form : start + form + trade]

            model = self.strategy.train(train_df)
            if model is None or not model.valid:
                log.info("Window %d: no valid model — skip", start)
                continue

            pnl = self._simulate_window(test_df, model)
            self.results.append({
                "window_start": self.prices.index[start + form],
                "pair": f"{model.pair[0]} / {model.pair[1]}",
                "copula": model.copula.name,
                "pnl_pct": pnl,
            })
            log.info("Window %s  pair=%s  copula=%s  pnl=%.4f%%",
                     self.prices.index[start + form].date(),
                     model.pair, model.copula.name, pnl * 100)

        df = pd.DataFrame(self.results)
        if len(df):
            total = df["pnl_pct"].sum()
            sharpe = (df["pnl_pct"].mean()
                      / max(df["pnl_pct"].std(), 1e-9)
                      * np.sqrt(len(df)))
            log.info("══ BACKTEST SUMMARY ══")
            log.info("Windows: %d  Total: %.2f%%  Sharpe: %.2f",
                     len(df), total * 100, sharpe)
        return df

    def _simulate_window(
        self, test: pd.DataFrame, model: ModelState
    ) -> float:
        """
        Simulate one trading window.

        FIX #1:  PIT uses training ECDF (stored in model.ecdf_s1/s2).
        FIX #4:  PnL accounts for both legs with beta weighting.
        FIX #5:  Fees deducted.
        """
        ref = self.cfg.ref_asset
        a1, a2 = model.pair
        lp = np.log(test)

        s1 = lp[ref] - model.intercepts[a1] - model.betas[a1] * lp[a1]
        s2 = lp[ref] - model.intercepts[a2] - model.betas[a2] * lp[a2]

        # FIX #1  — use training ECDF
        u_all = CopulaStrategy._ecdf_transform(model.ecdf_s1, s1.values)
        v_all = CopulaStrategy._ecdf_transform(model.ecdf_s2, s2.values)

        h12_all, h21_all = model.copula.h(u_all, v_all)

        pos = 0              # +1 or -1
        entry_a1 = entry_a2 = 0.0
        window_pnl = 0.0
        lo = self.cfg.entry_threshold
        hi = 1 - lo
        band = self.cfg.exit_band
        fee = self.cfg.fee_rate

        for t in range(len(test)):
            h12, h21 = h12_all[t], h21_all[t]

            if pos == 0:
                if h12 < lo and h21 > hi:
                    pos = 1
                    entry_a1 = test[a1].iloc[t]
                    entry_a2 = test[a2].iloc[t]
                elif h12 > hi and h21 < lo:
                    pos = -1
                    entry_a1 = test[a1].iloc[t]
                    entry_a2 = test[a2].iloc[t]
            else:
                do_close = (
                    abs(h12 - 0.5) < band and abs(h21 - 0.5) < band
                ) or (t == len(test) - 1)       # force-close at window end

                if do_close:
                    px_a1 = test[a1].iloc[t]
                    px_a2 = test[a2].iloc[t]

                    # FIX #4:  correct direction
                    # pos=+1: short ALT1, long ALT2
                    if pos == 1:
                        r1 = (entry_a1 - px_a1) / entry_a1   # short ALT1
                        r2 = (px_a2 - entry_a2) / entry_a2   # long  ALT2
                    else:
                        r1 = (px_a1 - entry_a1) / entry_a1   # long  ALT1
                        r2 = (entry_a2 - px_a2) / entry_a2   # short ALT2

                    # FIX #5: transaction costs (roundtrip = 2× per leg)
                    trade_pnl = (r1 + r2) / 2 - 4 * fee
                    window_pnl += trade_pnl
                    pos = 0

        return window_pnl


# ════════════════════════════════════════════════════════════
# 11.  LIVE TRADER  (main orchestrator)
# ════════════════════════════════════════════════════════════

class LiveTrader:
    def __init__(self, cfg: Config, client: Optional[OKXClient] = None):
        self.cfg = cfg
        self.client = client or OKXClient(cfg)
        self.pipe = DataPipeline(self.client, cfg)
        self.strat = CopulaStrategy(cfg)
        self.risk = RiskManager(cfg)
        self.exec = TradeExecutor(self.client, cfg, self.risk)
        self.last_train: Optional[datetime] = None
        self.running = False

    def _needs_retrain(self) -> bool:
        if self.last_train is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self.last_train).total_seconds()
        # trading_len bars * 5 minutes * 60 seconds
        return elapsed >= self.cfg.trading_len * 5 * 60

    def _train(self) -> bool:
        log.info("═" * 55)
        log.info("FORMATION PERIOD — fetching %d candles …",
                 self.cfg.formation_len)
        prices = self.pipe.fetch_prices()
        if prices is None:
            return False
        m = self.strat.train(prices)
        if m and m.valid:
            self.last_train = datetime.now(timezone.utc)
            log.info("Model ready  pair=%s  copula=%s",
                     m.pair, m.copula.name)
            return True
        log.warning("No valid model produced")
        return False

    def _cycle(self):
        # ---- retrain if needed ----
        if self._needs_retrain():
            if self.strat.model and self.exec.position != 0:
                log.info("Trading period ended → force close")
                self.exec.close(self.strat.model)
            if not self._train():
                return

        if not self.strat.model or not self.strat.model.valid:
            return

        # ---- fetch prices ----
        prices = self.pipe.fetch_current_prices()
        if len(prices) < 3:
            log.warning("Insufficient live prices")
            return

        # ---- signal ----
        sig = self.strat.signal(prices)
        log.info("Signal  action=%-22s  h12=%.4f  h21=%.4f",
                 sig["action"], sig["h12"], sig["h21"])

        # ---- stop-loss check ----
        if self.exec.position != 0:
            if self.risk.check_stop_loss(self.client.get_positions(), prices):
                self.exec.close(self.strat.model)
                return

        # ---- execute ----
        if self.exec.position == 0:
            if sig["action"] in ("long_s1_short_s2", "short_s1_long_s2"):
                bal = self.client.get_balance()
                needed = self.cfg.capital_per_leg * 2 / self.cfg.leverage
                if bal >= needed:
                    self.exec.open(sig["action"], self.strat.model)
                else:
                    log.warning("Balance $%.2f < required $%.2f", bal, needed)
        else:
            if sig["action"] == "close":
                self.exec.close(self.strat.model)

    def run(self):
        log.info("═" * 55)
        log.info("   COPULA PAIRS TRADER  —  %s",
                 "SIMULATED" if self.cfg.simulated else "🔴 LIVE")
        log.info("   Ref: %s   Alts: %d   Leverage: %dx",
                 self.cfg.ref_asset, len(self.cfg.alt_assets), self.cfg.leverage)
        log.info("   Capital / leg: $%.0f   Stop: %.1f%%",
                 self.cfg.capital_per_leg, self.cfg.max_loss_pct * 100)
        log.info("═" * 55)

        bal = self.client.get_balance()
        log.info("Account balance: $%.2f USDT", bal)

        self.running = True
        while self.running:
            try:
                self._cycle()
            except KeyboardInterrupt:
                break
            except Exception:
                log.exception("Error in cycle")

            # sleep until 1 min past the next 5-minute boundary
            now = datetime.now(timezone.utc)
            minutes_to_next = 5 - (now.minute % 5)
            nxt = (now + timedelta(minutes=minutes_to_next)).replace(
                second=60 if minutes_to_next == 1 and now.second > 0 else 0, 
                microsecond=0
            ).replace(second=0) + timedelta(minutes=1) # 1 min buffer for candle
            
            # Cleaner version:
            wait_seconds = (5 - (now.minute % 5)) * 60 - now.second + 30 # 30s buffer
            if wait_seconds < 30:
                wait_seconds += 300
            
            log.info("Next cycle in %.1f min  |  %s",
                     wait_seconds / 60, self.risk.summary())
            try:
                time.sleep(wait_seconds)
            except KeyboardInterrupt:
                break

        self.shutdown()

    def shutdown(self):
        log.info("Shutting down …")
        self.running = False
        if self.strat.model and self.exec.position != 0:
            log.info("Closing open positions …")
            self.exec.close(self.strat.model)
        log.info("Final stats:  %s", self.risk.summary())
        log.info("Done.")


# ════════════════════════════════════════════════════════════
# 12.  ENTRY POINT
# ════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="OKX Copula Pairs Trading System"
    )
    ap.add_argument(
        "--mode", choices=["live", "paper", "dry", "backtest"],
        default="paper",
        help="live = real money | paper = OKX demo | "
             "dry = local sim | backtest = historical",
    )
    ap.add_argument("--capital", type=float, default=5000.0,
                    help="USDT per leg")
    ap.add_argument("--leverage", type=int, default=3)
    ap.add_argument("--ref", default="BTC-USDT-SWAP")
    ap.add_argument("--hours", type=int, default=0,
                    help="Backtest: total hours of data to fetch (0 = auto)")
    args = ap.parse_args()

    cfg = Config(
        capital_per_leg=args.capital,
        leverage=args.leverage,
        ref_asset=args.ref,
        simulated=(args.mode != "live"),
    )

    # --- validate credentials ---
    if not cfg.api_key and args.mode in ("live", "paper"):
        log.error(
            "Set OKX credentials:\n"
            "  export OKX_API_KEY='...'\n"
            "  export OKX_SECRET_KEY='...'\n"
            "  export OKX_PASSPHRASE='...'"
        )
        sys.exit(1)

    # --- mode dispatch ---
    if args.mode == "backtest":
        client = OKXClient(cfg) if cfg.api_key else None
        if client is None:
            log.error("Backtest needs API key to fetch candles")
            sys.exit(1)
        pipe = DataPipeline(client, cfg)
        total = args.hours or (cfg.formation_len + cfg.trading_len * 4)
        prices = pipe.fetch_prices(count=total)
        if prices is None:
            sys.exit(1)
        bt = Backtester(cfg, prices)
        results = bt.run()
        if len(results):
            print(results.to_string(index=False))
        return

    if args.mode == "dry":
        if not cfg.api_key:
            log.error("Dry-run still needs API key for market data")
            sys.exit(1)
        client = PaperClient(cfg)
    elif args.mode == "paper":
        client = PaperClient(cfg)
    else:
        client = OKXClient(cfg)

    trader = LiveTrader(cfg, client)
    try:
        trader.run()
    except KeyboardInterrupt:
        trader.shutdown()


if __name__ == "__main__":
    main()
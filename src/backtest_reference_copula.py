from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.copula_model import (
    fit_best_marginal,
    fit_copula_candidates,
    h_functions_numerical,
)
from src.data_io import load_closes_from_dir
from src.stats_tests import cointegration_with_reference


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    symbol_long: str
    symbol_short: str
    qty_long: float
    qty_short: float
    entry_price_long: float
    entry_price_short: float
    exit_price_long: float
    exit_price_short: float
    fees: float
    pnl: float


@dataclass
class BacktestConfig:
    reference_symbol: str = "SPY"  # "SPY" for stocks, "BTCUSDT" for crypto
    interval: str = "1h"
    formation_days: float = 21.0
    trading_days: float = 7.0
    step_days: float = 7.0
    # EG can be too restrictive in short rolling windows; allow disabling by setting to 1.0.
    eg_alpha: float = 1.00
    adf_alpha: float = 0.10
    kss_critical_10pct: float = -1.92
    use_intercept_beta: bool = False
    alpha1: float = 0.20
    alpha2: float = 0.10
    capital_per_side: float = 20_000.0
    initial_capital: float = 20_000.0
    fee_rate: float = 0.0004  # 4 bps per leg per trade, adjustable


def kendall_tau(x: pd.Series, y: pd.Series) -> float:
    a, b = x.align(y, join="inner")
    a = a.to_numpy(dtype=float)
    b = b.to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 100:
        return float("nan")
    tau, _p = stats.kendalltau(a[mask], b[mask])
    return float(tau) if tau is not None else float("nan")


def pick_pair(
    window_prices: pd.DataFrame,
    cfg: BacktestConfig,
) -> tuple[str, str, dict[str, float]] | None:
    """Select two altcoins for the cycle.

    Paper logic (implementation methodology section):
    - Identify assets cointegrated with reference using EG + KSS (and ADF on spread)
    - Rank by Kendall tau with reference
    - Pick top 2

    Returns (sym1, sym2, betas) where betas map alt symbol -> beta.
    """

    ref = window_prices[cfg.reference_symbol].dropna()
    candidates = [c for c in window_prices.columns if c != cfg.reference_symbol]

    stats_rows: list[tuple[str, float, float]] = []  # (symbol, tau_with_ref, beta)
    betas: dict[str, float] = {}

    for sym in candidates:
        res = cointegration_with_reference(
            ref,
            window_prices[sym],
            eg_alpha=cfg.eg_alpha,
            adf_alpha=cfg.adf_alpha,
            kss_critical_10pct=cfg.kss_critical_10pct,
            use_intercept=cfg.use_intercept_beta,
        )
        if res is None:
            continue

        tau = kendall_tau(ref, window_prices[sym])
        if not np.isfinite(tau):
            continue

        stats_rows.append((sym, tau, res.beta))
        betas[sym] = res.beta

    if len(stats_rows) < 2:
        return None

    stats_rows.sort(key=lambda r: r[1], reverse=True)
    s1, _tau1, _b1 = stats_rows[0]
    s2, _tau2, _b2 = stats_rows[1]
    return s1, s2, betas


def position_sizes(beta1: float, beta2: float, p1: float, p2: float, capital_per_side: float) -> tuple[float, float]:
    """Compute quantities so each leg's notional is <= capital_per_side.

    Implements the paper's Table 4 logic up to a scaling factor k:
      beta2 * P2 - beta1 * P1

    Choose k so max(beta1*P1, beta2*P2) * k ≈ capital_per_side.

    Returns (q1, q2) where q1 is quantity in asset1, q2 in asset2.
    """
    denom = max(abs(beta1 * p1), abs(beta2 * p2))
    if denom <= 0:
        return 0.0, 0.0
    k = capital_per_side / denom
    q1 = k * beta1
    q2 = k * beta2
    return float(q1), float(q2)


def run_cycle(prices: pd.DataFrame, start_ts: pd.Timestamp, cfg: BacktestConfig) -> tuple[list[Trade], dict, pd.Series]:
    formation_end = start_ts + pd.Timedelta(days=cfg.formation_days)
    trading_end = formation_end + pd.Timedelta(days=cfg.trading_days)

    formation = prices.loc[start_ts:formation_end].dropna(how="any")
    # Exclude the shared boundary bar to prevent overlap
    trading = prices.loc[formation_end:trading_end]
    if not trading.empty and trading.index[0] <= formation_end:
        trading = trading.iloc[1:]
    trading = trading.dropna(how="any")

    if formation.empty or trading.empty:
        return [], {"skipped": True, "reason": "missing_data"}, pd.Series(dtype=float)

    picked = pick_pair(formation, cfg)
    if picked is None:
        # Equity curve is flat at 0 over the trading window.
        return [], {"skipped": True, "reason": "no_cointegrated_pair"}, pd.Series(0.0, index=trading.index)

    sym1, sym2, betas = picked
    beta1 = float(betas[sym1])
    beta2 = float(betas[sym2])

    ref_f = formation[cfg.reference_symbol]
    p1_f = formation[sym1]
    p2_f = formation[sym2]

    s1 = ref_f - beta1 * p1_f
    s2 = ref_f - beta2 * p2_f

    try:
        m1 = fit_best_marginal(s1.to_numpy(dtype=float))
        m2 = fit_best_marginal(s2.to_numpy(dtype=float))

        u1 = m1.cdf(s1.to_numpy(dtype=float))
        u2 = m2.cdf(s2.to_numpy(dtype=float))
        u = np.column_stack([u1, u2])
        u = u[np.isfinite(u).all(axis=1)]
        if u.shape[0] < 50:
            raise ValueError("Not enough valid PIT samples to fit copula")

        fitted = fit_copula_candidates(u)

        best = None
        for cand in fitted:
            try:
                _ = cand.copula.cdf(np.array([[0.5, 0.5]], dtype=float))
                best = cand
                break
            except NotImplementedError:
                continue
            except Exception:
                continue
        if best is None:
            raise ValueError("No fitted copula supported CDF evaluation")
    except Exception as e:
        return (
            [],
            {
                "skipped": True,
                "reason": "marginal_or_copula_fit_failed",
                "pair": (sym1, sym2),
                "error": str(e),
            },
            pd.Series(0.0, index=trading.index),
        )

    trades: list[Trade] = []

    pos = None  # (long_sym, short_sym, q_long, q_short, entry_prices, entry_time, fees_paid)
    realized = 0.0
    # equity_series is PnL (not including initial capital) per hour during trading
    equity_points: list[tuple[pd.Timestamp, float]] = []

    for t, row in trading.iterrows():
        pref = float(row[cfg.reference_symbol])
        p1 = float(row[sym1])
        p2 = float(row[sym2])

        s1_t = pref - beta1 * p1
        s2_t = pref - beta2 * p2

        u1_t = float(m1.cdf(s1_t))
        u2_t = float(m2.cdf(s2_t))

        h1_2, h2_1 = h_functions_numerical(best.copula, u1_t, u2_t)

        open_long_s1_short_s2 = (h1_2 < cfg.alpha1) and (h2_1 > (1 - cfg.alpha1))
        open_short_s1_long_s2 = (h1_2 > (1 - cfg.alpha1)) and (h2_1 < cfg.alpha1)
        close_signal = (abs(h1_2 - 0.5) < cfg.alpha2) and (abs(h2_1 - 0.5) < cfg.alpha2)

        if pos is None:
            if open_long_s1_short_s2:
                # Table 4: long beta2*P2 and short beta1*P1
                q1, q2 = position_sizes(beta1, beta2, p1, p2, cfg.capital_per_side)
                qty_long = q2
                qty_short = -q1
                long_sym = sym2
                short_sym = sym1
                notional = abs(qty_long * p2) + abs(qty_short * p1)
                fees = cfg.fee_rate * notional
                # side=1: long_s1_short_s2 (entered with h1_2 low, h2_1 high)
                pos = (long_sym, short_sym, qty_long, qty_short, p2, p1, t, fees, 1)
                realized -= fees
            elif open_short_s1_long_s2:
                # Table 4: short beta2*P2 and long beta1*P1
                q1, q2 = position_sizes(beta1, beta2, p1, p2, cfg.capital_per_side)
                qty_long = q1
                qty_short = -q2
                long_sym = sym1
                short_sym = sym2
                notional = abs(qty_long * p1) + abs(qty_short * p2)
                fees = cfg.fee_rate * notional
                # side=-1: short_s1_long_s2 (entered with h1_2 high, h2_1 low)
                pos = (long_sym, short_sym, qty_long, qty_short, p1, p2, t, fees, -1)
                realized -= fees
            else:
                equity_points.append((t, realized))
                continue
        else:
            long_sym, short_sym, qty_long, qty_short, entry_p_long, entry_p_short, entry_t, fees_paid, side = pos

            # Handle price 'snaps' by checking if we've reached or overshot the exit target
            if side == 1:
                # Entered when h1_2 < alpha1 and h2_1 > 1-alpha1
                # Exit when they revert towards 0.5 (or snap past it)
                close_signal = (h1_2 >= 0.5 - cfg.alpha2) and (h2_1 <= 0.5 + cfg.alpha2)
            else:
                # Entered when h1_2 > 1-alpha1 and h2_1 < alpha1
                close_signal = (h1_2 <= 0.5 + cfg.alpha2) and (h2_1 >= 0.5 - cfg.alpha2)

            if close_signal:
                # Close at current prices
                px_long = float(row[long_sym])
                px_short = float(row[short_sym])
                notional = abs(qty_long * px_long) + abs(qty_short * px_short)
                fees = fees_paid + cfg.fee_rate * notional

                pnl_long = qty_long * (px_long - entry_p_long)
                pnl_short = qty_short * (px_short - entry_p_short)
                pnl = pnl_long + pnl_short - fees

                # Realize PnL and remove any previously accounted entry fee
                realized += pnl

                trades.append(
                    Trade(
                        entry_time=entry_time,
                        exit_time=t,
                        symbol_long=long_sym,
                        symbol_short=short_sym,
                        qty_long=qty_long,
                        qty_short=qty_short,
                        entry_price_long=entry_price_long,
                        entry_price_short=entry_price_short,
                        exit_price_long=px_long,
                        exit_price_short=px_short,
                        fees=fees,
                        pnl=pnl,
                    )
                )
                pos = None
                equity_points.append((t, realized))
            else:
                # Mark-to-market
                px_long = float(row[long_sym])
                px_short = float(row[short_sym])
                unreal = qty_long * (px_long - entry_price_long) + qty_short * (px_short - entry_price_short)
                equity_points.append((t, realized + float(unreal)))

    # Force-close at end of trading window, per paper.
    if pos is not None:
        long_sym, short_sym, qty_long, qty_short, entry_p_long, entry_p_short, entry_t, fees_paid, side = pos
        last_t = trading.index[-1]
        last_row = trading.iloc[-1]
        px_long = float(last_row[long_sym])
        px_short = float(last_row[short_sym])
        notional = abs(qty_long * px_long) + abs(qty_short * px_short)
        fees = fees_paid + cfg.fee_rate * notional

        pnl_long = qty_long * (px_long - entry_p_long)
        pnl_short = qty_short * (px_short - entry_p_short)
        pnl = pnl_long + pnl_short - fees

        realized += pnl

        trades.append(
            Trade(
                entry_time=entry_t,
                exit_time=last_t,
                symbol_long=long_sym,
                symbol_short=short_sym,
                qty_long=qty_long,
                qty_short=qty_short,
                entry_price_long=entry_p_long,
                entry_price_short=entry_p_short,
                exit_price_long=px_long,
                exit_price_short=px_short,
                fees=fees,
                pnl=pnl,
            )
        )

        equity_points.append((last_t, realized))

    meta = {
        "skipped": False,
        "pair": (sym1, sym2),
        "betas": (beta1, beta2),
        "copula": best.name,
        "copula_aic": best.aic,
        "formation_start": formation.index[0],
        "trading_start": trading.index[0],
        "trading_end": trading.index[-1],
    }

    equity = pd.Series({t: v for t, v in equity_points}).sort_index()
    # Ensure we cover the whole trading window (flat-fill if needed)
    equity = equity.reindex(trading.index).ffill().fillna(0.0)

    return trades, meta, equity


def performance_summary(trades: list[Trade], equity_pnl: pd.Series, cfg: BacktestConfig) -> dict:
    """Compute paper-style metrics from an hourly equity curve."""

    trades_n = int(len(trades))
    pnl_total = float(equity_pnl.iloc[-1]) if not equity_pnl.empty else 0.0

    if equity_pnl.empty:
        return {
            "trades": trades_n,
            "pnl": pnl_total,
            "total_return": float("nan"),
            "annual_return": float("nan"),
            "annual_vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "win_rate": float("nan"),
        }

    equity = cfg.initial_capital + equity_pnl.astype(float)
    rets = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    # Annualization: use actual time duration to handle stock market gaps correctly
    n_days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400.0
    years = n_days / 365.25

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    if years > 0:
        annual_return = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)
        # Estimate period frequency (bars per year) from the data itself
        est_periods_per_year = len(equity) / years
        mean_r = float(rets.mean()) if len(rets) else 0.0
        vol_r = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
        annual_vol = float(vol_r * np.sqrt(est_periods_per_year)) if vol_r > 0 else float("nan")
        sharpe = float((annual_return) / annual_vol) if (annual_vol > 0 and not np.isnan(annual_vol)) else float("nan")
    else:
        annual_return = 0.0
        annual_vol = float("nan")
        sharpe = float("nan")

    # Max drawdown on equity
    eq = equity.to_numpy(dtype=float)
    peaks = np.maximum.accumulate(eq)
    drawdowns = (eq - peaks) / peaks
    max_dd = float(np.min(drawdowns))

    pnls = np.array([t.pnl for t in trades], dtype=float) if trades else np.array([])
    win_rate = float((pnls > 0).mean()) if pnls.size else float("nan")

    return {
        "trades": trades_n,
        "pnl": pnl_total,
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reference-asset-based copula pairs backtest")
    p.add_argument("--data", required=True, help="Directory with *_1h.csv files")
    p.add_argument("--interval", default="1h")
    p.add_argument("--formation-days", type=float, default=21.0)
    p.add_argument("--trading-days", type=float, default=7.0)
    p.add_argument("--step-days", type=float, default=7.0)
    p.add_argument("--alpha1", type=float, default=0.20)
    p.add_argument("--alpha2", type=float, default=0.10)
    p.add_argument("--eg-alpha", type=float, default=1.00, help="Engle–Granger p-value threshold (set 1.0 to disable)")
    p.add_argument("--adf-alpha", type=float, default=0.10, help="ADF p-value threshold on spread")
    p.add_argument("--kss-critical", type=float, default=-1.92, help="KSS 10%% critical value (paper: -1.92)")
    p.add_argument("--fee", type=float, default=0.0004)
    p.add_argument("--capital", type=float, default=20000.0)
    p.add_argument(
        "--log-prices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use log prices for cointegration/spreads (default: enabled)",
    )
    p.add_argument("--start", default=None, help="Optional start timestamp (UTC, ISO8601)")
    p.add_argument("--end", default=None, help="Optional end timestamp (UTC, ISO8601)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cfg = BacktestConfig(
        interval=args.interval,
        formation_days=args.formation_days,
        trading_days=args.trading_days,
        step_days=args.step_days,
        alpha1=args.alpha1,
        alpha2=args.alpha2,
        eg_alpha=args.eg_alpha,
        adf_alpha=args.adf_alpha,
        kss_critical_10pct=args.kss_critical,
        fee_rate=args.fee,
        capital_per_side=args.capital,
    )

    panel = load_closes_from_dir(Path(args.data), interval=cfg.interval)
    closes = panel.closes

    if args.start:
        closes = closes[closes.index >= pd.to_datetime(args.start, utc=True)]
    if args.end:
        closes = closes[closes.index < pd.to_datetime(args.end, utc=True)]

    required = {cfg.reference_symbol}
    if not required.issubset(set(closes.columns)):
        raise ValueError(f"Missing reference symbol {cfg.reference_symbol} in data")

    # Remove columns with too many NaNs early.
    closes = closes.dropna(axis=1, thresh=int(len(closes) * 0.95))

    if args.log_prices:
        # Cointegration/spread modeling is typically done in log-price space.
        closes = np.log(closes.astype(float))

    total_duration = cfg.formation_days + cfg.trading_days
    if (closes.index[-1] - closes.index[0]) < pd.Timedelta(days=total_duration):
        raise ValueError("Not enough data for a single formation+trading window")

    all_trades: list[Trade] = []
    cycle_metas: list[dict] = []
    equity_curves: list[pd.Series] = []

    cumulative_pnl = 0.0

    curr_ts = closes.index[0]
    while curr_ts + pd.Timedelta(days=cfg.formation_days + cfg.trading_days) <= closes.index[-1]:
        trades, meta, equity = run_cycle(closes, curr_ts, cfg)
        cycle_metas.append(meta)
        all_trades.extend(trades)

        # Convert per-cycle PnL to cumulative PnL across cycles.
        if not equity.empty:
            equity = equity + cumulative_pnl
            cumulative_pnl = float(equity.iloc[-1])
        equity_curves.append(equity)
        curr_ts += pd.Timedelta(days=cfg.step_days)

    equity_pnl = pd.concat(equity_curves).sort_index()
    # Trading windows are non-overlapping; if there are duplicates, keep last.
    equity_pnl = equity_pnl[~equity_pnl.index.duplicated(keep="last")]

    # Include formation gaps as flat equity for calendar-time metrics.
    if not equity_pnl.empty:
        start_ts = equity_pnl.index.min()
        end_ts = equity_pnl.index.max()
        full_index = closes.loc[start_ts:end_ts].index
        equity_pnl = equity_pnl.reindex(full_index).ffill().fillna(0.0)

    summary = performance_summary(all_trades, equity_pnl, cfg)
    print("Summary:", summary)
    print("Cycles:", len(cycle_metas), "Trades:", len(all_trades))

    skipped_reasons: dict[str, int] = {}
    skipped_errors: dict[str, int] = {}
    for m in cycle_metas:
        if not m.get("skipped"):
            continue
        r = str(m.get("reason", "unknown"))
        skipped_reasons[r] = skipped_reasons.get(r, 0) + 1
        if r == "marginal_or_copula_fit_failed":
            err = str(m.get("error", ""))
            if err:
                skipped_errors[err] = skipped_errors.get(err, 0) + 1
    if skipped_reasons:
        print("Skipped cycles by reason:", dict(sorted(skipped_reasons.items(), key=lambda kv: (-kv[1], kv[0]))))
    if skipped_errors:
        top = sorted(skipped_errors.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        print("Top fit-failure errors:", top)

    # Write trades CSV for inspection
    out = Path("data") / f"trades_{cfg.interval}_a1_{cfg.alpha1:.2f}_a2_{cfg.alpha2:.2f}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    tdf = pd.DataFrame([t.__dict__ for t in all_trades])
    if not tdf.empty:
        tdf.to_csv(out, index=False)
        print("Wrote", out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

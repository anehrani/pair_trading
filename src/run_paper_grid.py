"""Run a small experiment grid similar to the paper.

The paper evaluates alpha1 in {10%, 15%, 20%} with alpha2 fixed at 10%.

Example:
  /path/to/.venv/bin/python -m src.run_paper_grid --data data/binance_futures_1h --interval 1h
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import pandas as pd

from src.backtest_reference_copula import BacktestConfig, main as run_backtest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run paper-style alpha grid")
    p.add_argument("--data", required=True)
    p.add_argument("--interval", default="1h")
    p.add_argument("--fee", type=float, default=0.0004)
    p.add_argument("--capital", type=float, default=20000.0)
    p.add_argument(
        "--log-prices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use log prices for cointegration/spreads (default: enabled)",
    )
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    return p.parse_args(argv)


def run_one(args: argparse.Namespace, alpha1: float) -> dict:
    argv = [
        "--data",
        args.data,
        "--interval",
        args.interval,
        "--alpha1",
        str(alpha1),
        "--alpha2",
        "0.10",
        "--fee",
        str(args.fee),
        "--capital",
        str(args.capital),
    ]
    if args.log_prices:
        argv += ["--log-prices"]
    else:
        argv += ["--no-log-prices"]
    if args.start:
        argv += ["--start", args.start]
    if args.end:
        argv += ["--end", args.end]

    # backtest_reference_copula prints its own summary; we want a programmatic one.
    # For now, just run it and let it write outputs; metrics will be printed.
    run_backtest(argv)
    return {"alpha1": alpha1}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    for a1 in (0.10, 0.15, 0.20):
        print("\n" + "=" * 20 + f" alpha1={a1:.2f} " + "=" * 20)
        run_one(args, a1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Download Binance USDⓈ-M futures kline (candlestick) data.

This repo's strategy (per Tadi & Witzany, Financial Innovation 2025) uses:
- A reference asset (BTCUSDT) price series
- Candidate altcoin price series
- Hourly closes for formation/trading cycles (the paper also mentions 5-min data)

This script downloads PUBLIC data from Binance Futures API (no API key required).

Example:
  /path/to/.venv/bin/python -m get_data.download_binance_futures \
    --interval 1h \
    --start 2021-01-01 \
    --end 2023-01-19 \
    --out data/binance_futures_1h

  /path/to/.venv/bin/python -m get_data.download_binance_futures \
    --interval 5m \
    --start 2021-01-01 \
    --end 2023-01-19 \
    --out data/binance_futures_5m

Notes:
- Endpoint: https://fapi.binance.com/fapi/v1/klines
- Limits: max 1500 candles per request; this script paginates.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests


BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"
KLINES_PATH = "/fapi/v1/klines"

# Symbols used in the paper (Table 1). BTCUSDT is the reference.
DEFAULT_SYMBOLS: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "BCHUSDT",
    "XRPUSDT",
    "EOSUSDT",
    "LTCUSDT",
    "TRXUSDT",
    "ETCUSDT",
    "LINKUSDT",
    "XLMUSDT",
    "ADAUSDT",
    "XMRUSDT",
    "DASHUSDT",
    "ZECUSDT",
    "XTZUSDT",
    "ATOMUSDT",
    "BNBUSDT",
    "ONTUSDT",
    "IOTAUSDT",
    "BATUSDT",
)


@dataclass(frozen=True)
class Kline:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time_ms: int


def _parse_utc_date(date_str: str) -> datetime:
    # Accept YYYY-MM-DD
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.replace(tzinfo=timezone.utc)


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def fetch_klines(
    *,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    session: requests.Session,
    limit: int = 1500,
    pause_s: float = 0.2,
) -> list[Kline]:
    """Fetch klines for a single symbol and interval between [start_ms, end_ms)."""

    out: list[Kline] = []
    next_start = start_ms

    while next_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": next_start,
            "endTime": end_ms,
            "limit": limit,
        }

        resp = session.get(f"{BINANCE_FUTURES_BASE_URL}{KLINES_PATH}", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        for row in data:
            # https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-data
            out.append(
                Kline(
                    open_time_ms=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    close_time_ms=int(row[6]),
                )
            )

        # Advance start to just after the last candle open time.
        last_open = int(data[-1][0])
        if last_open == next_start:
            # Defensive guard to avoid infinite loops if API returns a single repeated candle.
            break
        next_start = last_open + 1

        time.sleep(pause_s)

    return out


def write_csv(path: Path, klines: Iterable[Kline]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "open_time_ms",
            "open_time_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time_ms",
            "close_time_utc",
        ])
        for k in klines:
            open_dt = datetime.fromtimestamp(k.open_time_ms / 1000, tz=timezone.utc)
            close_dt = datetime.fromtimestamp(k.close_time_ms / 1000, tz=timezone.utc)
            writer.writerow([
                k.open_time_ms,
                open_dt.isoformat(),
                k.open,
                k.high,
                k.low,
                k.close,
                k.volume,
                k.close_time_ms,
                close_dt.isoformat(),
            ])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Binance USDⓈ-M futures klines to CSV")
    parser.add_argument("--interval", default="1h", help="Kline interval, e.g. 1h or 5m")
    parser.add_argument("--start", required=True, help="Start date (UTC) YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date (UTC) YYYY-MM-DD")
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated symbols (default: paper's Table 1 set)",
    )
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--pause", type=float, default=0.2, help="Pause between API calls (seconds)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    start_dt = _parse_utc_date(args.start)
    end_dt = _parse_utc_date(args.end)
    start_ms = _to_ms(start_dt)
    end_ms = _to_ms(end_dt)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    out_dir = Path(args.out)

    with requests.Session() as session:
        for symbol in symbols:
            klines = fetch_klines(
                symbol=symbol,
                interval=args.interval,
                start_ms=start_ms,
                end_ms=end_ms,
                session=session,
                pause_s=args.pause,
            )
            out_path = out_dir / f"{symbol}_{args.interval}.csv"
            write_csv(out_path, klines)
            print(f"{symbol}: wrote {len(klines)} rows -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

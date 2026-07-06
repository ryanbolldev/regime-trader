#!/usr/bin/env python3
"""
scripts/tastytrade_parity_check.py
-----------------------------------
Compare WheelOptionLeg output from the Alpaca and tastytrade providers for the
same underlying, so we can trust flipping OPTIONS_DATA_PROVIDER to "tastytrade".

Matches contracts by (expiration, strike) and reports, for the overlap, the
per-field differences in delta, IV, bid, ask, and open interest.

Read-only. Hits Alpaca (paper options data) and tastytrade (live) — no orders.

Usage
-----
    python scripts/tastytrade_parity_check.py            # MSTR
    python scripts/tastytrade_parity_check.py MSTR 21 45 # symbol min_dte max_dte

Note: run during market hours for a meaningful quote/greek comparison. After
hours, bid/ask/delta may be stale or absent on one side; strikes, expirations,
and open interest remain comparable.
"""

from __future__ import annotations

import sys
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from dotenv import load_dotenv

from config.credentials import enable_os_trust_store

enable_os_trust_store()
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

from wheel_scanner import options_data, options_data_tastytrade


def _bar(char: str = "─", width: int = 76) -> str:
    return char * width


def _key(leg) -> tuple:
    return (leg.expiration, round(leg.strike, 2))


def _diffs(label: str, values: list[float]) -> str:
    if not values:
        return f"  {label:<16}: no overlap with both values present"
    med = statistics.median(values)
    mx = max(values)
    return f"  {label:<16}: n={len(values):<4} median|Δ|={med:<10.4f} max|Δ|={mx:.4f}"


def run(symbol: str, min_dte: int, max_dte: int) -> None:
    print(_bar("═"))
    print(f"  Provider parity — {symbol}  ({min_dte}-{max_dte} DTE)")
    print(_bar("═"))

    alpaca_client = options_data._build_options_client()
    tt_session = options_data_tastytrade.build_session()

    alpaca_legs = options_data.fetch_put_chain(symbol, alpaca_client, min_dte, max_dte)
    tt_legs = options_data_tastytrade.fetch_put_chain(symbol, tt_session, min_dte, max_dte)

    print(f"  Alpaca puts    : {len(alpaca_legs)}")
    print(f"  tastytrade puts: {len(tt_legs)}")

    a_by = {_key(l): l for l in alpaca_legs}
    t_by = {_key(l): l for l in tt_legs}
    common = sorted(set(a_by) & set(t_by))
    print(f"  Matched (exp,strike): {len(common)}")
    print(_bar())

    d_delta, d_iv, d_bid, d_ask, d_oi = [], [], [], [], []
    for k in common:
        a, t = a_by[k], t_by[k]
        if a.delta is not None and t.delta is not None:
            d_delta.append(abs(a.delta - t.delta))
        if a.implied_volatility is not None and t.implied_volatility is not None:
            d_iv.append(abs(a.implied_volatility - t.implied_volatility))
        if a.bid is not None and t.bid is not None:
            d_bid.append(abs(a.bid - t.bid))
        if a.ask is not None and t.ask is not None:
            d_ask.append(abs(a.ask - t.ask))
        if a.open_interest is not None and t.open_interest is not None:
            d_oi.append(abs(a.open_interest - t.open_interest))

    print("  Absolute differences on matched contracts (Alpaca vs tastytrade):")
    print(_diffs("delta", d_delta))
    print(_diffs("implied_vol", d_iv))
    print(_diffs("bid", d_bid))
    print(_diffs("ask", d_ask))
    print(_diffs("open_interest", [float(x) for x in d_oi]))

    print(_bar("═"))


if __name__ == "__main__":
    args = sys.argv[1:]
    sym = args[0].upper() if len(args) > 0 else "MSTR"
    lo = int(args[1]) if len(args) > 1 else 21
    hi = int(args[2]) if len(args) > 2 else 45
    try:
        run(sym, lo, hi)
    except Exception as exc:
        print(f"\n  PARITY CHECK FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

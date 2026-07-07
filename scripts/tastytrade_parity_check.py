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


def _pairs(common, a_by, t_by, attr: str) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for k in common:
        a = getattr(a_by[k], attr)
        t = getattr(t_by[k], attr)
        if a is not None and t is not None:
            out.append((float(a), float(t)))
    return out


def _corr(pairs: list[tuple[float, float]]) -> str:
    if len(pairs) < 2:
        return "n/a"
    try:
        return f"{statistics.correlation([a for a, _ in pairs], [t for _, t in pairs]):+.3f}"
    except statistics.StatisticsError:
        return "n/a"   # a constant series (e.g. all-equal) has undefined correlation


def _stat_line(label: str, pairs: list[tuple[float, float]]) -> str:
    if not pairs:
        return f"  {label:<14}: no overlap with both values present"
    diffs = [t - a for a, t in pairs]
    absd  = [abs(d) for d in diffs]
    return (
        f"  {label:<14}: n={len(pairs):<4} "
        f"median|Δ|={statistics.median(absd):<8.4f} "
        f"bias(tt−alp)={statistics.fmean(diffs):+.4f} "
        f"max|Δ|={max(absd):<8.4f} corr={_corr(pairs)}"
    )


def _iv_by_delta_bucket(common, a_by, t_by) -> str:
    """IV gap split by moneyness (via |delta|) — isolates wing vs ATM behaviour."""
    buckets: dict[str, list[tuple[float, float]]] = {
        "deep OTM |Δ|<.15": [], "OTM .15–.35": [],
        "ATM .35–.65": [],      "ITM |Δ|>.65": [],
    }
    for k in common:
        a, t = a_by[k], t_by[k]
        if a.implied_volatility is None or t.implied_volatility is None:
            continue
        d = t.delta if t.delta is not None else a.delta
        if d is None:
            continue
        ad  = abs(d)
        key = ("deep OTM |Δ|<.15" if ad < 0.15 else
               "OTM .15–.35"      if ad < 0.35 else
               "ATM .35–.65"      if ad < 0.65 else "ITM |Δ|>.65")
        buckets[key].append((float(a.implied_volatility), float(t.implied_volatility)))

    lines = []
    for name, pairs in buckets.items():
        if not pairs:
            lines.append(f"    {name:<18}: (none)")
            continue
        diffs = [t - a for a, t in pairs]
        lines.append(
            f"    {name:<18}: n={len(pairs):<3} "
            f"median|Δ|={statistics.median([abs(d) for d in diffs]):<7.4f} "
            f"bias={statistics.fmean(diffs):+.4f}"
        )
    return "\n".join(lines)


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

    print("  Field comparison on matched contracts — bias = mean(tt − alpaca):")
    print(_stat_line("delta",         _pairs(common, a_by, t_by, "delta")))
    print(_stat_line("implied_vol",   _pairs(common, a_by, t_by, "implied_volatility")))
    print(_stat_line("bid",           _pairs(common, a_by, t_by, "bid")))
    print(_stat_line("ask",           _pairs(common, a_by, t_by, "ask")))
    print(_stat_line("open_interest", _pairs(common, a_by, t_by, "open_interest")))

    print(_bar())
    print("  Implied-vol gap by moneyness (delta bucket):")
    print(_iv_by_delta_bucket(common, a_by, t_by))

    print(_bar())
    print("  Reading it:")
    print("    small median|Δ| + corr≈1 + bias≈0  → parity, safe to flip provider")
    print("    small median|Δ| + corr≈1 + bias≠0  → systematic offset (re-baseline IV Rank)")
    print("    large median|Δ| + low corr         → staleness/noise (re-run in market hours)")
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

#!/usr/bin/env python3
"""
scripts/tastytrade_connection_test.py
--------------------------------------
Connectivity smoke test for the tastytrade OAuth2 options-data integration.

Verifies, without ever reading or printing credential values:
  1. OAuth2 session   — proves TT_SECRET / TT_REFRESH are valid (Session() does
                        the token refresh on construction; bad creds raise here).
  2. REST option chain — proves market-data access and yields streamer symbols.
  3. DXLink streaming  — confirms Greeks (delta, IV), Quote (bid/ask), and
                        Summary (open_interest) events flow for real contracts.

Credentials load from .env into the process environment via python-dotenv; this
script never accesses their values. The tastytrade SDK reads $TT_SECRET and
$TT_REFRESH directly.

Usage
-----
    python scripts/tastytrade_connection_test.py            # defaults to SPY
    python scripts/tastytrade_connection_test.py MSTR       # custom underlying

Streaming values (step 3) are richest during market hours. Outside hours,
Summary/open_interest is typically still available while Quote/Greeks may be
sparse; steps 1-2 succeeding already proves credentials and data access.
"""

from __future__ import annotations

import asyncio
import datetime
import sys
from pathlib import Path

# Allow running directly from scripts/ or from project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

# Windows consoles default to cp1252, which cannot encode the box-drawing glyphs.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# Validate TLS against the OS trust store. Local TLS-inspection proxies (AV,
# corporate) inject a root CA present in the Windows store but not in certifi,
# which otherwise breaks the tastytrade handshake with CERTIFICATE_VERIFY_FAILED.
import truststore

truststore.inject_into_ssl()

from dotenv import load_dotenv

# Load credentials into os.environ without ever exposing their values here.
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

from tastytrade import DXLinkStreamer, Session
from tastytrade.dxfeed import Greeks, Quote, Summary
from tastytrade.instruments import OptionType, get_option_chain


# ---------------------------------------------------------------------------
# Display helpers (matches scripts/connection_test.py style)
# ---------------------------------------------------------------------------

def _bar(char: str = "─", width: int = 64) -> str:
    return char * width


def _step(n: int, title: str) -> None:
    print(f"\n{_bar()}")
    print(f"  Step {n} — {title}")
    print(_bar())


def _ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def _info(label: str, value: str) -> None:
    print(f"  {label:<24}: {value}")


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------

_TARGET_DTE = 35        # aim near the wheel's put-selling window
_N_CONTRACTS = 5
_STREAM_TIMEOUT = 15.0  # seconds to wait for streaming events


def _pick_expiration(chain: dict) -> datetime.date:
    today = datetime.date.today()
    return min(chain.keys(), key=lambda d: abs((d - today).days - _TARGET_DTE))


def _select_puts(options: list) -> list:
    puts = sorted(
        (o for o in options if o.option_type == OptionType.PUT),
        key=lambda o: o.strike_price,
    )
    if not puts:
        return []
    mid = len(puts) // 2
    lo = max(0, mid - _N_CONTRACTS // 2)
    return puts[lo : lo + _N_CONTRACTS]


async def _collect(streamer, event_type, n_expected: int, timeout: float) -> dict:
    """Collect one event per symbol until all arrive or timeout elapses."""
    out: dict = {}
    try:
        async with asyncio.timeout(timeout):
            async for ev in streamer.listen(event_type):
                out[ev.event_symbol] = ev
                if len(out) >= n_expected:
                    break
    except (asyncio.TimeoutError, TimeoutError):
        pass
    return out


async def _run_data(session: Session, symbol: str) -> None:
    _step(2, f"REST option chain — {symbol}")
    chain = await get_option_chain(session, symbol)
    if not chain:
        raise RuntimeError(f"empty option chain for {symbol}")
    _ok("Credentials confirmed — first authenticated request succeeded (TT_SECRET / TT_REFRESH valid)")

    exp = _pick_expiration(chain)
    contracts = _select_puts(chain[exp])
    _info("Expirations available", str(len(chain)))
    _info("Chosen expiration", f"{exp}  (~{(exp - datetime.date.today()).days} DTE)")
    _info("Put contracts sampled", str(len(contracts)))
    if not contracts:
        raise RuntimeError("no put contracts found in chosen expiration")
    _ok("Option chain fetched — streamer symbols obtained")

    _step(3, "DXLink streaming — Greeks / Quote / Summary")
    symbols = [c.streamer_symbol for c in contracts]
    async with DXLinkStreamer(session) as streamer:
        await streamer.subscribe(Greeks, symbols)
        await streamer.subscribe(Quote, symbols)
        await streamer.subscribe(Summary, symbols)
        greeks, quotes, summaries = await asyncio.gather(
            _collect(streamer, Greeks, len(symbols), _STREAM_TIMEOUT),
            _collect(streamer, Quote, len(symbols), _STREAM_TIMEOUT),
            _collect(streamer, Summary, len(symbols), _STREAM_TIMEOUT),
        )

    print(f"  {'contract':<26} {'delta':>7} {'IV':>7} {'bid':>7} {'ask':>7} {'OI':>8}")
    for c in contracts:
        s = c.streamer_symbol
        g, q, m = greeks.get(s), quotes.get(s), summaries.get(s)
        delta_s = f"{float(g.delta):+.3f}" if g and g.delta is not None else "n/a"
        iv_s = f"{float(g.volatility) * 100:5.1f}%" if g and g.volatility is not None else "n/a"
        bid_s = f"{float(q.bid_price):.2f}" if q and q.bid_price is not None else "n/a"
        ask_s = f"{float(q.ask_price):.2f}" if q and q.ask_price is not None else "n/a"
        oi_s = str(m.open_interest) if m and m.open_interest is not None else "n/a"
        print(f"  {s:<26} {delta_s:>7} {iv_s:>7} {bid_s:>7} {ask_s:>7} {oi_s:>8}")

    counts = {
        "Greeks (delta/IV)": len(greeks),
        "Quote (bid/ask)": len(quotes),
        "Summary (open_interest)": len(summaries),
    }
    print()
    for label, n in counts.items():
        _info(label, f"{n}/{len(symbols)} contracts")

    if not any(counts.values()):
        print("\n  ⚠  No streaming events received — likely outside market hours.")
        print("     Steps 1-2 passed, so credentials and data access are confirmed.")
    else:
        _ok("Streaming events received")


def run(symbol: str) -> None:
    print(_bar("═"))
    print("  tastytrade Connection Test")
    print(_bar("═"))

    _step(1, "OAuth2 session")
    session = Session()
    _ok("Session constructed (SDK refreshes the token on the first API call)")

    asyncio.run(_run_data(session, symbol))

    print(f"\n{_bar('═')}")
    print("  CONNECTION TEST PASSED")
    print(_bar("═"))


if __name__ == "__main__":
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"
    try:
        run(ticker)
    except KeyboardInterrupt:
        print("\n  Interrupted.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n{_bar('═')}", file=sys.stderr)
        print("  CONNECTION TEST FAILED", file=sys.stderr)
        print(_bar("═"), file=sys.stderr)
        print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

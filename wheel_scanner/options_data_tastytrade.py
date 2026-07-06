"""
wheel_scanner/options_data_tastytrade.py
------------------------------------------
tastytrade options-data provider for the wheel scanner.

Mirrors the fetching surface of wheel_scanner/options_data.py — fetch_put_chain,
count_expiry_cycles, compute_ivr — but sources option data from tastytrade
instead of Alpaca. Emits the same WheelOptionLeg dataclass so the scanner is
provider-agnostic.

Why tastytrade: it exposes open interest (via the DXLink Summary event), which
Alpaca's option snapshots do not. Chain structure comes from a REST call
(get_option_chain); live per-contract values arrive over DXLink streaming:

    Greeks  → delta, volatility (IV)
    Quote   → bid_price, ask_price
    Summary → open_interest
    Trade   → day_volume (today's contracts traded)

OHLCV/stock bars stay on Alpaca: compute_ivr takes tastytrade's current IV and
combines it with realized_vol_range() (Alpaca bars) so IV Rank is computed
identically to the Alpaca provider.

Credentials load via config.credentials.load_tastytrade_credentials(); TLS is
routed through the OS trust store (enable_os_trust_store) to survive local
TLS-inspection proxies. The .env file is never read here directly.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Optional

import numpy as np

from config.credentials import enable_os_trust_store, load_tastytrade_credentials
from wheel_scanner.options_data import WheelOptionLeg, _safe_float, realized_vol_range

log = logging.getLogger(__name__)

_STREAM_TIMEOUT = 15.0    # seconds to wait for streaming events per fetch

# A tastytrade Session's async HTTP client binds to the first event loop it runs
# on, so asyncio.run() (which closes its loop) breaks the next call. Reuse one
# persistent loop for every call instead.
_loop: Optional[asyncio.AbstractEventLoop] = None


def _run(coro):
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
    return _loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

def build_session(is_test: bool = False):
    """Construct an authenticated tastytrade Session.

    Enables the OS trust store first so the TLS handshake succeeds behind local
    inspection proxies. Credential values never leave config.credentials.
    """
    enable_os_trust_store()
    # tastytrade attaches its own DEBUG handler that bypasses the root config and
    # logs every websocket frame; keep it quiet unless explicitly raised.
    logging.getLogger("tastytrade").setLevel(logging.WARNING)
    creds = load_tastytrade_credentials(is_test=is_test)
    from tastytrade import Session

    return Session(creds.provider_secret, creds.refresh_token, is_test=creds.is_test)


# ---------------------------------------------------------------------------
# Chain fetching
# ---------------------------------------------------------------------------

def fetch_put_chain(
    symbol:   str,
    session,
    min_dte:  int = 21,
    max_dte:  int = 45,
) -> list[WheelOptionLeg]:
    """Fetch put options in the [min_dte, max_dte] window for symbol.

    Returns an empty list on any error (caller treats as no options available).
    """
    try:
        return _run(_fetch_put_chain_async(session, symbol, min_dte, max_dte))
    except Exception as exc:
        log.warning("fetch_put_chain [%s]: %s", symbol, exc)
        return []


async def _fetch_put_chain_async(
    session, symbol: str, min_dte: int, max_dte: int
) -> list[WheelOptionLeg]:
    from tastytrade import DXLinkStreamer
    from tastytrade.dxfeed import Greeks, Quote, Summary, Trade
    from tastytrade.instruments import OptionType, get_option_chain

    today = datetime.date.today()
    chain = await get_option_chain(session, symbol)
    if not chain:
        return []

    puts: list[tuple] = []
    for expiration, options in chain.items():
        dte = (expiration - today).days
        if dte < min_dte or dte > max_dte:
            continue
        puts.extend((o, dte) for o in options if o.option_type == OptionType.PUT)

    if not puts:
        return []

    symbols = [o.streamer_symbol for o, _ in puts]
    greeks, quotes, summaries, trades = await _stream_events(
        session, symbols, (Greeks, Quote, Summary, Trade)
    )

    legs: list[WheelOptionLeg] = []
    for o, dte in puts:
        s = o.streamer_symbol
        g, q, m, t = greeks.get(s), quotes.get(s), summaries.get(s), trades.get(s)
        legs.append(WheelOptionLeg(
            occ_symbol         = o.symbol,
            underlying         = symbol,
            expiration         = o.expiration_date,
            strike             = float(o.strike_price),
            option_type        = "put",
            delta              = _safe_float(g.delta) if g else None,
            implied_volatility = _safe_float(g.volatility) if g else None,
            bid                = _safe_float(q.bid_price) if q else None,
            ask                = _safe_float(q.ask_price) if q else None,
            open_interest      = int(m.open_interest) if m and m.open_interest is not None else None,
            volume_today       = int(t.day_volume) if t and t.day_volume is not None else None,
            dte                = dte,
        ))
    return legs


def count_expiry_cycles(symbol: str, session) -> int:
    """Return the number of distinct expiry dates for symbol's options.

    An empty chain (no options, API error) returns 0.
    """
    try:
        return _run(_count_expiry_cycles_async(session, symbol))
    except Exception as exc:
        log.debug("count_expiry_cycles [%s]: %s", symbol, exc)
        return 0


async def _count_expiry_cycles_async(session, symbol: str) -> int:
    from tastytrade.instruments import get_option_chain

    chain = await get_option_chain(session, symbol)
    return len(chain) if chain else 0


# ---------------------------------------------------------------------------
# IV Rank
# ---------------------------------------------------------------------------

def compute_ivr(
    symbol:        str,
    session,
    stock_client,
    lookback_days: int = 252,
) -> Optional[float]:
    """Compute IV Rank (0–100) for symbol.

    Current IV is the median implied vol of options in the 21–60 DTE window
    (from tastytrade Greeks). The historical IV range is realized-vol-based and
    comes from Alpaca bars via realized_vol_range(), matching the Alpaca
    provider exactly. Returns None when data is insufficient.
    """
    try:
        current_iv = _run(_current_iv_async(session, symbol))
    except Exception as exc:
        log.debug("compute_ivr [%s]: chain/stream failed: %s", symbol, exc)
        return None

    if current_iv is None:
        return None

    vol_range = realized_vol_range(symbol, stock_client, lookback_days)
    if vol_range is None:
        return None
    iv_low, iv_high = vol_range

    ivr = (current_iv - iv_low) / (iv_high - iv_low) * 100.0
    return round(max(0.0, min(100.0, ivr)), 1)


async def _current_iv_async(
    session, symbol: str, min_dte: int = 21, max_dte: int = 60
) -> Optional[float]:
    from tastytrade.dxfeed import Greeks
    from tastytrade.instruments import get_option_chain

    today = datetime.date.today()
    chain = await get_option_chain(session, symbol)
    if not chain:
        return None

    symbols: list[str] = []
    for expiration, options in chain.items():
        dte = (expiration - today).days
        if min_dte <= dte <= max_dte:
            symbols.extend(o.streamer_symbol for o in options)

    if not symbols:
        return None

    (greeks,) = await _stream_events(session, symbols, (Greeks,))
    ivs = [float(g.volatility) for g in greeks.values() if g.volatility is not None]
    if not ivs:
        return None
    return float(np.median(ivs))


# ---------------------------------------------------------------------------
# Streaming plumbing
# ---------------------------------------------------------------------------

async def _stream_events(session, symbols: list[str], event_types: tuple) -> tuple[dict, ...]:
    """Subscribe to each event type for symbols and collect one event per
    symbol (or until _STREAM_TIMEOUT). Returns a dict per event type, in order.
    """
    from tastytrade import DXLinkStreamer

    async with DXLinkStreamer(session) as streamer:
        for event_type in event_types:
            await streamer.subscribe(event_type, symbols)
        return tuple(
            await asyncio.gather(*(
                _collect(streamer, event_type, len(symbols), _STREAM_TIMEOUT)
                for event_type in event_types
            ))
        )


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

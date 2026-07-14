"""
wheel_scanner/options_data.py
------------------------------
Options chain fetching and processing for the wheel scanner.

Uses alpaca-py's OptionHistoricalDataClient directly to capture fields
(open_interest, daily bar volume) that the existing AlpacaClient wrapper
does not expose in its OptionContract dataclass.

Reuses _parse_occ_symbol from broker/alpaca_client to avoid duplication.

Data available from Alpaca option snapshots:
    delta, implied_volatility, bid, ask       ✓  (from greeks + latest_quote)
    open_interest                              ✓  (from snapshot.open_interest)
    today's contract volume                    ✓  (from snapshot.daily_bar.volume)
    average daily options volume (historical)  ✗  stub — see TODO below

TODO: For true average daily options volume, integrate a Tradier API call:
    GET https://api.tradier.com/v1/markets/history?symbol={occ}&interval=daily
    This requires TRADIER_API_KEY in the .env file. The current implementation
    uses today's snapshot volume as a same-day proxy when available.
"""

from __future__ import annotations

import datetime
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from alpaca.data.enums import DataFeed
from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from broker.alpaca_client import _parse_occ_symbol
from config.credentials import load_credentials

log = logging.getLogger(__name__)

_RATE_LIMIT_PAUSE = 62.0    # seconds to wait on HTTP 429
_INTER_CALL_DELAY = 0.5     # seconds between option chain requests


@dataclass(frozen=True)
class WheelOptionLeg:
    """Rich option contract snapshot including open interest and volume."""

    occ_symbol:         str
    underlying:         str
    expiration:         datetime.date
    strike:             float
    option_type:        str             # "put" or "call"
    delta:              Optional[float]
    implied_volatility: Optional[float]
    bid:                Optional[float]
    ask:                Optional[float]
    open_interest:      Optional[int]
    volume_today:       Optional[int]   # contracts traded today (snapshot daily bar)
    dte:                int             # days to expiration from today


# ---------------------------------------------------------------------------
# Client factory (credentials loaded once from .env)
# ---------------------------------------------------------------------------

def _build_options_client() -> OptionHistoricalDataClient:
    creds = load_credentials()
    return OptionHistoricalDataClient(
        api_key    = creds.api_key,
        secret_key = creds.api_secret,
    )


def _build_stock_client() -> StockHistoricalDataClient:
    creds = load_credentials()
    return StockHistoricalDataClient(
        api_key    = creds.api_key,
        secret_key = creds.api_secret,
    )


# ---------------------------------------------------------------------------
# Chain fetching
# ---------------------------------------------------------------------------

def fetch_put_chain(
    symbol:         str,
    options_client: OptionHistoricalDataClient,
    min_dte:        int = 21,
    max_dte:        int = 45,
) -> list[WheelOptionLeg]:
    """Fetch put options in the [min_dte, max_dte] window for symbol.

    Returns an empty list on any error (caller should treat as no options
    available and exclude the ticker).
    """
    today   = datetime.date.today()
    exp_gte = today + datetime.timedelta(days=min_dte)
    exp_lte = today + datetime.timedelta(days=max_dte)

    try:
        chain = options_client.get_option_chain(
            OptionChainRequest(
                underlying_symbol  = symbol,
                expiration_date_gte = exp_gte,
                expiration_date_lte = exp_lte,
            )
        )
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        if status == 429:
            log.warning(
                "fetch_put_chain [%s]: rate limited — pausing %.0fs",
                symbol, _RATE_LIMIT_PAUSE,
            )
            time.sleep(_RATE_LIMIT_PAUSE)
            return []
        if status == 422 or status == 404:
            log.debug("fetch_put_chain [%s]: no options data (status %s)", symbol, status)
            return []
        log.warning("fetch_put_chain [%s]: %s", symbol, exc)
        return []

    legs: list[WheelOptionLeg] = []
    for occ_sym, snap in chain.items():
        parsed = _parse_occ_symbol(occ_sym)
        if parsed is None:
            continue
        _, expiration_str, opt_type, strike = parsed

        if opt_type != "put":
            continue

        expiration = datetime.date.fromisoformat(expiration_str)
        dte        = (expiration - today).days

        greeks   = getattr(snap, "greeks", None)
        delta    = _safe_float(greeks.delta if greeks else None)
        iv_raw   = getattr(snap, "implied_volatility", None)
        iv       = _safe_float(iv_raw)
        quote    = getattr(snap, "latest_quote", None)
        bid      = _safe_float(getattr(quote, "bid_price", None) if quote else None)
        ask      = _safe_float(getattr(quote, "ask_price", None) if quote else None)
        oi_raw   = getattr(snap, "open_interest", None)
        oi       = int(float(oi_raw)) if oi_raw is not None else None

        # Daily bar volume (today's contracts traded — proxy for average volume)
        # TODO: replace with Tradier historical average for accuracy
        daily_bar = getattr(snap, "daily_bar", None)
        vol_today = (
            int(float(daily_bar.volume))
            if daily_bar and getattr(daily_bar, "volume", None) is not None
            else None
        )

        legs.append(WheelOptionLeg(
            occ_symbol         = occ_sym,
            underlying         = symbol,
            expiration         = expiration,
            strike             = strike,
            option_type        = "put",
            delta              = delta,
            implied_volatility = iv,
            bid                = bid,
            ask                = ask,
            open_interest      = oi,
            volume_today       = vol_today,
            dte                = dte,
        ))

    return legs


def count_expiry_cycles(
    symbol:         str,
    options_client: OptionHistoricalDataClient,
) -> int:
    """Return the number of distinct expiry dates for symbol's options.

    Used to enforce the "at least 2 active expiry cycles" filter.
    An empty chain (no options, API error) returns 0.
    """
    try:
        chain = options_client.get_option_chain(
            OptionChainRequest(underlying_symbol=symbol)
        )
    except Exception as exc:
        log.debug("count_expiry_cycles [%s]: %s", symbol, exc)
        return 0

    expiries: set[str] = set()
    for occ_sym in chain.keys():
        parsed = _parse_occ_symbol(occ_sym)
        if parsed:
            expiries.add(parsed[1])     # expiration YYYY-MM-DD
    return len(expiries)


# ---------------------------------------------------------------------------
# Target put selection
# ---------------------------------------------------------------------------

def find_target_put(
    legs:         list[WheelOptionLeg],
    target_delta: float = -0.30,
) -> Optional[WheelOptionLeg]:
    """Return the put closest to target_delta among the provided legs.

    Requires delta to be present. Returns None if no legs have delta data.
    """
    candidates = [leg for leg in legs if leg.delta is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda leg: abs(leg.delta - target_delta))


# ---------------------------------------------------------------------------
# Price / yield helpers
# ---------------------------------------------------------------------------

def mid_price(leg: WheelOptionLeg) -> Optional[float]:
    """Compute bid-ask mid price. Returns bid or ask alone when one is absent."""
    if leg.bid is not None and leg.ask is not None:
        return (leg.bid + leg.ask) / 2.0
    return leg.bid if leg.bid is not None else leg.ask


def annualized_yield_pct(put_mid: float, strike: float, dte: int) -> float:
    """(put_mid / strike) × (365 / dte) × 100 — annualized premium yield."""
    if dte <= 0 or strike <= 0 or put_mid < 0:
        return 0.0
    return (put_mid / strike) * (365.0 / dte) * 100.0


def bid_ask_spread_pct(leg: WheelOptionLeg) -> Optional[float]:
    """Bid-ask spread as a percentage of mid price."""
    if leg.bid is None or leg.ask is None:
        return None
    m = mid_price(leg)
    if m is None or m <= 0:
        return None
    return (leg.ask - leg.bid) / m * 100.0


# ---------------------------------------------------------------------------
# IV Rank
# ---------------------------------------------------------------------------

def compute_ivr(
    symbol:         str,
    options_client: OptionHistoricalDataClient,
    stock_client:   StockHistoricalDataClient,
    lookback_days:  int = 252,
) -> Optional[float]:
    """Compute IV Rank (0–100) for symbol.

    Matches the approach in AlpacaClient.get_iv_rank():
      current_iv  = median implied vol of puts in 21–60 DTE window
      iv_low/high = min/max of 20-day rolling realized vol over lookback period
      ivr         = (current_iv - iv_low) / (iv_high - iv_low) × 100

    Uses realized volatility as the historical IV proxy because Alpaca does
    not store a historical IV time series (only live snapshots are available).

    Returns None when data is insufficient (< 30 bars or no chain IV data).
    """
    today   = datetime.date.today()
    exp_gte = today + datetime.timedelta(days=21)
    exp_lte = today + datetime.timedelta(days=60)

    # Current IV from live options chain
    try:
        chain = options_client.get_option_chain(
            OptionChainRequest(
                underlying_symbol   = symbol,
                expiration_date_gte = exp_gte,
                expiration_date_lte = exp_lte,
            )
        )
    except Exception as exc:
        log.debug("compute_ivr [%s]: chain fetch failed: %s", symbol, exc)
        return None

    ivs = [
        float(getattr(snap, "implied_volatility"))
        for snap in chain.values()
        if getattr(snap, "implied_volatility", None) is not None
    ]
    if not ivs:
        log.debug("compute_ivr [%s]: no IV data in 21-60 DTE chain", symbol)
        return None

    current_iv = float(np.median(ivs))

    vol_range = realized_vol_range(symbol, stock_client, lookback_days)
    if vol_range is None:
        return None
    iv_low, iv_high = vol_range

    ivr = (current_iv - iv_low) / (iv_high - iv_low) * 100.0
    return round(max(0.0, min(100.0, ivr)), 1)


def realized_vol_range(
    symbol:        str,
    stock_client:  StockHistoricalDataClient,
    lookback_days: int = 252,
) -> Optional[tuple[float, float]]:
    """Return (iv_low, iv_high): the min/max of 20-day rolling realized vol over
    the lookback window, used as the historical IV-range proxy for IV Rank.

    Provider-agnostic — shared by the Alpaca and tastytrade options providers so
    both compute IV Rank identically; only the current-IV source differs.
    Returns None when fewer than 30 bars are available or the range is degenerate.
    """
    end   = datetime.datetime.now(datetime.timezone.utc)
    start = end - datetime.timedelta(days=int(lookback_days * 1.6))

    try:
        resp = stock_client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols = symbol,
                timeframe         = TimeFrame.Day,
                start             = start,
                end               = end,
                feed              = DataFeed.IEX,   # paper accounts are denied recent SIP data
            )
        )
        try:
            bars = list(resp[symbol])
        except (KeyError, TypeError):
            bars = []
    except Exception as exc:
        log.debug("realized_vol_range [%s]: bar fetch failed: %s", symbol, exc)
        return None

    if len(bars) < 30:
        return None

    closes   = np.array([b.close for b in bars], dtype=float)
    log_rets = np.log(closes[1:] / closes[:-1])

    roll = 20
    rv_series = [
        log_rets[i - roll : i].std() * np.sqrt(252)
        for i in range(roll, len(log_rets) + 1)
    ]

    iv_low  = float(min(rv_series))
    iv_high = float(max(rv_series))

    if iv_high <= iv_low:
        return None

    return iv_low, iv_high


# ---------------------------------------------------------------------------
# OHLCV + SMA helpers
# ---------------------------------------------------------------------------

def fetch_ohlcv_for_sma(
    symbol:       str,
    stock_client: StockHistoricalDataClient,
    n_bars:       int = 260,
) -> Optional["pd.DataFrame"]:
    """Fetch n_bars of daily OHLCV data for SMA computation.

    Returns a DataFrame with a 'close' column, or None on failure.
    """
    import pandas as pd

    end   = datetime.datetime.now(datetime.timezone.utc)
    start = end - datetime.timedelta(days=int(n_bars * 1.8))

    try:
        resp = stock_client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols = symbol,
                timeframe         = TimeFrame.Day,
                start             = start,
                end               = end,
                feed              = DataFeed.IEX,   # paper accounts are denied recent SIP data
            )
        )
        try:
            bars = list(resp[symbol])
        except (KeyError, TypeError):
            bars = []
    except Exception as exc:
        log.debug("fetch_ohlcv_for_sma [%s]: %s", symbol, exc)
        return None

    if len(bars) < 55:
        log.debug("fetch_ohlcv_for_sma [%s]: only %d bars (need ≥55)", symbol, len(bars))
        return None

    df = pd.DataFrame(
        {"close": [float(b.close) for b in bars],
         "volume": [float(b.volume) for b in bars]}
    )
    return df


def compute_sma_trend(df: "pd.DataFrame") -> tuple[float, float, float, bool]:
    """Compute price, SMA50, SMA200, and whether SMA50 is declining.

    Returns (price, sma50, sma200, sma50_declining).
    sma50_declining = True when SMA50 today < SMA50 five bars ago.

    Returns (price, NaN, NaN, False) when insufficient bars for SMA200.
    """
    import numpy as np

    closes = df["close"]
    price  = float(closes.iloc[-1])

    if len(closes) < 50:
        return price, float("nan"), float("nan"), False

    sma50 = float(closes.rolling(50).mean().iloc[-1])

    sma200_series = closes.rolling(200).mean()
    if sma200_series.iloc[-1] != sma200_series.iloc[-1]:  # NaN check
        sma200 = float("nan")
    else:
        sma200 = float(sma200_series.iloc[-1])

    # SMA50 declining: compare today to 5 bars ago
    sma50_5_ago = float(closes.rolling(50).mean().iloc[-6]) if len(closes) >= 55 else float("nan")
    sma50_declining = (
        sma50_5_ago == sma50_5_ago          # not NaN
        and sma50 < sma50_5_ago
    )

    return price, sma50, sma200, sma50_declining


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _safe_float(val) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None

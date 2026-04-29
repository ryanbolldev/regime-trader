"""
core/market_data.py
--------------------
Data acquisition layer: historical bars, latest bars, market hours, and
option chain.  All network calls use the alpaca-py SDK with 3-attempt
exponential backoff and dedicated 429 rate-limit handling.

Credentials are loaded once from .env via config/credentials.py.
Module-level client singletons are intentionally exposed so tests can
replace them via monkeypatch without hitting real credentials.

Public interface:
  get_historical_bars(symbol, start, end, timeframe) -> pd.DataFrame
  get_latest_bar(symbol, timeframe="1Day")           -> pd.Series
  get_latest_bars(symbol, n_bars, timeframe="1Day")  -> pd.DataFrame
  is_market_open()                                   -> bool
  get_option_chain(symbol)                           -> list[OptionContract]
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Optional

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config.credentials import load_credentials

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons (replaced with mocks in tests)
# ---------------------------------------------------------------------------

_stock_client: Optional[StockHistoricalDataClient] = None
_alpaca_client = None  # broker.alpaca_client.AlpacaClient; imported lazily

# ---------------------------------------------------------------------------
# Timeframe mapping
# ---------------------------------------------------------------------------

_TIMEFRAME_MAP: dict[str, TimeFrame] = {
    "1Day":  TimeFrame.Day,
    "1Hour": TimeFrame.Hour,
    "5Min":  TimeFrame(5, TimeFrameUnit.Minute),
}

# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

_RETRY_DELAYS   = (1.0, 2.0)   # waits between attempt 1→2 and 2→3
_RATE_LIMIT_WAIT = 60.0         # pause on HTTP 429 before retrying


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_stock_client() -> StockHistoricalDataClient:
    global _stock_client
    if _stock_client is None:
        creds = load_credentials()
        _stock_client = StockHistoricalDataClient(
            api_key    = creds.api_key,
            secret_key = creds.api_secret,
        )
    return _stock_client


def _get_alpaca_client():
    global _alpaca_client
    if _alpaca_client is None:
        from broker.alpaca_client import AlpacaClient
        _alpaca_client = AlpacaClient()
    return _alpaca_client


def _parse_timeframe(timeframe: str) -> TimeFrame:
    try:
        return _TIMEFRAME_MAP[timeframe]
    except KeyError:
        raise ValueError(
            f"Unknown timeframe: {timeframe!r}. "
            f"Expected one of {list(_TIMEFRAME_MAP)}"
        )


def _bars_to_df(bar_set, symbol: str) -> pd.DataFrame:
    """Convert an alpaca BarSet to an OHLCV DataFrame with UTC DatetimeIndex."""
    try:
        raw = bar_set[symbol]
    except (KeyError, TypeError):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    records    = []
    timestamps = []
    for b in raw:
        records.append({
            "open":   float(b.open),
            "high":   float(b.high),
            "low":    float(b.low),
            "close":  float(b.close),
            "volume": float(b.volume),
        })
        ts = b.timestamp
        if hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        timestamps.append(ts)

    df = pd.DataFrame(
        records,
        index=pd.DatetimeIndex(timestamps, tz="UTC"),
    )
    df.index.name = "timestamp"
    return df


def _with_retry(fn):
    """Retry up to 3 attempts with exponential backoff.

    HTTP 429 pauses for _RATE_LIMIT_WAIT seconds before retrying.
    All other exceptions use delays of 1 s then 2 s.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        for attempt, delay in enumerate((*_RETRY_DELAYS, None), start=1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                if getattr(exc, "status_code", None) == 429:
                    log.warning(
                        "Rate limited on attempt %d/3 — waiting %.0fs before retry",
                        attempt, _RATE_LIMIT_WAIT,
                    )
                    time.sleep(_RATE_LIMIT_WAIT)
                    continue
                if delay is None:
                    raise
                log.warning(
                    "Request failed on attempt %d/3: %s — retrying in %.0fs",
                    attempt, exc, delay,
                )
                time.sleep(delay)
    return wrapper


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@_with_retry
def get_historical_bars(
    symbol:    str,
    start:     datetime,
    end:       datetime,
    timeframe: str,
) -> pd.DataFrame:
    """Fetch OHLCV bars from the Alpaca historical data API.

    Returns a DataFrame with columns open, high, low, close, volume and a
    UTC DatetimeIndex.  Raises ValueError if no data is returned.
    """
    tf      = _parse_timeframe(timeframe)
    request = StockBarsRequest(
        symbol_or_symbols = symbol,
        timeframe         = tf,
        start             = start,
        end               = end,
        feed              = DataFeed.IEX,
    )
    bar_set = _get_stock_client().get_stock_bars(request)
    df      = _bars_to_df(bar_set, symbol)

    if df.empty:
        raise ValueError(
            f"No bars returned for {symbol!r} between {start} and {end} "
            f"(timeframe={timeframe})"
        )
    return df


@_with_retry
def get_latest_bar(symbol: str, timeframe: str = "1Day") -> pd.Series:
    """Fetch the most recently completed bar for symbol.

    Returns a Series with fields open, high, low, close, volume and the
    bar timestamp as the series name.
    """
    _parse_timeframe(timeframe)  # validate early; value used in future feed version
    request = StockLatestBarRequest(symbol_or_symbols=symbol, feed=DataFeed.IEX)
    result  = _get_stock_client().get_stock_latest_bar(request)

    try:
        bar = result[symbol]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"No latest bar returned for {symbol!r}") from exc

    ts = bar.timestamp
    if hasattr(ts, "tzinfo") and ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    return pd.Series(
        {
            "open":   float(bar.open),
            "high":   float(bar.high),
            "low":    float(bar.low),
            "close":  float(bar.close),
            "volume": float(bar.volume),
        },
        name=ts,
    )


def get_latest_bars(
    symbol:    str,
    n_bars:    int,
    timeframe: str = "1Day",
) -> pd.DataFrame:
    """Return the most recent n_bars OHLCV rows for symbol.

    Fetches historical data with a generous look-back window then trims to
    n_bars rows.  Inherits the retry behaviour of get_historical_bars.
    """
    end = datetime.now(tz=timezone.utc)

    if timeframe == "1Day":
        start = end - timedelta(days=int(n_bars * 2))
    elif timeframe == "1Hour":
        # ~6.5 trading hours/day; ×2 buffer
        start = end - timedelta(hours=int(n_bars / 6.5 * 24 * 2))
    else:  # 5Min — ~78 bars per trading day
        start = end - timedelta(days=max(1, int(n_bars / 78 * 2)))

    df = get_historical_bars(symbol, start, end, timeframe)
    return df.tail(n_bars)


def is_market_open() -> bool:
    """Return True if the US equity market is currently open."""
    return _get_alpaca_client().is_market_open()


def get_option_chain(symbol: str) -> list:
    """Return the option chain for symbol as a list of OptionContract."""
    return _get_alpaca_client().get_option_chain(symbol)

"""
core/scanner/universe.py
------------------------
Universe management for the nightly scanner.

Live universe fetch:
  fetch_universe() pulls the current S&P 500 and Nasdaq 100 constituent
  lists from Wikipedia using pandas.read_html() and deduplicates them.
  Ticker symbols are normalised from Wikipedia's dot-separated format
  (e.g. BRK.B) to Alpaca's slash format (e.g. BRK/B).

  If either Wikipedia page is unreachable the affected index falls back
  to the static SP500_NASDAQ100_UNIVERSE defined in config/settings.py.
  If both fail, the full static list is used.

Volume / price filter:
  get_tradeable() then filters the live universe by minimum average daily
  volume and minimum price using batched 20-day bar requests (50 tickers
  per call).

Exclusion counts are accumulated in self.exclusion_counts (dict[str, int])
after each get_tradeable() call for downstream reporting.
"""

from __future__ import annotations

import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    SCANNER_DATA_FEED,
    SCANNER_MIN_PRICE,
    SCANNER_MIN_VOLUME,
    SP500_NASDAQ100_UNIVERSE,
)

log = logging.getLogger(__name__)

# Wikipedia source URLs
_SP500_URL   = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_NDX100_URL  = "https://en.wikipedia.org/wiki/Nasdaq-100"

# Alpaca-compatible ticker normalisation: replace dots with slashes
# (e.g. BRK.B → BRK/B).  Other characters are left unchanged.
def _normalise(ticker: str) -> str:
    return ticker.replace(".", "/").strip().upper()


def fetch_universe() -> list[str]:
    """Return a deduplicated list of S&P 500 + Nasdaq 100 tickers.

    Pulls constituent lists from Wikipedia and normalises symbols to
    Alpaca format.  Falls back to the static SP500_NASDAQ100_UNIVERSE
    for any index whose Wikipedia page cannot be fetched.

    Returns
    -------
    list[str]
        Deduplicated, normalised tickers in the order S&P 500 first,
        then any Nasdaq 100 tickers not already in the S&P 500 list.
    """
    sp500  = _fetch_sp500()
    ndx100 = _fetch_nasdaq100()

    # Deduplicate: preserve order, Nasdaq additions after S&P 500
    seen: set[str] = set()
    combined: list[str] = []
    for ticker in sp500 + ndx100:
        if ticker not in seen:
            seen.add(ticker)
            combined.append(ticker)

    log.info(
        "fetch_universe: %d tickers (S&P 500=%d Nasdaq 100=%d combined=%d)",
        len(combined), len(sp500), len(ndx100), len(combined),
    )
    return combined


def _fetch_sp500() -> list[str]:
    """Fetch S&P 500 constituents from Wikipedia; return static fallback on failure."""
    try:
        tables = pd.read_html(_SP500_URL, attrs={"id": "constituents"})
        df     = tables[0]
        col    = _find_column(df, ["Symbol", "Ticker"])
        tickers = [_normalise(t) for t in df[col].dropna().tolist() if str(t).strip()]
        log.info("fetch_universe: fetched %d S&P 500 tickers from Wikipedia", len(tickers))
        return tickers
    except Exception as exc:
        log.warning(
            "fetch_universe: S&P 500 Wikipedia fetch failed (%s) — using static fallback",
            exc,
        )
        return [_normalise(t) for t in SP500_NASDAQ100_UNIVERSE]


def _fetch_nasdaq100() -> list[str]:
    """Fetch Nasdaq 100 constituents from Wikipedia; return empty list on failure."""
    try:
        tables = pd.read_html(_NDX100_URL)
        # Find the table that has a ticker/symbol column
        for df in tables:
            col = _find_column(df, ["Ticker", "Symbol"], required=False)
            if col is None:
                continue
            tickers = [_normalise(t) for t in df[col].dropna().tolist() if str(t).strip()]
            if len(tickers) >= 90:   # Nasdaq 100 always has ≥ 100 components
                log.info(
                    "fetch_universe: fetched %d Nasdaq 100 tickers from Wikipedia",
                    len(tickers),
                )
                return tickers
        raise ValueError("No Nasdaq 100 ticker table found on page")
    except Exception as exc:
        log.warning(
            "fetch_universe: Nasdaq 100 Wikipedia fetch failed (%s) — skipping",
            exc,
        )
        return []


def _find_column(df, candidates: list[str], *, required: bool = True) -> Optional[str]:
    """Return the first column name from *candidates* that exists in *df*.

    Case-insensitive match.  Raises ValueError if *required* and none found.
    """
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    if required:
        raise ValueError(
            f"None of {candidates} found in columns {list(df.columns)}"
        )
    return None


class UniverseManager:
    """Return a filtered list of tradeable tickers.

    On each get_tradeable() call:
      1. Fetch the live S&P 500 + Nasdaq 100 universe from Wikipedia
         (falls back to static list on network failure).
      2. Apply volume and price filters using recent Alpaca bar data.

    After get_tradeable() completes, self.exclusion_counts holds per-reason
    counts of every ticker that was removed from the universe.

    Parameters
    ----------
    client : optional AlpacaClient instance.  When None, all API filters are
             skipped (useful in tests with pre-screened universes).
    """

    def __init__(self, client=None) -> None:
        self._client         = client
        self.exclusion_counts: dict[str, int] = {}
        self._cached_universe: Optional[list[str]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tradeable(
        self,
        universe: list[str] | None = None,
        *,
        min_volume: float = SCANNER_MIN_VOLUME,
        min_price: float  = SCANNER_MIN_PRICE,
    ) -> list[str]:
        """Return tickers that pass all filters; populate self.exclusion_counts.

        If *universe* is provided it is used directly (skipping the live
        Wikipedia fetch).  Otherwise the live universe is fetched once and
        cached for the lifetime of this UniverseManager instance.
        """
        self.exclusion_counts = {}

        if universe is not None:
            candidates = list(dict.fromkeys(universe))
        else:
            candidates = self._get_universe()

        if self._client is not None:
            candidates, vp_counts = self._filter_volume_price(
                candidates, min_volume=min_volume, min_price=min_price
            )
            _add_counts(self.exclusion_counts, vp_counts)
            log.info("[SCANNER] Universe pipeline: %d after volume/price filter", len(candidates))

        log.info(
            "UniverseManager: %d tickers after filters (volume>=%.0f price>=%.0f)",
            len(candidates), min_volume, min_price,
        )
        return candidates

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_universe(self) -> list[str]:
        """Return cached live universe, fetching from Wikipedia if not yet cached."""
        if self._cached_universe is None:
            self._cached_universe = fetch_universe()
        return list(self._cached_universe)

    def _filter_volume_price(
        self,
        tickers: list[str],
        *,
        min_volume: float,
        min_price: float,
    ) -> tuple[list[str], dict[str, int]]:
        """Fetch 20-day bar history in batches; remove tickers below volume or price floors.

        Returns (passing_tickers, {reason: count}).
        """
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        end   = datetime.datetime.now(datetime.timezone.utc)
        start = end - datetime.timedelta(days=30)

        passing: list[str] = []
        n_low_vol   = 0
        n_low_price = 0

        chunk_size = 50
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i : i + chunk_size]
            try:
                resp = self._client._stocks.get_stock_bars(
                    StockBarsRequest(
                        symbol_or_symbols=chunk,
                        timeframe=TimeFrame.Day,
                        start=start,
                        end=end,
                        feed=SCANNER_DATA_FEED,
                    )
                )
            except Exception as exc:
                log.warning(
                    "Universe filter: batch fetch failed for chunk %s…: %s — keeping all",
                    chunk[:3], exc,
                )
                passing.extend(chunk)
                continue

            for ticker in chunk:
                try:
                    try:
                        bars = list(resp[ticker])
                    except KeyError:
                        bars = []
                    if len(bars) < 5:
                        log.debug("Universe drop %s: insufficient bars (%d)", ticker, len(bars))
                        continue
                    avg_volume = float(np.mean([b.volume for b in bars]))
                    avg_close  = float(np.mean([b.close  for b in bars]))
                    if avg_volume < min_volume:
                        log.debug(
                            "[SCANNER] %s excluded — low volume (%.0f < %.0f ADV)",
                            ticker, avg_volume, min_volume,
                        )
                        n_low_vol += 1
                        continue
                    if avg_close < min_price:
                        log.debug(
                            "[SCANNER] %s excluded — price below floor ($%.2f < $%.2f)",
                            ticker, avg_close, min_price,
                        )
                        n_low_price += 1
                        continue
                    passing.append(ticker)
                except Exception as exc:
                    log.warning("Universe filter error for %s (kept): %s", ticker, exc)
                    passing.append(ticker)

        return passing, {"low_volume": n_low_vol, "low_price": n_low_price}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    """Merge source counts into target in-place."""
    for k, v in source.items():
        target[k] = target.get(k, 0) + v

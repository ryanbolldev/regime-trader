"""
core/scanner/universe.py
------------------------
Universe management for the nightly scanner.

Filters SP500_NASDAQ100_UNIVERSE by minimum average daily volume and
minimum price using batched 20-day bar requests (50 tickers per call).

Exclusion counts are accumulated in self.exclusion_counts (dict[str, int])
after each get_tradeable() call for downstream reporting.
"""

from __future__ import annotations

import datetime
import logging

from config.settings import (
    SCANNER_DATA_FEED,
    SCANNER_MIN_PRICE,
    SCANNER_MIN_VOLUME,
    SP500_NASDAQ100_UNIVERSE,
)

log = logging.getLogger(__name__)


class UniverseManager:
    """Return a filtered list of tradeable tickers.

    After get_tradeable() completes, self.exclusion_counts holds per-reason
    counts of every ticker that was removed from the universe.

    Parameters
    ----------
    client : optional AlpacaClient instance.  When None, all API filters are
             skipped (useful in tests with pre-screened universes).
    """

    def __init__(self, client=None) -> None:
        self._client        = client
        self.exclusion_counts: dict[str, int] = {}

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
        """Return tickers that pass all filters; populate self.exclusion_counts."""
        self.exclusion_counts = {}
        candidates = list(dict.fromkeys(universe or SP500_NASDAQ100_UNIVERSE))

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
        import numpy as np
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

"""
core/scanner/universe.py
------------------------
Universe management for the nightly scanner.

Filters SP500_NASDAQ100_UNIVERSE by minimum average daily volume, minimum
price, and a configurable earnings-proximity buffer.

Earnings filter hardening (fail-safe):
  - Both estimated_date and confirmed_date are checked; if EITHER is within
    SCANNER_EARNINGS_BUFFER_DAYS, the ticker is excluded.
  - If the earnings API call fails, ALL tickers in that batch are excluded
    with reason "earnings_data_unavailable" (fail-safe, not fail-open).
  - Every exclusion is logged with the specific date and reason.

Exclusion counts are accumulated in self.exclusion_counts (dict[str, int])
after each get_tradeable() call for downstream reporting.
"""

from __future__ import annotations

import datetime
import logging
from typing import Optional

from config.settings import (
    SCANNER_DATA_FEED,
    SCANNER_EARNINGS_BUFFER_DAYS,
    SCANNER_MIN_PRICE,
    SCANNER_MIN_VOLUME,
    SP500_NASDAQ100_UNIVERSE,
)

log = logging.getLogger(__name__)

# Maps raw API field names to human-readable labels for log messages
_DATE_FIELDS = [
    ("estimated_date", "estimated"),
    ("confirmed_date",  "confirmed"),
    ("date",            "confirmed"),   # fallback field name used by some endpoints
]


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
        earnings_buffer_days: int = SCANNER_EARNINGS_BUFFER_DAYS,
    ) -> list[str]:
        """Return tickers that pass all filters; populate self.exclusion_counts."""
        self.exclusion_counts = {}
        candidates = list(dict.fromkeys(universe or SP500_NASDAQ100_UNIVERSE))

        if self._client is not None:
            candidates, vp_counts = self._filter_volume_price(
                candidates, min_volume=min_volume, min_price=min_price
            )
            _add_counts(self.exclusion_counts, vp_counts)

            candidates, earn_counts = self._filter_earnings(
                candidates, buffer_days=earnings_buffer_days
            )
            _add_counts(self.exclusion_counts, earn_counts)

        log.info(
            "UniverseManager: %d tickers after filters "
            "(volume>=%.0f price>=%.0f earnings_buf=%dd)",
            len(candidates), min_volume, min_price, earnings_buffer_days,
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
                    bars = list(resp[ticker]) if ticker in resp else []
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

    def _filter_earnings(
        self,
        tickers: list[str],
        *,
        buffer_days: int,
    ) -> tuple[list[str], dict[str, int]]:
        """Remove tickers with an earnings announcement within buffer_days.

        Checks both estimated_date and confirmed_date fields; excludes the
        ticker if EITHER is within buffer_days of today.

        Returns (passing_tickers, {reason: count}).

        FAIL-SAFE: if the earnings API call fails, ALL tickers are excluded
        with reason "earnings_data_unavailable".
        """
        if buffer_days <= 0:
            return tickers, {}

        today     = datetime.date.today()
        cutoff_lo = today - datetime.timedelta(days=buffer_days)
        cutoff_hi = today + datetime.timedelta(days=buffer_days)

        # _fetch_near_earnings returns None on API failure → fail safe
        earnings_data = self._fetch_near_earnings(cutoff_lo, cutoff_hi)

        if earnings_data is None:
            # API failed: cannot determine earnings status — exclude all
            log.warning(
                "[SCANNER] Earnings API unavailable — excluding all %d tickers (fail-safe)",
                len(tickers),
            )
            for ticker in tickers:
                log.info("[SCANNER] %s excluded — earnings data unavailable", ticker)
            return [], {"earnings_data_unavailable": len(tickers)}

        passing:      list[str] = []
        n_within    = 0
        n_unavail   = 0

        for ticker in tickers:
            ticker_dates = earnings_data.get(ticker, [])  # list of (date_str, label)
            excluded     = False

            for date_str, label in ticker_dates:
                try:
                    ann_date  = datetime.date.fromisoformat(date_str)
                    days_away = (ann_date - today).days
                    if abs(days_away) <= buffer_days:
                        log.info(
                            "[SCANNER] %s excluded — earnings in %d days (%s %s)",
                            ticker, abs(days_away), date_str, label,
                        )
                        n_within += 1
                        excluded  = True
                        break
                except Exception:
                    pass

            if not excluded:
                passing.append(ticker)

        counts: dict[str, int] = {}
        if n_within:
            counts["earnings_within_7_days"] = n_within
        if n_unavail:
            counts["earnings_data_unavailable"] = n_unavail
        return passing, counts

    def _fetch_near_earnings(
        self,
        date_lo: datetime.date,
        date_hi: datetime.date,
    ) -> Optional[dict[str, list[tuple[str, str]]]]:
        """Fetch corporate-actions announcements for the date window.

        Returns a dict mapping symbol -> list of (date_str, label) pairs,
        where label is "estimated" or "confirmed".

        Returns None on any failure (network error, non-2xx response) so the
        caller can apply fail-safe exclusion.
        """
        try:
            url  = f"{self._client._base_url}/corporate-actions/announcements"
            resp = self._client._session.get(
                url,
                params={
                    "ca_types": "dividend,earnings",
                    "since":    date_lo.isoformat(),
                    "until":    date_hi.isoformat(),
                },
                timeout=10,
            )
            if not resp.ok:
                log.warning(
                    "Earnings fetch HTTP %d — applying fail-safe exclusion",
                    resp.status_code,
                )
                return None

            data  = resp.json()
            items = data if isinstance(data, list) else data.get("announcements", [])

            result: dict[str, list[tuple[str, str]]] = {}
            for item in items:
                symbol = item.get("symbol", "")
                if not symbol:
                    continue
                dates: list[tuple[str, str]] = []
                for field, label in _DATE_FIELDS:
                    d = item.get(field)
                    if d:
                        dates.append((str(d), label))
                if dates:
                    result.setdefault(symbol, []).extend(dates)

            return result

        except Exception as exc:
            log.warning("Earnings fetch failed — applying fail-safe exclusion: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    """Merge source counts into target in-place."""
    for k, v in source.items():
        target[k] = target.get(k, 0) + v

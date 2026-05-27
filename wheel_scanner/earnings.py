"""
wheel_scanner/earnings.py
--------------------------
Earnings calendar lookup and exclusion logic.

Data source: yfinance (optional dependency).

If yfinance is not installed, all functions return None / False and log a
one-time warning. The scanner treats missing earnings data conservatively:
  - earnings_within_14d defaults to False (ticker not excluded)
  - days_to_earnings defaults to None (no "near_earnings" flag added)

Install yfinance to enable earnings filtering:
    pip install yfinance>=0.2.40
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

log = logging.getLogger(__name__)

# Guard: warn once if yfinance is absent, not on every call
_yf_warned = False


def _get_yfinance():
    """Return the yfinance module or None if not installed."""
    global _yf_warned
    try:
        import yfinance as yf
        return yf
    except ImportError:
        if not _yf_warned:
            log.warning(
                "yfinance not installed — earnings filtering disabled. "
                "Run: pip install yfinance>=0.2.40"
            )
            _yf_warned = True
        return None


def get_next_earnings_date(ticker: str) -> Optional[date]:
    """Return the next confirmed earnings date for ticker, or None if unavailable.

    Uses yfinance Ticker.calendar which returns a dict containing
    'Earnings Date' as a list of timestamps.
    """
    yf = _get_yfinance()
    if yf is None:
        return None

    try:
        info = yf.Ticker(ticker)
        cal = info.calendar

        if cal is None:
            return None

        # yfinance returns calendar as a dict with 'Earnings Date' key
        # containing a list of Timestamp objects
        if isinstance(cal, dict):
            earnings_dates = cal.get("Earnings Date", [])
            if not earnings_dates:
                return None
            # Take the nearest upcoming date
            today = date.today()
            future_dates = []
            for d in earnings_dates:
                try:
                    earnings_date = d.date() if hasattr(d, "date") else date.fromisoformat(str(d)[:10])
                    if earnings_date >= today:
                        future_dates.append(earnings_date)
                except Exception:
                    continue
            return min(future_dates) if future_dates else None

        # Fallback: handle DataFrame format from older yfinance versions
        if hasattr(cal, "columns"):
            if "Earnings Date" in cal.columns:
                for val in cal["Earnings Date"]:
                    try:
                        d = val.date() if hasattr(val, "date") else date.fromisoformat(str(val)[:10])
                        if d >= date.today():
                            return d
                    except Exception:
                        continue

        return None

    except Exception as exc:
        log.debug("get_next_earnings_date [%s]: %s", ticker, exc)
        return None


def days_until_earnings(ticker: str) -> Optional[int]:
    """Return the number of calendar days until the next earnings date, or None."""
    next_date = get_next_earnings_date(ticker)
    if next_date is None:
        return None
    return (next_date - date.today()).days


def is_earnings_within(ticker: str, days: int) -> bool:
    """Return True if earnings are within the next *days* calendar days.

    Returns False (not excluded) when earnings data is unavailable.
    """
    d = days_until_earnings(ticker)
    if d is None:
        return False
    return 0 <= d <= days


def batch_earnings_filter(
    tickers: list[str],
    exclusion_days: int = 14,
) -> tuple[list[str], list[str]]:
    """Split tickers into (safe, excluded_near_earnings).

    safe              — earnings not within exclusion_days
    excluded          — earnings confirmed within exclusion_days

    Tickers with unavailable earnings data are treated as safe.
    """
    yf = _get_yfinance()
    if yf is None:
        return list(tickers), []

    safe: list[str] = []
    excluded: list[str] = []

    for ticker in tickers:
        if is_earnings_within(ticker, exclusion_days):
            log.debug(
                "Earnings filter: excluding %s (earnings within %d days)",
                ticker, exclusion_days,
            )
            excluded.append(ticker)
        else:
            safe.append(ticker)

    log.info(
        "Earnings filter: %d safe, %d excluded (within %d days)",
        len(safe), len(excluded), exclusion_days,
    )
    return safe, excluded

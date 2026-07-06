"""
wheel_scanner/filters.py
-------------------------
Universe construction and multi-stage filtering for the wheel scanner.

Filtering pipeline (applied in order):
  1. Universe fetch       — S&P MidCap 400 from Wikipedia + static fallback
  2. Price / ADV          — $15–$150 price, ≥500k average daily shares (Alpaca)
  3. Market cap           — $2B–$10B (yfinance; optional — skipped if not installed)
  4. Security type        — exclude ETFs, REITs, SPACs, Chinese ADRs (yfinance; optional)
  5. Earnings proximity   — exclude if earnings within 14 days (yfinance; optional)
  6. Options availability — must have ≥2 active expiry cycles (Alpaca)

Each stage logs how many tickers were excluded and why.
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Universe constants
# ---------------------------------------------------------------------------

_SP400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"

# Static fallback — well-known mid-cap names with liquid options.
# Used when Wikipedia is unreachable. NOT a curated watchlist — it is only
# the seed from which the filter pipeline selects candidates.
_FALLBACK_SP400: list[str] = [
    "AXON", "DECK", "SAIA", "RGEN", "PODD", "IDCC", "GLOB", "EPAM",
    "PAYC", "BURL", "GTLB", "RH", "CRUS", "AAON", "HALO", "MATX",
    "LSTR", "CNX", "EXLS", "EXPO", "FIVN", "GH", "ICUI", "INSP",
    "LFUS", "NARI", "PLMR", "PRGO", "RRC", "SLGN", "TMHC", "TRNO",
    "UFP", "VCYT", "WERN", "WSFS", "WTFC", "AMG", "AYI", "BJ",
    "CATY", "CWK", "DLB", "ELF", "ENSG", "FLR", "FRPT", "GEO",
    "HBI", "HCC", "HRI", "IBP", "IBTX", "ITRI", "JBT", "KBH",
    "KFY", "KMPR", "LCII", "MAN", "MCY", "MDU", "MHO", "MMS",
    "MOG/A", "MSA", "NBTB", "NEU", "NFG", "NHI", "NRC", "NSIT",
    "OGE", "OMCL", "OUT", "PAR", "PATK", "PCVX", "PEB", "PINC",
    "PJT", "POWL", "PRDO", "PSMT", "PUMP", "PVH", "RDN", "RLGY",
    "ROAD", "ROCK", "RUSHA", "RXRX", "SCI", "SHOO", "SIG", "SMTC",
    "SPNT", "SUPN", "SWX", "TDC", "TEX", "TFIN", "TGNA", "THR",
    "TNC", "TRMK", "TRN", "TRUP", "TTEC", "TWI", "UFPI", "UFPT",
    "UNF", "UNVR", "VCEL", "VGR", "VITL", "VPG", "WCC", "WK",
]

# ---------------------------------------------------------------------------
# Filter parameters (matching spec)
# ---------------------------------------------------------------------------

MIN_PRICE_USD       = 15.0
MAX_PRICE_USD       = 150.0
MIN_ADV_SHARES      = 500_000
MIN_MARKET_CAP_B    = 2.0
MAX_MARKET_CAP_B    = 10.0
MIN_OPTION_CYCLES   = 2
EARNINGS_EXCL_DAYS  = 14

# Security type exclusion values from yfinance quoteType
_EXCLUDED_QUOTE_TYPES = {"ETF", "MUTUALFUND", "CURRENCY", "FUTURE", "INDEX"}
_REIT_INDUSTRIES       = {"REIT", "Real Estate Investment Trust"}

# Chinese ADR exchange identifiers from yfinance
_CHINESE_EXCHANGES = {"SHZ", "SHH", "HKG"}
_ADR_COUNTRY       = "China"


# ---------------------------------------------------------------------------
# Universe fetch
# ---------------------------------------------------------------------------

def fetch_sp400_universe() -> list[str]:
    """Fetch S&P MidCap 400 constituents from Wikipedia.

    Falls back to _FALLBACK_SP400 on any fetch error.
    Returns a deduplicated list of normalised ticker symbols.
    """
    try:
        tables = pd.read_html(_SP400_URL)
        for df in tables:
            col = _find_column(df, ["Ticker", "Symbol"], required=False)
            if col is None:
                continue
            tickers = [
                _normalise(t) for t in df[col].dropna().tolist()
                if str(t).strip() and str(t) != "nan"
            ]
            if len(tickers) >= 300:
                log.info(
                    "fetch_sp400_universe: %d tickers from Wikipedia S&P 400",
                    len(tickers),
                )
                return _dedup(tickers)
        raise ValueError("No S&P 400 ticker table found on Wikipedia page")
    except Exception as exc:
        log.warning(
            "fetch_sp400_universe: Wikipedia fetch failed (%s) — using static fallback (%d tickers)",
            exc, len(_FALLBACK_SP400),
        )
        return _dedup([_normalise(t) for t in _FALLBACK_SP400])


# ---------------------------------------------------------------------------
# Stage 1: Price and average daily volume (Alpaca bars)
# ---------------------------------------------------------------------------

def filter_price_and_volume(
    tickers:      list[str],
    stock_client,
    min_price:    float = MIN_PRICE_USD,
    max_price:    float = MAX_PRICE_USD,
    min_adv:      float = MIN_ADV_SHARES,
    lookback_days: int  = 20,
) -> tuple[list[str], dict[str, int]]:
    """Filter tickers by price range and average daily volume.

    Uses Alpaca StockHistoricalDataClient in 50-ticker batches.
    Returns (passing_tickers, {reason: count}).
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    end   = datetime.datetime.now(datetime.timezone.utc)
    start = end - datetime.timedelta(days=lookback_days + 14)  # buffer for weekends

    passing:     list[str] = []
    n_low_price  = 0
    n_high_price = 0
    n_low_vol    = 0

    chunk_size = 50
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        try:
            resp = stock_client.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols = chunk,
                    timeframe         = TimeFrame.Day,
                    start             = start,
                    end               = end,
                )
            )
        except Exception as exc:
            log.warning(
                "filter_price_and_volume: batch fetch failed (%s) — keeping chunk %s…",
                exc, chunk[:3],
            )
            passing.extend(chunk)
            continue

        for ticker in chunk:
            try:
                try:
                    bars = list(resp[ticker])
                except (KeyError, TypeError):
                    bars = []

                if len(bars) < 5:
                    log.debug("filter_price_and_volume: drop %s (only %d bars)", ticker, len(bars))
                    continue

                avg_close  = float(np.mean([b.close  for b in bars]))
                avg_volume = float(np.mean([b.volume for b in bars]))

                if avg_close < min_price:
                    log.debug("filter: %s price $%.2f < $%.2f", ticker, avg_close, min_price)
                    n_low_price += 1
                    continue
                if avg_close > max_price:
                    log.debug("filter: %s price $%.2f > $%.2f", ticker, avg_close, max_price)
                    n_high_price += 1
                    continue
                if avg_volume < min_adv:
                    log.debug("filter: %s ADV %.0f < %.0f", ticker, avg_volume, min_adv)
                    n_low_vol += 1
                    continue

                passing.append(ticker)

            except Exception as exc:
                log.warning("filter_price_and_volume: error for %s (kept): %s", ticker, exc)
                passing.append(ticker)

    counts = {"low_price": n_low_price, "high_price": n_high_price, "low_volume": n_low_vol}
    log.info(
        "filter_price_and_volume: %d/%d passed (low_price=%d, high_price=%d, low_vol=%d)",
        len(passing), len(tickers), n_low_price, n_high_price, n_low_vol,
    )
    return passing, counts


# ---------------------------------------------------------------------------
# Stage 2: Market cap (yfinance — optional)
# ---------------------------------------------------------------------------

def filter_market_cap(
    tickers:   list[str],
    min_cap_B: float = MIN_MARKET_CAP_B,
    max_cap_B: float = MAX_MARKET_CAP_B,
) -> tuple[list[str], dict[str, int]]:
    """Filter by market capitalisation using yfinance.

    If yfinance is not installed, returns all tickers unfiltered with a warning.

    TODO: Replace with a data source that supports bulk market cap queries to
    reduce runtime (yfinance requires one HTTP call per ticker).
    """
    yf = _get_yfinance()
    if yf is None:
        log.warning(
            "filter_market_cap: yfinance not installed — skipping market cap filter. "
            "Run: pip install yfinance>=0.2.40"
        )
        return list(tickers), {}

    passing:   list[str] = []
    n_too_small = 0
    n_too_large = 0
    n_unknown   = 0

    for ticker in tickers:
        try:
            info   = yf.Ticker(ticker).info
            cap    = info.get("marketCap") or info.get("market_cap")
            if cap is None:
                log.debug("filter_market_cap: %s — market cap unavailable (kept)", ticker)
                n_unknown += 1
                passing.append(ticker)
                continue

            cap_B = cap / 1e9
            if cap_B < min_cap_B:
                log.debug("filter_market_cap: %s cap $%.1fB < $%.1fB", ticker, cap_B, min_cap_B)
                n_too_small += 1
            elif cap_B > max_cap_B:
                log.debug("filter_market_cap: %s cap $%.1fB > $%.1fB", ticker, cap_B, max_cap_B)
                n_too_large += 1
            else:
                passing.append(ticker)

            time.sleep(0.2)   # avoid yfinance rate limiting

        except Exception as exc:
            log.debug("filter_market_cap: %s error (kept): %s", ticker, exc)
            n_unknown += 1
            passing.append(ticker)

    counts = {
        "market_cap_too_small": n_too_small,
        "market_cap_too_large": n_too_large,
        "market_cap_unknown":   n_unknown,
    }
    log.info(
        "filter_market_cap: %d/%d passed (too_small=%d, too_large=%d, unknown=%d)",
        len(passing), len(tickers), n_too_small, n_too_large, n_unknown,
    )
    return passing, counts


# ---------------------------------------------------------------------------
# Stage 3: Security type (yfinance — optional)
# ---------------------------------------------------------------------------

def filter_security_type(
    tickers: list[str],
) -> tuple[list[str], dict[str, int]]:
    """Exclude ETFs, REITs, SPACs, and Chinese ADRs using yfinance.

    If yfinance is not installed, returns all tickers unfiltered with a warning.
    """
    yf = _get_yfinance()
    if yf is None:
        log.warning(
            "filter_security_type: yfinance not installed — "
            "ETF/REIT/SPAC/ADR exclusion disabled."
        )
        return list(tickers), {}

    passing:    list[str] = []
    n_etf       = 0
    n_reit      = 0
    n_spac      = 0
    n_china_adr = 0

    for ticker in tickers:
        try:
            info       = yf.Ticker(ticker).info
            quote_type = (info.get("quoteType") or "").upper()
            sector     = info.get("sector") or ""
            industry   = info.get("industry") or ""
            country    = info.get("country") or ""
            long_name  = (info.get("longName") or "").upper()
            exchange   = (info.get("exchange") or "").upper()

            if quote_type in _EXCLUDED_QUOTE_TYPES:
                log.debug("filter_security_type: %s excluded (quoteType=%s)", ticker, quote_type)
                n_etf += 1
                time.sleep(0.1)
                continue

            if sector == "Real Estate" or any(r in industry for r in _REIT_INDUSTRIES):
                log.debug("filter_security_type: %s excluded (REIT)", ticker)
                n_reit += 1
                time.sleep(0.1)
                continue

            # SPAC detection: company name contains "Acquisition" or sector is blank
            # and price is low with tiny market cap — heuristic, not perfect
            if "ACQUISITION" in long_name and quote_type == "EQUITY":
                log.debug("filter_security_type: %s excluded (likely SPAC)", ticker)
                n_spac += 1
                time.sleep(0.1)
                continue

            if country == _ADR_COUNTRY or exchange in _CHINESE_EXCHANGES:
                log.debug("filter_security_type: %s excluded (Chinese ADR/stock)", ticker)
                n_china_adr += 1
                time.sleep(0.1)
                continue

            passing.append(ticker)
            time.sleep(0.1)

        except Exception as exc:
            log.debug("filter_security_type: %s error (kept): %s", ticker, exc)
            passing.append(ticker)

    counts = {
        "excluded_etf":       n_etf,
        "excluded_reit":      n_reit,
        "excluded_spac":      n_spac,
        "excluded_china_adr": n_china_adr,
    }
    log.info(
        "filter_security_type: %d/%d passed (etf=%d, reit=%d, spac=%d, china=%d)",
        len(passing), len(tickers), n_etf, n_reit, n_spac, n_china_adr,
    )
    return passing, counts


# ---------------------------------------------------------------------------
# Stage 4: Options availability (Alpaca)
# ---------------------------------------------------------------------------

def filter_options_available(
    tickers:        list[str],
    options_client,
    min_cycles:     int = MIN_OPTION_CYCLES,
) -> tuple[list[str], dict[str, int]]:
    """Exclude tickers without at least min_cycles active expiry cycles.

    Calls count_expiry_cycles() from the active provider for each ticker.
    Includes a small inter-call delay to stay within rate limits.
    """
    from wheel_scanner.options_provider import count_expiry_cycles

    passing:   list[str] = []
    n_no_options = 0

    for ticker in tickers:
        cycles = count_expiry_cycles(ticker, options_client)
        if cycles < min_cycles:
            log.debug(
                "filter_options_available: %s excluded (%d cycles < %d)",
                ticker, cycles, min_cycles,
            )
            n_no_options += 1
        else:
            passing.append(ticker)
        time.sleep(_inter_call_delay())

    counts = {"no_listed_options": n_no_options}
    log.info(
        "filter_options_available: %d/%d passed (no_options=%d)",
        len(passing), len(tickers), n_no_options,
    )
    return passing, counts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_yf_warned = False


def _get_yfinance():
    global _yf_warned
    try:
        import yfinance as yf
        return yf
    except ImportError:
        if not _yf_warned:
            log.warning(
                "yfinance not installed. Market cap, security type, and earnings "
                "filters will be skipped. Install with: pip install yfinance>=0.2.40"
            )
            _yf_warned = True
        return None


def _normalise(ticker: str) -> str:
    """Convert dot-separated tickers to Alpaca slash format (BRK.B → BRK/B)."""
    return ticker.replace(".", "/").strip().upper()


def _dedup(tickers: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def _find_column(df: pd.DataFrame, candidates: list[str], *, required: bool = True) -> Optional[str]:
    """Case-insensitive column name lookup."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    if required:
        raise ValueError(f"None of {candidates} found in columns {list(df.columns)}")
    return None


def _inter_call_delay() -> float:
    """Delay between Alpaca options API calls to stay within rate limits."""
    return 0.5

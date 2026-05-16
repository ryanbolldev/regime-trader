"""
core/scanner/options_enricher.py
---------------------------------
Enriches TickerResult objects with a combined volatility rank estimate and
ATM bid-ask spread (options liquidity flag).

Vol rank is computed from three components that require no options API access:
  1. Realized vol percentile rank  (50% weight)
  2. VIX/VIXY percentile rank      (30% weight)
  3. Vol term structure score       (20% weight)

Available weights are redistributed across whichever components have data,
so the estimate degrades gracefully when VIXY or OHLCV is unavailable.

The combined estimate is stored on TickerResult as both `vol_rank` and
`iv_rank` (backward compat for scorer) so downstream code needs no changes.
`vol_estimated = True` signals the reporter to show a tilde prefix.

Public interface:
  compute_vol_estimate(close, vix_series, *, ...) -> tuple[float|None, dict]
  OptionsEnricher(client, ..., vix_series, ohlcv_map).enrich(results) -> list[TickerResult]
"""

from __future__ import annotations

import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    SCANNER_MAX_IV_RANK,
    SCANNER_MAX_WORKERS,
    SCANNER_OPTIONS_SPREAD_MAX,
    SCANNER_VIX_LOOKBACK,
    SCANNER_VOL_LOOKBACK,
    SCANNER_VOL_REALIZED_WEIGHT,
    SCANNER_VOL_TERM_WEIGHT,
    SCANNER_VOL_VIX_WEIGHT,
    SCANNER_VOL_WINDOW_LONG,
    SCANNER_VOL_WINDOW_MID,
    SCANNER_VOL_WINDOW_SHORT,
)
from core.scanner.batch_trainer import TickerResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vol estimator — module-level functions (public)
# ---------------------------------------------------------------------------

def _realized_vol_rank(
    close: pd.Series,
    vol_window: int,
    lookback: int,
) -> float | None:
    """Percentile rank of current realized vol vs its own lookback history.

    Returns None when close is too short to compute a meaningful rank.
    """
    if len(close) < vol_window + lookback:
        return None
    log_ret = np.log(close / close.shift(1)).dropna()
    current_vol = log_ret.iloc[-vol_window:].std() * np.sqrt(252)
    historical_vols = [
        log_ret.iloc[i : i + vol_window].std() * np.sqrt(252)
        for i in range(len(log_ret) - vol_window - lookback, len(log_ret) - vol_window)
    ]
    if not historical_vols or np.std(historical_vols) == 0:
        return None
    rank = sum(v < current_vol for v in historical_vols) / len(historical_vols) * 100
    return round(rank, 1)


def _vix_percentile_rank(
    vix_series: pd.Series | None,
    lookback: int,
) -> float | None:
    """Percentile rank of current VIXY close vs its lookback window.

    Returns None when vix_series is None or has fewer than 2 bars.
    """
    if vix_series is None or len(vix_series) < 2:
        return None
    series = vix_series.tail(lookback)
    current = series.iloc[-1]
    historical = series.iloc[:-1]
    rank = sum(v < current for v in historical) / len(historical) * 100
    return round(rank, 1)


def _vol_term_structure_score(
    close: pd.Series,
    short_window: int,
    long_window: int,
) -> float | None:
    """Score 0–100 based on short-vol / long-vol ratio.

    Ratio > 1 (short-term vol elevated vs long-term) scores toward 100,
    signalling stress or event risk. Ratio < 1 scores toward 0.
    Ratio is clamped to [0.5, 2.0] before normalising to avoid extremes
    from thin data distorting the score.
    """
    if len(close) < long_window + 10:
        return None
    log_ret = np.log(close / close.shift(1)).dropna()
    short_vol = log_ret.iloc[-short_window:].std() * np.sqrt(252)
    long_vol  = log_ret.iloc[-long_window:].std()  * np.sqrt(252)
    if long_vol == 0:
        return None
    ratio = short_vol / long_vol
    ratio_clamped = max(0.5, min(2.0, ratio))
    score = (ratio_clamped - 0.5) / 1.5 * 100
    return round(score, 1)


def compute_vol_estimate(
    close: pd.Series,
    vix_series: pd.Series | None,
    *,
    realized_weight: float,
    vix_weight: float,
    term_weight: float,
    vol_window_short: int,
    vol_window_mid: int,
    vol_window_long: int,
    vol_lookback: int,
    vix_lookback: int,
) -> tuple[float | None, dict]:
    """Combine three vol components into a single 0–100 percentile rank estimate.

    Returns (estimate, components_dict).  estimate is None when all components
    fail.  Available weights are redistributed so the estimate is always fully
    weighted on the data that is present.
    """
    realized = _realized_vol_rank(close, vol_window_mid, vol_lookback)
    vix_rank  = _vix_percentile_rank(vix_series, vix_lookback)
    term      = _vol_term_structure_score(close, vol_window_short, vol_window_long)

    components = {
        'realized': realized,
        'vix':      vix_rank,
        'term':     term,
    }

    available = {k: v for k, v in components.items() if v is not None}
    if not available:
        return None, components

    weight_map = {
        'realized': realized_weight,
        'vix':      vix_weight,
        'term':     term_weight,
    }
    total_weight = sum(weight_map[k] for k in available)
    estimate = sum(
        available[k] * weight_map[k] / total_weight
        for k in available
    )
    return round(estimate, 1), components


# ---------------------------------------------------------------------------
# OptionsEnricher
# ---------------------------------------------------------------------------

class OptionsEnricher:
    """Attach vol rank estimate and options liquidity flags to TickerResult objects.

    Parameters
    ----------
    client      : AlpacaClient instance (used for ATM spread fetch only).
    max_workers : thread-pool size for parallel enrichment.
    spread_max  : maximum ATM bid-ask spread ($) before flagging illiquid.
    vix_series  : VIXY daily close series for VIX component (optional).
    ohlcv_map   : {ticker: DataFrame} from fetch_ohlcv — used for vol estimate.
    """

    def __init__(
        self,
        client,
        max_workers: int = SCANNER_MAX_WORKERS,
        spread_max:  float = SCANNER_OPTIONS_SPREAD_MAX,
        vix_series:  Optional[pd.Series] = None,
        ohlcv_map:   Optional[dict] = None,
    ) -> None:
        self._client      = client
        self._max_workers = max_workers
        self._spread_max  = spread_max
        self._vix_series  = vix_series
        self._ohlcv_map   = ohlcv_map or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich(self, results: list[TickerResult]) -> list[TickerResult]:
        """Mutate *results* in-place with vol_rank, iv_rank, spread, and flags.

        Failed-fit tickers are skipped.  Errors for any single ticker are
        swallowed; the ticker proceeds with vol_rank = None.
        """
        to_enrich = [r for r in results if not r.fit_failed]

        futures = {}
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            for result in to_enrich:
                fut = pool.submit(self._enrich_one, result.ticker)
                futures[fut] = result

            for fut in as_completed(futures):
                result = futures[fut]
                try:
                    (
                        iv_rank, spread, low_liq, high_iv,
                        vol_rank, vol_rank_available, vol_components, vol_estimated,
                    ) = fut.result()
                    result.iv_rank               = iv_rank
                    result.spread                = spread
                    result.low_liquidity_options = low_liq
                    result.high_iv_event_risk    = high_iv
                    result.vol_rank              = vol_rank
                    result.vol_rank_available    = vol_rank_available
                    result.vol_components        = vol_components
                    result.vol_estimated         = vol_estimated
                except Exception as exc:
                    log.warning("OptionsEnricher: error for %s (skipped): %s", result.ticker, exc)

        log.info(
            "OptionsEnricher: enriched %d tickers (%d low-liquidity, %d high-vol)",
            len(to_enrich),
            sum(1 for r in to_enrich if r.low_liquidity_options),
            sum(1 for r in to_enrich if r.high_iv_event_risk),
        )
        return results

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _enrich_one(self, ticker: str) -> tuple:
        """Return enrichment tuple for one ticker.

        Returns:
            (iv_rank, spread, low_liquidity_options, high_iv_event_risk,
             vol_rank, vol_rank_available, vol_components, vol_estimated)
        """
        spread  = None
        low_liq = False
        high_iv = False
        vol_components: dict = {}

        # ── Vol estimate (replaces options-API iv_rank) ────────────────
        close: Optional[pd.Series] = None
        df = self._ohlcv_map.get(ticker)
        if df is not None and 'close' in df.columns and len(df) > 0:
            close = df['close']

        vol_rank: Optional[float] = None
        if close is not None:
            vol_rank, vol_components = compute_vol_estimate(
                close,
                self._vix_series,
                realized_weight  = SCANNER_VOL_REALIZED_WEIGHT,
                vix_weight       = SCANNER_VOL_VIX_WEIGHT,
                term_weight      = SCANNER_VOL_TERM_WEIGHT,
                vol_window_short = SCANNER_VOL_WINDOW_SHORT,
                vol_window_mid   = SCANNER_VOL_WINDOW_MID,
                vol_window_long  = SCANNER_VOL_WINDOW_LONG,
                vol_lookback     = SCANNER_VOL_LOOKBACK,
                vix_lookback     = SCANNER_VIX_LOOKBACK,
            )

        # iv_rank is set to vol_rank for backward compatibility with scorer
        iv_rank           = vol_rank
        vol_rank_available = vol_rank is not None
        vol_estimated     = True   # always an estimate; never true market IV

        if vol_rank is not None and vol_rank > SCANNER_MAX_IV_RANK:
            high_iv = True
            log.info(
                "[SCANNER] %s flagged high_iv_event_risk — vol rank %.1f > %d",
                ticker, vol_rank, SCANNER_MAX_IV_RANK,
            )

        # ── ATM bid-ask spread (options chain — liquidity flag only) ───
        try:
            chain = self._client.get_option_chain(ticker)
            today = datetime.date.today()
            atm_contracts = [
                c for c in chain
                if (c.bid is not None and c.ask is not None)
                and 30 <= (_safe_date_days(c.expiration, today)) <= 45
            ]
            if atm_contracts:
                spreads = [c.ask - c.bid for c in atm_contracts]
                spread  = float(min(spreads))
                low_liq = spread > self._spread_max
        except Exception as exc:
            log.debug("OptionsEnricher [%s]: spread fetch failed: %s", ticker, exc)

        log.debug(
            "OptionsEnricher [%s]: vol_rank=%s spread=%s low_liq=%s high_iv=%s components=%s",
            ticker, vol_rank, spread, low_liq, high_iv, vol_components,
        )
        return (
            iv_rank, spread, low_liq, high_iv,
            vol_rank, vol_rank_available, vol_components, vol_estimated,
        )


def _safe_date_days(expiration: str, today: datetime.date) -> int:
    """Return days until expiration, or -1 on parse error."""
    try:
        return (datetime.date.fromisoformat(expiration) - today).days
    except Exception:
        return -1

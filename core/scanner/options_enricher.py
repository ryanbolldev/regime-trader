"""
core/scanner/options_enricher.py
---------------------------------
Enriches TickerResult objects with IV rank and ATM bid-ask spread.

Calls AlpacaClient.get_iv_rank() (already implemented) and inspects the
live option chain for ATM 30-45 DTE bid-ask spread.  Tickers with spread
above SCANNER_OPTIONS_SPREAD_MAX are flagged as low_liquidity_options.

Public interface:
  OptionsEnricher.enrich(results, client) -> list[TickerResult]
"""

from __future__ import annotations

import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from config.settings import SCANNER_MAX_WORKERS, SCANNER_OPTIONS_SPREAD_MAX
from core.scanner.batch_trainer import TickerResult

log = logging.getLogger(__name__)


class OptionsEnricher:
    """Attach IV rank and options liquidity flags to TickerResult objects.

    Parameters
    ----------
    client      : AlpacaClient instance for market data.
    max_workers : thread-pool size for parallel enrichment.
    spread_max  : maximum ATM bid-ask spread ($) before flagging illiquid.
    """

    def __init__(
        self,
        client,
        max_workers: int = SCANNER_MAX_WORKERS,
        spread_max:  float = SCANNER_OPTIONS_SPREAD_MAX,
    ) -> None:
        self._client      = client
        self._max_workers = max_workers
        self._spread_max  = spread_max

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich(self, results: list[TickerResult]) -> list[TickerResult]:
        """Mutate *results* in-place with iv_rank, spread, low_liquidity_options.

        Failed-fit tickers are skipped.  Network errors for any single ticker
        are swallowed (iv_rank stays None, low_liquidity_options stays False).
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
                    iv_rank, spread, low_liq = fut.result()
                    result.iv_rank              = iv_rank
                    result.spread               = spread
                    result.low_liquidity_options = low_liq
                except Exception as exc:
                    log.warning("OptionsEnricher: error for %s (skipped): %s", result.ticker, exc)

        log.info(
            "OptionsEnricher: enriched %d tickers (%d low-liquidity)",
            len(to_enrich),
            sum(1 for r in to_enrich if r.low_liquidity_options),
        )
        return results

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _enrich_one(
        self, ticker: str
    ) -> tuple[Optional[float], Optional[float], bool]:
        """Return (iv_rank, atm_spread, low_liquidity_options)."""
        iv_rank = None
        spread  = None
        low_liq = False

        # IV rank via existing AlpacaClient method
        try:
            iv_rank = self._client.get_iv_rank(ticker)
        except Exception as exc:
            log.debug("OptionsEnricher [%s]: iv_rank failed: %s", ticker, exc)

        # ATM bid-ask spread from live chain (30-45 DTE options)
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
                spread  = float(min(spreads))  # best (tightest) spread available
                low_liq = spread > self._spread_max
        except Exception as exc:
            log.debug("OptionsEnricher [%s]: spread fetch failed: %s", ticker, exc)

        log.debug(
            "OptionsEnricher [%s]: iv_rank=%s spread=%s low_liq=%s",
            ticker, iv_rank, spread, low_liq,
        )
        return iv_rank, spread, low_liq


def _safe_date_days(expiration: str, today: datetime.date) -> int:
    """Return days until expiration, or -1 on parse error."""
    try:
        return (datetime.date.fromisoformat(expiration) - today).days
    except Exception:
        return -1

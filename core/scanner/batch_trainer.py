"""
core/scanner/batch_trainer.py
------------------------------
Parallel HMM training across a universe of tickers.

Uses ThreadPoolExecutor(max_workers=SCANNER_MAX_WORKERS) to stay within
Alpaca's rate limits while training one HMMEngine per ticker.  Each result
captures the fitted regime, BIC score, convergence status, and regime duration.

Rate-limit resilience:
  - Tickers are submitted in batches of max_workers.  After each batch
    completes, the thread pool sleeps SCANNER_BATCH_SLEEP_SECS before
    processing the next batch.
  - If _train_one raises RateLimitError (HTTP 429), it is retried up to
    SCANNER_MAX_RETRIES times with exponential backoff (0.5 s, 1 s, 2 s).
  - Tickers that exhaust all retries are excluded with reason
    "rate_limit_exhausted" and counted in total_retries.

Public interface:
  BatchTrainer.run(tickers, ohlcv_map) -> list[TickerResult]
  BatchTrainer.total_retries           -> int  (populated by run())
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config.settings import (
    SCANNER_BATCH_SLEEP_SECS,
    SCANNER_DURATION_HOLDOUT_BARS,
    SCANNER_MAX_RETRIES,
    SCANNER_MAX_WORKERS,
    SCANNER_TRAIN_BARS,
)
from core.hmm_engine import HMMEngine

log = logging.getLogger(__name__)

# Exponential backoff delays for successive retries
_BACKOFF_SECS = [0.5, 1.0, 2.0]


@dataclass
class TickerResult:
    ticker:               str
    current_regime:       int           # -1 if fit failed or not yet confirmed
    regime_duration_bars: int           # bars since last regime change
    bic_score:            float         # lower is better
    converged:            bool          # True if EM converged
    convergence_warning:  bool          # True when NOT converged
    n_states:             int           # winning state count
    fit_failed:           bool = False
    error_message:        str  = ""
    # enrichment fields (populated later by OptionsEnricher)
    iv_rank:              Optional[float] = None
    spread:               Optional[float] = None
    low_liquidity_options: bool           = False
    high_iv_event_risk:   bool            = False
    # vol estimator fields (populated by OptionsEnricher)
    vol_rank:             Optional[float] = None
    vol_rank_available:   bool            = False
    vol_components:       dict            = field(default_factory=dict)
    vol_estimated:        bool            = False


class BatchTrainer:
    """Train HMMEngine for each ticker in parallel batches.

    Parameters
    ----------
    max_workers  : thread-pool size (default from settings, 5)
    train_bars   : maximum history bars to pass to HMMEngine.fit()
    batch_sleep  : seconds to sleep between batches (throttles API calls)
    max_retries  : per-ticker retry limit on RateLimitError before exclusion
    """

    def __init__(
        self,
        max_workers:      int   = SCANNER_MAX_WORKERS,
        train_bars:       int   = SCANNER_TRAIN_BARS,
        batch_sleep:      float = SCANNER_BATCH_SLEEP_SECS,
        max_retries:      int   = SCANNER_MAX_RETRIES,
        duration_holdout: int   = SCANNER_DURATION_HOLDOUT_BARS,
    ) -> None:
        self._max_workers      = max_workers
        self._train_bars       = train_bars
        self._batch_sleep      = batch_sleep
        self._max_retries      = max_retries
        self._duration_holdout = duration_holdout

        self.total_retries: int = 0
        self._retry_lock        = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        tickers:   list[str],
        ohlcv_map: dict[str, pd.DataFrame],
    ) -> list[TickerResult]:
        """Train one HMMEngine per ticker in batches and return results.

        Tickers are submitted in groups of max_workers.  After each group
        completes, the loop sleeps batch_sleep seconds before the next group.
        """
        self.total_retries = 0
        results: list[TickerResult] = []

        # Separate tickers with sufficient data from those without
        valid:   list[str] = []
        invalid: list[str] = []
        for ticker in tickers:
            df = ohlcv_map.get(ticker)
            if df is None or len(df) < 30:
                invalid.append(ticker)
            else:
                valid.append(ticker)

        for ticker in invalid:
            log.warning("BatchTrainer: no/insufficient data for %s — skipping", ticker)
            results.append(_failed(ticker, "insufficient data"))

        # Process valid tickers in batches of max_workers
        batch_size = self._max_workers
        for batch_start in range(0, len(valid), batch_size):
            batch   = valid[batch_start : batch_start + batch_size]
            futures = {}

            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                for ticker in batch:
                    df  = ohlcv_map[ticker]
                    fut = pool.submit(self._train_with_retry, ticker, df)
                    futures[fut] = ticker

                for fut in as_completed(futures):
                    ticker = futures[fut]
                    try:
                        results.append(fut.result())
                    except Exception as exc:
                        log.error("BatchTrainer: unhandled error for %s: %s", ticker, exc)
                        results.append(_failed(ticker, str(exc)))

            # Throttle between batches (skip after the final batch)
            if batch_start + batch_size < len(valid):
                log.debug(
                    "BatchTrainer: batch done — sleeping %.1fs before next batch",
                    self._batch_sleep,
                )
                time.sleep(self._batch_sleep)

        n_ok   = sum(1 for r in results if not r.fit_failed)
        n_fail = sum(1 for r in results if r.fit_failed)
        log.info(
            "BatchTrainer: %d tickers — %d ok, %d failed, %d rate-limit retries",
            len(results), n_ok, n_fail, self.total_retries,
        )
        return results

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _train_with_retry(self, ticker: str, df: pd.DataFrame) -> TickerResult:
        """Wrap _train_one with exponential-backoff retry on RateLimitError."""
        from broker.alpaca_client import RateLimitError

        for attempt in range(self._max_retries + 1):
            try:
                return self._train_one(ticker, df)
            except RateLimitError:
                if attempt >= self._max_retries:
                    log.warning(
                        "[SCANNER] Rate limit exhausted on %s after %d retries — excluding",
                        ticker, self._max_retries,
                    )
                    return _failed(ticker, "rate_limit_exhausted")

                wait = _BACKOFF_SECS[min(attempt, len(_BACKOFF_SECS) - 1)]
                log.warning(
                    "[SCANNER] Rate limit hit on %s — retry %d/%d in %.1fs",
                    ticker, attempt + 1, self._max_retries, wait,
                )
                with self._retry_lock:
                    self.total_retries += 1
                time.sleep(wait)

        # Unreachable, but satisfies type checker
        return _failed(ticker, "rate_limit_exhausted")  # pragma: no cover

    def _train_one(self, ticker: str, df: pd.DataFrame) -> TickerResult:
        """Fit HMMEngine on training bars; count duration on out-of-sample holdout."""
        df = df.tail(self._train_bars).copy()

        # Split: fit on all-but-holdout, count duration on the trailing holdout only.
        # Ensures regime_duration_bars reflects genuine recent out-of-sample persistence
        # rather than trivially recounting the training data's own dominant label.
        _min_train = 30
        holdout  = min(self._duration_holdout, max(0, len(df) - _min_train))
        train_df   = df.iloc[:-holdout] if holdout > 0 else df
        holdout_df = df.iloc[-holdout:] if holdout > 0 else df.iloc[0:0]

        engine = HMMEngine()
        try:
            engine.fit(train_df)
        except Exception as exc:
            log.warning("HMMEngine.fit failed for %s: %s", ticker, exc)
            return _failed(ticker, str(exc))

        # Walk forward on holdout bars only — pure out-of-sample
        regimes: list[int] = []
        try:
            for i in range(len(holdout_df)):
                r = engine.predict_current(holdout_df.iloc[i])
                regimes.append(r)
        except Exception:
            pass

        current_regime  = regimes[-1] if regimes else -1
        regime_duration = 0
        for r in reversed(regimes):
            if r == current_regime:
                regime_duration += 1
            else:
                break

        model    = engine._model
        n_states = model.n_components if model is not None else 0
        monitor  = getattr(model, "monitor_", None)
        converged = bool(getattr(monitor, "converged", False)) if monitor else False

        bic = getattr(engine, "_last_bic", float("inf"))

        log.debug(
            "BatchTrainer [%s]: regime=%d dur=%d bic=%.1f converged=%s n=%d",
            ticker, current_regime, regime_duration, bic, converged, n_states,
        )
        return TickerResult(
            ticker=ticker,
            current_regime=current_regime,
            regime_duration_bars=regime_duration,
            bic_score=bic,
            converged=converged,
            convergence_warning=not converged,
            n_states=n_states,
        )


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _failed(ticker: str, reason: str) -> TickerResult:
    return TickerResult(
        ticker=ticker,
        current_regime=-1,
        regime_duration_bars=0,
        bic_score=float("inf"),
        converged=False,
        convergence_warning=True,
        n_states=0,
        fit_failed=True,
        error_message=reason,
    )

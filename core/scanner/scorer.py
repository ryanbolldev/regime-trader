"""
core/scanner/scorer.py
-----------------------
Composite scoring and strategy mapping for the nightly scanner.

Produces LONG and SHORT scores (0-100) for each TickerResult using:
  regime_score      40%  — regime label mapped to direction
  confirmation      20%  — HMM confirmation gate & flicker status
  regime_duration   15%  — bars in current regime (saturation at 20)
  iv_rank           15%  — options IV environment fit to direction
  model_quality     10%  — BIC / convergence quality

Tickers scoring < SCANNER_SCORE_THRESHOLD are excluded from the returned
list, but ALL eligible tickers are scored for the distribution summary
(accessible via Scorer.last_distribution after each score() call).

Public interface:
  Scorer.score(results) -> list[ScoredTicker]
  Scorer.last_distribution -> dict        (populated by score())
  get_suggested_strategy(direction, iv_rank) -> str
  build_score_distribution(all_scored, threshold) -> dict
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from config.settings import SCANNER_SCORE_THRESHOLD
from core.scanner.batch_trainer import TickerResult

log = logging.getLogger(__name__)

# Regime label 0=crash 1=bear 2=neutral 3=bull 4=euphoria
_REGIME_LONG_SCORES  = {0: 0,  1: 20, 2: 50, 3: 90, 4: 70}
_REGIME_SHORT_SCORES = {0: 95, 1: 80, 2: 50, 3: 10, 4: 30}

_REGIME_DURATION_SAT = 20   # bars at which duration bonus saturates

_BUCKETS = ["0-20", "20-40", "40-60", "60-80", "80-100"]


@dataclass
class ScoredTicker:
    ticker:               str
    current_regime:       int
    regime_name:          str
    long_score:           float
    short_score:          float
    direction:            str           # "LONG", "SHORT", or "NEUTRAL"
    suggested_strategy:   str
    iv_rank:              Optional[float]
    spread:               Optional[float]
    low_liquidity_options: bool
    regime_duration_bars: int
    bic_score:            float
    converged:            bool


class Scorer:
    """Compute composite LONG/SHORT scores for scanner results.

    Parameters
    ----------
    threshold : minimum score required to include a ticker (default from settings)
    """

    def __init__(self, threshold: float = SCANNER_SCORE_THRESHOLD) -> None:
        self._threshold     = threshold
        self.last_distribution: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, results: list[TickerResult]) -> list[ScoredTicker]:
        """Score all eligible TickerResult objects; return those above threshold.

        Failed-fit tickers and unconfirmed regimes (current_regime == -1) are
        excluded entirely.  ALL other tickers are scored for the distribution
        (stored in self.last_distribution); only those >= threshold are returned,
        sorted by best score descending.
        """
        all_scored: list[ScoredTicker] = []

        for r in results:
            if r.fit_failed or r.current_regime < 0:
                continue

            long_s  = self._compute_score(r, direction="LONG")
            short_s = self._compute_score(r, direction="SHORT")

            direction = (
                "LONG"  if long_s >= short_s else
                "SHORT"
            )

            strategy = get_suggested_strategy(direction, r.iv_rank)
            if r.low_liquidity_options and "WHEEL" in strategy:
                strategy = "EQUITY_ONLY"

            regime_name = _REGIME_NAMES.get(r.current_regime, f"state_{r.current_regime}")

            all_scored.append(ScoredTicker(
                ticker               = r.ticker,
                current_regime       = r.current_regime,
                regime_name          = regime_name,
                long_score           = round(long_s,  1),
                short_score          = round(short_s, 1),
                direction            = direction,
                suggested_strategy   = strategy,
                iv_rank              = r.iv_rank,
                spread               = r.spread,
                low_liquidity_options = r.low_liquidity_options,
                regime_duration_bars = r.regime_duration_bars,
                bic_score            = round(r.bic_score, 1),
                converged            = r.converged,
            ))

        # Build distribution across ALL scored tickers before filtering
        self.last_distribution = build_score_distribution(all_scored, self._threshold)

        # Filter to threshold passers and sort
        passed = [s for s in all_scored if max(s.long_score, s.short_score) >= self._threshold]
        passed.sort(key=lambda s: max(s.long_score, s.short_score), reverse=True)

        log.info(
            "Scorer: %d/%d tickers passed threshold=%.0f (long=%d short=%d)",
            len(passed),
            len(all_scored),
            self._threshold,
            sum(1 for s in passed if s.direction == "LONG"),
            sum(1 for s in passed if s.direction == "SHORT"),
        )
        return passed

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _compute_score(self, r: TickerResult, direction: str) -> float:
        """Weighted composite score for one direction."""
        regime = r.current_regime

        # 1. Regime alignment (40%)
        regime_map  = _REGIME_LONG_SCORES if direction == "LONG" else _REGIME_SHORT_SCORES
        regime_comp = regime_map.get(regime, 50)

        # 2. Confirmation quality (20%)
        if r.converged and not r.convergence_warning:
            confirm_comp = 100.0
        elif r.convergence_warning:
            confirm_comp = 50.0
        else:
            confirm_comp = 20.0

        # 3. Regime duration (15%): longer duration = more stable = higher score
        dur_pct      = min(r.regime_duration_bars / _REGIME_DURATION_SAT, 1.0)
        duration_comp = dur_pct * 100.0

        # 4. IV rank (15%): aligned if LONG and IV is moderate; SHORT if IV is high
        if r.iv_rank is None:
            iv_comp = 50.0
        elif direction == "LONG":
            iv_comp = max(0.0, 100.0 - r.iv_rank)
        else:
            iv_comp = r.iv_rank

        # 5. Model quality (10%)
        if r.bic_score == float("inf"):
            quality_comp = 0.0
        else:
            quality_comp = 80.0 if r.converged else 50.0

        score = (
            0.40 * regime_comp
            + 0.20 * confirm_comp
            + 0.15 * duration_comp
            + 0.15 * iv_comp
            + 0.10 * quality_comp
        )
        return min(100.0, score)


# ---------------------------------------------------------------------------
# Distribution builder
# ---------------------------------------------------------------------------

def build_score_distribution(
    all_scored: list[ScoredTicker],
    threshold: float,
) -> dict:
    """Build score distribution dict from ALL scored tickers (pre-filter).

    Returns a dict with keys 'long' and 'short', each containing:
      total, mean, buckets (dict of label->count), passed_threshold
    """
    if not all_scored:
        empty_dir = {
            "total":            0,
            "mean":             0.0,
            "buckets":          {b: 0 for b in _BUCKETS},
            "passed_threshold": 0,
        }
        return {"long": empty_dir, "short": empty_dir}

    def _bucket(score: float) -> str:
        if score < 20:   return "0-20"
        if score < 40:   return "20-40"
        if score < 60:   return "40-60"
        if score < 80:   return "60-80"
        return "80-100"

    def _dir_stats(scores: list[float]) -> dict:
        total   = len(scores)
        mean    = round(sum(scores) / total, 1) if total else 0.0
        buckets = {b: 0 for b in _BUCKETS}
        passed  = 0
        for s in scores:
            buckets[_bucket(s)] += 1
            if s >= threshold:
                passed += 1
        return {
            "total":            total,
            "mean":             mean,
            "buckets":          buckets,
            "passed_threshold": passed,
        }

    long_scores  = [s.long_score  for s in all_scored]
    short_scores = [s.short_score for s in all_scored]
    return {
        "long":  _dir_stats(long_scores),
        "short": _dir_stats(short_scores),
    }


# ---------------------------------------------------------------------------
# Strategy mapping
# ---------------------------------------------------------------------------

_REGIME_NAMES: dict[int, str] = {
    0: "crash",
    1: "bear",
    2: "neutral",
    3: "bull",
    4: "euphoria",
}


def get_suggested_strategy(direction: str, iv_rank: Optional[float]) -> str:
    """Map direction + IV environment to a strategy name."""
    iv = iv_rank if iv_rank is not None else 50.0

    if direction == "LONG":
        if iv >= 50:
            return "CASH_SECURED_PUT"
        return "BUY_EQUITY"

    if direction == "SHORT":
        if iv >= 50:
            return "COVERED_CALL"
        return "BEAR_SPREAD"

    # NEUTRAL
    if iv >= 60:
        return "IRON_CONDOR"
    return "WHEEL"

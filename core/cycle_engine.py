"""
core/cycle_engine.py
---------------------
Probabilistic 60-day cycle detection for BTC.

Detects cycle lows from price history, scores timing probability via a
Gaussian window, and combines three price-confirmation signals (Donchian,
Gaussian MA, Bollinger bands) with optional HMM regime context into a
composite cycle score.

Public interface:
  CycleLow     — dataclass for a detected cycle trough
  CycleSignal  — dataclass for the full cycle output
  CycleEngine  — detection and scoring engine

  engine = CycleEngine("BTC")
  signal = engine.get_cycle_signal(price_history_df, hmm_regime=2)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    CYCLE_60D_CENTER,
    CYCLE_60D_STD,
    CYCLE_BOLLINGER_WEIGHT,
    CYCLE_COMPOSITE_THRESHOLD,
    CYCLE_DONCHIAN_WEIGHT,
    CYCLE_GAUSSIAN_WEIGHT,
    CYCLE_LOW_CONFIRMATION_PCT,
    CYCLE_QUALITY_LOOKBACK,
)
from core import alerts

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Nov 2022 BTC 4-year cycle low — used for macro phase calculation
_BTC_4Y_LOW_DATE = datetime(2022, 11, 21, tzinfo=timezone.utc)

# Seeded historical major cycle lows (date, price)
_BTC_SEED_LOWS = [
    ("2018-12-15", 3_200.0),
    ("2020-03-13", 3_800.0),
    ("2022-11-21", 15_500.0),
]

# Module-level prev-score tracker for threshold-crossing alerts
_prev_cycle_score: Optional[float] = None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CycleLow:
    timestamp:          datetime
    price:              float
    days_from_prior_low: float
    confidence:         float    # 0–1, how clean the low structure was
    confirmed:          bool     # True when price rose >CYCLE_LOW_CONFIRMATION_PCT after


@dataclass
class CycleSignal:
    days_since_last_low:    int
    timing_probability:     float    # 0–1
    window_center_days:     int
    window_std_days:        int
    macro_phase:            str      # accumulation / markup / distribution / markdown
    days_since_cycle_start: int
    cycle_completion_pct:   float    # 0–1
    translation:            str      # left / right / unknown
    translation_confidence: float    # 0–1
    donchian_score:         float    # 0–1
    gaussian_score:         float    # 0–1
    bollinger_score:        float    # 0–1
    price_confirmation:     float    # weighted combination of the three scores above
    hmm_confirmation:       float    # 0–1
    composite_score:        float    # 0–1
    bias:                   str      # long / neutral / short
    failed_cycle:           bool
    cycle_quality_score:    float    # 0–1
    adaptive_window_center: int


# ---------------------------------------------------------------------------
# CycleEngine
# ---------------------------------------------------------------------------

class CycleEngine:
    """Probabilistic 60-day cycle detection engine."""

    def __init__(self, symbol: str = "BTC") -> None:
        self.symbol = symbol
        self._seed_lows: list[CycleLow] = self._build_seed_lows()

    # ------------------------------------------------------------------
    # Seed lows
    # ------------------------------------------------------------------

    def _build_seed_lows(self) -> list[CycleLow]:
        lows: list[CycleLow] = []
        prev_ts: Optional[datetime] = None
        for date_str, price in _BTC_SEED_LOWS:
            ts = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
            days = float((ts - prev_ts).days) if prev_ts is not None else 0.0
            lows.append(CycleLow(
                timestamp           = ts,
                price               = price,
                days_from_prior_low = days,
                confidence          = 0.90,
                confirmed           = True,
            ))
            prev_ts = ts
        return lows

    # ------------------------------------------------------------------
    # detect_cycle_lows
    # ------------------------------------------------------------------

    def detect_cycle_lows(self, price_history_df: pd.DataFrame) -> list[CycleLow]:
        """Return confirmed cycle lows detected in price_history_df.

        A candidate low is the minimum over a ±15-bar window.
        Confirmation: price rises >CYCLE_LOW_CONFIRMATION_PCT within 20 bars.
        """
        if price_history_df.empty or "close" not in price_history_df.columns:
            return []

        close = price_history_df["close"]
        n = len(close)
        half = 15
        confirmation_window = 20

        if n < half + confirmation_window + 1:
            return []

        confirmed_lows: list[CycleLow] = []

        for i in range(half, n - confirmation_window):
            window_vals = close.iloc[i - half : i + half + 1]
            candidate_price = float(close.iloc[i])
            if candidate_price != float(window_vals.min()):
                continue

            future_max = float(close.iloc[i + 1 : i + confirmation_window + 1].max())
            rise_pct = (future_max - candidate_price) / max(candidate_price, 1.0)
            if rise_pct < CYCLE_LOW_CONFIRMATION_PCT:
                continue

            confidence = float(max(0.10, min(1.0, rise_pct * 3.0)))

            ts = price_history_df.index[i]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            if hasattr(ts, "tzinfo") and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            prior_lows: list[CycleLow] = self._seed_lows + confirmed_lows
            prior_ts = prior_lows[-1].timestamp if prior_lows else None
            days_from_prior = float((ts - prior_ts).days) if prior_ts is not None else 0.0

            confirmed_lows.append(CycleLow(
                timestamp           = ts,
                price               = candidate_price,
                days_from_prior_low = days_from_prior,
                confidence          = confidence,
                confirmed           = True,
            ))

        return confirmed_lows

    # ------------------------------------------------------------------
    # calculate_timing_probability
    # ------------------------------------------------------------------

    def calculate_timing_probability(
        self,
        days_elapsed: int,
        center: int,
        std: int,
    ) -> float:
        """Gaussian probability peaked at center, returns 0–1."""
        if std <= 0:
            return 1.0 if days_elapsed == center else 0.0
        return float(math.exp(-0.5 * ((days_elapsed - center) / std) ** 2))

    # ------------------------------------------------------------------
    # update_adaptive_window
    # ------------------------------------------------------------------

    def update_adaptive_window(self, recent_cycle_lengths: list[float]) -> int:
        """Weighted average of last 3 cycle lengths, clamped to [45, 90].

        Weights [0.50, 0.30, 0.20] applied most-recent-first.
        """
        if not recent_cycle_lengths:
            return CYCLE_60D_CENTER

        lengths = recent_cycle_lengths[-3:]
        weights = [0.50, 0.30, 0.20][: len(lengths)]
        lengths_rev = list(reversed(lengths))
        w_sum = sum(weights[: len(lengths_rev)])
        avg = sum(l * w for l, w in zip(lengths_rev, weights)) / w_sum
        return int(max(45, min(90, round(avg))))

    # ------------------------------------------------------------------
    # evaluate_cycle_hypotheses
    # ------------------------------------------------------------------

    def evaluate_cycle_hypotheses(
        self,
        price_history: pd.DataFrame,
        candidates: list[CycleLow],
    ) -> Optional[CycleLow]:
        """Score last 3 candidates; return the highest-confidence hypothesis."""
        if not candidates:
            return None

        close = price_history.get("close") if isinstance(price_history, pd.DataFrame) else None
        if close is None or close.empty:
            return max(candidates[-3:], key=lambda c: c.confidence)

        best: Optional[CycleLow] = None
        best_score = -1.0

        for candidate in candidates[-3:]:
            ts = candidate.timestamp
            try:
                ts_lookup = (
                    pd.Timestamp(ts).tz_localize(None)
                    if ts.tzinfo is not None
                    else pd.Timestamp(ts)
                )
                if (
                    hasattr(price_history.index, "tz")
                    and price_history.index.tz is not None
                ):
                    ts_lookup = ts_lookup.tz_localize("UTC")
                idx_pos = price_history.index.get_indexer([ts_lookup], method="nearest")[0]
            except Exception:
                idx_pos = -1

            if idx_pos < 0 or idx_pos >= len(close):
                score = candidate.confidence * 0.5
            else:
                after = close.iloc[idx_pos:]
                if len(after) < 5:
                    score = candidate.confidence * 0.5
                else:
                    max_gain = (float(after.max()) - candidate.price) / max(candidate.price, 1.0)
                    revisit_penalty = min(0.5, (after < candidate.price * 1.01).sum() * 0.05)
                    score = float(candidate.confidence * min(1.0, max_gain) - revisit_penalty)
                    score = max(0.0, score)

            # Prefer later (more recent) candidates when scores are nearly equal
            if score >= best_score - 1e-9:
                best_score = score
                best = candidate

        return best or candidates[-1]

    # ------------------------------------------------------------------
    # calculate_donchian_score
    # ------------------------------------------------------------------

    def calculate_donchian_score(
        self,
        price_history: pd.DataFrame,
        window: int = 60,
    ) -> float:
        """Score based on position within Donchian channel.

        0.0  — price at or above midpoint.
        →1.0 — price approaching lower band.
        1.0  — price breached lower band then recovered above it.
        """
        close = price_history.get("close") if isinstance(price_history, pd.DataFrame) else None
        if close is None or len(close) < window:
            return 0.0

        upper = float(close.rolling(window).max().iloc[-1])
        lower = float(close.rolling(window).min().iloc[-1])
        current = float(close.iloc[-1])

        if upper <= lower:
            return 0.5

        midpoint = (upper + lower) / 2.0

        recent = close.iloc[-5:] if len(close) >= 5 else close
        breached_and_recovered = (
            float(recent.min()) <= lower * 1.01 and current > lower * 1.01
        )
        if breached_and_recovered:
            return 1.0

        if current >= midpoint:
            return 0.0

        score = (midpoint - current) / (midpoint - lower) * 0.95
        return float(max(0.0, min(1.0, score)))

    # ------------------------------------------------------------------
    # calculate_gaussian_score
    # ------------------------------------------------------------------

    def calculate_gaussian_score(
        self,
        price_history: pd.DataFrame,
        window: int = 60,
    ) -> float:
        """Score based on price position relative to Gaussian-weighted MA.

        1.0 — price crosses back above GMA after being below.
        0.5 — price riding the GMA.
        0.0 — price well below GMA.
        """
        close = price_history.get("close") if isinstance(price_history, pd.DataFrame) else None
        if close is None or len(close) < window:
            return 0.5

        # Centered Gaussian kernel: weights peak at the middle of the window.
        # This gives a smooth lagged average so "price above/below GMA" is meaningful.
        x = np.arange(window, dtype=float)
        center = (window - 1) / 2.0
        sigma = window / 6.0
        weights = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        weights /= weights.sum()

        series_now = close.values[-window:].astype(float)
        gma_now = float(np.dot(series_now, weights))
        current = float(close.iloc[-1])

        # Previous GMA for crossover detection
        prev_gma: Optional[float] = None
        if len(close) >= window + 1:
            series_prev = close.values[-(window + 1) : -1].astype(float)
            prev_gma = float(np.dot(series_prev, weights))

        prev_close = float(close.iloc[-2]) if len(close) >= 2 else current

        if prev_gma is not None and prev_close < prev_gma and current >= gma_now:
            return 1.0

        if gma_now > 0 and abs(current - gma_now) / gma_now < 0.01:
            return 0.5

        if current < gma_now:
            depth = (gma_now - current) / max(gma_now, 1.0)
            return float(max(0.0, 0.5 - depth * 5.0))

        return 0.3  # above GMA but no recent crossover — not a cycle low signal

    # ------------------------------------------------------------------
    # calculate_bollinger_score
    # ------------------------------------------------------------------

    def calculate_bollinger_score(
        self,
        price_history: pd.DataFrame,
        window: int = 20,
        std: float = 2.0,
    ) -> float:
        """Score based on Bollinger band position and band width change.

        1.0 — price touched lower band and bands are now expanding.
        0.5 — touched lower band but bands still contracting.
        0.0 — bands contracting or price near upper band.
        """
        close = price_history.get("close") if isinstance(price_history, pd.DataFrame) else None
        if close is None or len(close) < window + 5:
            return 0.0

        rolling_std = close.rolling(window).std()
        rolling_mean = close.rolling(window).mean()

        bw_now = rolling_std.iloc[-1]
        if bw_now is None or (isinstance(bw_now, float) and math.isnan(bw_now)):
            return 0.0
        bw_now = float(bw_now)

        valid_std = rolling_std.dropna()
        bw_prev = float(valid_std.iloc[-5]) if len(valid_std) >= 5 else bw_now

        lower_now = float(rolling_mean.iloc[-1]) - std * bw_now
        upper_now = float(rolling_mean.iloc[-1]) + std * bw_now
        current = float(close.iloc[-1])
        band_range = max(upper_now - lower_now, 1e-9)

        bands_expanding = bw_now > bw_prev * 1.001  # 0.1% threshold avoids float ties
        recent_low = float(close.iloc[-5:].min())
        # Tolerance proportional to band width — avoids false positives on tight bands
        touched_lower = recent_low <= lower_now + band_range * 0.05

        # Price in top 20% of band → not a cycle-low entry condition
        position_in_band = (current - lower_now) / band_range
        if position_in_band > 0.8 or current >= upper_now:
            return 0.0

        if touched_lower and bands_expanding:
            return 1.0
        if touched_lower:
            return 0.5
        if not bands_expanding:
            return 0.1
        return 0.2

    # ------------------------------------------------------------------
    # is_failed_cycle
    # ------------------------------------------------------------------

    def is_failed_cycle(
        self,
        current_price: float,
        prior_cycle_low: CycleLow,
    ) -> bool:
        """Return True if current price broke below the cycle low that started this cycle."""
        return current_price < prior_cycle_low.price

    # ------------------------------------------------------------------
    # measure_translation
    # ------------------------------------------------------------------

    def measure_translation(
        self,
        cycle_low: CycleLow,
        current_price: float,
        days_elapsed: int,
    ) -> tuple[str, float]:
        """Estimate cycle translation from days_elapsed alone.

        Returns ('unknown', 0.0) when the cycle is not mature enough (<30 days).
        For full translation detection, use _measure_translation_from_history.
        """
        if days_elapsed < 30:
            return ("unknown", 0.0)
        return ("unknown", 0.0)

    def _measure_translation_from_history(
        self,
        price_history: pd.DataFrame,
        cycle_low_ts: datetime,
        days_elapsed: int,
    ) -> tuple[str, float]:
        """Measure cycle translation using the full price series.

        'right' — peak occurred in second half (bullish structure).
        'left'  — peak occurred in first half (bearish structure).
        """
        if days_elapsed < 30:
            return ("unknown", 0.0)

        close = price_history.get("close") if isinstance(price_history, pd.DataFrame) else None
        if close is None or close.empty:
            return ("unknown", 0.0)

        since_low = close.iloc[-days_elapsed:] if len(close) >= days_elapsed else close

        if len(since_low) < 10:
            return ("unknown", 0.0)

        peak_pos = int(since_low.values.argmax())
        n = len(since_low)
        half = n / 2.0

        maturity = min(1.0, days_elapsed / 60.0)
        confidence = float(maturity * 0.8)

        translation = "right" if peak_pos > half else "left"
        return (translation, confidence)

    # ------------------------------------------------------------------
    # calculate_macro_phase
    # ------------------------------------------------------------------

    def calculate_macro_phase(self, days_since_4y_low: int) -> str:
        """Map days since 4-year cycle low to market phase."""
        if days_since_4y_low <= 365:
            return "accumulation"
        if days_since_4y_low <= 730:
            return "markup"
        if days_since_4y_low <= 1095:
            return "distribution"
        return "markdown"

    # ------------------------------------------------------------------
    # _cycle_quality_score
    # ------------------------------------------------------------------

    def _cycle_quality_score(self, all_lows: list[CycleLow]) -> float:
        """Score the regularity of the last CYCLE_QUALITY_LOOKBACK cycle lengths."""
        if len(all_lows) < 2:
            return 0.5

        recent = all_lows[-CYCLE_QUALITY_LOOKBACK:]
        lengths = [c.days_from_prior_low for c in recent if c.days_from_prior_low > 0]
        if len(lengths) < 2:
            return float(recent[-1].confidence)

        mean_len = sum(lengths) / len(lengths)
        if mean_len <= 0:
            return 0.5

        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        cv = math.sqrt(variance) / mean_len
        quality = max(0.0, min(1.0, 1.0 - cv))
        avg_conf = sum(c.confidence for c in recent) / len(recent)
        return float(quality * avg_conf)

    # ------------------------------------------------------------------
    # _hmm_to_confirmation
    # ------------------------------------------------------------------

    def _hmm_to_confirmation(
        self,
        hmm_regime: Optional[int],
        timing_prob: float,
    ) -> float:
        """Convert HMM regime to a cycle-low confirmation score.

        crash(0)/bear(1) at high timing probability → bullish for cycle low (1.0).
        neutral(2)                                  → 0.5 always.
        bull(3)/euphoria(4) at high timing          → bearish (0.0).
        None / -1 (unconfirmed)                     → 0.5.
        """
        if hmm_regime is None or hmm_regime == -1:
            return 0.5
        if hmm_regime in (0, 1):
            return float(min(1.0, 0.5 + timing_prob * 0.5))
        if hmm_regime == 2:
            return 0.5
        # bull (3) or euphoria (4)
        return float(max(0.0, 0.5 - timing_prob * 0.5))

    # ------------------------------------------------------------------
    # get_cycle_signal  (main entry point)
    # ------------------------------------------------------------------

    def get_cycle_signal(
        self,
        price_history_df: pd.DataFrame,
        hmm_regime: Optional[int] = None,
    ) -> CycleSignal:
        """Compute the full cycle signal for the most recent bar."""
        global _prev_cycle_score

        if price_history_df.empty or "close" not in price_history_df.columns:
            return self._neutral_signal()

        # Collect all cycle lows
        detected = self.detect_cycle_lows(price_history_df)
        all_lows = self._seed_lows + detected

        current_price = float(price_history_df["close"].iloc[-1])

        current_ts = price_history_df.index[-1]
        if hasattr(current_ts, "to_pydatetime"):
            current_ts = current_ts.to_pydatetime()
        if hasattr(current_ts, "tzinfo") and current_ts.tzinfo is None:
            current_ts = current_ts.replace(tzinfo=timezone.utc)

        # Current cycle start via hypothesis evaluation
        cycle_low = self.evaluate_cycle_hypotheses(price_history_df, all_lows)
        if cycle_low is None:
            cycle_low = all_lows[-1]

        days_since_low = max(0, (current_ts - cycle_low.timestamp).days)

        # Adaptive window center
        cycle_lengths = [c.days_from_prior_low for c in all_lows if c.days_from_prior_low > 0]
        adaptive_center = self.update_adaptive_window(cycle_lengths)

        # Timing probability
        timing_prob = self.calculate_timing_probability(
            days_since_low, adaptive_center, CYCLE_60D_STD
        )

        # Cycle completion
        cycle_completion = float(min(1.0, days_since_low / max(adaptive_center, 1)))

        # Price confirmation scores
        d_score = self.calculate_donchian_score(price_history_df)
        g_score = self.calculate_gaussian_score(price_history_df)
        b_score = self.calculate_bollinger_score(price_history_df)
        price_conf = float(
            CYCLE_DONCHIAN_WEIGHT * d_score
            + CYCLE_GAUSSIAN_WEIGHT * g_score
            + CYCLE_BOLLINGER_WEIGHT * b_score
        )

        # Failed cycle: current price broke below the cycle start low
        failed = self.is_failed_cycle(current_price, cycle_low)

        # HMM confirmation
        hmm_conf = self._hmm_to_confirmation(hmm_regime, timing_prob)

        # Cycle quality
        quality = self._cycle_quality_score(all_lows)

        # Translation
        translation, trans_conf = self._measure_translation_from_history(
            price_history_df, cycle_low.timestamp, days_since_low
        )

        # Macro phase (4-year cycle)
        days_since_4y = max(0, (current_ts - _BTC_4Y_LOW_DATE).days)
        macro_phase = self.calculate_macro_phase(days_since_4y)

        # Composite score
        composite = float(max(0.0, min(1.0,
            0.35 * timing_prob
            + 0.30 * price_conf
            + 0.20 * hmm_conf
            + 0.15 * quality
        )))

        # Bias
        if failed:
            bias = "short"
        elif composite > CYCLE_COMPOSITE_THRESHOLD:
            bias = "long"
        else:
            bias = "neutral"

        signal = CycleSignal(
            days_since_last_low    = days_since_low,
            timing_probability     = float(timing_prob),
            window_center_days     = adaptive_center,
            window_std_days        = CYCLE_60D_STD,
            macro_phase            = macro_phase,
            days_since_cycle_start = days_since_low,
            cycle_completion_pct   = cycle_completion,
            translation            = translation,
            translation_confidence = float(trans_conf),
            donchian_score         = float(d_score),
            gaussian_score         = float(g_score),
            bollinger_score        = float(b_score),
            price_confirmation     = price_conf,
            hmm_confirmation       = float(hmm_conf),
            composite_score        = composite,
            bias                   = bias,
            failed_cycle           = failed,
            cycle_quality_score    = float(quality),
            adaptive_window_center = adaptive_center,
        )

        # Fire alert on threshold crossing or failed cycle
        prev = _prev_cycle_score
        _prev_cycle_score = composite
        alerts.send_cycle_alert(signal, prev_score=prev if prev is not None else 0.0)

        return signal

    # ------------------------------------------------------------------
    # Neutral default
    # ------------------------------------------------------------------

    def _neutral_signal(self) -> CycleSignal:
        return CycleSignal(
            days_since_last_low    = 0,
            timing_probability     = 0.0,
            window_center_days     = CYCLE_60D_CENTER,
            window_std_days        = CYCLE_60D_STD,
            macro_phase            = "unknown",
            days_since_cycle_start = 0,
            cycle_completion_pct   = 0.0,
            translation            = "unknown",
            translation_confidence = 0.0,
            donchian_score         = 0.0,
            gaussian_score         = 0.5,
            bollinger_score        = 0.0,
            price_confirmation     = 0.0,
            hmm_confirmation       = 0.5,
            composite_score        = 0.0,
            bias                   = "neutral",
            failed_cycle           = False,
            cycle_quality_score    = 0.0,
            adaptive_window_center = CYCLE_60D_CENTER,
        )

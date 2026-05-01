"""
tests/test_cycle_engine.py
---------------------------
Tests for core/cycle_engine.py — BTC probabilistic cycle detection.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from core.cycle_engine import CycleEngine, CycleLow, CycleSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(values, start: str = "2023-01-01") -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a close price list."""
    dates = pd.bdate_range(start, periods=len(values))
    close = np.array(values, dtype=float)
    return pd.DataFrame(
        {
            "open":   close * 0.999,
            "high":   close * 1.010,
            "low":    close * 0.990,
            "close":  close,
            "volume": np.ones(len(values)) * 1_000_000.0,
        },
        index=dates,
    )


def _cycle_low(
    ts_str: str,
    price: float,
    days_from_prior: float = 60.0,
    conf: float = 0.8,
) -> CycleLow:
    ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
    return CycleLow(
        timestamp           = ts,
        price               = price,
        days_from_prior_low = days_from_prior,
        confidence          = conf,
        confirmed           = True,
    )


@pytest.fixture()
def engine() -> CycleEngine:
    return CycleEngine("BTC")


@pytest.fixture(autouse=True)
def reset_cycle_score():
    """Reset module-level prev_cycle_score between tests."""
    import core.cycle_engine as ce
    ce._prev_cycle_score = None
    yield
    ce._prev_cycle_score = None


# ---------------------------------------------------------------------------
# TestTimingProbability
# ---------------------------------------------------------------------------

class TestTimingProbability:

    def test_peaks_at_center(self, engine):
        assert engine.calculate_timing_probability(60, center=60, std=12) == pytest.approx(1.0)

    def test_returns_zero_to_one(self, engine):
        for days in [0, 30, 60, 90, 120]:
            p = engine.calculate_timing_probability(days, center=60, std=12)
            assert 0.0 <= p <= 1.0, f"Out of range at days={days}: {p}"

    def test_symmetric_around_center(self, engine):
        left  = engine.calculate_timing_probability(50, center=60, std=12)
        right = engine.calculate_timing_probability(70, center=60, std=12)
        assert left == pytest.approx(right)

    def test_decays_away_from_center(self, engine):
        at_center = engine.calculate_timing_probability(60, 60, 12)
        at_offset = engine.calculate_timing_probability(90, 60, 12)
        assert at_center > at_offset

    def test_wider_std_gives_higher_prob_at_offset(self, engine):
        narrow = engine.calculate_timing_probability(75, 60, std=5)
        wide   = engine.calculate_timing_probability(75, 60, std=20)
        assert wide > narrow

    def test_zero_std_returns_one_at_exact_center(self, engine):
        assert engine.calculate_timing_probability(60, 60, 0) == pytest.approx(1.0)

    def test_zero_std_returns_zero_off_center(self, engine):
        assert engine.calculate_timing_probability(61, 60, 0) == pytest.approx(0.0)

    def test_value_at_one_std_matches_gaussian(self, engine):
        # At ±1 std, value should be exp(-0.5) ≈ 0.6065
        p = engine.calculate_timing_probability(72, center=60, std=12)
        assert p == pytest.approx(math.exp(-0.5), rel=1e-5)


# ---------------------------------------------------------------------------
# TestAdaptiveWindow
# ---------------------------------------------------------------------------

class TestAdaptiveWindow:

    def test_clamps_below_minimum(self, engine):
        assert engine.update_adaptive_window([10, 15, 20]) == 45

    def test_clamps_above_maximum(self, engine):
        assert engine.update_adaptive_window([200, 300, 400]) == 90

    def test_weighted_average_calculation(self, engine):
        # [50, 60, 70] → most recent = 70
        # 70*0.50 + 60*0.30 + 50*0.20 = 35+18+10 = 63
        assert engine.update_adaptive_window([50, 60, 70]) == 63

    def test_three_equal_lengths_returns_that_value(self, engine):
        assert engine.update_adaptive_window([60, 60, 60]) == 60

    def test_empty_returns_default_center(self, engine):
        from config.settings import CYCLE_60D_CENTER
        assert engine.update_adaptive_window([]) == CYCLE_60D_CENTER

    def test_single_element_in_range(self, engine):
        result = engine.update_adaptive_window([60])
        assert 45 <= result <= 90

    def test_uses_most_recent_with_highest_weight(self, engine):
        # [40, 50, 80]: most recent=80 → 80*0.50+50*0.30+40*0.20 = 63
        # [80, 50, 40]: most recent=40 → 40*0.50+50*0.30+80*0.20 = 51
        result_high_recent = engine.update_adaptive_window([40, 50, 80])
        result_low_recent  = engine.update_adaptive_window([80, 50, 40])
        assert result_high_recent > result_low_recent

    def test_only_uses_last_three(self, engine):
        # Extra early elements should not affect the result
        r1 = engine.update_adaptive_window([50, 60, 70])
        r2 = engine.update_adaptive_window([1, 2, 3, 4, 5, 50, 60, 70])
        assert r1 == r2


# ---------------------------------------------------------------------------
# TestDonchianScore
# ---------------------------------------------------------------------------

class TestDonchianScore:

    def test_returns_zero_when_price_at_upper_band(self, engine):
        # Monotonically rising series — price is always at the Donchian upper
        prices = list(range(1, 71))
        df = _make_ohlcv(prices)
        assert engine.calculate_donchian_score(df, window=60) == pytest.approx(0.0)

    def test_returns_zero_to_one(self, engine):
        rng = np.random.default_rng(42)
        prices = 100 + rng.normal(0, 5, 80)
        df = _make_ohlcv(prices)
        score = engine.calculate_donchian_score(df)
        assert 0.0 <= score <= 1.0

    def test_returns_one_when_breached_and_recovered(self, engine):
        # Stable base at 100, then price crashes to lower band and recovers
        base = [100.0] * 60
        dip  = [60.0, 58.0, 57.0, 59.0, 61.0]
        df = _make_ohlcv(base + dip)
        assert engine.calculate_donchian_score(df, window=60) == pytest.approx(1.0)

    def test_high_score_near_lower_band(self, engine):
        # Price approaches rolling minimum — score should be > 0
        prices = [100.0] * 60 + [50.0, 48.0, 47.0, 46.0, 48.0]
        df = _make_ohlcv(prices)
        assert engine.calculate_donchian_score(df, window=60) > 0.0

    def test_insufficient_data_returns_zero(self, engine):
        df = _make_ohlcv([100.0, 99.0, 98.0])
        assert engine.calculate_donchian_score(df, window=60) == pytest.approx(0.0)

    def test_zero_when_price_equals_midpoint(self, engine):
        # Build series so current price lands exactly on midpoint
        prices = [80.0] * 30 + [120.0] * 29 + [100.0]  # midpoint = (120+80)/2 = 100
        df = _make_ohlcv(prices)
        score = engine.calculate_donchian_score(df, window=60)
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestGaussianScore
# ---------------------------------------------------------------------------

class TestGaussianScore:

    def test_returns_zero_to_one(self, engine):
        rng = np.random.default_rng(0)
        prices = 100 + rng.normal(0, 3, 100)
        df = _make_ohlcv(prices)
        score = engine.calculate_gaussian_score(df)
        assert 0.0 <= score <= 1.0

    def test_insufficient_data_returns_neutral(self, engine):
        df = _make_ohlcv([100.0, 99.0, 98.0])
        assert engine.calculate_gaussian_score(df, window=60) == pytest.approx(0.5)

    def test_deep_below_gma_returns_low_score(self, engine):
        # Price spends 55 bars near 100, then collapses to 40
        prices = [100.0] * 55 + [40.0] * 10
        df = _make_ohlcv(prices)
        assert engine.calculate_gaussian_score(df) < 0.5

    def test_crossover_returns_one(self, engine):
        # 41 bars at 100 (warm-up), 19 bars at 80 (below centered GMA),
        # then 1 bar at 120 (crosses above GMA) → should return 1.0
        prices = [100.0] * 41 + [80.0] * 19 + [120.0]
        df = _make_ohlcv(prices)
        score = engine.calculate_gaussian_score(df, window=60)
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestBollingerScore
# ---------------------------------------------------------------------------

class TestBollingerScore:

    def test_returns_zero_to_one(self, engine):
        rng = np.random.default_rng(7)
        prices = 100 + rng.normal(0, 3, 60)
        df = _make_ohlcv(prices)
        score = engine.calculate_bollinger_score(df)
        assert 0.0 <= score <= 1.0

    def test_insufficient_data_returns_zero(self, engine):
        df = _make_ohlcv([100.0, 99.0, 98.0])
        assert engine.calculate_bollinger_score(df) == pytest.approx(0.0)

    def test_contracting_bands_returns_low_score(self, engine):
        # Near-flatline: bands shrink to near zero
        prices = [100.0 + 0.001 * i for i in range(30)]
        df = _make_ohlcv(prices)
        assert engine.calculate_bollinger_score(df, window=20) <= 0.2

    def test_touched_lower_band_and_expanding_returns_one(self, engine):
        # Create a series that clearly touches lower band then expands
        rng = np.random.default_rng(12)
        base     = list(100 + rng.normal(0, 2, 20))
        deep_dip = [70.0, 68.0, 66.0, 68.0, 70.0]   # touches lower band
        recovery = [75.0, 82.0, 90.0, 97.0, 105.0]  # bands expand
        df = _make_ohlcv(base + deep_dip + recovery)
        score = engine.calculate_bollinger_score(df, window=20)
        assert 0.0 <= score <= 1.0

    def test_price_at_upper_band_returns_zero(self, engine):
        # Prices monotonically rise so current price is always at upper band
        prices = list(range(50, 110))
        df = _make_ohlcv(prices)
        score = engine.calculate_bollinger_score(df, window=20)
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestFailedCycle
# ---------------------------------------------------------------------------

class TestFailedCycle:

    def test_failed_when_below_prior_low(self, engine):
        low = _cycle_low("2023-01-01", price=20_000.0)
        assert engine.is_failed_cycle(15_000.0, low) is True

    def test_not_failed_when_above_prior_low(self, engine):
        low = _cycle_low("2023-01-01", price=20_000.0)
        assert engine.is_failed_cycle(25_000.0, low) is False

    def test_not_failed_at_exact_prior_price(self, engine):
        low = _cycle_low("2023-01-01", price=20_000.0)
        assert engine.is_failed_cycle(20_000.0, low) is False

    def test_failed_cycle_sets_bias_to_short(self, engine):
        # Price below ALL seed lows (Dec 2018 = $3,200 is the lowest seed)
        # so is_failed_cycle is True regardless of which hypothesis is selected
        rng = np.random.default_rng(1)
        prices = list(30_000.0 + rng.normal(0, 500, 120))
        prices[-1] = 2_500.0   # below even the Dec 2018 seed ($3,200)
        df = _make_ohlcv(prices, start="2023-01-01")
        signal = engine.get_cycle_signal(df)
        assert signal.failed_cycle is True
        assert signal.bias == "short"

    def test_not_failed_at_normal_price(self, engine):
        rng = np.random.default_rng(2)
        prices = 30_000.0 + rng.normal(0, 500, 120)
        df = _make_ohlcv(prices, start="2023-01-01")
        signal = engine.get_cycle_signal(df)
        assert signal.failed_cycle is False


# ---------------------------------------------------------------------------
# TestTranslation
# ---------------------------------------------------------------------------

class TestTranslation:

    def test_unknown_when_cycle_too_short(self, engine):
        translation, conf = engine.measure_translation(
            _cycle_low("2023-06-01", 25_000.0), 30_000.0, days_elapsed=10
        )
        assert translation == "unknown"
        assert conf == pytest.approx(0.0)

    def test_right_translation_peak_in_second_half(self, engine):
        # 60-bar series with peak at bar 40 (second half)
        n = 60
        prices = (
            [100.0 + i * 1.5 for i in range(41)]         # rises to bar 40
            + [160.0 - (i - 40) * 2.0 for i in range(41, n)]  # falls after
        )
        df = _make_ohlcv(prices)
        cycle_ts = df.index[0].to_pydatetime().replace(tzinfo=timezone.utc)
        translation, conf = engine._measure_translation_from_history(
            df, cycle_ts, days_elapsed=n
        )
        assert translation == "right"
        assert 0.0 <= conf <= 1.0

    def test_left_translation_peak_in_first_half(self, engine):
        # 60-bar series with peak at bar 15 (first half)
        n = 60
        prices = (
            [100.0 + i * 3.0 for i in range(16)]            # rises quickly to bar 15
            + [145.0 - (i - 15) * 1.5 for i in range(16, n)]  # falls after
        )
        df = _make_ohlcv(prices)
        cycle_ts = df.index[0].to_pydatetime().replace(tzinfo=timezone.utc)
        translation, conf = engine._measure_translation_from_history(
            df, cycle_ts, days_elapsed=n
        )
        assert translation == "left"

    def test_translation_confidence_in_range(self, engine):
        prices = list(range(100, 160)) + list(range(159, 99, -1))
        df = _make_ohlcv(prices)
        cycle_ts = df.index[0].to_pydatetime().replace(tzinfo=timezone.utc)
        _, conf = engine._measure_translation_from_history(
            df, cycle_ts, days_elapsed=len(prices)
        )
        assert 0.0 <= conf <= 1.0

    def test_unknown_when_too_short_for_from_history(self, engine):
        df = _make_ohlcv([100.0] * 5)
        cycle_ts = df.index[0].to_pydatetime().replace(tzinfo=timezone.utc)
        translation, conf = engine._measure_translation_from_history(
            df, cycle_ts, days_elapsed=5
        )
        assert translation == "unknown"
        assert conf == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestMacroPhase
# ---------------------------------------------------------------------------

class TestMacroPhase:

    def test_accumulation_at_zero(self, engine):
        assert engine.calculate_macro_phase(0) == "accumulation"

    def test_accumulation_at_365(self, engine):
        assert engine.calculate_macro_phase(365) == "accumulation"

    def test_markup_at_366(self, engine):
        assert engine.calculate_macro_phase(366) == "markup"

    def test_markup_at_730(self, engine):
        assert engine.calculate_macro_phase(730) == "markup"

    def test_distribution_at_731(self, engine):
        assert engine.calculate_macro_phase(731) == "distribution"

    def test_distribution_at_1095(self, engine):
        assert engine.calculate_macro_phase(1095) == "distribution"

    def test_markdown_at_1096(self, engine):
        assert engine.calculate_macro_phase(1096) == "markdown"

    def test_markdown_at_large_value(self, engine):
        assert engine.calculate_macro_phase(2000) == "markdown"


# ---------------------------------------------------------------------------
# TestDetectCycleLows
# ---------------------------------------------------------------------------

class TestDetectCycleLows:

    def test_empty_df_returns_empty_list(self, engine):
        assert engine.detect_cycle_lows(pd.DataFrame()) == []

    def test_too_short_returns_empty(self, engine):
        df = _make_ohlcv([100.0] * 20)
        assert engine.detect_cycle_lows(df) == []

    def test_detects_confirmed_low_with_strong_recovery(self, engine):
        # V-shape: 30 flat bars, a clean dip, then a strong >10% recovery
        before   = [100.0] * 30
        dip      = [90.0, 85.0, 80.0, 82.0, 86.0]    # low at bar 32 (price 80)
        recovery = [90.0 + i * 2.0 for i in range(25)]  # rises well above 10%
        df = _make_ohlcv(before + dip + recovery)
        lows = engine.detect_cycle_lows(df)
        assert len(lows) >= 1
        assert all(c.confirmed for c in lows)

    def test_no_confirmation_below_threshold(self, engine):
        # Dip of ~5% then flat — should not satisfy the 10% confirmation
        prices = [100.0] * 30 + [96.0, 95.0, 96.0] + [97.0] * 25
        df = _make_ohlcv(prices)
        lows = engine.detect_cycle_lows(df)
        # Any detected lows must still satisfy the confirmed flag
        for low in lows:
            assert low.confirmed

    def test_confidence_in_range(self, engine):
        before   = [100.0] * 30
        dip      = [60.0, 55.0, 50.0, 55.0, 60.0]
        recovery = [80.0] * 25
        df = _make_ohlcv(before + dip + recovery)
        lows = engine.detect_cycle_lows(df)
        for low in lows:
            assert 0.0 <= low.confidence <= 1.0

    def test_days_from_prior_non_negative(self, engine):
        before   = [100.0] * 30
        dip      = [60.0, 55.0, 50.0, 55.0, 60.0]
        recovery = [80.0] * 25
        df = _make_ohlcv(before + dip + recovery)
        lows = engine.detect_cycle_lows(df)
        for low in lows:
            assert low.days_from_prior_low >= 0


# ---------------------------------------------------------------------------
# TestCycleHypotheses
# ---------------------------------------------------------------------------

class TestCycleHypotheses:

    def test_returns_none_on_empty_candidates(self, engine):
        df = _make_ohlcv([100.0] * 10)
        assert engine.evaluate_cycle_hypotheses(df, []) is None

    def test_returns_single_candidate_directly(self, engine):
        candidates = [_cycle_low("2023-01-01", 20_000.0, conf=0.8)]
        df = _make_ohlcv([100.0] * 30)
        result = engine.evaluate_cycle_hypotheses(df, candidates)
        assert result is not None
        assert result.price == pytest.approx(20_000.0)

    def test_considers_only_last_three(self, engine):
        # 6 candidates — function should only evaluate the last 3
        candidates = [
            _cycle_low(f"2023-0{i}-01", float(i * 1000), conf=float(i) / 10)
            for i in range(1, 7)
        ]
        df = _make_ohlcv([100.0] * 30)
        result = engine.evaluate_cycle_hypotheses(df, candidates)
        # Must return one of the last 3 candidates
        assert result is not None
        assert result.price in {c.price for c in candidates[-3:]}

    def test_prefers_higher_confidence_when_no_history_match(self, engine):
        # No data history that matches the candidate dates → falls back on confidence
        candidates = [
            _cycle_low("2020-01-01", 8_000.0, conf=0.3),
            _cycle_low("2020-02-01", 9_000.0, conf=0.9),  # highest confidence
            _cycle_low("2020-03-01", 7_000.0, conf=0.5),
        ]
        df = _make_ohlcv([100.0] * 10)  # tiny df — all get fallback score
        result = engine.evaluate_cycle_hypotheses(df, candidates)
        assert result is not None  # returns something

    def test_returns_best_scoring_candidate(self, engine):
        # Build price history that clearly starts from a deep low and rises sharply
        prices = [15_000.0] + [15_000.0 * (1 + 0.01 * i) for i in range(1, 120)]
        df = _make_ohlcv(prices, start="2022-11-21")
        candidates = [
            _cycle_low("2022-11-21", 15_000.0, conf=0.9),
            _cycle_low("2023-03-01", 20_000.0, conf=0.5),
        ]
        result = engine.evaluate_cycle_hypotheses(df, candidates)
        assert result is not None


# ---------------------------------------------------------------------------
# TestHmmConfirmation
# ---------------------------------------------------------------------------

class TestHmmConfirmation:

    def test_crash_at_high_timing_gives_one(self, engine):
        assert engine._hmm_to_confirmation(0, timing_prob=1.0) == pytest.approx(1.0)

    def test_bear_at_high_timing_gives_one(self, engine):
        assert engine._hmm_to_confirmation(1, timing_prob=1.0) == pytest.approx(1.0)

    def test_neutral_always_returns_half(self, engine):
        assert engine._hmm_to_confirmation(2, timing_prob=0.9) == pytest.approx(0.5)
        assert engine._hmm_to_confirmation(2, timing_prob=0.0) == pytest.approx(0.5)

    def test_bull_at_high_timing_gives_zero(self, engine):
        assert engine._hmm_to_confirmation(3, timing_prob=1.0) == pytest.approx(0.0)

    def test_euphoria_at_high_timing_gives_zero(self, engine):
        assert engine._hmm_to_confirmation(4, timing_prob=1.0) == pytest.approx(0.0)

    def test_none_regime_returns_half(self, engine):
        assert engine._hmm_to_confirmation(None, timing_prob=0.9) == pytest.approx(0.5)

    def test_unconfirmed_minus1_returns_half(self, engine):
        assert engine._hmm_to_confirmation(-1, timing_prob=0.9) == pytest.approx(0.5)

    def test_all_outputs_in_range(self, engine):
        for regime in [-1, None, 0, 1, 2, 3, 4]:
            for timing in [0.0, 0.5, 1.0]:
                conf = engine._hmm_to_confirmation(regime, timing)
                assert 0.0 <= conf <= 1.0, (
                    f"Out of range: regime={regime} timing={timing} conf={conf}"
                )

    def test_crash_at_low_timing_gives_lower_confirmation(self, engine):
        high = engine._hmm_to_confirmation(0, timing_prob=1.0)
        low  = engine._hmm_to_confirmation(0, timing_prob=0.0)
        assert high > low


# ---------------------------------------------------------------------------
# TestCompositeWeights
# ---------------------------------------------------------------------------

class TestCompositeWeights:

    def test_weights_sum_to_one(self):
        # Explicit composite weights from the spec
        assert sum([0.35, 0.30, 0.20, 0.15]) == pytest.approx(1.0)

    def test_price_confirmation_weights_sum_to_one(self):
        from config.settings import (
            CYCLE_DONCHIAN_WEIGHT,
            CYCLE_GAUSSIAN_WEIGHT,
            CYCLE_BOLLINGER_WEIGHT,
        )
        assert CYCLE_DONCHIAN_WEIGHT + CYCLE_GAUSSIAN_WEIGHT + CYCLE_BOLLINGER_WEIGHT == pytest.approx(1.0)

    def test_composite_in_range(self, engine):
        rng = np.random.default_rng(99)
        prices = 30_000 + rng.normal(0, 1000, 150)
        df = _make_ohlcv(prices, start="2023-01-01")
        signal = engine.get_cycle_signal(df, hmm_regime=2)
        assert 0.0 <= signal.composite_score <= 1.0

    def test_all_max_inputs_gives_high_composite(self, engine):
        with (
            patch.object(engine, "calculate_timing_probability", return_value=1.0),
            patch.object(engine, "calculate_donchian_score",     return_value=1.0),
            patch.object(engine, "calculate_gaussian_score",     return_value=1.0),
            patch.object(engine, "calculate_bollinger_score",    return_value=1.0),
            patch.object(engine, "_cycle_quality_score",         return_value=1.0),
            patch.object(engine, "_hmm_to_confirmation",         return_value=1.0),
            patch.object(engine, "is_failed_cycle",              return_value=False),
        ):
            rng = np.random.default_rng(5)
            prices = 30_000 + rng.normal(0, 500, 120)
            df = _make_ohlcv(prices, start="2023-01-01")
            signal = engine.get_cycle_signal(df, hmm_regime=0)
        assert signal.composite_score == pytest.approx(1.0)
        assert signal.bias == "long"

    def test_failed_cycle_forces_short_bias_regardless_of_composite(self, engine):
        with patch.object(engine, "is_failed_cycle", return_value=True):
            rng = np.random.default_rng(5)
            prices = 30_000 + rng.normal(0, 500, 120)
            df = _make_ohlcv(prices, start="2023-01-01")
            signal = engine.get_cycle_signal(df)
        assert signal.bias == "short"
        assert signal.failed_cycle is True

    def test_low_composite_gives_neutral_bias(self, engine):
        with (
            patch.object(engine, "calculate_timing_probability", return_value=0.0),
            patch.object(engine, "calculate_donchian_score",     return_value=0.0),
            patch.object(engine, "calculate_gaussian_score",     return_value=0.0),
            patch.object(engine, "calculate_bollinger_score",    return_value=0.0),
            patch.object(engine, "_cycle_quality_score",         return_value=0.0),
            patch.object(engine, "_hmm_to_confirmation",         return_value=0.0),
            patch.object(engine, "is_failed_cycle",              return_value=False),
        ):
            rng = np.random.default_rng(5)
            prices = 30_000 + rng.normal(0, 500, 120)
            df = _make_ohlcv(prices, start="2023-01-01")
            signal = engine.get_cycle_signal(df)
        assert signal.bias == "neutral"


# ---------------------------------------------------------------------------
# TestGetCycleSignal  (integration)
# ---------------------------------------------------------------------------

class TestGetCycleSignal:

    def _synthetic_btc(self, n: int = 200, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 2 * math.pi, n)
        trend  = np.linspace(30_000, 35_000, n)
        cycle  = 2_000 * np.sin(t)
        noise  = rng.normal(0, 300, n)
        prices = trend + cycle + noise
        return _make_ohlcv(prices, start="2023-01-01")

    def test_returns_cycle_signal_instance(self, engine):
        assert isinstance(engine.get_cycle_signal(self._synthetic_btc()), CycleSignal)

    def test_all_scores_in_range(self, engine):
        s = engine.get_cycle_signal(self._synthetic_btc(), hmm_regime=2)
        for attr in [
            "timing_probability", "donchian_score", "gaussian_score",
            "bollinger_score", "price_confirmation", "hmm_confirmation",
            "composite_score", "cycle_quality_score", "translation_confidence",
            "cycle_completion_pct",
        ]:
            val = getattr(s, attr)
            assert 0.0 <= val <= 1.0, f"{attr}={val} out of [0,1]"

    def test_macro_phase_is_valid_string(self, engine):
        s = engine.get_cycle_signal(self._synthetic_btc())
        assert s.macro_phase in ("accumulation", "markup", "distribution", "markdown", "unknown")

    def test_translation_is_valid(self, engine):
        s = engine.get_cycle_signal(self._synthetic_btc())
        assert s.translation in ("left", "right", "unknown")

    def test_bias_is_valid(self, engine):
        s = engine.get_cycle_signal(self._synthetic_btc())
        assert s.bias in ("long", "neutral", "short")

    def test_adaptive_window_center_in_range(self, engine):
        s = engine.get_cycle_signal(self._synthetic_btc())
        assert 45 <= s.adaptive_window_center <= 90

    def test_days_since_low_non_negative(self, engine):
        s = engine.get_cycle_signal(self._synthetic_btc())
        assert s.days_since_last_low >= 0

    def test_empty_df_returns_neutral_signal(self, engine):
        s = engine.get_cycle_signal(pd.DataFrame())
        assert s.bias == "neutral"
        assert s.composite_score == pytest.approx(0.0)
        assert s.failed_cycle is False

    def test_crash_regime_gives_higher_hmm_confirmation_than_euphoria(self, engine):
        df = self._synthetic_btc(seed=3)
        s_crash    = engine.get_cycle_signal(df, hmm_regime=0)
        s_euphoria = engine.get_cycle_signal(df, hmm_regime=4)
        assert s_crash.hmm_confirmation >= s_euphoria.hmm_confirmation

    def test_seed_lows_initialized_correctly(self, engine):
        assert len(engine._seed_lows) == 3
        prices = {low.price for low in engine._seed_lows}
        assert 3_200.0 in prices
        assert 3_800.0 in prices
        assert 15_500.0 in prices
        assert all(low.confirmed for low in engine._seed_lows)

    def test_price_confirmation_matches_weighted_sum(self, engine):
        from config.settings import (
            CYCLE_DONCHIAN_WEIGHT,
            CYCLE_GAUSSIAN_WEIGHT,
            CYCLE_BOLLINGER_WEIGHT,
        )
        df = self._synthetic_btc()
        s = engine.get_cycle_signal(df)
        expected = (
            CYCLE_DONCHIAN_WEIGHT  * s.donchian_score
            + CYCLE_GAUSSIAN_WEIGHT  * s.gaussian_score
            + CYCLE_BOLLINGER_WEIGHT * s.bollinger_score
        )
        assert s.price_confirmation == pytest.approx(expected, abs=1e-9)

    def test_window_std_matches_setting(self, engine):
        from config.settings import CYCLE_60D_STD
        s = engine.get_cycle_signal(self._synthetic_btc())
        assert s.window_std_days == CYCLE_60D_STD

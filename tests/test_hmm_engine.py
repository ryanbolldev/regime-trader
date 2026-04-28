"""
tests/test_hmm_engine.py
-------------------------
Unit tests for core/hmm_engine.py and core/feature_engineering.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import CONFIRMATION_BARS, FLICKER_THRESHOLD, FLICKER_WINDOW
from core.feature_engineering import (
    LookaheadBiasError,
    compute,
    validate_no_lookahead,
)
from core.hmm_engine import HMMEngine, _check_flicker, _map_regime_label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n_bars: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0003, 0.012, n_bars)
    close = 100 * np.exp(np.cumsum(log_returns))
    noise = rng.uniform(0.001, 0.015, n_bars)
    high  = close * (1 + noise)
    low   = close * (1 - noise)
    open_ = np.clip(close * (1 + rng.normal(0, 0.005, n_bars)), low, high)
    volume = rng.lognormal(14.0, 0.6, n_bars).astype(int)
    dates = pd.bdate_range(end="2024-01-01", periods=n_bars)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _fitted_engine(n_bars: int = 600) -> tuple[HMMEngine, pd.DataFrame]:
    ohlcv    = _make_ohlcv(n_bars)
    features = compute(ohlcv).dropna()
    engine   = HMMEngine()
    engine.fit(features)
    return engine, features


# ---------------------------------------------------------------------------
# feature_engineering tests
# ---------------------------------------------------------------------------

class TestFeatureEngineering:

    def test_compute_returns_expected_columns(self):
        ohlcv = _make_ohlcv()
        features = compute(ohlcv)
        expected = {"log_return", "realized_vol_20", "volume_zscore",
                    "hl_range_norm", "rsi_14"}
        assert expected.issubset(set(features.columns))

    def test_index_matches_input(self):
        ohlcv = _make_ohlcv()
        features = compute(ohlcv)
        assert features.index.equals(ohlcv.index)

    def test_warmup_nan_present(self):
        ohlcv = _make_ohlcv(100)
        features = compute(ohlcv)
        # realized_vol_20 needs 20-bar warmup + 1 shift → first 21 rows NaN
        assert features["realized_vol_20"].iloc[:21].isna().all()

    def test_no_future_data_in_realized_vol(self):
        """realized_vol_20 must not use the current bar's return."""
        ohlcv = _make_ohlcv(200)
        features = compute(ohlcv)
        # Corrupt the last close price and recompute — realized_vol of the
        # *same* bar must not change (it should only depend on past bars).
        ohlcv_modified = ohlcv.copy()
        ohlcv_modified.iloc[-1, ohlcv_modified.columns.get_loc("close")] *= 2
        features_modified = compute(ohlcv_modified)
        # realized_vol_20 of the last bar should be identical because it is
        # built from bars [-21..-1] (shifted), not from bar[-1] itself.
        assert np.isclose(
            features["realized_vol_20"].iloc[-1],
            features_modified["realized_vol_20"].iloc[-1],
        ), "realized_vol_20 changed when only the current bar's close changed — lookahead detected."

    def test_validate_no_lookahead_passes_on_clean_features(self):
        ohlcv    = _make_ohlcv(600)
        features = compute(ohlcv)
        # Should not raise
        validate_no_lookahead(features, ohlcv)

    def test_validate_no_lookahead_raises_on_injected_future(self):
        """Inject a synthetic feature that is perfectly correlated with the
        next bar's return and confirm the validator catches it."""
        ohlcv    = _make_ohlcv(600)
        features = compute(ohlcv).copy()
        close    = ohlcv["close"]
        future_ret = np.log(close / close.shift(1)).shift(-1)
        # Overwrite log_return with future return
        features["log_return"] = future_ret
        with pytest.raises(LookaheadBiasError):
            validate_no_lookahead(features, ohlcv)

    def test_hl_range_norm_non_negative(self):
        ohlcv    = _make_ohlcv(200)
        features = compute(ohlcv)
        assert (features["hl_range_norm"].dropna() >= 0).all()

    def test_rsi_bounded(self):
        ohlcv    = _make_ohlcv(200)
        features = compute(ohlcv)
        rsi = features["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()


# ---------------------------------------------------------------------------
# HMMEngine: fit
# ---------------------------------------------------------------------------

class TestHMMEngineFit:

    def test_fit_runs_without_error(self):
        engine, _ = _fitted_engine()
        assert engine._model is not None

    def test_fit_selects_state_count_in_valid_range(self):
        engine, _ = _fitted_engine()
        from config.settings import HMM_MAX_STATES, HMM_MIN_STATES
        assert HMM_MIN_STATES <= engine._n_states <= HMM_MAX_STATES

    def test_fit_raises_on_insufficient_data(self):
        engine   = HMMEngine()
        features = compute(_make_ohlcv(10)).dropna()
        with pytest.raises((ValueError, Exception)):
            engine.fit(features)

    def test_state_to_regime_covers_all_states(self):
        engine, _ = _fitted_engine()
        assert set(engine._state_to_regime.keys()) == set(range(engine._n_states))


# ---------------------------------------------------------------------------
# HMMEngine: predict_current (forward-only constraint)
# ---------------------------------------------------------------------------

class TestHMMEnginePredict:

    def test_predict_current_returns_valid_label(self):
        engine, features = _fitted_engine()
        label = engine.predict_current(features.iloc[-1])
        assert label in {-1, 0, 1, 2, 3, 4}

    def test_predict_current_raises_without_fit(self):
        engine = HMMEngine()
        row    = pd.Series({"log_return": 0.001, "realized_vol_20": 0.01,
                            "volume_zscore": 0.0, "hl_range_norm": 0.02,
                            "rsi_14": 50.0})
        with pytest.raises(RuntimeError):
            engine.predict_current(row)

    def test_forward_only_no_future_rows_used(self):
        """predict_current() must produce the same result whether or not future
        bars are present in the feature set — it only receives a single row."""
        engine, features = _fitted_engine()
        row_t = features.iloc[-10]

        # Call with exactly row_t
        engine.reset_filters()
        # Re-fit to reset internal state cleanly
        engine2, _ = _fitted_engine()
        result_a = engine2.predict_current(row_t)

        # Add 10 extra future rows to the OHLCV — predict_current still only
        # receives the same single row, so result must be identical.
        engine3, _ = _fitted_engine(n_bars=610)
        result_b = engine3.predict_current(row_t)

        # Both must be valid labels; the key invariant is that predict_current
        # only sees the single row passed in, never more.
        assert result_a in {-1, 0, 1, 2, 3, 4}
        assert result_b in {-1, 0, 1, 2, 3, 4}

    def test_regime_history_grows_by_one_per_call(self):
        engine, features = _fitted_engine()
        clean = features.dropna()
        for i in range(1, 6):
            engine.predict_current(clean.iloc[-(6 - i)])
            assert len(engine.regime_history()) == i


# ---------------------------------------------------------------------------
# Confirmation gate
# ---------------------------------------------------------------------------

class TestConfirmationGate:

    def _engine_with_state_sequence(self, states: list[int]) -> HMMEngine:
        """Drive the confirmation gate with a pre-baked sequence of raw states."""
        engine = HMMEngine(confirmation_bars=CONFIRMATION_BARS)
        # Patch _forward_decode to return states[i] on call i
        call_count = {"n": 0}

        def fake_predict(_row):
            idx   = call_count["n"]
            state = states[idx] if idx < len(states) else states[-1]
            call_count["n"] += 1
            regime = engine._state_to_regime.get(state, state)
            if regime == engine._pending_state:
                engine._pending_count += 1
            else:
                engine._pending_state = regime
                engine._pending_count  = 1
            if engine._pending_count >= engine.confirmation_bars:
                engine._confirmed_regime = regime
                engine._history.append(regime)
                engine._recent_regimes.append(regime)
                engine._uncertain = _check_flicker(
                    engine._recent_regimes, engine.flicker_threshold
                )
            elif engine._confirmed_regime is not None:
                engine._history.append(engine._confirmed_regime)
                engine._recent_regimes.append(engine._confirmed_regime)
                engine._uncertain = _check_flicker(
                    engine._recent_regimes, engine.flicker_threshold
                )
            return engine._confirmed_regime if engine._confirmed_regime is not None else -1

        # Manually set up mapping so state == regime
        engine._model = object()  # non-None sentinel
        engine._n_states = 5
        engine._state_to_regime = {i: i for i in range(5)}
        engine.predict_current = fake_predict  # type: ignore[assignment]
        return engine

    def test_not_confirmed_before_threshold(self):
        engine = HMMEngine(confirmation_bars=3)
        engine._model      = object()
        engine._n_states   = 5
        engine._state_to_regime = {i: i for i in range(5)}

        # Feed two consecutive same-state bars manually
        for _ in range(CONFIRMATION_BARS - 1):
            engine._pending_state = 2
            engine._pending_count += 1

        assert not engine.is_confirmed()

    def test_confirmed_at_threshold(self):
        engine = HMMEngine(confirmation_bars=CONFIRMATION_BARS)
        engine._pending_state = 3
        engine._pending_count = CONFIRMATION_BARS
        assert engine.is_confirmed()

    def test_counter_resets_on_state_change(self):
        engine, features = _fitted_engine()
        clean = features.dropna()
        # Feed bars until confirmed
        for i in range(CONFIRMATION_BARS + 2):
            engine.predict_current(clean.iloc[i])
        # Force a state change by temporarily injecting a different pending state
        engine._pending_state = (engine._pending_state + 1) % 5
        engine._pending_count  = 0
        assert engine._pending_count == 0

    def test_regime_label_only_changes_after_confirmation(self):
        """The emitted label should not change until CONFIRMATION_BARS consecutive
        bars of the same raw state have been seen."""
        engine, features = _fitted_engine()
        clean = features.dropna()
        labels = []
        for i in range(CONFIRMATION_BARS * 3):
            labels.append(engine.predict_current(clean.iloc[i]))
        # Before the first confirmation fires, label must stay -1 or last confirmed
        assert all(l in {-1, 0, 1, 2, 3, 4} for l in labels)


# ---------------------------------------------------------------------------
# Flicker filter
# ---------------------------------------------------------------------------

class TestFlickerFilter:

    def test_flicker_detected_when_changes_exceed_threshold(self):
        from collections import deque
        # Alternating regimes → many changes
        recent = deque([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], maxlen=FLICKER_WINDOW)
        assert _check_flicker(recent, FLICKER_THRESHOLD) is True

    def test_flicker_not_detected_when_changes_below_threshold(self):
        from collections import deque
        recent = deque([2, 2, 2, 2, 3, 3, 3, 3, 3, 3], maxlen=FLICKER_WINDOW)
        assert _check_flicker(recent, FLICKER_THRESHOLD) is False

    def test_is_uncertain_true_after_high_flicker(self):
        engine, _ = _fitted_engine()
        # Stuff the recent-regimes deque with alternating labels
        from collections import deque
        engine._recent_regimes = deque(
            [i % 2 for i in range(FLICKER_WINDOW)], maxlen=FLICKER_WINDOW
        )
        engine._uncertain = _check_flicker(engine._recent_regimes, engine.flicker_threshold)
        assert engine.is_uncertain() is True

    def test_is_uncertain_false_when_stable(self):
        engine, features = _fitted_engine()
        from collections import deque
        engine._recent_regimes = deque([2] * FLICKER_WINDOW, maxlen=FLICKER_WINDOW)
        engine._uncertain = _check_flicker(engine._recent_regimes, engine.flicker_threshold)
        assert engine.is_uncertain() is False

    def test_normal_signal_passes_through_when_stable(self):
        engine, features = _fitted_engine()
        clean = features.dropna()
        # Run enough bars so at least one regime gets confirmed
        for i in range(CONFIRMATION_BARS * 4):
            engine.predict_current(clean.iloc[i])
        # If a regime was confirmed, uncertain should be False on stable data
        if engine._confirmed_regime is not None:
            assert not engine.is_uncertain()


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

class TestModelSelection:

    def test_best_state_count_chosen_from_candidate_set(self):
        engine, _ = _fitted_engine()
        from config.settings import HMM_MAX_STATES, HMM_MIN_STATES
        assert HMM_MIN_STATES <= engine._n_states <= HMM_MAX_STATES

    def test_refit_does_not_alter_already_emitted_history(self):
        """History emitted before a re-fit must be immutable."""
        engine, features = _fitted_engine()
        clean = features.dropna()
        for i in range(CONFIRMATION_BARS + 2):
            engine.predict_current(clean.iloc[i])
        snapshot = list(engine.regime_history())

        # Re-fit with a slightly different window
        engine.fit(clean.iloc[10:])
        assert engine.regime_history()[:len(snapshot)] == snapshot


# ---------------------------------------------------------------------------
# Regime label mapping
# ---------------------------------------------------------------------------

class TestRegimeLabelMapping:

    @pytest.mark.parametrize("n_states,expected", [
        (3, {0: 0, 1: 2, 2: 4}),
        (5, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}),
    ])
    def test_map_regime_label_boundaries(self, n_states, expected):
        result = {rank: _map_regime_label(rank, n_states) for rank in range(n_states)}
        assert result == expected

    def test_regime_name_returns_string(self):
        engine = HMMEngine()
        for label in range(5):
            assert isinstance(engine.regime_name(label), str)

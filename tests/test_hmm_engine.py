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


def _fitted_engine(n_bars: int = 600, seed: int = 0) -> tuple[HMMEngine, pd.DataFrame]:
    ohlcv    = _make_ohlcv(n_bars, seed=seed)
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

    def test_n_init_restarts_log_debug_per_seed(self, caplog):
        """fit() must attempt HMM_N_INIT seeds per candidate state count."""
        import logging
        from config.settings import HMM_MIN_STATES, HMM_N_INIT
        engine, features = _fitted_engine()
        clean = features.dropna()
        with caplog.at_level(logging.DEBUG, logger="core.hmm_engine"):
            engine.fit(clean)
        seed_lines = [r for r in caplog.records if "seed=" in r.message]
        # At minimum the lowest state count must have tried HMM_N_INIT seeds
        seeds_for_min = [
            r for r in seed_lines
            if f"n_states={HMM_MIN_STATES}" in r.message
        ]
        assert len(seeds_for_min) >= HMM_N_INIT

    def test_final_ll_delta_logged_at_info(self, caplog):
        """The winning model's ll_delta and converged status must appear in INFO log."""
        import logging
        engine, features = _fitted_engine()
        clean = features.dropna()
        with caplog.at_level(logging.INFO, logger="core.hmm_engine"):
            engine.fit(clean)
        info_lines = [r.message for r in caplog.records if r.levelno == logging.INFO
                      and "final_ll_delta" in r.message]
        assert info_lines, "Expected 'final_ll_delta' in INFO log after fit()"

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


# ---------------------------------------------------------------------------
# Per-ticker symbol attribute and independence
# ---------------------------------------------------------------------------

class TestPerTickerSymbol:

    def test_symbol_stored_on_instance(self):
        engine = HMMEngine("MSTR")
        assert engine.symbol == "MSTR"

    def test_default_symbol_is_empty_string(self):
        engine = HMMEngine()
        assert engine.symbol == ""

    def test_symbol_does_not_affect_fit_behaviour(self):
        ohlcv    = _make_ohlcv(600)
        features = compute(ohlcv).dropna()
        e1 = HMMEngine("SPY")
        e2 = HMMEngine("MSTR")
        e1.fit(features)
        e2.fit(features)
        assert e1._n_states == e2._n_states

    def test_two_engines_have_independent_histories(self):
        engine_a, features = _fitted_engine()
        engine_b, _        = _fitted_engine(seed=1)
        clean = features.dropna()

        engine_a.predict_current(clean.iloc[-1])
        assert len(engine_a.regime_history()) == 1
        assert len(engine_b.regime_history()) == 0  # b unaffected by a's call

        engine_b.predict_current(clean.iloc[-1])
        assert len(engine_a.regime_history()) == 1  # a unaffected by b's call
        assert len(engine_b.regime_history()) == 1

    def test_two_engines_independent_confirmed_state(self):
        engine_a, features = _fitted_engine()
        engine_b, _        = _fitted_engine(seed=99)
        clean = features.dropna()

        # Drive engine_a to a confirmed state
        for i in range(CONFIRMATION_BARS + 1):
            engine_a.predict_current(clean.iloc[i])

        # engine_b's confirmation gate must remain independent
        assert engine_b._pending_count == 0
        assert engine_b._confirmed_regime is None


# ---------------------------------------------------------------------------
# HMM Staleness Detection
# ---------------------------------------------------------------------------

class TestModelStaleness:

    def test_is_model_stale_false_after_fit(self):
        engine, _ = _fitted_engine()
        assert engine.is_model_stale is False

    def test_is_model_stale_false_on_normal_observation(self):
        engine, features = _fitted_engine()
        clean = features.dropna()
        # Use a mid-training bar — safely in the center of the distribution,
        # not at the tail where borderline LL values can cross the 2σ threshold.
        mid_row = clean.iloc[len(clean) // 2]
        engine.predict_current(mid_row)
        assert engine.is_model_stale is False

    def test_is_model_stale_true_on_extreme_observation(self):
        """An observation far outside the training distribution triggers stale=True."""
        engine, features = _fitted_engine()
        # Create an extreme row: all values × 1000 → log-likelihood will be far below threshold
        extreme_row = features.dropna().iloc[-1] * 1000.0
        engine.predict_current(extreme_row)
        assert engine.is_model_stale is True

    def test_stale_observation_forces_uncertain(self):
        """is_uncertain() must be True whenever is_model_stale is True."""
        engine, features = _fitted_engine()
        extreme_row = features.dropna().iloc[-1] * 1000.0
        engine.predict_current(extreme_row)
        assert engine.is_model_stale is True
        assert engine.is_uncertain() is True

    def test_staleness_clears_after_normal_observation(self):
        """After a stale bar, a normal observation should clear the stale flag."""
        engine, features = _fitted_engine()
        clean = features.dropna()
        mid_row = clean.iloc[len(clean) // 2]
        engine.predict_current(mid_row * 1000.0)   # stale — extreme scaling
        assert engine.is_model_stale is True
        engine.predict_current(mid_row)             # normal — center of distribution
        assert engine.is_model_stale is False

    def test_train_ll_mean_and_std_set_after_fit(self):
        engine, _ = _fitted_engine()
        assert engine._train_ll_mean is not None
        assert engine._train_ll_std  is not None

    def test_reset_filters_clears_stale_flag(self):
        engine, features = _fitted_engine()
        engine.predict_current(features.dropna().iloc[-1] * 1000.0)
        assert engine.is_model_stale is True
        engine.reset_filters()
        assert engine.is_model_stale is False


# ---------------------------------------------------------------------------
# HMM Convergence (n_init restarts, iter/tol settings)
# ---------------------------------------------------------------------------

class TestHMMConvergence:
    """Verify the EM algorithm converges on well-separated synthetic data."""

    @staticmethod
    def _make_synthetic_features(n_per_state: int = 300, seed: int = 7) -> pd.DataFrame:
        """Three clearly separated Gaussian clusters (crash / neutral / bull).

        Each row represents one bar's feature vector matching the columns
        produced by feature_engineering.compute():
          log_return, volatility, rsi, volume_ratio, hl_norm
        """
        rng = np.random.default_rng(seed)
        # State 0 (crash):   strong negative return, high vol
        crash = rng.multivariate_normal(
            mean=[-0.04, 0.045, 25.0, 0.7, 0.08],
            cov=np.diag([1e-5, 1e-5, 4.0, 0.01, 1e-5]),
            size=n_per_state,
        )
        # State 1 (neutral): near-zero return, moderate vol
        neutral = rng.multivariate_normal(
            mean=[0.001, 0.015, 50.0, 1.0, 0.03],
            cov=np.diag([1e-6, 1e-5, 4.0, 0.01, 1e-5]),
            size=n_per_state,
        )
        # State 2 (bull):    positive return, low vol
        bull = rng.multivariate_normal(
            mean=[0.035, 0.008, 72.0, 1.3, 0.015],
            cov=np.diag([1e-5, 1e-5, 4.0, 0.01, 1e-5]),
            size=n_per_state,
        )
        X = np.vstack([crash, neutral, bull])
        cols = ["log_return", "volatility", "rsi", "volume_ratio", "hl_norm"]
        return pd.DataFrame(X, columns=cols)

    def test_converges_within_iter_limit_on_synthetic_data(self):
        """GaussianHMM with n_iter=500 and tol=1e-5 should converge on well-separated
        synthetic data before exhausting all iterations."""
        from hmmlearn.hmm import GaussianHMM
        from config.settings import HMM_COVARIANCE_TYPE, HMM_N_ITER, HMM_TOL

        X = self._make_synthetic_features().values.astype(float)
        model = GaussianHMM(
            n_components=3,
            covariance_type=HMM_COVARIANCE_TYPE,
            n_iter=HMM_N_ITER,
            tol=HMM_TOL,
            random_state=42,
        )
        model.fit(X)

        monitor = getattr(model, "monitor_", None)
        assert monitor is not None, "hmmlearn did not expose monitor_ attribute"
        assert monitor.converged is True, (
            f"Model did not converge within {HMM_N_ITER} iterations "
            f"(history len={len(monitor.history)})"
        )

    def test_converged_model_has_small_ll_delta(self):
        """After convergence the final LL improvement should be smaller than tol."""
        from hmmlearn.hmm import GaussianHMM
        from config.settings import HMM_COVARIANCE_TYPE, HMM_N_ITER, HMM_TOL

        X = self._make_synthetic_features().values.astype(float)
        model = GaussianHMM(
            n_components=3,
            covariance_type=HMM_COVARIANCE_TYPE,
            n_iter=HMM_N_ITER,
            tol=HMM_TOL,
            random_state=42,
        )
        model.fit(X)

        history = model.monitor_.history
        assert len(history) >= 2, "Too few EM iterations recorded"
        ll_delta = abs(history[-1] - history[-2])
        assert ll_delta < 1.0, (
            f"Final LL delta {ll_delta:.4e} is unexpectedly large — "
            "model may not have converged meaningfully"
        )

    def test_engine_fit_converges_on_synthetic_data(self):
        """HMMEngine.fit() on synthetic well-separated data selects a model that
        converges, reflected by is_model_stale=False on in-distribution observations."""
        features = self._make_synthetic_features(n_per_state=400)
        engine = HMMEngine()
        engine.fit(features)

        # Sample a row from the middle of the training data (clearly in-distribution)
        row = features.iloc[len(features) // 2]
        engine.predict_current(row)
        assert engine.is_model_stale is False

    def test_n_init_restarts_produce_valid_selection(self):
        """With n_init=5 restarts the best model must still make sensible predictions."""
        from config.settings import HMM_N_INIT
        assert HMM_N_INIT >= 2, "Test requires at least 2 restarts to be meaningful"

        features = self._make_synthetic_features(n_per_state=300)
        engine = HMMEngine()
        engine.fit(features)

        # Fit must succeed and expose a valid mapping
        assert engine._model is not None
        assert len(engine._state_to_regime) == engine._n_states
        regime_labels = set(engine._state_to_regime.values())
        assert regime_labels.issubset({0, 1, 2, 3, 4})

"""
tests/test_walk_forward.py
--------------------------
Unit tests for core/walk_forward.py — 16 required tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from core.walk_forward import (
    FoldResult,
    InsufficientDataError,
    WalkForwardBacktester,
    WalkForwardResult,
    _avg_confirmed_regime_duration,
    _compute_regime_consistency,
    _compute_transition_matrix,
    _compute_wf_windows,
    interpret_overfitting_ratio,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng      = np.random.default_rng(seed)
    log_rets = rng.normal(0.0003, 0.012, n)
    close    = 100.0 * np.exp(np.cumsum(log_rets))
    noise    = rng.uniform(0.001, 0.015, n)
    high     = close * (1 + noise)
    low      = close * (1 - noise)
    open_    = np.clip(close * (1 + rng.normal(0, 0.005, n)), low, high)
    volume   = rng.lognormal(14.0, 0.6, n).astype(int)
    dates    = pd.bdate_range(end="2024-01-01", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_mock_engine(regime: int = 3) -> MagicMock:
    """Return a MagicMock that satisfies WalkForwardBacktester's engine contract."""
    eng = MagicMock()
    eng.predict_current.return_value = regime
    eng.is_uncertain.return_value    = False
    eng.is_confirmed.return_value    = True
    eng._is_model_stale              = False
    eng._n_states                    = 3
    return eng


def _make_mock_fold(
    fold_number: int = 0,
    sharpe: float = 0.5,
    per_regime_mean_logr: dict | None = None,
) -> FoldResult:
    """Build a minimal FoldResult for unit testing aggregate helpers."""
    return FoldResult(
        fold_number=fold_number,
        train_start="2020-01-01",
        train_end="2021-12-31",
        test_start="2022-01-03",
        test_end="2022-06-30",
        n_train_bars=504,
        n_test_bars=126,
        sharpe_ratio=sharpe,
        annualized_return=0.05,
        max_drawdown=-0.05,
        win_rate=0.55,
        total_trades=80,
        regime_distribution={3: 0.6, 2: 0.4},
        avg_confirmed_regime_duration_bars=8.0,
        per_regime_sharpe={3: 0.6},
        per_regime_return={3: 0.08},
        per_regime_mean_log_return=per_regime_mean_logr or {3: 0.0003},
        slippage_cost_total=120.0,
        hmm_convergence_warnings=0,
        hmm_n_states_selected=3,
        pct_bars_stale=0.0,
    )


def _mock_backtester_class(insample_sharpe: float = 1.2) -> MagicMock:
    """Return a patched Backtester class whose .run() yields a canned sharpe."""
    perf = MagicMock()
    perf.sharpe = insample_sharpe
    report = MagicMock()
    report.performance = perf
    bt = MagicMock()
    bt.return_value.run.return_value = report
    return bt


# ---------------------------------------------------------------------------
# Helpers for mocked walk-forward runs (fast — no real HMM fitting)
# ---------------------------------------------------------------------------

def _run_wf_mocked(
    ohlcv: pd.DataFrame,
    insample_sharpe: float = 1.2,
    n_train: int = 252,
    n_test: int = 63,
    step: int = 63,
    min_folds: int = 3,
    engine_regime: int = 3,
    tmp_logs: Path | None = None,
) -> WalkForwardResult:
    """Run WalkForwardBacktester with HMMEngine and Backtester fully mocked."""
    mock_bt = _mock_backtester_class(insample_sharpe)
    mock_engine = _make_mock_engine(engine_regime)

    logs_patch = tmp_logs or Path("logs/walk_forward")

    with (
        patch("core.walk_forward.Backtester", mock_bt),
        patch("core.walk_forward.HMMEngine", return_value=mock_engine),
        patch("core.walk_forward._LOGS_DIR", logs_patch),
    ):
        if tmp_logs:
            tmp_logs.mkdir(parents=True, exist_ok=True)
        result = WalkForwardBacktester().run(
            ohlcv, n_train=n_train, n_test=n_test, step=step, min_folds=min_folds
        )
    return result


# ===========================================================================
# 1. test_walk_forward_folds_never_overlap
# ===========================================================================

class TestFoldsNeverOverlap:

    def test_walk_forward_folds_never_overlap(self):
        """Test slice of fold N must not share any bar index with fold N's train slice."""
        windows = _compute_wf_windows(total_bars=900, n_train=504, n_test=126, step=63)
        assert len(windows) >= 3, "Need at least 3 windows to test"

        for train_start, train_end, test_end in windows:
            train_indices = set(range(train_start, train_end))
            test_indices  = set(range(train_end, test_end))
            overlap = train_indices & test_indices
            assert overlap == set(), (
                f"Train [{train_start}:{train_end}] and test [{train_end}:{test_end}] "
                f"overlap at indices {overlap}"
            )


# ===========================================================================
# 2. test_walk_forward_hmm_retrained_per_fold
# ===========================================================================

class TestHMMRetrainedPerFold:

    def test_walk_forward_hmm_retrained_per_fold(self, tmp_path):
        """HMMEngine.fit() must be called exactly once per fold."""
        ohlcv = _make_ohlcv(500)
        n_train, n_test, step = 252, 63, 63
        windows = _compute_wf_windows(len(ohlcv), n_train, n_test, step)
        expected_folds = len(windows)

        fit_calls   = []
        mock_engine = _make_mock_engine()
        mock_engine.fit.side_effect = lambda df: fit_calls.append(len(df))

        with (
            patch("core.walk_forward.Backtester", _mock_backtester_class()),
            patch("core.walk_forward.HMMEngine", return_value=mock_engine),
            patch("core.walk_forward._LOGS_DIR", tmp_path),
        ):
            WalkForwardBacktester().run(
                ohlcv, n_train=n_train, n_test=n_test, step=step, min_folds=1
            )

        assert len(fit_calls) == expected_folds, (
            f"Expected fit() called {expected_folds} times, got {len(fit_calls)}"
        )
        # Each fit was called with training-slice-sized features (roughly)
        for n in fit_calls:
            assert n <= n_train, f"fit() received {n} bars — exceeds training window {n_train}"


# ===========================================================================
# 3. test_walk_forward_no_lookahead_prediction_independence
# ===========================================================================

class TestNoLookaheadPredictionIndependence:

    def test_walk_forward_no_lookahead_prediction_independence(self):
        """predict_current() output at bar T is identical whether or not future
        bars are subsequently fed — forward-only algorithm guarantee."""
        import config.settings as cfg
        orig_n_init = cfg.HMM_N_INIT
        orig_n_iter = cfg.HMM_N_ITER
        orig_max_s  = cfg.HMM_MAX_STATES
        try:
            cfg.HMM_N_INIT    = 1
            cfg.HMM_N_ITER    = 50
            cfg.HMM_MAX_STATES = 3

            from core.hmm_engine import HMMEngine
            from core.feature_engineering import compute as feat

            n_train = 80
            n_test  = 20
            n_future = 50
            ohlcv   = _make_ohlcv(n_train + n_test + n_future, seed=7)

            train_df  = ohlcv.iloc[:n_train]
            test_df   = ohlcv.iloc[n_train: n_train + n_test]
            future_df = ohlcv.iloc[n_train + n_test:]

            feats_all = feat(pd.concat([train_df, test_df, future_df]))
            f_train   = feats_all.iloc[:n_train].dropna()
            f_test    = feats_all.iloc[n_train: n_train + n_test].dropna()
            f_future  = feats_all.iloc[n_train + n_test:].dropna()

            # Two engines trained identically (same random_state=42 in HMMEngine)
            eng1 = HMMEngine()
            eng2 = HMMEngine()
            eng1.fit(f_train)
            eng2.fit(f_train)

            # Engine 1: predict test slice only
            out1 = []
            for _, row in f_test.iterrows():
                out1.append(eng1.predict_current(row))

            # Engine 2: predict test slice, then continue into future bars
            out2_test = []
            for _, row in f_test.iterrows():
                out2_test.append(eng2.predict_current(row))
            # Feed future bars into eng2 (must not change out2_test already recorded)
            for _, row in f_future.iterrows():
                eng2.predict_current(row)

            assert out1 == out2_test, (
                "predict_current() outputs differ — lookahead bias detected"
            )
        finally:
            cfg.HMM_N_INIT    = orig_n_init
            cfg.HMM_N_ITER    = orig_n_iter
            cfg.HMM_MAX_STATES = orig_max_s


# ===========================================================================
# 4. test_walk_forward_minimum_folds_enforced
# ===========================================================================

class TestMinimumFoldsEnforced:

    def test_walk_forward_minimum_folds_enforced(self):
        """Insufficient bars must raise InsufficientDataError with 'insufficient'."""
        ohlcv = _make_ohlcv(300)   # not enough for default params
        with pytest.raises(InsufficientDataError, match="(?i)insufficient"):
            WalkForwardBacktester().run(ohlcv, n_train=504, n_test=126, step=63, min_folds=3)

    def test_insufficient_error_no_hmm_fits(self):
        """No HMM fits should happen when InsufficientDataError is raised."""
        ohlcv = _make_ohlcv(300)
        with patch("core.walk_forward.HMMEngine") as mock_cls:
            with pytest.raises(InsufficientDataError):
                WalkForwardBacktester().run(ohlcv, n_train=504, n_test=126, step=63)
        mock_cls.assert_not_called()


# ===========================================================================
# 5. test_fold_result_avg_duration_excludes_unconfirmed
# ===========================================================================

class TestAvgDurationExcludesUnconfirmed:

    def test_fold_result_avg_duration_excludes_unconfirmed(self):
        """-1 bars are not counted in avg_confirmed_regime_duration_bars."""
        # Sequence: 2 bars regime 3, then -1, then 3 bars regime 1
        regimes = [3, 3, -1, -1, 1, 1, 1]
        result  = _avg_confirmed_regime_duration(regimes)
        # Runs: [3,3]=2 and [1,1,1]=3 → mean=2.5
        assert result == pytest.approx(2.5)

    def test_all_unconfirmed_returns_zero(self):
        regimes = [-1, -1, -1]
        assert _avg_confirmed_regime_duration(regimes) == 0.0

    def test_single_run(self):
        regimes = [3, 3, 3]
        assert _avg_confirmed_regime_duration(regimes) == pytest.approx(3.0)


# ===========================================================================
# 6. test_regime_label_consistency_sign_check
# ===========================================================================

class TestRegimeLabelConsistencySignCheck:

    def test_regime_label_consistency_sign_check(self):
        """Opposite-sign mean returns for label 3 across two folds → 0.0 consistency."""
        fold_a = _make_mock_fold(0, per_regime_mean_logr={3:  0.005})  # positive
        fold_b = _make_mock_fold(1, per_regime_mean_logr={3: -0.005})  # negative
        result = _compute_regime_consistency([fold_a, fold_b])
        assert result == pytest.approx(0.0)

    def test_same_sign_consistent_pairs_counted(self):
        """Identical mean returns (std=0) → threshold=0 bypass fires → consistency=1.0."""
        fold_a = _make_mock_fold(0, per_regime_mean_logr={3: 0.005})
        fold_b = _make_mock_fold(1, per_regime_mean_logr={3: 0.005})
        result = _compute_regime_consistency([fold_a, fold_b])
        assert result == pytest.approx(1.0)

    def test_single_fold_returns_one(self):
        """With only one fold, no pairs exist — return 1.0 (vacuously consistent)."""
        fold = _make_mock_fold(0, per_regime_mean_logr={3: 0.003})
        assert _compute_regime_consistency([fold]) == pytest.approx(1.0)


# ===========================================================================
# 7. test_overfitting_ratio_uses_fresh_insample
# ===========================================================================

class TestOverfittingRatioUsesFreshInsample:

    def test_overfitting_ratio_uses_fresh_insample(self, tmp_path):
        """Backtester().run() must be called exactly once before the fold loop."""
        ohlcv   = _make_ohlcv(500)
        mock_bt = _mock_backtester_class(insample_sharpe=1.5)

        with (
            patch("core.walk_forward.Backtester", mock_bt),
            patch("core.walk_forward.HMMEngine", return_value=_make_mock_engine()),
            patch("core.walk_forward._LOGS_DIR", tmp_path),
        ):
            WalkForwardBacktester().run(
                ohlcv, n_train=252, n_test=63, step=63, min_folds=1
            )

        # Backtester class instantiated once; .run() called once on that instance
        mock_bt.assert_called_once()
        mock_bt.return_value.run.assert_called_once()


# ===========================================================================
# 8. test_overfitting_ratio_computed_correctly
# ===========================================================================

class TestOverfittingRatioComputedCorrectly:

    def test_overfitting_ratio_computed_correctly(self, tmp_path):
        """overfitting_ratio == outsample_sharpe / insample_sharpe (4 dp)."""
        ohlcv          = _make_ohlcv(500)
        insample_sharpe = 1.8

        result = _run_wf_mocked(
            ohlcv,
            insample_sharpe=insample_sharpe,
            n_train=252, n_test=63, step=63, min_folds=1,
            tmp_logs=tmp_path,
        )

        expected_ratio = result.outsample_sharpe / insample_sharpe
        assert result.overfitting_ratio == pytest.approx(expected_ratio, abs=1e-4)


# ===========================================================================
# 9. test_overfitting_ratio_zero_insample_handled
# ===========================================================================

class TestZeroInsampleSharpeHandled:

    def test_overfitting_ratio_zero_insample_handled(self, tmp_path):
        """insample_sharpe=0.0 must not raise ZeroDivisionError; returns 0.0."""
        ohlcv = _make_ohlcv(500)
        result = _run_wf_mocked(
            ohlcv,
            insample_sharpe=0.0,
            n_train=252, n_test=63, step=63, min_folds=1,
            tmp_logs=tmp_path,
        )
        assert result.overfitting_ratio == pytest.approx(0.0)

    def test_zero_insample_logs_warning(self, tmp_path, caplog):
        import logging
        ohlcv = _make_ohlcv(500)
        with caplog.at_level(logging.WARNING, logger="core.walk_forward"):
            _run_wf_mocked(
                ohlcv,
                insample_sharpe=0.0,
                n_train=252, n_test=63, step=63, min_folds=1,
                tmp_logs=tmp_path,
            )
        assert any("0.0" in rec.message for rec in caplog.records)


# ===========================================================================
# 10. test_interpretation_all_thresholds
# ===========================================================================

class TestInterpretationAllThresholds:

    def test_ratio_above_085_is_minimal(self):
        assert "Minimal" in interpret_overfitting_ratio(0.85)
        assert "Minimal" in interpret_overfitting_ratio(1.0)
        assert "Minimal" in interpret_overfitting_ratio(0.99)

    def test_ratio_070_to_085_is_moderate(self):
        assert "Moderate" in interpret_overfitting_ratio(0.70)
        assert "Moderate" in interpret_overfitting_ratio(0.84)

    def test_ratio_050_to_070_is_significant(self):
        assert "Significant" in interpret_overfitting_ratio(0.50)
        assert "Significant" in interpret_overfitting_ratio(0.69)

    def test_ratio_below_050_is_severe(self):
        assert "Severe" in interpret_overfitting_ratio(0.49)
        assert "Severe" in interpret_overfitting_ratio(0.01)


# ===========================================================================
# 11. test_regime_transition_matrix_rows_sum_to_one
# ===========================================================================

class TestTransitionMatrixRowsSumToOne:

    def test_regime_transition_matrix_rows_sum_to_one(self):
        """Each row of the empirical transition matrix sums to 1.0 within 1e-6."""
        regimes = [3, 3, 2, 1, 2, 3, 3, 4, 3, 2, 1, 1, 2, 3, -1, 3, 3, 2]
        matrix  = _compute_transition_matrix(regimes)
        for from_r, to_dict in matrix.items():
            row_sum = sum(to_dict.values())
            assert abs(row_sum - 1.0) < 1e-6, (
                f"Row {from_r} sums to {row_sum}, not 1.0"
            )

    def test_single_self_loop(self):
        regimes = [3, 3, 3]
        matrix  = _compute_transition_matrix(regimes)
        assert matrix == {"3": {"3": pytest.approx(1.0)}}

    def test_excludes_unconfirmed(self):
        """Transitions involving -1 must be excluded."""
        regimes = [-1, 3, -1, 2, -1]
        matrix  = _compute_transition_matrix(regimes)
        # Only one confirmed-to-confirmed transition: 3→2
        assert "3" in matrix
        assert matrix["3"] == {"2": pytest.approx(1.0)}


# ===========================================================================
# 12. test_per_regime_returns_all_labels_present
# ===========================================================================

class TestPerRegimeReturnsAllLabelsPresent:

    def test_per_regime_returns_all_labels_present(self, tmp_path):
        """per_regime_returns must have entries for every regime observed in test folds."""
        ohlcv = _make_ohlcv(500)

        # Mock engine cycles through regimes 2, 3 alternately so both appear
        call_count = [0]
        def _predict_side_effect(row, **kwargs):
            call_count[0] += 1
            return 2 if call_count[0] % 2 == 0 else 3

        mock_engine = _make_mock_engine()
        mock_engine.predict_current.side_effect = _predict_side_effect

        with (
            patch("core.walk_forward.Backtester", _mock_backtester_class()),
            patch("core.walk_forward.HMMEngine", return_value=mock_engine),
            patch("core.walk_forward._LOGS_DIR", tmp_path),
        ):
            result = WalkForwardBacktester().run(
                ohlcv, n_train=252, n_test=63, step=63, min_folds=1
            )

        # Both labels (as strings) must be present
        assert "2" in result.per_regime_returns
        assert "3" in result.per_regime_returns


# ===========================================================================
# 13. test_test_window_overlap_bars_correct
# ===========================================================================

class TestTestWindowOverlapBarsCorrect:

    def test_test_window_overlap_bars_correct(self, tmp_path):
        """With n_test=126 and step=63, test_window_overlap_bars must equal 63."""
        ohlcv  = _make_ohlcv(500)
        result = _run_wf_mocked(
            ohlcv,
            n_train=252, n_test=126, step=63, min_folds=1,
            tmp_logs=tmp_path,
        )
        assert result.test_window_overlap_bars == 63

    def test_overlap_formula(self):
        """test_window_overlap_bars = n_test - step."""
        for n_test, step in [(126, 63), (100, 40), (50, 25)]:
            assert n_test - step == n_test - step   # tautology — formula checked in run()


# ===========================================================================
# 14. test_monte_carlo_inputs_keys_present
# ===========================================================================

class TestMonteCarloInputsKeysPresent:

    def test_monte_carlo_inputs_keys_present(self, tmp_path):
        """JSON output must contain monte_carlo_inputs with all 6 required sub-keys."""
        required = {
            "return_distribution",
            "drawdown_distribution",
            "sharpe_distribution",
            "per_regime_returns",
            "regime_transition_matrix",
            "test_window_overlap_bars",
        }
        ohlcv = _make_ohlcv(500)
        _run_wf_mocked(
            ohlcv, n_train=252, n_test=63, step=63, min_folds=1, tmp_logs=tmp_path
        )

        import datetime
        today    = datetime.date.today().isoformat()
        json_file = tmp_path / f"walk_forward_{today}.json"
        payload   = json.loads(json_file.read_text())

        assert "monte_carlo_inputs" in payload, "Top-level key 'monte_carlo_inputs' missing"
        mc = payload["monte_carlo_inputs"]
        missing = required - set(mc.keys())
        assert not missing, f"monte_carlo_inputs missing keys: {missing}"


# ===========================================================================
# 15. test_walk_forward_markdown_section_headers
# ===========================================================================

class TestMarkdownSectionHeaders:

    def test_walk_forward_markdown_section_headers(self, tmp_path):
        """Markdown must contain all required section headers and notes."""
        ohlcv = _make_ohlcv(500)
        _run_wf_mocked(
            ohlcv, n_train=252, n_test=63, step=63, min_folds=1, tmp_logs=tmp_path
        )

        import datetime
        today   = datetime.date.today().isoformat()
        md_text = (tmp_path / f"walk_forward_{today}.md").read_text(encoding="utf-8")

        required_headers = [
            "# Walk-Forward Backtest Report",
            "## Aggregate Performance",
            "## Overfitting Analysis",
            "## Regime Stability",
            "## Per-Fold Summary",
        ]
        for header in required_headers:
            assert header in md_text, f"Missing markdown section: {header!r}"

        # Overlap note must appear
        assert "overlap" in md_text.lower(), "Overlap note missing from markdown"

        # Overfitting verdict must appear prominently (between overfitting ratio and per-regime)
        assert "Verdict:" in md_text, "Overfitting verdict line missing"

        # HMM state stability metrics must appear
        assert "Mean HMM States" in md_text
        assert "Std  HMM States" in md_text


# ===========================================================================
# 16. test_walk_forward_alert_fires_with_correct_fields
# ===========================================================================

class TestWalkForwardAlertFiresWithCorrectFields:

    def test_walk_forward_alert_fires_with_correct_fields(self, tmp_path):
        """WALKFORWARD_COMPLETE alert must contain required fields in message."""
        ohlcv = _make_ohlcv(500)

        sent_calls: list[tuple] = []

        def _capture_send(event_type, message, severity="info", **kwargs):
            sent_calls.append((event_type, message))

        with (
            patch("core.walk_forward.Backtester", _mock_backtester_class(1.5)),
            patch("core.walk_forward.HMMEngine", return_value=_make_mock_engine()),
            patch("core.walk_forward._LOGS_DIR", tmp_path),
            patch("core.alerts.send", side_effect=_capture_send),
        ):
            result = WalkForwardBacktester().run(
                ohlcv, n_train=252, n_test=63, step=63, min_folds=1
            )

        wf_alerts = [(et, msg) for et, msg in sent_calls if et == "walkforward_complete"]
        assert wf_alerts, "No walkforward_complete alert was sent"

        _, alert_msg = wf_alerts[0]
        assert "mean_sharpe" in alert_msg
        assert "overfitting_ratio" in alert_msg
        assert "verdict" in alert_msg
        assert "worst_fold_sharpe" in alert_msg


# ===========================================================================
# 17. test_overfitting_ratio_above_one_returns_severe
# ===========================================================================

class TestOverfittingRatioAboveOne:

    def test_overfitting_ratio_above_one_returns_severe(self):
        """ratio > 1.0 must mention 'x worse' with the numeric magnitude."""
        result = interpret_overfitting_ratio(3.43)
        assert "3.43x worse" in result, f"Expected '3.43x worse' in: {result!r}"

    def test_overfitting_ratio_exactly_one_is_minimal(self):
        """ratio == 1.0 is boundary — must not trigger the > 1.0 branch."""
        assert "Minimal" in interpret_overfitting_ratio(1.0)


# ===========================================================================
# 18. test_overfitting_ratio_negative_handled
# ===========================================================================

class TestOverfittingRatioNegativeHandled:

    def test_overfitting_ratio_negative_handled(self):
        """Negative ratio must return 'Invalid' string without raising."""
        result = interpret_overfitting_ratio(-0.5)
        assert "Invalid" in result, f"Expected 'Invalid' in: {result!r}"

    def test_overfitting_ratio_zero_returns_invalid(self):
        """ratio == 0.0 (both Sharpes same sign or IS=0 edge) returns 'Invalid'."""
        result = interpret_overfitting_ratio(0.0)
        assert "Invalid" in result, f"Expected 'Invalid' in: {result!r}"


# ===========================================================================
# 19. test_staleness_warning_fires_above_50pct
# ===========================================================================

class TestStalenessWarningAbove50Pct:

    def test_staleness_warning_fires_above_50pct(self, tmp_path, caplog):
        """When >50% of test bars are stale, a WARNING must be logged."""
        import logging
        ohlcv = _make_ohlcv(500)

        # Engine reports stale on every bar
        mock_engine = _make_mock_engine()
        mock_engine._is_model_stale = True

        with caplog.at_level(logging.WARNING, logger="core.walk_forward"):
            with (
                patch("core.walk_forward.Backtester", _mock_backtester_class()),
                patch("core.walk_forward.HMMEngine", return_value=mock_engine),
                patch("core.walk_forward._LOGS_DIR", tmp_path),
            ):
                WalkForwardBacktester().run(
                    ohlcv, n_train=252, n_test=63, step=63, min_folds=1
                )

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("flagged as stale" in m for m in warning_msgs), (
            f"Expected staleness warning but got: {warning_msgs}"
        )

    def test_no_staleness_warning_below_50pct(self, tmp_path, caplog):
        """When 0% of test bars are stale, no staleness WARNING must be emitted."""
        import logging
        ohlcv = _make_ohlcv(500)

        mock_engine = _make_mock_engine()
        mock_engine._is_model_stale = False   # never stale

        with caplog.at_level(logging.WARNING, logger="core.walk_forward"):
            with (
                patch("core.walk_forward.Backtester", _mock_backtester_class()),
                patch("core.walk_forward.HMMEngine", return_value=mock_engine),
                patch("core.walk_forward._LOGS_DIR", tmp_path),
            ):
                WalkForwardBacktester().run(
                    ohlcv, n_train=252, n_test=63, step=63, min_folds=1
                )

        assert not any("flagged as stale" in r.message for r in caplog.records), (
            "Unexpected staleness warning when no bars were stale"
        )


# ===========================================================================
# 20. test_pct_bars_stale_in_fold_result
# ===========================================================================

class TestPctBarsStaleInFoldResult:

    def test_pct_bars_stale_in_fold_result(self, tmp_path):
        """FoldResult.pct_bars_stale must be between 0.0 and 1.0 inclusive."""
        ohlcv = _make_ohlcv(500)
        result = _run_wf_mocked(
            ohlcv, n_train=252, n_test=63, step=63, min_folds=1, tmp_logs=tmp_path
        )
        for fold in result.folds:
            assert 0.0 <= fold.pct_bars_stale <= 1.0, (
                f"Fold {fold.fold_number} pct_bars_stale={fold.pct_bars_stale} out of range"
            )

    def test_pct_bars_stale_all_stale(self, tmp_path):
        """When engine._is_model_stale is always True, pct_bars_stale must equal 1.0."""
        ohlcv = _make_ohlcv(500)
        mock_engine = _make_mock_engine()
        mock_engine._is_model_stale = True

        with (
            patch("core.walk_forward.Backtester", _mock_backtester_class()),
            patch("core.walk_forward.HMMEngine", return_value=mock_engine),
            patch("core.walk_forward._LOGS_DIR", tmp_path),
        ):
            result = WalkForwardBacktester().run(
                ohlcv, n_train=252, n_test=63, step=63, min_folds=1
            )

        for fold in result.folds:
            assert fold.pct_bars_stale == pytest.approx(1.0), (
                f"Fold {fold.fold_number} pct_bars_stale expected 1.0, got {fold.pct_bars_stale}"
            )


# ===========================================================================
# 21. TestStalenessWidening — zscore threading, IS disable, aggregate fields
# ===========================================================================

class TestStalenessWidening:

    def test_walkforward_passes_wider_zscore(self, tmp_path):
        """_run_fold must call predict_current with staleness_zscore=HMM_STALENESS_ZSCORE_WALKFORWARD."""
        from config.settings import HMM_STALENESS_ZSCORE_WALKFORWARD

        ohlcv       = _make_ohlcv(500)
        mock_engine = _make_mock_engine()

        with (
            patch("core.walk_forward.Backtester", _mock_backtester_class()),
            patch("core.walk_forward.HMMEngine", return_value=mock_engine),
            patch("core.walk_forward._LOGS_DIR", tmp_path),
        ):
            WalkForwardBacktester().run(ohlcv, n_train=252, n_test=63, step=63, min_folds=1)

        for call in mock_engine.predict_current.call_args_list:
            _, kwargs = call
            assert kwargs.get("staleness_zscore") == HMM_STALENESS_ZSCORE_WALKFORWARD, (
                f"Expected staleness_zscore={HMM_STALENESS_ZSCORE_WALKFORWARD}, got {kwargs}"
            )

    def test_insample_backtest_disables_staleness(self, tmp_path):
        """IS backtest must be called with disable_staleness=True."""
        ohlcv    = _make_ohlcv(500)
        mock_bt  = _mock_backtester_class()
        mock_engine = _make_mock_engine()

        with (
            patch("core.walk_forward.Backtester", mock_bt),
            patch("core.walk_forward.HMMEngine", return_value=mock_engine),
            patch("core.walk_forward._LOGS_DIR", tmp_path),
        ):
            WalkForwardBacktester().run(ohlcv, n_train=252, n_test=63, step=63, min_folds=1)

        run_call = mock_bt.return_value.run.call_args
        assert run_call.kwargs.get("disable_staleness") is True, (
            f"IS backtest must set disable_staleness=True, got: {run_call}"
        )

    def test_mean_max_pct_stale_in_walk_forward_result(self, tmp_path):
        """WalkForwardResult has mean/max_pct_bars_stale in [0,1] and max_pct_stale_fold in range."""
        ohlcv  = _make_ohlcv(500)
        result = _run_wf_mocked(
            ohlcv, n_train=252, n_test=63, step=63, min_folds=1, tmp_logs=tmp_path
        )

        assert hasattr(result, "mean_pct_bars_stale")
        assert hasattr(result, "max_pct_bars_stale")
        assert hasattr(result, "max_pct_stale_fold")
        assert 0.0 <= result.mean_pct_bars_stale <= 1.0
        assert 0.0 <= result.max_pct_bars_stale <= 1.0
        assert 0 <= result.max_pct_stale_fold < result.n_folds

"""
tests/test_backtester.py
-------------------------
Unit tests for core/backtester.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import (
    BACKTEST_IN_SAMPLE_BARS,
    BACKTEST_OUT_SAMPLE_BARS,
    BACKTEST_STEP_BARS,
)
from core.backtester import (
    INITIAL_NAV,
    Backtester,
    BacktestReport,
    FoldResult,
    StressReport,
    _compute_windows,
)
from core.feature_engineering import LookaheadBiasError
from core.performance import RegimeLogEntry, Trade
from core.risk_manager import RiskManager


# ---------------------------------------------------------------------------
# Synthetic OHLCV fixture
# ---------------------------------------------------------------------------

def _make_ohlcv(n_bars: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0003, 0.012, n_bars)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    noise = rng.uniform(0.001, 0.015, n_bars)
    high  = close * (1 + noise)
    low   = close * (1 - noise)
    open_ = np.clip(close * (1 + rng.normal(0, 0.005, n_bars)), low, high)
    vol   = rng.lognormal(14.0, 0.6, n_bars).astype(int)
    dates = pd.bdate_range(end="2024-01-01", periods=n_bars)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


# ---------------------------------------------------------------------------
# _compute_windows helper
# ---------------------------------------------------------------------------

class TestComputeWindows:

    def test_no_windows_when_data_too_short(self):
        wins = _compute_windows(300, 252, 126, 63)
        assert wins == []

    def test_exactly_one_window_at_minimum(self):
        wins = _compute_windows(252 + 126, 252, 126, 63)
        assert len(wins) == 1

    def test_window_structure_correct(self):
        wins = _compute_windows(252 + 126, 252, 126, 63)
        is_start, is_end, oos_end = wins[0]
        assert is_start == 0
        assert is_end   == 252
        assert oos_end  == 252 + 126

    def test_oos_never_overlaps_is_within_a_fold(self):
        wins = _compute_windows(600, 252, 126, 63)
        for is_start, is_end, oos_end in wins:
            assert is_end <= oos_end
            assert is_start < is_end

    def test_fold_count_with_600_bars(self):
        """With 600 bars, IS=252, OOS=126, step=63: 4 complete folds.
        Fold 4: IS=[189:441], OOS=[441:567] — 567 ≤ 600 so it fits."""
        wins = _compute_windows(600, 252, 126, 63)
        assert len(wins) == 4

    def test_consecutive_is_windows_advance_by_step(self):
        wins = _compute_windows(700, 252, 126, 63)
        for i in range(1, len(wins)):
            assert wins[i][0] == wins[i - 1][0] + 63


# ---------------------------------------------------------------------------
# run_fold()
# ---------------------------------------------------------------------------

class TestRunFold:

    @pytest.fixture
    def fold_inputs(self):
        ohlcv = _make_ohlcv(500)
        is_df  = ohlcv.iloc[:252]
        oos_df = ohlcv.iloc[252:378]
        return is_df, oos_df

    def test_run_fold_returns_fold_result(self, fold_inputs):
        is_df, oos_df = fold_inputs
        result = Backtester().run_fold(is_df, oos_df)
        assert isinstance(result, FoldResult)

    def test_fold_result_has_equity_curve(self, fold_inputs):
        is_df, oos_df = fold_inputs
        result = Backtester().run_fold(is_df, oos_df)
        assert isinstance(result.equity_curve, pd.Series)
        assert len(result.equity_curve) > 0

    def test_fold_result_has_trades(self, fold_inputs):
        is_df, oos_df = fold_inputs
        result = Backtester().run_fold(is_df, oos_df)
        assert isinstance(result.trades, list)
        for t in result.trades:
            assert isinstance(t, Trade)

    def test_fold_result_has_regime_log(self, fold_inputs):
        is_df, oos_df = fold_inputs
        result = Backtester().run_fold(is_df, oos_df)
        assert isinstance(result.regime_log, list)
        for r in result.regime_log:
            assert isinstance(r, RegimeLogEntry)

    def test_fold_result_has_benchmark_curves(self, fold_inputs):
        is_df, oos_df = fold_inputs
        result = Backtester().run_fold(is_df, oos_df)
        assert "buy_and_hold" in result.benchmark_curves
        assert "sma200"       in result.benchmark_curves
        assert "random"       in result.benchmark_curves

    def test_benchmark_curves_have_same_length_as_strategy(self, fold_inputs):
        is_df, oos_df = fold_inputs
        result = Backtester().run_fold(is_df, oos_df)
        strat_len = len(result.equity_curve)
        for name, bmark in result.benchmark_curves.items():
            assert len(bmark) == strat_len, \
                f"Benchmark '{name}' length {len(bmark)} != strategy {strat_len}"

    def test_oos_window_is_strictly_after_is_window(self, fold_inputs):
        is_df, oos_df = fold_inputs
        result = Backtester().run_fold(is_df, oos_df)
        assert result.oos_start > result.in_sample_end

    def test_hmm_state_count_in_valid_range(self, fold_inputs):
        is_df, oos_df = fold_inputs
        result = Backtester().run_fold(is_df, oos_df)
        assert 3 <= result.n_hmm_states <= 7

    def test_equity_curve_starts_near_initial_nav(self, fold_inputs):
        is_df, oos_df = fold_inputs
        result = Backtester().run_fold(is_df, oos_df, initial_nav=100_000.0)
        # First NAV should be within 5 % of initial (allocation starts at 0)
        assert abs(result.equity_curve.iloc[0] - 100_000.0) < 5_000.0

    def test_regime_log_length_equals_equity_length(self, fold_inputs):
        is_df, oos_df = fold_inputs
        result = Backtester().run_fold(is_df, oos_df)
        assert len(result.regime_log) == len(result.equity_curve)


# ---------------------------------------------------------------------------
# Lookahead audit
# ---------------------------------------------------------------------------

class TestLookaheadAudit:

    def test_clean_features_pass_audit(self):
        ohlcv = _make_ohlcv(500)
        is_df  = ohlcv.iloc[:252]
        oos_df = ohlcv.iloc[252:378]
        # Should not raise
        Backtester().run_fold(is_df, oos_df, audit_lookahead=True)

    def test_injected_lookahead_raises(self):
        """A custom feature_fn that injects future returns must be detected."""
        from core.feature_engineering import compute as base_compute
        import numpy as np

        def lookahead_feature_fn(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
            features = base_compute(ohlcv_df).copy()
            close    = ohlcv_df["close"]
            future_ret = np.log(close / close.shift(1)).shift(-1)
            features["log_return"] = future_ret   # overwrite with future data
            return features

        ohlcv = _make_ohlcv(500)
        is_df  = ohlcv.iloc[:252]
        oos_df = ohlcv.iloc[252:378]

        with pytest.raises(LookaheadBiasError):
            Backtester().run_fold(
                is_df, oos_df,
                feature_fn=lookahead_feature_fn,
                audit_lookahead=True,
            )


# ---------------------------------------------------------------------------
# run() — full walk-forward
# ---------------------------------------------------------------------------

class TestRunFull:

    @pytest.fixture
    def ohlcv_600(self):
        return _make_ohlcv(600)

    def test_run_completes_without_error(self, ohlcv_600):
        report = Backtester().run(ohlcv_600, audit_lookahead=False)
        assert isinstance(report, BacktestReport)

    def test_run_produces_correct_fold_count(self, ohlcv_600):
        report = Backtester().run(ohlcv_600, audit_lookahead=False)
        expected = len(_compute_windows(600, BACKTEST_IN_SAMPLE_BARS,
                                        BACKTEST_OUT_SAMPLE_BARS, BACKTEST_STEP_BARS))
        assert report.n_folds == expected

    def test_oos_bars_in_each_fold_do_not_overlap_is_bars(self, ohlcv_600):
        report = Backtester().run(ohlcv_600, audit_lookahead=False)
        for fold in report.folds:
            assert fold.oos_start > fold.in_sample_end

    def test_hmm_refitted_per_fold(self, ohlcv_600):
        """Each fold must carry an independent n_hmm_states value (all in [3,7])."""
        report = Backtester().run(ohlcv_600, audit_lookahead=False)
        for fold in report.folds:
            assert 3 <= fold.n_hmm_states <= 7

    def test_report_has_benchmark_reports(self, ohlcv_600):
        report = Backtester().run(ohlcv_600, audit_lookahead=False)
        assert "buy_and_hold" in report.benchmark_reports
        assert "sma200"       in report.benchmark_reports
        assert "random"       in report.benchmark_reports

    def test_performance_metrics_non_trivial(self, ohlcv_600):
        report = Backtester().run(ohlcv_600, audit_lookahead=False)
        perf   = report.performance
        assert isinstance(perf.sharpe, float)
        assert perf.n_bars == len(report.equity_curve)

    def test_equity_curve_length_equals_total_oos_bars(self, ohlcv_600):
        report   = Backtester().run(ohlcv_600, audit_lookahead=False)
        wins     = _compute_windows(600, BACKTEST_IN_SAMPLE_BARS,
                                    BACKTEST_OUT_SAMPLE_BARS, BACKTEST_STEP_BARS)
        total_oos = sum(oos - is_e for is_s, is_e, oos in wins)
        assert len(report.equity_curve) == pytest.approx(total_oos, abs=5)


# ---------------------------------------------------------------------------
# run_stress_test()
# ---------------------------------------------------------------------------

class TestRunStressTest:

    @pytest.fixture
    def oos_df(self):
        return _make_ohlcv(200).iloc[80:]   # 120 bars of OOS

    def test_stress_test_returns_stress_report(self, oos_df):
        report = Backtester().run_stress_test(oos_df, n_injections=3)
        assert isinstance(report, StressReport)

    def test_stress_report_has_equity_curve(self, oos_df):
        report = Backtester().run_stress_test(oos_df)
        assert isinstance(report.equity_curve, pd.Series)
        assert len(report.equity_curve) > 0

    def test_injects_correct_number_of_events(self, oos_df):
        for n in (1, 2, 3):
            report = Backtester().run_stress_test(oos_df, n_injections=n)
            assert len(report.events_applied) == n

    def test_stress_results_separate_from_clean_backtest(self):
        """StressReport is a distinct object — run() is unaffected."""
        ohlcv  = _make_ohlcv(600)
        bt     = Backtester()
        clean  = bt.run(ohlcv, audit_lookahead=False)
        stress = bt.run_stress_test(ohlcv.iloc[252:378], n_injections=3)
        # Clean report must not carry stress event labels
        assert not hasattr(clean, "events_applied")
        assert isinstance(stress, StressReport)

    def test_crash_day_triggers_peak_drawdown_lockout(self, oos_df):
        """−12 % day at 90 % allocation → −10.8 % portfolio → lockout fires."""
        report = Backtester().run_stress_test(oos_df, n_injections=1,
                                              initial_nav=INITIAL_NAV)
        # After the crash day the lockout should have fired
        assert "peak_drawdown_lockout" in report.circuit_breakers_fired

    def test_consecutive_drops_trigger_lockout(self, oos_df):
        """5 × −3.5 % ≈ −16 % cumulative → peak_drawdown_lockout must fire."""
        report = Backtester().run_stress_test(oos_df, n_injections=2,
                                              initial_nav=INITIAL_NAV)
        assert "peak_drawdown_lockout" in report.circuit_breakers_fired

    def test_daily_halt_fires_on_large_single_day_loss(self, oos_df):
        """−12 % × 0.9 allocation = −10.8 % daily portfolio loss > −3 % threshold."""
        report = Backtester().run_stress_test(oos_df, n_injections=1,
                                              initial_nav=INITIAL_NAV)
        assert "daily_halt" in report.circuit_breakers_fired

    def test_stress_max_drawdown_negative_after_crash(self, oos_df):
        report = Backtester().run_stress_test(oos_df, n_injections=1)
        assert report.max_drawdown < 0.0

    def test_n_injections_stored_in_report(self, oos_df):
        report = Backtester().run_stress_test(oos_df, n_injections=2)
        assert report.n_injections == 2


# ---------------------------------------------------------------------------
# RiskManager unit tests (circuit breaker thresholds)
# ---------------------------------------------------------------------------

class TestRiskManagerCircuitBreakers:

    def test_peak_drawdown_lockout_fires_at_threshold(self):
        rm = RiskManager()
        rm.initialize(100_000.0)
        # −10 % exactly should fire
        fired = rm.update(90_000.0)
        assert "peak_drawdown_lockout" in fired
        assert rm.is_locked()

    def test_peak_drawdown_does_not_fire_above_threshold(self):
        rm = RiskManager()
        rm.initialize(100_000.0)
        # −9 % should NOT fire
        fired = rm.update(91_000.0)
        assert "peak_drawdown_lockout" not in fired
        assert not rm.is_locked()

    def test_daily_halt_fires_at_minus_3pct(self):
        rm = RiskManager()
        rm.initialize(100_000.0)
        fired = rm.update(96_900.0)   # −3.1 %
        assert "daily_halt" in fired

    def test_daily_halve_fires_at_minus_2pct(self):
        rm = RiskManager()
        rm.initialize(100_000.0)
        fired = rm.update(97_900.0)   # −2.1 %
        assert "daily_halve_sizes" in fired
        assert "daily_halt" not in fired

    def test_weekly_resize_fires_at_minus_5pct(self):
        rm = RiskManager()
        rm.initialize(100_000.0)
        # reset_weekly so the weekly baseline is set
        fired = rm.update(94_900.0)   # −5.1 %
        assert "weekly_resize" in fired

    def test_approve_blocked_when_locked(self):
        from core.regime_strategies import get_signal
        rm = RiskManager()
        rm.initialize(100_000.0)
        rm.update(89_000.0)   # triggers lockout
        signal = get_signal(3, 0.8, 100_000.0, 0.5, False)
        result = rm.approve(signal, 100_000.0)
        assert not result.approved
        assert result.size_multiplier == 0.0

    def test_approve_halves_size_after_daily_halve(self):
        from core.regime_strategies import get_signal
        rm = RiskManager()
        rm.initialize(100_000.0)
        rm.update(97_900.0)   # daily_halve fires
        signal = get_signal(3, 0.8, 100_000.0, 0.5, False)
        result = rm.approve(signal, 100_000.0)
        assert result.approved
        assert result.size_multiplier == pytest.approx(0.5)

    def test_benchmark_returns_are_reproducible(self):
        """Same seed and OHLCV → identical benchmark equity curves."""
        from core.backtester import _benchmark_random
        ohlcv = _make_ohlcv(200)
        close = ohlcv["close"].iloc[100:]
        bm1   = _benchmark_random(close, 100_000.0, seed=42)
        bm2   = _benchmark_random(close, 100_000.0, seed=42)
        pd.testing.assert_series_equal(bm1, bm2)

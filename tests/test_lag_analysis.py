"""
tests/test_lag_analysis.py
--------------------------
Unit and integration tests for core/lag_analysis.py and the backtester's
regime transition lag tracking.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from core.lag_analysis import (
    LagTransition,
    build_lag_report,
    compute_lag_transitions,
    write_lag_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lag_inputs(
    raw: list[int],
    confirmed: list[int],
    closes: list[float] | None = None,
    navs: list[float] | None = None,
    allocs: list[float] | None = None,
    ticker: str = "SPY",
    bar_interval_secs: int = 86400,
):
    n = len(raw)
    closes = closes or [100.0] * n
    navs   = navs   or [100_000.0] * n
    allocs = allocs or [0.8] * n
    ts     = list(pd.date_range("2024-01-01", periods=n, freq="D"))
    return compute_lag_transitions(
        raw_regimes=raw,
        confirmed_regimes=confirmed,
        close_prices=closes,
        nav_history=navs,
        alloc_history=allocs,
        timestamps=ts,
        ticker=ticker,
        bar_interval_secs=bar_interval_secs,
    )


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
# TestComputeLagTransitions — unit tests on synthetic sequences
# ---------------------------------------------------------------------------

class TestComputeLagTransitions:

    def test_lag_analysis_identifies_raw_transition(self):
        """raw_transition_bar is the first bar where HMM raw state shows new regime."""
        # raw:       [3, 3, 0, 3, 0, 0, 0] — 0 first appears at bar 2, then wobbles
        # confirmed: [3, 3, 3, 3, 3, 3, 0] — confirmation fires at bar 6
        raw       = [3, 3, 0, 3, 0, 0, 0]
        confirmed = [3, 3, 3, 3, 3, 3, 0]

        transitions = _make_lag_inputs(raw, confirmed)

        assert len(transitions) == 1
        t = transitions[0]
        assert t.from_regime == 3
        assert t.to_regime   == 0
        assert t.raw_transition_bar       == 2   # first bar where raw == 0
        assert t.confirmed_transition_bar == 6

    def test_lag_analysis_confirmation_gate_counted(self):
        """lag_bars equals exactly confirmed_bar minus raw_transition_bar."""
        # Clean run: raw switches to 0 at bar 2 and stays; confirmed fires at bar 4
        # (CONFIRMATION_BARS=3 → bars 2, 3, 4 all raw=0)
        raw       = [3, 3, 0, 0, 0]
        confirmed = [3, 3, 3, 3, 0]

        transitions = _make_lag_inputs(raw, confirmed)

        assert len(transitions) == 1
        t = transitions[0]
        assert t.lag_bars == t.confirmed_transition_bar - t.raw_transition_bar
        assert t.raw_transition_bar       == 2
        assert t.confirmed_transition_bar == 4
        assert t.lag_bars                 == 2

    def test_lag_hours_derived_from_lag_bars_and_interval(self):
        """lag_hours = lag_bars * bar_interval_secs / 3600."""
        raw       = [3, 3, 0, 0, 0]
        confirmed = [3, 3, 3, 3, 0]
        transitions = _make_lag_inputs(raw, confirmed, bar_interval_secs=86400)
        assert transitions[0].lag_hours == pytest.approx(2 * 86400 / 3600)

    def test_price_change_computed_correctly(self):
        """price_change_pct = (price_at_confirmed - price_at_raw) / price_at_raw * 100."""
        raw       = [3, 3, 0, 0, 0]
        confirmed = [3, 3, 3, 3, 0]
        closes    = [100.0, 100.0, 98.0, 97.0, 95.0]

        transitions = _make_lag_inputs(raw, confirmed, closes=closes)
        t = transitions[0]

        # raw_bar=2 → price=98.0; confirmed_bar=4 → price=95.0
        expected_pct = (95.0 - 98.0) / 98.0 * 100.0
        assert t.price_change_pct == pytest.approx(expected_pct, rel=1e-4)

    def test_estimated_damage_uses_nav_and_allocation(self):
        """estimated_damage_usd = nav_at_raw * |price_change_pct/100| * alloc."""
        raw       = [3, 3, 0, 0, 0]
        confirmed = [3, 3, 3, 3, 0]
        closes    = [100.0, 100.0, 95.0, 94.0, 93.0]
        navs      = [100_000.0] * 5
        allocs    = [0.80] * 5

        transitions = _make_lag_inputs(raw, confirmed, closes=closes,
                                        navs=navs, allocs=allocs)
        t = transitions[0]
        expected = navs[2] * abs(t.price_change_pct / 100.0) * allocs[2]
        assert t.estimated_damage_usd == pytest.approx(expected, rel=1e-4)

    def test_lag_analysis_zero_lag_edge_case(self):
        """Handles zero lag (raw and confirmed transition on same bar) without errors."""
        # raw and confirmed both switch at bar 1
        raw       = [3, 0]
        confirmed = [3, 0]

        transitions = _make_lag_inputs(raw, confirmed)

        assert len(transitions) == 1
        t = transitions[0]
        assert t.lag_bars       == 0
        assert t.lag_hours      == 0.0
        assert t.raw_transition_bar == t.confirmed_transition_bar

    def test_no_transitions_when_regime_stable(self):
        """No transitions recorded when confirmed regime never changes."""
        raw       = [3, 3, 3, 3, 3]
        confirmed = [3, 3, 3, 3, 3]
        assert _make_lag_inputs(raw, confirmed) == []

    def test_no_transitions_while_unconfirmed(self):
        """Bars with confirmed == -1 are skipped."""
        raw       = [0, 0, 0, 0, 3]
        confirmed = [-1, -1, -1, -1, 3]
        assert _make_lag_inputs(raw, confirmed) == []

    def test_multiple_transitions_tracked_independently(self):
        """Two confirmed transitions produce two LagTransition records."""
        #           0  1  2  3  4  5  6  7  8  9  10
        raw       = [3, 3, 0, 0, 0, 0, 2, 2, 2, 2, 2]
        confirmed = [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 2]

        transitions = _make_lag_inputs(raw, confirmed)
        assert len(transitions) == 2
        assert transitions[0].from_regime == 3
        assert transitions[0].to_regime   == 0
        assert transitions[1].from_regime == 0
        assert transitions[1].to_regime   == 2

    def test_search_window_resets_after_each_transition(self):
        """raw_bar for the second transition is searched from after the first
        confirmed transition, not from bar 0."""
        #           0  1  2  3  4  5  6  7  8
        raw       = [3, 2, 2, 2, 2, 0, 0, 0, 0]
        confirmed = [3, 3, 3, 2, 2, 2, 2, 2, 0]

        transitions = _make_lag_inputs(raw, confirmed)
        assert len(transitions) == 2

        # Second transition: confirmed at bar 8, to_regime=0
        # Search starts from bar 4 (after first confirmed at bar 3)
        # First raw==0 in that window is bar 5
        t2 = transitions[1]
        assert t2.to_regime          == 0
        assert t2.raw_transition_bar == 5
        assert t2.confirmed_transition_bar == 8


# ---------------------------------------------------------------------------
# TestBuildLagReport
# ---------------------------------------------------------------------------

class TestBuildLagReport:

    def test_lag_analysis_crash_transitions_isolated(self):
        """by_transition_type correctly separates crash-bound from non-crash."""
        raw       = [3, 3, 0, 0, 0, 0, 0, 2, 2, 2, 2]
        confirmed = [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 2]

        transitions = _make_lag_inputs(raw, confirmed)
        report = build_lag_report(transitions, bar_interval_secs=86400)

        crash     = {k: v for k, v in report["by_transition_type"].items() if "to_0" in k}
        non_crash = {k: v for k, v in report["by_transition_type"].items() if "to_0" not in k}

        assert "3_to_0" in crash
        assert len(crash)     == 1
        assert "0_to_2" in non_crash
        assert len(non_crash) == 1

    def test_report_has_required_top_level_keys(self):
        transitions = _make_lag_inputs([3, 3, 0, 0, 0], [3, 3, 3, 3, 0])
        report = build_lag_report(transitions, bar_interval_secs=86400)
        for key in ("summary", "by_transition_type", "all_transitions"):
            assert key in report

    def test_summary_has_required_fields(self):
        transitions = _make_lag_inputs([3, 3, 0, 0, 0], [3, 3, 3, 3, 0])
        report = build_lag_report(transitions, bar_interval_secs=86400)
        s = report["summary"]
        for field in (
            "total_transitions", "crash_transitions",
            "mean_lag_bars", "mean_lag_hours",
            "worst_lag_bars", "worst_lag_hours",
            "mean_price_damage_pct", "worst_price_damage_pct",
            "mean_portfolio_damage_usd", "worst_portfolio_damage_usd",
        ):
            assert field in s, f"summary missing '{field}'"

    def test_empty_transitions_returns_zero_summary(self):
        report = build_lag_report([], bar_interval_secs=86400)
        s = report["summary"]
        assert s["total_transitions"] == 0
        assert s["crash_transitions"] == 0
        assert s["mean_lag_bars"]     == 0.0
        assert report["all_transitions"] == []

    def test_crash_transition_count_correct(self):
        # Two transitions: 3→0 (crash) and 0→2 (non-crash, no bounce between)
        raw       = [3, 3, 0, 0, 0, 2, 2, 2, 2]
        confirmed = [3, 3, 3, 3, 0, 0, 0, 0, 2]

        transitions = _make_lag_inputs(raw, confirmed)
        report = build_lag_report(transitions, bar_interval_secs=86400)
        assert report["summary"]["crash_transitions"] == 1
        assert report["summary"]["total_transitions"] == 2

    def test_all_transitions_serialisable_as_dicts(self):
        transitions = _make_lag_inputs([3, 3, 0, 0, 0], [3, 3, 3, 3, 0])
        report = build_lag_report(transitions, bar_interval_secs=86400)
        for item in report["all_transitions"]:
            assert isinstance(item, dict)
            for field in ("ticker", "date", "from_regime", "to_regime",
                          "lag_bars", "lag_hours", "price_change_pct",
                          "nav_at_transition", "estimated_damage_usd"):
                assert field in item


# ---------------------------------------------------------------------------
# TestWriteLagReport
# ---------------------------------------------------------------------------

class TestWriteLagReport:

    def test_lag_report_written_to_file(self, tmp_path):
        """JSON file is written to the specified directory."""
        transitions = _make_lag_inputs([3, 3, 0, 0, 0], [3, 3, 3, 3, 0])
        report = build_lag_report(transitions, bar_interval_secs=86400)

        out = write_lag_report(report, tmp_path)

        assert out.exists()
        assert out.name == "regime_lag_analysis.json"

    def test_lag_report_file_contains_required_keys(self, tmp_path):
        transitions = _make_lag_inputs([3, 3, 0, 0, 0], [3, 3, 3, 3, 0])
        report = build_lag_report(transitions, bar_interval_secs=86400)
        out = write_lag_report(report, tmp_path)

        loaded = json.loads(out.read_text(encoding="utf-8"))
        for key in ("summary", "by_transition_type", "all_transitions"):
            assert key in loaded

    def test_lag_report_creates_missing_directory(self, tmp_path):
        """write_lag_report creates the target directory if it doesn't exist."""
        new_dir = tmp_path / "nested" / "logs"
        transitions = _make_lag_inputs([3, 3, 0, 0, 0], [3, 3, 3, 3, 0])
        report = build_lag_report(transitions, bar_interval_secs=86400)
        write_lag_report(report, new_dir)
        assert (new_dir / "regime_lag_analysis.json").exists()


# ---------------------------------------------------------------------------
# TestBacktesterLagIntegration — end-to-end via Backtester.run_fold()
# ---------------------------------------------------------------------------

class TestBacktesterLagIntegration:

    @pytest.fixture
    def fold_inputs(self):
        ohlcv = _make_ohlcv(500)
        return ohlcv.iloc[:252], ohlcv.iloc[252:378]

    def test_fold_result_has_lag_transitions_field(self, fold_inputs):
        from core.backtester import Backtester
        is_df, oos_df = fold_inputs
        result = Backtester().run_fold(is_df, oos_df, audit_lookahead=False)
        assert hasattr(result, "lag_transitions")
        assert isinstance(result.lag_transitions, list)

    def test_lag_transitions_are_lag_transition_instances(self, fold_inputs):
        from core.backtester import Backtester
        is_df, oos_df = fold_inputs
        result = Backtester().run_fold(is_df, oos_df, audit_lookahead=False)
        for t in result.lag_transitions:
            assert isinstance(t, LagTransition)

    def test_lag_transitions_have_valid_regime_labels(self, fold_inputs):
        from core.backtester import Backtester
        is_df, oos_df = fold_inputs
        result = Backtester().run_fold(is_df, oos_df, audit_lookahead=False)
        for t in result.lag_transitions:
            assert 0 <= t.from_regime <= 4, f"from_regime {t.from_regime} out of range"
            assert 0 <= t.to_regime   <= 4, f"to_regime {t.to_regime} out of range"

    def test_lag_report_written_after_full_run(self, tmp_path, monkeypatch):
        """Backtester.run() writes regime_lag_analysis.json to LOGS_DIR."""
        import core.backtester as bt_mod
        from core.backtester import Backtester
        monkeypatch.setattr(bt_mod, "LOGS_DIR", tmp_path)

        ohlcv = _make_ohlcv(600)
        Backtester().run(ohlcv, audit_lookahead=False)

        report_file = tmp_path / "regime_lag_analysis.json"
        assert report_file.exists(), "regime_lag_analysis.json was not written"

        loaded = json.loads(report_file.read_text(encoding="utf-8"))
        for key in ("summary", "by_transition_type", "all_transitions"):
            assert key in loaded, f"report missing top-level key '{key}'"

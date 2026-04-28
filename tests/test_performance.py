"""
tests/test_performance.py
--------------------------
Unit tests for core/performance.py.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.performance import (
    RISK_FREE_RATE,
    TRADING_DAYS,
    BucketStats,
    ComparisonTable,
    PerformanceReport,
    RegimeLogEntry,
    RegimeStats,
    Trade,
    compare_benchmarks,
    compute,
    _max_drawdown,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_equity(n: int = 252, nav: float = 100_000.0) -> pd.Series:
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(np.full(n, nav), index=dates)


def _growing_equity(n: int = 252, daily_ret: float = 0.001) -> pd.Series:
    dates = pd.bdate_range("2022-01-03", periods=n)
    rng   = np.random.default_rng(42)
    noisy = daily_ret + rng.normal(0, 0.005, n)
    nav   = 100_000.0 * np.cumprod(1 + noisy)
    return pd.Series(nav, index=dates)


def _drawdown_equity() -> pd.Series:
    """Series with a known -20 % drawdown."""
    nav = np.array([100, 110, 120, 100, 96, 90, 95, 100, 105], dtype=float) * 1_000
    dates = pd.bdate_range("2022-01-03", periods=len(nav))
    return pd.Series(nav, index=dates)


def _make_trades(
    n: int,
    pnl_per_trade: float,
    confidence: float = 0.8,
    regime: int = 3,
) -> list[Trade]:
    dates = pd.bdate_range("2022-01-03", periods=n)
    return [
        Trade(timestamp=d, regime=regime, confidence=confidence,
              pnl=pnl_per_trade, return_pct=pnl_per_trade / 1_000)
        for d in dates
    ]


def _make_regime_log(equity: pd.Series, regime: int = 3) -> list[RegimeLogEntry]:
    return [
        RegimeLogEntry(timestamp=ts, regime=regime, confidence=0.8,
                       is_uncertain=False, allocation_pct=0.9)
        for ts in equity.index
    ]


# ---------------------------------------------------------------------------
# compute() — required fields
# ---------------------------------------------------------------------------

class TestComputeRequiredFields:

    def test_returns_performance_report(self):
        eq  = _growing_equity()
        rpt = compute(eq, [], [])
        assert isinstance(rpt, PerformanceReport)

    def test_all_numeric_fields_present(self):
        eq  = _growing_equity()
        rpt = compute(eq, [], [])
        for attr in ("sharpe", "sortino", "calmar", "max_drawdown",
                     "max_drawdown_duration", "win_rate", "avg_win", "avg_loss",
                     "win_loss_ratio", "profit_factor", "annualized_return",
                     "annualized_vol", "total_return", "n_trades", "n_bars"):
            assert hasattr(rpt, attr), f"Missing field: {attr}"
            assert rpt.__dict__[attr] is not None, f"Field is None: {attr}"

    def test_n_bars_matches_equity_length(self):
        eq  = _growing_equity(200)
        rpt = compute(eq, [], [])
        assert rpt.n_bars == 200

    def test_n_trades_matches_trades_list(self):
        eq     = _growing_equity()
        trades = _make_trades(50, 10.0)
        rpt    = compute(eq, trades, [])
        assert rpt.n_trades == 50


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:

    def test_sharpe_zero_for_flat_equity(self):
        eq  = _flat_equity()
        rpt = compute(eq, [], [])
        assert rpt.sharpe == pytest.approx(0.0)

    def test_sharpe_positive_for_growing_equity(self):
        eq  = _growing_equity(252, daily_ret=0.002)
        rpt = compute(eq, [], [])
        assert rpt.sharpe > 0.0

    def test_sharpe_manual_verification(self):
        """
        Synthetic daily returns all = 0.002 (20 bp).
        RF daily ≈ (1.045)^(1/252) − 1 ≈ 1.741e-4
        Excess daily = 0.002 − 1.741e-4 ≈ 1.826e-3
        With zero variance the helper returns 0; add tiny noise to validate formula.
        """
        rng      = np.random.default_rng(0)
        rets     = 0.002 + rng.normal(0, 0.001, 252)
        nav      = 100_000 * np.cumprod(1 + rets)
        eq       = pd.Series(nav, index=pd.bdate_range("2022-01-03", periods=252))
        rpt      = compute(eq, [], [])
        # Manual Sharpe
        rf_d     = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1
        excess   = rets - rf_d
        expected = float(np.mean(excess) / np.std(excess, ddof=1) * math.sqrt(TRADING_DAYS))
        assert rpt.sharpe == pytest.approx(expected, rel=0.02)

    def test_sharpe_negative_for_declining_equity(self):
        rng  = np.random.default_rng(7)
        rets = -0.001 + rng.normal(0, 0.002, 252)
        nav  = 100_000 * np.cumprod(1 + rets)
        eq   = pd.Series(nav, index=pd.bdate_range("2022-01-03", periods=252))
        rpt  = compute(eq, [], [])
        assert rpt.sharpe < 0.0


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:

    def test_max_drawdown_zero_for_monotone_increasing(self):
        nav  = np.cumsum(np.ones(100)) * 1_000
        eq   = pd.Series(nav, index=pd.bdate_range("2022-01-03", periods=100))
        rpt  = compute(eq, [], [])
        assert rpt.max_drawdown == pytest.approx(0.0)

    def test_max_drawdown_matches_known_value(self):
        """Peak = 120k, trough = 90k → drawdown = −25 %."""
        eq   = _drawdown_equity()
        rpt  = compute(eq, [], [])
        assert rpt.max_drawdown == pytest.approx(-0.25, abs=0.005)

    def test_max_drawdown_negative_or_zero(self):
        eq  = _growing_equity()
        rpt = compute(eq, [], [])
        assert rpt.max_drawdown <= 0.0

    def test_max_drawdown_duration_non_negative(self):
        eq  = _drawdown_equity()
        rpt = compute(eq, [], [])
        assert rpt.max_drawdown_duration >= 0

    def test_max_drawdown_direct_helper(self):
        nav = np.array([100, 90, 80, 85, 90, 100], dtype=float)
        dd, dur = _max_drawdown(nav)
        assert dd  == pytest.approx(-0.20, abs=0.005)
        assert dur >= 2


# ---------------------------------------------------------------------------
# Win rate
# ---------------------------------------------------------------------------

class TestWinRate:

    def test_win_rate_one_when_all_winning(self):
        eq     = _growing_equity()
        trades = _make_trades(50, pnl_per_trade=100.0)
        rpt    = compute(eq, trades, [])
        assert rpt.win_rate == pytest.approx(1.0)

    def test_win_rate_zero_when_all_losing(self):
        eq     = _growing_equity()
        trades = _make_trades(50, pnl_per_trade=-100.0)
        rpt    = compute(eq, trades, [])
        assert rpt.win_rate == pytest.approx(0.0)

    def test_win_rate_half_when_mixed(self):
        dates  = pd.bdate_range("2022-01-03", periods=4)
        trades = [
            Trade(d, 3, 0.8, pnl=100.0, return_pct=0.001) for d in dates[:2]
        ] + [
            Trade(d, 3, 0.8, pnl=-100.0, return_pct=-0.001) for d in dates[2:]
        ]
        rpt = compute(_growing_equity(), trades, [])
        assert rpt.win_rate == pytest.approx(0.5)

    def test_win_rate_zero_for_no_trades(self):
        rpt = compute(_growing_equity(), [], [])
        assert rpt.win_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Regime breakdown
# ---------------------------------------------------------------------------

class TestRegimeBreakdown:

    def test_regime_breakdown_included_in_report(self):
        eq  = _growing_equity(100)
        log = _make_regime_log(eq, regime=3)
        rpt = compute(eq, [], log)
        assert isinstance(rpt.regime_breakdown, dict)

    def test_regime_breakdown_sums_to_total_bars(self):
        eq  = _growing_equity(100)
        # Split between two regimes
        log = (
            _make_regime_log(eq.iloc[:60], regime=3) +
            _make_regime_log(eq.iloc[60:], regime=2)
        )
        rpt = compute(eq, [], log)
        total = sum(s.n_bars for s in rpt.regime_breakdown.values())
        assert total == len(log)

    def test_regime_pct_times_sum_to_one(self):
        eq  = _growing_equity(100)
        log = (
            _make_regime_log(eq.iloc[:50], regime=3) +
            _make_regime_log(eq.iloc[50:], regime=2)
        )
        rpt = compute(eq, [], log)
        total_pct = sum(s.pct_time for s in rpt.regime_breakdown.values())
        assert total_pct == pytest.approx(1.0, abs=0.01)

    def test_regime_stats_has_required_fields(self):
        eq  = _growing_equity(50)
        log = _make_regime_log(eq)
        rpt = compute(eq, [], log)
        for s in rpt.regime_breakdown.values():
            assert isinstance(s, RegimeStats)
            assert s.n_bars > 0
            assert isinstance(s.regime_name, str)


# ---------------------------------------------------------------------------
# Confidence bucket analysis
# ---------------------------------------------------------------------------

class TestConfidenceBuckets:

    def test_always_three_buckets(self):
        eq  = _growing_equity()
        rpt = compute(eq, [], [])
        assert len(rpt.confidence_buckets) == 3

    def test_bucket_labels_are_high_medium_low(self):
        eq  = _growing_equity()
        rpt = compute(eq, [], [])
        assert set(rpt.confidence_buckets.keys()) == {"high", "medium", "low"}

    def test_bucket_stats_type(self):
        eq  = _growing_equity()
        rpt = compute(eq, [], [])
        for b in rpt.confidence_buckets.values():
            assert isinstance(b, BucketStats)

    def test_high_confidence_bucket_contains_high_conf_trades(self):
        eq     = _growing_equity()
        trades = [Trade(eq.index[i], 3, 0.9, 100.0, 0.001) for i in range(10)]
        rpt    = compute(eq, trades, [])
        assert rpt.confidence_buckets["high"].n_trades == 10
        assert rpt.confidence_buckets["low"].n_trades  == 0

    def test_low_confidence_bucket_contains_low_conf_trades(self):
        eq     = _growing_equity()
        trades = [Trade(eq.index[i], 2, 0.2, -50.0, -0.0005) for i in range(5)]
        rpt    = compute(eq, trades, [])
        assert rpt.confidence_buckets["low"].n_trades    == 5
        assert rpt.confidence_buckets["high"].n_trades   == 0
        assert rpt.confidence_buckets["medium"].n_trades == 0


# ---------------------------------------------------------------------------
# compare_benchmarks
# ---------------------------------------------------------------------------

class TestCompareBenchmarks:

    def _make_report(self, _name: str, sharpe: float) -> PerformanceReport:
        eq  = _growing_equity(100)
        rpt = compute(eq, [], [])
        # Override sharpe for testing
        rpt.sharpe = sharpe
        return rpt

    def test_returns_comparison_table(self):
        s_rpt = self._make_report("strategy", 1.0)
        bmark = {"buy_and_hold": self._make_report("bh", 0.5)}
        tbl   = compare_benchmarks(s_rpt, bmark)
        assert isinstance(tbl, ComparisonTable)

    def test_one_row_per_strategy(self):
        s_rpt = self._make_report("strategy", 1.0)
        bmarks = {
            "buy_and_hold": self._make_report("bh",   0.5),
            "sma200":       self._make_report("sma",  0.6),
            "random":       self._make_report("rand", 0.3),
        }
        tbl = compare_benchmarks(s_rpt, bmarks)
        assert len(tbl.rows) == 4   # strategy + 3 benchmarks

    def test_strategy_is_first_row(self):
        s_rpt = self._make_report("strategy", 1.5)
        bmark = {"bh": self._make_report("bh", 0.4)}
        tbl   = compare_benchmarks(s_rpt, bmark)
        assert tbl.rows[0].name == "strategy"

    def test_row_has_all_required_fields(self):
        s_rpt = self._make_report("strategy", 1.0)
        tbl   = compare_benchmarks(s_rpt, {})
        row   = tbl.rows[0]
        assert hasattr(row, "sharpe")
        assert hasattr(row, "total_return")
        assert hasattr(row, "max_drawdown")
        assert hasattr(row, "annualized_return")
        assert hasattr(row, "annualized_vol")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

class TestCSVExport:

    def test_csv_produces_valid_file(self, tmp_path: Path):
        s_rpt = compute(_growing_equity(), [], [])
        bmark = {"buy_and_hold": compute(_growing_equity(150), [], [])}
        tbl   = compare_benchmarks(s_rpt, bmark)
        out   = tmp_path / "comparison.csv"
        tbl.to_csv(out)
        assert out.exists()

    def test_csv_has_expected_columns(self, tmp_path: Path):
        s_rpt = compute(_growing_equity(), [], [])
        tbl   = compare_benchmarks(s_rpt, {})
        out   = tmp_path / "comparison.csv"
        tbl.to_csv(out)
        with open(out, newline="") as fh:
            reader = csv.DictReader(fh)
            cols   = set(reader.fieldnames or [])
        expected = {"name", "sharpe", "total_return", "max_drawdown",
                    "annualized_return", "annualized_vol"}
        assert expected.issubset(cols)

    def test_csv_row_count_matches_table(self, tmp_path: Path):
        s_rpt = compute(_growing_equity(), [], [])
        bmarks = {
            "bh":  compute(_growing_equity(120), [], []),
            "sma": compute(_growing_equity(130), [], []),
        }
        tbl = compare_benchmarks(s_rpt, bmarks)
        out = tmp_path / "cmp.csv"
        tbl.to_csv(out)
        with open(out, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == len(tbl.rows)

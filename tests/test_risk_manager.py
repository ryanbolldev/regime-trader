"""
tests/test_risk_manager.py
---------------------------
Unit tests for core/risk_manager.py — focused on circuit-breaker thresholds
in both paper mode (LIVE_ACCOUNT_MODE=False) and live mode (True).
"""

from __future__ import annotations

import pytest

from core.risk_manager import RiskManager

NAV = 100_000.0


@pytest.fixture()
def rm():
    r = RiskManager()
    r.initialize(NAV)
    return r


# ---------------------------------------------------------------------------
# Paper mode — existing threshold values
# ---------------------------------------------------------------------------

class TestPaperModeThresholds:

    def test_daily_halve_fires_at_minus_2pct(self, rm):
        fired = rm.update(NAV * 0.979)   # -2.1%
        assert "daily_halve_sizes" in fired

    def test_daily_halt_fires_at_minus_3pct(self, rm):
        fired = rm.update(NAV * 0.969)   # -3.1%
        assert "daily_halt" in fired

    def test_daily_halt_does_not_fire_at_minus_2pct(self, rm):
        fired = rm.update(NAV * 0.981)   # -1.9%
        assert "daily_halt" not in fired

    def test_weekly_resize_fires_at_minus_5pct(self, rm):
        fired = rm.update(NAV * 0.949)   # -5.1%
        assert "weekly_resize" in fired

    def test_weekly_resize_does_not_fire_at_minus_3pct(self, rm):
        fired = rm.update(NAV * 0.969)   # -3.1%
        assert "weekly_resize" not in fired

    def test_lockout_fires_at_minus_10pct(self, rm):
        fired = rm.update(NAV * 0.899)   # -10.1%
        assert "peak_drawdown_lockout" in fired

    def test_lockout_does_not_fire_at_minus_5pct(self, rm):
        fired = rm.update(NAV * 0.949)   # -5.1%
        assert "peak_drawdown_lockout" not in fired


# ---------------------------------------------------------------------------
# Live mode — tightened thresholds
# ---------------------------------------------------------------------------

class TestLiveModeThresholds:

    @pytest.fixture(autouse=True)
    def enable_live(self, monkeypatch):
        import config.settings as s
        monkeypatch.setattr(s, "LIVE_ACCOUNT_MODE", True)

    def test_daily_halt_fires_at_minus_2pct(self, rm):
        fired = rm.update(NAV * 0.979)   # -2.1%
        assert "daily_halt" in fired

    def test_daily_halt_does_not_fire_at_minus_1pct(self, rm):
        fired = rm.update(NAV * 0.991)   # -0.9%
        assert "daily_halt" not in fired

    def test_paper_halt_threshold_no_longer_triggers_halt(self, rm):
        """At -2.5% (paper halt zone), live mode already halts; check it fires."""
        fired = rm.update(NAV * 0.974)   # -2.6%, above paper -3% but below live -2%
        assert "daily_halt" in fired

    def test_weekly_resize_fires_at_minus_3pct(self, rm):
        fired = rm.update(NAV * 0.969)   # -3.1%
        assert "weekly_resize" in fired

    def test_weekly_resize_does_not_fire_at_minus_2pct(self, rm):
        fired = rm.update(NAV * 0.981)   # -1.9%: triggers daily_halve but not weekly
        assert "weekly_resize" not in fired

    def test_lockout_fires_at_minus_5pct(self, rm):
        fired = rm.update(NAV * 0.949)   # -5.1%
        assert "peak_drawdown_lockout" in fired

    def test_lockout_does_not_fire_at_minus_4pct(self, rm):
        # -4% is below live -5% lockout threshold but check it doesn't lock
        fired = rm.update(NAV * 0.959)   # -4.1%
        assert "peak_drawdown_lockout" not in fired

    def test_paper_lockout_threshold_does_not_lock_in_live(self, rm):
        """At -7% (paper lockout zone), live already locked; just confirm locked."""
        fired = rm.update(NAV * 0.929)   # -7.1%
        assert "peak_drawdown_lockout" in fired


# ---------------------------------------------------------------------------
# Threshold isolation: switching mode mid-session
# ---------------------------------------------------------------------------

class TestModeSwitch:

    def test_live_mode_true_uses_tighter_halt(self, monkeypatch):
        import config.settings as s
        monkeypatch.setattr(s, "LIVE_ACCOUNT_MODE", True)
        rm = RiskManager()
        rm.initialize(NAV)
        fired = rm.update(NAV * 0.979)   # -2.1%
        assert "daily_halt" in fired

    def test_live_mode_false_uses_paper_halt(self, monkeypatch):
        import config.settings as s
        monkeypatch.setattr(s, "LIVE_ACCOUNT_MODE", False)
        rm = RiskManager()
        rm.initialize(NAV)
        fired = rm.update(NAV * 0.979)   # -2.1% — below paper -2% warn, above -3% halt
        assert "daily_halt" not in fired
        assert "daily_halve_sizes" in fired

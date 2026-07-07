"""
tests/test_wheel_main.py
-------------------------
Unit tests for wheel_main.py — the wheel-only orchestrator (Phase 1, scan-only).

All external I/O is mocked. Emphasis on two guarantees:
  1. The scan runs with the LIVE HMM regime (not the scanner's neutral default).
  2. No non-wheel trading paths are wired in (negative guarantee).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from wheel_main import WheelTrader


def _mock_account(status: str = "ACTIVE", nav: float = 100_000.0) -> MagicMock:
    a = MagicMock()
    a.account_id      = "acc-test"
    a.status          = status
    a.portfolio_value = nav
    return a


def _synthetic_ohlcv(n: int = 60) -> pd.DataFrame:
    rng   = np.random.default_rng(0)
    close = 400.0 + np.cumsum(rng.normal(0, 1, n))
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(
        {"open": close * 0.999, "high": close * 1.005,
         "low": close * 0.995, "close": close, "volume": np.full(n, 1e7)},
        index=dates,
    )


def _mock_candidate(ticker: str = "MSTR", score: float = 72.0) -> MagicMock:
    c = MagicMock()
    c.ticker               = ticker
    c.composite_score      = score
    c.ivr                  = 55.0
    c.target_put_strike    = 93.0
    c.target_expiry        = "2026-08-07"
    c.dte                  = 33
    c.annualized_yield_pct = 41.0
    return c


@pytest.fixture()
def mock_client() -> MagicMock:
    c = MagicMock()
    c.get_account.return_value    = _mock_account()
    c.is_market_open.return_value = True
    c.get_positions.return_value  = []
    return c


@pytest.fixture()
def mock_hmm() -> MagicMock:
    h = MagicMock()
    h.predict_current.return_value = 3   # bull
    h.regime_name.return_value     = "bull"
    return h


@pytest.fixture(autouse=True)
def patched(monkeypatch, mock_hmm) -> dict:
    ohlcv = _synthetic_ohlcv()

    md = MagicMock()
    md.get_historical_bars.return_value = ohlcv

    # compute() must return a real DataFrame — the regime warm-up iterates its
    # tail rows through predict_current.
    feat_df = pd.DataFrame(
        np.random.default_rng(1).normal(size=(30, 3)),
        columns=["f0", "f1", "f2"],
    )
    fe = MagicMock()
    fe.compute.return_value = feat_df

    al = MagicMock()

    scanner_cls = MagicMock()
    scanner_cls.return_value.run.return_value = [_mock_candidate()]

    sched_cls = MagicMock()

    monkeypatch.setattr("wheel_main.HMMEngine", MagicMock(return_value=mock_hmm))
    monkeypatch.setattr("wheel_main.market_data", md)
    monkeypatch.setattr("wheel_main.feature_engineering", fe)
    monkeypatch.setattr("wheel_main.alerts", al)
    monkeypatch.setattr("wheel_main.WheelScanner", scanner_cls)
    monkeypatch.setattr("wheel_main.ScannerScheduler", sched_cls)

    return {"md": md, "fe": fe, "al": al, "hmm": mock_hmm,
            "scanner_cls": scanner_cls, "sched_cls": sched_cls}


@pytest.fixture()
def trader(mock_client, tmp_path) -> WheelTrader:
    return WheelTrader(
        client          = mock_client,
        risk_manager    = MagicMock(),
        lockfile        = tmp_path / "wheel_trading.lock",
        scan_on_startup = False,
    )


class TestLockfile:

    def test_startup_raises_when_lockfile_present(self, trader, tmp_path):
        (tmp_path / "wheel_trading.lock").write_text("stale")
        with pytest.raises(SystemExit):
            trader.startup()

    def test_lockfile_written_and_removed(self, trader, tmp_path):
        trader.startup()
        assert (tmp_path / "wheel_trading.lock").exists()
        trader.shutdown("test")
        assert not (tmp_path / "wheel_trading.lock").exists()


class TestStartup:

    def test_starts_scheduler_with_in_process_callback(self, trader, patched):
        trader.startup()
        sched_cls = patched["sched_cls"]
        sched_cls.assert_called_once()
        assert sched_cls.call_args.kwargs.get("on_fire") == trader._run_scan
        sched_cls.return_value.start.assert_called_once()

    def test_rejects_untradeable_account(self, trader, mock_client):
        mock_client.get_account.return_value = _mock_account(status="INACTIVE")
        with pytest.raises(RuntimeError):
            trader.startup()

    def test_initialises_risk_nav(self, trader):
        trader.startup()
        trader._risk.initialize.assert_called_once_with(100_000.0)


class TestScanRegimeWiring:

    def test_scan_uses_live_regime(self, trader, patched):
        patched["hmm"].predict_current.return_value = 3
        trader._run_scan()
        assert patched["scanner_cls"].call_args.kwargs["regime_label"] == 3

    def test_unconfirmed_regime_passes_none(self, trader, patched):
        patched["hmm"].predict_current.return_value = -1
        trader._run_scan()
        assert patched["scanner_cls"].call_args.kwargs["regime_label"] is None

    def test_scan_retrains_each_cycle(self, trader, patched):
        trader._run_scan()
        trader._run_scan()
        # HMM retrained (fit called) on each scan → non-stale model
        assert patched["hmm"].fit.call_count == 2


class TestScanEffects:

    def test_scan_alerts_top_candidates(self, trader, patched):
        trader._run_scan()
        events = [c.args[0] for c in patched["al"].send.call_args_list]
        assert "wheel_scan" in events

    def test_no_candidates_alert(self, trader, patched):
        patched["scanner_cls"].return_value.run.return_value = []
        trader._run_scan()
        msgs = [c.args[1] for c in patched["al"].send.call_args_list]
        assert any("no candidates" in m for m in msgs)

    def test_scan_writes_wheel_state(self, trader, patched, tmp_path, monkeypatch):
        monkeypatch.setattr("wheel_main.LOG_DIR", tmp_path)
        trader._run_scan()
        state_file = tmp_path / "wheel_state.json"
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["mode"] == "wheel_only"
        assert data["regime"] == "bull"
        assert data["candidate_count"] == 1
        assert data["candidates"][0]["ticker"] == "MSTR"

    def test_scan_failure_is_contained(self, trader, patched):
        patched["scanner_cls"].return_value.run.side_effect = RuntimeError("boom")
        trader._run_scan()  # must not raise
        events = [c.args[0] for c in patched["al"].send.call_args_list]
        assert "wheel_scan_error" in events

    def test_overlapping_scan_is_skipped(self, trader, patched):
        trader._scan_lock.acquire()
        try:
            trader._run_scan()
            patched["scanner_cls"].assert_not_called()
        finally:
            trader._scan_lock.release()


class TestNoTradingPaths:
    """Negative guarantee: wheel_main wires in no non-wheel trading paths."""

    @pytest.mark.parametrize(
        "name",
        ["order_executor", "btc_strategy", "regime_strategies",
         "BTCStrategy", "CrashStrategy", "position_tracker"],
    )
    def test_module_does_not_import_trading_paths(self, name):
        import wheel_main
        assert not hasattr(wheel_main, name), (
            f"wheel_main must not import {name} in the scan-only phase"
        )

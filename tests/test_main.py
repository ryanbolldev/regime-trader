"""
tests/test_main.py
-------------------
Unit tests for main.py — the orchestrator.

All external I/O is mocked: broker client, market data, alerts,
order executor, position tracker, and time.sleep.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from broker.alpaca_client import BrokerUnavailableError
from main import (
    API_RETRY_WAIT_SECS,
    DATA_MAX_RETRIES,
    DATA_RETRY_WAIT_SECS,
    RegimeTrader,
)


# ---------------------------------------------------------------------------
# Shared factories
# ---------------------------------------------------------------------------

def _mock_account(status: str = "ACTIVE", nav: float = 100_000.0) -> MagicMock:
    a = MagicMock()
    a.account_id     = "acc-test"
    a.status         = status
    a.portfolio_value = nav
    return a


def _synthetic_ohlcv(n: int = 60) -> pd.DataFrame:
    rng   = np.random.default_rng(0)
    close = 400.0 + np.cumsum(rng.normal(0, 1, n))
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(
        {
            "open":   close * 0.999,
            "high":   close * 1.005,
            "low":    close * 0.995,
            "close":  close,
            "volume": np.full(n, 1e7),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_client() -> MagicMock:
    c = MagicMock()
    c.get_account.return_value   = _mock_account()
    c.is_market_open.return_value = True
    c.get_positions.return_value  = []
    return c


@pytest.fixture()
def mock_hmm() -> MagicMock:
    h = MagicMock()
    h.predict_current.return_value = 3      # bull
    h.is_confirmed.return_value    = True
    h.is_uncertain.return_value    = False
    h.regime_name.return_value     = "bull"
    return h


@pytest.fixture()
def mock_risk() -> MagicMock:
    r = MagicMock()
    r.update.return_value  = []
    r.approve.return_value = MagicMock(
        approved=True, size_multiplier=1.0, reason="approved"
    )
    return r


@pytest.fixture()
def tmp_lockfile(tmp_path: Path) -> Path:
    return tmp_path / "trading.lock"


@pytest.fixture(autouse=True)
def patch_modules(monkeypatch, mock_hmm) -> dict:
    """Replace every stub module with a MagicMock for the duration of each test."""
    ohlcv  = _synthetic_ohlcv()
    series = MagicMock(spec=pd.Series)

    md = MagicMock()
    md.get_historical_bars.return_value = ohlcv
    md.get_latest_bars.return_value     = ohlcv

    fe = MagicMock()
    fe.compute.return_value        = MagicMock()
    fe.compute_latest.return_value = series

    al = MagicMock()

    oe = MagicMock()
    oe.cancel_all.return_value = [True, True]

    pt = MagicMock()
    pt.get_nav.return_value         = 100_000.0
    pt.get_daily_pnl.return_value   = 500.0
    pt.get_open_positions.return_value = []

    # Patch HMMEngine constructor so startup() returns the shared mock instead
    # of creating real engines that would fail fitting on MagicMock feature data.
    hmm_cls = MagicMock(return_value=mock_hmm)
    monkeypatch.setattr("main.HMMEngine", hmm_cls)

    monkeypatch.setattr("main.market_data",        md)
    monkeypatch.setattr("main.feature_engineering", fe)
    monkeypatch.setattr("main.alerts",             al)
    monkeypatch.setattr("main.order_executor",     oe)
    monkeypatch.setattr("main.position_tracker",   pt)

    return {"md": md, "fe": fe, "al": al, "oe": oe, "pt": pt, "hmm_cls": hmm_cls}


@pytest.fixture()
def trader(mock_client, mock_hmm, mock_risk, tmp_lockfile) -> RegimeTrader:
    return RegimeTrader(
        client       = mock_client,
        hmm          = mock_hmm,
        risk_manager = mock_risk,
        lockfile     = tmp_lockfile,
        bar_interval = 0,
    )


# ---------------------------------------------------------------------------
# TestLockfile
# ---------------------------------------------------------------------------

class TestLockfile:

    def test_startup_raises_system_exit_when_lockfile_present(
        self, trader, tmp_lockfile
    ):
        tmp_lockfile.write_text("stale lock")
        with pytest.raises(SystemExit):
            trader.startup()

    def test_lockfile_written_during_startup(self, trader, tmp_lockfile):
        trader.startup()
        assert tmp_lockfile.exists()

    def test_lockfile_contains_pid(self, trader, tmp_lockfile):
        import os
        trader.startup()
        content = tmp_lockfile.read_text()
        assert str(os.getpid()) in content

    def test_lockfile_removed_on_shutdown(self, trader, tmp_lockfile):
        trader.startup()
        assert tmp_lockfile.exists()
        trader.shutdown("test")
        assert not tmp_lockfile.exists()

    def test_lockfile_present_fires_alert(self, trader, tmp_lockfile, patch_modules):
        tmp_lockfile.write_text("stale lock")
        with pytest.raises(SystemExit):
            trader.startup()
        call_args = patch_modules["al"].send.call_args
        assert call_args.args[0] == "lockfile_present"
        assert call_args.args[2] == "critical"

    def test_lockfile_written_fires_alert(self, trader, patch_modules):
        trader.startup()
        calls = [c.args[0] for c in patch_modules["al"].send.call_args_list]
        assert "lockfile_written" in calls


# ---------------------------------------------------------------------------
# TestStartup
# ---------------------------------------------------------------------------

class TestStartup:

    def test_inactive_account_raises_runtime_error(self, trader, mock_client):
        mock_client.get_account.return_value = _mock_account(status="INACTIVE")
        with pytest.raises(RuntimeError, match="not tradeable"):
            trader.startup()

    def test_active_account_proceeds(self, trader):
        trader.startup()   # should not raise

    def test_approved_account_status_also_proceeds(self, trader, mock_client):
        mock_client.get_account.return_value = _mock_account(status="APPROVED")
        trader.startup()   # should not raise

    def test_hmm_training_fetches_historical_bars(self, trader, patch_modules):
        from config.settings import TICKERS
        trader.startup()
        md = patch_modules["md"]
        assert md.get_historical_bars.call_count == len(TICKERS)
        called_symbols = {c.args[0] for c in md.get_historical_bars.call_args_list}
        assert called_symbols == set(TICKERS)
        assert all(c.args[3] == "1Day" for c in md.get_historical_bars.call_args_list)

    def test_hmm_fit_called_with_computed_features(
        self, trader, mock_hmm, patch_modules
    ):
        from config.settings import TICKERS
        trader.startup()
        assert patch_modules["fe"].compute.call_count == len(TICKERS)
        assert mock_hmm.fit.call_count == len(TICKERS)

    def test_risk_manager_initialized_with_account_nav(
        self, trader, mock_risk, mock_client
    ):
        mock_client.get_account.return_value = _mock_account(nav=123_456.0)
        trader.startup()
        mock_risk.initialize.assert_called_once_with(pytest.approx(123_456.0))

    def test_startup_fires_startup_alert(self, trader, patch_modules):
        trader.startup()
        event_names = [c.args[0] for c in patch_modules["al"].send.call_args_list]
        assert "startup" in event_names

    def test_market_open_status_stored(self, trader, mock_client):
        mock_client.is_market_open.return_value = True
        trader.startup()
        assert trader._market_was_open is True

    def test_market_closed_status_stored(self, trader, mock_client):
        mock_client.is_market_open.return_value = False
        trader.startup()
        assert trader._market_was_open is False


# ---------------------------------------------------------------------------
# TestBarProcessing
# ---------------------------------------------------------------------------

class TestBarProcessing:

    def test_regime_change_fires_alert(
        self, trader, mock_hmm, patch_modules
    ):
        trader._current_regime = 2       # was neutral
        mock_hmm.predict_current.return_value = 3    # now bull
        mock_hmm.regime_name.return_value = "bull"
        trader._run_bar()
        event_names = [c.args[0] for c in patch_modules["al"].send.call_args_list]
        assert "regime_change" in event_names

    def test_no_alert_when_regime_unchanged(
        self, trader, mock_hmm, patch_modules
    ):
        trader._current_regime = 3
        mock_hmm.predict_current.return_value = 3
        trader._run_bar()
        event_names = [c.args[0] for c in patch_modules["al"].send.call_args_list]
        assert "regime_change" not in event_names

    def test_approved_signal_calls_order_executor(
        self, trader, mock_hmm, mock_risk, patch_modules
    ):
        trader._current_regime = 3
        mock_hmm.predict_current.return_value = 3
        mock_risk.approve.return_value = MagicMock(
            approved=True, size_multiplier=1.0, reason="approved"
        )
        trader._run_bar()
        patch_modules["oe"].submit.assert_called()

    def test_blocked_signal_skips_order_executor(
        self, trader, mock_hmm, mock_risk, patch_modules
    ):
        trader._current_regime = 3
        mock_hmm.predict_current.return_value = 3
        mock_risk.approve.return_value = MagicMock(
            approved=False, size_multiplier=0.0, reason="daily_halt_active"
        )
        trader._run_bar()
        patch_modules["oe"].submit.assert_not_called()

    def test_circuit_breaker_fires_alert(
        self, trader, mock_risk, patch_modules
    ):
        mock_risk.update.return_value = ["peak_drawdown_lockout"]
        trader._run_bar()
        event_names = [c.args[0] for c in patch_modules["al"].send.call_args_list]
        assert "circuit_breaker" in event_names

    def test_circuit_breaker_alert_contains_breaker_name(
        self, trader, mock_risk, patch_modules
    ):
        mock_risk.update.return_value = ["daily_halt"]
        trader._run_bar()
        alert_msgs = [c.args[1] for c in patch_modules["al"].send.call_args_list
                      if c.args[0] == "circuit_breaker"]
        assert any("daily_halt" in msg for msg in alert_msgs)

    def test_hmm_prediction_failure_skips_ticker(
        self, trader, mock_hmm, patch_modules
    ):
        mock_hmm.predict_current.side_effect = RuntimeError("model not fitted")
        trader._run_bar()
        patch_modules["oe"].submit.assert_not_called()

    def test_unconfirmed_regime_minus1_skips_signal(
        self, trader, mock_hmm, patch_modules
    ):
        mock_hmm.predict_current.return_value = -1
        trader._run_bar()
        patch_modules["oe"].submit.assert_not_called()

    def test_trade_placed_alert_includes_ticker_and_regime(
        self, trader, mock_hmm, patch_modules
    ):
        from config.settings import TICKERS, REFERENCE_TICKERS
        tradeable = next(t for t in TICKERS if t not in REFERENCE_TICKERS)
        trader._current_regime = 2
        mock_hmm.predict_current.return_value = 3
        mock_hmm.regime_name.return_value = "bull"
        trader._run_bar()
        alert_msgs = [c.args[1] for c in patch_modules["al"].send.call_args_list
                      if c.args[0] == "trade_placed"]
        assert any(tradeable in msg for msg in alert_msgs)

    def test_position_tracker_on_fill_called_after_order(
        self, trader, mock_hmm, mock_risk, patch_modules
    ):
        trader._current_regime = 3
        mock_hmm.predict_current.return_value = 3
        mock_risk.approve.return_value = MagicMock(
            approved=True, size_multiplier=1.0, reason="approved"
        )
        trader._run_bar()
        patch_modules["pt"].on_fill.assert_called()


# ---------------------------------------------------------------------------
# TestDataFeedRetry
# ---------------------------------------------------------------------------

class TestDataFeedRetry:

    def test_retries_data_fetch_three_times(
        self, trader, patch_modules
    ):
        patch_modules["md"].get_latest_bars.side_effect = OSError("timeout")
        with patch("main.time.sleep"):
            trader._run_bar()
        assert patch_modules["md"].get_latest_bars.call_count >= DATA_MAX_RETRIES

    def test_data_feed_drop_fires_alert(
        self, trader, patch_modules
    ):
        patch_modules["md"].get_latest_bars.side_effect = OSError("timeout")
        with patch("main.time.sleep"):
            trader._run_bar()
        event_names = [c.args[0] for c in patch_modules["al"].send.call_args_list]
        assert "data_feed_drop" in event_names

    def test_retry_sleeps_between_attempts(
        self, trader, patch_modules
    ):
        patch_modules["md"].get_latest_bars.side_effect = OSError("timeout")
        with patch("main.time.sleep") as mock_sleep:
            trader._run_bar()
        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert DATA_RETRY_WAIT_SECS in sleep_calls

    def test_succeeds_on_second_attempt(
        self, trader, mock_hmm, patch_modules
    ):
        ohlcv = _synthetic_ohlcv()
        patch_modules["md"].get_latest_bars.side_effect = [
            OSError("timeout"), ohlcv
        ]
        with patch("main.time.sleep"):
            trader._run_bar()
        mock_hmm.predict_current.assert_called()


# ---------------------------------------------------------------------------
# TestDailyPnL
# ---------------------------------------------------------------------------

class TestDailyPnL:

    def test_pnl_alert_fires_on_market_close(
        self, trader, mock_client, patch_modules
    ):
        trader._market_was_open = True
        mock_client.is_market_open.return_value = False
        trader._run_bar()
        event_names = [c.args[0] for c in patch_modules["al"].send.call_args_list]
        assert "daily_pnl" in event_names

    def test_pnl_alert_not_fired_when_market_stays_open(
        self, trader, mock_client, patch_modules
    ):
        trader._market_was_open = True
        mock_client.is_market_open.return_value = True
        trader._run_bar()
        event_names = [c.args[0] for c in patch_modules["al"].send.call_args_list]
        assert "daily_pnl" not in event_names

    def test_pnl_alert_not_fired_when_market_stays_closed(
        self, trader, mock_client, patch_modules
    ):
        trader._market_was_open = False
        mock_client.is_market_open.return_value = False
        trader._run_bar()
        event_names = [c.args[0] for c in patch_modules["al"].send.call_args_list]
        assert "daily_pnl" not in event_names

    def test_pnl_alert_message_contains_dollar_amount(
        self, trader, mock_client, patch_modules
    ):
        trader._market_was_open = True
        mock_client.is_market_open.return_value = False
        patch_modules["pt"].get_daily_pnl.return_value = 1234.56
        trader._run_bar()
        alert_msgs = [c.args[1] for c in patch_modules["al"].send.call_args_list
                      if c.args[0] == "daily_pnl"]
        assert any("1,234.56" in msg for msg in alert_msgs)


# ---------------------------------------------------------------------------
# TestRunLoop
# ---------------------------------------------------------------------------

class TestRunLoop:

    def test_api_outage_sleeps_and_retries(self, trader):
        call_count = [0]

        def run_bar_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                raise BrokerUnavailableError("connection refused")
            trader._running = False

        trader._run_bar = run_bar_side_effect
        with patch("main.time.sleep") as mock_sleep:
            trader.run()
        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert API_RETRY_WAIT_SECS in sleep_calls

    def test_api_outage_fires_alert(self, trader, patch_modules):
        call_count = [0]

        def run_bar_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                raise BrokerUnavailableError("down")
            trader._running = False

        trader._run_bar = run_bar_side_effect
        with patch("main.time.sleep"):
            trader.run()
        event_names = [c.args[0] for c in patch_modules["al"].send.call_args_list]
        assert "api_outage" in event_names

    def test_unhandled_exception_fires_critical_alert(
        self, trader, patch_modules
    ):
        trader._run_bar = MagicMock(side_effect=ValueError("surprise"))
        with patch("main.time.sleep"):
            trader.run()
        event_names = [c.args[0] for c in patch_modules["al"].send.call_args_list]
        assert "critical_error" in event_names

    def test_unhandled_exception_triggers_shutdown(self, trader):
        trader._run_bar = MagicMock(side_effect=ValueError("surprise"))
        with patch("main.time.sleep"):
            trader.run()
        assert not trader._running

    def test_loop_sleeps_between_bars(self, trader):
        call_count = [0]

        def run_bar_side_effect():
            call_count[0] += 1
            trader._running = False

        trader._run_bar = run_bar_side_effect
        with patch("main.time.sleep") as mock_sleep:
            trader.run()
        mock_sleep.assert_called_with(trader._bar_interval)


# ---------------------------------------------------------------------------
# TestShutdown
# ---------------------------------------------------------------------------

class TestShutdown:

    def test_shutdown_stops_running_flag(self, trader):
        trader._running = True
        trader.shutdown("test")
        assert not trader._running

    def test_shutdown_stores_reason(self, trader):
        trader.shutdown("test_reason")
        assert trader._shutdown_reason == "test_reason"

    def test_shutdown_fires_alert(self, trader, patch_modules):
        trader.shutdown("graceful")
        event_names = [c.args[0] for c in patch_modules["al"].send.call_args_list]
        assert "shutdown" in event_names

    def test_shutdown_alert_contains_reason(self, trader, patch_modules):
        trader.shutdown("sigterm")
        alert_msgs = [c.args[1] for c in patch_modules["al"].send.call_args_list
                      if c.args[0] == "shutdown"]
        assert any("sigterm" in msg for msg in alert_msgs)

    def test_shutdown_removes_lockfile(self, trader, tmp_lockfile):
        tmp_lockfile.write_text("lock")
        trader.shutdown("test")
        assert not tmp_lockfile.exists()

    def test_shutdown_cancels_orders_when_configured(
        self, mock_client, mock_hmm, mock_risk, tmp_lockfile, patch_modules
    ):
        t = RegimeTrader(
            client                    = mock_client,
            hmm                       = mock_hmm,
            risk_manager              = mock_risk,
            lockfile                  = tmp_lockfile,
            close_positions_on_shutdown = True,
        )
        tmp_lockfile.write_text("lock")
        t.shutdown("test")
        patch_modules["oe"].cancel_all.assert_called_once()

    def test_shutdown_skips_cancel_by_default(
        self, trader, patch_modules
    ):
        trader.shutdown("test")
        patch_modules["oe"].cancel_all.assert_not_called()

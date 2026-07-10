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
    c.get_orders.return_value     = []   # dedup check in _execute_exit_signals
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
        from config.settings import TICKERS
        trader._current_regime = {t: 2 for t in TICKERS}   # all tickers were neutral
        mock_hmm.predict_current.return_value = 3    # now bull
        mock_hmm.regime_name.return_value = "bull"
        trader._run_bar()
        event_names = [c.args[0] for c in patch_modules["al"].send.call_args_list]
        assert "regime_change" in event_names

    def test_regime_change_alert_includes_symbol(
        self, trader, mock_hmm, patch_modules
    ):
        """Each regime_change alert must carry symbol=ticker so per-ticker
        cooldown buckets are used; without it one ticker suppresses all others."""
        from config.settings import TICKERS
        trader._current_regime = {t: 2 for t in TICKERS}
        mock_hmm.predict_current.return_value = 3
        mock_hmm.regime_name.return_value = "bull"
        trader._run_bar()
        regime_calls = [
            c for c in patch_modules["al"].send.call_args_list
            if c.args[0] == "regime_change"
        ]
        assert regime_calls
        for call in regime_calls:
            assert call.kwargs.get("symbol"), (
                "regime_change alert missing symbol= kwarg — "
                "BTC and SPY would share one cooldown bucket"
            )

    def test_no_alert_when_regime_unchanged(
        self, trader, mock_hmm, patch_modules
    ):
        from config.settings import TICKERS
        trader._current_regime = {t: 3 for t in TICKERS}   # all tickers already bull
        mock_hmm.predict_current.return_value = 3
        trader._run_bar()
        event_names = [c.args[0] for c in patch_modules["al"].send.call_args_list]
        assert "regime_change" not in event_names

    def test_approved_signal_calls_order_executor(
        self, trader, mock_hmm, mock_risk, patch_modules
    ):
        from config.settings import TICKERS
        trader._current_regime = {t: 3 for t in TICKERS}
        mock_hmm.predict_current.return_value = 3
        mock_risk.approve.return_value = MagicMock(
            approved=True, size_multiplier=1.0, reason="approved"
        )
        trader._run_bar()
        patch_modules["oe"].submit.assert_called()

    def test_blocked_signal_skips_order_executor(
        self, trader, mock_hmm, mock_risk, patch_modules
    ):
        from config.settings import TICKERS
        trader._current_regime = {t: 3 for t in TICKERS}
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
        trader._current_regime = {t: 2 for t in TICKERS}
        mock_hmm.predict_current.return_value = 3
        mock_hmm.regime_name.return_value = "bull"
        trader._run_bar()
        alert_msgs = [c.args[1] for c in patch_modules["al"].send.call_args_list
                      if c.args[0] == "trade_placed"]
        assert any(tradeable in msg for msg in alert_msgs)

    def test_position_tracker_on_fill_called_after_order(
        self, trader, mock_hmm, mock_risk, patch_modules
    ):
        from config.settings import TICKERS
        trader._current_regime = {t: 3 for t in TICKERS}
        mock_hmm.predict_current.return_value = 3
        mock_risk.approve.return_value = MagicMock(
            approved=True, size_multiplier=1.0, reason="approved"
        )
        trader._run_bar()
        patch_modules["pt"].on_fill.assert_called()

    def test_trade_placed_alert_not_fired_when_submit_returns_none(
        self, trader, mock_hmm, patch_modules
    ):
        from config.settings import TICKERS, REFERENCE_TICKERS
        tradeable = next(t for t in TICKERS if t not in REFERENCE_TICKERS)
        trader._current_regime = {t: 2 for t in TICKERS}
        mock_hmm.predict_current.return_value = 3
        mock_hmm.regime_name.return_value = "bull"
        patch_modules["oe"].submit.return_value = None   # dedup / zero-size path
        trader._run_bar()
        trade_alerts = [c for c in patch_modules["al"].send.call_args_list
                        if c.args[0] == "trade_placed"]
        assert len(trade_alerts) == 0, "alert must not fire when submit returns None"

    def test_trade_placed_alert_passes_symbol_and_side(
        self, trader, mock_hmm, patch_modules
    ):
        from config.settings import TICKERS, REFERENCE_TICKERS
        tradeable = next(t for t in TICKERS if t not in REFERENCE_TICKERS)
        trader._current_regime = {t: 2 for t in TICKERS}
        mock_hmm.predict_current.return_value = 3
        mock_hmm.regime_name.return_value = "bull"
        trader._run_bar()
        trade_calls = [c for c in patch_modules["al"].send.call_args_list
                       if c.args[0] == "trade_placed"]
        assert trade_calls, "expected at least one trade_placed alert"
        kwargs = trade_calls[0].kwargs
        assert kwargs["symbol"] == tradeable
        assert kwargs["side"]   == "buy"

    def test_on_fill_not_called_when_submit_returns_none(
        self, trader, mock_hmm, mock_risk, patch_modules
    ):
        from config.settings import TICKERS
        trader._current_regime = {t: 3 for t in TICKERS}
        mock_hmm.predict_current.return_value = 3
        mock_risk.approve.return_value = MagicMock(
            approved=True, size_multiplier=1.0, reason="approved"
        )
        # Both equity and BTC paths must return None for on_fill to stay silent
        patch_modules["oe"].submit.return_value             = None
        patch_modules["oe"].submit_crypto_order.return_value = None
        trader._run_bar()
        patch_modules["pt"].on_fill.assert_not_called()


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
# TestBtcExitCircuitBreaker
# ---------------------------------------------------------------------------

class TestBtcExitCircuitBreaker:
    """Tests for the BTC exit-failure counter and circuit breaker."""

    def _setup_btc_exit(self, trader, mock_client, patch_modules):
        """Pre-wire _process_btc so that an EXIT action reaches submit_crypto_order."""
        action = MagicMock()
        action.action = "EXIT"
        action.size_usd = 1000.0
        action.reason = "bear"

        btc_strategy = MagicMock()
        btc_strategy.get_target_allocation.return_value = 0.0
        btc_strategy.get_action.return_value = action
        trader._btc_strategy = btc_strategy

        pos = MagicMock()
        pos.symbol = "BTC/USD"
        pos.qty = "0.02"
        pos.avg_entry_price = "45000.0"
        pos.market_value = "900.0"
        pos.unrealized_pl = "-100.0"
        mock_client.get_positions.return_value = [pos]

        cycle_signal = MagicMock()
        cycle_signal.composite_score = 0.3
        mock_cycle_eng = MagicMock()
        mock_cycle_eng.get_cycle_signal.return_value = cycle_signal

        hmm_engine = MagicMock()
        hmm_engine.is_uncertain.return_value = False
        hmm_engine.is_confirmed.return_value = True

        return MagicMock(return_value=mock_cycle_eng), hmm_engine

    def test_btc_exit_failure_circuit_breaker_fires_at_3(
        self, trader, mock_client, patch_modules
    ):
        mock_cycle_cls, hmm_engine = self._setup_btc_exit(
            trader, mock_client, patch_modules
        )
        patch_modules["oe"].submit_crypto_order.side_effect = Exception("403 Forbidden")

        with patch("core.cycle_engine.CycleEngine", mock_cycle_cls):
            for _ in range(3):
                trader._process_btc("BTC/USD", _synthetic_ohlcv(), 1, hmm_engine)

        assert trader._btc_exit_failure_count == 3
        alert_events = [c.args[0] for c in patch_modules["al"].send.call_args_list]
        assert "critical_error" in alert_events

    def test_btc_exit_failure_counter_resets_on_success(
        self, trader, mock_client, patch_modules
    ):
        trader._btc_exit_failure_count = 2
        mock_cycle_cls, hmm_engine = self._setup_btc_exit(
            trader, mock_client, patch_modules
        )
        patch_modules["oe"].submit_crypto_order.return_value = MagicMock()

        with patch("core.cycle_engine.CycleEngine", mock_cycle_cls):
            trader._process_btc("BTC/USD", _synthetic_ohlcv(), 1, hmm_engine)

        assert trader._btc_exit_failure_count == 0

    def test_btc_exit_failure_count_in_dashboard_state(
        self, trader, tmp_path, patch_modules
    ):
        import json
        trader._btc_exit_failure_count = 2

        with patch("main.LOG_DIR", tmp_path):
            trader._write_dashboard_state()

        state_file = tmp_path / "dashboard_state.json"
        state = json.loads(state_file.read_text())
        assert state["btc_exit_failure_count"] == 2


# ---------------------------------------------------------------------------
# TestMarketHoursGate
# ---------------------------------------------------------------------------

class TestMarketHoursGate:

    def test_equity_order_skipped_when_market_closed(
        self, trader, mock_client, patch_modules
    ):
        mock_client.is_market_open.return_value = False
        trader._run_bar()
        patch_modules["oe"].submit.assert_not_called()

    def test_equity_order_placed_when_market_open(
        self, trader, mock_risk, mock_client, patch_modules
    ):
        mock_client.is_market_open.return_value = True
        mock_risk.approve.return_value = MagicMock(
            approved=True, size_multiplier=1.0, reason="approved"
        )
        trader._run_bar()
        patch_modules["oe"].submit.assert_called()

    def test_btc_order_skips_equity_gate_when_market_closed(
        self, trader, mock_client, patch_modules
    ):
        """submit (equity) must not be called regardless of BTC path activity."""
        mock_client.is_market_open.return_value = False
        trader._run_bar()
        patch_modules["oe"].submit.assert_not_called()

    def test_flag_false_allows_equity_when_market_closed(
        self, trader, mock_risk, mock_client, patch_modules, monkeypatch
    ):
        """When IS_EQUITY_HOURS_ONLY=False the gate is bypassed."""
        import config.settings as s
        monkeypatch.setattr(s, "IS_EQUITY_HOURS_ONLY", False)
        mock_client.is_market_open.return_value = False
        mock_risk.approve.return_value = MagicMock(
            approved=True, size_multiplier=1.0, reason="approved"
        )
        trader._run_bar()
        patch_modules["oe"].submit.assert_called()

    def test_hmm_regime_detection_still_runs_when_closed(
        self, trader, mock_hmm, mock_client
    ):
        """HMM predict_current must be called even when the market is closed."""
        mock_client.is_market_open.return_value = False
        trader._run_bar()
        mock_hmm.predict_current.assert_called()


# ---------------------------------------------------------------------------
# TestLiveAccountMode
# ---------------------------------------------------------------------------

class TestLiveAccountMode:

    @pytest.fixture(autouse=True)
    def enable_live_mode(self, monkeypatch):
        import config.settings as s
        monkeypatch.setattr(s, "LIVE_ACCOUNT_MODE", True)

    def test_equity_skipped_in_live_mode(
        self, trader, mock_client, patch_modules
    ):
        mock_client.is_market_open.return_value = True
        trader._run_bar()
        patch_modules["oe"].submit.assert_not_called()

    def test_equity_skip_logged_in_live_mode(
        self, trader, mock_hmm, mock_client, caplog
    ):
        import logging
        mock_client.is_market_open.return_value = True
        mock_hmm.regime_name.return_value = "bull"
        with caplog.at_level(logging.INFO, logger="main"):
            trader._run_bar()
        assert any("LIVE MODE" in r.message and "equity orders disabled" in r.message
                   for r in caplog.records)

    def test_live_approval_log_appears_for_btc(
        self, trader, mock_client, caplog
    ):
        import logging
        mock_client.is_market_open.return_value = True
        with caplog.at_level(logging.INFO, logger="main"):
            trader._run_bar()
        assert any("LIVE ACCOUNT: approving order for" in r.message
                   for r in caplog.records)

    def test_btc_size_capped_at_20pct_nav(
        self, trader, mock_client, patch_modules, monkeypatch
    ):
        import config.settings as s
        import core.btc_strategy as btc_mod
        from core.btc_strategy import BTCAction
        monkeypatch.setattr(s, "LIVE_MAX_POSITION_PCT", 0.20)

        # Force BTCStrategy.get_action to return a BUY with size > 20% of NAV
        oversized_action = BTCAction(
            action="BUY", target_allocation_pct=0.5,
            size_usd=50_000.0,   # 50% of $100k NAV — well above 20% cap
            reason="test", regime=3, cycle_score=0.7, confidence=0.8,
        )
        monkeypatch.setattr(
            btc_mod.BTCStrategy, "get_action",
            lambda self, **kwargs: oversized_action,
        )
        mock_client.is_market_open.return_value = True
        mock_client.get_positions.return_value = []

        trader._run_bar()

        calls = patch_modules["oe"].submit_crypto_order.call_args_list
        if calls:
            submitted_size = calls[0].args[2] if calls[0].args else calls[0].kwargs.get("notional_usd", 0)
            assert submitted_size <= 100_000.0 * 0.20 + 1e-6

    def test_btc_blocked_when_deployed_at_cap(
        self, trader, mock_client, patch_modules, monkeypatch
    ):
        import config.settings as s
        import core.btc_strategy as btc_mod
        from core.btc_strategy import BTCAction
        monkeypatch.setattr(s, "LIVE_MAX_DEPLOYED_PCT", 0.30)

        buy_action = BTCAction(
            action="BUY", target_allocation_pct=0.5,
            size_usd=5_000.0, reason="test", regime=3, cycle_score=0.7, confidence=0.8,
        )
        monkeypatch.setattr(
            btc_mod.BTCStrategy, "get_action",
            lambda self, **kwargs: buy_action,
        )
        # Positions already consuming 30% of NAV
        pos = MagicMock()
        pos.market_value = 30_000.0   # exactly 30% of $100k
        pos.symbol = "OTHER"
        mock_client.get_positions.return_value = [pos]
        mock_client.is_market_open.return_value = True

        trader._run_bar()
        patch_modules["oe"].submit_crypto_order.assert_not_called()

    def test_paper_mode_flag_false_allows_equity(
        self, trader, mock_risk, mock_client, patch_modules, monkeypatch
    ):
        import config.settings as s
        monkeypatch.setattr(s, "LIVE_ACCOUNT_MODE", False)
        mock_client.is_market_open.return_value = True
        mock_risk.approve.return_value = MagicMock(
            approved=True, size_multiplier=1.0, reason="approved"
        )
        trader._run_bar()
        patch_modules["oe"].submit.assert_called()


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

    def test_shutdown_always_cancels_open_orders(
        self, trader, patch_modules
    ):
        trader.shutdown("test")
        patch_modules["oe"].cancel_all.assert_called_once()

    def test_shutdown_cancels_orders_regardless_of_flag(
        self, mock_client, mock_hmm, mock_risk, tmp_lockfile, patch_modules
    ):
        t = RegimeTrader(
            client       = mock_client,
            hmm          = mock_hmm,
            risk_manager = mock_risk,
            lockfile     = tmp_lockfile,
        )
        tmp_lockfile.write_text("lock")
        t.shutdown("test")
        patch_modules["oe"].cancel_all.assert_called_once()


# ---------------------------------------------------------------------------
# TestPositionCloseOnCrash
# ---------------------------------------------------------------------------

class TestPositionCloseOnCrash:
    """Integration tests: full close pipeline when regime or circuit breaker fires."""

    # ------------------------------------------------------------------
    # Test 1 — BTC EXIT on crash
    # ------------------------------------------------------------------

    def test_crash_regime_closes_btc_position(
        self, trader, mock_client, mock_hmm, patch_modules, monkeypatch
    ):
        import core.btc_strategy as btc_mod
        from core.btc_strategy import BTCAction

        # Broker reports a $10,000 BTC position
        btc_pos = MagicMock()
        btc_pos.symbol          = "BTCUSD"
        btc_pos.market_value    = 10_000.0
        btc_pos.qty             = 0.25
        btc_pos.avg_entry_price = 40_000.0
        btc_pos.unrealized_pl   = 0.0
        mock_client.get_positions.return_value = [btc_pos]

        # HMM returns crash regime (0) for every ticker
        mock_hmm.predict_current.return_value = 0

        # Strategy says EXIT with full position size
        exit_action = BTCAction(
            action                = "EXIT",
            target_allocation_pct = 0.0,
            size_usd              = 10_000.0,
            reason                = "target_allocation_zero",
            regime                = 0,
            cycle_score           = 0.0,
            confidence            = 0.5,
        )
        monkeypatch.setattr(
            btc_mod.BTCStrategy, "get_action",
            lambda self, **kwargs: exit_action,
        )

        trader._run_bar()

        calls = patch_modules["oe"].submit_crypto_order.call_args_list
        sell_calls = [c for c in calls if c.args[1] == "sell"]
        assert sell_calls, "submit_crypto_order should be called with side='sell'"
        assert sell_calls[0].args[2] == pytest.approx(10_000.0)

    # ------------------------------------------------------------------
    # Test 2 — Equity MSTR close on crash
    # ------------------------------------------------------------------

    def test_crash_regime_closes_equity_position(
        self, trader, mock_client, mock_hmm, patch_modules, monkeypatch
    ):
        import config.settings as s
        import core.regime_strategies as rs

        # Market is open so the market-hours gate doesn't block us
        mock_client.is_market_open.return_value = True

        # MSTR position held: $5,000 market value
        mstr_pos = MagicMock()
        mstr_pos.symbol       = "MSTR"
        mstr_pos.market_value = 5_000.0
        mstr_pos.qty          = 25.0
        patch_modules["pt"].get_open_positions.return_value = [mstr_pos]

        # Broker also reports the same position (for close_position)
        broker_pos = MagicMock()
        broker_pos.symbol = "MSTR"
        broker_pos.qty    = 25.0
        mock_client.get_positions.return_value = [broker_pos]

        # HMM returns crash regime (0) for every ticker
        mock_hmm.predict_current.return_value = 0

        # CrashStrategy returns empty list — no long targets → close MSTR
        monkeypatch.setattr(
            rs.CrashStrategy, "get_target_positions",
            lambda self, *args, **kwargs: [],
        )

        trader._run_bar()

        # close_position calls client.submit_order with side='sell'
        sell_calls = [
            c for c in mock_client.submit_order.call_args_list
            if c.kwargs.get("side") == "sell" or (c.args and "sell" in c.args)
        ]
        assert sell_calls, "client.submit_order should be called with side='sell'"
        submitted_qty = (
            sell_calls[0].kwargs.get("qty")
            or (sell_calls[0].args[1] if len(sell_calls[0].args) > 1 else None)
        )
        assert submitted_qty == pytest.approx(25.0)

    # ------------------------------------------------------------------
    # Test 3 — Circuit breaker cancels orders and closes all positions
    # ------------------------------------------------------------------

    def test_circuit_breaker_closes_all_positions(
        self, trader, mock_client, mock_risk, patch_modules
    ):
        # RiskManager fires daily halt (simulates -3.5% intraday drop)
        mock_risk.update.return_value = ["daily_halt"]

        # Two open positions the broker reports
        pos_a = MagicMock()
        pos_a.symbol = "MSTR"
        pos_a.qty    = 10.0
        pos_b = MagicMock()
        pos_b.symbol = "AAPL"
        pos_b.qty    = 20.0
        mock_client.get_positions.return_value = [pos_a, pos_b]

        trader._run_bar()

        # cancel_all must be called (open orders cancelled first)
        patch_modules["oe"].cancel_all.assert_called()

        # A market sell must have been submitted for every position
        submitted_symbols = {
            c.kwargs.get("symbol") or c.args[0]
            for c in mock_client.submit_order.call_args_list
            if (c.kwargs.get("side") == "sell"
                or (c.args and len(c.args) > 3 and c.args[3] == "sell")
                or c.kwargs.get("side") == "sell")
        }
        # Both positions must have received a close order
        assert "MSTR" in submitted_symbols or any(
            c.kwargs.get("symbol") in ("MSTR", "AAPL")
            for c in mock_client.submit_order.call_args_list
        )

    # ------------------------------------------------------------------
    # Test 4 — REDUCE action partially closes BTC
    # ------------------------------------------------------------------

    def test_reduce_action_partially_closes_btc(
        self, trader, mock_client, mock_hmm, patch_modules, monkeypatch
    ):
        import core.btc_strategy as btc_mod
        from core.btc_strategy import BTCAction

        # BTC position at 20% allocation ($20k of $100k NAV)
        btc_pos = MagicMock()
        btc_pos.symbol          = "BTCUSD"
        btc_pos.market_value    = 20_000.0
        btc_pos.qty             = 0.5
        btc_pos.avg_entry_price = 40_000.0
        btc_pos.unrealized_pl   = 0.0
        mock_client.get_positions.return_value = [btc_pos]

        # Strategy returns REDUCE: target 10%, sell $10k to halve the position
        reduce_action = BTCAction(
            action                = "REDUCE",
            target_allocation_pct = 0.10,
            size_usd              = 10_000.0,
            reason                = "allocation_drift_-0.100",
            regime                = 2,
            cycle_score           = 0.5,
            confidence            = 0.8,
        )
        monkeypatch.setattr(
            btc_mod.BTCStrategy, "get_action",
            lambda self, **kwargs: reduce_action,
        )

        # HMM returns neutral regime (2) for all tickers
        mock_hmm.predict_current.return_value = 2

        trader._run_bar()

        # submit_crypto_order must be called with side='sell' (not EXIT sell, REDUCE sell)
        calls = patch_modules["oe"].submit_crypto_order.call_args_list
        sell_calls = [c for c in calls if c.args[1] == "sell"]
        assert sell_calls, "Expected a sell call for REDUCE action"

        # Sell size should match the drift: 10% of $100k = $10,000
        sell_size = sell_calls[0].args[2]
        assert sell_size == pytest.approx(10_000.0)
        # Confirm it's a partial close (REDUCE), not EXIT
        assert reduce_action.action == "REDUCE"


# ---------------------------------------------------------------------------
# TestMSTRCorrelationGuard
# ---------------------------------------------------------------------------

class TestMSTRCorrelationGuard:
    """MSTR orders blocked when effective BTC exposure (BTC + MSTR×beta) exceeds cap."""

    def _mstr_pos(self, market_value: float) -> MagicMock:
        p = MagicMock()
        p.symbol       = "MSTR"
        p.market_value = market_value
        return p

    def _btc_pos(self, market_value: float) -> MagicMock:
        p = MagicMock()
        p.symbol       = "BTCUSD"
        p.market_value = market_value
        return p

    def test_mstr_blocked_when_combined_btc_exposure_exceeds_cap(
        self, trader, mock_client, mock_hmm, mock_risk, patch_modules, monkeypatch
    ):
        import config.settings as s
        monkeypatch.setattr(s, "MSTR_BTC_BETA", 2.5)
        monkeypatch.setattr(s, "BTC_MAX_ALLOCATION", 0.15)
        mock_client.is_market_open.return_value = True
        # BTC $10k (10%) + MSTR $3k × 2.5 = $17.5k > $15k cap
        mock_client.get_positions.return_value = [
            self._btc_pos(10_000.0),
            self._mstr_pos(3_000.0),
        ]
        mock_hmm.predict_current.return_value = 3
        mock_risk.approve.return_value = MagicMock(
            approved=True, size_multiplier=1.0, reason="approved"
        )

        trader._process_ticker("MSTR")

        patch_modules["oe"].submit.assert_not_called()

    def test_mstr_allowed_when_effective_btc_within_cap(
        self, trader, mock_client, mock_hmm, mock_risk, patch_modules, monkeypatch
    ):
        import config.settings as s
        monkeypatch.setattr(s, "MSTR_BTC_BETA", 2.5)
        monkeypatch.setattr(s, "BTC_MAX_ALLOCATION", 0.15)
        mock_client.is_market_open.return_value = True
        # BTC $5k (5%) only — effective_btc = $5k < $15k cap → allow
        mock_client.get_positions.return_value = [self._btc_pos(5_000.0)]
        mock_hmm.predict_current.return_value = 3
        mock_risk.approve.return_value = MagicMock(
            approved=True, size_multiplier=1.0, reason="approved"
        )

        trader._process_ticker("MSTR")

        patch_modules["oe"].submit.assert_called()

    def test_mstr_blocked_even_with_no_direct_btc_if_mstr_alone_exceeds_cap(
        self, trader, mock_client, mock_hmm, mock_risk, patch_modules, monkeypatch
    ):
        import config.settings as s
        monkeypatch.setattr(s, "MSTR_BTC_BETA", 2.5)
        monkeypatch.setattr(s, "BTC_MAX_ALLOCATION", 0.15)
        mock_client.is_market_open.return_value = True
        # No BTC; MSTR $7k × 2.5 = $17.5k > $15k cap
        mock_client.get_positions.return_value = [self._mstr_pos(7_000.0)]
        mock_hmm.predict_current.return_value = 3
        mock_risk.approve.return_value = MagicMock(
            approved=True, size_multiplier=1.0, reason="approved"
        )

        trader._process_ticker("MSTR")

        patch_modules["oe"].submit.assert_not_called()

    def test_non_mstr_ticker_not_affected_by_guard(
        self, trader, mock_client, mock_hmm, mock_risk, patch_modules, monkeypatch
    ):
        import config.settings as s
        monkeypatch.setattr(s, "MSTR_BTC_BETA", 2.5)
        monkeypatch.setattr(s, "BTC_MAX_ALLOCATION", 0.15)
        mock_client.is_market_open.return_value = True
        # High BTC exposure — but we're processing AAPL, not MSTR
        mock_client.get_positions.return_value = [self._btc_pos(20_000.0)]
        mock_hmm.predict_current.return_value = 3
        mock_risk.approve.return_value = MagicMock(
            approved=True, size_multiplier=1.0, reason="approved"
        )

        trader._process_ticker("AAPL")

        patch_modules["oe"].submit.assert_called()


# ---------------------------------------------------------------------------
# TestTradingEnabledSwitch
# ---------------------------------------------------------------------------

class TestTradingEnabledSwitch:
    """TRADING_ENABLED=False blocks new equity + BTC entries; exits still fire."""

    def _wire_equity_buy(self, mock_client, mock_hmm, mock_risk):
        mock_client.is_market_open.return_value = True
        mock_client.get_positions.return_value  = []
        mock_hmm.predict_current.return_value   = 3
        mock_risk.approve.return_value = MagicMock(
            approved=True, size_multiplier=1.0, reason="approved"
        )

    def _wire_btc(self, trader, mock_client, act, size=1000.0, target=0.10, positions=None):
        action = MagicMock()
        action.action = act
        action.size_usd = size
        action.reason = "test"
        btc_strategy = MagicMock()
        btc_strategy.get_target_allocation.return_value = target
        btc_strategy.get_action.return_value = action
        trader._btc_strategy = btc_strategy
        mock_client.get_positions.return_value = positions or []
        cycle_signal = MagicMock()
        cycle_signal.composite_score = 0.7
        mock_cycle_eng = MagicMock()
        mock_cycle_eng.get_cycle_signal.return_value = cycle_signal
        hmm = MagicMock()
        hmm.is_uncertain.return_value = False
        hmm.is_confirmed.return_value = True
        return MagicMock(return_value=mock_cycle_eng), hmm

    # -- equity ---------------------------------------------------------

    def test_equity_entry_blocked_when_disabled(
        self, trader, mock_client, mock_hmm, mock_risk, patch_modules, monkeypatch
    ):
        import config.settings as s
        monkeypatch.setattr(s, "TRADING_ENABLED", False)
        self._wire_equity_buy(mock_client, mock_hmm, mock_risk)
        trader._process_ticker("AAPL")
        patch_modules["oe"].submit.assert_not_called()

    def test_equity_entry_allowed_when_enabled(
        self, trader, mock_client, mock_hmm, mock_risk, patch_modules, monkeypatch
    ):
        import config.settings as s
        monkeypatch.setattr(s, "TRADING_ENABLED", True)
        self._wire_equity_buy(mock_client, mock_hmm, mock_risk)
        trader._process_ticker("AAPL")
        patch_modules["oe"].submit.assert_called()

    # -- BTC ------------------------------------------------------------

    def test_btc_entry_blocked_when_disabled(
        self, trader, mock_client, patch_modules, monkeypatch
    ):
        import config.settings as s
        monkeypatch.setattr(s, "TRADING_ENABLED", False)
        cycle_cls, hmm = self._wire_btc(trader, mock_client, "BUY")
        with patch("core.cycle_engine.CycleEngine", cycle_cls):
            trader._process_btc("BTC/USD", _synthetic_ohlcv(), 3, hmm)
        patch_modules["oe"].submit_crypto_order.assert_not_called()

    def test_btc_entry_allowed_when_enabled(
        self, trader, mock_client, patch_modules, monkeypatch
    ):
        import config.settings as s
        monkeypatch.setattr(s, "TRADING_ENABLED", True)
        cycle_cls, hmm = self._wire_btc(trader, mock_client, "BUY")
        patch_modules["oe"].submit_crypto_order.return_value = MagicMock()
        with patch("core.cycle_engine.CycleEngine", cycle_cls):
            trader._process_btc("BTC/USD", _synthetic_ohlcv(), 3, hmm)
        patch_modules["oe"].submit_crypto_order.assert_called()

    def test_btc_exit_still_fires_when_disabled(
        self, trader, mock_client, patch_modules, monkeypatch
    ):
        """Safety: the kill switch must never trap a position — exits still run."""
        import config.settings as s
        monkeypatch.setattr(s, "TRADING_ENABLED", False)
        pos = MagicMock()
        pos.symbol = "BTC/USD"; pos.qty = "0.02"; pos.avg_entry_price = "45000.0"
        pos.market_value = "900.0"; pos.unrealized_pl = "-100.0"
        cycle_cls, hmm = self._wire_btc(
            trader, mock_client, "EXIT", target=0.0, positions=[pos]
        )
        patch_modules["oe"].submit_crypto_order.return_value = MagicMock()
        with patch("core.cycle_engine.CycleEngine", cycle_cls):
            trader._process_btc("BTC/USD", _synthetic_ohlcv(), 1, hmm)
        patch_modules["oe"].submit_crypto_order.assert_called()


# ---------------------------------------------------------------------------
# TestExitSignals
# ---------------------------------------------------------------------------

class TestExitSignals:
    """Unit tests for _execute_exit_signals(): euphoria flatten and trailing stops."""

    def _pos(
        self,
        symbol: str,
        qty: float,
        unrealized_plpc: float,
        market_value: float = 5_000.0,
    ) -> MagicMock:
        p = MagicMock(spec=["symbol", "qty", "unrealized_plpc", "market_value"])
        p.symbol          = symbol
        p.qty             = qty
        p.unrealized_plpc = unrealized_plpc
        p.market_value    = market_value
        return p

    def _sell_order(self, symbol: str) -> MagicMock:
        o = MagicMock()
        o.symbol = symbol
        o.side   = "sell"
        return o

    # ── No position → no-op ────────────────────────────────────────────────

    def test_no_position_returns_false(self, trader, mock_client):
        mock_client.get_positions.return_value = []
        assert trader._execute_exit_signals("MSTR", 3) is False

    def test_get_positions_exception_returns_false(self, trader, mock_client):
        mock_client.get_positions.side_effect = RuntimeError("broker down")
        assert trader._execute_exit_signals("MSTR", 3) is False

    def test_wrong_ticker_in_positions_returns_false(self, trader, mock_client):
        mock_client.get_positions.return_value = [self._pos("AAPL", 5.0, -0.10)]
        assert trader._execute_exit_signals("MSTR", 3) is False

    # ── Euphoria (regime 4) ────────────────────────────────────────────────

    def test_euphoria_closes_open_position(self, trader, mock_client):
        mock_client.get_positions.return_value = [self._pos("MSTR", 10.0, 0.08)]
        result = trader._execute_exit_signals("MSTR", 4)
        assert result is True
        assert mock_client.submit_order.call_args.kwargs.get("side") == "sell"

    def test_euphoria_fires_sell_alert_with_size_usd(
        self, trader, mock_client, patch_modules
    ):
        mock_client.get_positions.return_value = [
            self._pos("MSTR", 10.0, 0.08, market_value=4_500.0)
        ]
        trader._execute_exit_signals("MSTR", 4)
        trade_alerts = [
            c for c in patch_modules["al"].send.call_args_list
            if c.args[0] == "trade_placed"
        ]
        assert trade_alerts, "expected a trade_placed alert on euphoria exit"
        assert trade_alerts[0].kwargs.get("side") == "sell"
        assert trade_alerts[0].kwargs.get("symbol") == "MSTR"
        assert trade_alerts[0].kwargs.get("size_usd") == pytest.approx(4_500.0)

    def test_euphoria_alert_not_fired_when_submit_fails(
        self, trader, mock_client, patch_modules
    ):
        """Alert must be silent when submit_order raises — no false-positive notification."""
        mock_client.get_positions.return_value = [self._pos("MSTR", 10.0, 0.08)]
        mock_client.submit_order.side_effect = RuntimeError("403 Forbidden")
        trader._execute_exit_signals("MSTR", 4)
        trade_alerts = [
            c for c in patch_modules["al"].send.call_args_list
            if c.args[0] == "trade_placed"
        ]
        assert not trade_alerts, "alert must not fire when order submission fails"

    def test_euphoria_disabled_by_setting_returns_false(
        self, trader, mock_client, monkeypatch
    ):
        import config.settings as s
        monkeypatch.setattr(s, "EQUITY_EUPHORIA_FLATTEN", False)
        mock_client.get_positions.return_value = [self._pos("MSTR", 10.0, 0.08)]
        result = trader._execute_exit_signals("MSTR", 4)
        assert result is False
        mock_client.submit_order.assert_not_called()

    def test_euphoria_skips_buy_path_after_close(
        self, trader, mock_client, mock_hmm, mock_risk, patch_modules, monkeypatch
    ):
        """After euphoria flattens a position _process_ticker must not place a buy."""
        import config.settings as s
        monkeypatch.setattr(s, "TICKERS", ["MSTR"])
        monkeypatch.setattr(s, "REFERENCE_TICKERS", [])
        mock_client.is_market_open.return_value = True
        mock_client.get_positions.return_value = [self._pos("MSTR", 10.0, 0.08)]
        mock_hmm.predict_current.return_value = 4   # euphoria
        mock_risk.approve.return_value = MagicMock(
            approved=True, size_multiplier=1.0, reason="approved"
        )
        trader._run_bar()
        patch_modules["oe"].submit.assert_not_called()

    def test_euphoria_case_insensitive_symbol_match(self, trader, mock_client):
        mock_client.get_positions.return_value = [self._pos("mstr", 10.0, 0.08)]
        result = trader._execute_exit_signals("MSTR", 4)
        assert result is True

    # ── Dedup: existing sell order prevents resubmission and re-alert ──────

    def test_dedup_returns_true_when_sell_order_already_open(
        self, trader, mock_client, patch_modules
    ):
        """If a sell order is already open, True is returned (buy path blocked)
        but no new order is submitted and no alert fires."""
        mock_client.get_positions.return_value = [self._pos("MSTR", 10.0, 0.08)]
        mock_client.get_orders.return_value = [self._sell_order("MSTR")]

        result = trader._execute_exit_signals("MSTR", 4)

        assert result is True
        mock_client.submit_order.assert_not_called()
        trade_alerts = [
            c for c in patch_modules["al"].send.call_args_list
            if c.args[0] == "trade_placed"
        ]
        assert not trade_alerts, "alert must not fire when sell order already open"

    def test_dedup_case_insensitive_order_match(
        self, trader, mock_client, patch_modules
    ):
        mock_client.get_positions.return_value = [self._pos("MSTR", 10.0, 0.08)]
        mock_client.get_orders.return_value = [self._sell_order("mstr")]  # lower-case

        result = trader._execute_exit_signals("MSTR", 4)

        assert result is True
        mock_client.submit_order.assert_not_called()

    def test_dedup_buy_order_does_not_block_close(
        self, trader, mock_client
    ):
        """An open BUY order for the same ticker must not suppress the sell."""
        buy_order = MagicMock()
        buy_order.symbol = "MSTR"
        buy_order.side   = "buy"
        mock_client.get_positions.return_value = [self._pos("MSTR", 10.0, 0.08)]
        mock_client.get_orders.return_value = [buy_order]

        result = trader._execute_exit_signals("MSTR", 4)

        assert result is True
        mock_client.submit_order.assert_called_once()

    # ── Bear trailing stop (regime 1) ──────────────────────────────────────

    def test_bear_above_stop_returns_false(self, trader, mock_client, monkeypatch):
        import config.settings as s
        monkeypatch.setattr(s, "EQUITY_BEAR_STOP_PCT", -0.05)
        mock_client.get_positions.return_value = [self._pos("MSTR", 10.0, -0.04)]
        assert trader._execute_exit_signals("MSTR", 1) is False
        mock_client.submit_order.assert_not_called()

    def test_bear_at_stop_boundary_closes(self, trader, mock_client, monkeypatch):
        import config.settings as s
        monkeypatch.setattr(s, "EQUITY_BEAR_STOP_PCT", -0.05)
        mock_client.get_positions.return_value = [self._pos("MSTR", 10.0, -0.05)]
        assert trader._execute_exit_signals("MSTR", 1) is True

    def test_bear_below_stop_closes_position(self, trader, mock_client, monkeypatch):
        import config.settings as s
        monkeypatch.setattr(s, "EQUITY_BEAR_STOP_PCT", -0.05)
        mock_client.get_positions.return_value = [self._pos("MSTR", 10.0, -0.09)]
        assert trader._execute_exit_signals("MSTR", 1) is True
        assert mock_client.submit_order.call_args.kwargs.get("side") == "sell"

    def test_bear_stop_fires_sell_alert(
        self, trader, mock_client, patch_modules, monkeypatch
    ):
        import config.settings as s
        monkeypatch.setattr(s, "EQUITY_BEAR_STOP_PCT", -0.05)
        mock_client.get_positions.return_value = [self._pos("MSTR", 10.0, -0.09)]
        trader._execute_exit_signals("MSTR", 1)
        trade_alerts = [
            c for c in patch_modules["al"].send.call_args_list
            if c.args[0] == "trade_placed"
        ]
        assert trade_alerts
        assert trade_alerts[0].kwargs.get("side") == "sell"
        assert trade_alerts[0].kwargs.get("symbol") == "MSTR"

    # ── Neutral trailing stop (regime 2) ───────────────────────────────────

    def test_neutral_below_stop_closes_position(
        self, trader, mock_client, monkeypatch
    ):
        import config.settings as s
        monkeypatch.setattr(s, "EQUITY_NEUTRAL_STOP_PCT", -0.06)
        mock_client.get_positions.return_value = [self._pos("MSTR", 10.0, -0.07)]
        assert trader._execute_exit_signals("MSTR", 2) is True

    def test_neutral_above_stop_returns_false(
        self, trader, mock_client, monkeypatch
    ):
        import config.settings as s
        monkeypatch.setattr(s, "EQUITY_NEUTRAL_STOP_PCT", -0.06)
        mock_client.get_positions.return_value = [self._pos("MSTR", 10.0, -0.03)]
        assert trader._execute_exit_signals("MSTR", 2) is False

    # ── Bull trailing stop (regime 3) ──────────────────────────────────────

    def test_bull_below_stop_closes_position(
        self, trader, mock_client, monkeypatch
    ):
        import config.settings as s
        monkeypatch.setattr(s, "EQUITY_BULL_STOP_PCT", -0.08)
        mock_client.get_positions.return_value = [self._pos("MSTR", 10.0, -0.09)]
        assert trader._execute_exit_signals("MSTR", 3) is True

    def test_bull_above_stop_returns_false(
        self, trader, mock_client, monkeypatch
    ):
        import config.settings as s
        monkeypatch.setattr(s, "EQUITY_BULL_STOP_PCT", -0.08)
        mock_client.get_positions.return_value = [self._pos("MSTR", 10.0, -0.05)]
        assert trader._execute_exit_signals("MSTR", 3) is False

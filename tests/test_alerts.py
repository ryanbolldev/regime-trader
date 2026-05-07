"""
tests/test_alerts.py
---------------------
Unit tests for core/alerts.py.
All network I/O is mocked — no real HTTP or SMTP calls are made.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import requests

import core.alerts as alerts_mod
from config.credentials import ConfigurationError


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_state():
    """Clear module-level cooldown state before and after every test."""
    alerts_mod._last_sent.clear()
    alerts_mod._overrides.clear()
    yield
    alerts_mod._last_sent.clear()
    alerts_mod._overrides.clear()


def _smtp_mock(smtp_cls: MagicMock) -> MagicMock:
    """Return the instance that the SMTP context manager yields."""
    instance = MagicMock()
    smtp_cls.return_value.__enter__ = MagicMock(return_value=instance)
    smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
    return instance


def _smtp_env(monkeypatch) -> None:
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USER", "user@example.com")
    monkeypatch.setenv("SMTP_PASS", "secret")
    monkeypatch.setenv("EMAIL_TO",  "admin@example.com")


def _http_ok() -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# TestSendWebhook
# ---------------------------------------------------------------------------

class TestSendWebhook:
    def test_posts_to_configured_url(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send_webhook({"event": "TEST"})
        mock_post.assert_called_once()
        assert mock_post.call_args.args[0] == "https://hooks.example.com/abc"

    def test_correct_payload_fields(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        payload = {"event": "STARTUP", "message": "hello", "regime": None}
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send_webhook(payload)
        sent = mock_post.call_args.kwargs["json"]
        assert sent["event"] == "STARTUP"
        assert sent["message"] == "hello"

    def test_missing_url_raises_configuration_error(self, monkeypatch):
        monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
        with pytest.raises(ConfigurationError):
            alerts_mod.send_webhook({"event": "TEST"})

    def test_empty_url_raises_configuration_error(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "")
        with pytest.raises(ConfigurationError):
            alerts_mod.send_webhook({"event": "TEST"})

    def test_network_error_is_silent(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        with patch("requests.post", side_effect=requests.ConnectionError("down")):
            alerts_mod.send_webhook({"event": "TEST"})  # must not raise

    def test_timeout_is_silent(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        with patch("requests.post", side_effect=requests.Timeout("timed out")):
            alerts_mod.send_webhook({"event": "TEST"})  # must not raise


# ---------------------------------------------------------------------------
# TestSendEmail
# ---------------------------------------------------------------------------

class TestSendEmail:
    def test_sends_with_correct_subject_and_body(self, monkeypatch):
        _smtp_env(monkeypatch)
        with patch("smtplib.SMTP") as smtp_cls:
            inst = _smtp_mock(smtp_cls)
            alerts_mod.send_email("Test Subject", "Test body")
        inst.starttls.assert_called_once()
        inst.login.assert_called_once_with("user@example.com", "secret")
        inst.sendmail.assert_called_once()
        from_addr, recipients, raw = inst.sendmail.call_args.args
        assert from_addr == "user@example.com"
        assert "admin@example.com" in recipients
        assert "Test Subject" in raw
        assert "Test body" in raw

    def test_correct_recipients(self, monkeypatch):
        _smtp_env(monkeypatch)
        monkeypatch.setenv("EMAIL_TO", "a@example.com, b@example.com")
        with patch("smtplib.SMTP") as smtp_cls:
            inst = _smtp_mock(smtp_cls)
            alerts_mod.send_email("subj", "body")
        recipients = inst.sendmail.call_args.args[1]
        assert "a@example.com" in recipients
        assert "b@example.com" in recipients

    def test_missing_host_raises_configuration_error(self, monkeypatch):
        _smtp_env(monkeypatch)
        monkeypatch.setenv("SMTP_HOST", "")
        with pytest.raises(ConfigurationError, match="SMTP_HOST"):
            alerts_mod.send_email("subj", "body")

    def test_missing_user_raises_configuration_error(self, monkeypatch):
        _smtp_env(monkeypatch)
        monkeypatch.setenv("SMTP_USER", "")
        with pytest.raises(ConfigurationError, match="SMTP_USER"):
            alerts_mod.send_email("subj", "body")

    def test_missing_to_raises_configuration_error(self, monkeypatch):
        _smtp_env(monkeypatch)
        monkeypatch.setenv("EMAIL_TO", "")
        with pytest.raises(ConfigurationError, match="EMAIL_TO"):
            alerts_mod.send_email("subj", "body")

    def test_uses_starttls(self, monkeypatch):
        _smtp_env(monkeypatch)
        with patch("smtplib.SMTP") as smtp_cls:
            inst = _smtp_mock(smtp_cls)
            alerts_mod.send_email("subj", "body")
        inst.starttls.assert_called_once()


# ---------------------------------------------------------------------------
# TestSend
# ---------------------------------------------------------------------------

class TestSend:
    def test_dispatches_webhook(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        monkeypatch.delenv("SMTP_HOST", raising=False)
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send("regime_change", "bull detected", "info")
        mock_post.assert_called_once()

    def test_dispatches_email_when_configured(self, monkeypatch):
        monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
        _smtp_env(monkeypatch)
        import config.settings as s
        monkeypatch.setattr(s, "ALERT_EMAIL_MIN_SEVERITY", "info")  # allow info for this test
        with patch("smtplib.SMTP") as smtp_cls:
            inst = _smtp_mock(smtp_cls)
            alerts_mod.send("startup", "system started", "info")
        inst.sendmail.assert_called_once()

    def test_no_raise_when_unconfigured(self, monkeypatch):
        monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
        monkeypatch.delenv("SMTP_HOST", raising=False)
        alerts_mod.send("shutdown", "stopping", "info")  # must not raise

    def test_cooldown_suppresses_duplicate(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send("regime_change", "msg1", "info")
            alerts_mod.send("regime_change", "msg2", "info")  # within cooldown
        assert mock_post.call_count == 1

    def test_different_events_not_suppressed(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send("regime_change", "msg1", "info")
            alerts_mod.send("trade_placed",  "msg2", "info")
        assert mock_post.call_count == 2

    def test_cooldown_resets_after_period(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        alerts_mod.set_cooldown("regime_change", 1)
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send("regime_change", "first", "info")
            time.sleep(1.1)
            alerts_mod.send("regime_change", "second", "info")
        assert mock_post.call_count == 2

    def test_canonical_event_name_in_payload(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send("regime_change", "bull", "info")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["event"] == "REGIME_CHANGE"

    def test_payload_has_required_fields(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send("trade_placed", "bought SPY", "info")
        payload = mock_post.call_args.kwargs["json"]
        for field in ("event", "message", "regime", "timestamp", "data"):
            assert field in payload, f"Missing field: {field}"

    def test_structured_trade_fields_in_data(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send(
                "trade_placed", "bought SPY", "info",
                symbol="SPY", side="buy", size_usd=5000.0, entry_price=452.10,
            )
        data = mock_post.call_args.kwargs["json"]["data"]
        assert data["symbol"]      == "SPY"
        assert data["side"]        == "buy"
        assert data["size_usd"]    == pytest.approx(5000.0)
        assert data["entry_price"] == pytest.approx(452.10)

    def test_missing_structured_fields_default_to_none(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send("regime_change", "bull", "info")
        data = mock_post.call_args.kwargs["json"]["data"]
        assert data["symbol"]      is None
        assert data["side"]        is None
        assert data["size_usd"]    is None
        assert data["entry_price"] is None

    def test_per_symbol_cooldown_for_trade_placed(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send("trade_placed", "SPY buy", "info", symbol="SPY")
            alerts_mod.send("trade_placed", "SPY buy", "info", symbol="SPY")   # suppressed
            alerts_mod.send("trade_placed", "AAPL buy", "info", symbol="AAPL") # different ticker
        assert mock_post.call_count == 2


class TestSendBtcTradeAlert:

    def _make_action(self, act="BUY", size=10_000.0, target=0.10, reason="test"):
        a = MagicMock()
        a.action                = act
        a.size_usd              = size
        a.target_allocation_pct = target
        a.reason                = reason
        return a

    def _make_order_result(self, filled_avg_price=None):
        r = MagicMock()
        r.filled_avg_price = filled_avg_price
        return r

    def test_fires_btc_trade_event(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send_btc_trade_alert(self._make_action())
        payload = mock_post.call_args.kwargs["json"]
        assert payload["event"] == "BTC_TRADE"

    def test_entry_price_populated_from_order_result(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        order_result = self._make_order_result(filled_avg_price=65_432.10)
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send_btc_trade_alert(
                self._make_action(), order_result=order_result
            )
        data = mock_post.call_args.kwargs["json"]["data"]
        assert data["entry_price"] == pytest.approx(65_432.10)

    def test_entry_price_none_when_no_order_result(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send_btc_trade_alert(self._make_action())
        data = mock_post.call_args.kwargs["json"]["data"]
        assert data["entry_price"] is None

    def test_entry_price_none_when_filled_avg_price_is_none(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        order_result = self._make_order_result(filled_avg_price=None)
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send_btc_trade_alert(
                self._make_action(), order_result=order_result
            )
        data = mock_post.call_args.kwargs["json"]["data"]
        assert data["entry_price"] is None

    def test_buy_action_sets_side_buy(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send_btc_trade_alert(self._make_action(act="BUY"))
        data = mock_post.call_args.kwargs["json"]["data"]
        assert data["side"] == "buy"

    def test_reduce_action_sets_side_sell(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send_btc_trade_alert(self._make_action(act="REDUCE"))
        data = mock_post.call_args.kwargs["json"]["data"]
        assert data["side"] == "sell"

    def test_size_usd_in_data(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send_btc_trade_alert(self._make_action(size=7_500.0))
        data = mock_post.call_args.kwargs["json"]["data"]
        assert data["size_usd"] == pytest.approx(7_500.0)


# ---------------------------------------------------------------------------
# TestSetCooldown
# ---------------------------------------------------------------------------

class TestSetCooldown:
    def test_zero_cooldown_allows_immediate_resend(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        alerts_mod.set_cooldown("startup", 0)
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send("startup", "first",  "info")
            alerts_mod.send("startup", "second", "info")
        assert mock_post.call_count == 2

    def test_per_event_independence(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        alerts_mod.set_cooldown("startup", 0)
        # "shutdown" still has default cooldown — second call suppressed
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send("startup",  "a", "info")
            alerts_mod.send("startup",  "b", "info")   # allowed (cooldown=0)
            alerts_mod.send("shutdown", "a", "info")
            alerts_mod.send("shutdown", "b", "info")   # suppressed (default cooldown)
        assert mock_post.call_count == 3


# ---------------------------------------------------------------------------
# TestCooldownOverrides (settings-level zero-cooldown for critical events)
# ---------------------------------------------------------------------------

class TestCooldownOverrides:
    def test_circuit_breaker_never_suppressed(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send("circuit_breaker", "first",  "critical")
            alerts_mod.send("circuit_breaker", "second", "critical")
        assert mock_post.call_count == 2

    def test_critical_error_never_suppressed(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send("critical_error", "first",  "critical")
            alerts_mod.send("critical_error", "second", "critical")
        assert mock_post.call_count == 2

    def test_lockfile_present_never_suppressed(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send("lockfile_present", "first",  "critical")
            alerts_mod.send("lockfile_present", "second", "critical")
        assert mock_post.call_count == 2

    def test_regular_event_still_suppressed_by_default_cooldown(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send("regime_change", "first",  "info")
            alerts_mod.send("regime_change", "second", "info")  # suppressed
        assert mock_post.call_count == 1

    def test_settings_override_independent_from_set_cooldown(self, monkeypatch):
        """settings override fires unconditionally; set_cooldown() acts on different namespace."""
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        alerts_mod.set_cooldown("circuit_breaker", 9999)  # runtime override: should win over settings
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send("circuit_breaker", "first",  "critical")
            alerts_mod.send("circuit_breaker", "second", "critical")  # suppressed by 9999s runtime override
        assert mock_post.call_count == 1


# ---------------------------------------------------------------------------
# TestEmailSeverityGate (ALERT_EMAIL_MIN_SEVERITY)
# ---------------------------------------------------------------------------

class TestEmailSeverityGate:

    def test_info_severity_skips_email(self, monkeypatch):
        import config.settings as s
        monkeypatch.setattr(s, "ALERT_EMAIL_MIN_SEVERITY", "warning")
        _smtp_env(monkeypatch)
        monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
        with patch("smtplib.SMTP") as smtp_cls:
            inst = _smtp_mock(smtp_cls)
            alerts_mod.send("regime_change", "info message", "info")
        inst.sendmail.assert_not_called()

    def test_warning_severity_triggers_email(self, monkeypatch):
        import config.settings as s
        monkeypatch.setattr(s, "ALERT_EMAIL_MIN_SEVERITY", "warning")
        _smtp_env(monkeypatch)
        monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
        with patch("smtplib.SMTP") as smtp_cls:
            inst = _smtp_mock(smtp_cls)
            alerts_mod.send("circuit_breaker", "warning message", "warning")
        inst.sendmail.assert_called_once()

    def test_critical_severity_triggers_email(self, monkeypatch):
        import config.settings as s
        monkeypatch.setattr(s, "ALERT_EMAIL_MIN_SEVERITY", "warning")
        _smtp_env(monkeypatch)
        monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
        with patch("smtplib.SMTP") as smtp_cls:
            inst = _smtp_mock(smtp_cls)
            alerts_mod.send("critical_error", "critical message", "critical")
        inst.sendmail.assert_called_once()

    def test_webhook_fires_regardless_of_severity(self, monkeypatch):
        import config.settings as s
        monkeypatch.setattr(s, "ALERT_EMAIL_MIN_SEVERITY", "warning")
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example.com/abc")
        with patch("requests.post", return_value=_http_ok()) as mock_post:
            alerts_mod.send("regime_change", "info event", "info")
        mock_post.assert_called_once()

    def test_min_severity_info_allows_all_emails(self, monkeypatch):
        import config.settings as s
        monkeypatch.setattr(s, "ALERT_EMAIL_MIN_SEVERITY", "info")
        _smtp_env(monkeypatch)
        monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
        with patch("smtplib.SMTP") as smtp_cls:
            inst = _smtp_mock(smtp_cls)
            alerts_mod.send("regime_change", "info message", "info")
        inst.sendmail.assert_called_once()

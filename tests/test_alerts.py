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

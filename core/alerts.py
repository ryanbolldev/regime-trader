"""
core/alerts.py
---------------
Notification dispatcher for critical system events.

Supported channels:
  - Email via SMTP (credentials loaded from environment)
  - Webhook (POST to ALERT_WEBHOOK_URL)

Public interface:
  send(event_type, message, severity) -> None
  send_email(subject, body) -> None
  send_webhook(payload) -> None
  set_cooldown(event_type, seconds) -> None
"""

from __future__ import annotations

import logging
import os
import smtplib
import time
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Any

import requests

from config.credentials import ConfigurationError

log = logging.getLogger(__name__)

# Load .env once at import time so individual functions see env vars without
# calling load_dotenv() again (which would defeat monkeypatch.delenv in tests).
try:
    from dotenv import load_dotenv as _ld
    _ld(override=False)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Module-level cooldown state (cleared in tests via .clear())
# ---------------------------------------------------------------------------
_last_sent: dict[str, float] = {}   # key absent or None == never sent
_overrides: dict[str, int]   = {}

# ---------------------------------------------------------------------------
# Canonical event names (raw_event → payload "event" field)
# ---------------------------------------------------------------------------
_CANONICAL: dict[str, str] = {
    "regime_change":    "REGIME_CHANGE",
    "trade_placed":     "TRADE_PLACED",
    "circuit_breaker":  "CIRCUIT_BREAKER",
    "daily_pnl":        "DAILY_PNL",
    "lockfile_written": "LOCKFILE",
    "lockfile_present": "LOCKFILE",
    "startup":          "STARTUP",
    "shutdown":         "SHUTDOWN",
    "api_outage":       "CIRCUIT_BREAKER",
    "data_feed_drop":   "CIRCUIT_BREAKER",
    "critical_error":   "CIRCUIT_BREAKER",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _webhook_url() -> str:
    url = os.getenv("ALERT_WEBHOOK_URL", "").strip()
    if not url:
        raise ConfigurationError("ALERT_WEBHOOK_URL is not configured")
    return url


def _cooldown_for(event_type: str) -> int:
    if event_type in _overrides:
        return _overrides[event_type]
    try:
        from config import settings
        return settings.ALERT_COOLDOWN_SECONDS
    except Exception:
        return 300


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def set_cooldown(event_type: str, seconds: int) -> None:
    """Override the default cooldown period for a specific event type."""
    _overrides[event_type] = seconds


def send_webhook(payload: dict[str, Any]) -> None:
    """POST payload as JSON to ALERT_WEBHOOK_URL.

    Raises ConfigurationError if URL is not set.
    Network errors are logged as warnings and swallowed.
    """
    url = _webhook_url()
    try:
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Webhook delivery failed (non-fatal): %s", exc)


def send_email(subject: str, body: str) -> None:
    """Send a plain-text email via SMTP with STARTTLS.

    Raises ConfigurationError if SMTP_HOST, SMTP_USER, or EMAIL_TO is absent.
    """
    host     = os.getenv("SMTP_HOST", "").strip()
    port_str = os.getenv("SMTP_PORT", "587").strip()
    user     = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASS", "")
    to_raw   = os.getenv("EMAIL_TO",  "").strip()

    if not host:
        raise ConfigurationError("SMTP_HOST is not configured")
    if not user:
        raise ConfigurationError("SMTP_USER is not configured")
    if not to_raw:
        raise ConfigurationError("EMAIL_TO is not configured")

    recipients = [r.strip() for r in to_raw.split(",") if r.strip()]

    msg            = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = user
    msg["To"]      = ", ".join(recipients)

    with smtplib.SMTP(host, int(port_str)) as smtp:
        smtp.starttls()
        if password:
            smtp.login(user, password)
        smtp.sendmail(user, recipients, msg.as_string())


def send(event_type: str, message: str, severity: str = "info") -> None:
    """Dispatch an alert to all configured channels, respecting cooldown.

    Silently suppresses if the same event_type was sent within the cooldown
    window.  Never raises — channel errors are logged as warnings.
    """
    cooldown = _cooldown_for(event_type)
    last     = _last_sent.get(event_type)  # None (key absent) == never sent

    if last is not None and (time.monotonic() - last) < cooldown:
        log.debug("Alert suppressed (cooldown active): event=%s", event_type)
        return

    _last_sent[event_type] = time.monotonic()

    canonical = _CANONICAL.get(event_type, event_type.upper())
    payload: dict[str, Any] = {
        "event":     canonical,
        "message":   message,
        "regime":    None,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "data":      {"severity": severity, "raw_event": event_type},
    }

    try:
        send_webhook(payload)
    except ConfigurationError:
        pass
    except Exception as exc:
        log.warning("Webhook error: %s", exc)

    subject = f"[Regime Trader] {canonical}"
    try:
        send_email(subject, message)
    except ConfigurationError:
        pass
    except Exception as exc:
        log.warning("Email error: %s", exc)

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
    "btc_trade":        "BTC_TRADE",
    "circuit_breaker":  "CIRCUIT_BREAKER",
    "daily_pnl":        "DAILY_PNL",
    "lockfile_written": "LOCKFILE",
    "lockfile_present": "LOCKFILE",
    "startup":          "STARTUP",
    "shutdown":         "SHUTDOWN",
    "api_outage":       "CIRCUIT_BREAKER",
    "data_feed_drop":   "CIRCUIT_BREAKER",
    "critical_error":   "CIRCUIT_BREAKER",
    "onchain_signal":   "ONCHAIN_SIGNAL",
    "cycle_signal":     "CYCLE_SIGNAL",
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


def send(
    event_type: str,
    message: str,
    severity: str = "info",
    *,
    symbol: str = "",
) -> None:
    """Dispatch an alert to all configured channels, respecting per-symbol cooldown.

    The cooldown key is ``event_type:symbol`` so each ticker fires independently.
    Callers that omit symbol use an empty-symbol key and share one cooldown bucket.
    Never raises — channel errors are logged as warnings.
    """
    cooldown_key = f"{event_type}:{symbol}"
    cooldown = _cooldown_for(event_type)
    last     = _last_sent.get(cooldown_key)  # None (key absent) == never sent

    if last is not None and (time.monotonic() - last) < cooldown:
        log.debug(
            "Alert suppressed (cooldown active): event=%s symbol=%s",
            event_type, symbol or "(none)",
        )
        return

    _last_sent[cooldown_key] = time.monotonic()

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


def send_btc_trade_alert(
    action: object,
    *,
    regime_name: str = "",
    cycle_score: float = 0.0,
) -> None:
    """Fire a BTC_TRADE alert for BUY / SELL / REDUCE / EXIT actions.

    ``action`` must be a BTCAction dataclass instance.  Using 'object' to
    avoid a circular import; duck-typing is safe here.
    """
    try:
        act         = str(getattr(action, "action", ""))
        size_usd    = float(getattr(action, "size_usd", 0.0))
        target_alloc = float(getattr(action, "target_allocation_pct", 0.0))
        reason      = str(getattr(action, "reason", ""))

        msg = (
            f"BTC TRADE\n"
            f"Action: {act}\n"
            f"Size: ${size_usd:,.2f}\n"
            f"Target Allocation: {target_alloc:.1%}\n"
            f"Regime: {regime_name}\n"
            f"Cycle Score: {cycle_score * 100:.1f}%\n"
            f"Reason: {reason}"
        )
        send("btc_trade", msg, "info", symbol="BTC")
    except Exception as exc:
        log.warning("BTC trade alert error: %s", exc)


def send_cycle_alert(signal: object, prev_score: float = 0.0) -> None:
    """Fire a CYCLE_SIGNAL alert on composite_score threshold crossing or failed cycle.

    signal must be a CycleSignal dataclass instance. Using 'object' to avoid
    a circular import; duck-typing is safe here.
    """
    try:
        from config.settings import CYCLE_COMPOSITE_THRESHOLD

        failed   = getattr(signal, "failed_cycle", False)
        score    = float(getattr(signal, "composite_score", 0.0))
        crossing = prev_score < CYCLE_COMPOSITE_THRESHOLD <= score

        if not (failed or crossing):
            return

        translation_map = {"right": "right (bullish)", "left": "left (bearish)"}
        translation_str = translation_map.get(
            getattr(signal, "translation", "unknown"), "unknown"
        )

        severity = "critical" if failed else "info"
        msg = (
            "🔄 CYCLE SIGNAL\n"
            f"60-Day Timing Probability: {getattr(signal, 'timing_probability', 0.0) * 100:.1f}%\n"
            f"Donchian Score: {getattr(signal, 'donchian_score', 0.0) * 100:.1f}%\n"
            f"Gaussian Score: {getattr(signal, 'gaussian_score', 0.0) * 100:.1f}%\n"
            f"Bollinger Score: {getattr(signal, 'bollinger_score', 0.0) * 100:.1f}%\n"
            f"Price Confirmation: {getattr(signal, 'price_confirmation', 0.0) * 100:.1f}%\n"
            f"Composite Score: {score * 100:.1f}%\n"
            f"Translation: {translation_str}\n"
            f"Days Since Low: {getattr(signal, 'days_since_last_low', 0)} "
            f"of ~{getattr(signal, 'adaptive_window_center', 60)}\n"
            f"Macro Phase: {getattr(signal, 'macro_phase', 'unknown')}\n"
            f"Bias: {getattr(signal, 'bias', 'neutral')}"
        )
        send("cycle_signal", msg, severity)
    except Exception as exc:
        log.warning("Cycle alert error: %s", exc)

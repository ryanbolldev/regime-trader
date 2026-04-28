"""
core/alerts.py
---------------
Notification dispatcher for critical system events.

Supported channels:
  - Email via SMTP (credentials loaded from .env through credentials.py)
  - Webhook (POST to a configurable URL, e.g., Slack incoming webhook)

Events that trigger alerts:
  - Regime change confirmed (with new regime label and confidence)
  - Circuit breaker activated: daily -2%, daily -3%, weekly -5%
  - Peak-drawdown lockfile written (requires human intervention)
  - Order submission failure after max retries
  - Stale / missing market data detected
  - Process crash or unhandled exception in main loop
  - Lockfile manually deleted and trading resumed

Alert deduplication:
  - Each alert type has a cooldown period (configurable in settings.py) to
    prevent notification storms during volatile periods.

Public interface (to be implemented):
  send(event_type, message, severity) -> None
  send_email(subject, body) -> None
  send_webhook(payload) -> None
  set_cooldown(event_type, seconds) -> None
"""

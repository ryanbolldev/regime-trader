"""
tests/test_alerts.py
---------------------
Unit tests for core/alerts.py.

Test cases to implement:
  - send() dispatches to both email and webhook channels when both are configured
  - send_email() calls the SMTP client with correct subject and body
  - send_webhook() sends a POST request to the configured URL with correct payload
  - Cooldown: second send() for the same event type within cooldown window is
    silently suppressed (no duplicate dispatch)
  - Cooldown resets after the cooldown period elapses
  - set_cooldown() overrides the default cooldown for the given event type
  - Missing SMTP config raises a clear ConfigurationError at send_email() time,
    not at import time
  - Missing webhook URL raises a clear ConfigurationError at send_webhook() time
  - All tests mock SMTP and HTTP; no real network calls
"""

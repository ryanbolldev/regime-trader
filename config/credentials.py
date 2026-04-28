"""
config/credentials.py
---------------------
Sole point of entry for all credentials and secrets.

Responsibilities:
  - Load the .env file at startup using python-dotenv
  - Expose named accessors (functions or properties) for each required secret:
      - Alpaca API key and secret
      - Email SMTP credentials (for alerts)
      - Webhook URL (for alerts)
  - Raise a clear ConfigurationError at import time if any required variable is
    absent, so the system fails fast before any market-facing code runs
  - Never log, print, or pass credential values to any external service
  - Never fall back to hardcoded defaults for secrets

This module must not be imported by tests that run in CI; use environment
variable mocking (monkeypatch / os.environ overrides) instead.
"""

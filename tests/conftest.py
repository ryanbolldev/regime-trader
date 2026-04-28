"""
tests/conftest.py
------------------
Shared pytest fixtures and configuration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv(request) -> pd.DataFrame:
    """Generate a realistic single-ticker OHLCV DataFrame with DatetimeIndex.

    Accepts an optional indirect parameter dict with keys:
        n_bars  (default 600)
        seed    (default 0)
    """
    params   = getattr(request, "param", {}) or {}
    n_bars   = params.get("n_bars", 600)
    seed     = params.get("seed", 0)

    rng = np.random.default_rng(seed)

    # Simulate a log-price random walk
    log_returns = rng.normal(0.0003, 0.012, n_bars)
    log_price   = np.cumsum(log_returns)
    close       = 100 * np.exp(log_price)

    # Build realistic OHLC from close
    noise = rng.uniform(0.001, 0.015, n_bars)
    high  = close * (1 + noise)
    low   = close * (1 - noise)
    open_ = close * (1 + rng.normal(0, 0.005, n_bars))
    open_ = np.clip(open_, low, high)

    volume = rng.lognormal(mean=14.0, sigma=0.6, size=n_bars).astype(int)

    dates = pd.bdate_range(end="2024-01-01", periods=n_bars, freq="B")

    return pd.DataFrame(
        {
            "open":   open_,
            "high":   high,
            "low":    low,
            "close":  close,
            "volume": volume,
        },
        index=dates,
    )


@pytest.fixture
def mock_broker_client():
    """MagicMock implementing a minimal BrokerClient interface."""
    from unittest.mock import MagicMock

    client = MagicMock()
    client.submit_order.return_value = {"id": "order-1", "status": "filled"}
    client.get_positions.return_value = []
    client.get_account.return_value   = {"equity": 100_000.0}
    return client


@pytest.fixture
def env_credentials(monkeypatch):
    """Inject fake credential env vars so imports of credentials.py don't fail."""
    monkeypatch.setenv("ALPACA_API_KEY",    "fake-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "fake-secret")
    monkeypatch.setenv("SMTP_HOST",         "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT",         "587")
    monkeypatch.setenv("SMTP_USER",         "test@example.com")
    monkeypatch.setenv("SMTP_PASSWORD",     "fake-pass")
    monkeypatch.setenv("WEBHOOK_URL",       "https://example.com/webhook")


@pytest.fixture
def tmp_lockfile_path(tmp_path):
    """Return a Path inside pytest's tmp_path for lockfile tests."""
    return tmp_path / "regime_trader.lock"

"""
tests/test_order_executor.py
-----------------------------
Unit tests for core/order_executor.py — focusing on the per-trade risk cap
that must prevent any equity order from exceeding NAV × PER_TRADE_RISK_CAP.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from config.settings import PER_TRADE_RISK_CAP
from core import order_executor
from core.regime_strategies import get_signal

NAV = 100_000.0
RISK_CAP = NAV * PER_TRADE_RISK_CAP   # $1,000 at 1%


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(position_size_usd: float, regime: int = 3):
    """Return a real Signal with the given position_size_usd overridden."""
    sig = get_signal(regime, 0.9, NAV, 0.0, False)
    # Use object.__setattr__ since Signal is a dataclass (not frozen).
    object.__setattr__(sig, "position_size_usd", position_size_usd)
    return sig


def _make_client() -> MagicMock:
    """Minimal mock AlpacaClient that returns no existing orders/positions."""
    client = MagicMock()
    client.get_orders.return_value = []
    client.get_positions.return_value = []
    client.submit_order.return_value = MagicMock(
        order_id="abc123",
        client_order_id="cid",
        symbol="MSTR",
        qty=5.0,
        side="buy",
        order_type="market",
        status="accepted",
        filled_qty=0.0,
        filled_avg_price=None,
        request_id="rid",
    )
    return client


# ---------------------------------------------------------------------------
# Cap enforcement: correct allocation produces ≤ RISK_CAP shares
# ---------------------------------------------------------------------------

class TestRiskCapEnforcement:

    def test_correctly_capped_signal_at_177_produces_5_shares(self):
        """$1,000 cap ÷ $177 = 5 shares (int truncation)."""
        signal = _make_signal(RISK_CAP)  # exactly at cap
        client = _make_client()

        with patch("core.position_tracker.get_nav", return_value=NAV), \
             patch("core.market_data.get_latest_bar", return_value={"close": 177.0}):
            order_executor._submit_equity_order(signal, client, "MSTR")

        call_kwargs = client.submit_order.call_args
        assert call_kwargs is not None, "submit_order was not called"
        qty = call_kwargs.kwargs.get("qty") or call_kwargs.args[1]
        assert qty == 5.0, f"Expected 5 shares, got {qty}"

    def test_correctly_capped_signal_at_190_produces_5_shares(self):
        """$1,000 cap ÷ $190 = 5 shares."""
        signal = _make_signal(RISK_CAP)
        client = _make_client()

        with patch("core.position_tracker.get_nav", return_value=NAV), \
             patch("core.market_data.get_latest_bar", return_value={"close": 190.0}):
            order_executor._submit_equity_order(signal, client, "CVX")

        call_kwargs = client.submit_order.call_args
        assert call_kwargs is not None
        qty = call_kwargs.kwargs.get("qty") or call_kwargs.args[1]
        assert qty == 5.0

    def test_inflated_signal_raises_assertion(self):
        """An inflated position_size_usd (e.g. $7,260) must raise AssertionError."""
        signal = _make_signal(7_260.0)   # old buggy value at $726k inflated NAV

        with patch("core.position_tracker.get_nav", return_value=NAV), \
             patch("core.market_data.get_latest_bar", return_value={"close": 177.0}):
            with pytest.raises(AssertionError, match="exceeds"):
                order_executor._submit_equity_order(signal, client=_make_client(), symbol="MSTR")

    def test_signal_just_above_cap_raises_assertion(self):
        """Any signal that slips past get_signal but exceeds the cap is caught here."""
        signal = _make_signal(RISK_CAP + 0.02)  # 2 cents over tolerance

        with patch("core.position_tracker.get_nav", return_value=NAV), \
             patch("core.market_data.get_latest_bar", return_value={"close": 100.0}):
            with pytest.raises(AssertionError):
                order_executor._submit_equity_order(signal, client=_make_client(), symbol="SPY")

    def test_signal_at_tolerance_boundary_passes(self):
        """$1,000.01 (within +0.01 tolerance) is accepted and clamped."""
        signal = _make_signal(RISK_CAP + 0.005)  # within tolerance
        client = _make_client()

        with patch("core.position_tracker.get_nav", return_value=NAV), \
             patch("core.market_data.get_latest_bar", return_value={"close": 200.0}):
            order_executor._submit_equity_order(signal, client, "QQQ")

        # Should have been clamped; 5 shares at $200
        call_kwargs = client.submit_order.call_args
        assert call_kwargs is not None
        qty = call_kwargs.kwargs.get("qty") or call_kwargs.args[1]
        assert qty == 5.0

    def test_clamped_size_used_for_share_calculation(self):
        """Even if signal.position_size_usd is at the cap, shares use the clamped value."""
        signal = _make_signal(RISK_CAP)
        client = _make_client()  # $1,000 / $333 = 3 shares

        with patch("core.position_tracker.get_nav", return_value=NAV), \
             patch("core.market_data.get_latest_bar", return_value={"close": 333.0}):
            order_executor._submit_equity_order(signal, client, "NVDA")

        call_kwargs = client.submit_order.call_args
        assert call_kwargs is not None
        qty = call_kwargs.kwargs.get("qty") or call_kwargs.args[1]
        assert qty == 3.0


# ---------------------------------------------------------------------------
# Cap scales with NAV
# ---------------------------------------------------------------------------

class TestCapScalesWithNav:

    def test_cap_at_50k_nav(self):
        """At $50k NAV, cap is $500; $500 / $100 = 5 shares."""
        nav = 50_000.0
        risk_cap = nav * PER_TRADE_RISK_CAP  # $500
        signal = _make_signal(risk_cap)
        client = _make_client()

        with patch("core.position_tracker.get_nav", return_value=nav), \
             patch("core.market_data.get_latest_bar", return_value={"close": 100.0}):
            order_executor._submit_equity_order(signal, client, "SPY")

        call_kwargs = client.submit_order.call_args
        assert call_kwargs is not None
        qty = call_kwargs.kwargs.get("qty") or call_kwargs.args[1]
        assert qty == 5.0

    def test_inflated_signal_raises_at_50k_nav(self):
        """$1,000 signal raises when NAV is only $50k (cap is $500)."""
        signal = _make_signal(1_000.0)

        with patch("core.position_tracker.get_nav", return_value=50_000.0), \
             patch("core.market_data.get_latest_bar", return_value={"close": 100.0}):
            with pytest.raises(AssertionError):
                order_executor._submit_equity_order(signal, client=_make_client(), symbol="SPY")


# ---------------------------------------------------------------------------
# submit_crypto_order
# ---------------------------------------------------------------------------

class TestSubmitCryptoOrder:

    def test_zero_notional_returns_none(self):
        client = MagicMock()
        result = order_executor.submit_crypto_order("BTC/USD", "buy", 0.0, client)
        assert result is None
        client.submit_order_notional.assert_not_called()

    def test_negative_notional_returns_none(self):
        client = MagicMock()
        result = order_executor.submit_crypto_order("BTC/USD", "buy", -100.0, client)
        assert result is None
        client.submit_order_notional.assert_not_called()

    def test_positive_notional_calls_broker(self):
        client = MagicMock()
        client.get_orders.return_value = []
        order_executor.submit_crypto_order("BTC/USD", "buy", 500.0, client)
        client.submit_order_notional.assert_called_once_with(
            symbol="BTC/USD", notional_usd=500.0, side="buy"
        )

    def test_dedup_skips_if_open_order_exists(self):
        existing = MagicMock()
        existing.symbol = "BTC/USD"
        client = MagicMock()
        client.get_orders.return_value = [existing]
        result = order_executor.submit_crypto_order("BTC/USD", "buy", 500.0, client)
        assert result is None
        client.submit_order_notional.assert_not_called()

    def test_dedup_is_case_insensitive(self):
        existing = MagicMock()
        existing.symbol = "btc/usd"
        client = MagicMock()
        client.get_orders.return_value = [existing]
        result = order_executor.submit_crypto_order("BTC/USD", "buy", 500.0, client)
        assert result is None

    def test_sell_not_blocked_by_position(self):
        """Crypto sell should not be blocked by an existing position — only open orders."""
        client = MagicMock()
        client.get_orders.return_value = []   # no open orders
        order_executor.submit_crypto_order("BTC/USD", "sell", 500.0, client)
        client.submit_order_notional.assert_called_once()

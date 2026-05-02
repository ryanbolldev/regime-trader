"""
core/order_executor.py
-----------------------
Translates approved trade signals into broker API calls.

Public interface:
  submit(signal, client=None, *, symbol="") -> OrderResult | None
  cancel(order_id, client=None) -> bool
  cancel_all(client=None) -> list[bool]
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from broker.alpaca_client import AlpacaClient, OrderResult
    from core.regime_strategies import Signal

from core.wheel_strategy import WheelActionType

log = logging.getLogger(__name__)

_client: Optional[AlpacaClient] = None


def _get_client() -> AlpacaClient:
    global _client
    if _client is None:
        from broker.alpaca_client import AlpacaClient
        _client = AlpacaClient()
    return _client


def submit(
    signal: Signal,
    client: Optional[AlpacaClient] = None,
    *,
    symbol: str = "",
) -> Optional[OrderResult]:
    """Submit an order derived from signal.

    Returns None when the signal calls for no action (WAIT / SIT_OUT wheel
    actions, zero position size, or equity signals without a symbol).
    """
    c = client if client is not None else _get_client()

    if signal.wheel_action is not None:
        return _submit_wheel_order(signal, c)

    return _submit_equity_order(signal, c, symbol)


def _submit_wheel_order(signal: Signal, client: AlpacaClient) -> Optional[OrderResult]:
    action = signal.wheel_action

    if action.action in (WheelActionType.WAIT, WheelActionType.SIT_OUT):
        log.debug(
            "No wheel order: action=%s reason=%s",
            action.action.value, action.reason,
        )
        return None

    contract = action.contract
    if contract is None:
        log.warning("Wheel action %s has no contract — skipping", action.action.value)
        return None

    if action.action in (WheelActionType.SELL_PUT, WheelActionType.SELL_CALL):
        side = "sell"
    elif action.action == WheelActionType.CLOSE:
        side = "buy"
    else:
        log.warning("Unhandled wheel action: %s", action.action)
        return None

    log.info(
        "Wheel order: symbol=%s side=%s qty=1 action=%s regime=%s",
        contract.symbol, side, action.action.value, signal.regime_name,
    )
    return client.submit_order(
        symbol=contract.symbol,
        qty=1.0,
        side=side,
        order_type="market",
    )


def _submit_equity_order(
    signal: Signal,
    client: AlpacaClient,
    symbol: str,
) -> Optional[OrderResult]:
    if not symbol:
        log.warning("Equity order has no symbol — skipping")
        return None

    if signal.position_size_usd <= 0:
        log.debug(
            "Position size $0 for %s regime=%s — skipping",
            symbol, signal.regime_name,
        )
        return None

    # Guard: regime must allow long equity positions.
    from core.regime_strategies import _PROFILES
    profile = _PROFILES.get(signal.regime)
    if profile is not None and not profile.allow_long:
        log.debug(
            "Regime %s (allow_long=False) — skipping equity buy for %s",
            signal.regime_name, symbol,
        )
        return None

    # Guard: reject orders that would exceed available buying power.
    try:
        from core import position_tracker
        nav = position_tracker.get_nav(client)
    except Exception as exc:
        log.warning("Could not fetch NAV for buying-power check: %s — skipping", exc)
        return None

    from config.settings import PER_TRADE_RISK_CAP
    risk_cap = nav * PER_TRADE_RISK_CAP
    assert signal.position_size_usd <= risk_cap + 0.01, (
        f"position_size_usd ${signal.position_size_usd:.2f} exceeds "
        f"NAV×{PER_TRADE_RISK_CAP:.0%} cap ${risk_cap:.2f} for {symbol} "
        f"regime={signal.regime_name}"
    )
    position_size_usd = min(signal.position_size_usd, risk_cap)

    # Guard: skip if an open order or position already exists for this symbol.
    if _has_existing_order_or_position(symbol, client):
        log.warning(
            "Skipping %s: existing open order or position already found", symbol
        )
        return None

    # Convert dollar size to whole-share quantity using the latest bar price.
    try:
        from core.market_data import get_latest_bar
        bar = get_latest_bar(symbol)
        current_price = float(bar["close"])
    except Exception as exc:
        log.warning("Could not fetch price for %s: %s — skipping", symbol, exc)
        return None

    shares = int(position_size_usd / current_price)
    if shares < 1:
        log.debug(
            "Size $%.2f < 1 share at $%.2f for %s — skipping",
            position_size_usd, current_price, symbol,
        )
        return None

    log.info(
        "Equity order: symbol=%s side=buy shares=%d ($%.2f @ $%.2f) regime=%s",
        symbol, shares, position_size_usd, current_price, signal.regime_name,
    )
    return client.submit_order(
        symbol=symbol,
        qty=float(shares),
        side="buy",
        order_type="market",
    )


def _has_existing_order_or_position(symbol: str, client: AlpacaClient) -> bool:
    """Return True if there is already an open order or live position for symbol."""
    try:
        orders = client.get_orders()
        if any(o.symbol.upper() == symbol.upper() for o in orders):
            return True
    except Exception as exc:
        log.warning("Could not check existing orders for %s: %s", symbol, exc)

    try:
        positions = client.get_positions()
        if any(p.symbol.upper() == symbol.upper() for p in positions):
            return True
    except Exception as exc:
        log.warning("Could not check existing positions for %s: %s", symbol, exc)

    return False


def submit_crypto_order(
    symbol:       str,
    side:         str,
    notional_usd: float,
    client: Optional[AlpacaClient] = None,
) -> Optional[OrderResult]:
    """Submit a notional crypto order.  Returns None if notional <= 0 or a
    duplicate open order already exists for the symbol.
    """
    if notional_usd <= 0:
        log.debug(
            "Crypto order skipped: notional $%.2f <= 0 for %s", notional_usd, symbol
        )
        return None

    c = client if client is not None else _get_client()

    # Dedup: skip if an open order already exists for this symbol.
    try:
        orders = c.get_orders()
        if any(o.symbol.upper() == symbol.upper() for o in orders):
            log.warning(
                "Skipping %s %s: existing open order found", symbol, side
            )
            return None
    except Exception as exc:
        log.warning("Could not check existing orders for %s: %s", symbol, exc)

    log.info(
        "Crypto order: symbol=%s side=%s notional=$%.2f", symbol, side, notional_usd
    )
    return c.submit_order_notional(
        symbol=symbol,
        notional_usd=notional_usd,
        side=side,
    )


def cancel(order_id: str, client: Optional[AlpacaClient] = None) -> bool:
    """Cancel a single order by ID. Returns True if cancelled, False if not found."""
    c = client if client is not None else _get_client()
    return c.cancel_order(order_id)


def cancel_all(client: Optional[AlpacaClient] = None) -> list[bool]:
    """Cancel all open orders. Returns a per-order list of cancel results."""
    c = client if client is not None else _get_client()
    orders = c.get_orders()
    return [c.cancel_order(o.order_id) for o in orders]

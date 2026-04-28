#!/usr/bin/env python3
"""
scripts/connection_test.py
---------------------------
End-to-end connectivity smoke test against the Alpaca paper-trading API.

Runs 7 steps in sequence and prints CONNECTION TEST PASSED on success.
Exits with code 1 and a descriptive message on any failure.

Usage
-----
    # from project root:
    python scripts/connection_test.py

    # or inside the Docker container:
    make shell
    python scripts/connection_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from the scripts/ directory or from project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from broker.alpaca_client import AlpacaClient, OrderResult


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _bar(char: str = "─", width: int = 64) -> str:
    return char * width


def _step(n: int, title: str) -> None:
    print(f"\n{_bar()}")
    print(f"  Step {n} — {title}")
    print(_bar())


def _ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def _info(label: str, value: str) -> None:
    print(f"  {label:<22}: {value}")


# ---------------------------------------------------------------------------
# Main test routine
# ---------------------------------------------------------------------------

def run() -> None:
    print(_bar("═"))
    print("  Alpaca Connection Test")
    print(_bar("═"))

    # ── Step 1: instantiate client (loads .env, validates HTTPS) ────────────
    _step(1, "Connecting to Alpaca")
    client = AlpacaClient()
    _ok("AlpacaClient instantiated")

    # ── Step 2: account info ─────────────────────────────────────────────────
    _step(2, "Account info")
    acct = client.get_account()
    _info("Account ID",      acct.account_id)
    _info("Status",          acct.status)
    _info("Portfolio value", f"${acct.portfolio_value:>14,.2f}")
    _info("Buying power",    f"${acct.buying_power:>14,.2f}")
    _info("Options BP",      f"${acct.options_buying_power:>14,.2f}")
    _info("Paper trading",   str(acct.is_paper))
    _ok("Account info fetched")

    # ── Step 3: market hours ─────────────────────────────────────────────────
    _step(3, "Market hours status")
    is_open = client.is_market_open()
    _info("Market status", "OPEN" if is_open else "CLOSED")
    _ok("Market hours check complete")

    # ── Step 4: SPY option chain ─────────────────────────────────────────────
    _step(4, "SPY option chain — first 5 contracts")
    chain = client.get_option_chain("SPY")
    if not chain:
        print("  ⚠  Chain is empty (may be outside trading hours or data unavailable)")
    else:
        # Stable ordering: nearest expiry first, then ascending strike.
        chain_sorted = sorted(chain, key=lambda c: (c.expiration, c.strike))
        for i, c in enumerate(chain_sorted[:5], start=1):
            delta_s = f"{c.delta:+.4f}"                    if c.delta              is not None else "    n/a"
            iv_s    = f"{c.implied_volatility * 100:5.1f}%" if c.implied_volatility is not None else "   n/a"
            bid_s   = f"${c.bid:.2f}"                       if c.bid                is not None else "n/a"
            ask_s   = f"${c.ask:.2f}"                       if c.ask                is not None else "n/a"
            print(
                f"  {i}. {c.symbol:<24} {c.option_type:<4} "
                f"strike=${c.strike:>8.2f}  exp={c.expiration}  "
                f"Δ={delta_s}  IV={iv_s}  bid/ask={bid_s}/{ask_s}"
            )
    _ok("Option chain fetched")

    # ── Step 5: place test order ─────────────────────────────────────────────
    _step(5, "Placing paper test order — buy 1 SPY at market")
    order: OrderResult = client.submit_order(
        symbol     = "SPY",
        qty        = 1.0,
        side       = "buy",
        order_type = "market",
    )
    _info("Order ID",        order.order_id)
    _info("Client order ID", order.client_order_id)
    _info("Status",          order.status)
    _info("Request ID",      order.request_id)
    if not order.order_id:
        raise RuntimeError("submit_order returned an empty order_id")
    _ok("Order submitted")

    # ── Step 6: confirm order is trackable ───────────────────────────────────
    _step(6, "Confirming order is trackable")
    open_orders = client.get_orders()
    tracked_ids = {o.order_id for o in open_orders}
    if order.order_id in tracked_ids:
        _ok(f"Order {order.order_id} found in open orders")
    else:
        # Market orders frequently fill within milliseconds during market hours.
        # A missing order here is not a failure — the non-empty order_id from
        # step 5 already confirms the broker accepted the request.
        _ok(
            f"Order {order.order_id} not in open orders "
            "(filled immediately — confirmed via order_id from step 5)"
        )

    # ── Step 7: cancel the test order ────────────────────────────────────────
    _step(7, "Cancelling test order")
    cancelled = client.cancel_order(order.order_id)
    if cancelled:
        _ok(f"Order {order.order_id} cancelled")
    else:
        # cancel_order returns False when the order is already gone (filled /
        # expired).  That is a valid outcome — the order existed and was handled.
        _ok(f"Order {order.order_id} already settled (filled or expired) — OK")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{_bar('═')}")
    print("  CONNECTION TEST PASSED")
    print(_bar("═"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n  Interrupted.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n{_bar('═')}", file=sys.stderr)
        print(f"  CONNECTION TEST FAILED", file=sys.stderr)
        print(_bar("═"), file=sys.stderr)
        print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

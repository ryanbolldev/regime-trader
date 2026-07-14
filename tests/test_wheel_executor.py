"""
tests/test_wheel_executor.py
-----------------------------
Phase 2a wheel executor: entry routing + NAV-scaled sizing under caps,
put management (close), open-order dedup, and assignment handoff.

The WheelStrategy is mocked so each test controls the returned action; the
strategy's own decision logic is covered by test_wheel_strategy.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from config import settings
from core.wheel_executor import WheelExecutor
from core.wheel_position_store import WheelPositionStore
from core.wheel_strategy import WheelAction, WheelActionType

PUT = "MSTR260821P00100000"   # MSTR 2026-08-21 put, strike 100


def _contract(symbol=PUT, strike=100.0, bid=2.45, ask=2.55, delta=-0.28, otype="put"):
    # default is a tight two-sided market (mid 2.50, 4% spread) that clears the
    # execution spread gate; individual tests widen it to exercise rejection.
    return SimpleNamespace(symbol=symbol, underlying="MSTR", expiration="2026-08-21",
                           strike=strike, option_type=otype, delta=delta,
                           bid=bid, ask=ask, implied_volatility=0.9)


def _pos(symbol, qty, avg=0.0, mv=0.0):
    return SimpleNamespace(symbol=symbol, qty=qty, avg_entry_price=avg,
                           market_value=mv, unrealized_pl=0.0, side="")


@pytest.fixture()
def env(tmp_path):
    client = MagicMock()
    client.get_positions.return_value = []
    client.get_orders.return_value    = []
    client.get_account.return_value   = SimpleNamespace(
        portfolio_value=100_000.0, options_buying_power=100_000.0)
    client.get_option_chain.return_value = []
    client.get_iv_rank.return_value      = 50.0
    client.is_market_open.return_value   = True
    store = WheelPositionStore(path=tmp_path / "wp.json")
    strat = MagicMock()
    return client, store, strat


def _sell_put(contract):
    return WheelAction(WheelActionType.SELL_PUT, contract, "delta", 3)


class TestEntry:

    def test_sizes_and_submits_sell_to_open(self, env):
        client, store, strat = env
        strat.get_next_action.return_value = _sell_put(_contract())
        WheelExecutor(client, store, strat).run_once(["MSTR"], 3, False)

        client.submit_order.assert_called_once()
        kw = client.submit_order.call_args.kwargs
        # budget = min(0.35*100k, 0.70*100k, 100k)=35k ; strike 100 → 10k/contract → 3
        assert kw["qty"] == 3
        assert kw["side"] == "sell"
        assert kw["position_intent"] == "sell_to_open"
        assert kw["order_type"] == "limit"
        assert kw["limit_price"] == 2.5   # mid of 2/3

    def test_no_entry_when_budget_below_one_contract(self, env):
        client, store, strat = env
        strat.get_next_action.return_value = _sell_put(_contract(strike=100_000.0))
        WheelExecutor(client, store, strat).run_once(["MSTR"], 3, False)
        client.submit_order.assert_not_called()

    def test_no_entry_at_max_positions(self, env, monkeypatch):
        client, store, strat = env
        monkeypatch.setattr(settings, "MAX_WHEEL_POSITIONS", 0)
        strat.get_next_action.return_value = _sell_put(_contract())
        WheelExecutor(client, store, strat).run_once(["MSTR"], 3, False)
        client.submit_order.assert_not_called()

    def test_total_deployed_cap_limits_size(self, env, monkeypatch):
        client, store, strat = env
        # already $68k deployed (existing short put) → remaining 70k-68k = 2k → 0 contracts
        client.get_positions.return_value = [_pos("AAPL250620P00680000", -1, avg=5.0)]
        monkeypatch.setattr(settings, "MAX_WHEEL_POSITIONS", 5)
        strat.get_next_action.return_value = _sell_put(_contract())
        WheelExecutor(client, store, strat).run_once(["MSTR"], 3, False)
        client.submit_order.assert_not_called()

    def test_no_order_when_strategy_waits(self, env):
        client, store, strat = env
        strat.get_next_action.return_value = WheelAction(WheelActionType.WAIT, None, "iv_low", 3)
        WheelExecutor(client, store, strat).run_once(["MSTR"], 3, False)
        client.submit_order.assert_not_called()

    def test_blocks_entry_when_iv_rank_none(self, env):
        # missing IV data must block, not silently bypass the IV gate
        client, store, strat = env
        client.get_iv_rank.return_value = None
        strat.get_next_action.return_value = _sell_put(_contract())
        WheelExecutor(client, store, strat).run_once(["MSTR"], 3, False)
        client.submit_order.assert_not_called()

    def test_skips_wide_spread(self, env):
        client, store, strat = env
        strat.get_next_action.return_value = _sell_put(_contract(bid=1.0, ask=3.0))  # 100% spread
        WheelExecutor(client, store, strat).run_once(["MSTR"], 3, False)
        client.submit_order.assert_not_called()


class TestManagement:

    def test_closes_open_put_on_close_action(self, env):
        client, store, strat = env
        client.get_positions.return_value    = [_pos(PUT, -1, avg=2.5)]     # PUT_SOLD, premium 2.5
        client.get_option_chain.return_value = [_contract(bid=1.0, ask=1.2)]  # mark 1.1 (profit)
        strat.get_next_action.return_value   = WheelAction(WheelActionType.CLOSE, None, "profit", 3)

        WheelExecutor(client, store, strat).run_once(["MSTR"], 3, False)
        kw = client.submit_order.call_args.kwargs
        assert kw["side"] == "buy"
        assert kw["position_intent"] == "buy_to_close"
        assert kw["qty"] == 1
        assert kw["limit_price"] == 1.2   # marketable — at the ask, not the mid

    def test_holds_put_on_wait(self, env):
        client, store, strat = env
        client.get_positions.return_value    = [_pos(PUT, -1, avg=2.5)]
        client.get_option_chain.return_value = [_contract(bid=1.0, ask=1.2)]
        strat.get_next_action.return_value   = WheelAction(WheelActionType.WAIT, None, "hold", 3)
        WheelExecutor(client, store, strat).run_once(["MSTR"], 3, False)
        client.submit_order.assert_not_called()

    def test_missing_mark_alerts_and_suspends_pnl(self, env, monkeypatch):
        client, store, strat = env
        sent = []
        monkeypatch.setattr("core.wheel_executor.alerts",
                            SimpleNamespace(send=lambda *a, **k: sent.append(a)))
        client.get_positions.return_value    = [_pos(PUT, -1, avg=2.5)]   # PUT_SOLD
        client.get_option_chain.return_value = []                          # active leg absent → no mark
        strat.get_next_action.return_value   = WheelAction(WheelActionType.WAIT, None, "hold", 3)
        WheelExecutor(client, store, strat).run_once(["MSTR"], 3, False)
        assert any("cannot price" in a[1] for a in sent)
        client.submit_order.assert_not_called()

    def test_order_failure_alerts(self, env, monkeypatch):
        client, store, strat = env
        sent = []
        monkeypatch.setattr("core.wheel_executor.alerts",
                            SimpleNamespace(send=lambda *a, **k: sent.append(a)))
        client.submit_order.side_effect = Exception("403 Forbidden")
        strat.get_next_action.return_value = _sell_put(_contract())
        WheelExecutor(client, store, strat).run_once(["MSTR"], 3, False)   # must not raise
        assert any("ORDER FAILED" in a[1] for a in sent)


class TestGuards:

    def test_cancels_stale_order_then_reprices(self, env):
        client, store, strat = env
        client.get_orders.return_value = [
            SimpleNamespace(symbol=PUT, status="new", order_id="abc12345")
        ]
        strat.get_next_action.return_value = _sell_put(_contract())
        WheelExecutor(client, store, strat).run_once(["MSTR"], 3, False)
        client.cancel_order.assert_called_once_with("abc12345")  # stale order pulled
        client.submit_order.assert_called_once()                 # re-priced entry

    def test_non_option_order_not_cancelled(self, env):
        client, store, strat = env
        # a plain-equity resting order (non-OCC symbol) must never be touched
        client.get_orders.return_value = [
            SimpleNamespace(symbol="MSTR", status="new", order_id="equity01")
        ]
        strat.get_next_action.return_value = _sell_put(_contract())
        WheelExecutor(client, store, strat).run_once(["MSTR"], 3, False)
        client.cancel_order.assert_not_called()

    def test_assigned_places_no_order(self, env):
        client, store, strat = env
        client.get_positions.return_value = [_pos("MSTR", 100, avg=100.0, mv=9500.0)]
        WheelExecutor(client, store, strat).run_once(["MSTR"], 3, False)
        client.submit_order.assert_not_called()

    def test_assignment_transition_alerts(self, env, monkeypatch):
        client, store, strat = env
        sent = []
        monkeypatch.setattr("core.wheel_executor.alerts",
                            SimpleNamespace(send=lambda *a, **k: sent.append(a)))
        ex = WheelExecutor(client, store, strat)
        # first pass: short put present → PUT_SOLD
        client.get_positions.return_value = [_pos(PUT, -1, avg=2.5)]
        ex.run_once(["MSTR"], 3, False)
        # second pass: shares present, put gone → ASSIGNED (transition alert)
        client.get_positions.return_value = [_pos("MSTR", 100, avg=100.0, mv=9500.0)]
        ex.run_once(["MSTR"], 3, False)
        assert any("ASSIGNED" in a[1] for a in sent)


class TestMarketHours:

    def test_market_closed_places_no_orders(self, env):
        client, store, strat = env
        client.is_market_open.return_value = False
        strat.get_next_action.return_value = _sell_put(_contract())
        WheelExecutor(client, store, strat).run_once(["MSTR"], 3, False)
        client.submit_order.assert_not_called()
        client.cancel_order.assert_not_called()

    def test_market_closed_still_reconciles_and_alerts(self, env, monkeypatch):
        client, store, strat = env
        client.is_market_open.return_value = False
        sent = []
        monkeypatch.setattr("core.wheel_executor.alerts",
                            SimpleNamespace(send=lambda *a, **k: sent.append(a)))
        ex = WheelExecutor(client, store, strat)
        client.get_positions.return_value = [_pos(PUT, -1, avg=2.5)]       # PUT_SOLD
        ex.run_once(["MSTR"], 3, False)
        client.get_positions.return_value = [_pos("MSTR", 100, avg=100.0, mv=9500.0)]  # assigned
        ex.run_once(["MSTR"], 3, False)
        # assignment detected + alerted even though the market is closed …
        assert any("ASSIGNED" in a[1] for a in sent)
        # … but no orders were placed
        client.submit_order.assert_not_called()

"""
tests/test_wheel_position_store.py
-----------------------------------
Hybrid wheel state store: broker-derived phase + open-leg premium, transition
detection (assignment / expiry), broker-wins reconciliation, lifetime-premium
accumulation, and persistence.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.wheel_position_store import WheelPositionStore, _broker_state
from core.wheel_strategy import WheelState

PUT  = "MSTR260821P00400000"   # MSTR 2026-08-21 put, strike 400
CALL = "MSTR260821C00500000"   # MSTR 2026-08-21 call, strike 500


def _pos(symbol, qty, avg=0.0):
    return SimpleNamespace(symbol=symbol, qty=qty, avg_entry_price=avg,
                           market_value=0.0, unrealized_pl=0.0, side="")


@pytest.fixture()
def store(tmp_path):
    return WheelPositionStore(path=tmp_path / "wheel_positions.json")


class TestBrokerState:

    def test_empty_is_cash(self):
        assert _broker_state("MSTR", [])[0] == WheelState.CASH

    def test_short_put_reports_premium_and_qty(self):
        phase, active, shares, cost, prem, qty = _broker_state("MSTR", [_pos(PUT, -2, avg=2.5)])
        assert phase == WheelState.PUT_SOLD
        assert active == PUT
        assert prem == 2.5 and qty == 2      # avg_entry_price = premium/share; |qty| = contracts

    def test_hundred_shares_is_assigned(self):
        phase, _, shares, cost, _, _ = _broker_state("MSTR", [_pos("MSTR", 100, avg=400.0)])
        assert phase == WheelState.ASSIGNED
        assert shares == 100 and cost == 400.0

    def test_short_call_with_shares_is_call_sold(self):
        phase, active, *_ = _broker_state(
            "MSTR", [_pos("MSTR", 100, avg=400.0), _pos(CALL, -1, avg=3.0)]
        )
        assert phase == WheelState.CALL_SOLD and active == CALL

    def test_other_underlying_ignored(self):
        assert _broker_state("MSTR", [_pos("AAPL250117P00150000", -1)])[0] == WheelState.CASH

    def test_long_option_not_counted_as_short(self):
        assert _broker_state("MSTR", [_pos(PUT, +1)])[0] == WheelState.CASH


class TestReconcile:

    def test_fresh_cash_no_transition(self, store):
        res = store.reconcile("MSTR", [])
        assert res.record.phase == WheelState.CASH
        assert res.transition is None

    def test_entering_put_derives_economics_from_broker(self, store):
        res = store.reconcile("MSTR", [_pos(PUT, -1, avg=2.5)], current_regime=3)
        assert res.record.phase == WheelState.PUT_SOLD
        assert res.transition == "CASH->PUT_SOLD"
        assert res.record.active_contract == PUT
        assert res.record.active_contract_premium == 2.5   # for the executor's P&L
        assert res.record.contracts == 1
        assert res.record.premium_collected_total == 250.0  # 2.5 × 100 × 1
        assert res.record.entry_regime == 3

    def test_broker_wins_when_put_disappears(self, store):
        store.reconcile("MSTR", [_pos(PUT, -1, avg=2.5)], current_regime=3)   # PUT_SOLD
        res = store.reconcile("MSTR", [])                                     # expired worthless
        assert res.record.phase == WheelState.CASH
        assert res.transition == "PUT_SOLD->CASH"
        assert res.record.active_contract is None
        assert res.record.active_contract_premium == 0.0
        assert res.record.premium_collected_total == 250.0                    # lifetime retained

    def test_assignment_detected(self, store):
        store.reconcile("MSTR", [_pos(PUT, -1, avg=2.5)], current_regime=3)   # PUT_SOLD
        res = store.reconcile("MSTR", [_pos("MSTR", 100, avg=400.0)])         # assigned
        assert res.record.phase == WheelState.ASSIGNED
        assert res.transition == "PUT_SOLD->ASSIGNED"
        assert res.record.shares_owned == 100
        assert res.record.cost_basis == 400.0
        assert res.record.active_contract is None
        assert res.record.premium_collected_total == 250.0

    def test_lifetime_premium_accumulates_over_cycles(self, store):
        store.reconcile("MSTR", [_pos(PUT, -1, avg=2.0)], current_regime=3)   # +200
        store.reconcile("MSTR", [])                                           # closed
        store.reconcile("MSTR", [_pos(PUT, -1, avg=3.0)], current_regime=3)   # +300
        assert store.get("MSTR").premium_collected_total == 500.0


class TestPersistence:

    def test_round_trip(self, tmp_path):
        path = tmp_path / "wheel_positions.json"
        s1 = WheelPositionStore(path=path)
        s1.reconcile("MSTR", [_pos(PUT, -2, avg=2.5)], current_regime=3)

        s2 = WheelPositionStore(path=path)
        rec = s2.get("MSTR")
        assert rec is not None
        assert rec.phase == WheelState.PUT_SOLD
        assert rec.active_contract == PUT
        assert rec.contracts == 2
        assert rec.premium_collected_total == 500.0
        assert rec.entry_regime == 3

    def test_to_wheel_position_maps_fields(self, store):
        store.reconcile("MSTR", [_pos(PUT, -1, avg=2.5)], current_regime=3)
        wp = store.get("MSTR").to_wheel_position()
        assert wp.symbol == "MSTR"
        assert wp.phase == WheelState.PUT_SOLD
        assert wp.active_contract == PUT
        assert wp.premium_collected_total == 250.0

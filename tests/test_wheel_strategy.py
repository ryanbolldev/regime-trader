"""
tests/test_wheel_strategy.py
-----------------------------
Tests for core/wheel_strategy.py, risk_manager wheel validators,
position_tracker wheel extensions, and the regime_strategies integration.
"""

from __future__ import annotations

import datetime as _dt
from unittest.mock import MagicMock, patch

import pytest

from broker.alpaca_client import OptionContract
from core.position_tracker import (
    _wheel_positions,
    get_wheel_position,
    get_wheel_state,
    track_wheel_position,
    update_on_assignment,
    update_on_close,
    update_on_expiry,
)
from core.regime_strategies import Signal, get_signal
from core.risk_manager import InsufficientFundsError, RiskManager
from core.wheel_strategy import (
    WheelAction,
    WheelActionType,
    WheelPosition,
    WheelState,
    WheelStrategy,
    _dte_from_occ,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOCK_TODAY = _dt.date(2024, 12, 1)   # fixed reference date for all DTE calcs
# Expiry 40 DTE from MOCK_TODAY → 2025-01-10
EXPIRY_40 = "2025-01-10"             # 2025-01-10 − 2024-12-01 = 40 days
OCC_DATE_40 = "250110"               # YYMMDD portion for 2025-01-10
# Expiry 5 DTE from MOCK_TODAY → 2024-12-06  (inside gamma-risk zone)
EXPIRY_5 = "2024-12-06"
OCC_DATE_5 = "241206"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fixed_today():
    """Patch _today() so every DTE calculation is deterministic."""
    with patch("core.wheel_strategy._today", return_value=MOCK_TODAY):
        yield MOCK_TODAY


@pytest.fixture(autouse=True)
def clear_wheel_positions():
    """Reset position_tracker wheel state before every test."""
    _wheel_positions.clear()
    yield
    _wheel_positions.clear()


@pytest.fixture
def ws():
    return WheelStrategy()


def _put(underlying: str, strike: float, delta: float,
         expiration: str = EXPIRY_40) -> OptionContract:
    strike_int = int(strike * 1000)
    yymmdd = expiration.replace("-", "")[2:]
    occ = f"{underlying}{yymmdd}P{strike_int:08d}"
    return OptionContract(
        symbol=occ, underlying=underlying, expiration=expiration,
        strike=strike, option_type="put", delta=delta,
        implied_volatility=0.20, bid=1.50, ask=1.60,
    )


def _call(underlying: str, strike: float, delta: float,
          expiration: str = EXPIRY_40) -> OptionContract:
    strike_int = int(strike * 1000)
    yymmdd = expiration.replace("-", "")[2:]
    occ = f"{underlying}{yymmdd}C{strike_int:08d}"
    return OptionContract(
        symbol=occ, underlying=underlying, expiration=expiration,
        strike=strike, option_type="call", delta=delta,
        implied_volatility=0.20, bid=1.50, ask=1.60,
    )


def _cash_position(symbol: str = "SPY") -> WheelPosition:
    return WheelPosition(
        symbol=symbol, phase=WheelState.CASH, shares_owned=0, cost_basis=0.0,
        active_contract=None, premium_collected_total=0.0, entry_regime=3,
    )


def _put_sold_position(symbol: str = "SPY", occ: str | None = None) -> WheelPosition:
    contract = occ or f"{symbol}{OCC_DATE_40}P00580000"
    return WheelPosition(
        symbol=symbol, phase=WheelState.PUT_SOLD, shares_owned=0, cost_basis=0.0,
        active_contract=contract, premium_collected_total=200.0, entry_regime=3,
    )


def _assigned_position(symbol: str = "SPY", cost_basis: float = 580.0) -> WheelPosition:
    return WheelPosition(
        symbol=symbol, phase=WheelState.ASSIGNED, shares_owned=100,
        cost_basis=cost_basis, active_contract=None,
        premium_collected_total=200.0, entry_regime=3,
    )


def _call_sold_position(symbol: str = "SPY", occ: str | None = None) -> WheelPosition:
    contract = occ or f"{symbol}{OCC_DATE_40}C00595000"
    return WheelPosition(
        symbol=symbol, phase=WheelState.CALL_SOLD, shares_owned=100,
        cost_basis=580.0, active_contract=contract,
        premium_collected_total=400.0, entry_regime=3,
    )


# ---------------------------------------------------------------------------
# TestDteFromOcc
# ---------------------------------------------------------------------------

class TestDteFromOcc:
    def test_round_trip_40_dte(self):
        occ = f"SPY{OCC_DATE_40}P00580000"
        assert _dte_from_occ(occ, MOCK_TODAY) == 40

    def test_round_trip_5_dte(self):
        occ = f"SPY{OCC_DATE_5}P00580000"
        assert _dte_from_occ(occ, MOCK_TODAY) == 5

    def test_bad_symbol_returns_none(self):
        assert _dte_from_occ("not-an-occ", MOCK_TODAY) is None


# ---------------------------------------------------------------------------
# TestGetPutToSell — delta target and DTE window
# ---------------------------------------------------------------------------

class TestGetPutToSell:
    def test_selects_closest_delta_to_target(self, ws):
        chain = [
            _put("SPY", 590.0, -0.35),   # further from -0.28
            _put("SPY", 585.0, -0.29),   # closest
            _put("SPY", 575.0, -0.20),   # further from -0.28
        ]
        result = ws.get_put_to_sell("SPY", chain, 100_000, 60_000, regime=3)
        assert result is not None
        assert result.strike == 585.0

    def test_rejects_contract_outside_dte_window_too_short(self, ws):
        # 5 DTE — below WHEEL_MIN_DTE=30
        chain = [_put("SPY", 580.0, -0.28, expiration=EXPIRY_5)]
        assert ws.get_put_to_sell("SPY", chain, 100_000, 60_000, regime=3) is None

    def test_rejects_contract_outside_dte_window_too_long(self, ws):
        # 90 DTE — above WHEEL_MAX_DTE=45
        expiry_90 = "2025-03-01"   # ~90 days from MOCK_TODAY
        chain = [_put("SPY", 580.0, -0.28, expiration=expiry_90)]
        assert ws.get_put_to_sell("SPY", chain, 100_000, 60_000, regime=3) is None

    def test_accepts_contract_at_min_dte_boundary(self, ws):
        # 30 DTE exactly
        expiry_30 = "2025-01-01"   # 31 days from 2024-12-01... let's compute properly
        # 2024-12-01 + 30 = 2024-12-31
        expiry_30 = "2024-12-31"
        chain = [_put("SPY", 580.0, -0.28, expiration=expiry_30)]
        result = ws.get_put_to_sell("SPY", chain, 100_000, 60_000, regime=3)
        assert result is not None

    def test_only_runs_in_bull_regime(self, ws):
        chain = [_put("SPY", 580.0, -0.28)]
        assert ws.get_put_to_sell("SPY", chain, 100_000, 60_000, regime=3) is not None

    def test_only_runs_in_neutral_regime(self, ws):
        chain = [_put("SPY", 580.0, -0.28)]
        assert ws.get_put_to_sell("SPY", chain, 100_000, 60_000, regime=2) is not None

    def test_blocked_in_bear_regime(self, ws):
        chain = [_put("SPY", 580.0, -0.28)]
        assert ws.get_put_to_sell("SPY", chain, 100_000, 60_000, regime=1) is None

    def test_blocked_in_crash_regime(self, ws):
        chain = [_put("SPY", 580.0, -0.28)]
        assert ws.get_put_to_sell("SPY", chain, 100_000, 60_000, regime=0) is None

    def test_blocked_in_euphoria_regime(self, ws):
        chain = [_put("SPY", 580.0, -0.28)]
        assert ws.get_put_to_sell("SPY", chain, 100_000, 60_000, regime=4) is None

    def test_blocked_when_buying_power_insufficient(self, ws):
        # strike=580 → assignment cost=$58,000; buying_power=50,000 → insufficient
        chain = [_put("SPY", 580.0, -0.28)]
        assert ws.get_put_to_sell("SPY", chain, 100_000, 50_000, regime=3) is None

    def test_returns_none_when_chain_empty(self, ws):
        assert ws.get_put_to_sell("SPY", [], 100_000, 60_000, regime=3) is None

    def test_filters_by_underlying(self, ws):
        chain = [_put("QQQ", 450.0, -0.28)]   # wrong underlying
        assert ws.get_put_to_sell("SPY", chain, 100_000, 60_000, regime=3) is None

    def test_skips_contracts_with_no_delta(self, ws):
        bad = OptionContract(
            symbol=f"SPY{OCC_DATE_40}P00580000", underlying="SPY",
            expiration=EXPIRY_40, strike=580.0, option_type="put",
            delta=None, implied_volatility=None, bid=None, ask=None,
        )
        assert ws.get_put_to_sell("SPY", [bad], 100_000, 60_000, regime=3) is None


# ---------------------------------------------------------------------------
# TestGetCallToSell — cost-basis and shares constraints
# ---------------------------------------------------------------------------

class TestGetCallToSell:
    def test_selects_closest_delta_to_target(self, ws):
        chain = [
            _call("SPY", 595.0, 0.30),
            _call("SPY", 600.0, 0.28),   # exact target
            _call("SPY", 610.0, 0.20),
        ]
        result = ws.get_call_to_sell("SPY", chain, 100, 580.0, regime=2)
        assert result is not None
        assert result.strike == 600.0

    def test_never_sells_below_cost_basis(self, ws):
        # cost_basis=590; strike=585 is below cost_basis → reject
        chain = [_call("SPY", 585.0, 0.28)]
        assert ws.get_call_to_sell("SPY", chain, 100, 590.0, regime=2) is None

    def test_strike_equal_to_cost_basis_rejected(self, ws):
        # must be strictly above cost_basis
        chain = [_call("SPY", 580.0, 0.28)]
        assert ws.get_call_to_sell("SPY", chain, 100, 580.0, regime=2) is None

    def test_strike_above_cost_basis_accepted(self, ws):
        chain = [_call("SPY", 581.0, 0.28)]
        result = ws.get_call_to_sell("SPY", chain, 100, 580.0, regime=2)
        assert result is not None

    def test_blocked_when_fewer_than_100_shares(self, ws):
        chain = [_call("SPY", 595.0, 0.28)]
        assert ws.get_call_to_sell("SPY", chain, 99, 580.0, regime=2) is None

    def test_allowed_with_exactly_100_shares(self, ws):
        chain = [_call("SPY", 595.0, 0.28)]
        assert ws.get_call_to_sell("SPY", chain, 100, 580.0, regime=2) is not None

    def test_only_runs_in_neutral_regime(self, ws):
        chain = [_call("SPY", 595.0, 0.28)]
        assert ws.get_call_to_sell("SPY", chain, 100, 580.0, regime=2) is not None

    def test_only_runs_in_euphoria_regime(self, ws):
        chain = [_call("SPY", 595.0, 0.28)]
        assert ws.get_call_to_sell("SPY", chain, 100, 580.0, regime=4) is not None

    def test_blocked_in_bull_regime(self, ws):
        chain = [_call("SPY", 595.0, 0.28)]
        assert ws.get_call_to_sell("SPY", chain, 100, 580.0, regime=3) is None

    def test_blocked_in_bear_regime(self, ws):
        chain = [_call("SPY", 595.0, 0.28)]
        assert ws.get_call_to_sell("SPY", chain, 100, 580.0, regime=1) is None

    def test_rejects_outside_dte_window(self, ws):
        chain = [_call("SPY", 595.0, 0.28, expiration=EXPIRY_5)]
        assert ws.get_call_to_sell("SPY", chain, 100, 580.0, regime=2) is None


# ---------------------------------------------------------------------------
# TestShouldCloseEarly
# ---------------------------------------------------------------------------

class TestShouldCloseEarly:
    def test_closes_on_bear_regime(self, ws):
        pos = _put_sold_position()
        assert ws.should_close_early(pos, current_regime=1, current_pnl_pct=0.0)

    def test_closes_on_crash_regime(self, ws):
        pos = _put_sold_position()
        assert ws.should_close_early(pos, current_regime=0, current_pnl_pct=0.0)

    def test_does_not_close_in_bull_regime(self, ws):
        pos = _put_sold_position()
        assert not ws.should_close_early(pos, current_regime=3, current_pnl_pct=0.1)

    def test_closes_at_50_pct_profit(self, ws):
        pos = _put_sold_position()
        assert ws.should_close_early(pos, current_regime=3, current_pnl_pct=0.50)

    def test_does_not_close_below_50_pct_profit(self, ws):
        pos = _put_sold_position()
        assert not ws.should_close_early(pos, current_regime=3, current_pnl_pct=0.49)

    def test_closes_at_200_pct_loss(self, ws):
        pos = _put_sold_position()
        assert ws.should_close_early(pos, current_regime=3, current_pnl_pct=-2.00)

    def test_does_not_close_below_200_pct_loss_threshold(self, ws):
        pos = _put_sold_position()
        assert not ws.should_close_early(pos, current_regime=3, current_pnl_pct=-1.99)

    def test_closes_on_gamma_risk_with_loss(self, ws):
        # active contract expiring in 5 DTE, position at a loss
        occ_5dte = f"SPY{OCC_DATE_5}P00580000"
        pos = _put_sold_position(occ=occ_5dte)
        assert ws.should_close_early(pos, current_regime=3, current_pnl_pct=-0.10)

    def test_no_gamma_close_when_profitable(self, ws):
        # < 7 DTE but profitable → do not close
        occ_5dte = f"SPY{OCC_DATE_5}P00580000"
        pos = _put_sold_position(occ=occ_5dte)
        assert not ws.should_close_early(pos, current_regime=3, current_pnl_pct=0.10)

    def test_no_gamma_close_when_no_contract(self, ws):
        pos = _assigned_position()  # no active_contract
        assert not ws.should_close_early(pos, current_regime=3, current_pnl_pct=-0.10)


# ---------------------------------------------------------------------------
# TestUncertaintyModifier
# ---------------------------------------------------------------------------

class TestUncertaintyModifier:
    def test_wait_returned_when_uncertain_in_cash_phase(self, ws):
        chain = [_put("SPY", 580.0, -0.28)]
        action = ws.get_next_action(
            _cash_position(), current_regime=3, option_chain=chain,
            portfolio_nav=100_000, buying_power=60_000, is_uncertain=True,
        )
        assert action.action == WheelActionType.WAIT
        assert action.reason == "uncertain_regime"

    def test_wait_returned_when_uncertain_in_assigned_phase(self, ws):
        chain = [_call("SPY", 595.0, 0.28)]
        action = ws.get_next_action(
            _assigned_position(), current_regime=2, option_chain=chain,
            portfolio_nav=100_000, buying_power=60_000, is_uncertain=True,
        )
        assert action.action == WheelActionType.WAIT
        assert action.reason == "uncertain_regime"

    def test_no_new_put_entry_during_uncertain_regime(self, ws):
        # even with a perfect chain, uncertainty blocks new put entries
        chain = [_put("SPY", 580.0, -0.28)]
        action = ws.get_next_action(
            _cash_position(), current_regime=3, option_chain=chain,
            portfolio_nav=100_000, buying_power=60_000, is_uncertain=True,
        )
        assert action.action != WheelActionType.SELL_PUT


# ---------------------------------------------------------------------------
# TestGetNextAction — per-phase routing
# ---------------------------------------------------------------------------

class TestGetNextAction:
    def test_none_position_returns_sit_out(self, ws):
        action = ws.get_next_action(
            None, current_regime=3, option_chain=[],
            portfolio_nav=100_000, buying_power=60_000,
        )
        assert action.action == WheelActionType.SIT_OUT

    def test_cash_with_good_chain_returns_sell_put(self, ws):
        chain = [_put("SPY", 580.0, -0.28)]
        action = ws.get_next_action(
            _cash_position(), current_regime=3, option_chain=chain,
            portfolio_nav=100_000, buying_power=60_000,
        )
        assert action.action == WheelActionType.SELL_PUT
        assert action.contract is not None

    def test_cash_with_empty_chain_returns_sit_out(self, ws):
        action = ws.get_next_action(
            _cash_position(), current_regime=3, option_chain=[],
            portfolio_nav=100_000, buying_power=60_000,
        )
        assert action.action == WheelActionType.SIT_OUT

    def test_put_sold_holding_returns_wait(self, ws):
        action = ws.get_next_action(
            _put_sold_position(), current_regime=3, option_chain=[],
            portfolio_nav=100_000, buying_power=60_000, current_pnl_pct=0.10,
        )
        assert action.action == WheelActionType.WAIT

    def test_put_sold_triggers_close_on_bear(self, ws):
        action = ws.get_next_action(
            _put_sold_position(), current_regime=1, option_chain=[],
            portfolio_nav=100_000, buying_power=60_000, current_pnl_pct=0.0,
        )
        assert action.action == WheelActionType.CLOSE

    def test_assigned_with_good_chain_returns_sell_call(self, ws):
        chain = [_call("SPY", 595.0, 0.28)]
        action = ws.get_next_action(
            _assigned_position(), current_regime=2, option_chain=chain,
            portfolio_nav=100_000, buying_power=60_000,
        )
        assert action.action == WheelActionType.SELL_CALL
        assert action.contract is not None

    def test_assigned_no_good_chain_returns_wait(self, ws):
        action = ws.get_next_action(
            _assigned_position(), current_regime=2, option_chain=[],
            portfolio_nav=100_000, buying_power=60_000,
        )
        assert action.action == WheelActionType.WAIT

    def test_call_sold_holding_returns_wait(self, ws):
        action = ws.get_next_action(
            _call_sold_position(), current_regime=2, option_chain=[],
            portfolio_nav=100_000, buying_power=60_000, current_pnl_pct=0.10,
        )
        assert action.action == WheelActionType.WAIT

    def test_call_sold_triggers_close_at_50_pct_profit(self, ws):
        action = ws.get_next_action(
            _call_sold_position(), current_regime=2, option_chain=[],
            portfolio_nav=100_000, buying_power=60_000, current_pnl_pct=0.50,
        )
        assert action.action == WheelActionType.CLOSE

    def test_action_carries_current_regime(self, ws):
        chain = [_put("SPY", 580.0, -0.28)]
        action = ws.get_next_action(
            _cash_position(), current_regime=3, option_chain=chain,
            portfolio_nav=100_000, buying_power=60_000,
        )
        assert action.regime == 3


# ---------------------------------------------------------------------------
# TestRiskManagerWheelValidators
# ---------------------------------------------------------------------------

class TestRiskManagerWheelValidators:
    @pytest.fixture
    def rm(self):
        r = RiskManager()
        r.initialize(100_000)
        return r

    def test_validate_put_sale_passes_with_sufficient_funds(self, rm):
        # strike=500, qty=1 → need $50,000; buying_power=$60,000 → OK
        rm.validate_put_sale(strike=500.0, qty=1, buying_power=60_000)

    def test_validate_put_sale_raises_when_insufficient(self, rm):
        # strike=580, qty=1 → need $58,000; buying_power=$50,000 → fail
        with pytest.raises(InsufficientFundsError):
            rm.validate_put_sale(strike=580.0, qty=1, buying_power=50_000)

    def test_validate_put_sale_exact_boundary_passes(self, rm):
        # buying_power exactly equals assignment cost → allowed
        rm.validate_put_sale(strike=500.0, qty=1, buying_power=50_000)

    def test_validate_put_sale_multi_contract(self, rm):
        # qty=2, strike=300 → need $60,000; buying_power=$59,999 → fail
        with pytest.raises(InsufficientFundsError):
            rm.validate_put_sale(strike=300.0, qty=2, buying_power=59_999)

    def test_validate_call_sale_passes_when_assigned(self, rm):
        track_wheel_position("SPY", WheelState.PUT_SOLD, "SPY250110P00580000", 200.0)
        update_on_assignment("SPY", shares=100, cost_basis=580.0)
        pt = MagicMock()
        pt.get_wheel_state.return_value = WheelState.ASSIGNED
        rm.validate_call_sale("SPY", 1, pt)  # must not raise

    def test_validate_call_sale_raises_when_not_assigned(self, rm):
        pt = MagicMock()
        pt.get_wheel_state.return_value = WheelState.CASH
        with pytest.raises(InsufficientFundsError):
            rm.validate_call_sale("SPY", 1, pt)

    def test_validate_call_sale_raises_when_state_is_none(self, rm):
        pt = MagicMock()
        pt.get_wheel_state.return_value = None
        with pytest.raises(InsufficientFundsError):
            rm.validate_call_sale("SPY", 1, pt)


# ---------------------------------------------------------------------------
# TestPositionTracker
# ---------------------------------------------------------------------------

class TestPositionTrackerWheelExtensions:
    def test_track_creates_new_position(self):
        track_wheel_position("SPY", WheelState.PUT_SOLD, "SPY250110P00580000", 200.0)
        assert get_wheel_state("SPY") == WheelState.PUT_SOLD

    def test_get_wheel_state_returns_none_for_unknown(self):
        assert get_wheel_state("AAPL") is None

    def test_track_accumulates_premium(self):
        track_wheel_position("SPY", WheelState.PUT_SOLD, "SPY250110P00580000", 200.0)
        track_wheel_position("SPY", WheelState.CALL_SOLD, "SPY250110C00595000", 150.0)
        pos = get_wheel_position("SPY")
        assert pos.premium_collected_total == 350.0

    def test_update_on_assignment(self):
        track_wheel_position("SPY", WheelState.PUT_SOLD, "SPY250110P00580000", 200.0)
        update_on_assignment("SPY", shares=100, cost_basis=580.0)
        pos = get_wheel_position("SPY")
        assert pos.phase == WheelState.ASSIGNED
        assert pos.shares_owned == 100
        assert pos.cost_basis == 580.0
        assert pos.active_contract is None

    def test_update_on_expiry_put_to_cash(self):
        track_wheel_position("SPY", WheelState.PUT_SOLD, "SPY250110P00580000", 200.0)
        update_on_expiry("SPY")
        assert get_wheel_state("SPY") == WheelState.CASH

    def test_update_on_expiry_call_to_assigned(self):
        track_wheel_position("SPY", WheelState.CALL_SOLD, "SPY250110C00595000", 150.0)
        update_on_expiry("SPY")
        assert get_wheel_state("SPY") == WheelState.ASSIGNED

    def test_update_on_close_put_goes_to_cash(self):
        track_wheel_position("SPY", WheelState.PUT_SOLD, "SPY250110P00580000", 200.0)
        update_on_close("SPY", closing_cost=100.0)
        pos = get_wheel_position("SPY")
        assert pos.phase == WheelState.CASH
        assert pos.premium_collected_total == 100.0   # 200 − 100

    def test_update_on_close_call_goes_to_assigned(self):
        track_wheel_position("SPY", WheelState.CALL_SOLD, "SPY250110C00595000", 400.0)
        update_on_close("SPY", closing_cost=50.0)
        pos = get_wheel_position("SPY")
        assert pos.phase == WheelState.ASSIGNED
        assert pos.premium_collected_total == 350.0

    def test_update_on_expiry_noop_for_unknown_symbol(self):
        update_on_expiry("AAPL")   # must not raise


# ---------------------------------------------------------------------------
# TestRegimeStrategiesIntegration
# ---------------------------------------------------------------------------

class TestRegimeStrategiesIntegration:
    def test_signal_has_wheel_action_when_strategy_provided(self):
        chain = [_put("SPY", 580.0, -0.28)]
        ws = WheelStrategy()
        sig = get_signal(
            3, 0.8, 100_000, 0.5, False,
            wheel_strategy=ws,
            wheel_position=_cash_position(),
            option_chain=chain,
            buying_power=60_000,
        )
        assert sig.wheel_action is not None
        assert sig.wheel_action.action == WheelActionType.SELL_PUT

    def test_signal_wheel_action_none_without_strategy(self):
        sig = get_signal(3, 0.8, 100_000, 0.5, False)
        assert sig.wheel_action is None

    def test_uncertain_regime_propagates_to_wheel_action(self):
        chain = [_put("SPY", 580.0, -0.28)]
        ws = WheelStrategy()
        sig = get_signal(
            3, 0.8, 100_000, 0.5, True,   # is_uncertain=True
            wheel_strategy=ws,
            wheel_position=_cash_position(),
            option_chain=chain,
            buying_power=60_000,
        )
        assert sig.wheel_action.action == WheelActionType.WAIT
        assert sig.wheel_action.reason == "uncertain_regime"

    def test_existing_signal_fields_unaffected(self):
        sig = get_signal(3, 0.8, 100_000, 0.5, False)
        assert sig.regime == 3
        assert sig.regime_name == "bull"
        assert sig.wheel_action is None


# ---------------------------------------------------------------------------
# TestFullWheelCycle
# ---------------------------------------------------------------------------

class TestFullWheelCycle:
    """Simulate the complete cash → put sold → assigned → call sold → cash cycle."""

    def test_full_cycle(self, ws):
        # ── Phase 1: CASH → SELL_PUT ───────────────────────────────────────
        put_chain = [_put("SPY", 580.0, -0.28)]
        action = ws.get_next_action(
            _cash_position(), current_regime=3,
            option_chain=put_chain, portfolio_nav=100_000, buying_power=60_000,
        )
        assert action.action == WheelActionType.SELL_PUT
        assert action.contract.strike == 580.0

        # Record the put sale in position tracker
        track_wheel_position("SPY", WheelState.PUT_SOLD, action.contract.symbol, 200.0)
        assert get_wheel_state("SPY") == WheelState.PUT_SOLD

        # ── Phase 2: PUT_SOLD — hold (no early-close trigger) ──────────────
        pos_put = get_wheel_position("SPY")
        action2 = ws.get_next_action(
            pos_put, current_regime=3, option_chain=[],
            portfolio_nav=100_000, buying_power=60_000, current_pnl_pct=0.10,
        )
        assert action2.action == WheelActionType.WAIT

        # ── Phase 3: Assignment → ASSIGNED ────────────────────────────────
        update_on_assignment("SPY", shares=100, cost_basis=580.0)
        assert get_wheel_state("SPY") == WheelState.ASSIGNED

        # ── Phase 4: ASSIGNED → SELL_CALL ─────────────────────────────────
        call_chain = [_call("SPY", 595.0, 0.28)]
        pos_assigned = get_wheel_position("SPY")
        action3 = ws.get_next_action(
            pos_assigned, current_regime=2,
            option_chain=call_chain, portfolio_nav=100_000, buying_power=60_000,
        )
        assert action3.action == WheelActionType.SELL_CALL
        assert action3.contract.strike == 595.0
        assert action3.contract.strike > pos_assigned.cost_basis

        # Record the call sale
        track_wheel_position("SPY", WheelState.CALL_SOLD, action3.contract.symbol, 150.0)
        assert get_wheel_state("SPY") == WheelState.CALL_SOLD

        # ── Phase 5: CALL_SOLD — 50 % profit → early close ────────────────
        pos_call = get_wheel_position("SPY")
        action4 = ws.get_next_action(
            pos_call, current_regime=2, option_chain=[],
            portfolio_nav=100_000, buying_power=60_000, current_pnl_pct=0.50,
        )
        assert action4.action == WheelActionType.CLOSE

        # ── Phase 6: Close → back to ASSIGNED (call closed, shares retained)
        update_on_close("SPY", closing_cost=75.0)
        assert get_wheel_state("SPY") == WheelState.ASSIGNED
        pos_final = get_wheel_position("SPY")
        assert pos_final.active_contract is None
        assert pos_final.premium_collected_total == 275.0   # 200 + 150 − 75

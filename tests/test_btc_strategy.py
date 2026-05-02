"""
tests/test_btc_strategy.py
---------------------------
Unit tests for core/btc_strategy.py and submit_crypto_order in order_executor.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.btc_strategy import BTCAction, BTCPosition, BTCStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cycle(
    composite_score: float = 0.0,
    failed_cycle: bool = False,
) -> object:
    sig = MagicMock()
    sig.composite_score = composite_score
    sig.failed_cycle    = failed_cycle
    return sig


def _position(
    shares_held: float = 1.0,
    avg_cost: float = 50_000.0,
    current_price: float = 50_000.0,
) -> BTCPosition:
    cost = avg_cost
    return BTCPosition(
        symbol             = "BTC/USD",
        shares_held        = shares_held,
        avg_cost           = cost,
        current_price      = current_price,
        unrealized_pnl     = (current_price - cost) * shares_held,
        unrealized_pnl_pct = (current_price - cost) / cost if cost else 0.0,
        entry_regime       = 3,
        entry_cycle_score  = 0.7,
    )


STRATEGY = BTCStrategy()


# ---------------------------------------------------------------------------
# TestBTCStrategyAllocations
# ---------------------------------------------------------------------------

class TestBTCStrategyAllocations:

    def test_crash_allocation_is_zero(self):
        assert BTCStrategy.REGIME_ALLOCATIONS[0] == 0.0

    def test_bear_allocation_is_0_25(self):
        assert BTCStrategy.REGIME_ALLOCATIONS[1] == pytest.approx(0.25)

    def test_neutral_allocation_is_0_50(self):
        assert BTCStrategy.REGIME_ALLOCATIONS[2] == pytest.approx(0.50)

    def test_bull_allocation_is_0_75(self):
        assert BTCStrategy.REGIME_ALLOCATIONS[3] == pytest.approx(0.75)

    def test_euphoria_allocation_is_0_40(self):
        assert BTCStrategy.REGIME_ALLOCATIONS[4] == pytest.approx(0.40)

    def test_euphoria_less_than_bull(self):
        assert BTCStrategy.REGIME_ALLOCATIONS[4] < BTCStrategy.REGIME_ALLOCATIONS[3]


# ---------------------------------------------------------------------------
# TestGetTargetAllocation
# ---------------------------------------------------------------------------

class TestGetTargetAllocation:

    def test_crash_always_zero_no_cycle(self):
        assert STRATEGY.get_target_allocation(0, _cycle(), False) == 0.0

    def test_crash_zero_even_with_high_cycle_score(self):
        assert STRATEGY.get_target_allocation(0, _cycle(composite_score=0.9), False) == 0.0

    def test_crash_zero_with_uncertainty(self):
        assert STRATEGY.get_target_allocation(0, _cycle(composite_score=0.9), True) == 0.0

    def test_bear_base_no_boost(self):
        result = STRATEGY.get_target_allocation(1, _cycle(composite_score=0.0), False)
        assert result == pytest.approx(0.25)

    def test_neutral_base_no_boost(self):
        result = STRATEGY.get_target_allocation(2, _cycle(composite_score=0.0), False)
        assert result == pytest.approx(0.50)

    def test_bull_base_no_boost(self):
        result = STRATEGY.get_target_allocation(3, _cycle(composite_score=0.0), False)
        assert result == pytest.approx(0.75)

    def test_euphoria_base_no_boost(self):
        result = STRATEGY.get_target_allocation(4, _cycle(composite_score=0.0), False)
        assert result == pytest.approx(0.40)

    # -- cycle boost (one tier up) --

    def test_cycle_boost_bear_to_neutral_allocation(self):
        result = STRATEGY.get_target_allocation(1, _cycle(composite_score=0.8), False)
        assert result == pytest.approx(0.50)   # REGIME_ALLOCATIONS[2]

    def test_cycle_boost_neutral_to_bull_allocation(self):
        result = STRATEGY.get_target_allocation(2, _cycle(composite_score=0.8), False)
        assert result == pytest.approx(0.75)   # REGIME_ALLOCATIONS[3]

    def test_cycle_boost_bull_to_euphoria_allocation(self):
        result = STRATEGY.get_target_allocation(3, _cycle(composite_score=0.8), False)
        assert result == pytest.approx(0.40)   # REGIME_ALLOCATIONS[4]

    def test_cycle_boost_euphoria_stays_at_euphoria(self):
        result = STRATEGY.get_target_allocation(4, _cycle(composite_score=0.8), False)
        assert result == pytest.approx(0.40)   # already at max tier

    # -- cycle reduce (one tier down) --

    def test_failed_cycle_neutral_reduces_to_bear(self):
        result = STRATEGY.get_target_allocation(2, _cycle(failed_cycle=True), False)
        assert result == pytest.approx(0.25)   # REGIME_ALLOCATIONS[1]

    def test_failed_cycle_bull_reduces_to_neutral(self):
        result = STRATEGY.get_target_allocation(3, _cycle(failed_cycle=True), False)
        assert result == pytest.approx(0.50)   # REGIME_ALLOCATIONS[2]

    def test_failed_cycle_bear_reduces_to_crash(self):
        result = STRATEGY.get_target_allocation(1, _cycle(failed_cycle=True), False)
        assert result == 0.0   # REGIME_ALLOCATIONS[0]

    def test_failed_cycle_euphoria_reduces_to_bull(self):
        result = STRATEGY.get_target_allocation(4, _cycle(failed_cycle=True), False)
        assert result == pytest.approx(0.75)   # REGIME_ALLOCATIONS[3]

    # -- uncertainty --

    def test_uncertainty_halves_neutral_allocation(self):
        result = STRATEGY.get_target_allocation(2, _cycle(), True)
        assert result == pytest.approx(0.25)   # 0.50 * 0.50

    def test_uncertainty_with_boost_halves_boosted(self):
        # neutral + boost → bull (0.75), uncertainty → 0.75 * 0.50 = 0.375
        result = STRATEGY.get_target_allocation(2, _cycle(composite_score=0.8), True)
        assert result == pytest.approx(0.375)

    def test_uncertainty_with_failed_cycle(self):
        # bull + failed → neutral (0.50), uncertainty → 0.50 * 0.50 = 0.25
        result = STRATEGY.get_target_allocation(3, _cycle(failed_cycle=True), True)
        assert result == pytest.approx(0.25)

    # -- BTC_MAX_ALLOCATION cap --

    def test_result_never_exceeds_max_allocation(self):
        from config.settings import BTC_MAX_ALLOCATION
        for regime in range(5):
            for cs in [0.0, 0.5, 0.9]:
                result = STRATEGY.get_target_allocation(
                    regime, _cycle(composite_score=cs), False
                )
                assert result <= BTC_MAX_ALLOCATION, (
                    f"regime={regime} cs={cs} result={result}"
                )

    # -- BTC_CYCLE_TIER_BOOST disabled --

    def test_no_boost_when_tier_boost_disabled(self):
        sig = _cycle(composite_score=0.9)
        with patch("core.btc_strategy.BTC_CYCLE_TIER_BOOST", False):
            result = BTCStrategy().get_target_allocation(2, sig, False)
        assert result == pytest.approx(0.50)   # unchanged base

    def test_no_reduce_when_tier_boost_disabled(self):
        sig = _cycle(failed_cycle=True)
        with patch("core.btc_strategy.BTC_CYCLE_TIER_BOOST", False):
            result = BTCStrategy().get_target_allocation(3, sig, False)
        assert result == pytest.approx(0.75)   # unchanged base

    # -- failed_cycle takes precedence over high score --

    def test_failed_cycle_takes_precedence_over_high_composite(self):
        sig = _cycle(composite_score=0.9, failed_cycle=True)
        result = STRATEGY.get_target_allocation(3, sig, False)
        assert result == pytest.approx(0.50)   # bull reduced to neutral, not boosted


# ---------------------------------------------------------------------------
# TestGetAction
# ---------------------------------------------------------------------------

class TestGetAction:

    NAV   = 100_000.0
    PRICE = 50_000.0

    def test_hold_when_no_position_and_target_zero(self):
        action = STRATEGY.get_action(
            current_position  = None,
            target_allocation = 0.0,
            portfolio_nav     = self.NAV,
            buying_power      = 50_000.0,
            current_price     = self.PRICE,
        )
        assert action.action == "HOLD"

    def test_hold_when_drift_within_threshold(self):
        # current = 0.50 * NAV, target = 0.52 → drift = 0.02 < 0.05
        pos = _position(shares_held=1.0, current_price=self.PRICE)
        action = STRATEGY.get_action(
            current_position  = pos,
            target_allocation = 0.52,
            portfolio_nav     = self.NAV,
            buying_power      = 50_000.0,
            current_price     = self.PRICE,
        )
        assert action.action == "HOLD"

    def test_hold_at_exact_threshold(self):
        # current = 0.45, target = 0.50 → drift = 0.05 which is NOT > 0.05
        pos = _position(shares_held=0.9, current_price=self.PRICE)
        # current_value = 0.9 * 50_000 = 45_000; alloc = 45_000/100_000 = 0.45
        action = STRATEGY.get_action(
            current_position  = pos,
            target_allocation = 0.50,
            portfolio_nav     = self.NAV,
            buying_power      = 50_000.0,
            current_price     = self.PRICE,
        )
        assert action.action == "HOLD"

    def test_buy_when_no_position_and_target_above_threshold(self):
        action = STRATEGY.get_action(
            current_position  = None,
            target_allocation = 0.50,
            portfolio_nav     = self.NAV,
            buying_power      = 60_000.0,
            current_price     = self.PRICE,
        )
        assert action.action == "BUY"

    def test_buy_size_equals_drift_times_nav(self):
        # current = 0, target = 0.50 → size = 0.50 * 100_000 = 50_000
        action = STRATEGY.get_action(
            current_position  = None,
            target_allocation = 0.50,
            portfolio_nav     = self.NAV,
            buying_power      = 60_000.0,
            current_price     = self.PRICE,
        )
        assert action.size_usd == pytest.approx(50_000.0)

    def test_buy_size_capped_at_buying_power(self):
        # drift * nav = 50_000 but buying_power = 20_000
        action = STRATEGY.get_action(
            current_position  = None,
            target_allocation = 0.50,
            portfolio_nav     = self.NAV,
            buying_power      = 20_000.0,
            current_price     = self.PRICE,
        )
        assert action.action == "BUY"
        assert action.size_usd == pytest.approx(20_000.0)

    def test_reduce_when_over_target(self):
        # current = 1.4 BTC × $50_000 = $70_000 = 70%; target = 50%
        pos = _position(shares_held=1.4, current_price=self.PRICE)
        action = STRATEGY.get_action(
            current_position  = pos,
            target_allocation = 0.50,
            portfolio_nav     = self.NAV,
            buying_power      = 0.0,
            current_price     = self.PRICE,
        )
        assert action.action == "REDUCE"

    def test_reduce_size_is_drift_times_nav(self):
        pos = _position(shares_held=1.4, current_price=self.PRICE)
        action = STRATEGY.get_action(
            current_position  = pos,
            target_allocation = 0.50,
            portfolio_nav     = self.NAV,
            buying_power      = 0.0,
            current_price     = self.PRICE,
        )
        # drift = 0.50 - 0.70 = -0.20; size = 0.20 * 100_000 = 20_000
        assert action.size_usd == pytest.approx(20_000.0)

    def test_exit_when_target_zero_and_position_held(self):
        pos = _position(shares_held=0.1, current_price=self.PRICE)
        action = STRATEGY.get_action(
            current_position  = pos,
            target_allocation = 0.0,
            portfolio_nav     = self.NAV,
            buying_power      = 0.0,
            current_price     = self.PRICE,
        )
        assert action.action == "EXIT"

    def test_exit_size_equals_full_position_value(self):
        pos = _position(shares_held=0.2, current_price=self.PRICE)
        action = STRATEGY.get_action(
            current_position  = pos,
            target_allocation = 0.0,
            portfolio_nav     = self.NAV,
            buying_power      = 0.0,
            current_price     = self.PRICE,
        )
        # size = 0.2 * 50_000 = 10_000
        assert action.size_usd == pytest.approx(10_000.0)

    def test_hold_when_zero_nav(self):
        action = STRATEGY.get_action(
            current_position  = None,
            target_allocation = 0.50,
            portfolio_nav     = 0.0,
            buying_power      = 0.0,
            current_price     = self.PRICE,
        )
        assert action.action == "HOLD"
        assert action.reason  == "zero_nav"

    def test_action_stores_regime_and_cycle_score(self):
        action = STRATEGY.get_action(
            current_position  = None,
            target_allocation = 0.0,
            portfolio_nav     = self.NAV,
            buying_power      = 0.0,
            current_price     = self.PRICE,
            regime            = 3,
            cycle_score       = 0.72,
        )
        assert action.regime      == 3
        assert action.cycle_score == pytest.approx(0.72)

    def test_buy_target_allocation_stored_in_action(self):
        action = STRATEGY.get_action(
            current_position  = None,
            target_allocation = 0.50,
            portfolio_nav     = self.NAV,
            buying_power      = 60_000.0,
            current_price     = self.PRICE,
        )
        assert action.target_allocation_pct == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# TestShouldRebalance
# ---------------------------------------------------------------------------

class TestShouldRebalance:

    def test_rebalance_true_when_drift_above_threshold(self):
        assert STRATEGY.should_rebalance(0.40, 0.50) is True

    def test_rebalance_false_when_drift_below_threshold(self):
        assert STRATEGY.should_rebalance(0.48, 0.50) is False

    def test_rebalance_false_at_exact_threshold(self):
        # |0.50 - 0.45| == 0.05, which is NOT > 0.05
        assert STRATEGY.should_rebalance(0.45, 0.50) is False

    def test_rebalance_true_when_over_allocated(self):
        assert STRATEGY.should_rebalance(0.70, 0.50) is True

    def test_rebalance_false_when_exactly_equal(self):
        assert STRATEGY.should_rebalance(0.50, 0.50) is False


# ---------------------------------------------------------------------------
# TestBTCPositionDataclass
# ---------------------------------------------------------------------------

class TestBTCPositionDataclass:

    def test_stores_all_required_fields(self):
        pos = BTCPosition(
            symbol             = "BTC/USD",
            shares_held        = 0.5,
            avg_cost           = 40_000.0,
            current_price      = 45_000.0,
            unrealized_pnl     = 2_500.0,
            unrealized_pnl_pct = 0.125,
            entry_regime       = 3,
            entry_cycle_score  = 0.7,
        )
        assert pos.symbol             == "BTC/USD"
        assert pos.shares_held        == pytest.approx(0.5)
        assert pos.unrealized_pnl     == pytest.approx(2_500.0)
        assert pos.entry_regime       == 3
        assert pos.entry_cycle_score  == pytest.approx(0.7)

    def test_timestamp_defaults_to_utc_now(self):
        pos = BTCPosition(
            symbol="BTC/USD", shares_held=0.1, avg_cost=40_000.0,
            current_price=40_000.0, unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0, entry_regime=2, entry_cycle_score=0.5,
        )
        assert pos.timestamp is not None
        assert pos.timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# TestBTCActionDataclass
# ---------------------------------------------------------------------------

class TestBTCActionDataclass:

    def test_all_fields_stored(self):
        action = BTCAction(
            action                = "BUY",
            target_allocation_pct = 0.50,
            size_usd              = 25_000.0,
            reason                = "allocation_drift_0.050",
            regime                = 2,
            cycle_score           = 0.6,
            confidence            = 0.8,
        )
        assert action.action                == "BUY"
        assert action.size_usd              == pytest.approx(25_000.0)
        assert action.regime                == 2
        assert action.confidence            == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# TestSubmitCryptoOrder
# ---------------------------------------------------------------------------

class TestSubmitCryptoOrder:

    def test_calls_submit_order_notional_on_client(self):
        from core.order_executor import submit_crypto_order

        mock_client = MagicMock()
        mock_client.get_orders.return_value      = []
        mock_client.submit_order_notional.return_value = MagicMock()

        submit_crypto_order("BTC/USD", "buy", 10_000.0, mock_client)

        mock_client.submit_order_notional.assert_called_once_with(
            symbol       = "BTC/USD",
            notional_usd = 10_000.0,
            side         = "buy",
        )

    def test_returns_none_for_zero_notional(self):
        from core.order_executor import submit_crypto_order
        assert submit_crypto_order("BTC/USD", "buy", 0.0) is None

    def test_returns_none_for_negative_notional(self):
        from core.order_executor import submit_crypto_order
        assert submit_crypto_order("BTC/USD", "buy", -100.0) is None

    def test_returns_none_when_open_order_exists(self):
        from core.order_executor import submit_crypto_order

        mock_order          = MagicMock()
        mock_order.symbol   = "BTC/USD"
        mock_client         = MagicMock()
        mock_client.get_orders.return_value = [mock_order]

        result = submit_crypto_order("BTC/USD", "buy", 10_000.0, mock_client)

        assert result is None
        mock_client.submit_order_notional.assert_not_called()

    def test_case_insensitive_dedup_check(self):
        from core.order_executor import submit_crypto_order

        mock_order          = MagicMock()
        mock_order.symbol   = "btc/usd"          # lower-case from broker
        mock_client         = MagicMock()
        mock_client.get_orders.return_value = [mock_order]

        result = submit_crypto_order("BTC/USD", "buy", 10_000.0, mock_client)

        assert result is None

    def test_sell_order_skipped_when_open_order_exists(self):
        from core.order_executor import submit_crypto_order

        mock_order          = MagicMock()
        mock_order.symbol   = "BTC/USD"
        mock_client         = MagicMock()
        mock_client.get_orders.return_value = [mock_order]

        result = submit_crypto_order("BTC/USD", "sell", 10_000.0, mock_client)

        assert result is None


# ---------------------------------------------------------------------------
# TestCurrentAllocationGuard
# ---------------------------------------------------------------------------

class TestCurrentAllocationGuard:
    """Verify the defensive current_allocation guard added to get_action()."""

    def _call(self, current_allocation: float, target_allocation: float) -> BTCAction:
        return STRATEGY.get_action(
            current_position   = None,
            target_allocation  = target_allocation,
            portfolio_nav      = 100_000.0,
            buying_power       = 100_000.0,
            current_price      = 50_000.0,
            regime             = 1,
            cycle_score        = 0.5,
            confidence         = 1.0,
            current_allocation = current_allocation,
        )

    def test_hold_when_allocation_equals_target(self):
        action = self._call(current_allocation=0.25, target_allocation=0.25)
        assert action.action == "HOLD"
        assert action.reason == "at_target_allocation"

    def test_hold_when_allocation_within_threshold_below_target(self):
        # 0.25 target, 0.21 current → diff = 0.04 < threshold 0.05 → HOLD
        action = self._call(current_allocation=0.21, target_allocation=0.25)
        assert action.action == "HOLD"
        assert action.reason == "at_target_allocation"

    def test_hold_when_allocation_exceeds_target(self):
        # Over-allocated: should still HOLD (not BUY more)
        action = self._call(current_allocation=0.30, target_allocation=0.25)
        assert action.action == "HOLD"
        assert action.reason == "at_target_allocation"

    def test_buy_when_allocation_well_below_target(self):
        # 0.25 target, 0.10 current → diff = 0.15 > threshold → BUY
        action = self._call(current_allocation=0.10, target_allocation=0.25)
        assert action.action == "BUY"

    def test_buy_when_no_position_and_allocation_zero(self):
        # current_allocation=0.0 with no position — should BUY toward 0.25
        action = self._call(current_allocation=0.0, target_allocation=0.25)
        assert action.action == "BUY"

    def test_exit_fires_before_guard_when_target_zero_and_position_held(self):
        # EXIT must still fire even when current_allocation >= target - threshold.
        # target=0.0, current_allocation=0.30 → guard: 0.30 >= 0.0 - 0.05 = -0.05 is True
        # BUT the EXIT check comes first and current_value > 0, so EXIT wins.
        action = STRATEGY.get_action(
            current_position   = _position(shares_held=1.0, current_price=50_000.0),
            target_allocation  = 0.0,
            portfolio_nav      = 100_000.0,
            buying_power       = 100_000.0,
            current_price      = 50_000.0,
            regime             = 0,
            cycle_score        = 0.0,
            confidence         = 1.0,
            current_allocation = 0.50,
        )
        assert action.action == "EXIT"
        assert action.reason == "target_allocation_zero"

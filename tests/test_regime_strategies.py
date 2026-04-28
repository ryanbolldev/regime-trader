"""
tests/test_regime_strategies.py
---------------------------------
Unit tests for core/regime_strategies.py.
"""

from __future__ import annotations

import pytest

from config.settings import (
    MAX_POSITIONS,
    PER_TRADE_RISK_CAP,
    REBALANCE_DRIFT_THRESHOLD,
    UNCERTAINTY_ALLOCATION_FACTOR,
)
from core.regime_strategies import (
    BullStrategy,
    CrashStrategy,
    EuphoriaStrategy,
    ExitSignal,
    Signal,
    StrategyBase,
    TargetPosition,
    get_signal,
    get_strategy,
    _PROFILES,
)

NAV = 100_000.0
TICKERS = ["SPY", "QQQ", "IWM", "GLD", "TLT"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _signal(regime: int, *, confidence: float = 0.8,
            nav: float = NAV, current_alloc: float = 0.5,
            uncertain: bool = False) -> Signal:
    return get_signal(regime, confidence, nav, current_alloc, uncertain)


# ---------------------------------------------------------------------------
# Signal dataclass: always fully populated
# ---------------------------------------------------------------------------

class TestSignalFullyPopulated:

    @pytest.mark.parametrize("regime", range(5))
    def test_all_fields_present(self, regime):
        sig = _signal(regime)
        assert isinstance(sig.regime, int)
        assert isinstance(sig.regime_name, str) and sig.regime_name
        assert isinstance(sig.allocation_pct, float)
        assert isinstance(sig.leverage, float)
        assert isinstance(sig.position_size_usd, float)
        assert isinstance(sig.confidence, float)
        assert isinstance(sig.is_uncertain, bool)
        assert isinstance(sig.needs_rebalance, bool)
        assert isinstance(sig.rationale, str) and len(sig.rationale) > 0
        assert sig.timestamp is not None

    @pytest.mark.parametrize("regime", range(5))
    def test_allocation_pct_in_unit_range(self, regime):
        sig = _signal(regime)
        assert 0.0 <= sig.allocation_pct <= 1.0

    @pytest.mark.parametrize("regime", range(5))
    def test_leverage_at_least_one(self, regime):
        sig = _signal(regime)
        assert sig.leverage >= 1.0

    @pytest.mark.parametrize("regime", range(5))
    def test_position_size_non_negative(self, regime):
        sig = _signal(regime)
        assert sig.position_size_usd >= 0.0

    def test_confidence_echoed_in_signal(self):
        for conf in (0.0, 0.5, 1.0):
            sig = _signal(2, confidence=conf)
            assert sig.confidence == conf

    def test_is_uncertain_echoed_in_signal(self):
        assert _signal(2, uncertain=True).is_uncertain is True
        assert _signal(2, uncertain=False).is_uncertain is False

    def test_invalid_regime_raises(self):
        with pytest.raises(ValueError):
            get_signal(99, 0.5, NAV, 0.5, False)

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            get_signal(2, 1.5, NAV, 0.5, False)

    def test_negative_nav_raises(self):
        with pytest.raises(ValueError):
            get_signal(2, 0.8, -1.0, 0.5, False)


# ---------------------------------------------------------------------------
# Uncertainty modifier
# ---------------------------------------------------------------------------

class TestUncertaintyModifier:

    @pytest.mark.parametrize("regime", range(5))
    def test_uncertain_reduces_allocation(self, regime):
        normal   = _signal(regime, uncertain=False)
        uncertain = _signal(regime, uncertain=True)
        assert pytest.approx(uncertain.allocation_pct, rel=1e-6) == (
            normal.allocation_pct * UNCERTAINTY_ALLOCATION_FACTOR
        )

    @pytest.mark.parametrize("regime", range(5))
    def test_uncertain_reduces_by_40_percent(self, regime):
        normal    = _signal(regime, uncertain=False)
        uncertain = _signal(regime, uncertain=True)
        ratio = uncertain.allocation_pct / normal.allocation_pct
        assert pytest.approx(ratio, abs=1e-9) == UNCERTAINTY_ALLOCATION_FACTOR

    def test_uncertainty_mentioned_in_rationale(self):
        sig = _signal(2, uncertain=True)
        assert "uncertainty" in sig.rationale.lower() or "uncertain" in sig.rationale.lower()

    def test_certainty_not_mentioned_when_not_uncertain(self):
        sig = _signal(2, uncertain=False)
        assert "uncertain" not in sig.rationale.lower()


# ---------------------------------------------------------------------------
# Regime-specific allocation values
# ---------------------------------------------------------------------------

class TestRegimeAllocations:

    def test_crash_allocation_is_10_pct(self):
        sig = _signal(0, uncertain=False)
        assert pytest.approx(sig.allocation_pct) == 0.10

    def test_bear_allocation_is_30_pct(self):
        sig = _signal(1, uncertain=False)
        assert pytest.approx(sig.allocation_pct) == 0.30

    def test_neutral_allocation_is_60_pct(self):
        sig = _signal(2, uncertain=False)
        assert pytest.approx(sig.allocation_pct) == 0.60

    def test_bull_allocation_is_90_pct(self):
        sig = _signal(3, uncertain=False)
        assert pytest.approx(sig.allocation_pct) == 0.90

    def test_euphoria_allocation_is_70_pct(self):
        sig = _signal(4, uncertain=False)
        assert pytest.approx(sig.allocation_pct) == 0.70

    def test_bull_leverage_is_1_1(self):
        sig = _signal(3, uncertain=False)
        assert pytest.approx(sig.leverage) == 1.1

    def test_crash_no_leverage(self):
        sig = _signal(0, uncertain=False)
        assert pytest.approx(sig.leverage) == 1.0

    def test_euphoria_never_exceeds_bull_allocation(self):
        bull_sig  = _signal(3, uncertain=False)
        euph_sig  = _signal(4, uncertain=False)
        assert euph_sig.allocation_pct < bull_sig.allocation_pct

    def test_euphoria_uncertain_never_exceeds_bull_uncertain(self):
        bull_sig  = _signal(3, uncertain=True)
        euph_sig  = _signal(4, uncertain=True)
        assert euph_sig.allocation_pct < bull_sig.allocation_pct


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

class TestPositionSizing:

    @pytest.mark.parametrize("regime", range(5))
    def test_position_size_never_exceeds_1pct_nav(self, regime):
        sig = _signal(regime, nav=NAV)
        assert sig.position_size_usd <= NAV * PER_TRADE_RISK_CAP + 1e-9

    def test_position_size_scales_with_nav(self):
        small = _signal(3, nav=10_000)
        large = _signal(3, nav=1_000_000)
        assert large.position_size_usd >= small.position_size_usd

    def test_zero_nav_gives_zero_position(self):
        sig = get_signal(2, 0.8, 0.0, 0.0, False)
        assert sig.position_size_usd == 0.0


# ---------------------------------------------------------------------------
# Rebalancing logic
# ---------------------------------------------------------------------------

class TestRebalancing:

    def test_no_rebalance_when_within_threshold(self):
        # neutral target = 60 %; current 63 % → drift 3 % < 5 %
        sig = get_signal(2, 0.8, NAV, 0.63, False)
        assert not sig.needs_rebalance

    def test_rebalance_triggered_when_over_threshold(self):
        # neutral target = 60 %; current 30 % → drift 30 % > 5 %
        sig = get_signal(2, 0.8, NAV, 0.30, False)
        assert sig.needs_rebalance

    def test_rebalance_just_below_threshold_is_not_triggered(self):
        target  = _PROFILES[2].allocation_pct          # 0.60
        current = target + REBALANCE_DRIFT_THRESHOLD - 0.001  # 0.649 — inside
        sig = get_signal(2, 0.8, NAV, current, False)
        assert not sig.needs_rebalance

    def test_rebalance_clearly_over_threshold_is_triggered(self):
        target  = _PROFILES[2].allocation_pct
        current = target + REBALANCE_DRIFT_THRESHOLD + 0.01  # 0.66 — clearly over
        sig = get_signal(2, 0.8, NAV, current, False)
        assert sig.needs_rebalance

    def test_rebalance_mentioned_in_rationale_when_needed(self):
        sig = get_signal(2, 0.8, NAV, 0.10, False)
        assert "rebalance" in sig.rationale.lower()

    def test_uncertainty_modifier_affects_rebalance_threshold(self):
        # With uncertainty, target drops to 60 % × 0.6 = 36 %.
        # current = 60 % — drift = 24 % → must rebalance.
        sig = get_signal(2, 0.8, NAV, 0.60, is_uncertain=True)
        assert sig.needs_rebalance


# ---------------------------------------------------------------------------
# Strategy classes: target positions
# ---------------------------------------------------------------------------

class TestStrategyTargetPositions:

    @pytest.mark.parametrize("regime", range(5))
    def test_returns_list_of_target_positions(self, regime):
        strategy = get_strategy(regime)
        positions = strategy.get_target_positions(NAV, {}, TICKERS)
        assert isinstance(positions, list)
        for p in positions:
            assert isinstance(p, TargetPosition)

    @pytest.mark.parametrize("regime", range(5))
    def test_result_count_does_not_exceed_max_positions(self, regime):
        strategy = get_strategy(regime)
        big_current = {t: 10_000.0 for t in TICKERS}
        positions = strategy.get_target_positions(NAV, big_current, TICKERS)
        assert len(positions) <= MAX_POSITIONS

    def test_crash_never_returns_long(self):
        strategy = CrashStrategy()
        current  = {"SPY": 50_000.0, "QQQ": 20_000.0}
        positions = strategy.get_target_positions(NAV, current, TICKERS)
        for p in positions:
            assert p.direction != "long", (
                f"CrashStrategy returned a long position for {p.ticker}"
            )

    def test_crash_flattens_existing_longs(self):
        strategy = CrashStrategy()
        current  = {"SPY": 30_000.0}
        positions = strategy.get_target_positions(NAV, current, TICKERS)
        assert any(p.ticker == "SPY" and p.direction == "flat" for p in positions)

    def test_crash_ignores_short_positions(self):
        strategy = CrashStrategy()
        current  = {"SPY": -10_000.0}   # short position — no action needed
        positions = strategy.get_target_positions(NAV, current, TICKERS)
        assert all(p.ticker != "SPY" for p in positions)

    def test_bull_returns_at_least_one_long(self):
        from core.regime_strategies import BullStrategy
        strategy  = BullStrategy()
        positions = strategy.get_target_positions(NAV, {}, TICKERS)
        assert any(p.direction == "long" for p in positions)

    def test_euphoria_returns_no_new_longs(self):
        strategy = EuphoriaStrategy()
        current  = {"SPY": 40_000.0}
        positions = strategy.get_target_positions(NAV, current, TICKERS)
        for p in positions:
            assert p.direction != "long", (
                f"EuphoriaStrategy opened a new long for {p.ticker}"
            )

    def test_euphoria_with_no_positions_returns_empty(self):
        strategy  = EuphoriaStrategy()
        positions = strategy.get_target_positions(NAV, {}, TICKERS)
        assert positions == []

    def test_target_weights_non_negative(self):
        for regime in range(5):
            strategy = get_strategy(regime)
            for p in strategy.get_target_positions(NAV, {}, TICKERS):
                assert p.target_weight >= 0.0

    def test_direction_values_valid(self):
        valid = {"long", "short", "flat"}
        for regime in range(5):
            strategy = get_strategy(regime)
            for p in strategy.get_target_positions(NAV, {"SPY": 1000.0}, TICKERS):
                assert p.direction in valid


# ---------------------------------------------------------------------------
# Strategy classes: exit signals
# ---------------------------------------------------------------------------

class TestStrategyExitSignals:

    @pytest.mark.parametrize("regime", range(5))
    def test_returns_list_of_exit_signals(self, regime):
        strategy = get_strategy(regime)
        signals  = strategy.get_exit_signals({"SPY": 1000.0}, {})
        assert isinstance(signals, list)
        for s in signals:
            assert isinstance(s, ExitSignal)

    @pytest.mark.parametrize("regime", range(5))
    def test_exit_signal_has_ticker_and_reason(self, regime):
        strategy = get_strategy(regime)
        signals  = strategy.get_exit_signals({"SPY": 1000.0}, {})
        for s in signals:
            assert s.ticker and isinstance(s.ticker, str)
            assert s.reason and isinstance(s.reason, str)
            assert s.exit_type and isinstance(s.exit_type, str)

    @pytest.mark.parametrize("regime", range(5))
    def test_exit_count_does_not_exceed_max_positions(self, regime):
        strategy     = get_strategy(regime)
        many_positions = {f"TICK{i}": 1000.0 for i in range(MAX_POSITIONS + 5)}
        signals      = strategy.get_exit_signals(many_positions, {})
        assert len(signals) <= MAX_POSITIONS

    @pytest.mark.parametrize("regime", range(5))
    def test_empty_positions_returns_empty_exits(self, regime):
        strategy = get_strategy(regime)
        assert strategy.get_exit_signals({}, {}) == []


# ---------------------------------------------------------------------------
# Strategy interface: duck-type / ABC check
# ---------------------------------------------------------------------------

class TestStrategyInterface:

    @pytest.mark.parametrize("regime", range(5))
    def test_strategy_has_get_target_positions(self, regime):
        strategy = get_strategy(regime)
        assert callable(getattr(strategy, "get_target_positions", None))

    @pytest.mark.parametrize("regime", range(5))
    def test_strategy_has_get_exit_signals(self, regime):
        strategy = get_strategy(regime)
        assert callable(getattr(strategy, "get_exit_signals", None))

    @pytest.mark.parametrize("regime", range(5))
    def test_strategy_is_subclass_of_base(self, regime):
        strategy = get_strategy(regime)
        assert isinstance(strategy, StrategyBase)

    def test_invalid_regime_raises_in_get_strategy(self):
        with pytest.raises(ValueError):
            get_strategy(99)


# ---------------------------------------------------------------------------
# Statelessness: no side effects
# ---------------------------------------------------------------------------

class TestStatelessness:

    def test_get_signal_is_pure(self):
        """Calling get_signal twice with identical args must return identical values."""
        kwargs = dict(regime=2, confidence=0.7, portfolio_nav=NAV,
                      current_allocation=0.5, is_uncertain=False)
        s1 = get_signal(**kwargs)
        s2 = get_signal(**kwargs)
        assert s1.allocation_pct     == s2.allocation_pct
        assert s1.leverage           == s2.leverage
        assert s1.position_size_usd  == s2.position_size_usd
        assert s1.needs_rebalance    == s2.needs_rebalance
        assert s1.rationale          == s2.rationale

    def test_strategy_does_not_mutate_inputs(self):
        strategy  = get_strategy(3)
        original  = {"SPY": 10_000.0}
        positions_copy = dict(original)
        strategy.get_target_positions(NAV, original, TICKERS)
        assert original == positions_copy

    def test_get_signal_does_not_modify_profiles(self):
        alloc_before = _PROFILES[3].allocation_pct
        _signal(3, uncertain=True)
        assert _PROFILES[3].allocation_pct == alloc_before

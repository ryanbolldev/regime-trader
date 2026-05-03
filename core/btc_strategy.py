"""
core/btc_strategy.py
---------------------
BTC spot trading strategy with cycle-adjusted regime allocations.

REGIME_ALLOCATIONS:
  crash    0.00 — sit out entirely
  bear     0.25 — light long exposure
  neutral  0.50 — moderate exposure
  bull     0.75 — full exposure
  euphoria 0.40 — trim back (mean-reversion risk)

Cycle overlay:
  When BTC_CYCLE_TIER_BOOST is True, a strong cycle signal
  (composite_score >= CYCLE_COMPOSITE_THRESHOLD) boosts the target by one
  tier, and a failed cycle reduces it by one tier.  Tiers follow regime
  indices (0–4), so a bull regime (3) boosted moves to the euphoria
  allocation (4 → 0.40), encoding the take-profit logic.  A failed cycle
  in bear regime (1) reduces to crash (0 → 0.00).

Uncertainty:
  When HMM is uncertain, target allocation is multiplied by 0.50.

Cap:
  Final target is capped at BTC_MAX_ALLOCATION (default 0.75).

Public interface:
  BTCPosition   — dataclass snapshot of the current BTC spot position
  BTCAction     — dataclass for the recommended trading action
  BTCStrategy   — stateless strategy class

  strategy = BTCStrategy()
  target   = strategy.get_target_allocation(regime, cycle_signal, is_uncertain)
  action   = strategy.get_action(current_position, target, nav, buying_power, price)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

from config.settings import (
    BTC_CYCLE_TIER_BOOST,
    BTC_MAX_ALLOCATION,
    BTC_REBALANCE_THRESHOLD,
    CYCLE_COMPOSITE_THRESHOLD,
)

if TYPE_CHECKING:
    from core.cycle_engine import CycleSignal

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class BTCPosition:
    """Snapshot of the current BTC spot position."""
    symbol:             str
    shares_held:        float    # fractional BTC units
    avg_cost:           float    # USD per BTC at entry
    current_price:      float
    unrealized_pnl:     float
    unrealized_pnl_pct: float
    entry_regime:       int
    entry_cycle_score:  float
    timestamp:          datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )


@dataclass
class BTCAction:
    """Recommended action to reach target_allocation_pct."""
    action:                str     # 'BUY' | 'SELL' | 'REDUCE' | 'HOLD' | 'EXIT'
    target_allocation_pct: float   # 0.0–1.0
    size_usd:              float
    reason:                str
    regime:                int
    cycle_score:           float
    confidence:            float


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class BTCStrategy:
    """Regime-aware BTC spot trading strategy with 60-day cycle overlay."""

    REGIME_ALLOCATIONS: dict[int, float] = {
        0: 0.00,   # crash    — sit out
        1: 0.05,   # bear     — light exposure
        2: 0.10,   # neutral  — moderate exposure
        3: 0.15,   # bull     — full exposure
        4: 0.08,   # euphoria — trim (mean-reversion risk)
    }

    def get_target_allocation(
        self,
        regime: int,
        cycle_signal: "CycleSignal",
        is_uncertain: bool,
    ) -> float:
        """Return target BTC allocation as fraction of NAV (0–BTC_MAX_ALLOCATION).

        Crash regime (base == 0.0) always returns 0.0 regardless of cycle signal.
        """
        base = self.REGIME_ALLOCATIONS.get(regime, 0.0)
        if base == 0.0:
            return 0.0

        adj = base

        if BTC_CYCLE_TIER_BOOST:
            if cycle_signal.failed_cycle:
                lower = max(regime - 1, 0)
                adj = self.REGIME_ALLOCATIONS[lower]
            elif cycle_signal.composite_score >= CYCLE_COMPOSITE_THRESHOLD:
                higher = min(regime + 1, 4)
                adj = self.REGIME_ALLOCATIONS[higher]

        if is_uncertain:
            adj *= 0.50

        return min(adj, BTC_MAX_ALLOCATION)

    def get_action(
        self,
        current_position: Optional[BTCPosition],
        target_allocation: float,
        portfolio_nav: float,
        buying_power: float,
        current_price: float,
        *,
        regime: int = -1,
        cycle_score: float = 0.0,
        confidence: float = 1.0,
        current_allocation: float = 0.0,
    ) -> BTCAction:
        """Return the recommended BTCAction to move toward target_allocation."""
        if portfolio_nav <= 0:
            return BTCAction(
                action="HOLD",
                target_allocation_pct=target_allocation,
                size_usd=0.0,
                reason="zero_nav",
                regime=regime,
                cycle_score=cycle_score,
                confidence=confidence,
            )

        if current_position is not None and current_position.shares_held > 0:
            current_value = current_position.shares_held * current_price
            current_alloc = current_value / portfolio_nav
        else:
            current_value = 0.0
            current_alloc = 0.0

        drift = target_allocation - current_alloc

        # EXIT: target is zero and we hold BTC
        if target_allocation == 0.0 and current_value > 0:
            return BTCAction(
                action="EXIT",
                target_allocation_pct=0.0,
                size_usd=current_value,
                reason="target_allocation_zero",
                regime=regime,
                cycle_score=cycle_score,
                confidence=confidence,
            )

        # Defensive guard: when position lookup failed (current_position is None),
        # use broker-reported allocation to prevent spurious buys/reduces.
        # Asymmetric: suppresses action only when at or above target;
        # under-target always falls through to BUY logic.
        if current_position is None:
            if current_allocation >= target_allocation:
                excess = current_allocation - target_allocation
                if excess <= BTC_REBALANCE_THRESHOLD:
                    return BTCAction(
                        action="HOLD",
                        target_allocation_pct=target_allocation,
                        size_usd=0.0,
                        reason="within_threshold",
                        regime=regime,
                        cycle_score=cycle_score,
                        confidence=confidence,
                    )
                return BTCAction(
                    action="REDUCE",
                    target_allocation_pct=target_allocation,
                    size_usd=excess * portfolio_nav,
                    reason="over_target",
                    regime=regime,
                    cycle_score=cycle_score,
                    confidence=confidence,
                )

        # HOLD: within rebalance threshold (position-based drift only)
        if current_position is not None and abs(drift) <= BTC_REBALANCE_THRESHOLD:
            return BTCAction(
                action="HOLD",
                target_allocation_pct=target_allocation,
                size_usd=0.0,
                reason="within_threshold",
                regime=regime,
                cycle_score=cycle_score,
                confidence=confidence,
            )

        # BUY: need more BTC
        if drift > 0:
            size_usd = min(drift * portfolio_nav, max(buying_power, 0.0))
            return BTCAction(
                action="BUY",
                target_allocation_pct=target_allocation,
                size_usd=size_usd,
                reason=f"allocation_drift_{drift:.3f}",
                regime=regime,
                cycle_score=cycle_score,
                confidence=confidence,
            )

        # REDUCE: need less BTC
        size_usd = abs(drift) * portfolio_nav
        return BTCAction(
            action="REDUCE",
            target_allocation_pct=target_allocation,
            size_usd=size_usd,
            reason=f"allocation_drift_{drift:.3f}",
            regime=regime,
            cycle_score=cycle_score,
            confidence=confidence,
        )

    def should_rebalance(
        self,
        current_allocation: float,
        target_allocation: float,
    ) -> bool:
        """Return True if |drift| exceeds BTC_REBALANCE_THRESHOLD."""
        return abs(target_allocation - current_allocation) > BTC_REBALANCE_THRESHOLD

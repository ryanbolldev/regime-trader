"""
core/regime_strategies.py
--------------------------
Per-regime allocation and strategy logic.

This is the primary customization surface.  Each confirmed HMM regime maps to
a distinct strategy profile that governs:
  - Target gross exposure (% of portfolio NAV)
  - Leverage multiplier
  - Long / short / flat bias
  - Entry signal type (momentum, mean-reversion, defensive)
  - Exit signal type (trailing stop, time-based, regime change)
  - Maximum concurrent new positions per regime

Default regime profiles:
  crash    : 10 % invested, 1.0x (no leverage), cash-heavy defensive
  bear     : 30 % invested, 1.0x (no leverage), reduced exposure
  neutral  : 60 % invested, 1.0x, balanced mean-reversion posture
  bull     : 90 % invested, 1.1x, trend-following posture
  euphoria : 70 % invested, 1.0x, reduced from bull (mean-reversion risk)

Uncertainty modifier:
  If HMMEngine.is_uncertain() is True, effective allocation is multiplied by
  UNCERTAINTY_ALLOCATION_FACTOR (0.60), reducing target exposure by 40 %.

Rebalancing:
  Only triggered when |current_allocation - target_allocation| > 5 %.
  Every rebalance decision is logged with before/after values.

Public interface:
  get_signal(regime, confidence, portfolio_nav, current_allocation,
             is_uncertain) -> Signal
  get_strategy(regime) -> StrategyBase
  StrategyBase.get_target_positions(nav, current_positions, tickers)
      -> list[TargetPosition]
  StrategyBase.get_exit_signals(open_positions, market_data)
      -> list[ExitSignal]

The risk_manager retains absolute veto power; signals produced here are
advisory until approved by RiskManager.approve().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from config.settings import (
    MAX_POSITIONS,
    PER_TRADE_RISK_CAP,
    REBALANCE_DRIFT_THRESHOLD,
    UNCERTAINTY_ALLOCATION_FACTOR,
)

if TYPE_CHECKING:
    from core.wheel_strategy import WheelAction, WheelPosition, WheelStrategy

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeProfile:
    name: str
    allocation_pct: float   # base target gross exposure as fraction of NAV
    leverage: float         # position leverage multiplier (1.0 = no borrowing)
    allow_long: bool
    allow_short: bool
    max_new_positions: int


@dataclass
class Signal:
    """Output of get_signal().  Always fully populated; never has None fields."""
    regime: int
    regime_name: str
    allocation_pct: float       # effective allocation after all modifiers
    leverage: float
    position_size_usd: float    # max per-trade dollar size
    confidence: float
    is_uncertain: bool
    needs_rebalance: bool
    rationale: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    wheel_action: Optional[WheelAction] = None


@dataclass
class TargetPosition:
    ticker: str
    direction: str          # 'long' | 'short' | 'flat'
    target_weight: float    # fraction of NAV
    signal_type: str        # 'momentum' | 'mean_reversion' | 'defensive' | 'exit'


@dataclass
class ExitSignal:
    ticker: str
    reason: str
    exit_type: str          # 'trailing_stop' | 'regime_change' | 'time_based'


# ---------------------------------------------------------------------------
# Regime profiles
# ---------------------------------------------------------------------------

_PROFILES: dict[int, RegimeProfile] = {
    0: RegimeProfile("crash",    allocation_pct=0.10, leverage=1.0,
                     allow_long=False, allow_short=True,  max_new_positions=1),
    1: RegimeProfile("bear",     allocation_pct=0.30, leverage=1.0,
                     allow_long=True,  allow_short=False, max_new_positions=3),
    2: RegimeProfile("neutral",  allocation_pct=0.60, leverage=1.0,
                     allow_long=True,  allow_short=False, max_new_positions=5),
    3: RegimeProfile("bull",     allocation_pct=0.90, leverage=1.1,
                     allow_long=True,  allow_short=False, max_new_positions=5),
    4: RegimeProfile("euphoria", allocation_pct=0.70, leverage=1.0,
                     allow_long=False, allow_short=False, max_new_positions=0),
}


# ---------------------------------------------------------------------------
# Stateless orchestrator
# ---------------------------------------------------------------------------

def get_signal(
    regime: int,
    confidence: float,
    portfolio_nav: float,
    current_allocation: float,
    is_uncertain: bool,
    *,
    wheel_strategy: Optional[WheelStrategy] = None,
    wheel_position: Optional[WheelPosition] = None,
    option_chain: Optional[list] = None,
    buying_power: float = 0.0,
    current_pnl_pct: float = 0.0,
) -> Signal:
    """Return a Signal for the current regime state.

    Parameters
    ----------
    regime             : int 0–4 from HMMEngine
    confidence         : float 0–1, caller's confidence estimate
    portfolio_nav      : total portfolio NAV in dollars
    current_allocation : current gross exposure as fraction of NAV (0.0–1.0)
    is_uncertain       : True if HMMEngine.is_uncertain() is active

    Returns
    -------
    Signal — always fully populated, never raises on valid inputs.
    """
    if regime not in _PROFILES:
        raise ValueError(f"Unknown regime: {regime}. Expected 0–4.")
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence must be 0–1, got {confidence}.")
    if portfolio_nav < 0:
        raise ValueError("portfolio_nav must be non-negative.")

    profile = _PROFILES[regime]

    # Uncertainty modifier: reduce target allocation by 40 %
    uncertainty_factor = UNCERTAINTY_ALLOCATION_FACTOR if is_uncertain else 1.0
    effective_alloc = profile.allocation_pct * uncertainty_factor

    # Position size: per-position allocation bounded by 1 % NAV risk cap
    n_positions = max(profile.max_new_positions, 1)
    equity_budget = portfolio_nav * effective_alloc * profile.leverage
    per_position_usd = equity_budget / n_positions
    risk_cap_usd = portfolio_nav * PER_TRADE_RISK_CAP
    position_size_usd = max(min(per_position_usd, risk_cap_usd), 0.0)

    # Rebalance decision
    drift = abs(current_allocation - effective_alloc)
    needs_rebalance = drift > REBALANCE_DRIFT_THRESHOLD

    if needs_rebalance:
        log.info(
            "Rebalance: regime=%s  before=%.1f%%  after=%.1f%%  drift=%.1f%%",
            profile.name,
            current_allocation * 100,
            effective_alloc * 100,
            drift * 100,
        )

    rationale = _build_rationale(
        profile, effective_alloc, is_uncertain, uncertainty_factor,
        needs_rebalance, current_allocation,
    )

    wheel_action = None
    if wheel_strategy is not None:
        wheel_action = wheel_strategy.get_next_action(
            position=wheel_position,
            current_regime=regime,
            option_chain=option_chain or [],
            portfolio_nav=portfolio_nav,
            buying_power=buying_power,
            is_uncertain=is_uncertain,
            current_pnl_pct=current_pnl_pct,
        )

    return Signal(
        regime=regime,
        regime_name=profile.name,
        allocation_pct=effective_alloc,
        leverage=profile.leverage,
        position_size_usd=position_size_usd,
        confidence=confidence,
        is_uncertain=is_uncertain,
        needs_rebalance=needs_rebalance,
        rationale=rationale,
        wheel_action=wheel_action,
    )


def _build_rationale(
    profile: RegimeProfile,
    effective_alloc: float,
    is_uncertain: bool,
    uncertainty_factor: float,
    needs_rebalance: bool,
    current_allocation: float,
) -> str:
    parts = [
        f"Regime: {profile.name} "
        f"({profile.allocation_pct:.0%} base, {profile.leverage:.1f}x leverage)."
    ]
    if is_uncertain:
        reduction_pct = round((1 - uncertainty_factor) * 100)
        parts.append(
            f"Uncertainty active: allocation reduced {reduction_pct}% "
            f"→ effective {effective_alloc:.0%}."
        )
    if needs_rebalance:
        parts.append(
            f"Rebalance required: "
            f"current {current_allocation:.0%} → target {effective_alloc:.0%}."
        )
    else:
        parts.append(
            f"No rebalance needed "
            f"(current {current_allocation:.0%} within "
            f"{REBALANCE_DRIFT_THRESHOLD:.0%} of target {effective_alloc:.0%})."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Per-regime strategy classes
# ---------------------------------------------------------------------------

class StrategyBase:
    """Base strategy; subclasses override get_target_positions / get_exit_signals."""

    regime_label: str = "base"
    allow_long:   bool = True
    allow_short:  bool = False

    def get_target_positions(
        self,
        nav: float,
        current_positions: dict[str, float],
        tickers: list[str],
    ) -> list[TargetPosition]:
        raise NotImplementedError

    def get_exit_signals(
        self,
        open_positions: dict[str, float],
        market_data: dict[str, dict],
    ) -> list[ExitSignal]:
        raise NotImplementedError


class CrashStrategy(StrategyBase):
    """10 % allocation, defensive cash posture. No new longs; flatten existing longs."""

    regime_label = "crash"
    allow_long   = False
    allow_short  = True

    def get_target_positions(
        self, _nav: float, current_positions: dict[str, float], _tickers: list[str]
    ) -> list[TargetPosition]:
        positions: list[TargetPosition] = []
        for ticker, value in current_positions.items():
            if value > 0:
                positions.append(TargetPosition(
                    ticker=ticker, direction="flat",
                    target_weight=0.0, signal_type="defensive",
                ))
        return positions[:MAX_POSITIONS]

    def get_exit_signals(
        self, open_positions: dict[str, float], _market_data: dict[str, dict]
    ) -> list[ExitSignal]:
        return [
            ExitSignal(ticker=t, reason="crash_regime_active",
                       exit_type="regime_change")
            for t in open_positions
        ][:MAX_POSITIONS]


class BearStrategy(StrategyBase):
    """30 % allocation, reduced long exposure, defensive sectors."""

    regime_label = "bear"
    allow_long   = True
    allow_short  = False

    def get_target_positions(
        self, _nav: float, _current_positions: dict[str, float], tickers: list[str]
    ) -> list[TargetPosition]:
        profile = _PROFILES[1]
        per_weight = profile.allocation_pct / profile.max_new_positions
        return [
            TargetPosition(ticker=t, direction="long",
                           target_weight=per_weight, signal_type="defensive")
            for t in tickers[:profile.max_new_positions]
        ]

    def get_exit_signals(
        self, open_positions: dict[str, float], _market_data: dict[str, dict]
    ) -> list[ExitSignal]:
        return [
            ExitSignal(ticker=t, reason="bear_tighten_stops",
                       exit_type="trailing_stop")
            for t in open_positions
        ][:MAX_POSITIONS]


class NeutralStrategy(StrategyBase):
    """60 % allocation, balanced mean-reversion posture."""

    regime_label = "neutral"
    allow_long   = True
    allow_short  = False

    def get_target_positions(
        self, _nav: float, _current_positions: dict[str, float], tickers: list[str]
    ) -> list[TargetPosition]:
        profile = _PROFILES[2]
        per_weight = profile.allocation_pct / profile.max_new_positions
        return [
            TargetPosition(ticker=t, direction="long",
                           target_weight=per_weight, signal_type="mean_reversion")
            for t in tickers[:profile.max_new_positions]
        ]

    def get_exit_signals(
        self, open_positions: dict[str, float], _market_data: dict[str, dict]
    ) -> list[ExitSignal]:
        return [
            ExitSignal(ticker=t, reason="neutral_time_exit",
                       exit_type="time_based")
            for t in open_positions
        ][:MAX_POSITIONS]


class BullStrategy(StrategyBase):
    """90 % allocation, 1.1x leverage, momentum entries."""

    regime_label = "bull"
    allow_long   = True
    allow_short  = False

    def get_target_positions(
        self, _nav: float, _current_positions: dict[str, float], tickers: list[str]
    ) -> list[TargetPosition]:
        profile = _PROFILES[3]
        per_weight = profile.allocation_pct / profile.max_new_positions
        return [
            TargetPosition(ticker=t, direction="long",
                           target_weight=per_weight, signal_type="momentum")
            for t in tickers[:profile.max_new_positions]
        ]

    def get_exit_signals(
        self, open_positions: dict[str, float], _market_data: dict[str, dict]
    ) -> list[ExitSignal]:
        return [
            ExitSignal(ticker=t, reason="bull_trailing_stop",
                       exit_type="trailing_stop")
            for t in open_positions
        ][:MAX_POSITIONS]


class EuphoriaStrategy(StrategyBase):
    """70 % allocation, no new longs, profit-taking / mean-reversion exits."""

    regime_label = "euphoria"
    allow_long   = False
    allow_short  = False

    def get_target_positions(
        self, _nav: float, current_positions: dict[str, float], _tickers: list[str]
    ) -> list[TargetPosition]:
        # Flatten existing long positions — no new entries
        positions: list[TargetPosition] = []
        for ticker, value in current_positions.items():
            if value > 0:
                positions.append(TargetPosition(
                    ticker=ticker, direction="flat",
                    target_weight=0.0, signal_type="mean_reversion",
                ))
        return positions[:MAX_POSITIONS]

    def get_exit_signals(
        self, open_positions: dict[str, float], _market_data: dict[str, dict]
    ) -> list[ExitSignal]:
        return [
            ExitSignal(ticker=t, reason="euphoria_profit_taking",
                       exit_type="trailing_stop")
            for t in open_positions
        ][:MAX_POSITIONS]


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

_STRATEGY_MAP: dict[int, StrategyBase] = {
    0: CrashStrategy(),
    1: BearStrategy(),
    2: NeutralStrategy(),
    3: BullStrategy(),
    4: EuphoriaStrategy(),
}


def get_strategy(regime: int) -> StrategyBase:
    """Return the strategy singleton for a given regime label (0–4)."""
    if regime not in _STRATEGY_MAP:
        raise ValueError(f"Unknown regime: {regime}. Expected 0–4.")
    return _STRATEGY_MAP[regime]

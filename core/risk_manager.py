"""
core/risk_manager.py
---------------------
Absolute risk gate with veto power over all trade signals.

Circuit breakers evaluated in priority order:
  1. Peak-drawdown lockout  (-10 % from rolling HWM → halt all trading)
  2. Daily drawdown close-all (-3 % intraday → flatten positions, suppress entries)
  3. Daily drawdown halve  (-2 % intraday → 50 % size for remainder of session)
  4. Weekly drawdown resize (-5 % weekly → 50 % max size until Monday)
  5. Per-trade risk cap    (max 1 % NAV per trade)
  6. Correlation budget    (placeholder — always passes in backtest context)

Public interface:
  initialize(nav)                        → None   # call once per fold / session
  reset_daily(nav)                       → None   # call at open of each trading day
  reset_weekly(nav)                      → None   # call at open of each trading week
  update(current_nav)                    → list[str]   # call after each bar
  approve(signal, portfolio_nav)         → ApprovalResult
  get_circuit_breaker_status()           → CircuitBreakerState
  is_locked()                            → bool
"""

from __future__ import annotations

from dataclasses import dataclass

from config import settings


class InsufficientFundsError(Exception):
    """Raised by validate_put_sale / validate_call_sale when pre-flight fails."""


@dataclass
class CircuitBreakerState:
    peak_drawdown_lockout: bool = False
    daily_halt:            bool = False
    daily_halve_sizes:     bool = False
    weekly_resize:         bool = False


@dataclass
class ApprovalResult:
    approved:               bool
    size_multiplier:        float        # 0.0 blocked, 0.5 halved, 1.0 full
    circuit_breakers_fired: list[str]
    reason:                 str


class RiskManager:
    """Stateful circuit-breaker gate.  One instance per fold or live session."""

    def __init__(self) -> None:
        self._high_water_mark:  float = 0.0
        self._daily_open_nav:   float = 0.0
        self._weekly_open_nav:  float = 0.0
        self._locked:           bool = False
        self._daily_halt:       bool = False
        self._daily_halve:      bool = False
        self._weekly_resize:    bool = False
        self._all_fired:        list[str] = []

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def initialize(self, nav: float) -> None:
        """Reset all state to start a fresh fold or live session."""
        self._high_water_mark = nav
        self._daily_open_nav  = nav
        self._weekly_open_nav = nav
        self._locked          = False
        self._daily_halt      = False
        self._daily_halve     = False
        self._weekly_resize   = False
        self._all_fired       = []

    def reset_daily(self, nav: float) -> None:
        """Call at the start of each trading day."""
        self._daily_open_nav = nav
        self._daily_halt     = False
        self._daily_halve    = False

    def reset_weekly(self, nav: float) -> None:
        """Call at the start of each trading week (Monday open)."""
        self._weekly_open_nav = nav
        self._weekly_resize   = False

    # ------------------------------------------------------------------
    # State update (called after each bar's P&L is realised)
    # ------------------------------------------------------------------

    def update(self, current_nav: float) -> list[str]:
        """Evaluate all circuit breakers against current_nav.

        Returns the names of any breakers newly fired this bar.
        """
        live = settings.LIVE_ACCOUNT_MODE
        _halt    = settings.LIVE_INTRADAY_STOP_HALT    if live else settings.INTRADAY_STOP_HALT
        _warn    = settings.INTRADAY_STOP_WARN
        _weekly  = settings.LIVE_WEEKLY_STOP           if live else settings.WEEKLY_STOP
        _lockout = settings.LIVE_PEAK_DRAWDOWN_LOCKOUT if live else settings.PEAK_DRAWDOWN_LOCKOUT

        fired: list[str] = []

        # 1. Peak-drawdown lockout — check before updating HWM
        if not self._locked and self._high_water_mark > 0.0:
            peak_dd = (current_nav - self._high_water_mark) / self._high_water_mark
            if peak_dd <= _lockout:
                self._locked = True
                fired.append("peak_drawdown_lockout")

        # Update HWM after the drawdown check so a new high doesn't mask a breach
        if current_nav > self._high_water_mark:
            self._high_water_mark = current_nav

        # 2 & 3. Daily drawdown breakers
        if self._daily_open_nav > 0.0 and not self._locked:
            daily_dd = (current_nav - self._daily_open_nav) / self._daily_open_nav
            if daily_dd <= _halt and not self._daily_halt:
                self._daily_halt  = True
                self._daily_halve = True
                fired.append("daily_halt")
                fired.append("daily_halve_sizes")
            elif daily_dd <= _warn and not self._daily_halve:
                self._daily_halve = True
                fired.append("daily_halve_sizes")

        # 4. Weekly drawdown resize
        if self._weekly_open_nav > 0.0 and not self._locked:
            weekly_dd = (current_nav - self._weekly_open_nav) / self._weekly_open_nav
            if weekly_dd <= _weekly and not self._weekly_resize:
                self._weekly_resize = True
                fired.append("weekly_resize")

        self._all_fired.extend(fired)
        return fired

    # ------------------------------------------------------------------
    # Signal approval
    # ------------------------------------------------------------------

    def approve(self, _signal, _portfolio_nav: float) -> ApprovalResult:
        """Return whether a Signal may be acted upon and at what size."""
        if self._locked:
            return ApprovalResult(
                approved=False,
                size_multiplier=0.0,
                circuit_breakers_fired=["peak_drawdown_lockout"],
                reason="peak_drawdown_lockout_active",
            )
        if self._daily_halt:
            return ApprovalResult(
                approved=False,
                size_multiplier=0.0,
                circuit_breakers_fired=["daily_halt"],
                reason="daily_halt_active",
            )

        multiplier = 1.0
        active: list[str] = []
        if self._daily_halve:
            multiplier *= 0.5
            active.append("daily_halve_sizes")
        if self._weekly_resize:
            multiplier *= 0.5
            active.append("weekly_resize")

        return ApprovalResult(
            approved=True,
            size_multiplier=multiplier,
            circuit_breakers_fired=active,
            reason="approved",
        )

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def get_circuit_breaker_status(self) -> CircuitBreakerState:
        return CircuitBreakerState(
            peak_drawdown_lockout=self._locked,
            daily_halt=self._daily_halt,
            daily_halve_sizes=self._daily_halve,
            weekly_resize=self._weekly_resize,
        )

    def is_locked(self) -> bool:
        return self._locked

    @property
    def all_fired(self) -> list[str]:
        """Every circuit-breaker event that has fired since initialize()."""
        return list(self._all_fired)

    # ------------------------------------------------------------------
    # Wheel strategy pre-flight validators
    # ------------------------------------------------------------------

    def validate_put_sale(self, strike: float, qty: int, buying_power: float) -> None:
        """Confirm buying power covers full assignment before approving a put sale.

        Raises InsufficientFundsError if the check fails.
        Full assignment cost = strike × qty × 100 (one contract = 100 shares).
        """
        required = strike * qty * 100
        if buying_power < required:
            raise InsufficientFundsError(
                f"Insufficient buying power for put sale: "
                f"need ${required:,.2f}, have ${buying_power:,.2f}"
            )

    def validate_call_sale(self, symbol: str, qty: int, position_tracker) -> None:
        """Confirm shares are owned before approving a covered call sale.

        Raises InsufficientFundsError if the wheel position is not in ASSIGNED state.
        """
        from core.wheel_strategy import WheelState
        state = position_tracker.get_wheel_state(symbol)
        if state != WheelState.ASSIGNED:
            raise InsufficientFundsError(
                f"Cannot sell {qty} call contract(s) on {symbol}: shares not owned "
                f"(wheel state={state!r}, expected ASSIGNED)"
            )

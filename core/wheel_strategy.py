"""
core/wheel_strategy.py
-----------------------
Wheel options strategy layered on top of the regime-based allocation system.

The wheel cycle:
  CASH → sell cash-secured put → PUT_SOLD
  PUT_SOLD → assigned on put   → ASSIGNED
  ASSIGNED → sell covered call → CALL_SOLD
  CALL_SOLD → called away      → CASH (cycle repeats)

Regime gates:
  Put entry  : bull (3) or neutral (2) only.
  Call entry : neutral (2) or euphoria (4) only.
  Uncertain  : no new entries; existing positions hold (WAIT).

Early-close triggers (applies to both put and call legs):
  1. Regime deteriorates to bear or crash.
  2. Fifty percent of max profit captured.
  3. Loss exceeds 200 % of premium received.
  4. Fewer than WHEEL_GAMMA_RISK_DTE days to expiration with a loss.
"""

from __future__ import annotations

import datetime as _dt
import enum
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config.settings import (
    WHEEL_CALL_DELTA_TARGET,
    WHEEL_EARLY_CLOSE_LOSS_PCT,
    WHEEL_EARLY_CLOSE_PROFIT_PCT,
    WHEEL_GAMMA_RISK_DTE,
    WHEEL_MAX_DTE,
    WHEEL_MIN_DTE,
    WHEEL_PUT_DELTA_TARGET,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class WheelState(enum.Enum):
    CASH      = "CASH"
    PUT_SOLD  = "PUT_SOLD"
    ASSIGNED  = "ASSIGNED"
    CALL_SOLD = "CALL_SOLD"


class WheelActionType(enum.Enum):
    SELL_PUT  = "SELL_PUT"
    SELL_CALL = "SELL_CALL"
    CLOSE     = "CLOSE"
    WAIT      = "WAIT"
    SIT_OUT   = "SIT_OUT"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class WheelPosition:
    """Tracks the current state of one ticker inside the wheel cycle."""
    symbol:                  str
    phase:                   WheelState
    shares_owned:            int
    cost_basis:              float          # per-share average cost
    active_contract:         Optional[str]  # OCC symbol string, or None
    premium_collected_total: float
    entry_regime:            int
    timestamp:               datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )


@dataclass(frozen=True)
class WheelAction:
    """Recommended action returned by WheelStrategy.get_next_action()."""
    action:   WheelActionType
    contract: Optional[object]   # OptionContract at runtime, None otherwise
    reason:   str
    regime:   int


# ---------------------------------------------------------------------------
# OCC / DTE helpers (module-level so tests can patch _today)
# ---------------------------------------------------------------------------

_OCC_DATE_RE = re.compile(r"^[A-Z0-9]+(\d{6})[CP]\d{8}$")


def _today() -> _dt.date:
    """Return today's date.  Isolated here so tests can patch it."""
    return _dt.date.today()


def _dte_from_occ(occ: str, today: _dt.date) -> Optional[int]:
    """Return days to expiration parsed from an OCC symbol, or None."""
    m = _OCC_DATE_RE.match(occ)
    if not m:
        return None
    yymmdd = m.group(1)
    exp = _dt.date(2000 + int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:]))
    return (exp - today).days


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class WheelStrategy:
    """Stateless adviser: takes position context, returns a WheelAction."""

    _CRASH    = 0
    _BEAR     = 1
    _NEUTRAL  = 2
    _BULL     = 3
    _EUPHORIA = 4

    _PUT_REGIMES   = frozenset({_BULL, _NEUTRAL})
    _CALL_REGIMES  = frozenset({_NEUTRAL, _EUPHORIA})
    _CLOSE_REGIMES = frozenset({_CRASH, _BEAR})

    # ------------------------------------------------------------------
    # Entry finders
    # ------------------------------------------------------------------

    def get_put_to_sell(
        self,
        symbol: str,
        option_chain: list,
        portfolio_nav: float,
        buying_power: float,
        regime: int,
    ) -> Optional[object]:
        """Return the best cash-secured put or None.

        Selects the put in the 30–45 DTE window whose delta is closest to
        WHEEL_PUT_DELTA_TARGET, provided buying power covers full assignment.
        """
        if regime not in self._PUT_REGIMES:
            return None

        today = _today()
        candidates = []
        for c in option_chain:
            if c.option_type != "put" or c.underlying != symbol or c.delta is None:
                continue
            dte = (_dt.date.fromisoformat(c.expiration) - today).days
            if WHEEL_MIN_DTE <= dte <= WHEEL_MAX_DTE:
                candidates.append(c)

        if not candidates:
            return None

        best = min(candidates, key=lambda c: abs(c.delta - WHEEL_PUT_DELTA_TARGET))
        assignment_cost = best.strike * 100
        # Buying power must cover full assignment; also reject if it exceeds total NAV
        # (a sanity ceiling — never risk more than the whole portfolio on one put).
        if buying_power < assignment_cost or assignment_cost > portfolio_nav:
            return None
        return best

    def get_call_to_sell(
        self,
        symbol: str,
        option_chain: list,
        shares_owned: int,
        cost_basis: float,
        regime: int,
    ) -> Optional[object]:
        """Return the best covered call or None.

        Requires at least 100 shares owned (one contract) and a strike above
        cost_basis so that assignment guarantees a profit.
        """
        if regime not in self._CALL_REGIMES:
            return None
        if shares_owned < 100:
            return None

        today = _today()
        candidates = []
        for c in option_chain:
            if c.option_type != "call" or c.underlying != symbol or c.delta is None:
                continue
            if c.strike <= cost_basis:
                continue
            dte = (_dt.date.fromisoformat(c.expiration) - today).days
            if WHEEL_MIN_DTE <= dte <= WHEEL_MAX_DTE:
                candidates.append(c)

        if not candidates:
            return None

        return min(candidates, key=lambda c: abs(c.delta - WHEEL_CALL_DELTA_TARGET))

    # ------------------------------------------------------------------
    # Early-exit logic
    # ------------------------------------------------------------------

    def should_close_early(
        self,
        position: WheelPosition,
        current_regime: int,
        current_pnl_pct: float,
    ) -> bool:
        """True when the active option leg should be closed before expiry.

        current_pnl_pct is the realised fraction of premium received:
          +0.50 means 50 % of max profit captured.
          -2.00 means loss equals 200 % of premium received.
        """
        if current_regime in self._CLOSE_REGIMES:
            return True
        if current_pnl_pct >= WHEEL_EARLY_CLOSE_PROFIT_PCT:
            return True
        if current_pnl_pct <= -WHEEL_EARLY_CLOSE_LOSS_PCT:
            return True
        if position.active_contract and current_pnl_pct < 0:
            dte = _dte_from_occ(position.active_contract, _today())
            if dte is not None and dte < WHEEL_GAMMA_RISK_DTE:
                return True
        return False

    # ------------------------------------------------------------------
    # Cycle orchestrator
    # ------------------------------------------------------------------

    def get_next_action(
        self,
        position: Optional[WheelPosition],
        current_regime: int,
        option_chain: list,
        portfolio_nav: float,
        buying_power: float,
        is_uncertain: bool = False,
        current_pnl_pct: float = 0.0,
    ) -> WheelAction:
        """Return the recommended WheelAction for the current bar."""
        if is_uncertain:
            return WheelAction(
                action=WheelActionType.WAIT,
                contract=None,
                reason="uncertain_regime",
                regime=current_regime,
            )

        if position is None:
            return WheelAction(
                action=WheelActionType.SIT_OUT,
                contract=None,
                reason="no_position_context",
                regime=current_regime,
            )

        phase = position.phase

        # CASH — find a put to sell
        if phase == WheelState.CASH:
            contract = self.get_put_to_sell(
                symbol=position.symbol,
                option_chain=option_chain,
                portfolio_nav=portfolio_nav,
                buying_power=buying_power,
                regime=current_regime,
            )
            if contract:
                return WheelAction(
                    action=WheelActionType.SELL_PUT,
                    contract=contract,
                    reason=f"put_delta_{contract.delta:.3f}",
                    regime=current_regime,
                )
            return WheelAction(
                action=WheelActionType.SIT_OUT,
                contract=None,
                reason="no_suitable_put",
                regime=current_regime,
            )

        # PUT_SOLD — hold or close early
        if phase == WheelState.PUT_SOLD:
            if self.should_close_early(position, current_regime, current_pnl_pct):
                return WheelAction(
                    action=WheelActionType.CLOSE,
                    contract=None,
                    reason=f"early_close_put_pnl_{current_pnl_pct:.2f}",
                    regime=current_regime,
                )
            return WheelAction(
                action=WheelActionType.WAIT,
                contract=None,
                reason="holding_put",
                regime=current_regime,
            )

        # ASSIGNED — find a call to sell
        if phase == WheelState.ASSIGNED:
            contract = self.get_call_to_sell(
                symbol=position.symbol,
                option_chain=option_chain,
                shares_owned=position.shares_owned,
                cost_basis=position.cost_basis,
                regime=current_regime,
            )
            if contract:
                return WheelAction(
                    action=WheelActionType.SELL_CALL,
                    contract=contract,
                    reason=f"call_delta_{contract.delta:.3f}",
                    regime=current_regime,
                )
            return WheelAction(
                action=WheelActionType.WAIT,
                contract=None,
                reason="no_suitable_call",
                regime=current_regime,
            )

        # CALL_SOLD — hold or close early
        if phase == WheelState.CALL_SOLD:
            if self.should_close_early(position, current_regime, current_pnl_pct):
                return WheelAction(
                    action=WheelActionType.CLOSE,
                    contract=None,
                    reason=f"early_close_call_pnl_{current_pnl_pct:.2f}",
                    regime=current_regime,
                )
            return WheelAction(
                action=WheelActionType.WAIT,
                contract=None,
                reason="holding_call",
                regime=current_regime,
            )

        return WheelAction(
            action=WheelActionType.WAIT,
            contract=None,
            reason="unknown_phase",
            regime=current_regime,
        )

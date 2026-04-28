"""
core/position_tracker.py
-------------------------
Maintains the authoritative in-process view of open positions and P&L.

Responsibilities:
  - Store and update position records: ticker, direction, entry price, quantity,
    current market value, unrealised P&L, cost basis
  - Listen for fill events from order_executor.py and update positions
    accordingly (open, add-to, reduce, close)
  - Compute real-time portfolio metrics: gross/net exposure, NAV, daily P&L,
    high-water mark, current drawdown from peak
  - Provide the correlation matrix over trailing bars needed by risk_manager.py
  - Persist position state to disk so it survives process restarts; re-hydrate
    from the broker's live position feed on startup to reconcile any drift
  - Expose a snapshot suitable for the Streamlit dashboard
  - Track wheel strategy positions through their full lifecycle

Public interface (to be implemented):
  on_fill(fill_event) -> None
  get_open_positions() -> list[Position]
  get_portfolio_snapshot() -> PortfolioSnapshot
  get_nav() -> float
  get_daily_pnl() -> float
  get_drawdown_from_peak() -> float
  get_correlation_matrix(lookback_bars) -> pd.DataFrame

Wheel extensions:
  track_wheel_position(symbol, state, contract, premium) -> None
  get_wheel_state(symbol) -> Optional[WheelState]
  get_wheel_position(symbol) -> Optional[WheelPosition]
  update_on_assignment(symbol, shares, cost_basis) -> None
  update_on_expiry(symbol) -> None
  update_on_close(symbol, closing_cost) -> None
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from core.wheel_strategy import WheelPosition, WheelState


# ---------------------------------------------------------------------------
# Wheel position store (module-level; cleared in tests via _wheel_positions.clear())
# ---------------------------------------------------------------------------

_wheel_positions: dict[str, WheelPosition] = {}


def track_wheel_position(
    symbol: str,
    state: WheelState,
    contract: Optional[str],
    premium: float,
) -> None:
    """Create or update the wheel position record for symbol."""
    existing = _wheel_positions.get(symbol)
    if existing is None:
        _wheel_positions[symbol] = WheelPosition(
            symbol=symbol,
            phase=state,
            shares_owned=0,
            cost_basis=0.0,
            active_contract=contract,
            premium_collected_total=premium,
            entry_regime=-1,
            timestamp=datetime.now(tz=timezone.utc),
        )
    else:
        _wheel_positions[symbol] = WheelPosition(
            symbol=existing.symbol,
            phase=state,
            shares_owned=existing.shares_owned,
            cost_basis=existing.cost_basis,
            active_contract=contract,
            premium_collected_total=existing.premium_collected_total + premium,
            entry_regime=existing.entry_regime,
            timestamp=datetime.now(tz=timezone.utc),
        )


def get_wheel_state(symbol: str) -> Optional[WheelState]:
    """Return the current WheelState for symbol, or None if not tracked."""
    pos = _wheel_positions.get(symbol)
    return pos.phase if pos is not None else None


def get_wheel_position(symbol: str) -> Optional[WheelPosition]:
    """Return the full WheelPosition for symbol, or None if not tracked."""
    return _wheel_positions.get(symbol)


def update_on_assignment(symbol: str, shares: int, cost_basis: float) -> None:
    """Transition from PUT_SOLD → ASSIGNED after put exercise."""
    existing = _wheel_positions.get(symbol)
    if existing is None:
        _wheel_positions[symbol] = WheelPosition(
            symbol=symbol,
            phase=WheelState.ASSIGNED,
            shares_owned=shares,
            cost_basis=cost_basis,
            active_contract=None,
            premium_collected_total=0.0,
            entry_regime=-1,
            timestamp=datetime.now(tz=timezone.utc),
        )
    else:
        _wheel_positions[symbol] = WheelPosition(
            symbol=existing.symbol,
            phase=WheelState.ASSIGNED,
            shares_owned=shares,
            cost_basis=cost_basis,
            active_contract=None,
            premium_collected_total=existing.premium_collected_total,
            entry_regime=existing.entry_regime,
            timestamp=datetime.now(tz=timezone.utc),
        )


def update_on_expiry(symbol: str) -> None:
    """Handle option expiry: put expires worthless → CASH; call expires → ASSIGNED."""
    existing = _wheel_positions.get(symbol)
    if existing is None:
        return
    if existing.phase == WheelState.PUT_SOLD:
        next_phase = WheelState.CASH
    elif existing.phase == WheelState.CALL_SOLD:
        next_phase = WheelState.ASSIGNED
    else:
        next_phase = existing.phase
    _wheel_positions[symbol] = WheelPosition(
        symbol=existing.symbol,
        phase=next_phase,
        shares_owned=existing.shares_owned,
        cost_basis=existing.cost_basis,
        active_contract=None,
        premium_collected_total=existing.premium_collected_total,
        entry_regime=existing.entry_regime,
        timestamp=datetime.now(tz=timezone.utc),
    )


def update_on_close(symbol: str, closing_cost: float) -> None:
    """Record an early close; net P&L is premium_collected - closing_cost.
    Transitions back to CASH (put closed) or ASSIGNED (call closed).
    """
    existing = _wheel_positions.get(symbol)
    if existing is None:
        return
    next_phase = (
        WheelState.CASH if existing.phase == WheelState.PUT_SOLD
        else WheelState.ASSIGNED
    )
    _wheel_positions[symbol] = WheelPosition(
        symbol=existing.symbol,
        phase=next_phase,
        shares_owned=existing.shares_owned,
        cost_basis=existing.cost_basis,
        active_contract=None,
        premium_collected_total=existing.premium_collected_total - closing_cost,
        entry_regime=existing.entry_regime,
        timestamp=datetime.now(tz=timezone.utc),
    )

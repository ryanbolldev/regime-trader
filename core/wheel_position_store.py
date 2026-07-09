"""
core/wheel_position_store.py
-----------------------------
Hybrid state store for the wheel executor (Phase 2a).

WheelStrategy is stateless — it needs a WheelPosition (phase, shares, cost
basis, premium collected) each call. This store is hybrid:

  - the BROKER is the source of truth for the current bar, reconciled every
    call: phase, shares, cost basis, the active option leg, AND that leg's
    premium (a short put's avg_entry_price is the premium per share). Broker
    wins on conflict, so we never act on a phantom position.

  - a persisted JSON file retains what the broker does NOT: lifetime premium
    collected across closed legs, and the entry regime. These survive restarts.

Assignment / expiry are detected as phase transitions during reconcile: a short
put that disappears with 100 shares appearing = assigned; disappearing with no
shares = expired worthless. Because the open leg's premium comes from the
broker, the executor needs no separate fill-capture step — it just reconciles.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from broker.alpaca_client import _parse_occ_symbol
from core.wheel_strategy import WheelPosition, WheelState

log = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).parent.parent / "logs" / "wheel_positions.json"


@dataclass
class WheelRecord:
    """Per-ticker wheel state: broker-derived current position + persisted history."""
    symbol:                  str
    phase:                   WheelState
    shares_owned:            int
    cost_basis:              float
    active_contract:         Optional[str]
    active_contract_premium: float   # per-contract (per-share) premium of the open leg
    contracts:               int      # contracts on the open leg
    premium_collected_total: float    # lifetime premium collected (persisted)
    entry_regime:            int
    timestamp:               datetime

    def to_wheel_position(self) -> WheelPosition:
        return WheelPosition(
            symbol                  = self.symbol,
            phase                   = self.phase,
            shares_owned            = self.shares_owned,
            cost_basis              = self.cost_basis,
            active_contract         = self.active_contract,
            premium_collected_total = self.premium_collected_total,
            entry_regime            = self.entry_regime,
            timestamp               = self.timestamp,
        )


@dataclass(frozen=True)
class ReconcileResult:
    position:   WheelPosition
    record:     WheelRecord
    transition: Optional[str]   # "PUT_SOLD->ASSIGNED" etc., or None if unchanged


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _broker_state(ticker: str, positions: list):
    """Derive (phase, active_contract, shares, cost_basis, leg_premium,
    leg_contracts) for a ticker from live broker positions. Position-based only
    — pending orders never change phase (nothing is held until a fill)."""
    shares     = 0
    cost_basis = 0.0
    short_put  = short_call = None
    put_prem = call_prem = 0.0
    put_qty  = call_qty = 0

    for p in positions:
        parsed = _parse_occ_symbol(p.symbol)
        if parsed is None:                                   # equity leg
            if p.symbol.upper() == ticker.upper():
                shares     = int(round(p.qty))
                cost_basis = float(p.avg_entry_price)
            continue
        underlying, _exp, opt_type, _strike = parsed
        if underlying.upper() != ticker.upper():
            continue
        if p.qty < 0:                                        # short option leg
            if opt_type == "put":
                short_put, put_prem, put_qty = p.symbol, abs(float(p.avg_entry_price)), abs(int(round(p.qty)))
            elif opt_type == "call":
                short_call, call_prem, call_qty = p.symbol, abs(float(p.avg_entry_price)), abs(int(round(p.qty)))

    if short_put is not None:
        return WheelState.PUT_SOLD, short_put, shares, cost_basis, put_prem, put_qty
    if short_call is not None and shares >= 100:
        return WheelState.CALL_SOLD, short_call, shares, cost_basis, call_prem, call_qty
    if shares >= 100:
        return WheelState.ASSIGNED, None, shares, cost_basis, 0.0, 0
    return WheelState.CASH, None, 0, 0.0, 0.0, 0


class WheelPositionStore:
    """Persisted, broker-reconciled wheel state keyed by ticker."""

    def __init__(self, path: Path = _DEFAULT_PATH) -> None:
        self._path = Path(path)
        self._records: dict[str, WheelRecord] = {}
        self.load()

    # -- persistence ----------------------------------------------------

    def load(self) -> None:
        if not self._path.exists():
            self._records = {}
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._records = {t: _deserialize(d) for t, d in raw.items()}
        except Exception as exc:
            log.warning("WheelPositionStore: could not load %s: %s", self._path, exc)
            self._records = {}

    def save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {t: _serialize(r) for t, r in self._records.items()}
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self._path)
        except Exception as exc:
            log.warning("WheelPositionStore: could not save %s: %s", self._path, exc)

    # -- accessors ------------------------------------------------------

    def get(self, ticker: str) -> Optional[WheelRecord]:
        return self._records.get(ticker)

    def all(self) -> dict[str, WheelRecord]:
        return dict(self._records)

    # -- reconciliation (broker wins) — call at the START of each bar ---

    def reconcile(
        self, ticker: str, positions: list, current_regime: Optional[int] = None
    ) -> ReconcileResult:
        b_phase, b_active, b_shares, b_cost, leg_prem, leg_qty = _broker_state(ticker, positions)
        rec        = self._records.get(ticker)
        prev_phase = rec.phase if rec else WheelState.CASH

        if rec is None:
            rec = WheelRecord(
                symbol=ticker, phase=b_phase, shares_owned=b_shares, cost_basis=b_cost,
                active_contract=b_active, active_contract_premium=0.0, contracts=0,
                premium_collected_total=0.0, entry_regime=-1, timestamp=_now(),
            )
        else:
            rec.phase           = b_phase
            rec.shares_owned    = b_shares
            rec.active_contract = b_active
            rec.timestamp       = _now()
            if b_phase in (WheelState.ASSIGNED, WheelState.CALL_SOLD):
                rec.cost_basis = b_cost

        # Open-leg economics come straight from the broker.
        if b_phase in (WheelState.PUT_SOLD, WheelState.CALL_SOLD):
            rec.active_contract_premium = leg_prem
            rec.contracts               = leg_qty
        else:
            rec.active_contract_premium = 0.0
            rec.contracts               = 0

        transition = None
        if prev_phase != b_phase:
            transition = f"{prev_phase.value}->{b_phase.value}"
            log.info("Wheel [%s] phase transition: %s", ticker, transition)
            # New short-put entry — accumulate lifetime premium + stamp regime.
            if b_phase == WheelState.PUT_SOLD:
                rec.premium_collected_total += leg_prem * 100 * leg_qty
                if current_regime is not None:
                    rec.entry_regime = current_regime

        self._records[ticker] = rec
        self.save()
        return ReconcileResult(rec.to_wheel_position(), rec, transition)


# ---------------------------------------------------------------------------
# (De)serialization
# ---------------------------------------------------------------------------

def _serialize(rec: WheelRecord) -> dict:
    return {
        "symbol":                  rec.symbol,
        "phase":                   rec.phase.value,
        "shares_owned":            rec.shares_owned,
        "cost_basis":              rec.cost_basis,
        "active_contract":         rec.active_contract,
        "active_contract_premium": rec.active_contract_premium,
        "contracts":               rec.contracts,
        "premium_collected_total": rec.premium_collected_total,
        "entry_regime":            rec.entry_regime,
        "timestamp":               rec.timestamp.isoformat(),
    }


def _deserialize(d: dict) -> WheelRecord:
    return WheelRecord(
        symbol                  = d["symbol"],
        phase                   = WheelState(d["phase"]),
        shares_owned            = int(d["shares_owned"]),
        cost_basis              = float(d["cost_basis"]),
        active_contract         = d.get("active_contract"),
        active_contract_premium = float(d.get("active_contract_premium", 0.0)),
        contracts               = int(d.get("contracts", 0)),
        premium_collected_total = float(d.get("premium_collected_total", 0.0)),
        entry_regime            = int(d.get("entry_regime", -1)),
        timestamp               = datetime.fromisoformat(d["timestamp"]),
    )

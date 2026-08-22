"""
core/wheel_executor.py
-----------------------
Phase 2a wheel executor — cash-secured put entry + management.

One pass per interval, per candidate ticker:
  1. reconcile broker state (WheelPositionStore) — alert on assignment/expiry
  2. skip if a wheel order is already resting for the ticker
  3. route by phase:
       CASH      → maybe sell a new cash-secured put (regime + IV gate + caps + sizing)
       PUT_SOLD  → compute P&L, ask the strategy whether to close early
       ASSIGNED  → hold shares; covered calls are Phase 2b (handoff only)

Selection and pricing use Alpaca's chain — the execution venue — so limit prices
match where the order fills. The scanner's tastytrade data is upstream (candidate
discovery). Nothing here runs unless wheel_main enables execution.
"""

from __future__ import annotations

import logging
from typing import Optional

from broker.alpaca_client import _parse_occ_symbol, format_occ_symbol
from config import settings
from core import alerts
from core.wheel_position_store import WheelPositionStore, WheelRecord, _broker_state
from core.wheel_strategy import WheelActionType, WheelState, WheelStrategy

log = logging.getLogger(__name__)

_OPEN_ORDER_STATUSES = {
    "new", "accepted", "pending_new", "partially_filled", "held", "pending_replace",
}


def _mid(contract) -> Optional[float]:
    if contract is None or contract.bid is None or contract.ask is None:
        return None
    if contract.bid <= 0 and contract.ask <= 0:
        return None
    return (contract.bid + contract.ask) / 2.0


def _spread_ok(contract, max_pct: float) -> bool:
    """True if the contract has a real two-sided market with a spread no wider
    than max_pct of the mid — a mid off a one-sided or blown-out book is a
    meaningless limit price."""
    if contract is None or contract.bid is None or contract.ask is None:
        return False
    if contract.bid <= 0 or contract.ask <= 0 or contract.ask < contract.bid:
        return False
    mid = (contract.bid + contract.ask) / 2.0
    return mid > 0 and (contract.ask - contract.bid) / mid <= max_pct


def _open_wheel_tickers(positions: list) -> set[str]:
    """Underlyings the broker says hold a live wheel leg (short option, or a
    100+ share lot).

    Derived from broker positions rather than the store so a lost or emptied
    state file can never orphan an open position: the store only knows what it
    was told last run, and an unmanaged short put has no close path. Phase logic
    is _broker_state's, so this cannot drift from what reconcile() decides.
    """
    underlyings: set[str] = set()
    for p in positions:
        parsed = _parse_occ_symbol(p.symbol)
        underlyings.add((parsed[0] if parsed else p.symbol).upper())
    return {t for t in underlyings if _broker_state(t, positions)[0] != WheelState.CASH}


def _wheel_collateral(positions: list) -> float:
    """Approximate capital already committed to wheel legs: short-put collateral
    (strike × 100 × contracts) plus the market value of long shares."""
    total = 0.0
    for p in positions:
        parsed = _parse_occ_symbol(p.symbol)
        if parsed is None:
            if p.qty > 0:
                total += abs(float(p.market_value))
            continue
        _u, _e, opt_type, strike = parsed
        if opt_type == "put" and p.qty < 0:
            total += strike * 100 * abs(int(round(p.qty)))
    return total


class WheelExecutor:
    """Drives cash-secured put entry + management for a set of candidate tickers."""

    def __init__(self, client, store: Optional[WheelPositionStore] = None,
                 strategy: Optional[WheelStrategy] = None) -> None:
        self._client   = client
        self._store    = store or WheelPositionStore()
        self._strategy = strategy or WheelStrategy()

    # ------------------------------------------------------------------

    def open_tickers(self) -> list[str]:
        """Tickers the broker holds a wheel leg in — always managed, even if the
        scanner drops them from tonight's candidates and even if the state file
        was lost."""
        return sorted(_open_wheel_tickers(self._client.get_positions()))

    def open_positions(self) -> list[WheelRecord]:
        """Live (non-CASH) wheel records, for state/dashboard reporting. Store-
        backed because the persisted economics (lifetime premium, entry regime)
        are what reporting wants; phases are fresh because open_tickers() forces
        every broker-held leg through reconcile() on the pass just before."""
        return [r for r in self._store.all().values() if r.phase != WheelState.CASH]

    def run_once(self, candidates: list[str], regime: int, is_uncertain: bool) -> None:
        market_open = self._client.is_market_open()
        if not market_open:
            log.info("Wheel executor: market closed — reconciling state only, no orders")

        positions   = self._client.get_positions()
        open_orders = self._client.get_orders()
        acct        = self._client.get_account()
        nav         = float(acct.portfolio_value)
        bp          = float(acct.options_buying_power)

        deployed = _wheel_collateral(positions)
        # Broker-derived like the collateral above: counting the store here would
        # read 0 against a lost state file and let MAX_WHEEL_POSITIONS be breached
        # by exactly the legs it had forgotten.
        open_count = len(_open_wheel_tickers(positions))

        for ticker in candidates:
            try:
                committed = self._process(
                    ticker, positions, open_orders, regime, is_uncertain,
                    nav, bp, deployed, open_count, market_open,
                )
                if committed > 0:
                    deployed   += committed
                    open_count += 1
            except Exception:
                log.exception("Wheel executor error for %s", ticker)
                self._alert(ticker, "executor error — see logs", severity="warning")

    # ------------------------------------------------------------------

    def _process(self, ticker, positions, open_orders, regime, is_uncertain,
                 nav, bp, deployed, open_count, market_open) -> float:
        res = self._store.reconcile(ticker, positions, current_regime=regime)
        rec = res.record
        if res.transition:
            self._alert_transition(ticker, res.transition, rec)

        # Orders only during market hours — reconcile + alerts above still run so
        # overnight assignment/expiry is detected and alerted the next morning.
        if not market_open:
            return 0.0

        # Cancel any resting wheel order for this underlying so the decision below
        # re-prices on fresh quotes instead of leaving a stale limit that would
        # freeze the ticker until the day order expires.
        self._cancel_resting_orders(ticker, open_orders)

        if rec.phase == WheelState.CASH:
            return self._maybe_enter(ticker, rec, regime, is_uncertain, nav, bp, deployed, open_count)
        if rec.phase == WheelState.PUT_SOLD:
            self._manage_put(ticker, rec, regime, is_uncertain, nav, bp)
            return 0.0
        if rec.phase == WheelState.ASSIGNED:
            log.info("Wheel [%s]: ASSIGNED — holding %d shares (covered calls = Phase 2b)",
                     ticker, rec.shares_owned)
        return 0.0

    # -- entry ----------------------------------------------------------

    def _maybe_enter(self, ticker, rec, regime, is_uncertain, nav, bp, deployed, open_count) -> float:
        if open_count >= settings.MAX_WHEEL_POSITIONS:
            log.info("Wheel [%s]: at max positions (%d) — no new entry",
                     ticker, settings.MAX_WHEEL_POSITIONS)
            return 0.0

        # No IV data → block the entry rather than silently bypassing the IV
        # gate (get_next_action skips the gate when iv_rank is None).
        iv_rank = self._client.get_iv_rank(ticker)
        if iv_rank is None:
            log.info("Wheel [%s]: IV rank unavailable — blocking entry (no-data)", ticker)
            return 0.0

        chain  = self._client.get_option_chain(ticker)
        action = self._strategy.get_next_action(
            position=rec.to_wheel_position(), current_regime=regime, option_chain=chain,
            portfolio_nav=nav, buying_power=bp, is_uncertain=is_uncertain,
            current_pnl_pct=0.0, iv_rank=iv_rank,
        )
        if action.action != WheelActionType.SELL_PUT or action.contract is None:
            log.debug("Wheel [%s]: no entry (%s: %s)", ticker, action.action.value, action.reason)
            return 0.0

        contract = action.contract
        if not _spread_ok(contract, settings.WHEEL_MAX_SPREAD_PCT):
            log.info("Wheel [%s]: %s spread too wide / one-sided (bid=%s ask=%s) — skip",
                     ticker, contract.symbol, contract.bid, contract.ask)
            return 0.0

        collateral_each = contract.strike * 100
        budget = min(
            nav * settings.WHEEL_MAX_COLLATERAL_PCT,
            nav * settings.WHEEL_TOTAL_DEPLOYED_PCT - deployed,
            bp,
        )
        contracts = int(budget // collateral_each)
        if contracts < 1:
            log.info("Wheel [%s]: budget $%.0f < 1 contract collateral $%.0f — skip",
                     ticker, budget, collateral_each)
            return 0.0

        limit  = round(_mid(contract) * (1 - settings.WHEEL_LIMIT_SLIPPAGE_PCT), 2)
        result = self._submit(
            ticker, symbol=contract.symbol, qty=contracts, side="sell",
            order_type="limit", limit_price=limit, position_intent="sell_to_open",
        )
        if result is None:
            return 0.0
        committed = collateral_each * contracts
        log.info("Wheel [%s]: SELL_PUT %d× %s @ $%.2f (collateral $%.0f, %s)",
                 ticker, contracts, contract.symbol, limit, committed, action.reason)
        self._alert(ticker, f"SELL_PUT {contracts}× {format_occ_symbol(contract.symbol)} @ ${limit:.2f} "
                            f"(delta {contract.delta}, collateral ${committed:,.0f})")
        return committed

    # -- management -----------------------------------------------------

    def _manage_put(self, ticker, rec: WheelRecord, regime, is_uncertain, nav, bp) -> None:
        chain  = self._client.get_option_chain(ticker)
        active = next((c for c in chain if c.symbol == rec.active_contract), None)
        mark   = _mid(active)

        if mark is None:
            # Can't price the open leg → the 50% / 200% / gamma P&L triggers can't
            # evaluate. Surface it loudly rather than silently suspending stops;
            # the regime-based close is still assessed below (it needs no mark).
            log.warning("Wheel [%s]: no quote for open leg %s — P&L stops suspended this pass",
                        ticker, rec.active_contract)
            self._alert(ticker, f"cannot price open leg {format_occ_symbol(rec.active_contract)} — "
                                f"P&L stops suspended until quote returns", severity="warning")

        pnl_pct = 0.0
        if mark is not None and rec.active_contract_premium > 0:
            pnl_pct = (rec.active_contract_premium - mark) / rec.active_contract_premium

        action = self._strategy.get_next_action(
            position=rec.to_wheel_position(), current_regime=regime, option_chain=chain,
            portfolio_nav=nav, buying_power=bp, is_uncertain=is_uncertain,
            current_pnl_pct=pnl_pct, iv_rank=None,
        )
        if action.action != WheelActionType.CLOSE:
            log.debug("Wheel [%s]: holding put (pnl %.0f%%)", ticker, pnl_pct * 100)
            return

        # Buy-to-close at the ask so a decided exit actually fills — paying up is
        # worth the certainty, especially for a regime-driven close.
        close_limit = active.ask if (active and active.ask and active.ask > 0) else mark
        if close_limit is None:
            log.warning("Wheel [%s]: cannot price close for %s — skip", ticker, rec.active_contract)
            return
        close_limit = round(close_limit, 2)

        result = self._submit(
            ticker, symbol=rec.active_contract, qty=rec.contracts, side="buy",
            order_type="limit", limit_price=close_limit, position_intent="buy_to_close",
        )
        if result is None:
            return
        log.info("Wheel [%s]: CLOSE %d× %s @ $%.2f (pnl %.0f%%, %s)",
                 ticker, rec.contracts, rec.active_contract, close_limit, pnl_pct * 100, action.reason)
        self._alert(ticker, f"CLOSE {rec.contracts}× {format_occ_symbol(rec.active_contract)} @ ${close_limit:.2f} "
                            f"(pnl {pnl_pct * 100:.0f}%, {action.reason})")

    # -- helpers --------------------------------------------------------

    def _submit(self, ticker: str, **kwargs):
        """Submit an order, alerting on failure so a rejected entry/close is never
        silently swallowed."""
        try:
            return self._client.submit_order(**kwargs)
        except Exception as exc:
            log.exception("Wheel [%s]: order submit failed", ticker)
            self._alert(
                ticker,
                f"ORDER FAILED: {kwargs.get('side')} {format_occ_symbol(kwargs.get('symbol', ''))} "
                f"— {type(exc).__name__}: {exc}",
                severity="warning",
            )
            return None

    def _cancel_resting_orders(self, ticker: str, open_orders: list) -> None:
        """Cancel any resting option order on this underlying so the next decision
        re-prices on fresh quotes. On the wheel-dedicated account every option
        order is the wheel's; equity orders (non-OCC symbols) are left untouched."""
        for o in open_orders:
            if str(getattr(o, "status", "")).lower() not in _OPEN_ORDER_STATUSES:
                continue
            parsed = _parse_occ_symbol(getattr(o, "symbol", ""))
            if parsed and parsed[0].upper() == ticker.upper():
                try:
                    self._client.cancel_order(o.order_id)
                    log.info("Wheel [%s]: cancelled stale order %s (%s)",
                             ticker, str(o.order_id)[:8], o.symbol)
                except Exception:
                    log.warning("Wheel [%s]: cancel failed for order on %s", ticker, o.symbol)

    def _alert_transition(self, ticker, transition, rec: WheelRecord) -> None:
        if transition == "PUT_SOLD->ASSIGNED":
            self._alert(ticker, f"PUT ASSIGNED — now holding {rec.shares_owned} shares "
                                f"(cost ${rec.cost_basis:.2f}); covered-call phase (2b) not yet enabled",
                        severity="warning")
        elif transition == "PUT_SOLD->CASH":
            self._alert(ticker, f"put closed/expired — premium retained "
                                f"(lifetime ${rec.premium_collected_total:,.0f})")
        else:
            self._alert(ticker, f"phase transition {transition}")

    def _alert(self, ticker, message, severity="info") -> None:
        try:
            alerts.send("wheel_execution", f"[{ticker}] {message}", severity, symbol=ticker)
        except Exception:
            pass

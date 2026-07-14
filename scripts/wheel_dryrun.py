"""
scripts/wheel_dryrun.py
------------------------
Guarded ONE-SHOT live-paper dry-run of the wheel executor.

Calls WheelExecutor.run_once directly (so it never touches the global
WHEEL_EXECUTION_ENABLED flag), during market hours, to validate the live path:

  PASS 1  → reconcile + maybe place a cash-secured put on a scanner candidate
  PASS 2  → demonstrate the cancel-and-reprice lifecycle on the resting order
  CLEANUP → cancel every wheel order it placed (leaves no resting position)

INFO logs are printed so each gate decision (IV gate, spread, sizing, regime) is
visible. Uses a throwaway position-store path so the real one is untouched.

Must run during market hours (the executor's market-hours gate blocks orders when
closed). Optional argv: one or more tickers to override candidate selection.
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, r"c:\Users\bollr\regime_trader")

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

from config.credentials import enable_os_trust_store
enable_os_trust_store()

from broker.alpaca_client import AlpacaClient, _parse_occ_symbol
from core.risk_manager import RiskManager
from core.wheel_executor import WheelExecutor
from core.wheel_position_store import WheelPositionStore
from wheel_main import WheelTrader, _REGIME_NAMES

LOG_DIR = Path(r"c:\Users\bollr\regime_trader\logs")


def _bar(c="=", n=72):
    print(c * n)


def pick_candidates(argv) -> list[str]:
    if argv:
        return [a.upper() for a in argv]
    ws = LOG_DIR / "wheel_state.json"
    if ws.exists():
        try:
            data = json.loads(ws.read_text(encoding="utf-8"))
            cands = [c["ticker"] for c in data.get("candidates", []) if c.get("ticker")]
            if cands:
                return cands[:3]
        except Exception:
            pass
    files = sorted(LOG_DIR.glob("wheel_scanner/wheel_candidates_*.json"))
    if files:
        try:
            data = json.loads(files[-1].read_text(encoding="utf-8"))
            rows = data if isinstance(data, list) else data.get("candidates", [])
            cands = [r.get("ticker") for r in rows if r.get("ticker")]
            if cands:
                return cands[:3]
        except Exception:
            pass
    return ["F"]   # cheap, liquid fallback so sizing stays small


def wheel_orders(client, tickers):
    keep = {t.upper() for t in tickers}
    out = []
    for o in client.get_orders():
        parsed = _parse_occ_symbol(o.symbol)
        if parsed and parsed[0].upper() in keep:
            out.append(o)
    return out


def _show_orders(client, tickers, label):
    orders = wheel_orders(client, tickers)
    print(f"  {label}: {len(orders)} resting wheel order(s)")
    for o in orders:
        print(f"    {o.symbol}  {o.side} qty={o.qty} @ limit  status={o.status}  id={o.order_id[:8]}")
    return orders


def main():
    _bar()
    print("  WHEEL EXECUTOR — guarded live-paper dry-run")
    _bar()

    client = AlpacaClient()
    acct = client.get_account()
    market_open = client.is_market_open()
    print(f"  market_open={market_open}  NAV=${acct.portfolio_value:,.2f}  "
          f"options_bp=${acct.options_buying_power:,.2f}")
    if not market_open:
        print("  NOTE: market is CLOSED — the executor will reconcile only and place no orders.")

    candidates = pick_candidates(sys.argv[1:])
    print(f"  candidates: {candidates}")

    print("  training regime on", "SPY …")
    trader = WheelTrader(client=client, risk_manager=RiskManager(), scan_on_startup=False)
    regime = trader._train_and_predict_regime()
    is_uncertain = trader._hmm.is_uncertain()
    print(f"  regime={_REGIME_NAMES.get(regime, regime)} ({regime})  uncertain={is_uncertain}")

    store = WheelPositionStore(path=LOG_DIR / "wheel_dryrun_positions.json")
    ex = WheelExecutor(client=client, store=store)

    _bar("-")
    print("  PASS 1 — reconcile + maybe enter")
    _bar("-")
    ex.run_once(candidates, regime, is_uncertain)
    time.sleep(3)
    _show_orders(client, candidates, "after PASS 1")

    _bar("-")
    print("  PASS 2 — cancel resting order + re-price (lifecycle)")
    _bar("-")
    ex.run_once(candidates, regime, is_uncertain)
    time.sleep(3)
    _show_orders(client, candidates, "after PASS 2")

    _bar("-")
    print("  CLEANUP — cancelling every order this dry-run placed")
    _bar("-")
    for o in wheel_orders(client, candidates):
        ok = client.cancel_order(o.order_id)
        print(f"    cancel {o.order_id[:8]} ({o.symbol}) -> {ok}")
    time.sleep(2)

    _show_orders(client, candidates, "FINAL orders")
    opt_positions = [p for p in client.get_positions() if _parse_occ_symbol(p.symbol)]
    print(f"  FINAL option positions: {len(opt_positions)}")
    for p in opt_positions:
        print(f"    {p.symbol} qty={p.qty} mv=${p.market_value}")
    if opt_positions:
        print("  ⚠ A wheel order FILLED before cleanup — review/close manually if unwanted.")

    _bar()
    print("  DRY-RUN COMPLETE")
    _bar()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n  DRY-RUN FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

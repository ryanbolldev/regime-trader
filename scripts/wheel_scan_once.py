"""
scripts/wheel_scan_once.py
---------------------------
Run ONE wheel scan (regime-aware) and write logs/wheel_state.json, so we can
validate candidate generation during market hours. Mirrors what wheel_main does
on its scheduled fire, but as a standalone one-shot.
"""
import logging
import sys

sys.path.insert(0, r"c:\Users\bollr\regime_trader")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S", stream=sys.stdout,
)

from config.credentials import enable_os_trust_store
enable_os_trust_store()
from broker.alpaca_client import AlpacaClient
from core.risk_manager import RiskManager
from wheel_main import WheelTrader

t = WheelTrader(client=AlpacaClient(), risk_manager=RiskManager(), scan_on_startup=False)
print(f"market_open={t._client.is_market_open()}")
t._run_scan()

cands = t._last_candidates
print(f">>> SCAN DONE — {len(cands)} candidate(s)")
for c in cands[:15]:
    print(f"    {c.ticker:6} score={c.composite_score:.0f}  ivr={c.ivr}  "
          f"strike={c.target_put_strike}  dte={c.dte}")

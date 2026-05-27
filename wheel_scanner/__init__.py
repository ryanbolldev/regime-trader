"""
wheel_scanner/
--------------
Standalone wheel strategy candidate scanner for regime_trader.

Identifies mid-cap equities ($2B–$10B market cap, $15–$150 price) suitable
for the wheel strategy: selling cash-secured puts to accumulate shares then
selling covered calls for income.

Entry point:
    python -m wheel_scanner.scanner [--regime {0-4}] [--output-dir PATH]

Integration:
    from wheel_scanner import WheelScanner, WheelCandidate
    candidates = WheelScanner(regime_label=3).run()

regime_label integers match the existing HMM engine output:
    0=crash  1=bear  2=neutral  3=bull  4=euphoria
"""

from wheel_scanner.output import WheelCandidate
from wheel_scanner.scanner import WheelScanner

__all__ = ["WheelScanner", "WheelCandidate"]

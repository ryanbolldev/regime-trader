"""
wheel_scanner/output.py
-----------------------
WheelCandidate dataclass and result serialisation.

Output files are written to logs/wheel_scanner/ (mirrors logs/scanner/).
Each run produces:
  wheel_candidates_YYYY-MM-DD.json  — machine-readable ranked list
  wheel_candidates_YYYY-MM-DD.csv   — spreadsheet-friendly format
"""

from __future__ import annotations

import csv
import json
import logging
import pathlib
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Optional

log = logging.getLogger(__name__)

_LOGS_DIR = pathlib.Path("logs") / "wheel_scanner"


@dataclass
class WheelCandidate:
    """A single wheel strategy candidate with all scoring inputs and outputs."""

    ticker:               str
    composite_score:      float
    market_cap_B:         Optional[float]   # billions; None if unavailable
    price:                float
    ivr:                  Optional[float]   # 0–100; None if unavailable
    target_put_strike:    Optional[float]
    target_expiry:        Optional[str]     # YYYY-MM-DD
    dte:                  Optional[int]
    put_mid:              Optional[float]
    annualized_yield_pct: Optional[float]
    open_interest:        Optional[int]
    bid_ask_spread_pct:   Optional[float]
    regime:               str               # "bull", "bear", etc.
    trend_score:          str               # "above_both_smas", "recovering", "below_50sma", "downtrend"
    earnings_within_14d:  bool
    flags:                list[str]         = field(default_factory=list)

    # Component scores (for diagnostics / integration)
    score_ivr:            float = 0.0
    score_premium:        float = 0.0
    score_regime:         float = 0.0
    score_trend:          float = 0.0
    score_liquidity:      float = 0.0


def write_results(
    candidates: list[WheelCandidate],
    logs_dir: pathlib.Path | None = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write ranked candidates to JSON and CSV. Returns (json_path, csv_path)."""
    out_dir = logs_dir or _LOGS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()
    json_path = out_dir / f"wheel_candidates_{today}.json"
    csv_path  = out_dir / f"wheel_candidates_{today}.csv"

    payload: dict[str, Any] = {
        "date":        today,
        "total":       len(candidates),
        "candidates":  [asdict(c) for c in candidates],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("WheelScanner: JSON written to %s", json_path)

    if candidates:
        fieldnames = list(asdict(candidates[0]).keys())
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for c in candidates:
                row = asdict(c)
                row["flags"] = "|".join(row["flags"])
                writer.writerow(row)
        log.info("WheelScanner: CSV written to %s", csv_path)

    return json_path, csv_path


def print_summary(candidates: list[WheelCandidate]) -> None:
    """Print a ranked summary table to stdout."""
    if not candidates:
        print("No wheel candidates met the scoring threshold.")
        return

    header = (
        f"{'#':>3}  {'Ticker':<6}  {'Score':>5}  {'IVR':>5}  "
        f"{'Strike':>7}  {'Expiry':<11}  {'DTE':>3}  {'Yield%':>6}  "
        f"{'OI':>5}  {'Regime':<9}  {'Trend':<16}  Flags"
    )
    sep = "-" * len(header)
    print(f"\n{'WHEEL STRATEGY CANDIDATES':^{len(header)}}")
    print(sep)
    print(header)
    print(sep)

    for i, c in enumerate(candidates, 1):
        ivr_str    = f"{c.ivr:.0f}" if c.ivr is not None else "N/A"
        strike_str = f"{c.target_put_strike:.1f}" if c.target_put_strike else "N/A"
        expiry_str = c.target_expiry or "N/A"
        dte_str    = str(c.dte) if c.dte is not None else "N/A"
        yield_str  = f"{c.annualized_yield_pct:.1f}" if c.annualized_yield_pct is not None else "N/A"
        oi_str     = str(c.open_interest) if c.open_interest is not None else "N/A"
        flags_str  = ", ".join(c.flags) if c.flags else ""

        print(
            f"{i:>3}  {c.ticker:<6}  {c.composite_score:>5.1f}  {ivr_str:>5}  "
            f"{strike_str:>7}  {expiry_str:<11}  {dte_str:>3}  {yield_str:>6}  "
            f"{oi_str:>5}  {c.regime:<9}  {c.trend_score:<16}  {flags_str}"
        )

    print(sep)
    print(f"Total candidates: {len(candidates)}\n")

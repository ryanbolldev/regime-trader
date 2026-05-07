"""
core/lag_analysis.py
--------------------
Regime transition lag analysis for the walk-forward backtester.

Measures the delay between when the raw HMM output (pre-confirmation-gate)
first signals a new regime and when the confirmation gate fires, then
quantifies the price and portfolio damage incurred during that lag window.

Public interface:
  compute_lag_transitions(raw_regimes, confirmed_regimes, close_prices,
                          nav_history, alloc_history, timestamps, ticker,
                          bar_interval_secs) -> list[LagTransition]
  build_lag_report(transitions, bar_interval_secs) -> dict
  write_lag_report(report, logs_dir) -> pathlib.Path
  print_lag_summary(report) -> None
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class LagTransition:
    ticker:                   str
    date:                     str
    from_regime:              int
    to_regime:                int
    raw_transition_bar:       int
    confirmed_transition_bar: int
    lag_bars:                 int
    lag_hours:                float
    price_at_raw:             float
    price_at_confirmed:       float
    price_change_pct:         float
    nav_at_transition:        float
    estimated_damage_usd:     float


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_lag_transitions(
    raw_regimes:      list[int],
    confirmed_regimes: list[int],
    close_prices:     list[float],
    nav_history:      list[float],
    alloc_history:    list[float],
    timestamps:       list,
    ticker:           str,
    bar_interval_secs: int,
) -> list[LagTransition]:
    """Extract per-transition lag metrics from per-bar tracking data.

    Parameters
    ----------
    raw_regimes       : HMM raw mapped regime at each bar (engine._pending_state).
    confirmed_regimes : Confirmed regime at each bar (return of predict_current()).
                        -1 means not yet confirmed.
    close_prices      : Close price at each bar.
    nav_history       : Portfolio NAV at the START of each bar (before bar return).
    alloc_history     : Portfolio allocation fraction at the START of each bar.
    timestamps        : Timestamp (DatetimeIndex element) for each bar.
    ticker            : Symbol name, used for reporting.
    bar_interval_secs : Seconds per bar (86400 for daily).

    Returns
    -------
    List of LagTransition objects, one per confirmed regime change.
    """
    transitions: list[LagTransition] = []
    n = len(confirmed_regimes)

    # Start of the search window for each transition: we look for the first
    # raw appearance of the new regime since the previous confirmed change.
    prev_search_start = 0

    for i in range(1, n):
        prev_conf = confirmed_regimes[i - 1]
        curr_conf = confirmed_regimes[i]

        # Skip unconfirmed or unchanged bars
        if curr_conf == prev_conf or curr_conf < 0 or prev_conf < 0:
            continue

        confirmed_bar = i
        from_regime   = prev_conf
        to_regime     = curr_conf

        # Find the FIRST bar (since prev_search_start) where raw == to_regime.
        # This is the earliest possible signal before confirmation fired.
        raw_bar = confirmed_bar  # default: zero lag
        for j in range(prev_search_start, confirmed_bar + 1):
            if j < len(raw_regimes) and raw_regimes[j] == to_regime:
                raw_bar = j
                break

        lag_bars  = confirmed_bar - raw_bar
        lag_hours = lag_bars * bar_interval_secs / 3600.0

        price_at_raw       = float(close_prices[raw_bar])       if raw_bar       < len(close_prices) else 0.0
        price_at_confirmed = float(close_prices[confirmed_bar]) if confirmed_bar < len(close_prices) else 0.0

        if price_at_raw != 0.0:
            price_change_pct = (price_at_confirmed - price_at_raw) / price_at_raw * 100.0
        else:
            price_change_pct = 0.0

        nav_at_raw   = float(nav_history[raw_bar])   if raw_bar < len(nav_history)   else 0.0
        alloc_at_raw = float(alloc_history[raw_bar]) if raw_bar < len(alloc_history) else 0.0
        estimated_damage = nav_at_raw * abs(price_change_pct / 100.0) * alloc_at_raw

        ts = timestamps[confirmed_bar]
        date_str = str(ts.date()) if hasattr(ts, "date") else str(ts)

        transitions.append(LagTransition(
            ticker=ticker,
            date=date_str,
            from_regime=from_regime,
            to_regime=to_regime,
            raw_transition_bar=raw_bar,
            confirmed_transition_bar=confirmed_bar,
            lag_bars=lag_bars,
            lag_hours=round(lag_hours, 2),
            price_at_raw=round(price_at_raw, 4),
            price_at_confirmed=round(price_at_confirmed, 4),
            price_change_pct=round(price_change_pct, 4),
            nav_at_transition=round(nav_at_raw, 2),
            estimated_damage_usd=round(estimated_damage, 2),
        ))

        log.debug(
            "Lag transition [%s] %d→%d @ %s: raw_bar=%d confirmed_bar=%d "
            "lag=%d bars price_chg=%.2f%% damage=$%.0f",
            ticker, from_regime, to_regime, date_str,
            raw_bar, confirmed_bar, lag_bars, price_change_pct, estimated_damage,
        )

        prev_search_start = confirmed_bar + 1

    return transitions


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_lag_report(
    transitions:       list[LagTransition],
    bar_interval_secs: int,
) -> dict[str, Any]:
    """Build the JSON-serialisable lag report dict."""
    empty_summary: dict[str, Any] = {
        "total_transitions":          0,
        "crash_transitions":          0,
        "mean_lag_bars":              0.0,
        "mean_lag_hours":             0.0,
        "worst_lag_bars":             0,
        "worst_lag_hours":            0.0,
        "mean_price_damage_pct":      0.0,
        "worst_price_damage_pct":     0.0,
        "mean_portfolio_damage_usd":  0.0,
        "worst_portfolio_damage_usd": 0.0,
    }

    if not transitions:
        return {
            "summary":            empty_summary,
            "by_transition_type": {},
            "all_transitions":    [],
        }

    crash_transitions = [t for t in transitions if t.to_regime == 0]
    lags        = [t.lag_bars          for t in transitions]
    damages_pct = [t.price_change_pct  for t in transitions]
    damages_usd = [t.estimated_damage_usd for t in transitions]

    worst_lag   = int(max(lags))
    summary: dict[str, Any] = {
        "total_transitions":          len(transitions),
        "crash_transitions":          len(crash_transitions),
        "mean_lag_bars":              round(float(np.mean(lags)), 2),
        "mean_lag_hours":             round(float(np.mean(lags)) * bar_interval_secs / 3600.0, 2),
        "worst_lag_bars":             worst_lag,
        "worst_lag_hours":            round(worst_lag * bar_interval_secs / 3600.0, 2),
        "mean_price_damage_pct":      round(float(np.mean(damages_pct)), 4),
        "worst_price_damage_pct":     round(float(min(damages_pct)), 4),
        "mean_portfolio_damage_usd":  round(float(np.mean(damages_usd)), 2),
        "worst_portfolio_damage_usd": round(float(max(damages_usd)), 2),
    }

    # Group by "from_regime_to_to_regime"
    by_type: dict[str, list[LagTransition]] = {}
    for t in transitions:
        key = f"{t.from_regime}_to_{t.to_regime}"
        by_type.setdefault(key, []).append(t)

    by_transition_type: dict[str, Any] = {}
    for key, group in by_type.items():
        by_transition_type[key] = {
            "count":          len(group),
            "mean_lag_bars":  round(float(np.mean([t.lag_bars         for t in group])), 2),
            "mean_damage_pct": round(float(np.mean([t.price_change_pct for t in group])), 4),
        }

    return {
        "summary":            summary,
        "by_transition_type": by_transition_type,
        "all_transitions":    [asdict(t) for t in transitions],
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_lag_report(report: dict, logs_dir: pathlib.Path) -> pathlib.Path:
    """Serialise *report* to *logs_dir*/regime_lag_analysis.json.

    Creates *logs_dir* if it does not exist.  Returns the written path.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    out = logs_dir / "regime_lag_analysis.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Lag analysis written to %s", out)
    return out


def print_lag_summary(report: dict) -> None:
    """Print a human-readable lag summary to stdout."""
    s  = report.get("summary", {})
    by = report.get("by_transition_type", {})

    lines = [
        "",
        "=== REGIME TRANSITION LAG ANALYSIS ===",
        f"Total transitions analysed: {s.get('total_transitions', 0)}",
        f"Crash regime transitions: {s.get('crash_transitions', 0)}",
        "",
        f"Mean lag:  {s.get('mean_lag_bars', 0):.1f} bars  "
        f"({s.get('mean_lag_hours', 0):.1f} hours)",
        f"Worst lag: {s.get('worst_lag_bars', 0)} bars    "
        f"({s.get('worst_lag_hours', 0):.1f} hours)",
        "",
        f"Mean price damage during lag:   {s.get('mean_price_damage_pct', 0):.1f}%",
        f"Worst price damage during lag:  {s.get('worst_price_damage_pct', 0):.1f}%",
        "",
        f"Mean portfolio damage during lag:   ${s.get('mean_portfolio_damage_usd', 0):,.0f}",
        f"Worst portfolio damage during lag:  ${s.get('worst_portfolio_damage_usd', 0):,.0f}",
    ]

    crash_keys = sorted(k for k in by if k.endswith("_to_0"))
    if crash_keys:
        lines.append("")
        lines.append("By transition type:")
        for key in crash_keys:
            data  = by[key]
            parts = key.split("_to_")
            fr, to = parts[0], parts[1]
            lines.append(
                f"  {fr}→{to}: {data['count']} events | "
                f"mean lag {data['mean_lag_bars']:.1f} bars | "
                f"mean damage {data['mean_damage_pct']:.1f}%"
            )

    lines.append("======================================")
    print("\n".join(lines))

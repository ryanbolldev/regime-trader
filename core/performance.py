"""
core/performance.py
--------------------
Performance analytics for backtesting and live trading.

Metrics:
  - Sharpe ratio (annualised, RF = RISK_FREE_RATE from settings)
  - Sortino ratio
  - Calmar ratio  (annualised return / |max drawdown|)
  - Max drawdown depth and duration (bars)
  - Win rate and average win / loss
  - Profit factor
  - Annualised return and volatility
  - Regime breakdown  (return + Sharpe per regime)
  - Confidence bucket analysis (high >0.7, medium 0.4–0.7, low <0.4)
  - Benchmark comparison table

Public interface:
  compute(equity_series, trades, regime_log) -> PerformanceReport
  compare_benchmarks(strategy_report, benchmark_reports) -> ComparisonTable
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RISK_FREE_RATE      = 0.045   # 4.5 % annual
TRADING_DAYS        = 252

# Confidence bucket boundaries (task spec)
CONFIDENCE_HIGH_FLOOR   = 0.70
CONFIDENCE_MEDIUM_FLOOR = 0.40


# ---------------------------------------------------------------------------
# Shared data types (imported by backtester.py to avoid circular deps)
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """One tradeable bar — a bar where the portfolio had nonzero allocation."""
    timestamp:   pd.Timestamp
    regime:      int
    confidence:  float
    pnl:         float     # dollar P&L this bar
    return_pct:  float     # portfolio return this bar (alloc × price_return)


@dataclass
class RegimeLogEntry:
    """One entry per simulated bar."""
    timestamp:    pd.Timestamp
    regime:       int       # -1 = not yet confirmed
    confidence:   float
    is_uncertain: bool
    allocation_pct: float


# ---------------------------------------------------------------------------
# Report data types
# ---------------------------------------------------------------------------

@dataclass
class RegimeStats:
    regime:       int
    regime_name:  str
    n_bars:       int
    pct_time:     float
    total_return: float
    sharpe:       float


@dataclass
class BucketStats:
    label:           str
    confidence_min:  float
    confidence_max:  float
    n_trades:        int
    win_rate:        float
    avg_return:      float
    total_return:    float
    sharpe:          float


@dataclass
class PerformanceReport:
    sharpe:               float
    sortino:              float
    calmar:               float
    max_drawdown:         float    # negative fraction, e.g. -0.15
    max_drawdown_duration: int     # bars
    win_rate:             float
    avg_win:              float    # mean winning trade return (positive)
    avg_loss:             float    # mean losing trade return (negative)
    win_loss_ratio:       float
    profit_factor:        float
    annualized_return:    float
    annualized_vol:       float
    total_return:         float
    n_trades:             int
    n_bars:               int
    regime_breakdown:     dict[int, RegimeStats] = field(default_factory=dict)
    confidence_buckets:   dict[str, BucketStats] = field(default_factory=dict)


@dataclass
class ComparisonRow:
    name:              str
    sharpe:            float
    total_return:      float
    max_drawdown:      float
    annualized_return: float
    annualized_vol:    float


@dataclass
class ComparisonTable:
    rows: list[ComparisonRow] = field(default_factory=list)

    def to_csv(self, path: str | Path) -> None:
        """Write the comparison table to a CSV file."""
        cols = ["name", "sharpe", "total_return", "max_drawdown",
                "annualized_return", "annualized_vol"]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({
                    "name":              row.name,
                    "sharpe":            round(row.sharpe, 4),
                    "total_return":      round(row.total_return, 4),
                    "max_drawdown":      round(row.max_drawdown, 4),
                    "annualized_return": round(row.annualized_return, 4),
                    "annualized_vol":    round(row.annualized_vol, 4),
                })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute(
    equity_series: pd.Series,
    trades:        list[Trade],
    regime_log:    list[RegimeLogEntry],
) -> PerformanceReport:
    """Compute all performance metrics from an equity curve and trade / regime log.

    Parameters
    ----------
    equity_series : NAV indexed by timestamp (one value per bar).
    trades        : list of Trade objects (one per bar where allocation > 0).
    regime_log    : list of RegimeLogEntry (one per bar, parallel to equity_series).
    """
    nav = equity_series.values.astype(float)
    returns = np.diff(nav) / nav[:-1]   # length n-1

    sharpe        = _sharpe(returns)
    sortino       = _sortino(returns)
    max_dd, dd_dur = _max_drawdown(nav)
    ann_ret       = _annualized_return(nav)
    ann_vol       = _annualized_vol(returns)
    calmar        = ann_ret / abs(max_dd) if max_dd != 0.0 else 0.0
    total_ret     = (nav[-1] / nav[0]) - 1.0 if nav[0] != 0 else 0.0

    pnls    = np.array([t.pnl for t in trades])
    wins    = pnls[pnls > 0]
    losses  = pnls[pnls < 0]
    win_rate       = float(len(wins) / len(pnls)) if len(pnls) > 0 else 0.0
    avg_win        = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss       = float(np.mean(losses)) if len(losses) > 0 else 0.0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0.0 else float("inf")
    profit_factor  = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")

    regime_breakdown  = _regime_breakdown(equity_series, regime_log)
    confidence_buckets = _confidence_buckets(trades)

    return PerformanceReport(
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration=dd_dur,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=win_loss_ratio,
        profit_factor=profit_factor,
        annualized_return=ann_ret,
        annualized_vol=ann_vol,
        total_return=total_ret,
        n_trades=len(trades),
        n_bars=len(nav),
        regime_breakdown=regime_breakdown,
        confidence_buckets=confidence_buckets,
    )


def compare_benchmarks(
    strategy_report: PerformanceReport,
    benchmark_reports: dict[str, PerformanceReport],
) -> ComparisonTable:
    """Build a side-by-side comparison table.

    Parameters
    ----------
    strategy_report    : PerformanceReport for the main strategy.
    benchmark_reports  : dict mapping benchmark name → PerformanceReport.
    """
    def _row(name: str, r: PerformanceReport) -> ComparisonRow:
        return ComparisonRow(
            name=name,
            sharpe=r.sharpe,
            total_return=r.total_return,
            max_drawdown=r.max_drawdown,
            annualized_return=r.annualized_return,
            annualized_vol=r.annualized_vol,
        )

    rows = [_row("strategy", strategy_report)]
    for name, rpt in benchmark_reports.items():
        rows.append(_row(name, rpt))
    return ComparisonTable(rows=rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rf_daily() -> float:
    return (1.0 + RISK_FREE_RATE) ** (1.0 / TRADING_DAYS) - 1.0


def _sharpe(returns: np.ndarray) -> float:
    if len(returns) == 0 or np.std(returns) == 0.0:
        return 0.0
    excess = returns - _rf_daily()
    std    = float(np.std(excess, ddof=1))
    return 0.0 if std == 0.0 else float(np.mean(excess) / std * math.sqrt(TRADING_DAYS))


def _sortino(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    rf     = _rf_daily()
    excess = returns - rf
    down   = excess[excess < 0]
    if len(down) == 0:
        return float("inf") if np.mean(excess) > 0 else 0.0
    semi_std = math.sqrt(float(np.mean(down ** 2)))
    return 0.0 if semi_std == 0.0 else float(np.mean(excess) / semi_std * math.sqrt(TRADING_DAYS))


def _max_drawdown(nav: np.ndarray) -> tuple[float, int]:
    """Returns (max_drawdown as negative fraction, duration in bars)."""
    if len(nav) == 0:
        return 0.0, 0
    peak      = nav[0]
    peak_idx  = 0
    max_dd    = 0.0
    max_dur   = 0
    for i, v in enumerate(nav):
        if v >= peak:
            peak     = v
            peak_idx = i
        dd = (v - peak) / peak if peak != 0 else 0.0
        if dd < max_dd:
            max_dd  = dd
            max_dur = i - peak_idx
    return float(max_dd), int(max_dur)


def _annualized_return(nav: np.ndarray) -> float:
    if len(nav) < 2 or nav[0] == 0:
        return 0.0
    n_days   = len(nav) - 1
    total    = nav[-1] / nav[0]
    return float(total ** (TRADING_DAYS / n_days) - 1.0)


def _annualized_vol(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    return float(np.std(returns, ddof=1) * math.sqrt(TRADING_DAYS))


_REGIME_NAMES = {0: "crash", 1: "bear", 2: "neutral", 3: "bull", 4: "euphoria"}


def _regime_breakdown(
    equity_series: pd.Series,
    regime_log:    list[RegimeLogEntry],
) -> dict[int, RegimeStats]:
    if not regime_log:
        return {}

    nav     = equity_series.values.astype(float)
    n_total = len(regime_log)

    # Group bar indices by regime
    from collections import defaultdict
    regime_bars: dict[int, list[int]] = defaultdict(list)
    for i, entry in enumerate(regime_log):
        regime_bars[entry.regime].append(i)

    result: dict[int, RegimeStats] = {}
    for regime, idxs in sorted(regime_bars.items()):
        if regime < 0:    # unconfirmed bars — bucket under -1 but skip for report
            continue
        # Collect returns for these bars (bar i return = nav[i] / nav[i-1] - 1)
        bar_returns = []
        for idx in idxs:
            if idx > 0 and idx < len(nav):
                bar_returns.append((nav[idx] / nav[idx - 1]) - 1.0)

        arr      = np.array(bar_returns)
        tr       = float(np.prod(1 + arr) - 1) if len(arr) > 0 else 0.0
        s        = _sharpe(arr)
        n_bars   = len(idxs)
        result[regime] = RegimeStats(
            regime=regime,
            regime_name=_REGIME_NAMES.get(regime, f"state_{regime}"),
            n_bars=n_bars,
            pct_time=n_bars / n_total,
            total_return=tr,
            sharpe=s,
        )
    return result


def _confidence_buckets(trades: list[Trade]) -> dict[str, BucketStats]:
    """Always return exactly three buckets: high, medium, low."""
    buckets: dict[str, BucketStats] = {}
    for label, lo, hi in [
        ("high",   CONFIDENCE_HIGH_FLOOR,   1.01),
        ("medium", CONFIDENCE_MEDIUM_FLOOR, CONFIDENCE_HIGH_FLOOR),
        ("low",    -0.01,                   CONFIDENCE_MEDIUM_FLOOR),
    ]:
        subset = [t for t in trades if lo <= t.confidence < hi]
        pnls   = np.array([t.pnl for t in subset])
        rets   = np.array([t.return_pct for t in subset])
        n      = len(subset)
        wins   = (pnls > 0).sum()
        buckets[label] = BucketStats(
            label=label,
            confidence_min=lo if lo >= 0 else 0.0,
            confidence_max=hi if hi <= 1.0 else 1.0,
            n_trades=n,
            win_rate=float(wins / n) if n > 0 else 0.0,
            avg_return=float(np.mean(rets)) if n > 0 else 0.0,
            total_return=float(np.prod(1 + rets) - 1) if n > 0 else 0.0,
            sharpe=_sharpe(rets) if n > 1 else 0.0,
        )
    return buckets

"""
core/backtester.py
-------------------
Walk-forward backtest engine.

Window configuration (from settings.py):
  In-sample  : 252 bars  (HMM training)
  OOS        : 126 bars  (evaluation; never seen during fitting)
  Step size  : 63 bars   (quarterly re-training)

Anti-lookahead guarantees:
  - HMM is re-fitted inside each IS window using only data up to the
    IS boundary.  OOS evaluation runs in forward-only mode.
  - Feature computation for OOS bar t uses the combined IS+OOS prefix
    [0 … t], so no future bar is ever included.
  - An optional audit pass calls feature_engineering.validate_no_lookahead()
    and raises LookaheadBiasError if any feature leaks future data.

Benchmarks (run over every OOS window):
  1. Buy-and-hold       — 100 % allocation, hold throughout
  2. SMA-200 trend      — long when price > 200-bar SMA, cash otherwise
  3. Random entry       — random allocation from {0, 0.5, 1.0} per bar

Stress scenarios (injected into a copy of the OOS data):
  - crash_day           : single bar return = −12 %
  - consecutive_drops   : N consecutive bars at −3.5 % each
  - vol_spike           : HL range doubled for 10 bars (affects features)

Public interface:
  run(ohlcv_df, initial_nav, audit_lookahead) -> BacktestReport
  run_fold(is_df, oos_df, initial_nav, feature_fn, audit_lookahead)
      -> FoldResult
  run_stress_test(oos_df, n_injections, initial_nav) -> StressReport
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from config.settings import (
    BACKTEST_IN_SAMPLE_BARS,
    BACKTEST_OUT_SAMPLE_BARS,
    BACKTEST_STEP_BARS,
)
from core.feature_engineering import compute, validate_no_lookahead
from core.hmm_engine import HMMEngine
from core.performance import (
    PerformanceReport,
    RegimeLogEntry,
    Trade,
    compute as compute_performance,
)
from core.regime_strategies import get_signal
from core.risk_manager import RiskManager

log = logging.getLogger(__name__)

INITIAL_NAV = 100_000.0
SLIPPAGE    = 0.0005    # 0.05 % per side (entry + exit = 0.10 % round-trip)


# ---------------------------------------------------------------------------
# Result data types
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold_idx:        int
    in_sample_start: pd.Timestamp
    in_sample_end:   pd.Timestamp
    oos_start:       pd.Timestamp
    oos_end:         pd.Timestamp
    equity_curve:    pd.Series            # NAV indexed by date
    trades:          list[Trade]
    regime_log:      list[RegimeLogEntry]
    benchmark_curves: dict[str, pd.Series]
    regime_distribution: dict[int, int]   # regime → bar count
    n_hmm_states:    int


@dataclass
class BacktestReport:
    folds:            list[FoldResult]
    equity_curve:     pd.Series            # concatenated OOS NAV
    trades:           list[Trade]
    regime_log:       list[RegimeLogEntry]
    performance:      PerformanceReport
    benchmark_reports: dict[str, PerformanceReport]
    n_folds:          int


@dataclass
class StressReport:
    circuit_breakers_fired: list[str]
    max_drawdown:           float
    equity_curve:           pd.Series
    n_injections:           int
    events_applied:         list[str]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Backtester:

    def run(
        self,
        ohlcv_df:        pd.DataFrame,
        initial_nav:     float = INITIAL_NAV,
        audit_lookahead: bool  = True,
    ) -> BacktestReport:
        """Run full walk-forward backtest on ohlcv_df.

        Parameters
        ----------
        ohlcv_df        : OHLCV DataFrame with DatetimeIndex, sorted ascending.
        initial_nav     : starting portfolio NAV.
        audit_lookahead : if True, each fold calls validate_no_lookahead().
        """
        windows = _compute_windows(
            len(ohlcv_df),
            BACKTEST_IN_SAMPLE_BARS,
            BACKTEST_OUT_SAMPLE_BARS,
            BACKTEST_STEP_BARS,
        )
        if not windows:
            raise ValueError(
                f"Not enough data for even one fold. "
                f"Need at least {BACKTEST_IN_SAMPLE_BARS + BACKTEST_OUT_SAMPLE_BARS} bars."
            )

        folds: list[FoldResult] = []
        nav = initial_nav
        for idx, (is_start, is_end, oos_end) in enumerate(windows):
            is_df  = ohlcv_df.iloc[is_start:is_end]
            oos_df = ohlcv_df.iloc[is_end:oos_end]
            fold   = self.run_fold(is_df, oos_df, nav, audit_lookahead=audit_lookahead,
                                   fold_idx=idx)
            folds.append(fold)
            nav = float(fold.equity_curve.iloc[-1])

        # Concatenate OOS equity curves
        equity = pd.concat([f.equity_curve for f in folds])
        all_trades     = [t for f in folds for t in f.trades]
        all_regime_log = [r for f in folds for r in f.regime_log]

        perf = compute_performance(equity, all_trades, all_regime_log)

        # Benchmark reports (average across folds)
        bmark_names  = list(folds[0].benchmark_curves.keys())
        bmark_reports: dict[str, PerformanceReport] = {}
        for name in bmark_names:
            bmark_equity = pd.concat([f.benchmark_curves[name] for f in folds])
            bmark_trades = _equity_to_trades(bmark_equity)
            bmark_reports[name] = compute_performance(bmark_equity, bmark_trades, [])

        return BacktestReport(
            folds=folds,
            equity_curve=equity,
            trades=all_trades,
            regime_log=all_regime_log,
            performance=perf,
            benchmark_reports=bmark_reports,
            n_folds=len(folds),
        )

    def run_fold(
        self,
        is_df:           pd.DataFrame,
        oos_df:          pd.DataFrame,
        initial_nav:     float = INITIAL_NAV,
        feature_fn:      Callable = compute,
        audit_lookahead: bool  = True,
        fold_idx:        int   = 0,
    ) -> FoldResult:
        """Train HMM on is_df, simulate the strategy bar-by-bar over oos_df.

        Parameters
        ----------
        is_df           : in-sample OHLCV (for HMM training only).
        oos_df          : out-of-sample OHLCV (never seen during training).
        initial_nav     : starting NAV for this fold.
        feature_fn      : feature computation function; default = compute().
        audit_lookahead : if True, calls validate_no_lookahead().
        fold_idx        : fold index for logging.
        """
        # Combine IS + OOS so rolling features have warmup history
        combined = pd.concat([is_df, oos_df])

        # Compute features on the full combined window
        features_all = feature_fn(combined)

        if audit_lookahead:
            validate_no_lookahead(features_all, combined)

        # Slice IS features (for HMM training) and OOS features (for simulation)
        is_features  = features_all.iloc[: len(is_df)].dropna()
        oos_features = features_all.iloc[len(is_df):].dropna()

        if len(is_features) < 30:
            raise ValueError(f"Fold {fold_idx}: insufficient IS features after dropna.")

        # Fit HMM on IS data only
        engine = HMMEngine()
        engine.fit(is_features)
        log.info("Fold %d: HMM fitted with %d states.", fold_idx, engine._n_states)

        # Simulate OOS bars
        risk_mgr = RiskManager()
        risk_mgr.initialize(initial_nav)

        nav        = initial_nav
        allocation = 0.0
        prev_close: Optional[float] = None
        trades:     list[Trade]          = []
        regime_log: list[RegimeLogEntry] = []
        equity_nav: dict = {}

        # Align oos_features with oos_df by index
        oos_idx = oos_df.index.intersection(oos_features.index)

        for ts in oos_idx:
            feat_row = oos_features.loc[ts]
            close    = float(oos_df.loc[ts, "close"])

            # Bar return (close-to-close); 0 on first bar
            bar_return = 0.0
            if prev_close is not None and prev_close > 0:
                bar_return = (close / prev_close) - 1.0

            # Regime prediction (forward-only)
            regime_raw  = engine.predict_current(feat_row)
            is_uncertain = engine.is_uncertain()
            regime       = max(regime_raw, 0)
            confidence   = 0.7  # simplified; could derive from HMM posteriors

            # Strategy signal
            signal = get_signal(
                regime=regime,
                confidence=confidence,
                portfolio_nav=nav,
                current_allocation=allocation,
                is_uncertain=is_uncertain,
            )

            # Risk manager gate
            approval     = risk_mgr.approve(signal, nav)
            target_alloc = (
                signal.allocation_pct * approval.size_multiplier
                if approval.approved else 0.0
            )

            # Simulate fill for this bar
            alloc_change = target_alloc - allocation
            trade_cost   = abs(alloc_change) * nav * SLIPPAGE * 2  # round-trip
            port_return  = bar_return * allocation
            new_nav      = (nav - trade_cost) * (1.0 + port_return)

            if allocation > 0.0:
                trades.append(Trade(
                    timestamp=ts,
                    regime=regime,
                    confidence=confidence,
                    pnl=new_nav - nav,
                    return_pct=port_return,
                ))

            regime_log.append(RegimeLogEntry(
                timestamp=ts,
                regime=regime_raw,
                confidence=confidence,
                is_uncertain=is_uncertain,
                allocation_pct=target_alloc,
            ))

            # Update risk manager after realising P&L
            risk_mgr.update(new_nav)

            equity_nav[ts] = new_nav
            nav        = new_nav
            allocation = target_alloc
            prev_close = close

        equity_series = pd.Series(equity_nav)
        if equity_series.empty:
            equity_series = pd.Series({oos_df.index[0]: initial_nav})

        # Benchmark curves (same OOS window, seeded from initial_nav)
        close_series = oos_df["close"]
        bmark_curves = {
            "buy_and_hold": _benchmark_buy_hold(close_series, initial_nav),
            "sma200":       _benchmark_sma200(combined["close"], close_series, initial_nav),
            "random":       _benchmark_random(close_series, initial_nav, seed=fold_idx),
        }

        # Regime distribution
        dist: dict[int, int] = {}
        for entry in regime_log:
            dist[entry.regime] = dist.get(entry.regime, 0) + 1

        return FoldResult(
            fold_idx=fold_idx,
            in_sample_start=is_df.index[0],
            in_sample_end=is_df.index[-1],
            oos_start=oos_df.index[0],
            oos_end=oos_df.index[-1],
            equity_curve=equity_series,
            trades=trades,
            regime_log=regime_log,
            benchmark_curves=bmark_curves,
            regime_distribution=dist,
            n_hmm_states=engine._n_states,
        )

    def run_stress_test(
        self,
        oos_df:      pd.DataFrame,
        n_injections: int = 3,
        initial_nav:  float = INITIAL_NAV,
    ) -> StressReport:
        """Inject synthetic stress events into a copy of oos_df and simulate.

        Scenarios injected (one each, up to n_injections total):
          1. Single -12 % crash day         (bar index 5)
          2. Five consecutive -3.5 % days   (bars 20–24)
          3. Realised-vol spike ×2          (bars 40–49 HL range doubled)

        Parameters
        ----------
        oos_df        : clean out-of-sample OHLCV.
        n_injections  : how many distinct injection events to apply.
        initial_nav   : starting NAV for the stress simulation.
        """
        stressed = oos_df.copy()
        events_applied: list[str] = []

        # Scenario 1: single -12 % crash day
        if n_injections >= 1 and len(stressed) > 10:
            idx = 5
            prev_close = float(stressed["close"].iloc[idx - 1])
            crash_close = prev_close * (1.0 - 0.12)
            stressed.iloc[idx, stressed.columns.get_loc("close")] = crash_close
            stressed.iloc[idx, stressed.columns.get_loc("low")]   = crash_close * 0.99
            events_applied.append("crash_day_-12pct")

        # Scenario 2: five consecutive -3.5 % days
        if n_injections >= 2 and len(stressed) > 30:
            for k in range(5):
                i   = 20 + k
                prv = float(stressed["close"].iloc[i - 1])
                c   = prv * (1.0 - 0.035)
                stressed.iloc[i, stressed.columns.get_loc("close")] = c
                stressed.iloc[i, stressed.columns.get_loc("low")]   = c * 0.99
            events_applied.append("consecutive_5x_-3.5pct")

        # Scenario 3: vol spike (doubled HL range)
        if n_injections >= 3 and len(stressed) > 55:
            for k in range(10):
                i    = 40 + k
                row  = stressed.iloc[i]
                mid  = (float(row["high"]) + float(row["low"])) / 2.0
                rng  = float(row["high"]) - float(row["low"])
                stressed.iloc[i, stressed.columns.get_loc("high")] = mid + rng
                stressed.iloc[i, stressed.columns.get_loc("low")]  = mid - rng
            events_applied.append("vol_spike_10bars_2x")

        # Run simplified simulation (no HMM — use fixed high allocation to stress circuit breakers)
        risk_mgr = RiskManager()
        risk_mgr.initialize(initial_nav)

        nav       = initial_nav
        all_fired: list[str] = []
        equity: dict = {}
        prev_close: Optional[float] = None

        for ts, row in stressed.iterrows():
            close = float(row["close"])
            if prev_close is not None and prev_close > 0:
                bar_ret = (close / prev_close) - 1.0
                # Simulate 90 % allocation (bull regime) throughout
                port_ret = bar_ret * 0.90
                new_nav  = nav * (1.0 + port_ret)
                fired    = risk_mgr.update(new_nav)
                all_fired.extend(fired)
                nav = new_nav
            equity[ts] = nav
            prev_close = close

        equity_series     = pd.Series(equity)
        max_dd, _         = _max_dd_simple(equity_series.values)
        return StressReport(
            circuit_breakers_fired=all_fired,
            max_drawdown=max_dd,
            equity_curve=equity_series,
            n_injections=n_injections,
            events_applied=events_applied,
        )


# ---------------------------------------------------------------------------
# Walk-forward window computation
# ---------------------------------------------------------------------------

def _compute_windows(
    total_bars: int,
    is_bars:    int,
    oos_bars:   int,
    step:       int,
) -> list[tuple[int, int, int]]:
    """Return list of (is_start, is_end, oos_end) index tuples."""
    windows: list[tuple[int, int, int]] = []
    is_start = 0
    while True:
        is_end  = is_start + is_bars
        oos_end = is_end   + oos_bars
        if oos_end > total_bars:
            break
        windows.append((is_start, is_end, oos_end))
        is_start += step
    return windows


# ---------------------------------------------------------------------------
# Benchmark simulators
# ---------------------------------------------------------------------------

def _simulate_equity(
    price: pd.Series,
    alloc: pd.Series,
    initial_nav: float,
) -> pd.Series:
    """Simulate a buy/hold/sell equity curve given an allocation signal."""
    returns = price.pct_change().fillna(0.0).values
    allocs  = alloc.values
    nav_arr = np.empty(len(price))
    nav_arr[0] = initial_nav
    for i in range(1, len(price)):
        old_a  = allocs[i - 1]
        new_a  = allocs[i]
        cost   = abs(new_a - old_a) * nav_arr[i - 1] * SLIPPAGE * 2
        ret    = returns[i] * new_a
        nav_arr[i] = (nav_arr[i - 1] - cost) * (1.0 + ret)
    return pd.Series(nav_arr, index=price.index)


def _benchmark_buy_hold(close: pd.Series, initial_nav: float) -> pd.Series:
    alloc = pd.Series(1.0, index=close.index)
    return _simulate_equity(close, alloc, initial_nav)


def _benchmark_sma200(
    full_close: pd.Series,   # IS + OOS for SMA warmup
    oos_close:  pd.Series,
    initial_nav: float,
) -> pd.Series:
    sma   = full_close.rolling(200, min_periods=1).mean()
    signal = (full_close > sma).astype(float)
    oos_signal = signal.loc[oos_close.index]
    return _simulate_equity(oos_close, oos_signal, initial_nav)


def _benchmark_random(
    close:       pd.Series,
    initial_nav: float,
    seed:        int = 0,
) -> pd.Series:
    rng   = np.random.default_rng(seed)
    alloc = rng.choice([0.0, 0.5, 1.0], size=len(close))
    return _simulate_equity(close, pd.Series(alloc, index=close.index), initial_nav)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _equity_to_trades(equity: pd.Series) -> list[Trade]:
    """Convert a benchmark equity curve to a list of Trade objects."""
    trades: list[Trade] = []
    nav = equity.values
    for i in range(1, len(nav)):
        pnl  = float(nav[i] - nav[i - 1])
        ret  = float((nav[i] / nav[i - 1]) - 1.0) if nav[i - 1] != 0 else 0.0
        trades.append(Trade(
            timestamp=equity.index[i],
            regime=2,
            confidence=0.5,
            pnl=pnl,
            return_pct=ret,
        ))
    return trades


def _max_dd_simple(nav: np.ndarray) -> tuple[float, int]:
    peak   = nav[0]
    max_dd = 0.0
    dur    = 0
    pk_idx = 0
    for i, v in enumerate(nav):
        if v >= peak:
            peak   = v
            pk_idx = i
        dd = (v - peak) / peak if peak != 0 else 0.0
        if dd < max_dd:
            max_dd = dd
            dur    = i - pk_idx
    return float(max_dd), dur

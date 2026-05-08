"""
core/walk_forward.py
--------------------
Walk-forward backtesting framework. Additive — the standard Backtester is unchanged.

Public interface:
  WalkForwardBacktester.run(ohlcv_df, n_train, n_test, step, min_folds) -> WalkForwardResult
  interpret_overfitting_ratio(ratio) -> str
"""

from __future__ import annotations

import json
import logging
import math
import pathlib
import warnings
from dataclasses import asdict, dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import HMM_STALENESS_ZSCORE_WALKFORWARD, SLIPPAGE_BPS
from core.backtester import Backtester
from core.feature_engineering import compute as compute_features
from core.hmm_engine import HMMEngine
from core.regime_strategies import get_signal
from core.risk_manager import RiskManager

log = logging.getLogger(__name__)

_LOGS_DIR    = pathlib.Path("logs") / "walk_forward"
_TRADING_DAYS = 252
_RF_ANNUAL    = 0.045


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class WalkForwardError(Exception):
    """Unrecoverable walk-forward error (e.g. failed in-sample backtest)."""


class InsufficientDataError(Exception):
    """Insufficient bars to produce the minimum required number of folds."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold_number:                      int
    train_start:                      str
    train_end:                        str
    test_start:                       str
    test_end:                         str
    n_train_bars:                     int
    n_test_bars:                      int
    sharpe_ratio:                     float
    annualized_return:                float
    max_drawdown:                     float
    win_rate:                         float
    total_trades:                     int
    regime_distribution:              dict
    avg_confirmed_regime_duration_bars: float
    per_regime_sharpe:                dict
    per_regime_return:                dict
    per_regime_mean_log_return:       dict
    slippage_cost_total:              float
    hmm_convergence_warnings:         int
    hmm_n_states_selected:            int
    pct_bars_stale:                   float


@dataclass
class WalkForwardResult:
    n_folds:                    int
    total_bars_tested:          int
    test_window_overlap_bars:   int
    date_range_tested:          str

    mean_sharpe:                float
    std_sharpe:                 float
    min_sharpe:                 float
    max_sharpe:                 float
    median_sharpe:              float
    pct_folds_positive_sharpe:  float

    mean_max_drawdown:          float
    worst_max_drawdown:         float
    worst_drawdown_fold:        int

    mean_annualized_return:     float
    std_annualized_return:      float

    insample_sharpe:              float
    outsample_sharpe:             float
    overfitting_ratio:            float
    overfitting_verdict:          str
    ratio_interpretation_note:    str

    regime_label_consistency:   float
    mean_hmm_states_selected:   float
    std_hmm_states_selected:    float
    mean_convergence_warnings:  float
    folds_with_stale_model:     int
    mean_pct_bars_stale:        float
    max_pct_bars_stale:         float
    max_pct_stale_fold:         int

    return_distribution:        list
    drawdown_distribution:      list
    sharpe_distribution:        list
    per_regime_returns:         dict
    regime_transition_matrix:   dict

    folds:                      list


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def interpret_overfitting_ratio(ratio: float) -> str:
    if ratio <= 0:
        return (
            "🚨 Invalid — out-of-sample Sharpe is negative while in-sample is positive "
            "(or both negative with wrong sign). Strategy collapses out-of-sample."
        )
    if ratio > 1.0:
        return (
            f"🚨 Severe degradation — out-of-sample is {ratio:.2f}x worse than in-sample. "
            "Do not deploy."
        )
    if ratio >= 0.85:
        return "✅ Minimal overfitting — strategy is robust"
    if ratio >= 0.70:
        return "⚠️  Moderate degradation — acceptable but monitor closely"
    if ratio >= 0.50:
        return "🔴 Significant overfitting — do not deploy until investigated"
    return "🚨 Severe overfitting — strategy does not generalize"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class WalkForwardBacktester:

    def run(
        self,
        ohlcv_df:  pd.DataFrame,
        n_train:   int = 504,
        n_test:    int = 126,
        step:      int = 63,
        min_folds: int = 3,
    ) -> WalkForwardResult:
        """Run walk-forward backtest.

        Parameters
        ----------
        ohlcv_df  : OHLCV DataFrame with DatetimeIndex, sorted ascending.
        n_train   : training window in bars (~2 years daily).
        n_test    : test window in bars (~6 months daily).
        step      : roll-forward step in bars (~3 months daily).
        min_folds : minimum folds required; raises InsufficientDataError otherwise.
        """
        # Check feasibility before expensive in-sample run
        windows = _compute_wf_windows(len(ohlcv_df), n_train, n_test, step)
        if len(windows) < min_folds:
            raise InsufficientDataError(
                f"Insufficient data: {len(ohlcv_df)} bars produce {len(windows)} folds "
                f"(minimum {min_folds} required with n_train={n_train}, "
                f"n_test={n_test}, step={step})"
            )

        # 0. Fresh in-sample backtest for overfitting ratio denominator
        try:
            is_report     = Backtester().run(ohlcv_df, audit_lookahead=False, disable_staleness=True)
            insample_sharpe = float(is_report.performance.sharpe)
        except Exception as exc:
            raise WalkForwardError(
                "in-sample backtest failed — cannot compute overfitting ratio"
            ) from exc

        # Walk-forward loop
        folds: list[FoldResult]    = []
        all_returns:   list[float] = []
        all_drawdowns: list[float] = []
        all_regimes:   list[int]   = []
        per_regime_all: dict[int, list[float]] = {}
        folds_stale = 0

        for fold_num, (tr_start, tr_end, te_end) in enumerate(windows):
            train_df = ohlcv_df.iloc[tr_start:tr_end]
            test_df  = ohlcv_df.iloc[tr_end:te_end]

            fold_result, fold_rets, fold_dds, fold_regimes, fold_pr_rets, is_stale = (
                _run_fold(fold_num, train_df, test_df)
            )
            folds.append(fold_result)
            all_returns.extend(fold_rets)
            all_drawdowns.extend(fold_dds)
            all_regimes.extend(fold_regimes)
            for r, rets in fold_pr_rets.items():
                per_regime_all.setdefault(r, []).extend(rets)
            if is_stale:
                folds_stale += 1

        # Aggregate metrics
        sharpes  = [f.sharpe_ratio for f in folds]
        returns  = [f.annualized_return for f in folds]
        drawdowns = [f.max_drawdown for f in folds]

        outsample_sharpe = float(np.mean(sharpes))

        if insample_sharpe == 0.0:
            log.warning("WalkForward: insample_sharpe=0.0 — overfitting_ratio set to 0.0")
            overfitting_ratio = 0.0
        else:
            overfitting_ratio = outsample_sharpe / insample_sharpe

        overfitting_verdict = interpret_overfitting_ratio(overfitting_ratio)
        ratio_interpretation_note = (
            f"Out-of-sample Sharpe ({outsample_sharpe:.3f}) is "
            f"{overfitting_ratio:.1%} of in-sample Sharpe ({insample_sharpe:.3f})"
        )

        # Regime stability
        consistency = _compute_regime_consistency(folds)
        hmm_states  = [f.hmm_n_states_selected for f in folds]
        conv_warns  = [f.hmm_convergence_warnings for f in folds]

        # Date range tested
        test_first = ohlcv_df.index[windows[0][1]]
        test_last  = ohlcv_df.index[windows[-1][2] - 1]
        date_range = (
            f"{test_first.date().isoformat()} to {test_last.date().isoformat()}"
        )

        pct_stale_vals   = [f.pct_bars_stale for f in folds]
        mean_pct_stale   = float(np.mean(pct_stale_vals))
        max_pct_stale    = float(np.max(pct_stale_vals))
        max_pct_stale_fold = int(np.argmax(pct_stale_vals))

        worst_dd_fold = int(np.argmin(drawdowns))

        # Transition matrix (string keys for JSON)
        transition_matrix = _compute_transition_matrix(all_regimes)

        # Serialize per_regime_returns with string keys
        per_regime_returns_str = {str(k): v for k, v in per_regime_all.items()}

        result = WalkForwardResult(
            n_folds=len(folds),
            total_bars_tested=sum(f.n_test_bars for f in folds),
            test_window_overlap_bars=n_test - step,
            date_range_tested=date_range,

            mean_sharpe=float(np.mean(sharpes)),
            std_sharpe=float(np.std(sharpes)),
            min_sharpe=float(np.min(sharpes)),
            max_sharpe=float(np.max(sharpes)),
            median_sharpe=float(np.median(sharpes)),
            pct_folds_positive_sharpe=float(np.mean([s > 0 for s in sharpes])),

            mean_max_drawdown=float(np.mean(drawdowns)),
            worst_max_drawdown=float(np.min(drawdowns)),
            worst_drawdown_fold=worst_dd_fold,

            mean_annualized_return=float(np.mean(returns)),
            std_annualized_return=float(np.std(returns)),

            insample_sharpe=insample_sharpe,
            outsample_sharpe=outsample_sharpe,
            overfitting_ratio=overfitting_ratio,
            overfitting_verdict=overfitting_verdict,
            ratio_interpretation_note=ratio_interpretation_note,

            regime_label_consistency=consistency,
            mean_hmm_states_selected=float(np.mean(hmm_states)),
            std_hmm_states_selected=float(np.std(hmm_states)),
            mean_convergence_warnings=float(np.mean(conv_warns)),
            folds_with_stale_model=folds_stale,
            mean_pct_bars_stale=mean_pct_stale,
            max_pct_bars_stale=max_pct_stale,
            max_pct_stale_fold=max_pct_stale_fold,

            return_distribution=[float(r) for r in all_returns],
            drawdown_distribution=[float(d) for d in all_drawdowns],
            sharpe_distribution=[float(s) for s in sharpes],
            per_regime_returns=per_regime_returns_str,
            regime_transition_matrix=transition_matrix,

            folds=folds,
        )

        json_path, md_path = _write_output(result)
        _fire_alert(result, json_path)

        log.info(
            "WalkForward complete: %d folds | mean_sharpe=%.3f | "
            "overfitting_ratio=%.3f | %s",
            result.n_folds, result.mean_sharpe,
            result.overfitting_ratio, result.overfitting_verdict,
        )
        return result


# ---------------------------------------------------------------------------
# Window computation
# ---------------------------------------------------------------------------

def _compute_wf_windows(
    total_bars: int,
    n_train:    int,
    n_test:     int,
    step:       int,
) -> list[tuple[int, int, int]]:
    """Return list of (train_start, train_end, test_end) index tuples."""
    windows: list[tuple[int, int, int]] = []
    train_start = 0
    while True:
        train_end = train_start + n_train
        test_end  = train_end  + n_test
        if test_end > total_bars:
            break
        windows.append((train_start, train_end, test_end))
        train_start += step
    return windows


# ---------------------------------------------------------------------------
# Fold simulation
# ---------------------------------------------------------------------------

def _run_fold(
    fold_num:  int,
    train_df:  pd.DataFrame,
    test_df:   pd.DataFrame,
) -> tuple[FoldResult, list[float], list[float], list[int], dict[int, list[float]], bool]:
    """Train HMM on train_df, simulate strategy on test_df.

    Returns (FoldResult, portfolio_returns, rolling_drawdowns,
             regime_sequence, per_regime_portfolio_returns, has_stale_model)
    """
    # Feature computation: combine for rolling warmup, fit only on training slice
    combined      = pd.concat([train_df, test_df])
    features_all  = compute_features(combined)

    n_tr = len(train_df)
    features_train = features_all.iloc[:n_tr].dropna()
    features_test  = features_all.iloc[n_tr:].dropna()

    if len(features_train) < 30:
        raise WalkForwardError(
            f"Fold {fold_num}: insufficient IS features after dropna ({len(features_train)})."
        )

    # Fit a completely fresh HMMEngine on the training slice only
    engine = HMMEngine()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.fit(features_train)
    n_conv_warnings = len(caught)
    n_states        = engine._n_states

    # Simulate strategy bar-by-bar on the test slice
    risk_mgr = RiskManager()
    risk_mgr.initialize(100_000.0)

    nav        = 100_000.0
    allocation = 0.0
    prev_close: Optional[float] = None
    equity_nav: dict             = {}

    regimes_seq:     list[int]   = []
    portfolio_rets:  list[float] = []
    slippage_total               = 0.0
    n_trades                     = 0
    win_count                    = 0
    has_stale                    = False
    stale_bar_count              = 0

    per_regime_port: dict[int, list[float]] = {}
    per_regime_logr: dict[int, list[float]] = {}

    oos_idx = test_df.index.intersection(features_test.index)

    for ts in oos_idx:
        feat_row  = features_test.loc[ts]
        close     = float(test_df.loc[ts, "close"])

        bar_log_r = 0.0
        bar_ret   = 0.0
        if prev_close is not None and prev_close > 0:
            bar_log_r = math.log(close / prev_close)
            bar_ret   = math.exp(bar_log_r) - 1.0

        regime_raw   = engine.predict_current(feat_row, staleness_zscore=HMM_STALENESS_ZSCORE_WALKFORWARD)
        is_uncertain = engine.is_uncertain()

        if engine._is_model_stale:
            has_stale = True
            stale_bar_count += 1

        regimes_seq.append(regime_raw)

        regime_for_signal = max(regime_raw, 0) if regime_raw != -1 else 2
        confidence        = 0.7 if engine.is_confirmed() and not is_uncertain else 0.5

        signal    = get_signal(
            regime=regime_for_signal,
            confidence=confidence,
            portfolio_nav=nav,
            current_allocation=allocation,
            is_uncertain=is_uncertain,
        )
        approval  = risk_mgr.approve(signal, nav)
        tgt_alloc = (
            signal.allocation_pct * approval.size_multiplier
            if approval.approved else 0.0
        )

        alloc_change = tgt_alloc - allocation
        commission   = abs(alloc_change) * nav * SLIPPAGE_BPS / 10_000.0
        port_ret     = bar_ret * allocation
        new_nav      = (nav - commission) * (1.0 + port_ret)
        slippage_total += commission

        if allocation > 0.0:
            n_trades += 1
            if port_ret > 0:
                win_count += 1

        portfolio_rets.append(port_ret)

        if regime_raw != -1:
            per_regime_port.setdefault(regime_raw, []).append(port_ret)
            if prev_close is not None:
                per_regime_logr.setdefault(regime_raw, []).append(bar_log_r)

        risk_mgr.update(new_nav)
        equity_nav[ts] = new_nav
        nav        = new_nav
        allocation = tgt_alloc
        prev_close = close

    pct_bars_stale = stale_bar_count / len(oos_idx) if len(oos_idx) > 0 else 0.0
    if pct_bars_stale > 0.5:
        log.warning(
            "[WalkForward] Fold %d: %s of test bars flagged as stale — "
            "staleness threshold may still need widening",
            fold_num, f"{pct_bars_stale:.0%}",
        )

    equity_series = pd.Series(equity_nav)
    nav_arr = equity_series.values if len(equity_series) > 0 else np.array([100_000.0])
    rets_arr = (
        np.diff(nav_arr) / nav_arr[:-1]
        if len(nav_arr) > 1 else np.array([])
    )

    sharpe_r   = _wf_sharpe(rets_arr)
    ann_ret    = _wf_ann_return(nav_arr)
    max_dd     = _wf_max_drawdown(nav_arr)
    rolling_dd = list(_wf_rolling_drawdown(nav_arr))
    win_rate   = win_count / n_trades if n_trades > 0 else 0.0

    # Regime distribution (confirmed bars only, as fraction)
    confirmed_bars = [r for r in regimes_seq if r != -1]
    regime_dist: dict = {}
    if confirmed_bars:
        n_conf = len(confirmed_bars)
        for r in set(confirmed_bars):
            regime_dist[r] = confirmed_bars.count(r) / n_conf

    avg_duration = _avg_confirmed_regime_duration(regimes_seq)

    # Per-regime metrics
    per_regime_sharpe: dict = {}
    per_regime_return: dict = {}
    per_regime_mean_logr: dict = {}

    for r, rets in per_regime_port.items():
        arr = np.array(rets)
        per_regime_sharpe[r] = _wf_sharpe(arr)
        per_regime_return[r] = _wf_regime_ann_return(arr)

    for r, lrs in per_regime_logr.items():
        mlr = float(np.mean(lrs))
        per_regime_mean_logr[r] = mlr
        log.debug(
            "WF fold %d regime %d mean_log_return=%.6f n=%d",
            fold_num, r, mlr, len(lrs),
        )

    fold_result = FoldResult(
        fold_number=fold_num,
        train_start=train_df.index[0].date().isoformat(),
        train_end=train_df.index[-1].date().isoformat(),
        test_start=test_df.index[0].date().isoformat(),
        test_end=test_df.index[-1].date().isoformat(),
        n_train_bars=len(train_df),
        n_test_bars=len(test_df),
        sharpe_ratio=sharpe_r,
        annualized_return=ann_ret,
        max_drawdown=max_dd,
        win_rate=win_rate,
        total_trades=n_trades,
        regime_distribution=regime_dist,
        avg_confirmed_regime_duration_bars=avg_duration,
        per_regime_sharpe=per_regime_sharpe,
        per_regime_return=per_regime_return,
        per_regime_mean_log_return=per_regime_mean_logr,
        slippage_cost_total=slippage_total,
        hmm_convergence_warnings=n_conv_warnings,
        hmm_n_states_selected=n_states,
        pct_bars_stale=pct_bars_stale,
    )

    return fold_result, portfolio_rets, rolling_dd, regimes_seq, per_regime_port, has_stale


# ---------------------------------------------------------------------------
# Regime consistency
# ---------------------------------------------------------------------------

def _compute_regime_consistency(folds: list[FoldResult]) -> float:
    """Fraction of (label, fold-pair) combinations that are consistent.

    Consistent = same sign of mean_log_return AND |diff| < 0.5 * cross-fold std.
    """
    all_labels: set[int] = set()
    for f in folds:
        all_labels.update(int(k) for k in f.per_regime_mean_log_return)

    total_pairs     = 0
    consistent_pairs = 0

    for label in sorted(all_labels):
        values: list[float] = []
        for f in folds:
            v = f.per_regime_mean_log_return.get(label)
            if v is None:
                v = f.per_regime_mean_log_return.get(str(label))
            if v is not None:
                values.append(float(v))

        if len(values) < 2:
            continue

        cross_std = float(np.std(values, ddof=1))
        threshold = 0.5 * cross_std

        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                total_pairs += 1
                same_sign = (values[i] >= 0) == (values[j] >= 0)
                mag_diff  = abs(values[i] - values[j])
                if same_sign and (threshold == 0.0 or mag_diff < threshold):
                    consistent_pairs += 1

    return float(consistent_pairs / total_pairs) if total_pairs > 0 else 1.0


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------

def _compute_transition_matrix(regimes: list[int]) -> dict:
    """Empirical P(j | i) from a sequence of confirmed regimes. Rows sum to 1.0."""
    confirmed = [r for r in regimes if r != -1]
    counts: dict[int, dict[int, int]] = {}
    for a, b in zip(confirmed, confirmed[1:]):
        counts.setdefault(a, {})
        counts[a][b] = counts[a].get(b, 0) + 1

    matrix: dict[str, dict[str, float]] = {}
    for from_r, to_counts in counts.items():
        total = sum(to_counts.values())
        matrix[str(from_r)] = {
            str(to_r): cnt / total for to_r, cnt in to_counts.items()
        }
    return matrix


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _rf_daily() -> float:
    return (1.0 + _RF_ANNUAL) ** (1.0 / _TRADING_DAYS) - 1.0


def _wf_sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - _rf_daily()
    std = float(np.std(excess, ddof=1))
    return 0.0 if std == 0.0 else float(np.mean(excess) / std * math.sqrt(_TRADING_DAYS))


def _wf_ann_return(nav: np.ndarray) -> float:
    if len(nav) < 2 or nav[0] == 0:
        return 0.0
    total_r = nav[-1] / nav[0] - 1.0
    n = len(nav)
    return float((1.0 + total_r) ** (_TRADING_DAYS / n) - 1.0)


def _wf_regime_ann_return(returns: np.ndarray) -> float:
    """Annualized return from a sequence of per-bar portfolio returns."""
    if len(returns) == 0:
        return 0.0
    compound = float(np.prod(1.0 + returns))
    n = len(returns)
    return float(compound ** (_TRADING_DAYS / n) - 1.0)


def _wf_max_drawdown(nav: np.ndarray) -> float:
    peak   = nav[0]
    max_dd = 0.0
    for v in nav:
        if v >= peak:
            peak = v
        dd = (v - peak) / peak if peak != 0 else 0.0
        if dd < max_dd:
            max_dd = dd
    return float(max_dd)


def _wf_rolling_drawdown(nav: np.ndarray) -> np.ndarray:
    dd   = np.zeros(len(nav))
    peak = nav[0]
    for i, v in enumerate(nav):
        if v >= peak:
            peak = v
        dd[i] = (v - peak) / peak if peak != 0 else 0.0
    return dd


def _avg_confirmed_regime_duration(regimes: list[int]) -> float:
    """Average bars per confirmed-regime run, excluding unconfirmed (-1) bars."""
    durations: list[int] = []
    current  = None
    count    = 0
    for r in regimes:
        if r == -1:
            if current is not None and count > 0:
                durations.append(count)
            current = None
            count   = 0
        elif r == current:
            count += 1
        else:
            if current is not None and count > 0:
                durations.append(count)
            current = r
            count   = 1
    if current is not None and count > 0:
        durations.append(count)
    return float(np.mean(durations)) if durations else 0.0


# ---------------------------------------------------------------------------
# Output: JSON + Markdown
# ---------------------------------------------------------------------------

def _write_output(result: WalkForwardResult) -> tuple[pathlib.Path, pathlib.Path]:
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    today     = date.today().isoformat()
    json_path = _LOGS_DIR / f"walk_forward_{today}.json"
    md_path   = _LOGS_DIR / f"walk_forward_{today}.md"

    # ── JSON ────────────────────────────────────────────────────────────────
    def _serialise(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serialisable: {type(obj)}")

    payload = {
        "date":     today,
        "n_folds":  result.n_folds,
        "date_range_tested": result.date_range_tested,
        "aggregate": {
            "mean_sharpe":               result.mean_sharpe,
            "std_sharpe":                result.std_sharpe,
            "min_sharpe":                result.min_sharpe,
            "max_sharpe":                result.max_sharpe,
            "median_sharpe":             result.median_sharpe,
            "pct_folds_positive_sharpe": result.pct_folds_positive_sharpe,
            "mean_max_drawdown":         result.mean_max_drawdown,
            "worst_max_drawdown":        result.worst_max_drawdown,
            "worst_drawdown_fold":       result.worst_drawdown_fold,
            "mean_annualized_return":    result.mean_annualized_return,
            "std_annualized_return":     result.std_annualized_return,
        },
        "overfitting": {
            "insample_sharpe":   result.insample_sharpe,
            "outsample_sharpe":  result.outsample_sharpe,
            "overfitting_ratio": result.overfitting_ratio,
            "verdict":           result.overfitting_verdict,
        },
        "regime_stability": {
            "regime_label_consistency":  result.regime_label_consistency,
            "mean_hmm_states_selected":  result.mean_hmm_states_selected,
            "std_hmm_states_selected":   result.std_hmm_states_selected,
            "mean_convergence_warnings": result.mean_convergence_warnings,
            "folds_with_stale_model":    result.folds_with_stale_model,
        },
        "monte_carlo_inputs": {
            "return_distribution":      result.return_distribution,
            "drawdown_distribution":    result.drawdown_distribution,
            "sharpe_distribution":      result.sharpe_distribution,
            "per_regime_returns":       result.per_regime_returns,
            "regime_transition_matrix": result.regime_transition_matrix,
            "test_window_overlap_bars": result.test_window_overlap_bars,
        },
        "folds": [asdict(f) for f in result.folds],
    }
    json_path.write_text(
        json.dumps(payload, indent=2, default=_serialise), encoding="utf-8"
    )
    log.info("WalkForward: JSON written to %s", json_path)

    # ── Markdown ─────────────────────────────────────────────────────────────
    md_path.write_text(_render_markdown(result, today), encoding="utf-8")
    log.info("WalkForward: Markdown written to %s", md_path)

    return json_path, md_path


def _render_markdown(result: WalkForwardResult, today: str) -> str:
    lines: list[str] = []
    r = result

    lines += [
        f"# Walk-Forward Backtest Report — {today}",
        "",
        f"**Date range tested:** {r.date_range_tested}  ",
        f"**Folds:** {r.n_folds}  ",
        f"**Total bars tested:** {r.total_bars_tested}  ",
        f"**Test window overlap:** {r.test_window_overlap_bars} bars  ",
        "",
        (
            f"> Note: consecutive test windows overlap by {r.test_window_overlap_bars} bars "
            f"— fold results are not fully independent"
        ),
        "",
    ]

    # Aggregate performance
    lines += [
        "## Aggregate Performance",
        "",
        "```",
        f"  Mean Sharpe:               {r.mean_sharpe:>8.3f}",
        f"  Std Sharpe:                {r.std_sharpe:>8.3f}",
        f"  Min / Max Sharpe:          {r.min_sharpe:>8.3f}  /  {r.max_sharpe:.3f}",
        f"  Median Sharpe:             {r.median_sharpe:>8.3f}",
        f"  % Folds Positive Sharpe:   {r.pct_folds_positive_sharpe * 100:>7.1f}%",
        f"  Mean Ann. Return:          {r.mean_annualized_return * 100:>7.1f}%",
        f"  Std Ann. Return:           {r.std_annualized_return * 100:>7.1f}%",
        f"  Mean Max Drawdown:         {r.mean_max_drawdown * 100:>7.1f}%",
        f"  Worst Max Drawdown:        {r.worst_max_drawdown * 100:>7.1f}%  (fold {r.worst_drawdown_fold})",
        "```",
        "",
    ]

    # Overfitting analysis
    lines += [
        "## Overfitting Analysis",
        "",
        "```",
        f"  In-sample Sharpe:          {r.insample_sharpe:>8.3f}",
        f"  Out-of-sample Sharpe:      {r.outsample_sharpe:>8.3f}",
        f"  Overfitting Ratio:         {r.overfitting_ratio:>8.4f}",
        "```",
        "",
        f"**Verdict:** {r.overfitting_verdict}",
        "",
    ]

    # Regime stability
    lines += [
        "## Regime Stability",
        "",
        "```",
        f"  Regime Label Consistency:  {r.regime_label_consistency * 100:>7.1f}%",
        f"  Mean HMM States Selected:  {r.mean_hmm_states_selected:>8.2f}",
        f"  Std  HMM States Selected:  {r.std_hmm_states_selected:>8.2f}",
        f"  Mean Convergence Warnings: {r.mean_convergence_warnings:>8.2f}",
        f"  Folds with Stale Model:    {r.folds_with_stale_model:>8d}",
        f"  Mean % Bars Stale:         {r.mean_pct_bars_stale * 100:>7.1f}%",
        f"  Max  % Bars Stale:         {r.max_pct_bars_stale * 100:>7.1f}%  (fold {r.max_pct_stale_fold})",
        "```",
        "",
    ]

    # Per-regime returns summary
    if r.per_regime_returns:
        lines += ["## Per-Regime Returns (All Folds)", ""]
        lines += [
            "| Regime | N Bars | Mean Return | Std Return |",
            "|--------|--------|-------------|------------|",
        ]
        _NAMES = {0: "crash", 1: "bear", 2: "neutral", 3: "bull", 4: "euphoria"}
        for label_str, rets in sorted(r.per_regime_returns.items()):
            label    = int(label_str)
            name     = _NAMES.get(label, f"state_{label}")
            arr      = np.array(rets)
            mean_r   = float(np.mean(arr)) * 100
            std_r    = float(np.std(arr)) * 100
            n        = len(arr)
            lines.append(f"| {label} ({name}) | {n} | {mean_r:.4f}% | {std_r:.4f}% |")
        lines.append("")

    # Regime transition matrix
    if r.regime_transition_matrix:
        _NAMES = {0: "crash", 1: "bear", 2: "neutral", 3: "bull", 4: "euphoria"}
        states = sorted(r.regime_transition_matrix.keys(), key=int)
        col_labels = [f"{s} ({_NAMES.get(int(s), '?')})" for s in states]
        lines += ["## Regime Transition Matrix", ""]
        header = "| From \\ To | " + " | ".join(col_labels) + " |"
        sep    = "|-----------|" + "|".join("-" * (len(c) + 2) for c in col_labels) + "|"
        lines += [header, sep]
        for from_s in states:
            to_row  = r.regime_transition_matrix[from_s]
            row_lbl = f"{from_s} ({_NAMES.get(int(from_s), '?')})"
            cells   = " | ".join(
                f"{to_row.get(to_s, 0.0):.2f}" for to_s in states
            )
            lines.append(f"| {row_lbl} | {cells} |")
        lines.append("")

    # Per-fold summary
    lines += [
        "## Per-Fold Summary",
        "",
        f"> Note: consecutive test windows overlap by {r.test_window_overlap_bars} bars"
        " — fold results are not fully independent",
        "",
        "| Fold | Train Start | Test Start | Test End | Sharpe | Ann Ret | Max DD | Trades | States | Stale% |",
        "|------|-------------|------------|----------|--------|---------|--------|--------|--------|--------|",
    ]
    for f in r.folds:
        lines.append(
            f"| {f.fold_number} | {f.train_start} | {f.test_start} | {f.test_end} "
            f"| {f.sharpe_ratio:.3f} | {f.annualized_return * 100:.1f}% "
            f"| {f.max_drawdown * 100:.1f}% | {f.total_trades} | {f.hmm_n_states_selected} "
            f"| {f.pct_bars_stale:.0%} |"
        )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------

def _fire_alert(result: WalkForwardResult, json_path: pathlib.Path) -> None:
    worst_fold    = result.folds[result.worst_drawdown_fold]
    worst_range   = f"{worst_fold.test_start} to {worst_fold.test_end}"
    msg = (
        f"Walk-Forward Complete | "
        f"mean_sharpe={result.mean_sharpe:.3f} | "
        f"overfitting_ratio={result.overfitting_ratio:.3f} | "
        f"verdict={result.overfitting_verdict} | "
        f"worst_fold_sharpe={result.min_sharpe:.3f} ({worst_range}) | "
        f"output={json_path}"
    )
    try:
        from core import alerts
        alerts.send("walkforward_complete", msg, severity="info")
    except Exception as exc:
        log.warning("WalkForward: alert send failed (non-fatal): %s", exc)

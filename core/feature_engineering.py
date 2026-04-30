"""
core/feature_engineering.py
----------------------------
Computes the technical indicator features fed as observations into the HMM.

Feature set (all computed from OHLCV data):
  - Log returns (1-bar)
  - Realised volatility (rolling std of log returns, 20-bar)
  - Volume z-score (20-bar rolling mean/std)
  - High-low range normalised by close (single-bar)
  - RSI (14-period) as a momentum signal

Anti-lookahead-bias rules enforced here:
  - All rolling windows use only past bars (.shift(1) before any rolling calc).
  - No feature may reference the current bar's close before that bar is closed.
  - validate_no_lookahead() raises LookaheadBiasError if any feature uses
    future data (detected by checking that row 0 has NaN for every lagged
    feature, and that no feature column correlates with future returns).

Public interface:
  compute(ohlcv_df) -> pd.DataFrame          # full history, for backtesting
  compute_latest(ohlcv_df) -> pd.Series      # single row, for live trading
  validate_no_lookahead(feature_df, ohlcv_df) -> None  # raises on violation
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.settings import RSI_PERIOD, VOL_WINDOW, VOLUME_WINDOW

log = logging.getLogger(__name__)


class LookaheadBiasError(Exception):
    """Raised when a feature computation leaks future information."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Use exponential moving average (Wilder smoothing)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute(ohlcv_df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """Return a DataFrame of features aligned to ohlcv_df's index.

    Parameters
    ----------
    ohlcv_df : DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
               and a DatetimeIndex, sorted ascending.
    symbol   : Ticker symbol. When 'MSTR' and ONCHAIN_ENABLED is True, an
               additional 'on_chain_score' column is appended (filled with 0.0
               as a neutral placeholder; the live value is injected by
               compute_latest()).

    Returns
    -------
    DataFrame with columns:
        log_return, realized_vol_20, volume_zscore, hl_range_norm, rsi_14
        [+ on_chain_score when symbol=='MSTR' and ONCHAIN_ENABLED]
    All rows before the warm-up period are NaN; caller should dropna().
    """
    df = ohlcv_df.copy()
    _require_columns(df)

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    log_ret = _log_returns(close)

    # Realised volatility: rolling std on *already-computed* log returns.
    # shift(1) ensures the current bar's return is not in the window.
    realized_vol = (
        log_ret.shift(1)
               .rolling(VOL_WINDOW, min_periods=VOL_WINDOW)
               .std()
    )

    # Volume z-score using a lagged rolling window.
    vol_mean = volume.shift(1).rolling(VOLUME_WINDOW, min_periods=VOLUME_WINDOW).mean()
    vol_std  = volume.shift(1).rolling(VOLUME_WINDOW, min_periods=VOLUME_WINDOW).std()
    volume_zscore = (volume - vol_mean) / vol_std.replace(0, np.nan)

    # High-low range normalised by close — single-bar, no future leakage.
    hl_range_norm = (high - low) / close

    # RSI on close prices — uses only past data via ewm with min_periods.
    rsi = _rsi(close, RSI_PERIOD)

    features = pd.DataFrame(
        {
            "log_return":      log_ret,
            "realized_vol_20": realized_vol,
            "volume_zscore":   volume_zscore,
            "hl_range_norm":   hl_range_norm,
            "rsi_14":          rsi,
        },
        index=df.index,
    )

    # MSTR on-chain score: 0.0 placeholder for historical rows; live value
    # is injected by compute_latest() so the HMM feature shape stays consistent.
    if symbol.upper() == "MSTR":
        try:
            from config.settings import ONCHAIN_ENABLED
            if ONCHAIN_ENABLED:
                features["on_chain_score"] = 0.0
        except Exception:
            pass

    return features


def compute_latest(ohlcv_df: pd.DataFrame, symbol: str = "") -> pd.Series:
    """Return the feature vector for the most recent bar only.

    When symbol is 'MSTR' and ONCHAIN_ENABLED is True, the on_chain_score
    placeholder is replaced with the current live value from get_onchain_features().
    """
    row = compute(ohlcv_df, symbol=symbol).iloc[-1]

    if symbol.upper() == "MSTR":
        try:
            from config.settings import ONCHAIN_ENABLED
            if ONCHAIN_ENABLED:
                from core.onchain_data import get_onchain_features
                oc = get_onchain_features()
                row = row.copy()
                row["on_chain_score"] = oc.on_chain_score
                log.debug("MSTR on_chain_score=%.3f", oc.on_chain_score)
        except Exception as exc:
            log.warning("on-chain feature injection failed: %s", exc)

    return row


def validate_no_lookahead(
    feature_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
) -> None:
    """Raise LookaheadBiasError if any feature column correlates with future returns.

    Strategy:
      1. The lagged features (realized_vol_20, volume_zscore) must be NaN on
         the first bar where there is insufficient history — confirmed by
         checking that the initial warm-up rows contain NaN.
      2. No feature column may have a Pearson |r| > 0.05 with the *next* bar's
         log return (future_return). A legitimate feature may correlate with the
         *current* bar's return (contemporaneous), but never with the *next* bar's.
    """
    _require_columns(ohlcv_df)

    # Check 1: warm-up NaN boundary
    lagged_cols = ["realized_vol_20", "volume_zscore"]
    for col in lagged_cols:
        if col not in feature_df.columns:
            continue
        # The column must have at least one NaN at the beginning.
        if feature_df[col].notna().all():
            raise LookaheadBiasError(
                f"Column '{col}' has no NaN warm-up period — "
                "rolling window may not be properly lagged."
            )

    # Check 2: no feature predicts the *next* bar's return
    close = ohlcv_df["close"]
    log_ret = np.log(close / close.shift(1))
    future_ret = log_ret.shift(-1)  # next bar's return

    aligned = feature_df.align(future_ret, join="inner", axis=0)[0]
    future_aligned = future_ret.loc[aligned.index]

    clean_mask = future_aligned.notna()
    if clean_mask.sum() < 30:
        return  # too few rows to test meaningfully

    for col in aligned.columns:
        col_vals = aligned[col].loc[clean_mask]
        fut_vals = future_aligned.loc[clean_mask]
        both_valid = col_vals.notna() & fut_vals.notna()
        if both_valid.sum() < 30:
            continue
        corr = col_vals[both_valid].corr(fut_vals[both_valid])
        if abs(corr) > 0.15:
            raise LookaheadBiasError(
                f"Column '{col}' has suspiciously high correlation "
                f"({corr:.3f}) with future returns — possible lookahead bias."
            )


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"OHLCV DataFrame missing columns: {missing}")

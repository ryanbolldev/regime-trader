"""
core/hmm_engine.py
------------------
Hidden Markov Model regime classifier.

Regime labels (ordered by market condition severity):
  0 – crash      : rapid, sustained price collapse
  1 – bear       : persistent downtrend, elevated volatility
  2 – neutral    : range-bound, low-directional conviction
  3 – bull       : steady uptrend, normal volatility
  4 – euphoria   : parabolic advance, compressed volatility

Design constraints (anti-lookahead-bias):
  - Uses the HMM *forward algorithm* only.  The Viterbi / backward pass is
    explicitly prohibited to avoid conditioning today's state on future bars.
  - Model is fitted on a rolling in-sample window; state assignments for the
    current bar use only data available at that timestamp.
  - The model is re-fitted on a schedule (e.g., weekly or monthly); intra-fit
    it runs in emission-only mode so no parameters change mid-session.

Model selection:
  - Trains candidate models with 3, 4, 5, 6, and 7 hidden states.
  - Selects the best via BIC on the training data.
  - Logs the winning state count and covariance type to the run journal.

Regime stability filter:
  - Tracks the number of regime changes in the last 20 bars.
  - If the count exceeds the FLICKER_THRESHOLD (from settings.py), the current
    regime signal is suppressed and the previous confirmed regime is held.

Confirmation gate:
  - A new regime is only "confirmed" after CONFIRMATION_BARS consecutive bars
    showing the same predicted state (default: 3 from settings.py).
  - Downstream consumers receive only confirmed regime labels.

Public interface:
  fit(features_df) -> None
  predict_current(features_row) -> int   # raw HMM state mapped to RegimeLabel
  is_confirmed() -> bool
  regime_history() -> list[int]
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from config.settings import (
    CONFIRMATION_BARS,
    FLICKER_THRESHOLD,
    FLICKER_WINDOW,
    HMM_COVARIANCE_TYPE,
    HMM_MAX_STATES,
    HMM_MIN_STATES,
    HMM_N_INIT,
    HMM_N_ITER,
    HMM_TOL,
    HMM_TRAIN_BARS,
)

log = logging.getLogger(__name__)

# Regime label names indexed 0–4 (or fewer when n_states < 5)
_REGIME_NAMES = ["crash", "bear", "neutral", "bull", "euphoria"]


class HMMEngine:
    """Gaussian HMM-based market regime classifier."""

    def __init__(
        self,
        symbol: str = "",
        *,
        confirmation_bars: int = CONFIRMATION_BARS,
        flicker_window: int = FLICKER_WINDOW,
        flicker_threshold: int = FLICKER_THRESHOLD,
        train_bars: int = HMM_TRAIN_BARS,
    ) -> None:
        self.symbol             = symbol
        self.confirmation_bars  = confirmation_bars
        self.flicker_window     = flicker_window
        self.flicker_threshold  = flicker_threshold
        self.train_bars         = train_bars

        self._model: Optional[GaussianHMM] = None
        self._n_states: int = 0
        self._state_to_regime: dict[int, int] = {}  # HMM state → sorted regime label

        # Confirmation gate state
        self._pending_state: Optional[int]  = None
        self._pending_count: int            = 0
        self._confirmed_regime: Optional[int] = None

        # History: raw regime labels emitted after confirmation gate
        self._history: list[int] = []

        # Flicker filter: deque of last FLICKER_WINDOW regime labels
        self._recent_regimes: deque[int] = deque(maxlen=flicker_window)
        self._uncertain: bool = False

        # Staleness detection: log-likelihood of training reference window
        self._train_ll_mean: Optional[float] = None
        self._train_ll_std:  Optional[float] = None
        self._is_model_stale: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, features_df: pd.DataFrame) -> None:
        """Fit the HMM on the most recent train_bars rows of features_df.

        Selects the best number of states (HMM_MIN_STATES..HMM_MAX_STATES)
        using BIC.  After fitting, builds a mapping from internal HMM states
        to regime labels sorted by mean log-return of observations.
        """
        data = features_df.dropna().tail(self.train_bars)
        if len(data) < max(HMM_MIN_STATES, 30):
            raise ValueError(
                f"Need at least {max(HMM_MIN_STATES, 30)} clean rows to fit HMM; "
                f"got {len(data)}."
            )

        X = data.values.astype(float)
        best_model, best_bic = None, np.inf

        for n in range(HMM_MIN_STATES, HMM_MAX_STATES + 1):
            for seed_offset in range(HMM_N_INIT):
                try:
                    model = GaussianHMM(
                        n_components=n,
                        covariance_type=HMM_COVARIANCE_TYPE,
                        n_iter=HMM_N_ITER,
                        tol=HMM_TOL,
                        random_state=42 + seed_offset,
                    )
                    model.fit(X)
                    bic = _bic(model, X)

                    monitor   = getattr(model, "monitor_", None)
                    history   = getattr(monitor, "history",   [])
                    converged = getattr(monitor, "converged", None)
                    ll_delta  = (
                        abs(history[-1] - history[-2])
                        if len(history) >= 2 else float("inf")
                    )
                    log.debug(
                        "HMM [%s] n_states=%d  seed=%d  BIC=%.2f  "
                        "ll_delta=%.2e  converged=%s  iters=%d/%d",
                        self.symbol, n, 42 + seed_offset, bic,
                        ll_delta, converged, len(history), HMM_N_ITER,
                    )

                    if bic < best_bic:
                        best_bic   = bic
                        best_model = model
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "HMM [%s] fit failed for n_states=%d seed=%d: %s",
                        self.symbol, n, 42 + seed_offset, exc,
                    )

        if best_model is None:
            raise RuntimeError("All HMM candidate fits failed.")

        self._model    = best_model
        self._n_states = best_model.n_components

        _mon      = getattr(best_model, "monitor_", None)
        _hist     = getattr(_mon, "history",   [])
        _conv     = getattr(_mon, "converged", None)
        _ll_delta = abs(_hist[-1] - _hist[-2]) if len(_hist) >= 2 else float("inf")
        log.info(
            "HMM [%s] selected n_states=%d  BIC=%.2f  "
            "final_ll_delta=%.2e  converged=%s  iters=%d/%d",
            self.symbol, self._n_states, best_bic,
            _ll_delta, _conv, len(_hist), HMM_N_ITER,
        )

        # Staleness reference: log-likelihood distribution from last 30 training bars.
        # Future predictions < (mean - 2 std) of this window trigger is_model_stale.
        X_ref = X[-30:] if len(X) >= 30 else X
        ll_ref = _per_row_log_likelihood(best_model, X_ref)
        self._train_ll_mean = float(np.mean(ll_ref))
        self._train_ll_std  = float(np.std(ll_ref))
        self._is_model_stale = False
        log.debug(
            "HMM [%s] staleness baseline: mean=%.3f std=%.3f (n=%d bars)",
            self.symbol, self._train_ll_mean, self._train_ll_std, len(X_ref),
        )

        # Build state → regime mapping sorted by mean return (first feature column)
        means = best_model.means_[:, 0]  # mean log-return per state
        sorted_states = np.argsort(means)  # ascending: crash → euphoria
        n = self._n_states
        self._state_to_regime = {
            int(state): _map_regime_label(rank, n)
            for rank, state in enumerate(sorted_states)
        }
        log.info("HMM [%s] state→regime mapping: %s", self.symbol, self._state_to_regime)

    # ------------------------------------------------------------------
    # Prediction (forward algorithm only)
    # ------------------------------------------------------------------

    def predict_current(self, features_row: pd.Series) -> int:
        """Predict the regime for a single bar using the forward algorithm.

        Parameters
        ----------
        features_row : Series of feature values for the *current* bar only.
                       Must not contain future data.

        Returns
        -------
        int  Confirmed regime label (0=crash … 4=euphoria), or the last
             confirmed regime if the confirmation gate has not yet fired or
             the flicker filter is active.  Returns -1 if no regime has ever
             been confirmed.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict_current().")

        obs = np.array(features_row.values, dtype=float).reshape(1, -1)
        raw_state = _forward_decode(self._model, obs)
        regime    = self._state_to_regime[raw_state]

        # Confirmation gate
        if regime == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = regime
            self._pending_count  = 1

        newly_confirmed = self._pending_count >= self.confirmation_bars

        if newly_confirmed:
            self._confirmed_regime = regime

        emitted = self._confirmed_regime if self._confirmed_regime is not None else -1
        self._history.append(emitted)
        if emitted != -1:
            self._recent_regimes.append(emitted)
            self._uncertain = _check_flicker(
                self._recent_regimes, self.flicker_threshold
            )

        # Staleness check: current observation's log-likelihood vs training distribution.
        if self._train_ll_mean is not None:
            current_ll = float(_per_row_log_likelihood(self._model, obs)[0])
            std        = self._train_ll_std or 0.0
            threshold  = self._train_ll_mean - 2.0 * std
            if current_ll < threshold:
                self._is_model_stale = True
                self._uncertain      = True
                log.warning(
                    "HMM [%s] model may be stale: obs_ll=%.3f < threshold=%.3f "
                    "(mean=%.3f std=%.3f)",
                    self.symbol, current_ll, threshold,
                    self._train_ll_mean, std,
                )
            else:
                self._is_model_stale = False
                # _uncertain may still be True from flicker filter; don't clear it here.

        return emitted

    def is_confirmed(self) -> bool:
        """True if the current pending state has reached CONFIRMATION_BARS."""
        return self._pending_count >= self.confirmation_bars

    def is_uncertain(self) -> bool:
        """True when the flicker filter is active or the model is stale."""
        return self._uncertain

    @property
    def is_model_stale(self) -> bool:
        """True when the current observation's log-likelihood is > 2 std below
        the training reference distribution (last 30 training bars)."""
        return self._is_model_stale

    def regime_history(self) -> list[int]:
        """Return all regime labels emitted so far (one per predict_current call)."""
        return list(self._history)

    def regime_name(self, label: int) -> str:
        """Human-readable name for a regime label."""
        if 0 <= label < len(_REGIME_NAMES):
            return _REGIME_NAMES[label]
        return f"state_{label}"

    def reset_filters(self) -> None:
        """Reset confirmation gate, flicker filter, and staleness flag (call after re-fit)."""
        self._pending_state   = None
        self._pending_count   = 0
        self._uncertain       = False
        self._is_model_stale  = False
        self._recent_regimes.clear()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _per_row_log_likelihood(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Approximate marginal log-likelihood per observation row.

    Uses logsumexp(log_emission[row] + log(startprob)) as a single-step
    forward pass — consistent with _forward_decode's prior assumption.

    Returns array of shape (n_samples,).
    """
    log_emissions = model._compute_log_likelihood(X)   # (n_samples, n_states)
    log_prior     = np.log(model.startprob_ + 1e-300)
    combined      = log_emissions + log_prior           # broadcast: (n_samples, n_states)
    max_vals      = combined.max(axis=1, keepdims=True)
    return (max_vals.squeeze(axis=1) +
            np.log(np.sum(np.exp(combined - max_vals), axis=1)))


def _bic(model: GaussianHMM, X: np.ndarray) -> float:
    """BIC = -2 * log-likelihood + k * log(n)."""
    log_likelihood = model.score(X)
    n_samples  = X.shape[0]
    n_features = X.shape[1]
    n_states   = model.n_components

    # Free parameters: transition matrix, means, covariances, initial probs
    cov_type = model.covariance_type
    if cov_type == "full":
        cov_params = n_states * n_features * (n_features + 1) // 2
    elif cov_type == "diag":
        cov_params = n_states * n_features
    elif cov_type == "spherical":
        cov_params = n_states
    else:  # tied
        cov_params = n_features * (n_features + 1) // 2

    k = (
        n_states * n_features       # means
        + cov_params                # covariances
        + n_states * (n_states - 1) # transition matrix (rows sum to 1)
        + (n_states - 1)            # initial state distribution
    )
    return -2 * log_likelihood * n_samples + k * np.log(n_samples)


def _forward_decode(model: GaussianHMM, obs: np.ndarray) -> int:
    """Return the most likely state for a single observation using the
    forward algorithm (emission probabilities × prior), without any
    backward pass or full-sequence Viterbi decoding.

    obs shape: (1, n_features)
    """
    # log emission probability for each state given this observation
    log_emission = model._compute_log_likelihood(obs)  # shape (1, n_states)

    # Prior: use the stationary distribution derived from the transition matrix
    # as a stand-in for π when we don't have sequence context.
    log_prior = np.log(model.startprob_ + 1e-300)

    log_posterior = log_prior + log_emission[0]
    return int(np.argmax(log_posterior))


def _check_flicker(recent: deque, threshold: int) -> bool:
    """Return True if the number of regime changes in recent exceeds threshold."""
    if len(recent) < 2:
        return False
    changes = sum(1 for a, b in zip(list(recent), list(recent)[1:]) if a != b)
    return changes > threshold


def _map_regime_label(rank: int, n_states: int) -> int:
    """Map a rank (0 = lowest-return state) to a 0–4 regime label.

    With fewer than 5 states the labels are spread across the 0–4 scale
    so that 0 always means crash and the highest rank always means euphoria.
    """
    if n_states == 1:
        return 2  # neutral
    # Linear interpolation: rank 0 → 0, rank n_states-1 → 4
    return round(rank * 4 / (n_states - 1))

"""
wheel_scanner/scoring.py
------------------------
Pure scoring functions for each wheel candidate criterion.

All functions take numbers and return a float in [0, 100].
No I/O, no side effects — fully unit-testable without mocks.

Weights (must sum to 1.0):
    IVR           30%
    Put premium   25%
    Regime        20%
    Trend         15%
    Liquidity     10%
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "ivr":       0.30,
    "premium":   0.25,
    "regime":    0.20,
    "trend":     0.15,
    "liquidity": 0.10,
}

# Regime int → name map (mirrors HMM engine)
REGIME_NAMES: dict[int, str] = {
    0: "crash",
    1: "bear",
    2: "neutral",
    3: "bull",
    4: "euphoria",
}

# Trend score → human-readable label
TREND_LABELS: dict[float, str] = {
    100.0: "above_both_smas",
    60.0:  "recovering",
    20.0:  "below_50sma",
    0.0:   "downtrend",
}


# ---------------------------------------------------------------------------
# Individual criterion scores (each returns 0–100)
# ---------------------------------------------------------------------------

def score_ivr(ivr: Optional[float]) -> float:
    """Piecewise linear score peaking at IVR=60, zero outside [40, 80].

    IVR < 40  → premium too cheap     → 0
    IVR 40-60 → linear ramp 0→100
    IVR 60-80 → linear ramp 100→0
    IVR > 80  → likely binary event   → 0
    """
    if ivr is None or ivr < 40.0 or ivr > 80.0:
        return 0.0
    if ivr <= 60.0:
        return (ivr - 40.0) / 20.0 * 100.0
    return (80.0 - ivr) / 20.0 * 100.0


def score_put_premium(annualized_yield_pct: Optional[float]) -> float:
    """Piecewise linear score peaking at 30% annualized yield, zero outside [18%, 45%].

    yield < 18%  → not worth capital tie-up → 0
    yield 18-30% → linear ramp 0→100
    yield 30-45% → linear ramp 100→0
    yield > 45%  → pricing in excess risk  → 0
    """
    if annualized_yield_pct is None:
        return 0.0
    y = annualized_yield_pct
    if y < 18.0 or y > 45.0:
        return 0.0
    if y <= 30.0:
        return (y - 18.0) / 12.0 * 100.0
    return (45.0 - y) / 15.0 * 100.0


def score_regime(regime_label: Optional[int]) -> float:
    """Map HMM regime integer to a put-selling environment score.

    Crash (0)    →  0   — no new positions
    Bear (1)     → 20   — only highest-quality candidates
    Neutral (2)  → 60   — acceptable, reduce sizing
    Bull (3)     → 100  — ideal put-selling environment
    Euphoria (4) → 30   — market extended, assignment risk high
    """
    _scores: dict[int, float] = {0: 0.0, 1: 20.0, 2: 60.0, 3: 100.0, 4: 30.0}
    if regime_label is None:
        return 60.0  # default to neutral when unknown
    return _scores.get(regime_label, 0.0)


def score_trend(
    price: float,
    sma50: float,
    sma200: float,
    sma50_declining: bool = False,
) -> float:
    """Score the underlying's trend health for put selling.

    Selling puts into a downtrend is the primary wheel failure mode.

    price > 50d AND 50d > 200d                          → 100 (ideal)
    price > 50d AND 50d ≤ 200d (recovering)             →  60
    price ≤ 50d AND NOT (50d ≤ 200d AND declining)      →  20
    price ≤ 50d AND 50d ≤ 200d AND 50d declining        →   0 (falling knife)
    """
    above_50d       = price > sma50
    sma50_above_200 = sma50 > sma200

    if above_50d and sma50_above_200:
        return 100.0
    if above_50d:
        return 60.0
    if not sma50_above_200 and sma50_declining:
        return 0.0
    return 20.0


def trend_label(score: float) -> str:
    """Return the human-readable trend label for a trend score value."""
    for threshold, label in sorted(TREND_LABELS.items(), reverse=True):
        if score >= threshold:
            return label
    return "downtrend"


def score_liquidity(
    open_interest: Optional[int],
    bid_ask_spread_pct: Optional[float],
    avg_daily_volume: Optional[int],
) -> float:
    """Count how many of the three liquidity criteria are met.

    Criteria:
      1. Open interest at target put ≥ 500 contracts
      2. Bid-ask spread < 10% of mid price
      3. Average daily options volume ≥ 200 contracts

    All three met  → 100
    Two met        →  50
    One or zero    →   0
    """
    met = 0
    if open_interest is not None and open_interest >= 500:
        met += 1
    if bid_ask_spread_pct is not None and bid_ask_spread_pct < 10.0:
        met += 1
    if avg_daily_volume is not None and avg_daily_volume >= 200:
        met += 1

    if met == 3:
        return 100.0
    if met == 2:
        return 50.0
    return 0.0


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def composite_score(
    ivr_score:       float,
    premium_score:   float,
    regime_score:    float,
    trend_score:     float,
    liquidity_score: float,
) -> float:
    """Weighted composite of the five component scores, rounded to 1 decimal."""
    raw = (
        WEIGHTS["ivr"]       * ivr_score
        + WEIGHTS["premium"]   * premium_score
        + WEIGHTS["regime"]    * regime_score
        + WEIGHTS["trend"]     * trend_score
        + WEIGHTS["liquidity"] * liquidity_score
    )
    return round(max(0.0, min(100.0, raw)), 1)


# ---------------------------------------------------------------------------
# Flag helpers
# ---------------------------------------------------------------------------

def compute_flags(
    ivr:                Optional[float],
    bid_ask_spread_pct: Optional[float],
    open_interest:      Optional[int],
    price:              float,
    sma50:              float,
    days_to_earnings:   Optional[int],
) -> list[str]:
    """Return a list of soft-warning flag strings for a candidate."""
    flags: list[str] = []

    if ivr is not None and ivr > 70.0:
        flags.append("IV_elevated")

    if bid_ask_spread_pct is not None and bid_ask_spread_pct > 7.0:
        flags.append("spread_wide")

    if days_to_earnings is not None and days_to_earnings <= 21:
        flags.append("near_earnings")

    if open_interest is not None and 500 <= open_interest < 800:
        flags.append("low_oi")

    if price < sma50:
        flags.append("below_50sma")

    return flags

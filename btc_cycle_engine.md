# BTC 60-Day Cycle Engine — Deep Reference

How the `CycleEngine` detects cycle troughs, scores their timing, and produces
the composite signal that adjusts BTC allocation up or down by one tier.

This document goes one level below `btc_strategy.md`. Read that first if you
want the allocation and trade-action layer; this document focuses exclusively on
how the cycle signal is built.

---

## Table of Contents

1. [Why a Cycle Engine?](#why-a-cycle-engine)
2. [CycleSignal Fields](#cyclesignal-fields)
3. [Stage 1 — Seed Lows](#stage-1--seed-lows)
4. [Stage 2 — Detecting Cycle Lows from Price History](#stage-2--detecting-cycle-lows-from-price-history)
5. [Stage 3 — Hypothesis Evaluation (Picking the Best Low)](#stage-3--hypothesis-evaluation-picking-the-best-low)
6. [Stage 4 — Adaptive Window Center](#stage-4--adaptive-window-center)
7. [Stage 5 — Timing Probability (Gaussian)](#stage-5--timing-probability-gaussian)
8. [Stage 6 — Price Confirmation Signals](#stage-6--price-confirmation-signals)
   - [Donchian Channel Score](#donchian-channel-score-40-weight)
   - [Gaussian MA Score](#gaussian-ma-score-35-weight)
   - [Bollinger Band Score](#bollinger-band-score-25-weight)
9. [Stage 7 — HMM Confirmation](#stage-7--hmm-confirmation)
10. [Stage 8 — Cycle Quality Score](#stage-8--cycle-quality-score)
11. [Stage 9 — Translation (Left / Right)](#stage-9--translation-left--right)
12. [Stage 10 — Macro Phase (4-Year Cycle)](#stage-10--macro-phase-4-year-cycle)
13. [Stage 11 — Composite Score Assembly](#stage-11--composite-score-assembly)
14. [Failed Cycle Detection](#failed-cycle-detection)
15. [How the Signal Feeds into Allocation](#how-the-signal-feeds-into-allocation)
16. [Alerts Fired by the Cycle Engine](#alerts-fired-by-the-cycle-engine)
17. [Simulated Walk-Through — Three Points in a Cycle](#simulated-walk-through--three-points-in-a-cycle)

---

## Why a Cycle Engine?

BTC historically oscillates between local troughs at roughly 60-day intervals.
This is a shorter, intra-trend cycle that exists inside both bull and bear macro
phases. A pure HMM regime model can identify that the market is in a bull trend
but cannot tell you *where inside that trend* the price currently sits — whether
it is early in a new leg (buy) or overextended near a local peak (trim).

The cycle engine answers that question probabilistically. It identifies where the
last cycle low was, estimates whether the current bar is near the next expected
trough, and scores confidence using three independent price signals. The output
is a single `composite_score` (0–1) that the BTC strategy uses to shift
allocation one tier higher or lower.

---

## CycleSignal Fields

The full output dataclass. Every field is computed on each bar.

| Field | Type | Description |
|-------|------|-------------|
| `days_since_last_low` | int | Days from the identified cycle low to today |
| `timing_probability` | float 0–1 | Gaussian probability that today is at a cycle trough |
| `window_center_days` | int | Adaptive expected cycle length (default 60) |
| `window_std_days` | int | Gaussian std (fixed at 12) |
| `macro_phase` | str | accumulation / markup / distribution / markdown |
| `days_since_cycle_start` | int | Same as `days_since_last_low` |
| `cycle_completion_pct` | float 0–1 | `days_since_last_low / adaptive_center` (capped at 1.0) |
| `translation` | str | right / left / unknown |
| `translation_confidence` | float 0–1 | Confidence in the translation measurement |
| `donchian_score` | float 0–1 | Donchian channel signal |
| `gaussian_score` | float 0–1 | Gaussian MA crossover signal |
| `bollinger_score` | float 0–1 | Bollinger band touch signal |
| `price_confirmation` | float 0–1 | Weighted combination of the three above |
| `hmm_confirmation` | float 0–1 | Regime-derived cycle low confirmation |
| `composite_score` | float 0–1 | Final weighted aggregate used by strategy |
| `bias` | str | long / neutral / short |
| `failed_cycle` | bool | Price broke below the confirmed cycle low |
| `cycle_quality_score` | float 0–1 | Regularity of recent cycle lengths |
| `adaptive_window_center` | int | Same as `window_center_days` |

---

## Stage 1 — Seed Lows

The engine bootstraps with three known historical BTC cycle bottoms. These are
hard-coded because the detected cycle low algorithm requires at least
`15 + 20 + 1 = 36` bars of *surrounding* history to confirm a low, so recent
major bottoms near the start of the price series would be missed.

| Date | Price | Days from prior |
|------|-------|----------------|
| 2018-12-15 | $3,200 | — (first seed) |
| 2020-03-13 | $3,800 | ~454 days |
| 2022-11-21 | $15,500 | ~983 days |

All seed lows are given `confidence = 0.90` and `confirmed = True`.

These lows feed into:
- Adaptive window calculation (what is the typical cycle length?)
- Hypothesis evaluation (which of the recent lows is the current cycle start?)
- Failed cycle check (did price break below the cycle low we identified?)

---

## Stage 2 — Detecting Cycle Lows from Price History

`detect_cycle_lows(price_history_df)` scans the price series to find genuine
troughs that were followed by a meaningful recovery.

### Algorithm

For each bar `i` in the range `[15, n − 20]`:

**Step 1 — Local minimum test (±15-bar window)**
```
window = close[i-15 : i+16]   (31 bars centered on i)
candidate if: close[i] == min(window)
```
The bar must be the lowest price in a 31-bar window. This eliminates
noise and ensures the candidate is a genuine local trough, not just a
1-bar dip.

**Step 2 — Forward confirmation (20-bar window)**
```
future_max = max(close[i+1 : i+21])
rise_pct   = (future_max - close[i]) / close[i]
confirmed  if: rise_pct > 0.10   (CYCLE_LOW_CONFIRMATION_PCT)
```
The price must rise more than **10%** within the 20 bars following the candidate.
This separates genuine troughs from sideways lows that do not mark the start of
a new leg up.

**Step 3 — Confidence scoring**
```
confidence = clamp(rise_pct × 3.0, 0.10, 1.00)
```
A 10% rise (the minimum) gives `0.30`. A 33%+ rise gives `1.00`. Larger
recoveries produce higher-confidence lows.

### Examples

| Candidate price | Future max | Rise % | Confirmed? | Confidence |
|----------------|-----------|--------|-----------|------------|
| $28,000 | $30,000 | +7.1% | No | — |
| $28,000 | $31,500 | +12.5% | Yes | 0.375 |
| $28,000 | $36,000 | +28.6% | Yes | 0.858 |
| $28,000 | $46,000 | +64.3% | Yes | 1.00 (capped) |

### Why both tests?

The ±15-bar test alone catches every dip. The 10% forward confirmation
prevents the engine from treating a brief consolidation mid-downtrend as
a cycle bottom. Together they produce lows that are both structurally
significant and empirically followed by an upside move.

---

## Stage 3 — Hypothesis Evaluation (Picking the Best Low)

After combining seed lows and detected lows, there may be several candidates
from recent history. `evaluate_cycle_hypotheses()` scores the **last 3** and
returns the one that best represents the *current* cycle start.

### Scoring formula

For each candidate low:
```
max_gain        = (max price since the low) / candidate_price − 1.0
revisit_penalty = count(bars where price < candidate_price × 1.01) × 0.05
                  (capped at 0.50)

score = candidate.confidence × min(max_gain, 1.0) − revisit_penalty
score = max(0.0, score)
```

**Intuition:**
- A large gain since the low → high score (the market validated this as a real bottom)
- Many revisits close to or below the low → penalty (the low was "tested" repeatedly,
  meaning the market did not cleanly depart from it)
- When two candidates score nearly equal (`score >= best - 1e-9`), the more
  recent one wins — it is more likely to represent where the current cycle began

### Example

Three candidate lows:

| Low date | Low price | Confidence | Max gain since | Revisit bars | Score |
|----------|-----------|------------|---------------|-------------|-------|
| 45 days ago | $27,000 | 0.70 | +38% | 6 bars | 0.70 × 0.38 − 0.30 = **−0.03** (→ 0) |
| 30 days ago | $28,500 | 0.55 | +24% | 1 bar | 0.55 × 0.24 − 0.05 = **0.082** |
| 12 days ago | $29,200 | 0.40 | +8%  | 0 bars | 0.40 × 0.08 − 0.00 = **0.032** |

Winner: **30-days-ago low** (score 0.082). The 45-day low was repeatedly
revisited, signaling it was not a clean floor. The 12-day low has not had
enough time to prove itself.

---

## Stage 4 — Adaptive Window Center

The expected cycle length is not locked at 60 days. It adapts based on how long
recent cycles actually lasted.

```
cycle_lengths = [days_from_prior for each detected + seed low]
lengths       = last 3 values
weights       = [0.50, 0.30, 0.20]  applied most-recent-first

weighted_avg  = sum(length × weight) / sum(weights)
center        = clamp(round(weighted_avg), 45, 90)
```

| Scenario | Last 3 cycle lengths | Weighted avg | Center used |
|----------|---------------------|-------------|-------------|
| Typical | 58, 63, 61 | 60.5 | 61 |
| Compressing (bear market chop) | 45, 48, 50 | 47.9 | 48 |
| Extending (strong bull) | 70, 75, 80 | 74.0 | 74 |
| Clamped long | 95, 100, 92 | 95.9 → clamped | 90 |

The clamp at [45, 90] prevents the model from drifting into implausibly short
or long expectations due to anomalous readings.

---

## Stage 5 — Timing Probability (Gaussian)

This is the central signal: how probable is it that *today* sits at the trough
of the next 60-day cycle?

### Formula

```
timing_probability = exp(−0.5 × ((days_since_last_low − center) / std)²)
```

Where `center = adaptive_window_center` (≈60) and `std = CYCLE_60D_STD = 12`.

This is a unit-scaled Gaussian: it returns **1.0** when `days_since_last_low == center`
and decays symmetrically on both sides. It never reaches exactly 0.

### Probability at various days (center=60, std=12)

| Days since last low | Formula | Timing probability |
|--------------------|---------|-------------------|
| 0 | exp(−0.5 × (−60/12)²) = exp(−12.5) | ~0.000 |
| 24 | exp(−0.5 × (−36/12)²) = exp(−4.5) | ~0.011 |
| 36 | exp(−0.5 × (−24/12)²) = exp(−2.0) | ~0.135 |
| 48 | exp(−0.5 × (−12/12)²) = exp(−0.5) | ~0.607 |
| **60** | exp(−0.5 × (0/12)²) = exp(0) | **1.000** |
| 72 | exp(−0.5 × (12/12)²) = exp(−0.5) | ~0.607 |
| 84 | exp(−0.5 × (24/12)²) = exp(−2.0) | ~0.135 |
| 96 | exp(−0.5 × (36/12)²) = exp(−4.5) | ~0.011 |
| 120 | exp(−0.5 × (60/12)²) = exp(−12.5) | ~0.000 |

```
Timing probability curve (center = 60, std = 12):

1.00 |                     *
0.90 |                   *   *
0.80 |                 *       *
0.70 |               *           *
0.60 |             *               *
0.50 |           *                   *
0.40 |         *                       *
0.30 |       *                           *
0.20 |      *                             *
0.13 |    *                                 *
0.00 +--+--+--+--+--+--+--+--+--+--+--+--+--
     0  12 24 36 48 60 72 84 96 108 120
            days since last low
```

**Key design property:** The timing signal is forward-looking about cycle
position, not about price movement. It has no predictive power by itself —
it only says "if there is going to be a cycle low, it is most likely around day
60." It must be confirmed by the price signals in Stage 6 to carry weight.

---

## Stage 6 — Price Confirmation Signals

Three independent price-based signals ask: "does the current price action
*look like* a cycle trough?" They run regardless of timing probability and
are combined into `price_confirmation`.

### Donchian Channel Score (40% weight)

Measures where the current price sits within the 60-bar (≈3-month) high-low range.

```
upper     = rolling 60-bar max of close
lower     = rolling 60-bar min of close
midpoint  = (upper + lower) / 2

Case 1 — breach-and-recover:
  recent_low (last 5 bars) ≤ lower × 1.01  AND  current > lower × 1.01
  → score = 1.0  (strongest signal — price visited the floor then lifted off)

Case 2 — above midpoint:
  current ≥ midpoint
  → score = 0.0  (price not in the lower half; no cycle low signal)

Case 3 — below midpoint, approaching floor:
  score = (midpoint − current) / (midpoint − lower) × 0.95
  → ranges 0.0–0.95 (1.0 reserved for case 1)
```

**Why 60 bars?** A 60-bar Donchian channel roughly matches one expected cycle.
A price at the floor of this channel has underperformed its own 3-month range —
exactly the setup that precedes a cycle trough recovery.

**Breach-and-recover:** The most actionable signal. Price briefly broke below
the floor (capitulation) and recovered above it within 5 bars. This micro-pattern
is consistent with a final flush-and-bounce at a genuine low.

### Gaussian MA Score (35% weight)

Uses a centered Gaussian-weighted moving average of the last 60 bars.
Unlike a simple MA, the Gaussian kernel weights the middle of the window most
heavily, producing a smoother and less reactive average.

```
kernel    = Gaussian window centered at bar 30 of 60, σ = 10
GMA       = dot(close[-60:], kernel / kernel.sum())

Case 1 — bullish crossover:
  prev_close < prev_GMA  AND  current ≥ current_GMA
  → score = 1.0  (price just crossed back above the weighted MA)

Case 2 — riding near GMA:
  |current − GMA| / GMA < 0.01  (within 1%)
  → score = 0.5

Case 3 — below GMA:
  depth = (GMA − current) / GMA
  score = max(0.0, 0.5 − depth × 5.0)
  → ranges 0.0–0.5 (deeper below = lower score)

Case 4 — above GMA, no recent crossover:
  → score = 0.3  (holding above MA is mildly positive but not a cycle low signal)
```

**Why crossovers?** At a cycle trough, price typically falls below the MA
during the correction and then crosses back above as the new leg begins. The
crossover captures this transition directly.

### Bollinger Band Score (25% weight)

Uses a standard 20-bar Bollinger Band (2σ) and tracks both price position and
band width change.

```
BB_lower = 20-bar MA − 2 × 20-bar std
BB_upper = 20-bar MA + 2 × 20-bar std
band_range = BB_upper − BB_lower

bands_expanding = current_std > prev_std_5bars_ago × 1.001
touched_lower   = min(close[-5:]) ≤ BB_lower + band_range × 0.05

Case 1 — touched lower AND bands expanding:
  → score = 1.0  (capitulation low with expanding volatility — strongest Bollinger signal)

Case 2 — touched lower, bands still contracting:
  → score = 0.5  (price at lower band but not enough volatility expansion yet)

Case 3 — bands contracting, no touch:
  → score = 0.1

Case 4 — bands expanding, no touch:
  → score = 0.2

Guard — price in top 20% of band OR above upper band:
  → score = 0.0  (price is extended upward — no cycle low signal possible)
```

**Why band expansion matters?** Capitulation lows are typically accompanied by
a surge in volatility (bands widen). A price touch of the lower band during a
contraction phase often means a slow drift lower, not a true trough. Expanding
bands after a lower-band touch is the Bollinger pattern most consistent with a
flush-and-recover cycle low.

### Price Confirmation Assembly

```
price_confirmation = 0.40 × donchian_score
                   + 0.35 × gaussian_score
                   + 0.25 × bollinger_score
```

Donchian gets the most weight because it measures price position in an absolute
range, not relative to an MA. All three signals must broadly agree to push
`price_confirmation` above 0.6.

---

## Stage 7 — HMM Confirmation

The HMM regime at the time of the cycle signal provides a context check.
Intuitively: if the HMM detects a crash or bear regime *while* the timing
probability is high, that combination is *bullish* for a cycle low — the regime
confirms that the market is selling off at the expected trough time.

```
regime -1 or None → 0.50   (no information; neutral)
regime 0 (crash)  → 0.50 + timing_prob × 0.50   (crash at high timing = 1.0)
regime 1 (bear)   → same formula as crash
regime 2 (neutral)→ 0.50   (always; neutral never confirms or denies)
regime 3 (bull)   → 0.50 − timing_prob × 0.50   (bull at high timing = 0.0)
regime 4 (euphoria)→ same formula as bull
```

**Example at timing_prob = 0.80:**

| Regime | HMM confirmation |
|--------|-----------------|
| crash (0) | 0.50 + 0.80 × 0.50 = **0.90** |
| bear (1) | 0.50 + 0.80 × 0.50 = **0.90** |
| neutral (2) | **0.50** |
| bull (3) | 0.50 − 0.80 × 0.50 = **0.10** |
| euphoria (4) | 0.50 − 0.80 × 0.50 = **0.10** |

Bull or euphoria at high timing probability signals that the market has *not*
corrected into the expected trough — the cycle is running late or the trough
may not come. This reduces the HMM confirmation and drags the composite down.

---

## Stage 8 — Cycle Quality Score

Measures how *regular* recent cycles have been. Irregular cycle spacing reduces
confidence that the current timing estimate is reliable.

```
recent_lengths = [days_from_prior for last CYCLE_QUALITY_LOOKBACK (3) lows]
mean_len       = average(recent_lengths)
variance       = mean squared deviation from mean_len
cv             = sqrt(variance) / mean_len   (coefficient of variation)
quality        = clamp(1.0 − cv, 0.0, 1.0)
final_quality  = quality × avg_confidence_of_recent_lows
```

**Coefficient of variation (CV):** CV = 0 means all cycles were exactly the
same length (perfect regularity → quality = 1.0). CV = 1.0 means the std equals
the mean (maximum irregularity → quality = 0.0).

**Example:**

| Scenario | Cycle lengths | CV | Quality |
|----------|--------------|-----|---------|
| Highly regular | 59, 61, 60 | 0.01 | ~0.99 |
| Moderate | 50, 65, 58 | 0.12 | ~0.88 |
| Irregular | 40, 75, 55 | 0.30 | ~0.70 |
| Chaotic | 30, 90, 45 | 0.50 | ~0.50 |

---

## Stage 9 — Translation (Left / Right)

Cycle translation describes whether the price peak within a cycle occurred in
the first or second half of the cycle duration.

```
since_low = close bars from cycle_low_timestamp to today
peak_pos  = index of max(since_low)
half      = len(since_low) / 2

translation = "right" if peak_pos > half else "left"
confidence  = min(1.0, days_elapsed / 60.0) × 0.8
```

| Translation | Meaning | Implication |
|------------|---------|-------------|
| **right** | Peak in second half of cycle | Bullish — market held up longer before correcting |
| **left** | Peak in first half of cycle | Bearish — market peaked quickly and spent most of the cycle declining |
| **unknown** | < 30 days elapsed | Too early to measure; not enough of the cycle has unfolded |

Translation confidence increases linearly with cycle age. At 30 days (halfway
through the typical 60-day cycle), confidence is 0.40. At 60+ days, 0.80 is
the maximum.

**Note:** Translation is reported in `CycleSignal` but is not currently a direct
input to the composite score. It is available for dashboard display and future
strategy refinement.

---

## Stage 10 — Macro Phase (4-Year Cycle)

Bitcoin historically follows a ~4-year halving cycle. The cycle engine tracks
where the current date sits relative to the last major 4-year cycle low
(2022-11-21, $15,500).

```
days_since_4y_low = today − 2022-11-21

Phase boundaries:
  0–365 days   → accumulation  (buying the dip after the bottom)
  366–730 days → markup        (trend establishment and expansion)
  731–1095 days → distribution (late-cycle, smart money distributing)
  > 1095 days  → markdown      (beginning of the next bear cycle)
```

**Current reference dates** (from the 2022-11-21 low):

| Phase | Date range |
|-------|-----------|
| Accumulation | Nov 2022 – Nov 2023 |
| Markup | Nov 2023 – Nov 2024 |
| Distribution | Nov 2024 – Nov 2025 |
| Markdown | Nov 2025 → |

**As of today (May 2026):** approximately 1,276 days since the 4-year low →
**markdown** phase.

Macro phase is reported in `CycleSignal.macro_phase` and displayed in the
Telegram alert. It is not a direct input to the composite score but provides
strategic context — a high composite score during a markdown phase should be
treated more cautiously than the same score during an accumulation phase.

---

## Stage 11 — Composite Score Assembly

All prior stages feed into a single weighted composite:

```
composite = 0.35 × timing_probability
          + 0.30 × price_confirmation
          + 0.20 × hmm_confirmation
          + 0.15 × cycle_quality_score

composite = clamp(composite, 0.0, 1.0)
```

**Weight rationale:**

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Timing probability | 35% | The core hypothesis — we are near the expected trough |
| Price confirmation | 30% | Three price signals verify the hypothesis in the data |
| HMM confirmation | 20% | Regime context (crash/bear at trough timing is bullish) |
| Cycle quality | 15% | Confidence that timing model is reliable given recent regularity |

### Bias assignment

```
if failed_cycle:         bias = "short"
elif composite >= 0.65:  bias = "long"
else:                    bias = "neutral"
```

The `bias` field is a human-readable summary of the composite. The allocation
tier boost/reduce uses `composite_score >= 0.65` and `failed_cycle` directly,
not the `bias` string.

---

## Failed Cycle Detection

`is_failed_cycle(current_price, prior_cycle_low)` is a single comparison:

```
failed_cycle = current_price < prior_cycle_low.price
```

If BTC falls below the price at the identified cycle low, the engine flags a
failed cycle. This means the bottom we identified was not a genuine floor —
the market has made a new lower low, invalidating the cycle hypothesis.

**Effect on allocation:** A failed cycle triggers a **tier-down** shift regardless
of the composite score. Even if timing and price signals are high (perhaps
because the new low is attracting technical buyers), a price breakdown below a
confirmed floor is treated as a structural signal to reduce exposure.

**Example:** Cycle low identified at $28,000. BTC subsequently rallied to $36,000
then fell back to $27,200. `failed_cycle = True`. Even if the timing Gaussian
is peaking, the strategy shifts down one allocation tier until a new confirmed
low is established.

---

## How the Signal Feeds into Allocation

This is the bridge between the cycle engine and `BTCStrategy.get_target_allocation()`:

```python
# In BTCStrategy.get_target_allocation():
if BTC_CYCLE_TIER_BOOST:
    if cycle_signal.failed_cycle:
        lower = max(regime - 1, 0)
        adj = REGIME_ALLOCATIONS[lower]          # shift down
    elif cycle_signal.composite_score >= 0.65:
        higher = min(regime + 1, 4)
        adj = REGIME_ALLOCATIONS[higher]         # shift up
```

Only two fields from the entire `CycleSignal` dataclass directly change
allocation: `failed_cycle` and `composite_score`. Everything else feeds into
building those two values.

**Allocation shift table:**

| Regime | No shift | Composite ≥ 0.65 (shift up) | Failed cycle (shift down) |
|--------|----------|------------------------------|--------------------------|
| 0 crash | 0% | 0% (crash always stays 0%) | 0% |
| 1 bear | 5% | 10% (→ neutral tier) | 0% (→ crash tier) |
| 2 neutral | 10% | 15% (→ bull tier) | 5% (→ bear tier) |
| 3 bull | 15% | 8% (→ euphoria tier — take-profit) | 10% (→ neutral tier) |
| 4 euphoria | 8% | 8% (→ euphoria again, capped) | 15% (→ bull tier) |

Note the bull + boost case: a strong cycle signal at bull regime shifts to the
*euphoria allocation* (8%), which is lower. This is intentional — if the market
is in a bull trend and the 60-day cycle composite is firing at its peak, the
strategy interprets that as "we may be near the local cycle top" and pre-emptively
trims 7 percentage points.

---

## Alerts Fired by the Cycle Engine

`get_cycle_signal()` calls `alerts.send_cycle_alert()` on every bar. The alert
only fires to channels (webhook/Telegram) in two cases:

| Event | Condition |
|-------|-----------|
| **Threshold crossing** | `composite_score` crossed above or below `0.65` since last bar |
| **Failed cycle** | `failed_cycle` is newly True (price broke below cycle low) |

Routine bars where the composite is stable and no failure occurred are logged at
DEBUG level only — no alert is sent to prevent notification fatigue.

---

## Simulated Walk-Through — Three Points in a Cycle

The following traces a single 60-day cycle. The cycle low was identified on
**Day 0** at **$28,000** with confidence 0.72. The adaptive window center is
**62 days** (recent cycles averaged slightly long). NAV = $100,000, regime = bear.

---

### Day 15 — Early Cycle (Accumulation)

BTC has recovered to $31,500 since the trough. No indicators are signaling a
new low.

**Inputs:**
```
days_since_last_low = 15
current_price       = $31,500
regime              = 1 (bear)
adaptive_center     = 62
```

**Timing probability:**
```
exp(−0.5 × ((15 − 62) / 12)²) = exp(−0.5 × (−3.92)²) = exp(−7.67) ≈ 0.000
```

**Donchian score:**
```
60-bar range: high $38,000, low $28,000, midpoint $33,000
current $31,500 < midpoint $33,000 → case 3 (below midpoint)
score = (33,000 − 31,500) / (33,000 − 28,000) × 0.95 = 0.285
```

**Gaussian MA score:**
```
GMA ≈ $32,800 (smoothed recent downtrend)
current $31,500 < GMA $32,800
depth = (32,800 − 31,500) / 32,800 = 0.040
score = max(0, 0.5 − 0.040 × 5.0) = 0.30
```

**Bollinger score:**
```
Lower band ≈ $28,500, upper ≈ $35,500
current $31,500 in middle of band; recent low $29,000 touched lower band
bands still contracting (settling after the crash)
→ score = 0.50 (touched lower but no expansion)
```

**Price confirmation:**
```
= 0.40 × 0.285 + 0.35 × 0.30 + 0.25 × 0.50
= 0.114 + 0.105 + 0.125
= 0.344
```

**HMM confirmation (regime=1, timing=0.000):**
```
= 0.50 + 0.000 × 0.50 = 0.50
```

**Cycle quality:** 0.82 (recent cycles were moderately regular)

**Composite:**
```
= 0.35 × 0.000 + 0.30 × 0.344 + 0.20 × 0.50 + 0.15 × 0.82
= 0.000 + 0.103 + 0.100 + 0.123
= 0.326
```

**Outcome:** composite 0.326 < 0.65 → **no tier shift** → bear allocation stays 5%
The strategy holds the existing $5,000 BTC position (acquired at the prior cycle low).
No new order.

---

### Day 58 — Approaching the Expected Trough

BTC has pulled back to $30,200. Timing probability is near its peak.
The Bollinger lower band was touched 3 bars ago and bands are now widening.

**Inputs:**
```
days_since_last_low = 58
current_price       = $30,200
regime              = 1 (bear)
adaptive_center     = 62
```

**Timing probability:**
```
exp(−0.5 × ((58 − 62) / 12)²) = exp(−0.5 × 0.111) = exp(−0.056) ≈ 0.946
```

**Donchian score:**
```
60-bar range: high $36,800, low $28,000, midpoint $32,400
current $30,200 < midpoint → case 3
recent 5-bar low = $29,800; lower band = $28,000
no breach-and-recover yet
score = (32,400 − 30,200) / (32,400 − 28,000) × 0.95 = (2,200 / 4,400) × 0.95 = 0.475
```

**Gaussian MA score:**
```
GMA ≈ $31,600 (slow-moving; price has been below it for 12 days)
prev_close $30,400 < prev_GMA $31,700; current $30,200 still below
depth = (31,600 − 30,200) / 31,600 = 0.044
score = max(0, 0.5 − 0.044 × 5.0) = 0.28
```

**Bollinger score:**
```
Lower band ≈ $29,100
recent_low $29,800 > 29,100 × 1.05 = $30,555? No — wait:
recent_low $29,800 ≤ 29,100 + band_range × 0.05 → touched lower
bands expanding (std rising for 4 bars)
→ score = 1.00
```

**Price confirmation:**
```
= 0.40 × 0.475 + 0.35 × 0.28 + 0.25 × 1.00
= 0.190 + 0.098 + 0.250
= 0.538
```

**HMM confirmation (regime=1, timing=0.946):**
```
= 0.50 + 0.946 × 0.50 = 0.973
```

**Cycle quality:** 0.82

**Composite:**
```
= 0.35 × 0.946 + 0.30 × 0.538 + 0.20 × 0.973 + 0.15 × 0.82
= 0.331 + 0.161 + 0.195 + 0.123
= 0.810
```

**Outcome:** composite 0.810 ≥ 0.65 → **tier boost**
Bear regime + boost → neutral tier allocation = 10% = $10,000
Current position: $5,000 (5%)
drift = +5% → BUY $5,000

```
Order: BTCUSD  BUY  $5,000  (GTC notional)
Alert: BTC_TRADE — action=BUY size=$5,000 target=10.0% regime=bear cycle=0.810
Cycle alert: composite crossed above 0.65
```

---

### Day 82 — Late Cycle (Declining Timing, Monitoring for Exit)

BTC has rallied to $36,400. The timing signal is now well past center and
declining. The HMM has re-confirmed to bull. The composite is fading.

**Inputs:**
```
days_since_last_low = 82
current_price       = $36,400
regime              = 3 (bull)
adaptive_center     = 62
```

**Timing probability:**
```
exp(−0.5 × ((82 − 62) / 12)²) = exp(−0.5 × (1.67)²) = exp(−1.389) ≈ 0.249
```

**Donchian score:**
```
60-bar range: high $36,400 (current), low $28,000, midpoint $32,200
current $36,400 ≥ midpoint → score = 0.0
```

**Gaussian MA score:**
```
GMA ≈ $32,900
current $36,400 > GMA; no crossover this bar
→ score = 0.30 (above GMA but no recent crossover)
```

**Bollinger score:**
```
current $36,400 near upper band ≈ $36,200
position_in_band = (36,400 − 29,200) / (36,200 − 29,200) ≈ 1.03 → above upper
→ score = 0.0 (guard: price in top 20% or above band)
```

**Price confirmation:**
```
= 0.40 × 0.0 + 0.35 × 0.30 + 0.25 × 0.0
= 0.000 + 0.105 + 0.000
= 0.105
```

**HMM confirmation (regime=3/bull, timing=0.249):**
```
= 0.50 − 0.249 × 0.50 = 0.376
```

**Cycle quality:** 0.82

**Composite:**
```
= 0.35 × 0.249 + 0.30 × 0.105 + 0.20 × 0.376 + 0.15 × 0.82
= 0.087 + 0.032 + 0.075 + 0.123
= 0.317
```

**Outcome:** composite 0.317 < 0.65 → **no tier boost**
Bull regime + no boost → allocation = 15% = $15,000
Current position: $10,000 (10% of $100,000 NAV)
drift = +5% = BTC_REBALANCE_THRESHOLD (exactly) → not > threshold → **HOLD**

No order. The strategy holds and waits. If BTC keeps rising the NAV-based
current_allocation will naturally drift closer to target as BTC appreciates.

---

### Summary of the Three-Day Walk-Through

| Day | Days elapsed | Timing prob | Price conf | HMM conf | Composite | Action |
|-----|-------------|------------|------------|---------|-----------|--------|
| 15 | Early | 0.000 | 0.344 | 0.50 | 0.326 | HOLD — no tier shift |
| 58 | Near peak | 0.946 | 0.538 | 0.973 | 0.810 | **BUY** — tier boost fires |
| 82 | Late | 0.249 | 0.105 | 0.376 | 0.317 | HOLD — composite faded |

The cycle engine fires its strongest signal at day 58 — two days before the
62-day adaptive center, when timing probability is at 0.946, the Bollinger lower
band was recently touched with expanding bands, and the bear regime validates
the correction. The composite peaks above 0.80, well above the 0.65 tier-boost
threshold, and a BUY order doubles the BTC position from 5% to 10% of NAV.

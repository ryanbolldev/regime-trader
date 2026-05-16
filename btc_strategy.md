# BTC Trading Strategy — Technical Reference

How the Regime Trader enters, manages, and exits BTC spot positions.
Covers signal inputs, target allocation math, the action decision tree,
order execution, and six end-to-end simulated trade scenarios.

---

## Table of Contents

1. [Overview](#overview)
2. [Where BTC Fits in the Main Loop](#where-btc-fits-in-the-main-loop)
3. [Signal Inputs](#signal-inputs)
   - [HMM Regime](#hmm-regime)
   - [60-Day Cycle Engine](#60-day-cycle-engine)
4. [Target Allocation Calculation](#target-allocation-calculation)
5. [Action Decision Tree](#action-decision-tree)
6. [Order Execution](#order-execution)
7. [Live Account Mode Constraints](#live-account-mode-constraints)
8. [Simulated Trade Scenarios](#simulated-trade-scenarios)
9. [Parameter Reference](#parameter-reference)

---

## Overview

BTC is traded as a spot asset using notional (dollar-amount) orders via Alpaca's
crypto endpoint. Unlike the equity pipeline — which manages share counts,
regime-gated strategies, and a 1% per-trade risk cap — BTC uses an
**allocation-percentage model**: the strategy targets a specific fraction of
total portfolio NAV in BTC and rebalances toward it when drift exceeds a
threshold.

Two independent signals drive every decision:

| Signal | Source | Role |
|--------|--------|------|
| HMM regime (0–4) | `core/hmm_engine.py` | Sets the base allocation tier |
| 60-day cycle composite score (0–1) | `core/cycle_engine.py` | Shifts the allocation tier up or down |

The two signals are combined into a single **target allocation** (0–15% of NAV).
The strategy then computes the dollar drift from the current broker-reported
allocation and decides whether to BUY, HOLD, REDUCE, or EXIT.

---

## Where BTC Fits in the Main Loop

Every 300 seconds (`BAR_INTERVAL_SECS`), `main.py` calls `_run_bar()`, which
iterates over `settings.TICKERS`. When it reaches `"BTC"`, it routes to
`_process_btc()` instead of the equity pipeline. BTC bypasses both the market
hours gate (BTC trades 24/7) and the `LIVE_ACCOUNT_MODE` equity block.

```
_run_bar()
  └─ _process_ticker("BTC")
       └─ _process_btc()
            ├─ cycle_engine.get_cycle_signal(ohlcv)       → CycleSignal
            ├─ broker: get_account() + get_positions()    → nav, buying_power, current_alloc
            ├─ btc_strategy.get_target_allocation(...)    → target_pct
            ├─ btc_strategy.get_action(...)               → BTCAction
            └─ order_executor.submit_crypto_order(...)    → OrderResult | None
```

---

## Signal Inputs

### HMM Regime

The same `HMMEngine` that drives equity decisions classifies each BTC bar into
one of five regimes. Regime is confirmed only after `CONFIRMATION_BARS = 3`
consecutive bars agree on the same label. Until confirmed, the engine returns
`-1` (unconfirmed), and the bar is skipped entirely — no BTC trade is
considered while regime is ambiguous.

| Regime ID | Label | Meaning |
|-----------|-------|---------|
| -1 | unconfirmed | Not enough consecutive agreement — skip bar |
| 0 | crash | Sharp sell-off conditions |
| 1 | bear | Sustained negative trend |
| 2 | neutral | No clear directional trend |
| 3 | bull | Sustained positive trend |
| 4 | euphoria | Overextended — mean-reversion risk |

**Uncertainty flag:** When `HMMEngine.is_uncertain()` is True (more than
`FLICKER_THRESHOLD = 4` regime changes in the last `FLICKER_WINDOW = 20` bars),
the target allocation is scaled down by `UNCERTAINTY_ALLOCATION_FACTOR = 0.60`.
This is applied after regime lookup and tier adjustments.

---

### 60-Day Cycle Engine

`CycleEngine.get_cycle_signal()` returns a `CycleSignal` dataclass.
Two fields directly affect BTC allocation:

| Field | Type | Effect on allocation |
|-------|------|----------------------|
| `composite_score` | float 0–1 | ≥ 0.65 → shift up one allocation tier |
| `failed_cycle` | bool | True → shift down one allocation tier |

**How composite_score is built:**

```
composite = 0.35 × timing_probability     (Gaussian prob of being at a 60-day trough)
          + 0.30 × price_confirmation      (weighted average of three price signals)
          + 0.20 × hmm_confirmation        (crash/bear at cycle timing → bullish)
          + 0.15 × cycle_quality_score     (regularity of recent cycle lengths)
```

The three price confirmation signals:

| Signal | Weight | High score means... |
|--------|--------|---------------------|
| Donchian (60-bar) | 40% | Price near or at the channel floor |
| Gaussian MA crossover | 35% | Price just crossed back above the weighted MA |
| Bollinger Bands (20-bar, 2σ) | 25% | Price touched lower band and bands are expanding |

**Failed cycle:** Set to `True` when the current price falls below the price at
the most recently identified cycle low. This signals the prior low was not a
true bottom — the strategy treats it as a bear signal and reduces exposure.

---

## Target Allocation Calculation

`BTCStrategy.get_target_allocation(regime, cycle_signal, is_uncertain)` runs
these steps in order:

### Step 1 — Base allocation from regime

| Regime | Base allocation | Rationale |
|--------|----------------|-----------|
| 0 crash | 0% | Never hold BTC in a crash — no boost possible |
| 1 bear | 5% | Minimal exposure; preserves optionality at cycle lows |
| 2 neutral | 10% | Moderate core holding |
| 3 bull | 15% | Near-maximum exposure |
| 4 euphoria | 8% | Intentionally below bull — take-profit logic |

If `base == 0.0` (crash regime), the function returns `0.0` immediately.
No tier boost can override a crash regime.

### Step 2 — Cycle tier adjustment (when `BTC_CYCLE_TIER_BOOST = True`)

```
If failed_cycle:
    adj = REGIME_ALLOCATIONS[max(regime - 1, 0)]   ← shift down one tier

Elif composite_score >= 0.65:
    adj = REGIME_ALLOCATIONS[min(regime + 1, 4)]   ← shift up one tier

Else:
    adj = base                                      ← no adjustment
```

Note: Tiers follow regime indices 0–4, not allocation magnitudes.
A bull regime (3) boosted one tier moves to the **euphoria allocation (8%)**,
which is lower than bull (15%). This is intentional — a strong cycle signal at
the bull stage indicates the market may be approaching a peak, so the strategy
pre-emptively trims exposure.

### Step 3 — Uncertainty damping

```
If is_uncertain:
    adj = adj × 0.60
```

### Step 4 — Hard cap

```
target = min(adj, BTC_MAX_ALLOCATION)   # BTC_MAX_ALLOCATION = 0.15 (15%)
```

### Full formula

```
base  = REGIME_ALLOCATIONS[regime]
if base == 0.0: return 0.0

adj   = base
if failed_cycle:          adj = REGIME_ALLOCATIONS[max(regime - 1, 0)]
elif composite ≥ 0.65:    adj = REGIME_ALLOCATIONS[min(regime + 1, 4)]

if is_uncertain:          adj = adj × 0.60
return min(adj, 0.15)
```

---

## Action Decision Tree

`BTCStrategy.get_action()` receives the target allocation computed above plus
the broker-reported `current_allocation` (BTC market value / NAV). It evaluates
conditions in the following priority order:

| Priority | Condition | Action | Size |
|----------|-----------|--------|------|
| 1 | `portfolio_nav == 0` | HOLD | $0 — zero_nav |
| 2 | `target == 0.0` and BTC position is held | EXIT | Full current BTC value |
| 3 | Position object unknown AND `current_alloc ≥ target` AND excess ≤ 5% | HOLD | $0 — within_threshold |
| 4 | Position object unknown AND `current_alloc ≥ target` AND excess > 5% | REDUCE | excess × NAV |
| 5 | Position object unknown AND `current_alloc < target` | falls through to BUY |  |
| 6 | Position known AND `abs(drift) ≤ 5%` | HOLD | $0 — within_threshold |
| 7 | `drift > 0` (under target) | BUY | min(drift × NAV, buying_power) |
| 8 | `drift < 0` (over target) | REDUCE | abs(drift) × NAV |

Where `drift = target_allocation − current_allocation`.

**Priority 5 guard asymmetry:** When the broker position object is unavailable
(API failure or lag after a recent fill), the guard only suppresses action when
at or above target. Under-target always falls through to BUY. This prevents a
position lookup failure from blocking legitimate accumulation.

**Recent-buy unconfirmed guard:** Immediately after a BUY order is submitted,
`record_buy()` stamps a bar counter. For the next 2 bars
(`_BTC_UNCONFIRMED_BARS`), if the broker still reports no position (position
updates lag 1-2 bars on Alpaca), the strategy returns HOLD rather than
re-submitting the same order.

---

## Order Execution

Once `get_action()` returns a non-HOLD action, `_process_btc()` in `main.py`
calls `order_executor.submit_crypto_order()`:

```
BUY or REDUCE/EXIT:
  submit_crypto_order(symbol="BTCUSD", side="buy"/"sell", notional_usd=size_usd, client)

EXIT (full position):
  submit_crypto_order(..., qty=current_position.shares_held)
  ← passes exact units to avoid Alpaca rounding on notional sells
```

`submit_crypto_order()` applies two pre-flight guards before submitting:

1. **Zero-size guard** — returns `None` if `notional_usd ≤ 0`
2. **Duplicate guard** — returns `None` if an open order already exists for
   `BTCUSD` (checked via `client.get_orders()`)

The underlying `AlpacaClient.submit_order_notional()` uses `time_in_force: gtc`
(good-till-cancelled), which Alpaca requires for crypto notional orders.

A successful fill fires a `BTC_TRADE` alert and calls `position_tracker.on_fill(result)`.

---

## Live Account Mode Constraints

When `LIVE_ACCOUNT_MODE = True`, two additional caps are enforced in
`_process_btc()` before any order is submitted:

| Constraint | Threshold | Effect |
|------------|-----------|--------|
| Per-trade size cap | 20% of NAV (`LIVE_MAX_POSITION_PCT`) | Order size is clamped to `nav × 0.20` |
| Total deployed cap | 30% of NAV (`LIVE_MAX_DEPLOYED_PCT`) | BUY is skipped entirely if total broker market value already ≥ `nav × 0.30` |

The deployed cap only blocks BUY orders. REDUCE and EXIT are always permitted
regardless of total deployment. The 20% and 30% caps are intentionally
conservative for early live sessions before the strategy is validated in
production.

---

## Simulated Trade Scenarios

All examples use `BTC_REBALANCE_THRESHOLD = 0.05` (5%), `BTC_MAX_ALLOCATION = 0.15` (15%), and `CYCLE_COMPOSITE_THRESHOLD = 0.65`.

---

### Scenario 1 — Bear Regime + Strong Cycle Signal → Accumulate

**Situation:** Market is in a confirmed bear trend, but the cycle engine is
detecting a high-probability trough. Price touched the lower Bollinger band,
recovered above the Gaussian MA, and the timing Gaussian peaks near day 58 of
the 60-day window.

```
NAV:                $100,000
Regime:             1 — bear
Cycle composite:    0.71   (above 0.65 threshold)
failed_cycle:       False
is_uncertain:       False
Current BTC held:   $0 (0% allocation)
```

**Target allocation calculation:**

```
base         = REGIME_ALLOCATIONS[1] = 5%
composite >= 0.65 → shift to regime 2 → adj = REGIME_ALLOCATIONS[2] = 10%
not uncertain → no damping
target       = min(10%, 15%) = 10%  →  $10,000
```

**Action decision:**

```
drift = 10% - 0% = +10%
Position known (None = no position)
current_alloc (0%) < target (10%) → falls through to BUY (priority 5 → 7)
size = min(10% × $100,000, buying_power) = $10,000
```

**Order submitted:**

```
BTCUSD  BUY  $10,000 (notional, GTC)
Alert: BTC_TRADE — action=BUY size=$10,000 target=10.0% regime=bear cycle=0.71
```

---

### Scenario 2 — Bull Regime + Strong Cycle → Trim to Euphoria Level

**Situation:** BTC has rallied strongly. The HMM now reads bull. The cycle
engine's timing probability is peaking (day 55 of 60), all price signals are
firing. A composite above 0.65 shifts to the euphoria tier — which carries a
*lower* allocation than bull, encoding "we may be near a top."

```
NAV:                $100,000
Regime:             3 — bull
Cycle composite:    0.73   (above 0.65)
failed_cycle:       False
is_uncertain:       False
Current BTC:        $15,000 (15% allocation — was a full bull position)
```

**Target allocation calculation:**

```
base         = REGIME_ALLOCATIONS[3] = 15%
composite >= 0.65 → shift to regime 4 → adj = REGIME_ALLOCATIONS[4] = 8%
not uncertain → no damping
target       = min(8%, 15%) = 8%  →  $8,000
```

**Action decision:**

```
drift = 8% - 15% = -7%
Position known → priority 6: abs(-7%) = 7% > 5% threshold → not HOLD
drift < 0 → REDUCE (priority 8)
size = 7% × $100,000 = $7,000
```

**Order submitted:**

```
BTCUSD  SELL  $7,000 (notional, GTC)
After fill: position ~$8,000 (8% NAV)
Alert: BTC_TRADE — action=REDUCE size=$7,000 target=8.0% regime=bull cycle=0.73
```

---

### Scenario 3 — Crash Regime → Full Exit

**Situation:** A sharp sell-off has pushed the HMM into crash (regime 0).
The program holds BTC from a prior bull position.

```
NAV:                $80,000
Regime:             0 — crash
Cycle composite:    0.20   (irrelevant — crash always returns 0%)
failed_cycle:       False
is_uncertain:       False
Current BTC:        $9,600 (12% allocation)
```

**Target allocation calculation:**

```
base = REGIME_ALLOCATIONS[0] = 0%
base == 0.0 → return 0.0 immediately (no boost possible in crash)
target = 0%
```

**Action decision:**

```
target == 0.0 AND current_value ($9,600) > 0  →  EXIT (priority 2)
size = $9,600 (full current BTC value)
```

**Order submitted:**

```
BTCUSD  SELL  qty=0.096 BTC  (exact units passed to avoid rounding)
After fill: $0 BTC exposure
Alert: BTC_TRADE — action=EXIT size=$9,600 target=0.0% regime=crash cycle=0.20
```

---

### Scenario 4 — Uncertainty Damping → Smaller Buy

**Situation:** HMM has been flickering — more than 4 regime changes in the last
20 bars. `is_uncertain()` returns True. The current regime appears bull but the
engine is not confident.

```
NAV:                $100,000
Regime:             3 — bull
Cycle composite:    0.45   (below 0.65 — no tier boost)
failed_cycle:       False
is_uncertain:       True
Current BTC:        $0 (0% allocation)
```

**Target allocation calculation:**

```
base         = REGIME_ALLOCATIONS[3] = 15%
composite < 0.65 → no tier adjustment → adj = 15%
is_uncertain  → adj = 15% × 0.60 = 9%
target        = min(9%, 15%) = 9%  →  $9,000
```

**Action decision:**

```
drift = 9% - 0% = +9% > 5% threshold
BUY $9,000
```

**Order submitted:**

```
BTCUSD  BUY  $9,000 (notional, GTC)
Alert: BTC_TRADE — action=BUY size=$9,000 target=9.0% regime=bull cycle=0.45
       [note: uncertainty reduced position from $15,000 to $9,000]
```

---

### Scenario 5 — Failed Cycle → Reduce Exposure

**Situation:** BTC broke below the price that marked the prior cycle low. The
cycle engine flags `failed_cycle = True`, indicating the prior bottom was not a
genuine floor. The regime is neutral.

```
NAV:                $75,000
Regime:             2 — neutral
Cycle composite:    0.28   (low; failed cycle dominates)
failed_cycle:       True
is_uncertain:       False
Current BTC:        $9,000 (12% allocation)
```

**Target allocation calculation:**

```
base         = REGIME_ALLOCATIONS[2] = 10%
failed_cycle → shift to regime 1 → adj = REGIME_ALLOCATIONS[1] = 5%
not uncertain → no damping
target        = min(5%, 15%) = 5%  →  $3,750
```

**Action decision:**

```
drift = 5% - 12% = -7%
abs(-7%) = 7% > 5% threshold → REDUCE (priority 8)
size = 7% × $75,000 = $5,250
```

**Order submitted:**

```
BTCUSD  SELL  $5,250 (notional, GTC)
After fill: position ~$3,750 (5% NAV)
Alert: BTC_TRADE — action=REDUCE size=$5,250 target=5.0% regime=neutral cycle=0.28
```

---

### Scenario 6 — Within Threshold → Hold

**Situation:** BTC has drifted slightly below target but not enough to justify
a rebalancing trade.

```
NAV:                $100,000
Regime:             2 — neutral
Cycle composite:    0.50   (below 0.65 — no boost)
failed_cycle:       False
is_uncertain:       False
Current BTC:        $9,200 (9.2% allocation)
```

**Target allocation calculation:**

```
base   = REGIME_ALLOCATIONS[2] = 10%
target = 10%  →  $10,000
```

**Action decision:**

```
drift = 10% - 9.2% = +0.8%
Position known → priority 6: abs(0.8%) = 0.8% ≤ 5% threshold  →  HOLD
```

**No order submitted.** The system logs the HOLD at DEBUG level and moves on.
This prevents excessive churn from small day-to-day price moves that naturally
push allocation a few percent off target.

---

## Parameter Reference

All settings live in `config/settings.py`.

### Regime allocations (hard-coded in `BTCStrategy`)

| Regime | Allocation |
|--------|------------|
| 0 crash | 0% |
| 1 bear | 5% |
| 2 neutral | 10% |
| 3 bull | 15% |
| 4 euphoria | 8% |

### Tunable settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BTC_MAX_ALLOCATION` | 0.15 | Hard cap on any computed target allocation |
| `BTC_REBALANCE_THRESHOLD` | 0.05 | Minimum drift before a trade is placed (5%) |
| `BTC_CYCLE_TIER_BOOST` | True | Enable cycle-driven tier shifts |
| `BTC_TICKERS` | `["BTCUSD"]` | Alpaca crypto symbol used for orders |
| `MSTR_BTC_BETA` | 2.5 | MSTR carries ~2.5× BTC beta; MSTR position value × 2.5 is added to effective BTC exposure before approving new MSTR buys |
| `CYCLE_COMPOSITE_THRESHOLD` | 0.65 | Score required to trigger a tier boost |
| `CYCLE_60D_CENTER` | 60 | Expected cycle length in days |
| `CYCLE_60D_STD` | 12 | Gaussian std for timing probability |
| `UNCERTAINTY_ALLOCATION_FACTOR` | 0.60 | Allocation multiplier when HMM is uncertain |
| `LIVE_MAX_POSITION_PCT` | 0.20 | Per-trade BTC size cap in live mode (20% NAV) |
| `LIVE_MAX_DEPLOYED_PCT` | 0.30 | Total deployed cap; blocks BUY if exceeded (30% NAV) |

### Live vs. paper thresholds

| Constraint | Paper | Live |
|------------|-------|------|
| Per-trade size cap | none (risk manager: 1% equity cap only) | 20% NAV |
| Total deployed cap | none | 30% NAV |
| Daily halt | −3% | −2% |
| Weekly resize | −5% | −3% |
| Peak drawdown lockout | −10% | −5% |

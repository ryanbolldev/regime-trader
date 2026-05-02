# Regime Trader — Technical Reference

This document is the authoritative technical reference for the Regime Trader system. It covers architecture, signal detection, strategy logic, risk management, and safety protocols in depth.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Startup Sequence](#startup-sequence)
3. [Main Loop](#main-loop)
4. [Signal Detection](#signal-detection)
   - [HMM Regime Classification](#hmm-regime-classification)
   - [Feature Engineering](#feature-engineering)
   - [60-Day BTC Cycle Engine](#60-day-btc-cycle-engine)
5. [Trading Strategies](#trading-strategies)
   - [Equity Regime Strategies](#equity-regime-strategies)
   - [BTC Spot Strategy](#btc-spot-strategy)
   - [Wheel Options Strategy](#wheel-options-strategy)
6. [Risk Management & Safety Protocols](#risk-management--safety-protocols)
   - [Circuit Breakers](#circuit-breakers)
   - [Position Sizing](#position-sizing)
   - [Market Hours Gate](#market-hours-gate)
   - [Live Account Mode](#live-account-mode)
   - [Lockfile Guard](#lockfile-guard)
   - [Credential Security](#credential-security)
7. [Order Execution](#order-execution)
8. [Alerts & Monitoring](#alerts--monitoring)
9. [Broker Integration](#broker-integration)
10. [Anti-Lookahead-Bias Guarantees](#anti-lookahead-bias-guarantees)
11. [Configuration Reference](#configuration-reference)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                        main.py                          │
│              RegimeTrader (Orchestrator)                │
└──────┬──────────────────────────────────────┬──────────┘
       │                                      │
       ▼                                      ▼
┌─────────────┐                    ┌──────────────────────┐
│  HMMEngine  │◄──feature_eng──►   │    RiskManager       │
│  per ticker │    (OHLCV→obs)     │  circuit breakers    │
└──────┬──────┘                    └──────────┬───────────┘
       │ regime (0–4)                         │ approve / veto
       ▼                                      ▼
┌─────────────────┐             ┌─────────────────────────┐
│ regime_strat    │             │    order_executor        │
│ btc_strategy    │──signal──►  │  equity / crypto / wheel │
│ wheel_strategy  │             └──────────────┬──────────┘
│ cycle_engine    │                            │
└─────────────────┘                            ▼
                                    ┌──────────────────────┐
                                    │    alpaca_client      │
                                    │  REST API (paper/live)│
                                    └──────────────────────┘
```

The system is deliberately **data-flow, not event-driven**. Every 300 seconds (`BAR_INTERVAL_SECS`) the orchestrator pulls the latest bars, computes signals, gates through risk management, and submits approved orders. There are no async callbacks or WebSocket subscriptions — simplicity and testability are prioritised over sub-second latency, which is appropriate for a daily-bar HMM strategy.

---

## Startup Sequence

On launch, `RegimeTrader.startup()` runs seven steps in order. Any failure in steps 2–4 is fatal (raises `SystemExit` or `RuntimeError`).

| Step | Action | Failure mode |
|------|--------|--------------|
| 1 | Load `.env` via `AlpacaClient` constructor | `ConfigurationError` — bad or missing credentials |
| 2 | Check for lockfile at `trading.lock` | `SystemExit(1)` — another instance may be running |
| 3 | Verify Alpaca account status is ACTIVE or APPROVED | `RuntimeError` — account suspended or restricted |
| 4 | Log market hours status (informational, does not halt) | — |
| 5 | Fetch 2 years of daily bars and train one `HMMEngine` per ticker | Logged warning if any ticker fails; non-fatal |
| 6 | Initialize `RiskManager` with current portfolio NAV | — |
| 7 | Reconcile `position_tracker` from broker positions; cancel lingering orders | Logged warning if reconciliation fails; non-fatal |

After a successful startup, a `STARTUP` alert fires with NAV and market status.

---

## Main Loop

`RegimeTrader.run()` blocks indefinitely, calling `_run_bar()` every `BAR_INTERVAL_SECS` seconds (default 300).

### Per-bar sequence

```
_run_bar():
  1. is_market_open() — check for market close transition (fires daily P&L alert)
  2. For each ticker in settings.TICKERS:
       _process_ticker(ticker)
  3. risk_manager.update(current_nav) — evaluate circuit breakers post-bar
  4. For each newly-fired breaker: fire circuit_breaker alert
```

### Per-ticker pipeline (`_process_ticker`)

```
1. _fetch_bars_with_retry()     — up to 3 attempts, fires data_feed_drop alert on failure
2. feature_engineering.compute_latest()
3. hmm_engine.predict_current() → regime (-1 = unconfirmed, 0–4 = confirmed)
4. If regime != prior: fire regime_change alert
5. If regime == -1: return (no signal until confirmed)
6. If ticker in REFERENCE_TICKERS: return (context only, no trade)
7. If ticker == "BTC": → _process_btc() and return
8. If IS_EQUITY_HOURS_ONLY and market closed: return
9. If LIVE_ACCOUNT_MODE: return (equities disabled in live mode)
10. regime_strategies.get_signal() → Signal
11. risk_manager.approve(signal) → ApprovalResult
12. If not approved: log and return
13. order_executor.submit(signal, symbol=ticker)
14. position_tracker.on_fill(result)
15. Fire trade_placed alert
```

### BTC pipeline (`_process_btc`)

```
1. cycle_engine.get_cycle_signal(ohlcv) → CycleSignal
2. Fetch NAV and buying_power from broker
3. Resolve current BTC position from broker positions
4. btc_strategy.get_target_allocation(regime, cycle_signal, is_uncertain)
5. btc_strategy.get_action(position, target, nav, buying_power, price, ...)
6. If action == HOLD: log and return
7. [Live mode] Log full decision, check 30% deployed cap, cap size at 20% NAV
8. Log "LIVE ACCOUNT: approving order for ..." (live mode only)
9. order_executor.submit_crypto_order(symbol, side, notional_usd)
10. alerts.send_btc_trade_alert()
11. position_tracker.on_fill()
```

### Error handling

- `BrokerUnavailableError` in the main loop → sleep `API_RETRY_WAIT_SECS` (60s) and retry
- Any other unhandled exception → fire `critical_error` alert, call `shutdown()`, exit
- Per-ticker errors are caught independently — one bad ticker cannot stall the loop

---

## Signal Detection

### HMM Regime Classification

**File:** `core/hmm_engine.py`

The `HMMEngine` fits a Gaussian HMM to a sequence of multi-dimensional market observations and assigns each bar to one of five regimes.

#### Model selection

At training time, the engine tests all state counts from `HMM_MIN_STATES` (3) to `HMM_MAX_STATES` (7) and selects the model with the lowest Bayesian Information Criterion (BIC). This avoids overfitting to a fixed number of states.

#### State labelling

HMM states are inherently unordered. The engine maps them to regime labels 0–4 by sorting states on their mean log-return (ascending). The lowest-return state becomes regime 0 (crash), the highest becomes regime 4 (euphoria). When the model has fewer than 5 states, labels are spread linearly so the crash and euphoria endpoints are always covered.

#### Regime confirmation and stability filters

Raw HMM output can flicker between adjacent states on noisy bars. Two filters prevent acting on noise:

| Filter | Parameter | Behaviour |
|--------|-----------|-----------|
| **Confirmation gate** | `CONFIRMATION_BARS = 3` | Regime not confirmed until 3 consecutive bars agree |
| **Flicker suppressor** | `FLICKER_THRESHOLD = 4` in `FLICKER_WINDOW = 20` | `is_uncertain()` returns True if regime changed >4 times in the last 20 bars |

When `is_uncertain()` is True, position sizes are scaled by `UNCERTAINTY_ALLOCATION_FACTOR = 0.60`, reducing all new exposure by 40%.

#### Anti-lookahead guarantee

The engine uses the **forward algorithm only** (one bar at a time, left-to-right). The Viterbi algorithm — which conditions each state assignment on the entire future sequence — is explicitly prohibited. This ensures the live prediction path cannot access data it would not have seen at that timestamp.

---

### Feature Engineering

**File:** `core/feature_engineering.py`

All features are computed from OHLCV bars and fed as observations to the HMM.

| Feature | Computation | Anti-lookahead method |
|---------|-------------|----------------------|
| `log_return` | `ln(close_t / close_{t-1})` | Single-lag diff |
| `realized_vol_20` | Rolling 20-bar std of log returns | Window shifted +1 before rolling |
| `volume_zscore` | `(vol - rolling_mean) / rolling_std` | Means and stds computed on lagged window |
| `hl_range_norm` | `(high - low) / close` | Single bar, no rolling |
| `rsi_14` | 14-period RSI via exponential MA | Standard RSI (no future data) |

Additional symbol-specific features are injected at inference time via `compute_latest()`:

- **MSTR**: `on_chain_score` — placeholder for on-chain Bitcoin metrics
- **BTC**: Cycle detection features (`timing_probability`, `composite_score`, `days_since_last_low`, etc.)

The `validate_no_lookahead()` utility performs a statistical check: it computes the correlation between any feature column and the *next bar's* return. A correlation above the significance threshold raises `LookaheadBiasError` and aborts the run.

---

### 60-Day BTC Cycle Engine

**File:** `core/cycle_engine.py`

Bitcoin historically exhibits approximately 60-day cycles between local lows. The `CycleEngine` detects these cycles probabilistically and produces a composite score used by the BTC strategy to adjust its regime-based allocation.

#### Cycle low detection

1. Identify local minima in the price series using a ±15-bar window
2. Require price to rise >10% (`CYCLE_LOW_CONFIRMATION_PCT`) within 20 bars to confirm a low
3. Score the last 3 candidates (`CYCLE_QUALITY_LOOKBACK`) and select the best by quality

Seeded historical lows (known Bitcoin cycle bottoms) are pre-loaded to bootstrap the model before sufficient history accumulates:

| Date | Price |
|------|-------|
| 2018-12-15 | $3,200 |
| 2020-03-13 | $3,800 |
| 2022-11-21 | $15,500 |

#### Timing probability

Once a cycle low is identified, the engine computes the probability that the *current bar* is at a cycle trough using a Gaussian distribution centred at `CYCLE_60D_CENTER = 60` days with `CYCLE_60D_STD = 12` days. Probability peaks at day 60 and decays symmetrically.

A secondary 4-year macro cycle (`CYCLE_4Y_CENTER = 1458` days) modulates the macro phase label:

| Days from last major low | Macro phase |
|--------------------------|-------------|
| 0–365 | Accumulation |
| 365–730 | Markup |
| 730–1095 | Distribution |
| >1095 | Markdown |

#### Price confirmation scores

Three independent price-based signals are combined to confirm a cycle low:

| Signal | Weight | Logic |
|--------|--------|-------|
| **Donchian** | 40% | Score = 1 - (price position in 60-bar high-low channel). High score = price near channel bottom |
| **Gaussian MA** | 35% | 1.0 if price crosses above Gaussian-weighted MA; 0.5 if riding above; 0.0–0.3 if below |
| **Bollinger Bands** | 25% | Score based on touch of lower band and band expansion (expanding = 1.0) |

#### Composite score

```
composite = 0.35 × timing_probability
          + 0.30 × price_confirmation
          + 0.20 × hmm_regime_score
          + 0.15 × quality_score
```

A score above `CYCLE_COMPOSITE_THRESHOLD = 0.65` triggers a **tier boost** in the BTC strategy (advance one regime tier's allocation). A `failed_cycle` event (price breaks below the confirmed low) triggers a **tier reduction**.

#### CycleSignal output fields

| Field | Type | Description |
|-------|------|-------------|
| `timing_probability` | float 0–1 | Gaussian probability of being at cycle trough |
| `donchian_score` | float 0–1 | Price position in 60-bar channel |
| `gaussian_score` | float 0–1 | Price vs. Gaussian-weighted MA |
| `bollinger_score` | float 0–1 | Bollinger band signal |
| `composite_score` | float 0–1 | Weighted aggregate |
| `days_since_last_low` | int | Days elapsed since identified low |
| `adaptive_window_center` | int | Expected cycle length (default 60) |
| `failed_cycle` | bool | Price broke below confirmed cycle low |
| `translation` | str | "right" (bullish) or "left" (bearish) cycle translation |
| `macro_phase` | str | Accumulation / Markup / Distribution / Markdown |
| `bias` | str | "long", "short", or "neutral" |

---

## Trading Strategies

### Equity Regime Strategies

**File:** `core/regime_strategies.py`

Each HMM regime maps to an immutable `RegimeProfile` that governs all equity trading decisions.

| Regime | Name | Allocation | Leverage | Allow Long | Allow Short | Max New Positions |
|--------|------|------------|----------|------------|-------------|-------------------|
| 0 | Crash | 10% | 1.0× | No | Yes | 1 |
| 1 | Bear | 30% | 1.0× | Yes | No | 3 |
| 2 | Neutral | 60% | 1.0× | Yes | No | 5 |
| 3 | Bull | 90% | 1.1× | Yes | No | 5 |
| 4 | Euphoria | 70% | 1.0× | No | No | 0 |

**Uncertainty modifier:** When `HMMEngine.is_uncertain()` is True, the effective allocation is multiplied by `UNCERTAINTY_ALLOCATION_FACTOR = 0.60`, reducing exposure by 40%.

**Rebalancing threshold:** A signal only recommends rebalancing when `|current_allocation - target_allocation| > REBALANCE_DRIFT_THRESHOLD (5%)`. This prevents excessive churn from small regime oscillations.

**Position sizing formula:**

```
equity_budget     = portfolio_nav × effective_alloc × leverage
per_position_usd  = equity_budget / max_new_positions
risk_cap_usd      = portfolio_nav × PER_TRADE_RISK_CAP (1%)
position_size_usd = min(per_position_usd, risk_cap_usd)
```

The 1% per-trade cap is enforced twice: once here at signal generation, and again in `order_executor._submit_equity_order` as a hard assertion with clamp.

**Strategy class behaviours:**

| Class | Entry signal | Exit signal |
|-------|-------------|-------------|
| `CrashStrategy` | None — flatten all longs | Immediate regime change |
| `BearStrategy` | None — hold existing, tighten stops | Regime change or stop-loss |
| `NeutralStrategy` | Mean-reversion entries | Trailing stop or regime change |
| `BullStrategy` | Momentum entries | Trailing stop or regime change |
| `EuphoriaStrategy` | None — profit-taking only | Target profit or regime change |

---

### BTC Spot Strategy

**File:** `core/btc_strategy.py`

BTC uses a separate allocation table that is independently adjusted by the 60-day cycle score. Unlike equities, BTC trades 24/7 and uses notional (dollar-amount) orders rather than share quantities.

#### Base regime allocations

| Regime | Allocation | Rationale |
|--------|------------|-----------|
| 0 Crash | 0% | Never hold BTC in a crash — always 0 regardless of cycle |
| 1 Bear | 25% | Light exposure; cycle may still produce tradeable lows |
| 2 Neutral | 50% | Moderate; core holding |
| 3 Bull | 75% | Near-maximum exposure |
| 4 Euphoria | 40% | Trim back — euphoria encodes take-profit logic (lower than bull) |

Note that Euphoria (40%) < Bull (75%) intentionally. A strong cycle signal in bull regime boosts to the Euphoria tier's *allocation* (40%), encoding "we are likely near a top."

#### Cycle tier adjustments (`BTC_CYCLE_TIER_BOOST = True`)

```
Base allocation determined by regime (0–4)
If base == 0.0 (crash): return 0.0 immediately — no boost possible
If failed_cycle:        shift down one tier (regime - 1)
If composite_score ≥ CYCLE_COMPOSITE_THRESHOLD (0.65):
                        shift up one tier (regime + 1)
If is_uncertain:        × 0.50
Cap at BTC_MAX_ALLOCATION (0.75)
```

#### Rebalancing actions

| Condition | Action |
|-----------|--------|
| `abs(target - current) ≤ BTC_REBALANCE_THRESHOLD (5%)` | HOLD |
| `target == 0` and position held | EXIT |
| `target > current` | BUY (`size = min(drift × nav, buying_power)`) |
| `target < current` | REDUCE (`size = abs(drift) × nav`) |

#### BTCAction fields

| Field | Description |
|-------|-------------|
| `action` | BUY / SELL / REDUCE / HOLD / EXIT |
| `target_allocation_pct` | Target % of NAV in BTC |
| `size_usd` | Dollar amount to buy or sell |
| `reason` | Human-readable rationale string |
| `regime` | HMM regime at decision time |
| `cycle_score` | Composite cycle score at decision time |
| `confidence` | HMM confidence (0.5 uncertain, 0.8 confirmed) |

---

### Wheel Options Strategy

**File:** `core/wheel_strategy.py`

The wheel strategy generates income by selling cash-secured puts and covered calls in a repeating cycle. It is applied to `WHEEL_TICKERS = ["MSTR", "CVX"]`.

#### State machine

```
CASH ──(sell put)──► PUT_SOLD ──(assigned / early close)──► ASSIGNED
  ▲                                     │ (expiry worthless)     │
  │                                     └─────────────────────►  │
  │                              (sell call)                      │
  └──────────────────────────── CALL_SOLD ◄──────────────────────┘
          (expiry worthless or early close)
```

#### Regime gates

| Action | Allowed regimes |
|--------|----------------|
| Sell put (enter) | Bull (3), Neutral (2) |
| Sell call (enter) | Neutral (2), Euphoria (4) |
| Hold/wait | Any (no new entries when uncertain) |
| Sit out | Crash (0), Bear (1) |

#### Contract selection targets

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `WHEEL_PUT_DELTA_TARGET` | −0.28 | OTM puts — defensible strike below current price |
| `WHEEL_CALL_DELTA_TARGET` | +0.28 | OTM calls — strike above average cost basis |
| `WHEEL_MIN_DTE` | 30 days | Avoids accelerated gamma decay |
| `WHEEL_MAX_DTE` | 45 days | Optimal theta decay zone |

#### Early-close triggers (ordered by priority)

1. **Regime deteriorates** to Bear (1) or Crash (0) — close immediately at market
2. **50% of max profit** captured (`WHEEL_EARLY_CLOSE_PROFIT_PCT = 0.50`)
3. **Loss exceeds 200%** of premium received (`WHEEL_EARLY_CLOSE_LOSS_PCT = 2.00`)
4. **<7 DTE with a loss** (`WHEEL_GAMMA_RISK_DTE = 7`) — gamma risk closes the position

---

## Risk Management & Safety Protocols

### Circuit Breakers

**File:** `core/risk_manager.py`

The `RiskManager` holds absolute veto power over every trade. It evaluates five circuit breakers after each bar's P&L is realised, in priority order:

| Priority | Breaker | Paper Threshold | Live Threshold | Effect |
|----------|---------|-----------------|----------------|--------|
| 1 | Peak-drawdown lockout | −10% from HWM | −5% from HWM | Halt all trading; permanent for session |
| 2 | Daily halt | −3% intraday | −2% intraday | Halt all trading; resets at next `reset_daily()` |
| 3 | Daily halve | −2% intraday | −2% intraday | Halve all new position sizes |
| 4 | Weekly resize | −5% weekly | −3% weekly | Halve all new position sizes until Monday |
| 5 | Per-trade cap | 1% of NAV | 1% of NAV | Hard cap on any single order |

**Approval results:**

| State | `approved` | `size_multiplier` |
|-------|-----------|-------------------|
| Locked or daily halt | False | 0.0 |
| Daily halve only | True | 0.5 |
| Weekly resize only | True | 0.5 |
| Both halve + resize | True | 0.25 |
| Normal | True | 1.0 |

The peak-drawdown lockout **never resets mid-session**. Once it fires, the only way to resume trading is to restart the process (which requires manually deleting the lockfile after investigation).

**HWM (High-Water Mark)** is updated *after* the drawdown check on each bar. This means a new equity high on bar N does not mask a drawdown on the same bar.

---

### Position Sizing

Position sizes are enforced at two independent layers:

**Layer 1 — Signal generation** (`core/regime_strategies.py`):
```python
risk_cap_usd      = portfolio_nav × PER_TRADE_RISK_CAP   # e.g. $1,000 at $100k NAV
position_size_usd = min(per_position_usd, risk_cap_usd)
assert position_size_usd <= risk_cap_usd + 1e-9          # hard assertion
```

**Layer 2 — Order execution** (`core/order_executor.py`):
```python
risk_cap = nav × PER_TRADE_RISK_CAP
assert signal.position_size_usd <= risk_cap + 0.01       # hard assertion
position_size_usd = min(signal.position_size_usd, risk_cap)  # clamp
shares = int(position_size_usd / current_price)
```

Both layers assert, not just clamp. If a bug in `get_signal()` or inflation in the `portfolio_nav` input produces an oversized signal, the assertion fires and the order is rejected with a clear error message rather than silently over-sizing. This was implemented after observing orders placed at 7× the intended cap due to inflated NAV inputs from Alpaca's leveraged portfolio value field.

---

### Market Hours Gate

**File:** `main.py`, controlled by `config/settings.IS_EQUITY_HOURS_ONLY`

When `IS_EQUITY_HOURS_ONLY = True` (default), equity tickers skip the signal/order pipeline entirely when `alpaca_client.is_market_open()` returns False. HMM regime detection and feature computation still run — the system stays warm and regime-aware even during closed hours.

BTC is explicitly exempt: it is routed through `_process_btc()` before the market hours gate and always proceeds regardless of equity market status.

---

### Live Account Mode

**File:** `main.py` + `config/settings.py`, controlled by `LIVE_ACCOUNT_MODE`

When `LIVE_ACCOUNT_MODE = True`, the following constraints activate automatically:

| Constraint | Paper | Live |
|------------|-------|------|
| Equity trading | Enabled | Disabled (BTC only) |
| Per-trade position cap | 1% NAV | 20% NAV (BTC hard cap) |
| Total deployed cap | None | 30% NAV (BTC BUY blocked if exceeded) |
| Daily halt threshold | −3% | −2% |
| Weekly resize threshold | −5% | −3% |
| Peak drawdown lockout | −10% | −5% |
| Order decision logging | Standard | Full reasoning on every decision |
| Pre-submission validation | None | Logs "LIVE ACCOUNT: approving order for {symbol} size ${x}" |
| HOLD decisions | DEBUG level | INFO level with full context |

The live caps are intentionally conservative. The 20% per-trade cap exists to prevent a single BTC allocation from consuming an outsized fraction of the portfolio before the strategy is validated in production. The 30% total deployment cap ensures the system cannot fully commit to crypto exposure in its first live sessions.

`LIVE_ACCOUNT_MODE` defaults to `False`. Changing it requires a deliberate edit to `config/settings.py` — it cannot be toggled via environment variable.

---

### Lockfile Guard

On startup, the bot writes `trading.lock` containing the PID and ISO timestamp:

```
pid=12345 started=2025-01-15T09:30:00+00:00
```

If a lockfile already exists when the bot starts, it fires a `lockfile_present` alert at `critical` severity and raises `SystemExit(1)` immediately — before any trading activity. On clean shutdown (SIGINT, SIGTERM, or unhandled exception), the lockfile is removed and a `SHUTDOWN` alert fires.

**Do not script the deletion of this file.** Its presence after an unclean exit is intentional — it forces a human to investigate the cause before restarting.

---

### Credential Security

**File:** `config/credentials.py`

- All credentials are loaded exclusively from `.env` via `python-dotenv`
- API keys and secrets are **never stored as named instance attributes** on `AlpacaClient`
- API keys are **never logged** at any level
- The base URL is validated to be HTTPS — any plain HTTP URL raises `ConfigurationError` before a connection is attempted
- `.env` is in `.gitignore`

---

## Order Execution

**File:** `core/order_executor.py`

### Equity orders

`submit()` → `_submit_equity_order()` applies the following guards in sequence:

1. **Symbol check** — no empty symbol
2. **Position size check** — reject if `position_size_usd <= 0`
3. **Regime allows long** — check `_PROFILES[regime].allow_long`
4. **NAV fetch + risk cap assertion** — assert signal does not exceed `nav × PER_TRADE_RISK_CAP`; clamp and use local variable for share calculation
5. **Dedup check** — skip if open order or open position already exists for this symbol
6. **Price fetch** — get latest bar close price
7. **Share calculation** — `shares = int(position_size_usd / price)` — whole shares only
8. **Minimum shares** — skip if `shares < 1`

### Crypto orders

`submit_crypto_order()` uses notional (dollar-amount) orders, bypassing the share-calculation step:

1. **Notional check** — return None if `notional_usd <= 0`
2. **Dedup check** — skip if open order already exists for this symbol (positions are NOT checked for sells)
3. **Submit** via `alpaca_client.submit_order_notional()` using `time_in_force: gtc` (required for Alpaca crypto)

On any exception from `submit_order_notional`, the error is logged with full context (`symbol`, `side`, `notional_usd`, exception message) and re-raised.

### Wheel orders

`_submit_wheel_order()` maps `WheelActionType` to broker side:

| WheelActionType | Broker side | Qty |
|----------------|-------------|-----|
| SELL_PUT | sell | 1 contract |
| SELL_CALL | sell | 1 contract |
| CLOSE | buy | 1 contract |
| WAIT / SIT_OUT | — | (skipped) |

---

## Alerts & Monitoring

**File:** `core/alerts.py`

Alerts are dispatched to all configured channels simultaneously (webhook + email). Each channel failure is logged as a warning and swallowed — alert failures never propagate to the main loop.

### Cooldown mechanism

Each `(event_type, symbol)` pair has an independent cooldown bucket (default `ALERT_COOLDOWN_SECONDS = 300`). An alert is suppressed if the same event was sent for the same symbol within the cooldown window. Per-symbol cooldowns allow "BTC regime change" and "SPY regime change" to fire independently without mutual suppression.

Override example: `alerts.set_cooldown("circuit_breaker", 0)` — used in tests to disable cooldown.

### Event catalogue

| Event | Severity | Description |
|-------|----------|-------------|
| `STARTUP` | info | Bot started, includes NAV and market status |
| `SHUTDOWN` | info | Bot stopped, includes reason |
| `REGIME_CHANGE` | info | HMM confirmed regime transition |
| `TRADE_PLACED` | info | Equity or wheel order submitted |
| `BTC_TRADE` | info | BTC spot order (buy/sell/reduce/exit) |
| `CYCLE_SIGNAL` | info/critical | Cycle composite threshold crossed or failed cycle |
| `CIRCUIT_BREAKER` | warning | Any circuit breaker fired |
| `DAILY_PNL` | info | End-of-day unrealised P&L summary |
| `LOCKFILE` | critical | Lockfile present on startup attempt |
| `API_OUTAGE` | warning | Broker unavailable, retrying |
| `DATA_FEED_DROP` | warning | Data feed unavailable after 3 retries |
| `CRITICAL_ERROR` | critical | Unhandled exception in main loop |

---

## Broker Integration

**File:** `broker/alpaca_client.py`

### Supported operations

| Method | Description |
|--------|-------------|
| `get_account()` | Account status, NAV, buying power |
| `get_positions()` | All open positions with market value and P&L |
| `get_orders()` | All open orders |
| `submit_order()` | Market or limit equity/options order (qty-based) |
| `submit_order_notional()` | Market crypto order (dollar-amount, GTC) |
| `cancel_order()` | Cancel by order ID |
| `get_option_chain()` | Full options chain with greeks and quotes |
| `is_market_open()` | Alpaca clock endpoint |

### Error taxonomy

| Exception | HTTP codes | Meaning | Retried? |
|-----------|-----------|---------|---------|
| `AuthError` | 401, 403 | Bad credentials or permissions | No |
| `InsufficientFundsError` | 422 | Order rejected — insufficient buying power | No |
| `RateLimitError` | 429 | Too many requests | No (caller should back off) |
| `BrokerUnavailableError` | 5xx, network | Transient server error | Yes — 3 attempts |

All 422 responses log the full response body at `WARNING` level before classification, regardless of whether the error maps to `InsufficientFundsError`. This ensures the exact rejection reason is always visible in logs.

### Retry policy

The `@_with_retry` decorator retries `BrokerUnavailableError` only:
- Attempt 1: immediate
- Attempt 2: sleep 1s
- Attempt 3: sleep 2s
- After attempt 3: re-raise

---

## Anti-Lookahead-Bias Guarantees

Lookahead bias (accidentally using future data during training or live inference) is the most common source of inflated backtest performance in algorithmic trading. This system enforces three independent layers of protection:

### Layer 1 — Feature computation

All rolling windows are **shifted by one bar** before computing statistics:

```python
# WRONG (lookahead):  df["vol"] = df["log_ret"].rolling(20).std()
# RIGHT (safe):       df["vol"] = df["log_ret"].shift(1).rolling(20).std()
```

The shift ensures that on bar N, the rolling window contains bars N−20 through N−1, never bar N itself.

### Layer 2 — HMM inference algorithm

The `predict_current()` method uses the **forward algorithm** exclusively:

```
Forward algorithm:  α_t(i) = P(o_1,...,o_t, q_t=i | model)
                    — conditions only on past and current observations
```

The Viterbi algorithm (`argmax P(q_1,...,q_T | o_1,...,o_T)`) is prohibited because it performs a backward pass over the entire sequence, conditioning state assignments on future observations.

### Layer 3 — Validation utility

`feature_engineering.validate_no_lookahead(feature_df, ohlcv_df)` computes the correlation between each feature column and the next bar's log return. A statistically significant positive correlation implies the feature has predictive power *before* the fact — which is only possible through lookahead. Any violation raises `LookaheadBiasError` immediately.

---

## Configuration Reference

All parameters live in `config/settings.py`. Never put credentials here.

### Trading universe

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TICKERS` | `["SPY","MSTR","CVX","BTC"]` | Full universe |
| `REFERENCE_TICKERS` | `["SPY"]` | HMM training only |
| `BTC_TICKERS` | `["BTCUSD"]` | Alpaca crypto symbol |
| `WHEEL_TICKERS` | `["MSTR","CVX"]` | Wheel strategy candidates |

### HMM & stability

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HMM_MIN_STATES` | 3 | Minimum states to test |
| `HMM_MAX_STATES` | 7 | Maximum states to test |
| `HMM_TRAIN_BARS` | 504 | ~2 years of daily bars |
| `CONFIRMATION_BARS` | 3 | Bars before confirming regime |
| `FLICKER_WINDOW` | 20 | Lookback for flicker count |
| `FLICKER_THRESHOLD` | 4 | Max changes before uncertain flag |
| `UNCERTAINTY_ALLOCATION_FACTOR` | 0.60 | Size multiplier when uncertain |

### Risk limits

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PER_TRADE_RISK_CAP` | 0.01 | Max 1% NAV per trade |
| `MAX_POSITIONS` | 5 | Concurrent positions |
| `INTRADAY_STOP_WARN` | −0.02 | Halve sizes at −2% |
| `INTRADAY_STOP_HALT` | −0.03 | Halt trading at −3% |
| `WEEKLY_STOP` | −0.05 | Weekly resize at −5% |
| `PEAK_DRAWDOWN_LOCKOUT` | −0.10 | Lockout at −10% from HWM |

### Live account mode overrides

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LIVE_ACCOUNT_MODE` | False | Enable live-mode constraints |
| `LIVE_MAX_POSITION_PCT` | 0.20 | 20% NAV per-trade cap |
| `LIVE_MAX_DEPLOYED_PCT` | 0.30 | 30% NAV total deployment cap |
| `LIVE_INTRADAY_STOP_HALT` | −0.02 | Tighter daily halt |
| `LIVE_WEEKLY_STOP` | −0.03 | Tighter weekly resize |
| `LIVE_PEAK_DRAWDOWN_LOCKOUT` | −0.05 | Tighter lockout |

### BTC cycle detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CYCLE_60D_CENTER` | 60 | Expected cycle length (days) |
| `CYCLE_60D_STD` | 12 | Gaussian std for timing probability |
| `CYCLE_COMPOSITE_THRESHOLD` | 0.65 | Score required for tier boost |
| `CYCLE_DONCHIAN_WEIGHT` | 0.40 | Donchian weight in price confirmation |
| `CYCLE_GAUSSIAN_WEIGHT` | 0.35 | Gaussian MA weight |
| `CYCLE_BOLLINGER_WEIGHT` | 0.25 | Bollinger band weight |
| `BTC_MAX_ALLOCATION` | 0.75 | Hard allocation ceiling |
| `BTC_REBALANCE_THRESHOLD` | 0.05 | Min drift before rebalancing |
| `BTC_CYCLE_TIER_BOOST` | True | Enable cycle-driven tier adjustments |

### Wheel strategy

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WHEEL_PUT_DELTA_TARGET` | −0.28 | Target put delta |
| `WHEEL_CALL_DELTA_TARGET` | +0.28 | Target call delta |
| `WHEEL_MIN_DTE` | 30 | Minimum days to expiration |
| `WHEEL_MAX_DTE` | 45 | Maximum days to expiration |
| `WHEEL_EARLY_CLOSE_PROFIT_PCT` | 0.50 | Close at 50% of max profit |
| `WHEEL_EARLY_CLOSE_LOSS_PCT` | 2.00 | Stop at 200% of premium received |
| `WHEEL_GAMMA_RISK_DTE` | 7 | Close losing positions with <7 DTE |

### Alerts

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ALERT_COOLDOWN_SECONDS` | 300 | Per-symbol per-event suppression window |
| `IS_EQUITY_HOURS_ONLY` | True | Gate equity orders to market hours |

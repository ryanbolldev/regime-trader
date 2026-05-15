# Scanner Pipeline — `scripts/run_scanner.py`

Nightly HMM scanner that screens the live S&P 500 + Nasdaq 100 universe,
fits a Hidden Markov Model per ticker, and produces a ranked LONG/SHORT
watchlist with suggested trade structures.

Run via `python scripts/run_scanner.py` or `make scan`.
Output: `logs/scanner/watchlist_YYYY-MM-DD.{json,md}` + Telegram alert.

---

## Pipeline Overview

```
Stage 0  fetch_universe()   — live S&P 500 + Nasdaq 100 from Wikipedia (~530 tickers)
             │                 (falls back to static list on network failure)
             ▼
Stage 1  UniverseManager    — volume + price filter (20-day avg)
             │
             ▼
Stage 2  fetch_ohlcv        — daily bar history from Alpaca (252 bars)
             │
             ▼
Stage 3  BatchTrainer       — parallel HMM fit + regime classification
             │
             ▼
Stage 4  OptionsEnricher    — IV rank + ATM bid-ask spread
             │
             ▼
Stage 5  Scorer             — composite LONG/SHORT score (0–100)
             │
             ▼
Stage 6  Strategy mapping   — assign trade structure per ticker
             │
             ▼
Stage 7  Reporter           — JSON + Markdown watchlist + Telegram alert
```

---

## Stage 0 — Live Universe Fetch (`fetch_universe`)

`fetch_universe()` pulls current index membership from Wikipedia using
`pandas.read_html()` (requires `lxml`) before any filtering occurs.

| Index | Wikipedia source | Typical constituent count |
|-------|-----------------|--------------------------|
| S&P 500 | `List_of_S%26P_500_companies` — table `#constituents`, column `Symbol` | ~503 |
| Nasdaq 100 | `Nasdaq-100` — first table with a `Ticker` column and ≥ 90 rows | ~100 |

**Deduplication:** S&P 500 tickers are listed first; Nasdaq 100 tickers not
already present are appended. Combined unique count is typically **~530 tickers**.

**Symbol normalisation:** Wikipedia uses dots as separators (`BRK.B`, `BF.B`).
These are converted to Alpaca's slash format (`BRK/B`, `BF/B`) before any
downstream use.

**Fallback behaviour:**

| Failure | Result |
|---------|--------|
| S&P 500 fetch fails | Use static `SP500_NASDAQ100_UNIVERSE` from `config/settings.py` (~189 tickers) |
| Nasdaq 100 fetch fails | Skip Nasdaq additions; S&P 500 list (live or fallback) is still used |
| Both fetches fail | Full static `SP500_NASDAQ100_UNIVERSE` (~189 tickers) |

**Per-run cache:** the fetched list is cached on the `UniverseManager` instance.
Multiple calls to `get_tradeable()` within a single scanner run make only one
set of Wikipedia requests.

---

## Stage 1 — Volume / Price Filter (`UniverseManager`)

20-day daily bars are fetched from Alpaca in batches of 50 tickers.
Both metrics are computed as a simple mean across available bars in the window.

| Filter | Parameter | Default | Action on failure |
|--------|-----------|---------|-------------------|
| Average daily volume | `SCANNER_MIN_VOLUME` | 1,000,000 shares | Excluded — `low_volume` |
| Average closing price | `SCANNER_MIN_PRICE` | $10.00 | Excluded — `low_price` |

A ticker must return at least **5 bars** to be evaluated; fewer bars drops it
silently without incrementing an exclusion counter. If an entire batch API call
fails, all tickers in that batch are kept (fail-open) to avoid false drops on
transient errors.

---

## Stage 2 — OHLCV Fetch

Daily OHLCV bars are fetched for every ticker that survived Stage 1.

| Parameter | Setting | Default |
|-----------|---------|---------|
| Training bars requested | `SCANNER_TRAIN_BARS` | 252 bars (~1 year) |
| Fetch window | `SCANNER_TRAIN_BARS × 1.6` calendar days | ~403 days (accounts for weekends/holidays) |
| Data feed | `SCANNER_DATA_FEED` | `iex` (paper accounts cannot use SIP) |
| Batch size | hardcoded | 50 tickers per API call |
| Minimum bars to proceed | hardcoded | 30 bars |

Tickers with fewer than 30 bars after the fetch are dropped before HMM training.

---

## Stage 3 — HMM Training (`BatchTrainer`)

One `HMMEngine` is trained per ticker in parallel threads.

### Threading & Rate-Limit Handling

| Parameter | Setting | Default |
|-----------|---------|---------|
| Thread pool size | `SCANNER_MAX_WORKERS` | 5 workers |
| Sleep between batches | `SCANNER_BATCH_SLEEP_SECS` | 0.5 s |
| Per-ticker retries on HTTP 429 | `SCANNER_MAX_RETRIES` | 3 |
| Retry backoff | hardcoded | 0.5 s → 1.0 s → 2.0 s (exponential) |

Tickers that exhaust all retries are excluded with reason `rate_limit_exhausted`.

### Train / Holdout Split

```
Full bar history  (up to 252 bars)
  ├── Train set   → HMMEngine.fit()
  │     all bars except the last 40
  └── Holdout     → regime_duration_bars count
        last 40 bars, out-of-sample only
```

| Parameter | Setting | Default |
|-----------|---------|---------|
| Holdout bars | `SCANNER_DURATION_HOLDOUT_BARS` | 40 bars |
| Minimum train bars | hardcoded | 30 bars |

The holdout split prevents artificially inflated duration counts — the model
never sees the holdout data during fitting, so `regime_duration_bars` reflects
genuine out-of-sample regime persistence.

### HMM Model Selection

`HMMEngine` tests state counts from `HMM_MIN_STATES` (3) to `HMM_MAX_STATES`
(7) and selects the model with the lowest BIC score. Each candidate runs up to
`HMM_N_ITER` (500) EM iterations with `HMM_N_INIT` (5) random restarts.

### TickerResult Fields Produced

| Field | Meaning |
|-------|---------|
| `current_regime` | Regime label 0–4 from the holdout; `-1` if unconfirmed |
| `regime_duration_bars` | Consecutive holdout bars in the current regime |
| `bic_score` | Model selection metric — lower is better; `inf` on failure |
| `converged` | Whether EM converged (`monitor_.converged`) |
| `convergence_warning` | `True` when `not converged` |
| `n_states` | Winning state count (3–7) |
| `fit_failed` | `True` if training raised an exception |

---

## Stage 4 — Options Enrichment (`OptionsEnricher`)

IV rank and ATM bid-ask spread are fetched in parallel for every ticker that
did not fail HMM fitting. Network errors per ticker are swallowed (non-fatal);
the ticker proceeds with `iv_rank = None`.

### IV Rank Calculation

1. **Current IV** — median implied volatility of all live-chain options with
   30–45 DTE.
2. **Historical IV range** — 20-day rolling annualised realised volatility of
   the underlying over the lookback window, used as an IV proxy (Alpaca does
   not expose a full historical IV series).
3. **Formula:**

```
iv_rank = (current_iv − iv_low) / (iv_high − iv_low) × 100
```

Clamped to [0, 100]. Returns `None` when fewer than 30 price bars are
available or when no eligible options exist in the 30–45 DTE window.

| Parameter | Setting | Default |
|-----------|---------|---------|
| IV rank lookback | `WHEEL_IV_LOOKBACK_DAYS` | 252 days |
| DTE window | hardcoded | 30–45 DTE |

### ATM Bid-Ask Spread

Tightest (minimum) bid-ask spread across all 30–45 DTE contracts with both
bid and ask quoted.

| Parameter | Setting | Default |
|-----------|---------|---------|
| Maximum liquid spread | `SCANNER_OPTIONS_SPREAD_MAX` | $0.20 |

### Enrichment Flags

| Flag | Set when | Effect |
|------|----------|--------|
| `low_liquidity_options` | Best ATM spread > $0.20 | Strategy overridden to `EQUITY_ONLY` if WHEEL assigned |
| `high_iv_event_risk` | `iv_rank` > 70 | Ticker excluded from scoring entirely |
| `iv_rank is None` | API call failed / no eligible options | IV weight redistributed in scorer |

---

## Stage 5 — Composite Scoring (`Scorer`)

### Pre-Score Exclusions

| Condition | Reason |
|-----------|--------|
| `fit_failed == True` | HMM could not be fitted |
| `current_regime == -1` | Regime unconfirmed (too few holdout bars) |
| `high_iv_event_risk == True` | IV rank > 70; event risk too high |

### Score Components

Each ticker receives independent **LONG** and **SHORT** scores on a 0–100
scale. All weights sum to 1.0.

#### 1. Regime Alignment — 40% weight

| Regime | Label | LONG component | SHORT component |
|--------|-------|---------------|----------------|
| 0 | crash | 0 | 95 |
| 1 | bear | 20 | 80 |
| 2 | neutral | 50 | 50 |
| 3 | bull | 90 | 10 |
| 4 | euphoria | 70 | 30 |

#### 2. Confirmation Quality — 20% weight

| Condition | Component score |
|-----------|----------------|
| `converged == True` and `convergence_warning == False` | 100 |
| `convergence_warning == True` | 50 |
| `converged == False` | 20 |

#### 3. Regime Duration — 15% weight

```
duration_component = min(regime_duration_bars / 20, 1.0) × 100
```

Saturates at **20 bars** — full weight once the regime has persisted for 20
consecutive out-of-sample bars.

#### 4. IV Rank — 15% weight

| Direction | Formula | Rationale |
|-----------|---------|-----------|
| LONG | `max(0, 100 − iv_rank)` | Lower IV → cheaper entry |
| SHORT | `iv_rank` | Higher IV → richer premium to sell |

**When `iv_rank is None`:** the 15% IV weight is redistributed — +7.5% to
regime alignment and +7.5% to confirmation quality. The score remains fully
weighted on the data that is available.

#### 5. Model Quality — 10% weight

| Condition | Component score |
|-----------|----------------|
| `bic_score == inf` | 0 |
| `converged == True` | 80 |
| `converged == False` | 50 |

### Final Score Formula

```
score = (0.40 + regime_bonus)   × regime_component
      + (0.20 + confirm_bonus)  × confirm_component
      + 0.15                    × duration_component
      + iv_weight               × iv_component
      + 0.10                    × quality_component
```

`regime_bonus` and `confirm_bonus` are each 0.075 (and `iv_weight` is 0.0)
only when `iv_rank is None`; otherwise all bonuses are 0.0 and `iv_weight`
is 0.15. Score is capped at 100.0.

### Direction Assignment

```
direction = "LONG"  if long_score >= short_score
            "SHORT" otherwise
```

Every scored ticker is assigned exactly one direction — there is no NEUTRAL
output.

### Score Threshold

| Parameter | Setting | Default |
|-----------|---------|---------|
| Minimum qualifying score | `SCANNER_SCORE_THRESHOLD` | 60 |

All tickers are scored for the distribution report; only those at or above the
threshold appear in the watchlist. The returned list is sorted by
`max(long_score, short_score)` descending.

---

## Stage 6 — Strategy Mapping

A suggested trade structure is assigned based on direction and IV rank.

### With IV Data Available

| Direction | IV Rank | Strategy |
|-----------|---------|----------|
| LONG | ≥ 50 | `CASH_SECURED_PUT` |
| LONG | < 50 | `BUY_EQUITY` |
| SHORT | ≥ 50 | `COVERED_CALL` |
| SHORT | < 50 | `BEAR_SPREAD` |
| NEUTRAL | ≥ 60 | `IRON_CONDOR` |
| NEUTRAL | < 60 | `WHEEL` |

### Without IV Data (regime-only fallback)

| Direction | Regime | Strategy |
|-----------|--------|----------|
| LONG | crash (0) | `AVOID` |
| LONG | bear (1) | `UNDERWEIGHT` |
| LONG | neutral (2) | `WATCH` |
| LONG | bull (3) | `LONG_EQUITY` |
| LONG | euphoria (4) | `REDUCE_LONG` |
| SHORT | crash (0) | `PUT_DEBIT_SPREAD` |
| SHORT | bear (1) | `UNDERWEIGHT` |
| SHORT | neutral (2) | `WATCH` |
| SHORT | bull (3) | `AVOID_SHORT` |
| SHORT | euphoria (4) | `COVERED_CALL` |

### Liquidity Override

If `low_liquidity_options == True` and the assigned strategy contains the
string `"WHEEL"`, the strategy is overridden to `EQUITY_ONLY`.

---

## Stage 7 — Reporting (`Reporter`)

### Output Files

| File | Location | Contents |
|------|----------|----------|
| `watchlist_YYYY-MM-DD.json` | `logs/scanner/` | Full machine-readable results, metadata, score distribution, exclusion counts |
| `watchlist_YYYY-MM-DD.md` | `logs/scanner/` | Human-readable ranked watchlist with distribution chart and exclusion breakdown |

### Telegram Alert

A `scanner_briefing` alert fires via `core.alerts.send()` containing:
- Universe size and qualified ticker count
- Top 5 LONG candidates (ticker, regime, score, IV rank, strategy)
- Top 5 SHORT candidates

### Paper Validation Period

On first run, `logs/scanner/deployment_date.txt` is created. For
`SCANNER_PAPER_ONLY_DAYS` (30) calendar days afterward, every Markdown file
and Telegram alert prepends:

```
⚠️  PAPER VALIDATION PERIOD — Day N/30
Scanner output is for research only. Do not deploy real capital until
the 30-day paper validation window closes...
```

---

## Exclusion Accounting

Every ticker dropped at any stage is counted and included in the report.

| Reason key | Stage | Cause | Hard exclusion? |
|------------|-------|-------|----------------|
| `low_volume` | 1 — Universe | Avg daily volume < 1,000,000 shares | Yes |
| `low_price` | 1 — Universe | Avg close < $10.00 | Yes |
| `fit_failed` | 3 — BatchTrainer | HMM exception (non-rate-limit) | Yes |
| `rate_limit_exhausted` | 3 — BatchTrainer | 3 retries on HTTP 429 all failed | Yes |
| `high_iv_event_risk` | 5 — Scorer | IV rank > 70 | Yes |
| `low_liquidity_options` | 4 — OptionsEnricher | Best ATM spread > $0.20 | No — still scored; strategy overridden |
| `iv_data_unavailable` | 4 — OptionsEnricher | IV fetch failed or no eligible options | No — still scored; IV weight redistributed |

---

## Settings Reference

All parameters live in `config/settings.py`.

| Setting | Default | Stage |
|---------|---------|-------|
| `SP500_NASDAQ100_UNIVERSE` | ~189 curated tickers | 0 — static fallback only |
| `SCANNER_MIN_VOLUME` | 1,000,000 | 1 — UniverseManager |
| `SCANNER_MIN_PRICE` | 10.0 | 1 — UniverseManager |
| `SCANNER_DATA_FEED` | `iex` | 1, 2 — bar requests |
| `SCANNER_TRAIN_BARS` | 252 | 2 — OHLCV fetch, 3 — BatchTrainer |
| `SCANNER_DURATION_HOLDOUT_BARS` | 40 | 3 — BatchTrainer |
| `SCANNER_MAX_WORKERS` | 5 | 3 — BatchTrainer, 4 — OptionsEnricher |
| `SCANNER_BATCH_SLEEP_SECS` | 0.5 | 3 — BatchTrainer |
| `SCANNER_MAX_RETRIES` | 3 | 3 — BatchTrainer |
| `SCANNER_OPTIONS_SPREAD_MAX` | 0.20 | 4 — OptionsEnricher |
| `SCANNER_MAX_IV_RANK` | 70 | 4 — OptionsEnricher, 5 — Scorer |
| `SCANNER_SCORE_THRESHOLD` | 60 | 5 — Scorer |
| `SCANNER_PAPER_ONLY_DAYS` | 30 | 7 — Reporter |

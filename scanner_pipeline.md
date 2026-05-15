# Scanner Pipeline — `scripts/run_scanner.py`

Nightly HMM scanner that screens the S&P 500 + Nasdaq 100 universe, fits
a Hidden Markov Model per ticker, and produces a ranked LONG/SHORT watchlist.

Run via `python scripts/run_scanner.py` or `make scan`.
Output is written to `logs/scanner/watchlist_YYYY-MM-DD.{json,md}`.

---

## Pipeline Overview

```
fetch_universe()  — live S&P 500 + Nasdaq 100 from Wikipedia (~500+ tickers)
        │  (falls back to static list on network failure)
        ▼
1. UniverseManager   — volume + price filter
        │
        ▼
2. fetch_ohlcv       — daily bar history from Alpaca
        │
        ▼
3. BatchTrainer      — parallel HMM fit + regime classification
        │
        ▼
4. OptionsEnricher   — IV rank + ATM bid-ask spread
        │
        ▼
5. Scorer            — composite LONG/SHORT score
        │
        ▼
6. Reporter          — JSON + Markdown + Telegram alert
```

---

## Stage 0 — Live Universe Fetch (`fetch_universe`)

Before any filtering, `fetch_universe()` pulls current index membership from
Wikipedia using `pandas.read_html()` (requires `lxml`).

| Index | Wikipedia source | Typical count |
|-------|-----------------|---------------|
| S&P 500 | `List_of_S%26P_500_companies` — table `#constituents`, column `Symbol` | ~503 |
| Nasdaq 100 | `Nasdaq-100` — first table with a `Ticker` column and ≥ 90 rows | ~100 |

**Deduplication:** S&P 500 tickers are listed first; Nasdaq 100 tickers not
already present are appended. Combined unique count is typically **~530 tickers**.

**Symbol normalisation:** Wikipedia uses dots as separators (`BRK.B`, `BF.B`).
These are converted to Alpaca's slash format (`BRK/B`, `BF/B`) before any
downstream use.

**Fallback behaviour:**
- S&P 500 fetch fails → use static `SP500_NASDAQ100_UNIVERSE` from `config/settings.py`
- Nasdaq 100 fetch fails → skip Nasdaq additions (S&P 500 list still used)
- Both fail → full static `SP500_NASDAQ100_UNIVERSE` (~189 tickers)

**Per-run cache:** the fetched list is cached on the `UniverseManager` instance.
Multiple calls to `get_tradeable()` within a single scanner run make only one
Wikipedia request.

---

## Stage 1 — Volume / Price Filter (`UniverseManager`)

**Data fetched:** 20-day daily bar history in batches of 50 tickers.

### Inclusion Criteria

| Filter | Parameter | Default | Action on failure |
|--------|-----------|---------|-------------------|
| Average daily volume | `SCANNER_MIN_VOLUME` | 1,000,000 shares | Excluded — `low_volume` |
| Average closing price | `SCANNER_MIN_PRICE` | $10.00 | Excluded — `low_price` |

Both metrics are computed over the 20-day window.
If the Alpaca API call fails for a batch, all tickers in that batch are kept
(fail-open) to avoid dropping valid tickers on transient errors.

A ticker must have at least **5 bars** in the window to be evaluated;
anything below 5 bars is dropped without counting toward either exclusion bucket.

---

## Stage 2 — OHLCV Fetch

Daily OHLCV bars are fetched for every ticker that survived Stage 1.

| Parameter | Setting | Default |
|-----------|---------|---------|
| Training bars requested | `SCANNER_TRAIN_BARS` | 252 bars (~1 year) |
| Fetch window | `SCANNER_TRAIN_BARS × 1.6` days | ~403 calendar days (covers weekends/holidays) |
| Data feed | `SCANNER_DATA_FEED` | `iex` (paper accounts cannot use SIP) |
| Batch size | hardcoded | 50 tickers per API call |
| Minimum bars to proceed | hardcoded | 30 bars |

Tickers with fewer than 30 bars after the fetch are dropped before HMM training.

---

## Stage 3 — HMM Training (`BatchTrainer`)

One `HMMEngine` is trained per ticker using **parallel threads**.

### Threading & Rate-Limit Handling

| Parameter | Setting | Default |
|-----------|---------|---------|
| Thread pool size | `SCANNER_MAX_WORKERS` | 5 workers |
| Sleep between batches | `SCANNER_BATCH_SLEEP_SECS` | 0.5 seconds |
| Per-ticker retries on HTTP 429 | `SCANNER_MAX_RETRIES` | 3 retries |
| Retry backoff delays | hardcoded | 0.5 s → 1.0 s → 2.0 s |

Tickers that exhaust all retries are excluded with reason `rate_limit_exhausted`.

### Train / Holdout Split

```
Full bar history (up to 252 bars)
  ├── Train set  → HMMEngine.fit()
  │     (all bars except the last 40)
  └── Holdout    → regime_duration_bars count
        (last 40 bars, out-of-sample only)
```

| Parameter | Setting | Default |
|-----------|---------|---------|
| Holdout bars | `SCANNER_DURATION_HOLDOUT_BARS` | 40 bars |
| Minimum train bars | hardcoded | 30 bars |

The holdout split prevents artificially inflated duration counts — the model
never sees the holdout data during fitting.

### HMM Model Selection

The `HMMEngine` tests state counts from `HMM_MIN_STATES` to `HMM_MAX_STATES`
(3–7) and selects the model with the lowest BIC score. See `core/hmm_engine.py`
for full HMM configuration.

### TickerResult Fields

| Field | Meaning |
|-------|---------|
| `current_regime` | Regime label 0–4 (`-1` if fit failed or unconfirmed) |
| `regime_duration_bars` | Consecutive bars in current regime (out-of-sample) |
| `bic_score` | Model selection score — lower is better; `inf` on failure |
| `converged` | Whether EM algorithm converged (`monitor_.converged`) |
| `convergence_warning` | `True` when `not converged` |
| `n_states` | Winning state count (3–7) |
| `fit_failed` | `True` if training raised an exception |

---

## Stage 4 — Options Enrichment (`OptionsEnricher`)

IV rank and ATM bid-ask spread are fetched in parallel for every ticker
that did not fail HMM fitting. Network errors per ticker are swallowed
(non-fatal); the ticker proceeds without IV data.

### IV Rank

IV rank is computed by `AlpacaClient.get_iv_rank()`:

1. **Current IV** — median implied volatility of all options with 30–45 DTE
   from the live Alpaca option chain.
2. **Historical IV range** — 20-day rolling annualised realised volatility
   of the underlying stock over the lookback window, used as an IV proxy
   (Alpaca does not expose a full IV history).
3. **Formula:**
   ```
   iv_rank = (current_iv − iv_low) / (iv_high − iv_low) × 100
   ```
   Clamped to [0, 100]. Returns `None` when fewer than 30 bars or
   fewer than 1 eligible option are available.

| Parameter | Setting | Default |
|-----------|---------|---------|
| IV rank lookback | `WHEEL_IV_LOOKBACK_DAYS` | 252 days |
| DTE window for current IV | hardcoded | 30–45 DTE |

### ATM Bid-Ask Spread

The tightest (minimum) bid-ask spread across all 30–45 DTE contracts
with both bid and ask quoted.

| Parameter | Setting | Default |
|-----------|---------|---------|
| Max spread for liquid options | `SCANNER_OPTIONS_SPREAD_MAX` | $0.20 |

### Enrichment Flags

| Flag | Condition |
|------|-----------|
| `low_liquidity_options` | Best ATM spread > `SCANNER_OPTIONS_SPREAD_MAX` ($0.20) |
| `high_iv_event_risk` | `iv_rank` > `SCANNER_MAX_IV_RANK` (70) |
| `iv_rank is None` | API call failed or insufficient data |

---

## Stage 5 — Composite Scoring (`Scorer`)

### Pre-Score Exclusions

A ticker is dropped before scoring if any of the following apply:

| Condition | Reason |
|-----------|--------|
| `fit_failed == True` | HMM could not be fitted |
| `current_regime == -1` | Regime unconfirmed (too few holdout bars) |
| `high_iv_event_risk == True` | IV rank > 70; earnings/event risk too high |

### Score Components

Each ticker receives a **LONG score** and a **SHORT score** independently,
both on a 0–100 scale. The weights sum to 1.0.

#### 1. Regime Alignment — 40% weight

Maps the current HMM regime label to a directional score:

| Regime | Label | LONG score | SHORT score |
|--------|-------|-----------|------------|
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
| `converged == False` (and no warning flag) | 20 |

#### 3. Regime Duration — 15% weight

```
duration_component = min(regime_duration_bars / 20, 1.0) × 100
```

Score saturates at **20 bars** (full weight once a regime has persisted
for 20 out-of-sample bars).

#### 4. IV Rank — 15% weight

| Direction | Formula | Rationale |
|-----------|---------|-----------|
| LONG | `max(0, 100 − iv_rank)` | Prefer low IV; cheaper options entry |
| SHORT | `iv_rank` | Prefer high IV; richer premium to sell |

**When IV data is unavailable (`iv_rank is None`):**
The 15% IV weight is redistributed — +7.5% to regime alignment and +7.5%
to confirmation quality — so the total score remains fully weighted on the
available data.

#### 5. Model Quality — 10% weight

| Condition | Component score |
|-----------|----------------|
| `bic_score == inf` (fit failed) | 0 |
| `converged == True` | 80 |
| `converged == False` | 50 |

### Final Score Formula

```
score = 0.40 × regime_component
      + 0.20 × confirm_component
      + 0.15 × duration_component
      + 0.15 × iv_component        (or 0.0 when IV unavailable)
      + 0.10 × quality_component
```

Capped at 100.0.

### Direction Assignment

```
direction = "LONG"  if long_score >= short_score
            "SHORT" otherwise
```

There is no NEUTRAL output — every scored ticker is assigned one direction.

### Score Threshold

| Parameter | Setting | Default |
|-----------|---------|---------|
| Minimum score to qualify | `SCANNER_SCORE_THRESHOLD` | 60 |

Tickers below the threshold are scored (for the distribution report) but
excluded from the watchlist output. The returned list is sorted by
`max(long_score, short_score)` descending.

---

## Stage 6 — Strategy Mapping

After direction and IV rank are determined, a suggested trade structure
is assigned.

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

If `low_liquidity_options == True` and the assigned strategy contains
`"WHEEL"`, the strategy is overridden to `EQUITY_ONLY` (no options).

---

## Stage 7 — Reporting (`Reporter`)

### Output Files

| File | Location | Contents |
|------|----------|----------|
| `watchlist_YYYY-MM-DD.json` | `logs/scanner/` | Full machine-readable results + metadata + distribution + exclusions |
| `watchlist_YYYY-MM-DD.md` | `logs/scanner/` | Human-readable ranked watchlist + distribution chart |

### Telegram Alert

A `scanner_briefing` alert is fired via `core.alerts.send()` containing:
- Universe size and qualified ticker count
- Top 5 LONG candidates (ticker, regime, score, IV rank, strategy)
- Top 5 SHORT candidates

### Paper Validation Period

On first run, `logs/scanner/deployment_date.txt` is created with today's
date. For `SCANNER_PAPER_ONLY_DAYS` (30) calendar days after that date,
every Markdown file and Telegram alert prepends a warning banner:

```
⚠️  PAPER VALIDATION PERIOD — Day N/30
Scanner output is for research only...
```

---

## Exclusion Accounting

The final report tallies every reason a ticker was dropped:

| Reason | Stage | Cause |
|--------|-------|-------|
| `low_volume` | Universe | Avg daily volume < 1,000,000 |
| `low_price` | Universe | Avg close < $10.00 |
| `fit_failed` | BatchTrainer | HMM exception (not rate-limit) |
| `rate_limit_exhausted` | BatchTrainer | 3 retries on HTTP 429 all failed |
| `low_liquidity_options` | OptionsEnricher | ATM spread > $0.20 (informational — not excluded from watchlist) |
| `high_iv_event_risk` | Scorer | IV rank > 70 |
| `iv_data_unavailable` | OptionsEnricher | IV fetch failed or insufficient data (informational — ticker still scored) |

---

## Settings Reference

All parameters live in `config/settings.py`.

| Setting | Default | Used in |
|---------|---------|---------|
| `SCANNER_MIN_VOLUME` | 1,000,000 | UniverseManager |
| `SCANNER_MIN_PRICE` | 10.0 | UniverseManager |
| `SCANNER_TRAIN_BARS` | 252 | fetch_ohlcv, BatchTrainer |
| `SCANNER_DURATION_HOLDOUT_BARS` | 40 | BatchTrainer |
| `SCANNER_MAX_WORKERS` | 5 | BatchTrainer, OptionsEnricher |
| `SCANNER_BATCH_SLEEP_SECS` | 0.5 | BatchTrainer |
| `SCANNER_MAX_RETRIES` | 3 | BatchTrainer |
| `SCANNER_OPTIONS_SPREAD_MAX` | 0.20 | OptionsEnricher |
| `SCANNER_MAX_IV_RANK` | 70 | OptionsEnricher, Scorer |
| `SCANNER_SCORE_THRESHOLD` | 60 | Scorer |
| `SCANNER_PAPER_ONLY_DAYS` | 30 | Reporter |
| `SCANNER_DATA_FEED` | `iex` | fetch_ohlcv, UniverseManager |

# Wheel Strategy — TODO

**Focus:** tastytrade options-data integration → wheel-only application → live execution
**Last updated:** 2026-07-25

See also: [wheel execution memory](../.claude/memory/task_wheel_execution.md), [tastytrade integration memory](../.claude/memory/task_tastytrade_integration.md)

---

## tastytrade Options-Data Integration — COMPLETE

- [x] **Verify tastytrade OAuth2 connection + options data** — connection test green against production on MSTR; OAuth valid, 19 expirations, Greeks/Quote/Summary all 5/5 including live open interest. (`scripts/tastytrade_connection_test.py`)
- [x] **Commit checkpoint** — connection test + `tastytrade` / `truststore` deps committed.
- [x] **Extend `config/credentials.py`** — added `load_tastytrade_credentials()` + `TastytradeCredentials` (ALPACA independently required) and shared `enable_os_trust_store()` helper. Settings flag `OPTIONS_DATA_PROVIDER` + `TASTYTRADE_USE_SANDBOX`. `.env.example` documents `TT_SECRET`/`TT_REFRESH`.
- [x] **Build the adapter** — `wheel_scanner/options_data_tastytrade.py` maps Greeks/Quote/Summary/**Trade** → `WheelOptionLeg`. `volume_today` = `Trade.day_volume` (true parity, chosen). Verified live on MSTR (196 legs, delta/IV/bid/ask/OI populated). Refactored shared `realized_vol_range()` into `options_data.py` so IV Rank math is identical across providers.
- [x] **Wire the selector** — new `wheel_scanner/options_provider.py` dispatch (fails loud on bad value); `scanner.py` + `filters.py` route through it.
- [x] **Tests** — `tests/test_options_provider.py`: 14 mocked tests (dispatch routing/fail-loud/delegation + adapter `_leg_from_events` mapping + `compute_ivr` combination/clamp).
- [x] **Validate parity** — OTM bias (wheel's target delta −0.28 zone) ~0.045, small & systematic, not noise. **Flipped `OPTIONS_DATA_PROVIDER` to `tastytrade` 2026-07-07** — tastytrade now the scanner's discovery source (adds open interest); execution still prices off Alpaca (the actual fill venue).

## Wheel-only pivot — Phase 1 (strip-down, scan-only) — COMPLETE

- [x] **Fork `wheel_main.py`** — wheel-only orchestrator: trains HMM on `WHEEL_REGIME_TICKER` (SPY) for regime context → regime-aware wheel scan (startup + nightly) → candidate alerts + `logs/wheel_state.json`. No BTC/regime/equity trading. `main.py` untouched.
- [x] **Closes `gap_scanner_integration`** — scan runs **in-process with the live regime** via an additive `on_fire` callback on `ScannerScheduler`.
- [x] **Tests** — `tests/test_wheel_main.py` (19): regime wiring, retrain-each-cycle, scan effects, negative no-trading-imports guarantee.
- [x] **Live dry-run** — confirmed repeatedly via `scripts/wheel_scan_once.py`, and now runs continuously as part of the deployed Docker `wheel` service.

## Wheel-only pivot — Phase 2a (execution) — BUILT + DEPLOYED

- [x] **Position entry logic** — `core/wheel_executor.py`: scanner-driven CSP entry, NAV-scaled sizing under `WHEEL_MAX_COLLATERAL_PCT` / `WHEEL_TOTAL_DEPLOYED_PCT` / `MAX_WHEEL_POSITIONS` caps.
- [x] **Position management** — P&L-based early close (50%/200%), gamma-risk DTE close, regime-driven close.
- [x] **Hybrid state store** — `core/wheel_position_store.py`: broker-derived truth (phase, shares) reconciled against persisted economics (premium, cost basis) each pass.
- [x] **Hardening** — market-hours gate (reconciliation/alerts always run; order actions blocked when closed), resting-order cancel/reprice lifecycle.
- [x] **Safety batch** — missing IV rank blocks entry (no silent gate bypass), missing-mark alerts + suspends P&L stops, order-submit failures alert, buy-to-close prices at the ask (marketable), spread-sanity gate (`WHEEL_MAX_SPREAD_PCT`) rejects wide/one-sided books.
- [x] **Fixed 3 separate SIP-feed bugs** — `get_iv_rank`, `realized_vol_range`, `filter_price_and_volume` all defaulted to SIP, which paper accounts can't query; all three now request `DataFeed.IEX` explicitly. Full-repo sweep confirmed no remaining instances.
- [x] **Account dedicated to wheel only** — all other positions closed (NAV flat baseline), `TRADING_ENABLED=False` keeps the equity/BTC system off.
- [x] **Deployed as the `wheel` Docker service** — self-scheduling daemon (`core/scheduler.py` runs a UTC-clock background thread; no external cron/Task Scheduler needed, portable to the eventual Hetzner host). Lockfile on `tmpfs` so a crash never blocks the next auto-restart.
- [x] **Verified end-to-end live** — `WHEEL_EXECUTION_ENABLED=True` confirmed baked into the running container; regime flipped to bull, scanner produced real candidates, execution pass engaged, market-hours gate correctly held off (market closed).

## Wheel-only pivot — Phase 2b (assignment + covered calls) — NOT STARTED

- [ ] Assignment detection → `ASSIGNED` handoff → covered call selling → back to `CASH`.
- [ ] Verify Alpaca paper actually assigns ITM puts at expiry (unconfirmed — today assignment is detect-and-alert only).
- [ ] Assignment cost basis net of premium (latent bug: `get_call_to_sell` requires `strike > cost_basis`, not yet true net-of-premium).

---

## Notes / gotchas (carry forward)

- **TLS:** this machine has a TLS-inspection proxy; every tastytrade HTTPS call needs `truststore.inject_into_ssl()` first, or it fails with `CERTIFICATE_VERIFY_FAILED`. Applies to the adapter and live loop, not just the test.
- **Lazy token refresh:** `Session()` does not hit the network — the SDK refreshes the OAuth token on the first API call, so bad creds surface at the first real request, not at construction.
- **Windows console:** cp1252 can't encode box-drawing glyphs — reconfigure stdout/stderr to UTF-8.
- **`.env` is off-limits** — never read/print credential values; verify via connection tests.
- **Alpaca paper + SIP:** paper accounts are denied "recent SIP data" on stock bar fetches — every `StockBarsRequest` needs `feed=DataFeed.IEX` explicitly, or it silently returns `None`/empty instead of erroring. Hit this three separate times before sweeping the whole repo for it — grep for `StockBarsRequest(` first if a filter/rank looks suspiciously empty.
- **Docker non-root user:** the Dockerfile's `appuser` has `--no-create-home`, so anything that writes to `~/.cache` (yfinance, etc.) fails with `EACCES` unless `HOME` is redirected to a writable dir (`/app`).
- **`wheel_main.py` needs no external scheduler** — it's a self-scheduling daemon (`core/scheduler.py` ticks a UTC clock in a background thread). Just keep the process alive (`restart: unless-stopped`); don't reach for cron or Task Scheduler.

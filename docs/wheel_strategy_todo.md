# Wheel Strategy — TODO

**Focus:** tastytrade options-data integration → wheel-only application → live execution
**Last updated:** 2026-08-22

See also: [wheel execution memory](../.claude/memory/task_wheel_execution.md), [tastytrade integration memory](../.claude/memory/task_tastytrade_integration.md)

---

## tastytrade Options-Data Integration — COMPLETE

- [x] **Verify tastytrade OAuth2 connection + options data** — connection test green against production on MSTR; OAuth valid, 19 expirations, Greeks/Quote/Summary all 5/5 including live open interest. (`scripts/tastytrade_connection_test.py`)
- [x] **Commit checkpoint** — connection test + `tastytrade` / `truststore` deps committed.
- [x] **Extend `config/credentials.py`** — added `load_tastytrade_credentials()` + `TastytradeCredentials` (ALPACA independently required) and shared `enable_os_trust_store()` helper. Settings flag `OPTIONS_DATA_PROVIDER` + `TASTYTRADE_USE_SANDBOX`. `.env.example` documents `TT_SECRET`/`TT_REFRESH`.
- [x] **Build the adapter** — `wheel_scanner/options_data_tastytrade.py` maps Greeks/Quote/Summary/**Trade** → `WheelOptionLeg`. `volume_today` = `Trade.day_volume` (true parity, chosen). Verified live on MSTR (196 legs, delta/IV/bid/ask/OI populated). Refactored shared `realized_vol_range()` into `options_data.py` so IV Rank math is identical across providers.
- [x] **Wire the selector** — new `wheel_scanner/options_provider.py` dispatch (fails loud on bad value); `scanner.py` + `filters.py` route through it.
- [x] **Tests** — `tests/test_options_provider.py`: 15 mocked tests (dispatch routing/fail-loud/delegation + adapter `_leg_from_events` mapping + `compute_ivr` combination/clamp).
- [x] **Validate parity** — OTM bias (wheel's target delta −0.28 zone) ~0.045, small & systematic, not noise. **Flipped `OPTIONS_DATA_PROVIDER` to `tastytrade` 2026-07-07** — tastytrade now the scanner's discovery source (adds open interest); execution still prices off Alpaca (the actual fill venue).

## Wheel-only pivot — Phase 1 (strip-down, scan-only) — COMPLETE

- [x] **Fork `wheel_main.py`** — wheel-only orchestrator: trains HMM on `WHEEL_REGIME_TICKER` (SPY) for regime context → regime-aware wheel scan (startup + nightly) → candidate alerts + `logs/wheel_state.json`. No BTC/regime/equity trading. `main.py` untouched.
- [x] **Closes `gap_scanner_integration`** — scan runs **in-process with the live regime** via an additive `on_fire` callback on `ScannerScheduler`.
- [x] **Tests** — `tests/test_wheel_main.py` (23): regime wiring, retrain-each-cycle, scan effects, dashboard state rows, negative no-trading-imports guarantee.
- [x] **Live dry-run** — confirmed repeatedly via `scripts/wheel_scan_once.py`, and now runs continuously as part of the deployed Docker `wheel` service.

## Wheel-only pivot — Phase 2a (execution) — LIVE, TRADING

- [x] **Position entry logic** — `core/wheel_executor.py`: scanner-driven CSP entry, NAV-scaled sizing under `WHEEL_MAX_COLLATERAL_PCT` / `WHEEL_TOTAL_DEPLOYED_PCT` / `MAX_WHEEL_POSITIONS` caps.
- [x] **Position management** — P&L-based early close (50%/200%), gamma-risk DTE close, regime-driven close.
- [x] **Hybrid state store** — `core/wheel_position_store.py`: broker-derived truth (phase, shares) reconciled against persisted economics (premium, cost basis) each pass.
- [x] **Hardening** — market-hours gate (reconciliation/alerts always run; order actions blocked when closed), resting-order cancel/reprice lifecycle.
- [x] **Safety batch** — missing IV rank blocks entry (no silent gate bypass), missing-mark alerts + suspends P&L stops, order-submit failures alert, buy-to-close prices at the ask (marketable), spread-sanity gate (`WHEEL_MAX_SPREAD_PCT`) rejects wide/one-sided books.
- [x] **Fixed 3 separate SIP-feed bugs** — `get_iv_rank`, `realized_vol_range`, `filter_price_and_volume` all defaulted to SIP, which paper accounts can't query; all three now request `DataFeed.IEX` explicitly. Full-repo sweep confirmed no remaining instances.
- [x] **Account dedicated to wheel only** — all other positions closed (NAV flat baseline), `TRADING_ENABLED=False` keeps the equity/BTC system off.
- [x] **Deployed as the `wheel` Docker service** — self-scheduling daemon (`core/scheduler.py` runs a UTC-clock background thread; no external cron/Task Scheduler needed, portable to the eventual Hetzner host). Lockfile on `tmpfs` so a crash never blocks the next auto-restart.
- [x] **Verified end-to-end live** — `WHEEL_EXECUTION_ENABLED=True` baked into the running container; regime bull, scanner producing real candidates, execution pass engaged every `WHEEL_EXEC_INTERVAL_SECS` (900s), market-hours gate holding off orders when closed.
- [x] **First real CSPs sold** — the loop is no longer just gated dry-running; it has opened and closed live paper positions on its own (see live state below).
- [x] **Managed set + position cap made broker-derived** (2026-08-22) — `open_tickers()` and `run_once`'s `open_count` both read the store, so the "always managed even if the scanner drops them" guarantee was only as strong as `wheel_positions.json`. A lost state file could have orphaned an open short put (no early close, no gamma close, no regime close) *and* silently zeroed the position count, letting `MAX_WHEEL_POSITIONS` be breached by exactly the legs it had forgotten. Both now derive from `get_positions()` via `_open_wheel_tickers()`, which reuses `_broker_state()` so it cannot drift from what `reconcile()` decides.

### Live state — as of 2026-08-22 (`logs/wheel_positions.json`, `logs/wheel_state.json`)

- NAV **$92,900**, regime **bull**, last scan 2026-08-21 11:11 UTC, 24 candidates.
- **Open:** `PCVX` 5× $60 put exp 2026-09-18 ($3,635 premium), `ROAD` 3× $105 put exp 2026-09-18 ($2,370 premium). Both `PUT_SOLD`, 27 DTE.
- **Closed round trip:** `MSTR` back to `CASH`, $3,270 lifetime gross premium collected (`premium_collected_total` is gross — it is not netted against buy-to-close cost, so it is not P&L).
- 40 tickers tracked in the store; the rest sit at `CASH`.
- `MAX_WHEEL_POSITIONS = 2` is currently **binding** — every scan pass logs "at max positions (2) — no new entry" for all 20+ ranked candidates. Revisit the cap (or the log level) now that entry is proven.

## Wheel dashboard surfacing — COMPLETE (2026-08-12, commit `c9f598a`)

- [x] **Wheel panel in `dashboard/app.py`** — Row 4 reads `logs/wheel_state.json`: regime / NAV / open-leg count / candidate count, plus open-position and candidate tables. A missing file just means the wheel service isn't running.
- [x] **`wheel_main._position_rows()`** — formats live legs (phase, contract, contracts, DTE, shares, cost basis, premium) for the dashboard.
- [x] **State written on the execution cadence, not the scan cadence** — `_write_wheel_state()` now also runs at the end of each execution pass, so a position opened and closed between daily scans still shows up.
- [x] **OCC symbol helpers** — `format_occ_symbol()` / `_parse_occ_symbol()` in `broker/alpaca_client.py` turn `PCVX260918P00060000` into `PCVX $60 Put exp 2026-09-18` and back out the DTE.

## Wheel-only pivot — Phase 2b (assignment + covered calls) — NOT STARTED

- [ ] Assignment detection → `ASSIGNED` handoff → covered call selling → back to `CASH`.
- [ ] Verify Alpaca paper actually assigns ITM puts at expiry (unconfirmed — today assignment is detect-and-alert only).
- [ ] Assignment cost basis net of premium (latent bug: `core/wheel_strategy.py:198` skips any call with `strike <= cost_basis`; `cost_basis` is the raw assignment price, not net of premium collected — confirmed still present).

## Open infrastructure items

- [ ] **`SCANNER_RUN_UTC_HOUR = 11`** (`config/settings.py:196`) — 11:00 UTC is 6 AM ET / 5 AM CT, i.e. **pre-market**. Wrong window for options data quality; the scan feeding wheel entry runs before the book is real. Unfixed.
- [ ] **Legacy `scanner` compose service is broken** — `docker-compose.yml:44-58` runs `cron -f`, but the slim image has no cron binary, so it crash-loops. The container is not currently present on the host at all. Unrelated to the wheel path (that scan runs in-process via `ScannerScheduler`), but it means `scripts/run_scanner.py` never fires on schedule.
- [ ] **Revisit `MAX_WHEEL_POSITIONS = 2`** — binding on every pass now that two legs are open; nothing new can enter regardless of score.

---

## Notes / gotchas (carry forward)

- **TLS:** this machine has a TLS-inspection proxy; every tastytrade HTTPS call needs `truststore.inject_into_ssl()` first, or it fails with `CERTIFICATE_VERIFY_FAILED`. Applies to the adapter and live loop, not just the test.
- **Lazy token refresh:** `Session()` does not hit the network — the SDK refreshes the OAuth token on the first API call, so bad creds surface at the first real request, not at construction.
- **Windows console:** cp1252 can't encode box-drawing glyphs — reconfigure stdout/stderr to UTF-8.
- **`.env` is off-limits** — never read/print credential values; verify via connection tests.
- **Alpaca paper + SIP:** paper accounts are denied "recent SIP data" on stock bar fetches — every `StockBarsRequest` needs `feed=DataFeed.IEX` explicitly, or it silently returns `None`/empty instead of erroring. Hit this three separate times before sweeping the whole repo for it — grep for `StockBarsRequest(` first if a filter/rank looks suspiciously empty. Note two spellings are in use and both are correct: the wheel/live path passes `feed=DataFeed.IEX` directly, while `core/scanner/universe.py` and `scripts/run_scanner.py` pass `feed=SCANNER_DATA_FEED` (`'iex'` in `settings.py:195`). Don't "fix" the latter — but do flip it with the rest when the account goes live on SIP.
- **Docker non-root user:** the Dockerfile's `appuser` has `--no-create-home`, so anything that writes to `~/.cache` (yfinance, etc.) fails with `EACCES` unless `HOME` is redirected to a writable dir (`/app`).
- **`wheel_main.py` needs no external scheduler** — it's a self-scheduling daemon (`core/scheduler.py` ticks a UTC clock in a background thread). Just keep the process alive (`restart: unless-stopped`); don't reach for cron or Task Scheduler.
- **The wheel container stops when the host does** — `regime_trader-wheel-1` last exited 137 on SIGTERM at 2026-08-22 01:44 UTC (host/Docker Desktop shutdown), and stays down until Docker comes back up. `restart: unless-stopped` recovers it automatically, but on this Windows host that means *nothing runs while the machine is off* — a real gap for a strategy holding open short puts. This is the argument for the Hetzner move.
- **Wheel logs live only in `docker logs`** — `logs/regime_trader_*.log` are from the equity loop; the wheel service logs to stdout, so container recreation loses the history (current logs only reach back to 2026-08-20). `logs/wheel_positions.json` survives on the bind mount.
- **Know which store answers which question.** The **broker** is the truth for *what is open* — phase, shares, cost basis, active contract, contract count, and the open leg's premium are all rederived from `get_positions()` on every pass by `_broker_state()`, and broker wins on conflict. The **JSON** is the truth only for what the broker cannot tell you: `premium_collected_total` and `entry_regime`, both stamped on the phase-transition edge. Lose the file and the open leg self-heals (a `None` record replays the CASH→PUT_SOLD edge and re-reads the premium from `avg_entry_price`), but lifetime premium across *closed* legs is gone for good and you get one spurious transition alert.
- **`premium_collected_total` is gross, not P&L** — `wheel_position_store.py:189` only ever adds premium on sell-to-open; buy-to-close debits are never subtracted. Don't read it as profit.

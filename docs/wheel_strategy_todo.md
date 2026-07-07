# Wheel Strategy — TODO

**Focus:** tastytrade options-data integration → wheel-only application
**Last updated:** 2026-07

See also: [tastytrade integration memory](../.claude/memory/task_tastytrade_integration.md)

---

## tastytrade Options-Data Integration

- [x] **Verify tastytrade OAuth2 connection + options data** — connection test green against production on MSTR; OAuth valid, 19 expirations, Greeks/Quote/Summary all 5/5 including live open interest. (`scripts/tastytrade_connection_test.py`)
- [x] **Commit checkpoint** — connection test + `tastytrade` / `truststore` deps committed.
- [x] **Extend `config/credentials.py`** — added `load_tastytrade_credentials()` + `TastytradeCredentials` (ALPACA independently required) and shared `enable_os_trust_store()` helper. Settings flag `OPTIONS_DATA_PROVIDER` (defaults `alpaca`) + `TASTYTRADE_USE_SANDBOX`. `.env.example` documents `TT_SECRET`/`TT_REFRESH`.
- [x] **Build the adapter** — `wheel_scanner/options_data_tastytrade.py` maps Greeks/Quote/Summary/**Trade** → `WheelOptionLeg`. `volume_today` = `Trade.day_volume` (true parity, chosen). Verified live on MSTR (196 legs, delta/IV/bid/ask/OI populated). Refactored shared `realized_vol_range()` into `options_data.py` so IV Rank math is identical across providers.
- [x] **Wire the selector** — new `wheel_scanner/options_provider.py` dispatch (fails loud on bad value); `scanner.py` + `filters.py` route through it. Default `alpaca`; all 75 scanner tests pass.
- [x] **Tests** — `tests/test_options_provider.py`: 14 mocked tests (dispatch routing/fail-loud/delegation + adapter `_leg_from_events` mapping + `compute_ivr` combination/clamp). Full suite **994 passed**, no regressions.
- [~] **Validate parity** — harness enhanced (signed bias, corr, IV-by-moneyness). After-hours finding: the scary IV gap is **concentrated in ITM puts (0.16)**; in the **wheel's OTM zone (target delta −0.28) it's ~0.045 with a consistent −0.045 bias** — small & systematic, not noise. Delta/quotes parity. **Scheduled** Task Scheduler run `RegimeTrader-ParityCheck` fires **Tue 2026-07-07 08:35 CT** → `logs/parity_check_2026-07-07.txt`. Decision rule: OTM bias →0 = flip cleanly; bias holds ≈−0.045 = flip + nudge `WHEEL_MIN_IV_RANK` down ~2. **GATE still needs the market-hours file reviewed** before flipping `OPTIONS_DATA_PROVIDER`.

## Wheel-only pivot — Phase 1 (strip-down, scan-only)

- [x] **Fork `wheel_main.py`** — wheel-only orchestrator: trains HMM on `WHEEL_REGIME_TICKER` (SPY) for regime context → regime-aware wheel scan (startup + nightly) → candidate alerts + `logs/wheel_state.json`. No BTC/regime/equity trading, no execution. `main.py` untouched.
- [x] **Closes `gap_scanner_integration`** — scan now runs **in-process with the live regime** (was subprocess at default neutral). Added additive `on_fire` callback to `ScannerScheduler`.
- [x] **Tests** — `tests/test_wheel_main.py` (19): regime wiring, retrain-each-cycle, scan effects, and a negative no-trading-imports guarantee. Full suite **1013 passed**.
- [ ] **Optional live dry-run** — `python wheel_main.py` (startup + one real scan) to confirm end-to-end. Blocking/long; hits live data.

## Wheel-only pivot — Phase 2 (execution, LATER spec)

- [ ] **Wheel execution** — CSP entry → management → assignment → covered calls → risk sizing, via **Alpaca paper options**. Its own spec + verification gates.

---

## Notes / gotchas (carry forward)

- **TLS:** this machine has a TLS-inspection proxy; every tastytrade HTTPS call needs `truststore.inject_into_ssl()` first, or it fails with `CERTIFICATE_VERIFY_FAILED`. Applies to the adapter and live loop, not just the test.
- **Lazy token refresh:** `Session()` does not hit the network — the SDK refreshes the OAuth token on the first API call, so bad creds surface at the first real request, not at construction.
- **Windows console:** cp1252 can't encode box-drawing glyphs — reconfigure stdout/stderr to UTF-8.
- **`.env` is off-limits** — never read/print credential values; verify via connection tests.

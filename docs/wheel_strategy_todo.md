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
- [ ] **Wire the selector** — `OPTIONS_DATA_PROVIDER = "alpaca" | "tastytrade"` in `config/settings.py` + selection in `wheel_scanner`.
- [ ] **Tests** — mocked unit tests (match existing all-mocked suite) + a guarded live smoke test.
- [ ] **Validate parity** — diff `WheelOptionLeg` output tastytrade vs Alpaca on MSTR before flipping the provider flag.

## Follow-on (separate spec)

- [ ] **Wheel-only pivot** — disable HMM / regime / BTC trading so this becomes strictly a wheel-strategy app. Strategy change → verification gate + its own spec. Kept out of the integration above.

---

## Notes / gotchas (carry forward)

- **TLS:** this machine has a TLS-inspection proxy; every tastytrade HTTPS call needs `truststore.inject_into_ssl()` first, or it fails with `CERTIFICATE_VERIFY_FAILED`. Applies to the adapter and live loop, not just the test.
- **Lazy token refresh:** `Session()` does not hit the network — the SDK refreshes the OAuth token on the first API call, so bad creds surface at the first real request, not at construction.
- **Windows console:** cp1252 can't encode box-drawing glyphs — reconfigure stdout/stderr to UTF-8.
- **`.env` is off-limits** — never read/print credential values; verify via connection tests.

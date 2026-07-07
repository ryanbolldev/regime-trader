---
name: task_tastytrade_integration
description: tastytrade OAuth2 as read-only options-data provider for wheel_scanner (open interest source); design resolved, build pending
metadata:
  type: project
---

Adding **tastytrade as a read-only options-chain data provider** for `wheel_scanner`. It supplements Alpaca (does NOT replace it): Alpaca keeps equities/OHLCV/execution; tastytrade supplies options data — chiefly **open interest**, the field Alpaca lacks (see [[data_alpaca_options]]). Data-only, so running against **live** is acceptable; execution stays gated on Alpaca paper. Related: [[arch_wheel]], [[gap_wheel_live]], [[task_wheel_integration]].

**Auth (OAuth2):** use the maintained `tastyware/tastytrade` SDK (`pip install tastytrade`, v13.x). `Session(provider_secret=$TT_SECRET, refresh_token=$TT_REFRESH, is_test=False)`; the SDK auto-refreshes the 15-min access token. Two env vars — `TT_SECRET` (OAuth client secret) and `TT_REFRESH` (refresh token) — are present in `.env` (confirmed by Ryan, 2026-07). `.env` is off-limits — see [[feedback_env_offlimits]]; verify creds via a connection test, not by reading them.

**Fetch model (resolved — the key design fact):** tastytrade is NOT a one-shot snapshot like Alpaca. Two stages:
1. REST `NestedOptionChain` → chain structure (expirations, strikes, each contract's `streamer_symbol`).
2. DXLink websocket streaming → subscribe streamer symbols and collect events:
   - **Greeks** → `delta`, `volatility` (= IV)
   - **Quote** → `bid_price`, `ask_price`
   - **Summary** → `open_interest` (int, confirmed in SDK `dxfeed/summary.py`)

Caveat: Summary carries `prev_day_volume`, not today's volume. `WheelOptionLeg.volume_today` (today's contracts) needs the **Trade** event's `day_volume` if same-day accuracy matters.

**Build plan (spec approved, not yet built):** target contract is `WheelOptionLeg` in `wheel_scanner/options_data.py`. New `broker/tastytrade_client.py` (isolate all tastytrade specifics, mirror alpaca_client pattern) + new `wheel_scanner/options_data_tastytrade.py` emitting `WheelOptionLeg` + config flag `OPTIONS_DATA_PROVIDER = "alpaca" | "tastytrade"`. Extend `config/credentials.py` with `load_tastytrade_credentials()` (keep ALPACA_* required independently). OHLCV/stock bars stay on Alpaca; `compute_ivr` becomes hybrid (IV from tastytrade, realized-vol bars from Alpaca). Do NOT formalize a full BrokerClient ABC yet (data-only scope; premature). First deliverable: a connection test proving OAuth + a small chain pull.

**Connection verified (2026-07):** `scripts/tastytrade_connection_test.py` passes end-to-end against **production** on MSTR — OAuth valid, 19 expirations, and Greeks/Quote/Summary all 5/5 including live `open_interest` (confirmed the Alpaca gap is closed).

**Two non-obvious gotchas (both cost debugging time — carry forward to the adapter and live loop):**
1. **TLS:** this Windows machine has a TLS-inspection proxy (AV/corporate) whose root CA is in the Windows store but not `certifi`, so every tastytrade HTTPS call fails with `CERTIFICATE_VERIFY_FAILED` unless `truststore.inject_into_ssl()` runs first. Added `truststore` to requirements; call `inject_into_ssl()` at startup **anywhere** tastytrade makes network calls, not just the test.
2. **Lazy token refresh:** `Session()` construction does NOT hit the network — the SDK refreshes the OAuth token on the *first* API call. So bad creds surface at the first real request, not at construction. Don't claim "creds valid" just because `Session()` returned.
   (Also: Windows console is cp1252 — reconfigure stdout/stderr to UTF-8 for box-drawing output.)

**Adapter built (2026-07):** `wheel_scanner/options_data_tastytrade.py` mirrors the Alpaca provider's fetch surface (`fetch_put_chain`, `count_expiry_cycles`, `compute_ivr`, `build_session`) and emits `WheelOptionLeg`. Verified live on MSTR (196 legs; delta/IV/bid/ask/OI populated). Decisions/gotchas:
- **`volume_today` = `Trade.day_volume`** (4th DXLink subscription, chosen for true parity with Alpaca's today's-volume semantics). It's `None` when a contract hasn't traded today / outside market hours — correct, not a bug.
- **Persistent event loop required:** a tastytrade `Session`'s async httpx client binds to the first event loop it runs on, so `asyncio.run()` (closes its loop) breaks the *next* call with "Event loop is closed". The adapter reuses one module-level loop via `_run()`. Any future async tastytrade code must do the same.
- **Shared IV-Rank math:** extracted `realized_vol_range()` into `options_data.py`; both providers use it so only the current-IV source differs. All 75 scanner tests pass after the refactor.
- **Logging:** `build_session()` raises the `tastytrade` logger to WARNING (it attaches its own DEBUG handler that otherwise logs every websocket frame — ~200KB per fetch).
- Perf note: `_collect` waits for all symbols or `_STREAM_TIMEOUT` (15s); Trade events rarely arrive for every strike, so each `fetch_put_chain` tends to hit the full 15s. Fine for nightly/MSTR-only; revisit if scanning a large universe.

**Selector + tests done (2026-07):** `wheel_scanner/options_provider.py` dispatches on `OPTIONS_DATA_PROVIDER` (fails loud on bad value); `scanner.py`/`filters.py` route through it. `tests/test_options_provider.py` (14 tests). Full suite **994 passed**. Flag still defaults `alpaca`.

**Parity — GATE NOT YET PASSED.** `scripts/tastytrade_parity_check.py` diffs Alpaca vs tastytrade `WheelOptionLeg` by (expiration, strike). Preliminary **after-hours** MSTR run (21-45 DTE): delta median|Δ|=0.014 (good); **IV median|Δ|=0.19 (too large — must investigate)**; bid/ask gaps look like after-hours staleness; Alpaca returned `open_interest=None` on all 169 matched (reinforces why we want tastytrade OI, but makes OI uncomparable); tastytrade offers more strikes (196 vs 169). Do NOT flip `OPTIONS_DATA_PROVIDER` to tastytrade until a **market-hours** re-run resolves the IV gap. This is a human-sign-off verification gate (data-source change).

**Wheel-only pivot — Phase 1 DONE (2026-07).** Decisions (interview): fork a new entry point (not rewrite main.py), keep HMM as regime input, scan-only this phase, Alpaca paper options as the eventual execution venue. Built `wheel_main.py` (`WheelTrader`): trains HMM on `settings.WHEEL_REGIME_TICKER` (=SPY market proxy) → regime-aware `WheelScanner(regime_label=...)` at startup + nightly → candidate alerts + `logs/wheel_state.json`. No BTC/regime/equity/execution paths (enforced by a negative-import test). `main.py` untouched. Added additive `on_fire` callback to `ScannerScheduler` so the scan runs **in-process with live regime** — closes [[gap_scanner_integration]] (previously subprocess at default neutral). Retrains HMM each cycle (mitigates [[issue_hmm_staleness]]). `tests/test_wheel_main.py` (19 tests); full suite **1013 passed**. **Phase 2 (execution: CSP entry/management/assignment/covered calls via Alpaca paper options) is a separate later spec + verification gate.** See [[task_wheel_integration]].

**Dry-run (2026-07, exit 0) found + fixed two bugs in wheel_main:**
1. **Regime always unconfirmed → scan silently ran neutral.** `HMMEngine.predict_current` has a 3-bar confirmation gate carried as instance state; a single one-shot predict never confirms and returns -1. Fixed: `_train_and_predict_regime` now replays the last `CONFIRMATION_BARS + FLICKER_WINDOW` (~23) feature rows through predict_current so the gate settles like the live loop. Verified: SPY now returns a confirmed regime (2) instead of -1.
2. **cp1252 logging crash** on the HMM's `state→regime` arrow — fixed by UTF-8 file handler + stdout reconfigure in `wheel_main._setup_logging` (same class of bug as the connection-test console issue; [[env_tls_proxy]] is unrelated but note main.py's `_setup_logging` has the SAME latent cp1252 bug).

Dry-run also showed environment/data limits (not wheel_main bugs, scanner handles them): Wikipedia universe fetch 403 → static 120-ticker fallback; Alpaca paper "subscription does not permit SIP data"; market closed → 0 candidates (meaningless after-hours/paper). Full wheel scan took ~482s.

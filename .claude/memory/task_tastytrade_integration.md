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

**Bigger pivot flagged (separate spec):** Ryan intends this app to become "strictly a wheel strategy application" and to **disable all other trading** (HMM/regime/BTC engines). That's a strategy change hitting the verification gates and needs its own spec — deliberately kept out of this integration.

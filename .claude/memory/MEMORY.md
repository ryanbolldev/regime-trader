# Regime Trader — Memory Index

**Last Updated**: 2026-06-09  
**Project Root**: `c:\Users\bollr\regime_trader`

This index tracks decisions, architecture, blockers, and state across sessions. Each memory file is listed below with a one-line hook explaining its relevance.

---

## User & Collaboration

- [User preferences](user_preferences.md) — Ryan: direct feedback, values thorough specs before building, prefers parallel agents over sequential work
- [.env is off-limits](feedback_env_offlimits.md) — never read/print credential values; verify creds via connection tests, not by inspecting secrets

---

## Architecture & Design

- [HMM regime classification](arch_hmm.md) — 5-state Gaussian HMM (crash/bear/neutral/bull/euphoria); forward-only to avoid lookahead bias; trained daily on OHLCV
- [BTC cycle engine](arch_btc_cycle.md) — 60-day probabilistic cycle detection; composite score (timing + price + HMM + quality); threshold crossing alerts
- [Risk manager circuit breakers](arch_risk.md) — 4 layers: peak-drawdown lockout, daily halt, daily halve, weekly resize; different thresholds for paper vs live
- [Wheel strategy](arch_wheel.md) — Cash-secured puts (accumulation) + covered calls (income); currently disabled in live loop; MSTR-focused with IV gates

---

## Integration Points & Gaps

- [Scanner → Live loop disconnect](gap_scanner_integration.md) — Nightly scanner generates watchlist for S&P 500/NDX 100, but live trader ignores it (hardcoded TICKERS). Fix: wire scanner output into regime_strategies decision.
- [Wheel strategy not wired](gap_wheel_live.md) — Wheel module fully coded but never called in live loop. Blocker: need position entry trigger and sizing under risk manager.
- [Short selling disabled](gap_short_selling.md) — Crash regime allows shorts, but order_executor only submits BUYs. Not blocking (can flatten instead); needs design if reversed.

---

## Known Issues & Workarounds

- [Duplicate cycle_signal alerts](issue_cycle_alerts.md) — FIXED 2026-06-09: added fingerprint-based dedup in alerts.send_cycle_alert(). Only sends if signal (score/failed/days_since_low/timing_prob) has changed.
- [HMM trained once at startup](issue_hmm_staleness.md) — Model not retrained mid-session; long-running instances may see stale regimes. Workaround: restart daily. Fix: implement online refit schedule.
- [Live mode equity trading disabled](issue_live_equity.md) — LIVE_ACCOUNT_MODE=True forces BTC-only (equities blocked). Reason: strategy untested on live account. Fix: enable with feature flags after validation.

---

## Data Sources & Limitations

- [Alpaca options API limitations](data_alpaca_options.md) — Supports: chain fetch, IV, delta, bid-ask, Greeks. Missing: open interest field on OptionContract. Workaround: none yet.
- [IV Rank calculation](data_iv_rank.md) — Not true market IV; uses realized vol + VIX + term structure composite. Available via alpaca_client.get_iv_rank(). Scanner uses same approach (vol_rank).
- [Earnings data via yfinance](data_earnings.md) — wheel_scanner/earnings.py uses yfinance.Ticker.calendar. Requires yfinance>=0.2.40 (added to requirements.txt 2026-06-09).

---

## Outstanding Tasks & Blockers

- [Wheel strategy integration spec](task_wheel_integration.md) — Spec needed: position entry conditions, sizing under risk manager, exit rules. Blocker: unclear when scanner candidates should trigger trades.
- [Portfolio review logic design](task_portfolio_review.md) — Decision pending: quant rules-based (hard gates) vs agent-powered (Claude reviews nightly state). Each has tradeoffs; user needs to pick.
- [Trade review agent architecture](task_trade_agent.md) — Spec needed: inputs (scanner result + market context), outputs (proceed / reduce_size / skip), data sources (news API).
- [tastytrade options-data integration](task_tastytrade_integration.md) — OAuth2 read-only provider for wheel_scanner (open interest via DXLink Summary event); COMPLETE, flag flipped to tastytrade. Env vars TT_SECRET/TT_REFRESH.
- [Wheel execution (Phase 2)](task_wheel_execution.md) — WheelStrategy brain already built; Phase 2a = CSP entry+mgmt, hybrid state, scanner-driven+caps, NAV sizing. Step 0 PASSED: Alpaca paper CAN sell-to-open puts via API. Build pending sign-off.

---

## Testing & Validation

- [Test suite](testing_suite.md) — 900+ pytest tests across 23 files; all mocked (no real API calls). Coverage: HMM, regimes, BTC cycle, risk manager, alerts, scanner, backtester. Run: `pytest tests/ -v`.
- [Backtester & walk-forward](testing_backtest.md) — Fully implemented but separate from live trading. Use for strategy validation before going live. Entry: `python -m core.backtester` or `python scripts/run_walk_forward.py`.

---

## Configuration & Deployment

- [Settings & tuning](config_settings.md) — All parameters in config/settings.py; includes HMM tuning, regime allocations, risk limits (paper vs live), scanner filters. Edit here, not in code.
- [Paper vs live mode](config_paper_live.md) — Paper: IEX data, low min volume (20k), equity trades enabled, looser risk limits. Live: SIP data, high min volume (1M), BTC-only, tight risk limits. Toggle: LIVE_ACCOUNT_MODE.
- [TLS-inspection proxy](env_tls_proxy.md) — this host breaks ALL Python HTTPS (Alpaca/alerts/tastytrade) unless enable_os_trust_store() runs first; main.py/connection_test/run_scanner still lack it.
- [Alert system cooldown](config_alerts.md) — Default 300s cooldown per event:symbol. Overridable via ALERT_COOLDOWN_OVERRIDES in settings. Cycle_signal uses fingerprint dedup on top of cooldown.

---

## References & Documentation

- [Mirror & Feedback systems](ref_kinri.md) — From earlier Kinri project; not part of regime_trader. Reference only for workflow patterns.
- [Weekly meeting notes](ref_meetings.md) — TBD: will track strategy discussions, tuning decisions, live trading outcomes.

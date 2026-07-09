---
name: task_wheel_execution
description: Phase 2 wheel execution — spec + Step-0 feasibility (Alpaca paper CAN sell-to-open puts via API); build pending sign-off
metadata:
  type: project
---

Phase 2 = wiring the existing (tested) `WheelStrategy` brain into live paper trading. Key realization: the **decision logic already exists** — `core/wheel_strategy.py` `WheelStrategy.get_next_action()` returns SELL_PUT/SELL_CALL/CLOSE/WAIT with entry finders, early-close, IV gate, and the CASH→PUT_SOLD→ASSIGNED→CALL_SOLD state machine. `order_executor._submit_wheel_order` and `Signal.wheel_action` also exist. So Phase 2 is **execution + state plumbing**, not strategy.

**Spec decisions (interview):**
- **Split — Phase 2a first** = CSP entry + management only (CASH→sell put→hold/early-close/expire→CASH, or assigned→ASSIGNED *handoff*). Covered calls + assignment-acting = 2b.
- **State = hybrid** (Ryan confirmed): persist `WheelPosition` to a server-side file (economics: premium collected, cost basis, entry regime) AND reconcile against live broker positions/orders each bar (truth: phase, shares). Broker wins on conflict; assignment/expiry detected as position deltas. Neither pure approach works: file-only drifts from reality, broker-only loses premium/P&L history.
- **Underlyings = scanner-driven, capital-capped** (limited funds → hard limits on concurrent positions + collateral).
- **Sizing = NAV-scaled contracts** under caps (`MAX_WHEEL_POSITIONS`, `WHEEL_MAX_COLLATERAL_PCT`, `WHEEL_TOTAL_DEPLOYED_PCT`).
- **Execution = limit at mid, priced off ALPACA's quote** (venue), not tastytrade (data) — orders fill on Alpaca. Replace `_submit_wheel_order`'s market order with limit + qty.
- Planned new files: `core/wheel_executor.py`, `core/wheel_position_store.py`; execution behind `WHEEL_EXECUTION_ENABLED` flag (default False) in `wheel_main`.

**STEP 0 FEASIBILITY — PASSED (2026-07-07).** Alpaca **paper** account probe (`scratchpad/alpaca_options_probe.py`, non-filling limit orders): status ACTIVE, `options_buying_power ≈ $90,482`. **Both buy-to-open AND sell-to-open puts were ACCEPTED via the API and cancelled cleanly** — even though Ryan reports the Alpaca **web UI blocks manual options access**, the API route works. `submit_order` routes OCC symbols as-is; Alpaca infers `sell_to_open` from side=sell + no position. So the venue (Alpaca paper options) is validated for the wheel's short-put capability. Build note: add explicit `position_intent` to `submit_order` for buy-to-close safety.

**PHASE 2a BUILT 2026-07 (behind flag, 1036 tests pass).** New: `core/wheel_position_store.py` (hybrid state, reconcile-driven — broker supplies open-leg premium via short-put avg_entry_price; 13 tests) + `core/wheel_executor.py` (CASH→size+sell CSP / PUT_SOLD→P&L+close / ASSIGNED→handoff; 10 tests). Wired into `wheel_main.run()` behind `settings.WHEEL_EXECUTION_ENABLED` (default **False**) with `WHEEL_EXEC_INTERVAL_SECS`. Sizing caps: `MAX_WHEEL_POSITIONS=2`, `WHEEL_MAX_COLLATERAL_PCT=0.35`, `WHEEL_TOTAL_DEPLOYED_PCT=0.70`. Added `position_intent` to `AlpacaClient.submit_order`. **Design note:** executor selects+prices off ALPACA's chain (`get_option_chain`/`get_iv_rank`) = the execution venue; scanner stays on tastytrade for discovery. Executor calls `client.submit_order` directly (not `order_executor._submit_wheel_order`, now legacy/unused — candidate for cleanup).

**NOT YET ENABLED.** Flipping `WHEEL_EXECUTION_ENABLED=True` = live paper-trading activation gate → needs Ryan sign-off + a guarded paper dry-run first. See [[task_tastytrade_integration]], [[gap_wheel_live]].

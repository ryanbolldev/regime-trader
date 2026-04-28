"""
main.py
--------
Orchestrator: startup sequence, main trading loop, and graceful shutdown.

Startup sequence:
  1. Load credentials (config/credentials.py) — fail fast if any are missing.
  2. Validate settings (config/settings.py) — check ranges, required fields.
  3. Check for lockfile; if present, log and exit with a clear human message.
  4. Connect to broker (broker/alpaca_client.py) and verify account status.
  5. Reconcile open positions with broker (position_tracker.py).
  6. Fetch historical bars required to warm up the HMM and feature windows.
  7. Fit the initial HMM model (core/hmm_engine.py).
  8. Subscribe to real-time bar feed (core/market_data.py).
  9. Start the dashboard in a separate thread if ENABLE_DASHBOARD is True.

Main loop (default: 5-minute bars):
  On each new bar for each ticker:
    a. Update feature engineering (core/feature_engineering.py).
    b. Run HMM forward step; check confirmation and flicker filter
       (core/hmm_engine.py).
    c. If regime is confirmed and changed: log, alert, request new signals
       (core/regime_strategies.py).
    d. Pass signals through risk gate (core/risk_manager.py).
    e. Submit approved orders (core/order_executor.py).
    f. Update position tracker on fills.
    g. Evaluate circuit breakers; halt / resize as required.

Error handling:
  - API outage: pause bar processing, retry with back-off, alert after
    MAX_OUTAGE_RETRIES, gracefully exit if broker unreachable beyond
    MAX_OUTAGE_SECONDS.
  - Data drop (stale feed): hold positions, suppress new entries, alert.
  - Unhandled exception: log full traceback, send critical alert, exit cleanly
    so a process supervisor can restart.

Graceful shutdown (SIGINT / SIGTERM):
  - Cancel all open (unfilled) orders.
  - Log final portfolio snapshot.
  - Flush position state to disk.
  - Disconnect from data feed.
"""

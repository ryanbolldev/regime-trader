"""
dashboard/app.py
-----------------
Streamlit dashboard for real-time monitoring of the regime trader.

Panels:
  - Current Regime
      Displays the confirmed HMM regime label, the raw (unconfirmed) state,
      state-probability distribution as a bar chart, and flicker count over the
      last 20 bars.

  - Portfolio Value
      Real-time NAV, daily P&L (absolute and %), and an equity curve chart
      overlaid with regime colour bands.

  - Signal Feed
      Scrolling log of the last N trade signals: ticker, direction, regime at
      signal time, HMM confidence, size, status (pending / filled / rejected).

  - Risk Status
      Current circuit-breaker state (green / amber / red per level), drawdown
      from peak, and the daily / weekly drawdown meters.

  - Circuit Breaker Panel
      Prominent red banner if the peak-drawdown lockfile is active, with the
      lockfile path displayed so the operator knows what to delete to resume.

  - Position Table
      All open positions: ticker, size, entry price, current price, P&L.

Data source:
  - Reads from position_tracker.py snapshots and a shared state object updated
    by the main loop.  The dashboard never calls the broker API directly.

Run with:
  streamlit run dashboard/app.py
"""

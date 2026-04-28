"""
tests/test_position_tracker.py
--------------------------------
Unit tests for core/position_tracker.py.

Test cases to implement:
  - on_fill(buy_fill) opens a new position record
  - on_fill(sell_fill) reduces or closes the matching position
  - get_nav() returns cash + sum of market values of open positions
  - get_daily_pnl() resets at session open
  - get_drawdown_from_peak() is 0 when NAV equals the high-water mark
  - get_drawdown_from_peak() returns a negative fraction when below peak
  - get_correlation_matrix() returns a square DataFrame of shape (n, n)
  - Position state is persisted to disk and re-hydrated correctly on restart
  - Re-hydration reconciles against broker positions (broker wins on conflict)
  - get_portfolio_snapshot() contains all required fields for the dashboard
"""

"""
tests/test_risk_manager.py
---------------------------
Unit tests for core/risk_manager.py.

Test cases to implement:
  - approve() returns APPROVED for a valid signal within all limits
  - approve() returns REJECTED when daily P&L <= -3% (close-all breaker)
  - approve() returns RESIZED when daily P&L is between -2% and -3%
  - approve() returns REJECTED when lockfile is present
  - compute_position_size() caps risk at 1% of NAV per trade
  - compute_position_size() returns 0 when close-all breaker is active
  - Weekly drawdown at -5% triggers size halving on next approve() call
  - Peak drawdown at -10% creates lockfile and returns LOCKED
  - Lockfile is created at the expected path with readable content
  - Correlation check rejects a signal that would exceed MAX_CORR_BUDGET
  - Circuit breakers reset correctly at session open (daily) or week open
  - is_locked() returns True iff lockfile exists on disk
"""

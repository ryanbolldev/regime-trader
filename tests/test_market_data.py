"""
tests/test_market_data.py
--------------------------
Unit tests for core/market_data.py.

Test cases to implement:
  - get_historical_bars() returns a DataFrame with OHLCV columns and a
    DatetimeIndex sorted ascending
  - get_historical_bars() raises DataUnavailableError for an invalid ticker
  - get_latest_bars() returns exactly n_bars rows from the in-memory buffer
  - Buffer is updated correctly when a new bar arrives via subscribe callback
  - Forward-fill behaviour: gaps shorter than threshold are filled; longer gaps
    raise DataDropAlert
  - is_stale() returns True when the latest bar is older than the threshold
  - is_stale() returns False for a freshly updated ticker
  - subscribe() invokes the callback exactly once per new bar
  - unsubscribe() stops callback invocations for the removed tickers
  - All tests mock the broker HTTP layer; no live network calls
"""

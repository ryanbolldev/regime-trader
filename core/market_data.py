"""
core/market_data.py
--------------------
Data acquisition layer for both real-time and historical price feeds.

Responsibilities:
  - Fetch historical OHLCV bars for backtesting and model fitting via the
    broker's historical data API
  - Subscribe to real-time bar updates during live trading (5-minute default)
  - Maintain an in-memory rolling buffer of recent bars per ticker so that
    feature_engineering.py can be called without hitting the API on every bar
  - Handle data gaps (e.g., holidays, early closes): forward-fill or drop
    depending on the gap duration threshold in settings.py
  - Detect and log stale data: if the latest bar timestamp is older than
    STALENESS_THRESHOLD seconds, raise a DataDropAlert and notify alerts.py
  - Provide a unified interface so backtesting and live trading use the same
    feature_engineering and HMM code paths

Public interface (to be implemented):
  get_historical_bars(ticker, start, end, bar_size) -> pd.DataFrame
  get_latest_bars(ticker, n_bars) -> pd.DataFrame
  subscribe(tickers, bar_size, callback) -> None
  unsubscribe(tickers) -> None
  is_stale(ticker) -> bool
"""

"""
broker/alpaca_client.py
------------------------
Alpaca REST API wrapper.

Implements the abstract BrokerClient interface consumed by order_executor.py
and market_data.py, keeping all Alpaca-specific details isolated here.

Capabilities:
  - Account information: NAV, buying power, positions (live reconciliation)
  - Order management: submit, modify, cancel, list open orders
  - Historical bar data: daily and intraday OHLCV via the Alpaca Data API v2
  - Real-time streaming: WebSocket bar subscription per ticker
  - Paper trading / live trading toggle via ALPACA_BASE_URL in .env

Security rules enforced in this module:
  - Credentials are read exclusively from credentials.py (which loads .env);
    they are never passed as arguments, stored as class attributes, or logged.
  - The API key is redacted in all log messages (replaced with "***").
  - HTTPS is always enforced; plain HTTP base URLs raise a ConfigurationError.

Error handling:
  - 4xx responses: raise domain-specific exceptions (AuthError, RateLimitError,
    InsufficientFundsError) mapped from Alpaca error codes.
  - 5xx / network errors: raise BrokerUnavailableError; caller retries via
    order_executor.py's back-off logic.

Public interface (to be implemented):
  get_account() -> AccountInfo
  submit_order(order_request) -> OrderResult
  cancel_order(order_id) -> bool
  get_positions() -> list[BrokerPosition]
  get_historical_bars(ticker, start, end, timeframe) -> pd.DataFrame
  stream_bars(tickers, bar_size, callback) -> None
"""

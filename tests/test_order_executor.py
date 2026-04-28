"""
tests/test_order_executor.py
-----------------------------
Unit tests for core/order_executor.py.

Test cases to implement:
  - submit() calls the broker client with the correct order parameters
  - submit() returns an OrderResult with a non-null order_id on success
  - submit() retries up to MAX_RETRIES on BrokerUnavailableError
  - submit() raises AlertableError and does not retry on 4xx (client) errors
  - cancel() calls the broker cancel endpoint and returns True on success
  - cancel() returns False when the order is already filled
  - cancel_all() iterates all open orders and cancels each
  - modify() calls the broker modify endpoint with updated price/qty
  - get_open_orders() returns only unfilled, un-cancelled orders
  - Rate limiter: a burst of N+1 calls within one second raises RateLimitError
    or pauses appropriately (configurable behaviour)
  - All tests use a mock BrokerClient; no real HTTP calls
"""

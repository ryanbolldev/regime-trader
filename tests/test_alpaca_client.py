"""
tests/test_alpaca_client.py
----------------------------
Unit tests for broker/alpaca_client.py.

All Alpaca API calls are mocked — no real HTTP requests are made.
"""

from __future__ import annotations

import logging
import uuid
from unittest.mock import MagicMock, patch

import pytest
import requests

from broker.alpaca_client import (
    AccountInfo,
    AlpacaClient,
    AuthError,
    BrokerPosition,
    BrokerUnavailableError,
    InsufficientFundsError,
    OptionContract,
    OrderResult,
    RateLimitError,
    _parse_occ_symbol,
)
from config.credentials import Credentials


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

FAKE_CREDS = Credentials(
    api_key    = "TESTKEY123",
    api_secret = "TESTSECRET456",
    base_url   = "https://paper-api.alpaca.markets/v2",
)


class _APIError(Exception):
    """Simulates an alpaca-py APIError with a status_code attribute."""
    def __init__(self, status_code: int, msg: str = "error"):
        super().__init__(msg)
        self.status_code = status_code


def _mock_response(status_code: int, json_data: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = status_code < 400
    resp.json.return_value = json_data or {}
    resp.text = str(json_data or {})
    return resp


def _mock_position(
    symbol="AAPL", qty=10.0, market_value=1500.0,
    unrealized_pl=50.0, avg_entry_price=145.0, side="long",
) -> MagicMock:
    p = MagicMock()
    p.symbol = symbol
    p.qty = qty
    p.market_value = market_value
    p.unrealized_pl = unrealized_pl
    p.avg_entry_price = avg_entry_price
    p.side = side   # plain string so _enum_val falls through to str()
    return p


def _mock_order(
    symbol="AAPL", qty=10.0, side="buy", order_type="market",
    status="new", filled_qty=0.0, filled_avg_price=None,
) -> MagicMock:
    o = MagicMock()
    o.id = str(uuid.uuid4())
    o.client_order_id = str(uuid.uuid4())
    o.symbol = symbol
    o.qty = qty
    o.side = side
    o.order_type = order_type
    o.status = status
    o.filled_qty = filled_qty
    o.filled_avg_price = filled_avg_price
    return o


def _mock_account(
    account_id="acc-1", status="ACTIVE", buying_power=10_000.0,
    portfolio_value=100_000.0, options_buying_power=5_000.0, cash=10_000.0,
) -> MagicMock:
    a = MagicMock()
    a.id = account_id
    a.status = status
    a.buying_power = buying_power
    a.portfolio_value = portfolio_value
    a.options_buying_power = options_buying_power
    a.cash = cash
    return a


def _mock_snapshot(delta=0.4, iv=0.25, bid=1.5, ask=1.6) -> MagicMock:
    snap = MagicMock()
    snap.implied_volatility = iv
    snap.greeks = MagicMock()
    snap.greeks.delta = delta
    snap.latest_quote = MagicMock()
    snap.latest_quote.bid_price = bid
    snap.latest_quote.ask_price = ask
    return snap


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def client() -> AlpacaClient:
    """AlpacaClient with all external I/O mocked."""
    with patch("broker.alpaca_client.load_credentials", return_value=FAKE_CREDS), \
         patch("broker.alpaca_client.TradingClient"), \
         patch("broker.alpaca_client.OptionHistoricalDataClient"):
        c = AlpacaClient()
    # Replace the live session with a mock so tests control HTTP responses.
    c._session = MagicMock()
    return c


# ---------------------------------------------------------------------------
# TestOCCParser
# ---------------------------------------------------------------------------

class TestOCCParser:

    def test_call_option_parsed_correctly(self):
        result = _parse_occ_symbol("AAPL250117C00150000")
        assert result is not None
        underlying, expiration, option_type, strike = result
        assert underlying   == "AAPL"
        assert expiration   == "2025-01-17"
        assert option_type  == "call"
        assert strike       == pytest.approx(150.0)

    def test_put_option_parsed_correctly(self):
        underlying, expiration, option_type, strike = _parse_occ_symbol("SPY241220P00450000")
        assert underlying   == "SPY"
        assert expiration   == "2024-12-20"
        assert option_type  == "put"
        assert strike       == pytest.approx(450.0)

    def test_strike_divided_by_1000(self):
        _, _, _, strike = _parse_occ_symbol("QQQ250321C00500000")
        assert strike == pytest.approx(500.0)

    def test_fractional_strike(self):
        _, _, _, strike = _parse_occ_symbol("TSLA250117C00247500")
        assert strike == pytest.approx(247.5)

    def test_expiration_iso_date_format(self):
        _, expiration, _, _ = _parse_occ_symbol("SPY251231C00600000")
        assert expiration == "2025-12-31"

    def test_invalid_format_returns_none(self):
        assert _parse_occ_symbol("not_an_occ_symbol") is None

    def test_empty_string_returns_none(self):
        assert _parse_occ_symbol("") is None


# ---------------------------------------------------------------------------
# TestGetAccount
# ---------------------------------------------------------------------------

class TestGetAccount:

    def test_returns_account_info(self, client):
        client._trading.get_account.return_value = _mock_account()
        result = client.get_account()
        assert isinstance(result, AccountInfo)

    def test_account_info_fields_match_mock(self, client):
        mock_acct = _mock_account(
            account_id="id-999", status="ACTIVE",
            buying_power=50_000.0, portfolio_value=200_000.0,
            options_buying_power=25_000.0, cash=50_000.0,
        )
        client._trading.get_account.return_value = mock_acct
        result = client.get_account()
        assert result.account_id      == "id-999"
        assert result.status          == "ACTIVE"
        assert result.buying_power    == pytest.approx(50_000.0)
        assert result.portfolio_value == pytest.approx(200_000.0)
        assert result.cash            == pytest.approx(50_000.0)

    def test_is_paper_reflects_client_flag(self, client):
        client._trading.get_account.return_value = _mock_account()
        result = client.get_account()
        assert result.is_paper == client._is_paper

    def test_auth_error_on_401(self, client):
        client._trading.get_account.side_effect = _APIError(401)
        with pytest.raises(AuthError):
            client.get_account()

    def test_broker_unavailable_on_503(self, client):
        client._trading.get_account.side_effect = _APIError(503)
        with patch("broker.alpaca_client.time.sleep"):
            with pytest.raises(BrokerUnavailableError):
                client.get_account()


# ---------------------------------------------------------------------------
# TestGetPositions
# ---------------------------------------------------------------------------

class TestGetPositions:

    def test_returns_list_of_broker_positions(self, client):
        client._trading.get_all_positions.return_value = [_mock_position()]
        result = client.get_positions()
        assert isinstance(result, list)
        assert isinstance(result[0], BrokerPosition)

    def test_empty_positions_returns_empty_list(self, client):
        client._trading.get_all_positions.return_value = []
        assert client.get_positions() == []

    def test_position_fields_match_mock(self, client):
        mock_pos = _mock_position(
            symbol="SPY", qty=5.0, market_value=2200.0,
            unrealized_pl=-30.0, avg_entry_price=446.0, side="long",
        )
        client._trading.get_all_positions.return_value = [mock_pos]
        result = client.get_positions()[0]
        assert result.symbol          == "SPY"
        assert result.qty             == pytest.approx(5.0)
        assert result.market_value    == pytest.approx(2200.0)
        assert result.unrealized_pl   == pytest.approx(-30.0)
        assert result.avg_entry_price == pytest.approx(446.0)
        assert result.side            == "long"

    def test_multiple_positions_all_returned(self, client):
        client._trading.get_all_positions.return_value = [
            _mock_position("AAPL"), _mock_position("SPY"), _mock_position("QQQ"),
        ]
        assert len(client.get_positions()) == 3


# ---------------------------------------------------------------------------
# TestGetOrders
# ---------------------------------------------------------------------------

class TestGetOrders:

    def test_returns_list_of_order_results(self, client):
        client._trading.get_orders.return_value = [_mock_order()]
        result = client.get_orders()
        assert isinstance(result, list)
        assert isinstance(result[0], OrderResult)

    def test_empty_orders_returns_empty_list(self, client):
        client._trading.get_orders.return_value = []
        assert client.get_orders() == []

    def test_order_fields_match_mock(self, client):
        o = _mock_order(symbol="AAPL", qty=7.0, side="sell",
                        order_type="limit", status="open",
                        filled_qty=3.0, filled_avg_price=152.5)
        client._trading.get_orders.return_value = [o]
        result = client.get_orders()[0]
        assert result.symbol     == "AAPL"
        assert result.qty        == pytest.approx(7.0)
        assert result.side       == "sell"
        assert result.order_type == "limit"
        assert result.status     == "open"
        assert result.filled_qty == pytest.approx(3.0)
        assert result.filled_avg_price == pytest.approx(152.5)

    def test_unfilled_order_has_none_filled_price(self, client):
        client._trading.get_orders.return_value = [_mock_order(filled_avg_price=None)]
        result = client.get_orders()[0]
        assert result.filled_avg_price is None

    def test_auth_error_on_403(self, client):
        client._trading.get_orders.side_effect = _APIError(403)
        with pytest.raises(AuthError):
            client.get_orders()


# ---------------------------------------------------------------------------
# TestSubmitOrder
# ---------------------------------------------------------------------------

class TestSubmitOrder:

    _ORDER_JSON = {
        "id": "order-abc",
        "client_order_id": "coi-xyz",
        "symbol": "AAPL",
        "qty": "10",
        "side": "buy",
        "type": "market",
        "status": "accepted",
        "filled_qty": "0",
        "filled_avg_price": None,
    }

    def test_returns_order_result(self, client):
        client._session.post.return_value = _mock_response(200, self._ORDER_JSON)
        result = client.submit_order("AAPL", 10.0, "buy", "market")
        assert isinstance(result, OrderResult)

    def test_request_id_is_uuid(self, client):
        client._session.post.return_value = _mock_response(200, self._ORDER_JSON)
        result = client.submit_order("AAPL", 10.0, "buy", "market")
        uuid.UUID(result.request_id)   # raises ValueError if not a valid UUID

    def test_request_id_sent_in_header(self, client):
        client._session.post.return_value = _mock_response(200, self._ORDER_JSON)
        result = client.submit_order("AAPL", 10.0, "buy", "market")
        _, kwargs = client._session.post.call_args
        assert kwargs["headers"]["X-Request-Id"] == result.request_id

    def test_market_order_body_has_no_limit_price(self, client):
        client._session.post.return_value = _mock_response(200, self._ORDER_JSON)
        client.submit_order("AAPL", 10.0, "buy", "market")
        _, kwargs = client._session.post.call_args
        assert "limit_price" not in kwargs["json"]

    def test_limit_price_included_when_provided(self, client):
        client._session.post.return_value = _mock_response(200, self._ORDER_JSON)
        client.submit_order("AAPL", 10.0, "buy", "limit", limit_price=150.0)
        _, kwargs = client._session.post.call_args
        assert kwargs["json"]["limit_price"] == "150.0"

    def test_correct_url_used(self, client):
        client._session.post.return_value = _mock_response(200, self._ORDER_JSON)
        client.submit_order("AAPL", 10.0, "buy", "market")
        args, _ = client._session.post.call_args
        assert args[0] == "https://paper-api.alpaca.markets/v2/orders"

    def test_auth_error_on_401_response(self, client):
        client._session.post.return_value = _mock_response(401, {"message": "Unauthorized"})
        with pytest.raises(AuthError):
            client.submit_order("AAPL", 10.0, "buy", "market")

    def test_insufficient_funds_on_422(self, client):
        client._session.post.return_value = _mock_response(
            422, {"message": "insufficient buying power"}
        )
        with pytest.raises(InsufficientFundsError):
            client.submit_order("AAPL", 10.0, "buy", "market")

    def test_broker_unavailable_on_503(self, client):
        client._session.post.return_value = _mock_response(503, {})
        with patch("broker.alpaca_client.time.sleep"):
            with pytest.raises(BrokerUnavailableError):
                client.submit_order("AAPL", 10.0, "buy", "market")

    def test_network_error_raises_broker_unavailable(self, client):
        client._session.post.side_effect = requests.RequestException("timeout")
        with patch("broker.alpaca_client.time.sleep"):
            with pytest.raises(BrokerUnavailableError):
                client.submit_order("AAPL", 10.0, "buy", "market")


# ---------------------------------------------------------------------------
# TestCancelOrder
# ---------------------------------------------------------------------------

class TestCancelOrder:

    def test_returns_true_on_success(self, client):
        client._trading.cancel_order_by_id.return_value = None
        assert client.cancel_order("order-1") is True

    def test_returns_false_when_order_not_found(self, client):
        client._trading.cancel_order_by_id.side_effect = _APIError(404, "not found")
        assert client.cancel_order("order-999") is False

    def test_auth_error_on_403(self, client):
        client._trading.cancel_order_by_id.side_effect = _APIError(403)
        with pytest.raises(AuthError):
            client.cancel_order("order-1")

    def test_broker_unavailable_on_500(self, client):
        client._trading.cancel_order_by_id.side_effect = _APIError(500)
        with patch("broker.alpaca_client.time.sleep"):
            with pytest.raises(BrokerUnavailableError):
                client.cancel_order("order-1")


# ---------------------------------------------------------------------------
# TestGetOptionChain
# ---------------------------------------------------------------------------

class TestGetOptionChain:

    def test_returns_list_of_option_contracts(self, client):
        client._options.get_option_chain.return_value = {
            "AAPL250117C00150000": _mock_snapshot(),
        }
        result = client.get_option_chain("AAPL")
        assert isinstance(result, list)
        assert isinstance(result[0], OptionContract)

    def test_option_contract_fields_populated(self, client):
        client._options.get_option_chain.return_value = {
            "SPY241220P00450000": _mock_snapshot(delta=-0.3, iv=0.20, bid=2.1, ask=2.3),
        }
        contract = client.get_option_chain("SPY")[0]
        assert contract.symbol            == "SPY241220P00450000"
        assert contract.underlying        == "SPY"
        assert contract.expiration        == "2024-12-20"
        assert contract.strike            == pytest.approx(450.0)
        assert contract.option_type       == "put"
        assert contract.delta             == pytest.approx(-0.3)
        assert contract.implied_volatility == pytest.approx(0.20)
        assert contract.bid               == pytest.approx(2.1)
        assert contract.ask               == pytest.approx(2.3)

    def test_invalid_occ_symbol_is_skipped(self, client):
        client._options.get_option_chain.return_value = {
            "INVALID_SYMBOL":      _mock_snapshot(),
            "AAPL250117C00150000": _mock_snapshot(),
        }
        result = client.get_option_chain("AAPL")
        assert len(result) == 1
        assert result[0].symbol == "AAPL250117C00150000"

    def test_empty_chain_returns_empty_list(self, client):
        client._options.get_option_chain.return_value = {}
        assert client.get_option_chain("AAPL") == []

    def test_multiple_contracts_all_returned(self, client):
        client._options.get_option_chain.return_value = {
            "AAPL250117C00150000": _mock_snapshot(),
            "AAPL250117P00145000": _mock_snapshot(),
            "AAPL250117C00155000": _mock_snapshot(),
        }
        result = client.get_option_chain("AAPL")
        assert len(result) == 3

    def test_auth_error_on_401(self, client):
        client._options.get_option_chain.side_effect = _APIError(401)
        with pytest.raises(AuthError):
            client.get_option_chain("AAPL")


# ---------------------------------------------------------------------------
# TestIsMarketOpen
# ---------------------------------------------------------------------------

class TestIsMarketOpen:

    def test_returns_true_when_market_open(self, client):
        mock_clock = MagicMock()
        mock_clock.is_open = True
        client._trading.get_clock.return_value = mock_clock
        assert client.is_market_open() is True

    def test_returns_false_when_market_closed(self, client):
        mock_clock = MagicMock()
        mock_clock.is_open = False
        client._trading.get_clock.return_value = mock_clock
        assert client.is_market_open() is False

    def test_auth_error_on_401(self, client):
        client._trading.get_clock.side_effect = _APIError(401)
        with pytest.raises(AuthError):
            client.is_market_open()


# ---------------------------------------------------------------------------
# TestRetryLogic
# ---------------------------------------------------------------------------

class TestRetryLogic:

    def test_raises_broker_unavailable_after_three_failures(self, client):
        client._trading.get_clock.side_effect = _APIError(503)
        with patch("broker.alpaca_client.time.sleep"):
            with pytest.raises(BrokerUnavailableError):
                client.is_market_open()
        assert client._trading.get_clock.call_count == 3

    def test_sleep_called_with_correct_delays(self, client):
        client._trading.get_clock.side_effect = _APIError(503)
        with patch("broker.alpaca_client.time.sleep") as mock_sleep:
            with pytest.raises(BrokerUnavailableError):
                client.is_market_open()
        assert mock_sleep.call_count == 2
        calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert calls == [1.0, 2.0]

    def test_succeeds_on_second_attempt(self, client):
        open_clock = MagicMock()
        open_clock.is_open = True
        client._trading.get_clock.side_effect = [_APIError(503), open_clock]
        with patch("broker.alpaca_client.time.sleep"):
            result = client.is_market_open()
        assert result is True
        assert client._trading.get_clock.call_count == 2

    def test_non_broker_error_not_retried(self, client):
        client._trading.get_clock.side_effect = _APIError(401)
        with patch("broker.alpaca_client.time.sleep") as mock_sleep:
            with pytest.raises(AuthError):
                client.is_market_open()
        mock_sleep.assert_not_called()
        assert client._trading.get_clock.call_count == 1

    def test_rate_limit_not_retried(self, client):
        client._trading.get_clock.side_effect = _APIError(429)
        with patch("broker.alpaca_client.time.sleep") as mock_sleep:
            with pytest.raises(RateLimitError):
                client.is_market_open()
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# TestCredentialRedaction
# ---------------------------------------------------------------------------

class TestCredentialRedaction:

    def test_api_key_not_in_log_output(self, caplog):
        with caplog.at_level(logging.INFO, logger="broker.alpaca_client"):
            with patch("broker.alpaca_client.load_credentials", return_value=FAKE_CREDS), \
                 patch("broker.alpaca_client.TradingClient"), \
                 patch("broker.alpaca_client.OptionHistoricalDataClient"):
                AlpacaClient()
        assert FAKE_CREDS.api_key not in caplog.text

    def test_api_secret_not_in_log_output(self, caplog):
        with caplog.at_level(logging.INFO, logger="broker.alpaca_client"):
            with patch("broker.alpaca_client.load_credentials", return_value=FAKE_CREDS), \
                 patch("broker.alpaca_client.TradingClient"), \
                 patch("broker.alpaca_client.OptionHistoricalDataClient"):
                AlpacaClient()
        assert FAKE_CREDS.api_secret not in caplog.text

    def test_submit_order_log_excludes_credentials(self, client, caplog):
        client._session.post.return_value = _mock_response(200, {
            "id": "ord-1", "client_order_id": "coi-1",
            "symbol": "AAPL", "qty": "5", "side": "buy",
            "type": "market", "status": "accepted",
            "filled_qty": "0", "filled_avg_price": None,
        })
        with caplog.at_level(logging.INFO, logger="broker.alpaca_client"):
            client.submit_order("AAPL", 5.0, "buy", "market")
        assert FAKE_CREDS.api_key    not in caplog.text
        assert FAKE_CREDS.api_secret not in caplog.text


# ---------------------------------------------------------------------------
# TestErrorMapping
# ---------------------------------------------------------------------------

class TestErrorMapping:

    def test_401_maps_to_auth_error(self, client):
        client._trading.get_account.side_effect = _APIError(401)
        with pytest.raises(AuthError):
            client.get_account()

    def test_403_maps_to_auth_error(self, client):
        client._trading.get_account.side_effect = _APIError(403)
        with pytest.raises(AuthError):
            client.get_account()

    def test_429_maps_to_rate_limit_error(self, client):
        client._trading.get_account.side_effect = _APIError(429)
        with pytest.raises(RateLimitError):
            client.get_account()

    def test_500_maps_to_broker_unavailable(self, client):
        client._trading.get_account.side_effect = _APIError(500)
        with patch("broker.alpaca_client.time.sleep"):
            with pytest.raises(BrokerUnavailableError):
                client.get_account()

    def test_503_maps_to_broker_unavailable(self, client):
        client._trading.get_account.side_effect = _APIError(503)
        with patch("broker.alpaca_client.time.sleep"):
            with pytest.raises(BrokerUnavailableError):
                client.get_account()

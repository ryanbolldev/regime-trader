"""
broker/alpaca_client.py
------------------------
Alpaca REST API wrapper.

Implements the abstract BrokerClient interface consumed by order_executor.py
and market_data.py, keeping all Alpaca-specific details isolated here.

Security rules:
  - Credentials are read exclusively from credentials.py (which loads .env).
  - API key / secret are never stored as named instance attributes or logged.
  - HTTPS is always enforced; plain HTTP base URLs raise ConfigurationError.

Error handling:
  - 401/403  → AuthError
  - 422 (funds) → InsufficientFundsError
  - 429  → RateLimitError
  - 5xx / network → BrokerUnavailableError (retried up to 3 times)
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass
from functools import wraps
from typing import Optional

import requests
from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest

from config.credentials import load_credentials

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AuthError(Exception):
    """401/403 from Alpaca."""

class RateLimitError(Exception):
    """429 — too many requests."""

class InsufficientFundsError(Exception):
    """Order rejected due to insufficient buying power."""

class BrokerUnavailableError(Exception):
    """5xx or network error; caller may retry."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AccountInfo:
    account_id:           str
    status:               str
    buying_power:         float
    portfolio_value:      float
    options_buying_power: float
    cash:                 float
    is_paper:             bool


@dataclass(frozen=True)
class BrokerPosition:
    symbol:          str
    qty:             float
    market_value:    float
    unrealized_pl:   float
    avg_entry_price: float
    side:            str


@dataclass(frozen=True)
class OrderResult:
    order_id:         str
    client_order_id:  str
    symbol:           str
    qty:              float
    side:             str
    order_type:       str
    status:           str
    filled_qty:       float
    filled_avg_price: Optional[float]
    request_id:       str   # UUID sent as X-Request-Id header


@dataclass(frozen=True)
class OptionContract:
    symbol:             str   # OCC format e.g. AAPL250117C00150000
    underlying:         str
    expiration:         str   # YYYY-MM-DD
    strike:             float
    option_type:        str   # "call" or "put"
    delta:              Optional[float]
    implied_volatility: Optional[float]
    bid:                Optional[float]
    ask:                Optional[float]


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

_RETRY_DELAYS = (1.0, 2.0)   # pause before attempt 2, then attempt 3


def _with_retry(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        for attempt, delay in enumerate((*_RETRY_DELAYS, None), start=1):
            try:
                return fn(*args, **kwargs)
            except BrokerUnavailableError:
                if delay is None:
                    raise
                logger.warning(
                    "Broker unavailable (attempt %d/3) — retrying in %.0fs",
                    attempt, delay,
                )
                time.sleep(delay)
    return wrapper


# ---------------------------------------------------------------------------
# OCC symbol parser
# ---------------------------------------------------------------------------

_OCC_RE = re.compile(r"^([A-Z0-9]+)(\d{6})([CP])(\d{8})$")


def _parse_occ_symbol(symbol: str) -> tuple[str, str, str, float] | None:
    """Parse an OCC option symbol into (underlying, expiration, option_type, strike).

    Returns None if the symbol does not match the OCC format.
    """
    m = _OCC_RE.match(symbol)
    if not m:
        return None
    underlying, yymmdd, cp, strike_raw = m.groups()
    year  = 2000 + int(yymmdd[:2])
    month = int(yymmdd[2:4])
    day   = int(yymmdd[4:])
    return (
        underlying,
        f"{year:04d}-{month:02d}-{day:02d}",
        "call" if cp == "C" else "put",
        int(strike_raw) / 1000.0,
    )


# ---------------------------------------------------------------------------
# Internal error helpers
# ---------------------------------------------------------------------------

def _handle_sdk_error(exc: Exception) -> None:
    """Translate an alpaca-py APIError into a domain exception."""
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        raise AuthError(str(exc)) from exc
    if status == 422:
        msg = str(exc).lower()
        if "insufficient" in msg or "buying power" in msg or "funds" in msg:
            raise InsufficientFundsError(str(exc)) from exc
    if status == 429:
        raise RateLimitError(str(exc)) from exc
    if status is not None and status >= 500:
        raise BrokerUnavailableError(str(exc)) from exc
    raise exc


def _handle_http_error(resp: requests.Response) -> None:
    """Raise domain exceptions for non-2xx HTTP responses."""
    code = resp.status_code
    if code in (401, 403):
        raise AuthError(f"HTTP {code}: {resp.text}")
    if code == 422:
        logger.warning("Alpaca 422 rejection body: %s", resp.text)
        body = resp.text.lower()
        if "insufficient" in body or "buying power" in body or "funds" in body:
            raise InsufficientFundsError(f"HTTP 422: {resp.text}")
    if code == 429:
        raise RateLimitError(f"HTTP 429: {resp.text}")
    if code >= 500:
        raise BrokerUnavailableError(f"HTTP {code}: {resp.text}")
    resp.raise_for_status()


def _enum_val(obj) -> str:
    return obj.value if hasattr(obj, "value") else str(obj)


def _order_to_result(order, request_id: str = "") -> OrderResult:
    return OrderResult(
        order_id         = str(order.id),
        client_order_id  = str(order.client_order_id),
        symbol           = str(order.symbol),
        qty              = float(order.qty or 0),
        side             = _enum_val(order.side),
        order_type       = _enum_val(order.order_type),
        status           = _enum_val(order.status),
        filled_qty       = float(order.filled_qty or 0),
        filled_avg_price = float(order.filled_avg_price) if order.filled_avg_price is not None else None,
        request_id       = request_id,
    )


# ---------------------------------------------------------------------------
# AlpacaClient
# ---------------------------------------------------------------------------

class AlpacaClient:
    """Alpaca broker client — paper or live depending on ALPACA_BASE_URL."""

    def __init__(self) -> None:
        creds = load_credentials()

        self._is_paper = creds.is_paper
        self._base_url = creds.base_url

        self._trading = TradingClient(
            api_key    = creds.api_key,
            secret_key = creds.api_secret,
            paper      = creds.is_paper,
        )
        self._options = OptionHistoricalDataClient(
            api_key    = creds.api_key,
            secret_key = creds.api_secret,
        )
        self._session = requests.Session()
        self._session.headers.update({
            "APCA-API-KEY-ID":     creds.api_key,
            "APCA-API-SECRET-KEY": creds.api_secret,
        })

        logger.info(
            "AlpacaClient ready (paper=%s, base=%s)",
            self._is_paper,
            self._base_url,
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    @_with_retry
    def get_account(self) -> AccountInfo:
        try:
            acct = self._trading.get_account()
        except Exception as exc:
            _handle_sdk_error(exc)
            raise  # unreachable; satisfies type checker

        return AccountInfo(
            account_id           = str(acct.id),
            status               = str(acct.status),
            buying_power         = float(acct.buying_power),
            portfolio_value      = float(acct.portfolio_value),
            options_buying_power = float(getattr(acct, "options_buying_power", 0) or 0),
            cash                 = float(acct.cash),
            is_paper             = self._is_paper,
        )

    @_with_retry
    def get_positions(self) -> list[BrokerPosition]:
        try:
            positions = self._trading.get_all_positions()
        except Exception as exc:
            _handle_sdk_error(exc)
            raise

        return [
            BrokerPosition(
                symbol          = str(pos.symbol),
                qty             = float(pos.qty),
                market_value    = float(pos.market_value),
                unrealized_pl   = float(pos.unrealized_pl),
                avg_entry_price = float(pos.avg_entry_price),
                side            = _enum_val(pos.side),
            )
            for pos in positions
        ]

    @_with_retry
    def get_orders(self) -> list[OrderResult]:
        try:
            orders = self._trading.get_orders(GetOrdersRequest())
        except Exception as exc:
            _handle_sdk_error(exc)
            raise

        return [_order_to_result(o) for o in orders]

    @_with_retry
    def submit_order(
        self,
        symbol:      str,
        qty:         float,
        side:        str,
        order_type:  str,
        limit_price: Optional[float] = None,
    ) -> OrderResult:
        request_id = str(uuid.uuid4())

        body: dict = {
            "symbol":        symbol,
            "qty":           str(qty),
            "side":          side.lower(),
            "type":          order_type.lower(),
            "time_in_force": "day",
        }
        if limit_price is not None:
            body["limit_price"] = str(limit_price)

        url = f"{self._base_url}/orders"
        try:
            resp = self._session.post(
                url,
                json=body,
                headers={"X-Request-Id": request_id},
                timeout=10,
            )
        except requests.RequestException as exc:
            raise BrokerUnavailableError(str(exc)) from exc

        if not resp.ok:
            _handle_http_error(resp)

        data = resp.json()
        logger.info(
            "Order submitted: symbol=%s qty=%s side=%s type=%s request_id=%s",
            symbol, qty, side, order_type, request_id,
        )
        return OrderResult(
            order_id         = str(data.get("id", "")),
            client_order_id  = str(data.get("client_order_id", "")),
            symbol           = str(data.get("symbol", symbol)),
            qty              = float(data.get("qty") or qty),
            side             = str(data.get("side", side)),
            order_type       = str(data.get("type", order_type)),
            status           = str(data.get("status", "")),
            filled_qty       = float(data.get("filled_qty") or 0),
            filled_avg_price = float(data["filled_avg_price"]) if data.get("filled_avg_price") else None,
            request_id       = request_id,
        )

    @_with_retry
    def cancel_order(self, order_id: str) -> bool:
        try:
            self._trading.cancel_order_by_id(order_id)
            return True
        except Exception as exc:
            if getattr(exc, "status_code", None) == 404:
                return False
            _handle_sdk_error(exc)
            raise

    @_with_retry
    def get_option_chain(self, symbol: str) -> list[OptionContract]:
        try:
            chain = self._options.get_option_chain(
                OptionChainRequest(underlying_symbol=symbol)
            )
        except Exception as exc:
            _handle_sdk_error(exc)
            raise

        contracts: list[OptionContract] = []
        for occ_sym, snap in chain.items():
            parsed = _parse_occ_symbol(occ_sym)
            if parsed is None:
                logger.debug("Skipping unparseable OCC symbol: %s", occ_sym)
                continue
            underlying, expiration, option_type, strike = parsed

            greeks = getattr(snap, "greeks", None)
            delta  = float(greeks.delta) if greeks and getattr(greeks, "delta", None) is not None else None
            iv_raw = getattr(snap, "implied_volatility", None)
            iv     = float(iv_raw) if iv_raw is not None else None
            quote  = getattr(snap, "latest_quote", None)
            bid    = float(quote.bid_price) if quote and getattr(quote, "bid_price", None) is not None else None
            ask    = float(quote.ask_price) if quote and getattr(quote, "ask_price", None) is not None else None

            contracts.append(OptionContract(
                symbol             = occ_sym,
                underlying         = underlying,
                expiration         = expiration,
                strike             = strike,
                option_type        = option_type,
                delta              = delta,
                implied_volatility = iv,
                bid                = bid,
                ask                = ask,
            ))

        return contracts

    @_with_retry
    def submit_order_notional(
        self,
        symbol:       str,
        notional_usd: float,
        side:         str,
    ) -> OrderResult:
        """Submit a notional (dollar-amount) market order for fractional crypto.

        Uses ``notional`` instead of ``qty`` and ``gtc`` time-in-force, which
        are required for Alpaca crypto orders.
        """
        request_id = str(uuid.uuid4())

        body: dict = {
            "symbol":        symbol,
            "notional":      str(notional_usd),
            "side":          side.lower(),
            "type":          "market",
            "time_in_force": "gtc",
        }

        url = f"{self._base_url}/orders"
        try:
            resp = self._session.post(
                url,
                json=body,
                headers={"X-Request-Id": request_id},
                timeout=10,
            )
        except requests.RequestException as exc:
            raise BrokerUnavailableError(str(exc)) from exc

        if not resp.ok:
            _handle_http_error(resp)

        data = resp.json()
        logger.info(
            "Notional order submitted: symbol=%s notional=%s side=%s request_id=%s",
            symbol, notional_usd, side, request_id,
        )
        return OrderResult(
            order_id         = str(data.get("id", "")),
            client_order_id  = str(data.get("client_order_id", "")),
            symbol           = str(data.get("symbol", symbol)),
            qty              = float(data.get("qty") or 0),
            side             = str(data.get("side", side)),
            order_type       = str(data.get("type", "market")),
            status           = str(data.get("status", "")),
            filled_qty       = float(data.get("filled_qty") or 0),
            filled_avg_price = (
                float(data["filled_avg_price"])
                if data.get("filled_avg_price") else None
            ),
            request_id       = request_id,
        )

    @_with_retry
    def is_market_open(self) -> bool:
        try:
            clock = self._trading.get_clock()
            return bool(clock.is_open)
        except Exception as exc:
            _handle_sdk_error(exc)
            raise

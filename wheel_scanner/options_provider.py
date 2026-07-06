"""
wheel_scanner/options_provider.py
----------------------------------
Provider selector for options-chain data.

Routes the four provider-specific operations to either the Alpaca
(options_data) or tastytrade (options_data_tastytrade) implementation based on
settings.OPTIONS_DATA_PROVIDER. Both back-ends emit the same WheelOptionLeg and
share identical signatures, so this module is a thin pass-through and the single
source of truth for the provider switch.

The stock/OHLCV path and the pure helpers (find_target_put, mid_price, …) are
provider-agnostic and remain imported directly from options_data.
"""

from __future__ import annotations

from config.credentials import ConfigurationError
from config import settings


def _provider_module():
    provider = settings.OPTIONS_DATA_PROVIDER
    if provider == "alpaca":
        from wheel_scanner import options_data
        return options_data
    if provider == "tastytrade":
        from wheel_scanner import options_data_tastytrade
        return options_data_tastytrade
    raise ConfigurationError(
        f"unknown OPTIONS_DATA_PROVIDER: {provider!r} (expected 'alpaca' or 'tastytrade')"
    )


def build_options_client():
    """Construct the options-data client for the active provider.

    Alpaca → OptionHistoricalDataClient; tastytrade → authenticated Session.
    The stock client is always Alpaca and is built separately by the scanner.
    """
    provider = settings.OPTIONS_DATA_PROVIDER
    if provider == "tastytrade":
        from wheel_scanner.options_data_tastytrade import build_session
        return build_session(is_test=settings.TASTYTRADE_USE_SANDBOX)
    if provider == "alpaca":
        from alpaca.data.historical import OptionHistoricalDataClient
        from config.credentials import load_credentials
        creds = load_credentials()
        return OptionHistoricalDataClient(
            api_key    = creds.api_key,
            secret_key = creds.api_secret,
        )
    raise ConfigurationError(
        f"unknown OPTIONS_DATA_PROVIDER: {provider!r} (expected 'alpaca' or 'tastytrade')"
    )


def fetch_put_chain(symbol, client, min_dte: int = 21, max_dte: int = 45):
    return _provider_module().fetch_put_chain(symbol, client, min_dte, max_dte)


def count_expiry_cycles(symbol, client) -> int:
    return _provider_module().count_expiry_cycles(symbol, client)


def compute_ivr(symbol, client, stock_client, lookback_days: int = 252):
    return _provider_module().compute_ivr(symbol, client, stock_client, lookback_days)

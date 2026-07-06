"""
tests/test_options_provider.py
-------------------------------
Provider dispatch (options_provider) and the tastytrade adapter's pure mapping
logic. All mocked — no live tastytrade/Alpaca calls.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from config import settings
from config.credentials import ConfigurationError
import wheel_scanner.options_provider as op
import wheel_scanner.options_data as od
import wheel_scanner.options_data_tastytrade as tt


class TestProviderDispatch:

    def test_alpaca_route_resolves_options_data(self, monkeypatch):
        monkeypatch.setattr(settings, "OPTIONS_DATA_PROVIDER", "alpaca")
        assert op._provider_module() is od

    def test_tastytrade_route_resolves_adapter(self, monkeypatch):
        monkeypatch.setattr(settings, "OPTIONS_DATA_PROVIDER", "tastytrade")
        assert op._provider_module() is tt

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "OPTIONS_DATA_PROVIDER", "bogus")
        with pytest.raises(ConfigurationError):
            op._provider_module()

    def test_fetch_put_chain_delegates_with_args(self, monkeypatch):
        monkeypatch.setattr(settings, "OPTIONS_DATA_PROVIDER", "alpaca")
        captured = {}

        def fake(*a):
            captured["args"] = a
            return ["LEG"]

        monkeypatch.setattr(od, "fetch_put_chain", fake)
        result = op.fetch_put_chain("MSTR", "client", min_dte=21, max_dte=45)
        assert result == ["LEG"]
        assert captured["args"] == ("MSTR", "client", 21, 45)

    def test_count_expiry_cycles_delegates(self, monkeypatch):
        monkeypatch.setattr(settings, "OPTIONS_DATA_PROVIDER", "tastytrade")
        monkeypatch.setattr(tt, "count_expiry_cycles", lambda *a: 7)
        assert op.count_expiry_cycles("MSTR", "sess") == 7

    def test_compute_ivr_delegates_with_stock_client(self, monkeypatch):
        monkeypatch.setattr(settings, "OPTIONS_DATA_PROVIDER", "tastytrade")
        captured = {}

        def fake(*a):
            captured["args"] = a
            return 42.0

        monkeypatch.setattr(tt, "compute_ivr", fake)
        result = op.compute_ivr("MSTR", "sess", "stock", lookback_days=252)
        assert result == 42.0
        assert captured["args"] == ("MSTR", "sess", "stock", 252)

    def test_build_options_client_routes_to_tastytrade(self, monkeypatch):
        monkeypatch.setattr(settings, "OPTIONS_DATA_PROVIDER", "tastytrade")
        monkeypatch.setattr(settings, "TASTYTRADE_USE_SANDBOX", False)
        sentinel = object()
        captured = {}

        def fake(is_test):
            captured["is_test"] = is_test
            return sentinel

        monkeypatch.setattr(tt, "build_session", fake)
        assert op.build_options_client() is sentinel
        assert captured["is_test"] is False


def _option(symbol="MSTR  260807P00093000", strike="93.0", exp=date(2026, 8, 7)):
    return MagicMock(symbol=symbol, strike_price=Decimal(strike), expiration_date=exp)


class TestTastytradeLegMapping:

    def test_maps_all_event_fields(self):
        leg = tt._leg_from_events(
            _option(), "MSTR", 33,
            greeks  = MagicMock(delta=-0.30, volatility=0.92),
            quote   = MagicMock(bid_price=7.80, ask_price=8.05),
            summary = MagicMock(open_interest=Decimal("1512")),
            trade   = MagicMock(day_volume=Decimal("250")),
        )
        assert leg.underlying == "MSTR"
        assert leg.option_type == "put"
        assert leg.strike == 93.0
        assert leg.dte == 33
        assert leg.delta == pytest.approx(-0.30)
        assert leg.implied_volatility == pytest.approx(0.92)
        assert leg.bid == pytest.approx(7.80)
        assert leg.ask == pytest.approx(8.05)
        assert leg.open_interest == 1512 and isinstance(leg.open_interest, int)
        assert leg.volume_today == 250 and isinstance(leg.volume_today, int)

    def test_missing_events_degrade_to_none(self):
        leg = tt._leg_from_events(_option(), "MSTR", 33, None, None, None, None)
        assert leg.delta is None
        assert leg.implied_volatility is None
        assert leg.bid is None and leg.ask is None
        assert leg.open_interest is None
        assert leg.volume_today is None
        # Structural fields still populated from the contract itself.
        assert leg.strike == 93.0 and leg.dte == 33

    def test_absent_open_interest_and_volume_are_none(self):
        leg = tt._leg_from_events(
            _option(), "MSTR", 33,
            greeks  = MagicMock(delta=-0.25, volatility=0.8),
            quote   = MagicMock(bid_price=1.0, ask_price=1.2),
            summary = MagicMock(open_interest=None),
            trade   = MagicMock(day_volume=None),
        )
        assert leg.open_interest is None
        assert leg.volume_today is None


class TestTastytradeComputeIvr:

    def test_combines_current_iv_and_realized_range(self, monkeypatch):
        # current IV 0.9 within realized range [0.5, 1.5] → IVR 40.0
        monkeypatch.setattr(tt, "_current_iv_async", lambda *a, **k: "coro")
        monkeypatch.setattr(tt, "_run", lambda coro: 0.9)
        monkeypatch.setattr(tt, "realized_vol_range", lambda *a, **k: (0.5, 1.5))
        assert tt.compute_ivr("MSTR", MagicMock(), MagicMock()) == pytest.approx(40.0)

    def test_none_when_current_iv_missing(self, monkeypatch):
        monkeypatch.setattr(tt, "_current_iv_async", lambda *a, **k: "coro")
        monkeypatch.setattr(tt, "_run", lambda coro: None)
        monkeypatch.setattr(tt, "realized_vol_range", lambda *a, **k: (0.5, 1.5))
        assert tt.compute_ivr("MSTR", MagicMock(), MagicMock()) is None

    def test_none_when_realized_range_missing(self, monkeypatch):
        monkeypatch.setattr(tt, "_current_iv_async", lambda *a, **k: "coro")
        monkeypatch.setattr(tt, "_run", lambda coro: 0.9)
        monkeypatch.setattr(tt, "realized_vol_range", lambda *a, **k: None)
        assert tt.compute_ivr("MSTR", MagicMock(), MagicMock()) is None

    def test_ivr_clamped_to_100(self, monkeypatch):
        # current IV above the realized max → clamp to 100
        monkeypatch.setattr(tt, "_current_iv_async", lambda *a, **k: "coro")
        monkeypatch.setattr(tt, "_run", lambda coro: 2.0)
        monkeypatch.setattr(tt, "realized_vol_range", lambda *a, **k: (0.5, 1.5))
        assert tt.compute_ivr("MSTR", MagicMock(), MagicMock()) == 100.0

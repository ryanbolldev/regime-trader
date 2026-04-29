"""
tests/test_market_data.py
--------------------------
Unit tests for core/market_data.py.
All network calls are mocked — no real Alpaca API calls are made.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

import core.market_data as md


# ---------------------------------------------------------------------------
# Mock primitives
# ---------------------------------------------------------------------------

def _ts(year: int = 2024, month: int = 1, day: int = 2) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


class _Bar:
    """Minimal mock of an alpaca-py Bar."""
    def __init__(self, timestamp, open=100.0, high=105.0, low=99.0,
                 close=103.0, volume=1_000_000.0):
        self.timestamp = timestamp
        self.open   = open
        self.high   = high
        self.low    = low
        self.close  = close
        self.volume = volume


class _BarSet:
    """Minimal mock of an alpaca-py BarSet."""
    def __init__(self, symbol: str, bars: list):
        self._data = {symbol: bars}

    def __getitem__(self, key):
        return self._data[key]


def _make_bars(symbol: str, n: int = 3) -> _BarSet:
    bars = [_Bar(_ts(2024, 1, i + 1)) for i in range(n)]
    return _BarSet(symbol, bars)


# ---------------------------------------------------------------------------
# Fixtures: replace module-level singletons before every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_stock_client(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(md, "_stock_client", client)
    return client


@pytest.fixture(autouse=True)
def mock_alpaca_client(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(md, "_alpaca_client", client)
    return client


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Suppress all time.sleep calls so retry tests are instant."""
    monkeypatch.setattr("core.market_data.time.sleep", lambda _: None)


START = datetime(2024, 1, 1, tzinfo=timezone.utc)
END   = datetime(2024, 3, 31, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# TestGetHistoricalBars
# ---------------------------------------------------------------------------

class TestGetHistoricalBars:
    def test_returns_dataframe(self, mock_stock_client):
        mock_stock_client.get_stock_bars.return_value = _make_bars("SPY", 5)
        df = md.get_historical_bars("SPY", START, END, "1Day")
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self, mock_stock_client):
        mock_stock_client.get_stock_bars.return_value = _make_bars("SPY")
        df = md.get_historical_bars("SPY", START, END, "1Day")
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}

    def test_index_is_utc_datetimeindex(self, mock_stock_client):
        mock_stock_client.get_stock_bars.return_value = _make_bars("SPY")
        df = md.get_historical_bars("SPY", START, END, "1Day")
        assert isinstance(df.index, pd.DatetimeIndex)
        assert str(df.index.tz) == "UTC"

    def test_row_count_matches_bars(self, mock_stock_client):
        mock_stock_client.get_stock_bars.return_value = _make_bars("SPY", 10)
        df = md.get_historical_bars("SPY", START, END, "1Day")
        assert len(df) == 10

    def test_raises_value_error_on_empty_response(self, mock_stock_client):
        mock_stock_client.get_stock_bars.return_value = _BarSet("SPY", [])
        with pytest.raises(ValueError, match="No bars returned"):
            md.get_historical_bars("SPY", START, END, "1Day")

    def test_raises_value_error_on_missing_symbol(self, mock_stock_client):
        mock_stock_client.get_stock_bars.return_value = _BarSet("QQQ", [_Bar(_ts())])
        with pytest.raises(ValueError, match="No bars returned"):
            md.get_historical_bars("SPY", START, END, "1Day")

    def test_invalid_timeframe_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown timeframe"):
            md.get_historical_bars("SPY", START, END, "bad")

    def test_accepts_1day_timeframe(self, mock_stock_client):
        mock_stock_client.get_stock_bars.return_value = _make_bars("SPY")
        md.get_historical_bars("SPY", START, END, "1Day")  # must not raise

    def test_accepts_1hour_timeframe(self, mock_stock_client):
        mock_stock_client.get_stock_bars.return_value = _make_bars("SPY")
        md.get_historical_bars("SPY", START, END, "1Hour")

    def test_accepts_5min_timeframe(self, mock_stock_client):
        mock_stock_client.get_stock_bars.return_value = _make_bars("SPY")
        md.get_historical_bars("SPY", START, END, "5Min")

    def test_float_values_in_dataframe(self, mock_stock_client):
        mock_stock_client.get_stock_bars.return_value = _make_bars("SPY", 1)
        df = md.get_historical_bars("SPY", START, END, "1Day")
        assert df["close"].dtype == float
        assert df["volume"].dtype == float


# ---------------------------------------------------------------------------
# TestGetLatestBar
# ---------------------------------------------------------------------------

class TestGetLatestBar:
    def test_returns_series(self, mock_stock_client):
        mock_stock_client.get_stock_latest_bar.return_value = {"SPY": _Bar(_ts())}
        result = md.get_latest_bar("SPY")
        assert isinstance(result, pd.Series)

    def test_has_correct_fields(self, mock_stock_client):
        mock_stock_client.get_stock_latest_bar.return_value = {"SPY": _Bar(_ts())}
        s = md.get_latest_bar("SPY")
        assert set(s.index) == {"open", "high", "low", "close", "volume"}

    def test_series_name_is_timestamp(self, mock_stock_client):
        ts = _ts(2024, 6, 15)
        mock_stock_client.get_stock_latest_bar.return_value = {"SPY": _Bar(ts)}
        s = md.get_latest_bar("SPY")
        assert s.name == ts

    def test_values_are_floats(self, mock_stock_client):
        mock_stock_client.get_stock_latest_bar.return_value = {
            "SPY": _Bar(_ts(), open=100, high=110, low=99, close=105, volume=500_000)
        }
        s = md.get_latest_bar("SPY")
        assert s["close"] == 105.0
        assert s["volume"] == 500_000.0

    def test_raises_value_error_when_symbol_missing(self, mock_stock_client):
        mock_stock_client.get_stock_latest_bar.return_value = {}
        with pytest.raises(ValueError, match="No latest bar"):
            md.get_latest_bar("SPY")

    def test_invalid_timeframe_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown timeframe"):
            md.get_latest_bar("SPY", timeframe="bad")

    def test_naive_timestamp_gets_utc(self, mock_stock_client):
        naive_ts = datetime(2024, 1, 5)  # no tzinfo
        mock_stock_client.get_stock_latest_bar.return_value = {"SPY": _Bar(naive_ts)}
        s = md.get_latest_bar("SPY")
        assert s.name.tzinfo is not None


# ---------------------------------------------------------------------------
# TestGetLatestBars
# ---------------------------------------------------------------------------

class TestGetLatestBars:
    def test_returns_dataframe(self, mock_stock_client):
        mock_stock_client.get_stock_bars.return_value = _make_bars("SPY", 20)
        df = md.get_latest_bars("SPY", 10)
        assert isinstance(df, pd.DataFrame)

    def test_truncates_to_n_bars(self, mock_stock_client):
        mock_stock_client.get_stock_bars.return_value = _make_bars("SPY", 20)
        df = md.get_latest_bars("SPY", 5)
        assert len(df) == 5

    def test_returns_all_when_fewer_than_n_bars(self, mock_stock_client):
        mock_stock_client.get_stock_bars.return_value = _make_bars("SPY", 3)
        df = md.get_latest_bars("SPY", 10)
        assert len(df) == 3

    def test_passes_timeframe_to_historical(self, mock_stock_client):
        mock_stock_client.get_stock_bars.return_value = _make_bars("SPY", 5)
        md.get_latest_bars("SPY", 5, timeframe="1Hour")
        request = mock_stock_client.get_stock_bars.call_args[0][0]
        from alpaca.data.timeframe import TimeFrame
        assert str(request.timeframe) == str(TimeFrame.Hour)

    def test_propagates_value_error_on_no_data(self, mock_stock_client):
        mock_stock_client.get_stock_bars.return_value = _BarSet("SPY", [])
        with pytest.raises(ValueError):
            md.get_latest_bars("SPY", 10)


# ---------------------------------------------------------------------------
# TestIsMarketOpen
# ---------------------------------------------------------------------------

class TestIsMarketOpen:
    def test_delegates_to_alpaca_client(self, mock_alpaca_client):
        mock_alpaca_client.is_market_open.return_value = True
        assert md.is_market_open() is True

    def test_returns_false_when_closed(self, mock_alpaca_client):
        mock_alpaca_client.is_market_open.return_value = False
        assert md.is_market_open() is False

    def test_calls_is_market_open_once(self, mock_alpaca_client):
        mock_alpaca_client.is_market_open.return_value = True
        md.is_market_open()
        mock_alpaca_client.is_market_open.assert_called_once()


# ---------------------------------------------------------------------------
# TestGetOptionChain
# ---------------------------------------------------------------------------

class TestGetOptionChain:
    def test_delegates_to_alpaca_client(self, mock_alpaca_client):
        mock_alpaca_client.get_option_chain.return_value = ["contract1", "contract2"]
        result = md.get_option_chain("SPY")
        assert result == ["contract1", "contract2"]

    def test_passes_symbol_to_client(self, mock_alpaca_client):
        mock_alpaca_client.get_option_chain.return_value = []
        md.get_option_chain("QQQ")
        mock_alpaca_client.get_option_chain.assert_called_once_with("QQQ")

    def test_empty_chain_returned_as_empty_list(self, mock_alpaca_client):
        mock_alpaca_client.get_option_chain.return_value = []
        assert md.get_option_chain("SPY") == []


# ---------------------------------------------------------------------------
# TestRetryLogic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    def test_retries_three_times_on_error(self, mock_stock_client):
        err = Exception("network error")
        mock_stock_client.get_stock_bars.side_effect = [
            err, err, _make_bars("SPY")
        ]
        df = md.get_historical_bars("SPY", START, END, "1Day")
        assert not df.empty
        assert mock_stock_client.get_stock_bars.call_count == 3

    def test_raises_after_three_failures(self, mock_stock_client):
        mock_stock_client.get_stock_bars.side_effect = Exception("persistent error")
        with pytest.raises(Exception, match="persistent error"):
            md.get_historical_bars("SPY", START, END, "1Day")
        assert mock_stock_client.get_stock_bars.call_count == 3

    def test_sleeps_between_retries(self, mock_stock_client, monkeypatch):
        sleep_calls: list[float] = []
        monkeypatch.setattr("core.market_data.time.sleep", lambda s: sleep_calls.append(s))
        err = Exception("boom")
        mock_stock_client.get_stock_bars.side_effect = [err, err, _make_bars("SPY")]
        md.get_historical_bars("SPY", START, END, "1Day")
        assert sleep_calls == [1.0, 2.0]

    def test_rate_limit_429_waits_60_seconds(self, mock_stock_client, monkeypatch):
        sleep_calls: list[float] = []
        monkeypatch.setattr("core.market_data.time.sleep", lambda s: sleep_calls.append(s))
        rate_err = Exception("too many requests")
        rate_err.status_code = 429   # type: ignore[attr-defined]
        mock_stock_client.get_stock_bars.side_effect = [rate_err, _make_bars("SPY")]
        md.get_historical_bars("SPY", START, END, "1Day")
        assert 60.0 in sleep_calls

    def test_rate_limit_retry_succeeds(self, mock_stock_client):
        rate_err = Exception("rate limited")
        rate_err.status_code = 429   # type: ignore[attr-defined]
        mock_stock_client.get_stock_bars.side_effect = [rate_err, _make_bars("SPY")]
        df = md.get_historical_bars("SPY", START, END, "1Day")
        assert not df.empty

    def test_single_success_no_sleep(self, mock_stock_client, monkeypatch):
        sleep_calls: list[float] = []
        monkeypatch.setattr("core.market_data.time.sleep", lambda s: sleep_calls.append(s))
        mock_stock_client.get_stock_bars.return_value = _make_bars("SPY")
        md.get_historical_bars("SPY", START, END, "1Day")
        assert sleep_calls == []

    def test_get_latest_bar_retries(self, mock_stock_client):
        err = Exception("transient")
        mock_stock_client.get_stock_latest_bar.side_effect = [
            err, {"SPY": _Bar(_ts())}
        ]
        s = md.get_latest_bar("SPY")
        assert s is not None
        assert mock_stock_client.get_stock_latest_bar.call_count == 2

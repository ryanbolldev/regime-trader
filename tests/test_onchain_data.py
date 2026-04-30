"""
tests/test_onchain_data.py
---------------------------
Unit tests for core/onchain_data.py.
All HTTP calls are mocked — no real network requests are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import core.onchain_data as od
from core.onchain_data import (
    OnChainFeatures,
    _build_features,
    _compute_score,
    _fetch_blockchain_info,
    _fetch_coingecko,
    _fetch_mempool,
    _last_hashrate_from_data,
    _neutral_default,
    fire_signal_if_threshold,
    get_onchain_features,
)


# ---------------------------------------------------------------------------
# Sample API payloads
# ---------------------------------------------------------------------------

_CG_PAYLOAD = {
    "usd":             45_000.0,
    "usd_24h_change":  3.5,
    "usd_24h_vol":     25_000_000_000.0,
    "usd_market_cap":  875_000_000_000.0,
}

_FEES_PAYLOAD = {
    "fastestFee":  50,
    "halfHourFee": 30,
    "hourFee":     20,
    "economyFee":  10,
    "minimumFee":  5,
}

_HASHRATE_PAYLOAD = {
    "currentHashrate":    520e18,
    "currentDifficulty":  86_871_576,
    "hashrates": [
        {"timestamp": 1_700_000_000, "avgHashrate": 500e18},
        {"timestamp": 1_700_086_400, "avgHashrate": 510e18},
    ],
}

_HEIGHT_PAYLOAD = "850000"

_BC_PAYLOAD = {
    "market_price_usd":                   45_000.0,
    "hash_rate":                           500_000.0,
    "estimated_transaction_volume_usd":    12_000_000_000.0,
    "n_tx":                                350_000,
}


def _mock_response(json_data=None, text_data="", status_code=200):
    m = MagicMock()
    m.status_code = status_code
    m.raise_for_status.return_value = None
    if json_data is not None:
        m.json.return_value = json_data
    m.text = text_data
    return m


def _url_router(url, **kwargs):
    """Route requests.get calls by URL substring to the appropriate mock."""
    if "coingecko" in url:
        return _mock_response({"bitcoin": _CG_PAYLOAD})
    if "fees/recommended" in url:
        return _mock_response(_FEES_PAYLOAD)
    if "hashrate" in url:
        return _mock_response(_HASHRATE_PAYLOAD)
    if "blocks/tip/height" in url:
        return _mock_response(text_data=_HEIGHT_PAYLOAD)
    if "blockchain.info" in url:
        return _mock_response(_BC_PAYLOAD)
    raise ValueError(f"Unexpected URL in test: {url}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_cache():
    """Reset module-level cache and prev-score before every test."""
    od._cache_result = None
    od._cache_time   = 0.0
    od._prev_score   = None
    yield
    od._cache_result = None
    od._cache_time   = 0.0
    od._prev_score   = None


@pytest.fixture()
def mock_requests():
    with patch("core.onchain_data.requests.get", side_effect=_url_router) as m:
        yield m


# ---------------------------------------------------------------------------
# TestFetchCoingecko
# ---------------------------------------------------------------------------

class TestFetchCoingecko:

    def test_returns_btc_price(self, mock_requests):
        data = _fetch_coingecko()
        assert data["usd"] == 45_000.0

    def test_returns_24h_change(self, mock_requests):
        data = _fetch_coingecko()
        assert data["usd_24h_change"] == 3.5

    def test_returns_volume(self, mock_requests):
        data = _fetch_coingecko()
        assert data["usd_24h_vol"] == 25_000_000_000.0

    def test_returns_market_cap(self, mock_requests):
        data = _fetch_coingecko()
        assert data["usd_market_cap"] == 875_000_000_000.0

    def test_raises_on_http_error(self):
        bad = MagicMock()
        bad.raise_for_status.side_effect = Exception("HTTP 429")
        with patch("core.onchain_data.requests.get", return_value=bad):
            with pytest.raises(Exception):
                _fetch_coingecko()


# ---------------------------------------------------------------------------
# TestFetchMempool
# ---------------------------------------------------------------------------

class TestFetchMempool:

    def test_returns_fee_dict(self, mock_requests):
        data = _fetch_mempool()
        assert data["fees"]["fastestFee"] == 50

    def test_returns_hashrate_dict(self, mock_requests):
        data = _fetch_mempool()
        assert "hashrates" in data["hashrate"]

    def test_returns_block_height_as_int(self, mock_requests):
        data = _fetch_mempool()
        assert data["height"] == 850_000
        assert isinstance(data["height"], int)

    def test_raises_on_fees_http_error(self):
        def bad_router(url, **kwargs):
            m = MagicMock()
            if "fees/recommended" in url:
                m.raise_for_status.side_effect = Exception("503")
            else:
                m.raise_for_status.return_value = None
                m.json.return_value = {}
                m.text = "0"
            return m

        with patch("core.onchain_data.requests.get", side_effect=bad_router):
            with pytest.raises(Exception):
                _fetch_mempool()


# ---------------------------------------------------------------------------
# TestFetchBlockchainInfo
# ---------------------------------------------------------------------------

class TestFetchBlockchainInfo:

    def test_returns_tx_volume(self, mock_requests):
        data = _fetch_blockchain_info()
        assert data["estimated_transaction_volume_usd"] == 12_000_000_000.0

    def test_returns_n_tx(self, mock_requests):
        data = _fetch_blockchain_info()
        assert data["n_tx"] == 350_000

    def test_raises_on_http_error(self):
        bad = MagicMock()
        bad.raise_for_status.side_effect = Exception("HTTP 500")
        with patch("core.onchain_data.requests.get", return_value=bad):
            with pytest.raises(Exception):
                _fetch_blockchain_info()


# ---------------------------------------------------------------------------
# TestLastHashrate
# ---------------------------------------------------------------------------

class TestLastHashrate:

    def test_prefers_currentHashrate_field(self):
        data = {"currentHashrate": 520e18, "hashrates": [{"avgHashrate": 500e18}]}
        assert _last_hashrate_from_data(data) == 520e18

    def test_falls_back_to_last_hashrates_entry(self):
        data = {"hashrates": [{"avgHashrate": 490e18}, {"avgHashrate": 510e18}]}
        assert _last_hashrate_from_data(data) == 510e18

    def test_returns_zero_on_empty(self):
        assert _last_hashrate_from_data({}) == 0.0

    def test_returns_zero_on_empty_hashrates_list(self):
        assert _last_hashrate_from_data({"hashrates": []}) == 0.0


# ---------------------------------------------------------------------------
# TestBuildFeatures
# ---------------------------------------------------------------------------

class TestBuildFeatures:

    def test_returns_onchain_features_instance(self):
        cg = _CG_PAYLOAD
        mp = {"fees": _FEES_PAYLOAD, "hashrate": _HASHRATE_PAYLOAD, "height": 850_000}
        bc = _BC_PAYLOAD
        result = _build_features(cg, mp, bc)
        assert isinstance(result, OnChainFeatures)

    def test_price_populated(self):
        mp = {"fees": _FEES_PAYLOAD, "hashrate": _HASHRATE_PAYLOAD, "height": 850_000}
        result = _build_features(_CG_PAYLOAD, mp, _BC_PAYLOAD)
        assert result.btc_price_usd == 45_000.0

    def test_fee_pressure_is_ratio(self):
        mp = {"fees": {"fastestFee": 50, "minimumFee": 5}, "hashrate": {}, "height": 0}
        result = _build_features(_CG_PAYLOAD, mp, _BC_PAYLOAD)
        assert result.mempool_fee_pressure == pytest.approx(10.0)

    def test_fee_pressure_guards_zero_minimum(self):
        mp = {"fees": {"fastestFee": 50, "minimumFee": 0}, "hashrate": {}, "height": 0}
        result = _build_features(_CG_PAYLOAD, mp, _BC_PAYLOAD)
        assert result.mempool_fee_pressure == pytest.approx(50.0)

    def test_block_height_populated(self):
        mp = {"fees": _FEES_PAYLOAD, "hashrate": _HASHRATE_PAYLOAD, "height": 850_000}
        result = _build_features(_CG_PAYLOAD, mp, _BC_PAYLOAD)
        assert result.block_height == 850_000

    def test_tx_volume_populated(self):
        mp = {"fees": _FEES_PAYLOAD, "hashrate": _HASHRATE_PAYLOAD, "height": 0}
        result = _build_features(_CG_PAYLOAD, mp, _BC_PAYLOAD)
        assert result.tx_volume_usd == 12_000_000_000.0

    def test_score_is_float_in_range(self):
        mp = {"fees": _FEES_PAYLOAD, "hashrate": _HASHRATE_PAYLOAD, "height": 0}
        result = _build_features(_CG_PAYLOAD, mp, _BC_PAYLOAD)
        assert -1.0 <= result.on_chain_score <= 1.0


# ---------------------------------------------------------------------------
# TestComputeScore
# ---------------------------------------------------------------------------

class TestComputeScore:

    def test_bullish_conditions_positive(self):
        score = _compute_score(
            price_change_pct=10.0,  # strong positive momentum
            fee_pressure=1.2,       # low congestion
            hash_rate=500e18,       # above baseline
            tx_volume_usd=20e9,     # above baseline
        )
        assert score > 0.0

    def test_bearish_conditions_negative(self):
        score = _compute_score(
            price_change_pct=-10.0,  # strong negative momentum
            fee_pressure=9.0,        # high congestion
            hash_rate=100e18,        # well below baseline
            tx_volume_usd=1e9,       # well below baseline
        )
        assert score < 0.0

    def test_neutral_conditions_near_zero(self):
        score = _compute_score(
            price_change_pct=0.0,
            fee_pressure=3.0,    # midpoint of fee_score formula → 0.0
            hash_rate=400e18,    # exactly at baseline → 0.0
            tx_volume_usd=10e9,  # exactly at baseline → 0.0
        )
        assert abs(score) < 0.1

    def test_score_clamped_to_minus1(self):
        score = _compute_score(-100.0, 100.0, 0.0, 0.0)
        assert score >= -1.0

    def test_score_clamped_to_plus1(self):
        score = _compute_score(100.0, 1.0, 1000e18, 1000e9)
        assert score <= 1.0

    def test_positive_price_increases_score(self):
        base  = _compute_score(0.0, 3.0, 400e18, 10e9)
        high  = _compute_score(5.0, 3.0, 400e18, 10e9)
        assert high > base

    def test_high_fee_pressure_decreases_score(self):
        low_fee  = _compute_score(0.0, 1.5, 400e18, 10e9)
        high_fee = _compute_score(0.0, 8.0, 400e18, 10e9)
        assert high_fee < low_fee


# ---------------------------------------------------------------------------
# TestNeutralDefault
# ---------------------------------------------------------------------------

class TestNeutralDefault:

    def test_returns_onchain_features(self):
        d = _neutral_default()
        assert isinstance(d, OnChainFeatures)

    def test_score_is_zero(self):
        assert _neutral_default().on_chain_score == 0.0

    def test_fee_pressure_is_one(self):
        assert _neutral_default().mempool_fee_pressure == 1.0


# ---------------------------------------------------------------------------
# TestGetOnChainFeatures
# ---------------------------------------------------------------------------

class TestGetOnChainFeatures:

    def test_returns_onchain_features(self, mock_requests):
        result = get_onchain_features()
        assert isinstance(result, OnChainFeatures)

    def test_populates_price(self, mock_requests):
        result = get_onchain_features()
        assert result.btc_price_usd == 45_000.0

    def test_score_in_valid_range(self, mock_requests):
        result = get_onchain_features()
        assert -1.0 <= result.on_chain_score <= 1.0

    def test_never_raises_on_network_error(self):
        with patch("core.onchain_data.requests.get", side_effect=OSError("timeout")):
            result = get_onchain_features()
        assert isinstance(result, OnChainFeatures)

    def test_returns_neutral_default_on_first_failure(self):
        with patch("core.onchain_data.requests.get", side_effect=OSError("timeout")):
            result = get_onchain_features()
        assert result.on_chain_score == 0.0

    def test_returns_stale_cache_on_failure(self, mock_requests):
        # Prime the cache with a successful fetch
        first = get_onchain_features()
        assert first.btc_price_usd == 45_000.0

        # Now break the network — should return cached value
        with patch("core.onchain_data.requests.get", side_effect=OSError("down")):
            # Expire the cache manually so a fresh fetch is attempted
            od._cache_time = 0.0
            result = get_onchain_features()

        assert result.btc_price_usd == first.btc_price_usd

    def test_uses_cache_within_ttl(self, mock_requests):
        # First call populates cache
        get_onchain_features()
        call_count_after_first = mock_requests.call_count

        # Second call within TTL should hit cache — no new HTTP calls
        get_onchain_features()
        assert mock_requests.call_count == call_count_after_first

    def test_refetches_after_cache_expires(self, mock_requests):
        get_onchain_features()
        call_count_after_first = mock_requests.call_count

        # Expire the cache
        od._cache_time = 0.0
        get_onchain_features()
        assert mock_requests.call_count > call_count_after_first

    def test_makes_concurrent_requests_to_all_sources(self, mock_requests):
        get_onchain_features()
        called_urls = [c.args[0] for c in mock_requests.call_args_list]
        assert any("coingecko" in u for u in called_urls)
        assert any("mempool"   in u for u in called_urls)
        assert any("blockchain.info" in u for u in called_urls)


# ---------------------------------------------------------------------------
# TestFireSignalIfThreshold
# ---------------------------------------------------------------------------

class TestFireSignalIfThreshold:

    def test_fires_bullish_alert_on_upward_cross(self):
        with patch("core.onchain_data.alerts") as mock_alerts:
            fire_signal_if_threshold(score=0.6, prev_score=0.3)
            mock_alerts.send.assert_called_once()
            args = mock_alerts.send.call_args.args
            assert args[0] == "onchain_signal"
            assert "BULLISH" in args[1]

    def test_fires_bearish_alert_on_downward_cross(self):
        with patch("core.onchain_data.alerts") as mock_alerts:
            fire_signal_if_threshold(score=-0.6, prev_score=-0.3)
            mock_alerts.send.assert_called_once()
            args = mock_alerts.send.call_args.args
            assert args[0] == "onchain_signal"
            assert "BEARISH" in args[1]

    def test_no_alert_when_already_above_threshold(self):
        with patch("core.onchain_data.alerts") as mock_alerts:
            fire_signal_if_threshold(score=0.8, prev_score=0.6)
            mock_alerts.send.assert_not_called()

    def test_no_alert_when_already_below_threshold(self):
        with patch("core.onchain_data.alerts") as mock_alerts:
            fire_signal_if_threshold(score=-0.8, prev_score=-0.6)
            mock_alerts.send.assert_not_called()

    def test_no_alert_when_score_below_bullish_threshold(self):
        with patch("core.onchain_data.alerts") as mock_alerts:
            fire_signal_if_threshold(score=0.4, prev_score=0.1)
            mock_alerts.send.assert_not_called()

    def test_no_alert_when_score_above_bearish_threshold(self):
        with patch("core.onchain_data.alerts") as mock_alerts:
            fire_signal_if_threshold(score=-0.4, prev_score=-0.1)
            mock_alerts.send.assert_not_called()

    def test_bullish_alert_severity_is_info(self):
        with patch("core.onchain_data.alerts") as mock_alerts:
            fire_signal_if_threshold(score=0.7, prev_score=0.2)
            args = mock_alerts.send.call_args.args
            assert args[2] == "info"

    def test_bearish_alert_severity_is_warning(self):
        with patch("core.onchain_data.alerts") as mock_alerts:
            fire_signal_if_threshold(score=-0.7, prev_score=-0.2)
            args = mock_alerts.send.call_args.args
            assert args[2] == "warning"

    def test_exact_threshold_counts_as_cross(self):
        with patch("core.onchain_data.alerts") as mock_alerts:
            fire_signal_if_threshold(score=0.5, prev_score=0.3)
            mock_alerts.send.assert_called_once()

    def test_score_in_alert_message(self):
        with patch("core.onchain_data.alerts") as mock_alerts:
            fire_signal_if_threshold(score=0.72, prev_score=0.1)
            msg = mock_alerts.send.call_args.args[1]
            assert "0.720" in msg


# ---------------------------------------------------------------------------
# TestGetOnChainFeaturesSignalIntegration
# ---------------------------------------------------------------------------

class TestGetOnChainFeaturesSignalIntegration:

    def test_fires_alert_when_score_crosses_bullish_threshold(self, mock_requests):
        # Seed prev_score just below threshold
        od._prev_score = 0.3

        # Override score computation to return > 0.5
        def high_score_router(url, **kwargs):
            resp = _url_router(url, **kwargs)
            return resp

        # Patch _compute_score to guarantee a bullish result
        with patch("core.onchain_data._compute_score", return_value=0.8):
            with patch("core.onchain_data.alerts") as mock_alerts:
                get_onchain_features()
                mock_alerts.send.assert_called()
                call_event = mock_alerts.send.call_args.args[0]
                assert call_event == "onchain_signal"

    def test_prev_score_updated_after_fetch(self, mock_requests):
        assert od._prev_score is None
        get_onchain_features()
        assert od._prev_score is not None

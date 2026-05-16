"""
tests/test_vol_estimator.py
----------------------------
Tests for the three-component vol estimator in core/scanner/options_enricher.py
and its integration with OptionsEnricher and Reporter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from dataclasses import field
from unittest.mock import MagicMock, patch

from core.scanner.options_enricher import (
    _realized_vol_rank,
    _vix_percentile_rank,
    _vol_term_structure_score,
    compute_vol_estimate,
    OptionsEnricher,
)
from core.scanner.batch_trainer import TickerResult
from core.scanner.reporter import Reporter, _fmt_iv, _render_vol_footer_md
from core.scanner.scorer import ScoredTicker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_close(n: int = 300, seed: int = 42, drift: float = 0.0003) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(drift, 0.015, n)
    prices = 100.0 * np.exp(np.cumsum(log_rets))
    return pd.Series(prices)


def _make_ticker_result(ticker: str = "AAPL", regime: int = 3) -> TickerResult:
    return TickerResult(
        ticker               = ticker,
        current_regime       = regime,
        regime_duration_bars = 10,
        bic_score            = 1000.0,
        converged            = True,
        convergence_warning  = False,
        n_states             = 4,
    )


def _make_scored(ticker: str = "AAPL", iv_rank: float = 45.0, vol_estimated: bool = True, vol_components: dict | None = None) -> ScoredTicker:
    return ScoredTicker(
        ticker                = ticker,
        current_regime        = 3,
        regime_name           = "bull",
        long_score            = 75.0,
        short_score           = 25.0,
        direction             = "LONG",
        suggested_strategy    = "BUY_EQUITY",
        iv_rank               = iv_rank,
        spread                = None,
        low_liquidity_options = False,
        iv_data_available     = iv_rank is not None,
        regime_duration_bars  = 10,
        bic_score             = 1000.0,
        converged             = True,
        vol_estimated         = vol_estimated,
        vol_components        = vol_components or {'realized': 45.0, 'vix': None, 'term': 45.0},
    )


# ---------------------------------------------------------------------------
# _realized_vol_rank
# ---------------------------------------------------------------------------

class TestRealizedVolRank:
    def test_returns_float_in_range(self):
        close = _make_close(300)
        result = _realized_vol_rank(close, vol_window=20, lookback=252)
        assert result is not None
        assert 0.0 <= result <= 100.0

    def test_insufficient_data_returns_none(self):
        close = _make_close(50)
        result = _realized_vol_rank(close, vol_window=20, lookback=252)
        assert result is None

    def test_high_recent_vol_ranks_high(self):
        rng = np.random.default_rng(0)
        # Low-vol history then a spike
        calm = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, 272)))
        spike = calm[-1] * np.exp(np.cumsum(rng.normal(0, 0.05, 30)))
        close = pd.Series(np.concatenate([calm, spike]))
        result = _realized_vol_rank(close, vol_window=20, lookback=252)
        assert result is not None
        assert result > 50.0


# ---------------------------------------------------------------------------
# _vix_percentile_rank
# ---------------------------------------------------------------------------

class TestVixPercentileRank:
    def test_none_on_none_input(self):
        assert _vix_percentile_rank(None, lookback=252) is None

    def test_none_on_too_short_series(self):
        assert _vix_percentile_rank(pd.Series([15.0]), lookback=252) is None

    def test_high_current_scores_high(self):
        # 251 bars at 15, current bar at 40 (90th+ percentile)
        history = pd.Series([15.0] * 251 + [40.0])
        result = _vix_percentile_rank(history, lookback=252)
        assert result is not None
        assert result > 85.0

    def test_low_current_scores_low(self):
        # 251 bars at 40, current bar at 10
        history = pd.Series([40.0] * 251 + [10.0])
        result = _vix_percentile_rank(history, lookback=252)
        assert result is not None
        assert result < 15.0


# ---------------------------------------------------------------------------
# _vol_term_structure_score
# ---------------------------------------------------------------------------

class TestVolTermStructureScore:
    def test_elevated_short_vol_scores_high(self):
        # Calm long history, volatile recent bars → short_vol > long_vol
        rng = np.random.default_rng(1)
        calm   = 100 * np.exp(np.cumsum(rng.normal(0, 0.008, 80)))
        stress = calm[-1] * np.exp(np.cumsum(rng.normal(0, 0.035, 10)))
        close  = pd.Series(np.concatenate([calm, stress]))
        result = _vol_term_structure_score(close, short_window=10, long_window=60)
        assert result is not None
        assert result > 60.0

    def test_suppressed_short_vol_scores_low(self):
        # Volatile long history, calm recent bars → short_vol < long_vol
        rng = np.random.default_rng(2)
        stress = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, 80)))
        calm   = stress[-1] * np.exp(np.cumsum(rng.normal(0, 0.002, 10)))
        close  = pd.Series(np.concatenate([stress, calm]))
        result = _vol_term_structure_score(close, short_window=10, long_window=60)
        assert result is not None
        assert result < 40.0

    def test_insufficient_data_returns_none(self):
        close = pd.Series([100.0] * 20)
        assert _vol_term_structure_score(close, short_window=10, long_window=60) is None


# ---------------------------------------------------------------------------
# compute_vol_estimate
# ---------------------------------------------------------------------------

_KWARGS = dict(
    realized_weight  = 0.50,
    vix_weight       = 0.30,
    term_weight      = 0.20,
    vol_window_short = 10,
    vol_window_mid   = 20,
    vol_window_long  = 60,
    vol_lookback     = 252,
    vix_lookback     = 252,
)


class TestComputeVolEstimate:
    def test_all_components_produces_weighted_average(self):
        close = _make_close(300)
        vix   = pd.Series([15.0] * 252)
        estimate, components = compute_vol_estimate(close, vix, **_KWARGS)
        assert estimate is not None
        assert 0.0 <= estimate <= 100.0
        assert set(components.keys()) == {'realized', 'vix', 'term'}
        assert all(v is not None for v in components.values())

    def test_no_vix_redistributes_weight(self):
        close = _make_close(300)
        estimate, components = compute_vol_estimate(close, None, **_KWARGS)
        assert estimate is not None
        assert components['vix'] is None
        # Only realized and term contribute — estimate should still be 0–100
        assert 0.0 <= estimate <= 100.0

    def test_all_none_returns_none_estimate(self):
        # Too short for any component
        close = pd.Series([100.0] * 5)
        estimate, components = compute_vol_estimate(close, None, **_KWARGS)
        assert estimate is None
        assert all(v is None for v in components.values())

    def test_weight_redistribution_correct(self):
        """When only realized is available it should get 100% weight."""
        close = _make_close(300)
        # Patch the other two functions to return None
        with patch('core.scanner.options_enricher._vix_percentile_rank', return_value=None), \
             patch('core.scanner.options_enricher._vol_term_structure_score', return_value=None):
            realized_only = _realized_vol_rank(close, 20, 252)
            estimate, components = compute_vol_estimate(close, None, **_KWARGS)
        # estimate should equal realized_only (100% weight)
        assert estimate == realized_only

    def test_no_vix_weights_sum_to_one(self):
        """Redistributed weights across realized and term must sum to 1.0."""
        realized = 60.0
        term     = 40.0
        with patch('core.scanner.options_enricher._realized_vol_rank', return_value=realized), \
             patch('core.scanner.options_enricher._vix_percentile_rank', return_value=None), \
             patch('core.scanner.options_enricher._vol_term_structure_score', return_value=term):
            close = _make_close(300)
            estimate, _ = compute_vol_estimate(close, None, **_KWARGS)
        # realized weight 0.50, term weight 0.20; total 0.70
        # redistributed: realized = 60 * 0.50/0.70 + term = 40 * 0.20/0.70
        expected = round(60 * 0.50 / 0.70 + 40 * 0.20 / 0.70, 1)
        assert estimate == expected


# ---------------------------------------------------------------------------
# OptionsEnricher integration
# ---------------------------------------------------------------------------

class TestOptionsEnricherIntegration:
    def _make_enricher(self, vol_rank_override: float = 75.0) -> OptionsEnricher:
        client = MagicMock()
        client.get_option_chain.side_effect = Exception("no options")
        close  = _make_close(300)
        ohlcv  = {'AAPL': pd.DataFrame({'close': close})}
        # Patch compute_vol_estimate to return a predictable value
        self._patch_vol = patch(
            'core.scanner.options_enricher.compute_vol_estimate',
            return_value=(vol_rank_override, {'realized': vol_rank_override, 'vix': None, 'term': vol_rank_override}),
        )
        self._patch_vol.start()
        return OptionsEnricher(client=client, ohlcv_map=ohlcv)

    def teardown_method(self, _):
        try:
            self._patch_vol.stop()
        except Exception:
            pass

    def test_vol_ceiling_guard_uses_estimate(self):
        enricher = self._make_enricher(vol_rank_override=85.0)
        result   = _make_ticker_result()
        enricher.enrich([result])
        assert result.high_iv_event_risk is True

    def test_below_ceiling_not_flagged(self):
        enricher = self._make_enricher(vol_rank_override=55.0)
        result   = _make_ticker_result()
        enricher.enrich([result])
        assert result.high_iv_event_risk is False

    def test_enrichment_result_has_vol_components(self):
        enricher = self._make_enricher(vol_rank_override=50.0)
        result   = _make_ticker_result()
        enricher.enrich([result])
        assert isinstance(result.vol_components, dict)
        assert set(result.vol_components.keys()) == {'realized', 'vix', 'term'}

    def test_vol_estimated_is_true(self):
        enricher = self._make_enricher(vol_rank_override=50.0)
        result   = _make_ticker_result()
        enricher.enrich([result])
        assert result.vol_estimated is True

    def test_iv_rank_equals_vol_rank(self):
        enricher = self._make_enricher(vol_rank_override=63.0)
        result   = _make_ticker_result()
        enricher.enrich([result])
        assert result.iv_rank == result.vol_rank == 63.0


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

class TestReporter:
    def test_fmt_iv_shows_tilde_when_estimated(self):
        assert _fmt_iv(63.0, vol_estimated=True) == "~63"

    def test_fmt_iv_shows_plain_when_not_estimated(self):
        assert _fmt_iv(63.0, vol_estimated=False) == "63"

    def test_fmt_iv_shows_na_when_none(self):
        assert _fmt_iv(None) == "N/A"

    def test_reporter_shows_tilde_in_markdown(self):
        scored  = [_make_scored(iv_rank=55.0, vol_estimated=True, vol_components={'realized': 55.0, 'vix': None, 'term': 55.0})]
        reporter = Reporter(logs_dir=None)
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            reporter._logs_dir = pathlib.Path(tmp)
            _, md_path = reporter.write(scored, {}, {}, {})
            content = md_path.read_text(encoding='utf-8')
        assert "~55" in content

    def test_reporter_vix_unavailable_note(self):
        scored = [_make_scored(vol_components={'realized': 50.0, 'vix': None, 'term': 50.0})]
        lines  = _render_vol_footer_md(scored)
        assert any("VIX data unavailable" in ln for ln in lines)

    def test_reporter_vix_available_note(self):
        scored = [_make_scored(vol_components={'realized': 50.0, 'vix': 60.0, 'term': 50.0})]
        lines  = _render_vol_footer_md(scored)
        assert any("VIX rank (30%)" in ln for ln in lines)

    def test_reporter_empty_scored_no_footer(self):
        lines = _render_vol_footer_md([])
        assert lines == []

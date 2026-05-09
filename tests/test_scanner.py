"""
tests/test_scanner.py
---------------------
Unit tests for the nightly HMM scanner pipeline:
  - UniverseManager  (universe filtering)
  - BatchTrainer     (parallel HMM training)
  - OptionsEnricher  (IV rank + spread)
  - Scorer           (composite scoring)
  - Reporter         (file output + alert)
  - get_suggested_strategy
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from core.scanner.batch_trainer import BatchTrainer, TickerResult
from core.scanner.options_enricher import OptionsEnricher
from core.scanner.reporter import Reporter
from core.scanner.scorer import (
    Scorer,
    ScoredTicker,
    build_score_distribution,
    get_suggested_strategy,
)
from core.scanner.universe import UniverseManager


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lr  = rng.normal(0.0003, 0.012, n)
    c   = 100.0 * np.exp(np.cumsum(lr))
    noise = rng.uniform(0.001, 0.015, n)
    h   = c * (1 + noise)
    lo  = c * (1 - noise)
    o   = np.clip(c * (1 + rng.normal(0, 0.005, n)), lo, h)
    v   = rng.lognormal(14.0, 0.6, n).astype(int)
    idx = pd.bdate_range(end="2024-01-01", periods=n)
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": v}, index=idx)


def _ticker_result(
    ticker="SPY",
    regime=3,
    duration=10,
    bic=5000.0,
    converged=True,
    failed=False,
    iv_rank=50.0,
    spread=0.10,
    low_liq=False,
) -> TickerResult:
    return TickerResult(
        ticker=ticker,
        current_regime=regime,
        regime_duration_bars=duration,
        bic_score=bic,
        converged=converged,
        convergence_warning=not converged,
        n_states=4,
        fit_failed=failed,
        iv_rank=iv_rank,
        spread=spread,
        low_liquidity_options=low_liq,
    )


# ---------------------------------------------------------------------------
# 1. UniverseManager — filters
# ---------------------------------------------------------------------------

class TestUniverseManager:

    def test_no_client_returns_full_universe(self):
        """Without a client, all de-duped universe tickers are returned."""
        from config.settings import SP500_NASDAQ100_UNIVERSE
        mgr     = UniverseManager(client=None)
        result  = mgr.get_tradeable()
        unique  = list(dict.fromkeys(SP500_NASDAQ100_UNIVERSE))
        assert result == unique

    def test_volume_price_filter_drops_low_volume(self):
        """Tickers below min_volume threshold are excluded."""
        mock_client = MagicMock()

        def fake_bars(req):
            tickers = req.symbol_or_symbols
            data = {}
            for t in (tickers if isinstance(tickers, list) else [tickers]):
                if t == "LOW_VOL":
                    data[t] = [MagicMock(volume=100_000, close=50.0) for _ in range(10)]
                else:
                    data[t] = [MagicMock(volume=5_000_000, close=200.0) for _ in range(10)]
            return data

        mock_client._stocks.get_stock_bars.side_effect = fake_bars

        mgr    = UniverseManager(client=mock_client)
        result = mgr.get_tradeable(
            universe=["HIGH_VOL", "LOW_VOL"],
            min_volume=1_000_000,
            min_price=10.0,
        )
        assert "HIGH_VOL" in result
        assert "LOW_VOL"  not in result

    def test_barset_accessor_uses_bracket_notation(self):
        """_filter_volume_price accesses resp[ticker] — BarSet supports __getitem__."""
        mock_client = MagicMock()
        mock_client._stocks.get_stock_bars.return_value = {
            "AAPL": [MagicMock(volume=5_000_000, close=200.0) for _ in range(10)]
        }

        mgr    = UniverseManager(client=mock_client)
        result = mgr.get_tradeable(
            universe=["AAPL"], min_volume=1_000_000, min_price=10.0
        )
        assert "AAPL" in result

    def test_barset_missing_ticker_returns_empty(self):
        """Ticker absent from the batch response (KeyError) is dropped for insufficient bars."""
        mock_client = MagicMock()
        mock_client._stocks.get_stock_bars.return_value = {}  # no bars for "MISSING"

        mgr    = UniverseManager(client=mock_client)
        result = mgr.get_tradeable(
            universe=["MISSING"], min_volume=0, min_price=0.0
        )
        assert "MISSING" not in result

    def test_universe_pipeline_two_filters_only(self):
        """get_tradeable() runs only volume/price filter — no earnings method exists."""
        mgr = UniverseManager(client=None)
        assert not hasattr(mgr, "_filter_earnings"), \
            "_filter_earnings must be removed from UniverseManager"
        assert not hasattr(mgr, "_fetch_near_earnings"), \
            "_fetch_near_earnings must be removed from UniverseManager"


# ---------------------------------------------------------------------------
# 2. BatchTrainer — parallel HMM training
# ---------------------------------------------------------------------------

class TestBatchTrainer:

    def test_batch_trainer_returns_result_per_ticker(self):
        """One TickerResult per ticker in ohlcv_map."""
        ohlcv = {t: _make_ohlcv(300, seed=i) for i, t in enumerate(["SPY", "AAPL"])}
        trainer = BatchTrainer(max_workers=2, train_bars=252)
        results = trainer.run(["SPY", "AAPL"], ohlcv)
        assert len(results) == 2
        assert {r.ticker for r in results} == {"SPY", "AAPL"}

    def test_batch_trainer_missing_data_gives_fit_failed(self):
        """Tickers absent from ohlcv_map get fit_failed=True."""
        trainer = BatchTrainer(max_workers=1)
        results = trainer.run(["GHOST"], ohlcv_map={})
        assert len(results) == 1
        assert results[0].fit_failed is True

    def test_batch_trainer_result_has_valid_regime(self):
        """current_regime is -1 (unconfirmed) or 0-4 for valid fits."""
        ohlcv   = {"SPY": _make_ohlcv(300)}
        trainer = BatchTrainer(max_workers=1, train_bars=252)
        results = trainer.run(["SPY"], ohlcv)
        assert len(results) == 1
        r = results[0]
        if not r.fit_failed:
            assert -1 <= r.current_regime <= 4

    def test_batch_trainer_runs_in_parallel(self):
        """Parallel run with 3 tickers completes without deadlock."""
        tickers = ["SPY", "AAPL", "MSFT"]
        ohlcv   = {t: _make_ohlcv(300, seed=i) for i, t in enumerate(tickers)}
        trainer = BatchTrainer(max_workers=3, train_bars=252)
        results = trainer.run(tickers, ohlcv)
        assert len(results) == 3

    def test_duration_capped_at_holdout_bars(self):
        """regime_duration_bars never exceeds duration_holdout — it is out-of-sample only."""
        holdout = 10
        ohlcv   = {"SPY": _make_ohlcv(300)}
        trainer = BatchTrainer(max_workers=1, train_bars=200, duration_holdout=holdout)
        results = trainer.run(["SPY"], ohlcv)
        r = results[0]
        if not r.fit_failed:
            assert r.regime_duration_bars <= holdout

    def test_duration_holdout_default_from_settings(self):
        """Default duration_holdout comes from SCANNER_DURATION_HOLDOUT_BARS."""
        from config.settings import SCANNER_DURATION_HOLDOUT_BARS
        trainer = BatchTrainer()
        assert trainer._duration_holdout == SCANNER_DURATION_HOLDOUT_BARS


# ---------------------------------------------------------------------------
# 3. OptionsEnricher
# ---------------------------------------------------------------------------

class TestOptionsEnricher:

    def test_enricher_attaches_iv_rank(self):
        """iv_rank is populated from client.get_iv_rank()."""
        mock_client = MagicMock()
        mock_client.get_iv_rank.return_value = 65.0
        mock_client.get_option_chain.return_value = []

        result  = _ticker_result(iv_rank=None)
        enricher = OptionsEnricher(client=mock_client, max_workers=1)
        enricher.enrich([result])

        assert result.iv_rank == pytest.approx(65.0)

    def test_enricher_flags_wide_spread(self):
        """Spread above SCANNER_OPTIONS_SPREAD_MAX sets low_liquidity_options=True."""
        import datetime
        mock_client = MagicMock()
        mock_client.get_iv_rank.return_value = 50.0

        today      = datetime.date.today()
        expiry     = (today + datetime.timedelta(days=35)).isoformat()
        wide_contract = MagicMock(bid=1.00, ask=1.50, expiration=expiry)
        mock_client.get_option_chain.return_value = [wide_contract]

        result   = _ticker_result(iv_rank=None, spread=None, low_liq=False)
        enricher = OptionsEnricher(client=mock_client, max_workers=1, spread_max=0.20)
        enricher.enrich([result])

        assert result.low_liquidity_options is True
        assert result.spread == pytest.approx(0.50, rel=1e-3)

    def test_enricher_skips_failed_tickers(self):
        """fit_failed tickers are not enriched."""
        mock_client = MagicMock()
        result   = _ticker_result(failed=True, iv_rank=None)
        enricher = OptionsEnricher(client=mock_client, max_workers=1)
        enricher.enrich([result])

        mock_client.get_iv_rank.assert_not_called()
        assert result.iv_rank is None


# ---------------------------------------------------------------------------
# 4. Scorer
# ---------------------------------------------------------------------------

class TestScorer:

    def test_scorer_bull_regime_produces_high_long_score(self):
        """Bull regime (3) should score > threshold on the LONG side."""
        r      = _ticker_result(regime=3, converged=True, duration=15, iv_rank=30.0)
        scorer = Scorer(threshold=60)
        scored = scorer.score([r])
        assert len(scored) == 1
        assert scored[0].long_score >= 60

    def test_scorer_crash_regime_produces_high_short_score(self):
        """Crash regime (0) should score > threshold on the SHORT side."""
        r      = _ticker_result(regime=0, converged=True, duration=10, iv_rank=70.0)
        scorer = Scorer(threshold=60)
        scored = scorer.score([r])
        assert len(scored) == 1
        assert scored[0].short_score >= 60

    def test_scorer_excludes_below_threshold(self):
        """Neutral/uncertain result below threshold is excluded."""
        r      = _ticker_result(regime=2, converged=False, duration=1, iv_rank=50.0)
        scorer = Scorer(threshold=90)   # very high threshold
        scored = scorer.score([r])
        assert len(scored) == 0

    def test_scorer_excludes_fit_failed(self):
        r      = _ticker_result(failed=True)
        scored = Scorer().score([r])
        assert scored == []

    def test_low_liquidity_wheel_replaced_with_equity_only(self):
        """low_liquidity_options=True changes WHEEL strategy to EQUITY_ONLY."""
        r = _ticker_result(
            regime=2, converged=True, duration=15, iv_rank=65.0, low_liq=True
        )
        scored = Scorer(threshold=0).score([r])
        assert len(scored) == 1
        # WHEEL or IRON_CONDOR should be replaced
        if "WHEEL" in scored[0].suggested_strategy:
            pytest.fail("WHEEL strategy not replaced for low-liquidity ticker")


# ---------------------------------------------------------------------------
# 5. Reporter — file output + alert
# ---------------------------------------------------------------------------

class TestReporter:

    def _make_scored(self, ticker="SPY", direction="LONG") -> ScoredTicker:
        return ScoredTicker(
            ticker=ticker,
            current_regime=3,
            regime_name="bull",
            long_score=80.0,
            short_score=20.0,
            direction=direction,
            suggested_strategy="BUY_EQUITY",
            iv_rank=35.0,
            spread=0.10,
            low_liquidity_options=False,
            iv_data_available=True,
            regime_duration_bars=12,
            bic_score=4500.0,
            converged=True,
        )

    def test_reporter_writes_json(self, tmp_path):
        scored   = [self._make_scored()]
        reporter = Reporter(logs_dir=tmp_path)
        json_path, _ = reporter.write(scored, {"universe_size": 100})
        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "tickers" in data
        assert data["tickers"][0]["ticker"] == "SPY"

    def test_reporter_writes_markdown(self, tmp_path):
        scored   = [self._make_scored()]
        reporter = Reporter(logs_dir=tmp_path)
        _, md_path = reporter.write(scored, {"universe_size": 100})
        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")
        assert "SPY" in content
        assert "LONG" in content

    def test_reporter_creates_missing_directory(self, tmp_path):
        new_dir  = tmp_path / "nested" / "scanner"
        reporter = Reporter(logs_dir=new_dir)
        reporter.write([self._make_scored()], {})
        assert any(new_dir.iterdir())

    def test_reporter_send_alert_calls_alerts_send(self):
        scored = [self._make_scored()]
        with patch("core.alerts.send") as mock_send:
            reporter = Reporter()
            reporter.send_alert(scored, {"universe_size": 50})
            mock_send.assert_called_once()
            args = mock_send.call_args
            assert args[0][0] == "scanner_briefing"


# ---------------------------------------------------------------------------
# 6. get_suggested_strategy
# ---------------------------------------------------------------------------

class TestGetSuggestedStrategy:

    def test_long_low_iv_returns_buy_equity(self):
        assert get_suggested_strategy("LONG", iv_rank=20.0) == "BUY_EQUITY"

    def test_long_high_iv_returns_cash_secured_put(self):
        assert get_suggested_strategy("LONG", iv_rank=70.0) == "CASH_SECURED_PUT"

    def test_short_high_iv_returns_covered_call(self):
        assert get_suggested_strategy("SHORT", iv_rank=60.0) == "COVERED_CALL"

    def test_short_low_iv_returns_bear_spread(self):
        assert get_suggested_strategy("SHORT", iv_rank=30.0) == "BEAR_SPREAD"

    def test_neutral_very_high_iv_returns_iron_condor(self):
        assert get_suggested_strategy("NEUTRAL", iv_rank=75.0) == "IRON_CONDOR"

    def test_none_iv_rank_handled(self):
        # With bull regime (3) and no IV, falls back to regime-only strategy
        result = get_suggested_strategy("LONG", iv_rank=None, current_regime=3)
        assert result == "LONG_EQUITY"

    def test_strategy_none_iv_long_bull(self):
        assert get_suggested_strategy("LONG", iv_rank=None, current_regime=3) == "LONG_EQUITY"

    def test_strategy_none_iv_short_crash(self):
        assert get_suggested_strategy("SHORT", iv_rank=None, current_regime=0) == "PUT_DEBIT_SPREAD"

    def test_strategy_none_iv_long_crash(self):
        assert get_suggested_strategy("LONG", iv_rank=None, current_regime=0) == "AVOID"

    def test_strategy_real_iv_uses_existing_logic(self):
        # regime parameter ignored when iv_rank is provided
        assert get_suggested_strategy("LONG", iv_rank=45.0, current_regime=3) == "BUY_EQUITY"
        assert get_suggested_strategy("LONG", iv_rank=55.0, current_regime=3) == "CASH_SECURED_PUT"

    def test_strategy_none_iv_euphoria_short(self):
        assert get_suggested_strategy("SHORT", iv_rank=None, current_regime=4) == "COVERED_CALL"


# ---------------------------------------------------------------------------
# 7. Score distribution
# ---------------------------------------------------------------------------

class TestScoreDistribution:

    def _make_results_spread(self) -> list[TickerResult]:
        """Return 10 TickerResults spanning all five score buckets."""
        # regimes chosen to spread long/short scores across 0-100
        regime_iv = [
            (3, 10.0),   # bull  + low IV  → high long score
            (3, 20.0),
            (3, 40.0),
            (2, 50.0),   # neutral         → mid scores
            (2, 50.0),
            (2, 50.0),
            (1, 60.0),   # bear            → high short score
            (0, 70.0),   # crash           → high short score
            (0, 80.0),
            (0, 90.0),
        ]
        return [
            _ticker_result(
                ticker=f"T{i}",
                regime=reg,
                iv_rank=iv,
                converged=True,
                duration=15,
            )
            for i, (reg, iv) in enumerate(regime_iv)
        ]

    def test_score_distribution_all_buckets_present(self):
        """Distribution contains all five buckets for both directions."""
        results = self._make_results_spread()
        scorer  = Scorer(threshold=60)
        scorer.score(results)
        dist = scorer.last_distribution

        buckets = ["0-20", "20-40", "40-60", "60-80", "80-100"]
        for direction in ("long", "short"):
            d = dist[direction]
            for b in buckets:
                assert b in d["buckets"], f"{direction}: bucket '{b}' missing"

    def test_score_distribution_counts_sum_to_total(self):
        """Bucket counts sum to total tickers scored."""
        results = self._make_results_spread()
        scorer  = Scorer(threshold=60)
        scorer.score(results)
        dist = scorer.last_distribution

        for direction in ("long", "short"):
            d     = dist[direction]
            total = d["total"]
            bsum  = sum(d["buckets"].values())
            assert bsum == total, f"{direction}: bucket sum {bsum} != total {total}"

    def test_score_distribution_in_json_output(self, tmp_path):
        """score_distribution key is present in the written JSON."""
        results = self._make_results_spread()
        scorer  = Scorer(threshold=60)
        scored  = scorer.score(results)

        reporter = Reporter(logs_dir=tmp_path)
        json_path, _ = reporter.write(scored, {}, scorer.last_distribution)

        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "score_distribution" in data
        assert "long"  in data["score_distribution"]
        assert "short" in data["score_distribution"]

    def test_score_distribution_in_markdown_output(self, tmp_path):
        """Markdown file contains the distribution section header and all buckets."""
        results = self._make_results_spread()
        scorer  = Scorer(threshold=60)
        scored  = scorer.score(results)

        reporter  = Reporter(logs_dir=tmp_path)
        _, md_path = reporter.write(scored, {}, scorer.last_distribution)
        content   = md_path.read_text(encoding="utf-8")

        assert "SCORE DISTRIBUTION" in content
        for bucket in ("0-20", "20-40", "40-60", "60-80", "80-100"):
            assert bucket in content

    def test_build_score_distribution_empty_input(self):
        """build_score_distribution handles empty list without error."""
        dist = build_score_distribution([], threshold=60)
        assert dist["long"]["total"]  == 0
        assert dist["short"]["total"] == 0
        assert set(dist["long"]["buckets"].keys()) == {"0-20", "20-40", "40-60", "60-80", "80-100"}


# ---------------------------------------------------------------------------
# 10. Paper validation banner
# ---------------------------------------------------------------------------

class TestPaperValidationBanner:

    def _make_scored(self) -> ScoredTicker:
        return ScoredTicker(
            ticker="SPY", current_regime=3, regime_name="bull",
            long_score=80.0, short_score=20.0, direction="LONG",
            suggested_strategy="BUY_EQUITY", iv_rank=35.0,
            spread=0.10, low_liquidity_options=False, iv_data_available=True,
            regime_duration_bars=12, bic_score=4500.0, converged=True,
        )

    def test_paper_validation_banner_present_within_30_days(self, tmp_path):
        """Markdown and alert contain the warning banner when < 30 days elapsed."""
        import datetime
        deploy_file = tmp_path / "deployment_date.txt"
        deploy_file.write_text(datetime.date.today().isoformat(), encoding="utf-8")

        reporter  = Reporter(logs_dir=tmp_path)
        _, md_path = reporter.write([self._make_scored()], {})
        content   = md_path.read_text(encoding="utf-8")
        assert "PAPER VALIDATION PERIOD" in content

        with patch("core.alerts.send") as mock_send:
            reporter.send_alert([self._make_scored()], {})
            alert_msg = mock_send.call_args[0][1]
        assert "PAPER VALIDATION PERIOD" in alert_msg

    def test_paper_validation_banner_absent_after_30_days(self, tmp_path):
        """Banner is not present when 30+ days have elapsed since deployment."""
        import datetime
        old_date   = (datetime.date.today() - datetime.timedelta(days=31)).isoformat()
        deploy_file = tmp_path / "deployment_date.txt"
        deploy_file.write_text(old_date, encoding="utf-8")

        reporter  = Reporter(logs_dir=tmp_path)
        _, md_path = reporter.write([self._make_scored()], {})
        content   = md_path.read_text(encoding="utf-8")
        assert "PAPER VALIDATION PERIOD" not in content

    def test_deployment_date_written_on_first_run(self, tmp_path):
        """deployment_date.txt is created on the first write() call."""
        deploy_file = tmp_path / "deployment_date.txt"
        assert not deploy_file.exists()

        reporter = Reporter(logs_dir=tmp_path)
        reporter.write([self._make_scored()], {})
        assert deploy_file.exists()
        assert deploy_file.read_text(encoding="utf-8").strip() != ""

    def test_exclusions_breakdown_all_reasons_present(self, tmp_path):
        """Markdown exclusions section contains all five reason categories."""
        import datetime
        deploy_file = tmp_path / "deployment_date.txt"
        deploy_file.write_text(
            (datetime.date.today() - datetime.timedelta(days=31)).isoformat(),
            encoding="utf-8",
        )

        excl = {
            "low_volume":            5,
            "low_price":             2,
            "fit_failed":            1,
            "rate_limit_exhausted":  1,
            "low_liquidity_options": 4,
        }
        reporter  = Reporter(logs_dir=tmp_path)
        _, md_path = reporter.write([self._make_scored()], {}, exclusion_counts=excl)
        content   = md_path.read_text(encoding="utf-8")

        expected_labels = [
            "Low volume",
            "Price below",
            "HMM fit failure",
            "Rate limit exhausted",
            "Low liquidity",
        ]
        for label in expected_labels:
            assert label in content, f"Exclusions breakdown missing: {label}"


# ---------------------------------------------------------------------------
# 8. Rate-limit resilience in BatchTrainer
# ---------------------------------------------------------------------------

class TestBatchTrainerRateLimits:

    def test_batch_trainer_respects_max_workers(self):
        """ThreadPoolExecutor is initialised with the configured max_workers."""
        captured = {}

        original_init = ThreadPoolExecutor.__init__

        def recording_init(self_ex, *args, **kwargs):
            captured["max_workers"] = kwargs.get("max_workers", args[0] if args else None)
            original_init(self_ex, *args, **kwargs)

        with patch(
            "core.scanner.batch_trainer.ThreadPoolExecutor.__init__",
            recording_init,
        ):
            trainer = BatchTrainer(max_workers=5, train_bars=252)
            ohlcv   = {"SPY": _make_ohlcv(300)}
            trainer.run(["SPY"], ohlcv)

        assert captured.get("max_workers") == 5

    def test_batch_trainer_retries_on_rate_limit(self):
        """Ticker that raises RateLimitError on first attempt succeeds on retry."""
        from broker.alpaca_client import RateLimitError

        call_counts: dict[str, int] = {}
        original_train = BatchTrainer._train_one

        def flaky_train(self, ticker, df):
            call_counts[ticker] = call_counts.get(ticker, 0) + 1
            if call_counts[ticker] == 1:
                raise RateLimitError("429 simulated")
            return original_train(self, ticker, df)

        with patch.object(BatchTrainer, "_train_one", flaky_train):
            trainer = BatchTrainer(max_workers=1, max_retries=3, batch_sleep=0)
            results = trainer.run(["SPY"], {"SPY": _make_ohlcv(300)})

        assert len(results) == 1
        assert results[0].fit_failed is False, "ticker should succeed after one retry"
        assert trainer.total_retries >= 1

    def test_batch_trainer_excludes_after_max_retries(self):
        """Ticker that always raises RateLimitError is excluded with rate_limit_exhausted."""
        from broker.alpaca_client import RateLimitError

        def always_429(self, ticker, df):
            raise RateLimitError("429 always")

        with patch.object(BatchTrainer, "_train_one", always_429):
            trainer = BatchTrainer(max_workers=1, max_retries=3, batch_sleep=0)
            results = trainer.run(["DEAD"], {"DEAD": _make_ohlcv(300)})

        assert len(results) == 1
        assert results[0].fit_failed is True
        assert results[0].error_message == "rate_limit_exhausted"


# ---------------------------------------------------------------------------
# 11. IEX feed enforcement + SSL error handling
# ---------------------------------------------------------------------------

class TestIEXFeedAndSSLHandling:

    def test_universe_filter_uses_iex_feed(self):
        """_filter_volume_price must pass feed='iex' in every StockBarsRequest."""
        mock_client = MagicMock()
        mock_client._stocks.get_stock_bars.return_value = {}

        mgr = UniverseManager(client=mock_client)
        mgr._filter_volume_price(["SPY"], min_volume=0.0, min_price=0.0)

        mock_client._stocks.get_stock_bars.assert_called_once()
        request = mock_client._stocks.get_stock_bars.call_args[0][0]
        feed = request.feed
        assert feed == "iex" or (hasattr(feed, "value") and feed.value == "iex"), (
            f"Expected feed='iex' in StockBarsRequest, got {feed!r}"
        )

    def test_batch_trainer_uses_iex_feed(self):
        """fetch_ohlcv() in run_scanner must pass feed='iex' in StockBarsRequest."""
        from scripts.run_scanner import fetch_ohlcv

        mock_client = MagicMock()
        mock_client._stocks.get_stock_bars.return_value = {}

        fetch_ohlcv(mock_client, ["SPY"], train_bars=10)

        mock_client._stocks.get_stock_bars.assert_called_once()
        request = mock_client._stocks.get_stock_bars.call_args[0][0]
        feed = request.feed
        assert feed == "iex" or (hasattr(feed, "value") and feed.value == "iex"), (
            f"Expected feed='iex' in StockBarsRequest, got {feed!r}"
        )

    def test_scanner_ssl_error_exits_cleanly(self, caplog):
        """ssl.SSLError during pipeline produces exit code 1 and a clean log message."""
        import ssl
        import logging
        from scripts.run_scanner import main

        with (
            patch("broker.alpaca_client.AlpacaClient"),
            patch("scripts.run_scanner.UniverseManager") as mock_um,
            caplog.at_level(logging.ERROR, logger="scanner"),
        ):
            mock_um.return_value.get_tradeable.side_effect = ssl.SSLError(
                "certificate verify failed"
            )
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
        assert any(
            "[Scanner] SSL/network error" in r.message for r in caplog.records
        ), f"Expected SSL error log, got: {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# 11. IV rank ceiling guard
# ---------------------------------------------------------------------------

class TestHighIVEventRisk:

    def _result(self, ticker="AAPL", iv_rank=50.0, regime=3) -> TickerResult:
        return TickerResult(
            ticker=ticker, current_regime=regime, regime_duration_bars=10,
            bic_score=5000.0, converged=True, convergence_warning=False,
            n_states=3, iv_rank=iv_rank,
        )

    def test_high_iv_flag_set_above_threshold(self):
        """OptionsEnricher sets high_iv_event_risk=True when iv_rank > SCANNER_MAX_IV_RANK."""
        from config.settings import SCANNER_MAX_IV_RANK
        client = MagicMock()
        client.get_iv_rank.return_value = SCANNER_MAX_IV_RANK + 5.0
        client.get_option_chain.return_value = []

        result = self._result(iv_rank=None)
        OptionsEnricher(client=client, max_workers=1).enrich([result])
        assert result.high_iv_event_risk is True

    def test_high_iv_flag_not_set_below_threshold(self):
        """high_iv_event_risk=False when iv_rank <= SCANNER_MAX_IV_RANK."""
        from config.settings import SCANNER_MAX_IV_RANK
        client = MagicMock()
        client.get_iv_rank.return_value = SCANNER_MAX_IV_RANK - 5.0
        client.get_option_chain.return_value = []

        result = self._result(iv_rank=None)
        OptionsEnricher(client=client, max_workers=1).enrich([result])
        assert result.high_iv_event_risk is False

    def test_high_iv_flag_boundary_strictly_greater_than(self):
        """high_iv_event_risk=False when iv_rank == SCANNER_MAX_IV_RANK (strictly >)."""
        from config.settings import SCANNER_MAX_IV_RANK
        client = MagicMock()
        client.get_iv_rank.return_value = float(SCANNER_MAX_IV_RANK)
        client.get_option_chain.return_value = []

        result = self._result(iv_rank=None)
        OptionsEnricher(client=client, max_workers=1).enrich([result])
        assert result.high_iv_event_risk is False

    def test_high_iv_excluded_from_candidates(self):
        """Ticker with high_iv_event_risk=True never appears in scored output."""
        r = self._result()
        r.high_iv_event_risk = True
        scored = Scorer(threshold=0).score([r])
        tickers = [s.ticker for s in scored]
        assert r.ticker not in tickers

    def test_high_iv_exclusion_reason_logged(self, caplog):
        """Exclusion log contains the high_iv_event_risk label and the IV rank value."""
        import logging
        r = self._result(iv_rank=85.0)
        r.high_iv_event_risk = True
        with caplog.at_level(logging.INFO, logger="core.scanner.scorer"):
            Scorer(threshold=0).score([r])
        assert "high_iv_event_risk" in caplog.text
        assert "85" in caplog.text

    def test_high_iv_appears_in_exclusions_breakdown(self, tmp_path):
        """Markdown exclusions section includes 'High IV event risk' row."""
        import datetime
        deploy_file = tmp_path / "deployment_date.txt"
        deploy_file.write_text(
            (datetime.date.today() - datetime.timedelta(days=31)).isoformat(),
            encoding="utf-8",
        )
        excl = {"high_iv_event_risk": 3}
        reporter = Reporter(logs_dir=tmp_path)
        _, md_path = reporter.write([], {}, exclusion_counts=excl)
        assert "High IV event risk" in md_path.read_text(encoding="utf-8")

    def test_scanner_max_iv_rank_configurable(self):
        """Changing SCANNER_MAX_IV_RANK changes the exclusion threshold in the enricher."""
        from unittest.mock import patch as _patch
        client = MagicMock()
        client.get_iv_rank.return_value = 65.0  # between 60 and 70
        client.get_option_chain.return_value = []

        result_strict = self._result(iv_rank=None)
        with _patch("core.scanner.options_enricher.SCANNER_MAX_IV_RANK", 60):
            OptionsEnricher(client=client, max_workers=1).enrich([result_strict])
        assert result_strict.high_iv_event_risk is True  # 65 > 60

        result_loose = self._result(iv_rank=None)
        with _patch("core.scanner.options_enricher.SCANNER_MAX_IV_RANK", 70):
            OptionsEnricher(client=client, max_workers=1).enrich([result_loose])
        assert result_loose.high_iv_event_risk is False  # 65 <= 70


# ---------------------------------------------------------------------------
# IV data availability — None return, weight redistribution, flag propagation
# ---------------------------------------------------------------------------

class TestIVDataAvailability:

    def _result(self, iv_rank=None, regime=3) -> TickerResult:
        return TickerResult(
            ticker="AAPL", current_regime=regime, regime_duration_bars=10,
            bic_score=5000.0, converged=True, convergence_warning=False,
            n_states=3, iv_rank=iv_rank,
        )

    def test_iv_rank_returns_none_on_failure(self):
        """get_iv_rank returns None (not 50.0) when options chain fetch fails."""
        client = MagicMock()
        client.get_iv_rank.return_value = None

        result = self._result()
        OptionsEnricher(client=client, max_workers=1).enrich([result])
        assert result.iv_rank is None

    def test_scorer_redistributes_weight_on_none_iv(self):
        """Score with iv_rank=None uses redistributed weights and produces a valid float."""
        scorer = Scorer(threshold=0)
        r = self._result(iv_rank=None, regime=3)  # bull regime
        scored = scorer.score([r])
        assert len(scored) == 1
        s = scored[0]
        # Redistributed: regime 47.5%, confirm 27.5%, duration 15%, quality 10%
        # bull converged: regime_comp=90, confirm_comp=100, dur=50 (10/20), quality=80
        expected = 0.475 * 90 + 0.275 * 100 + 0.15 * 50 + 0.10 * 80
        assert abs(s.long_score - round(expected, 1)) < 0.2

    def test_scorer_normal_weight_with_real_iv(self):
        """Score with iv_rank=45 uses standard 15% IV weight."""
        scorer = Scorer(threshold=0)
        r = self._result(iv_rank=45.0, regime=3)
        scored = scorer.score([r])
        assert len(scored) == 1
        s = scored[0]
        # Standard weights: regime 40%, confirm 20%, dur 15%, iv 15%, quality 10%
        # bull converged LONG: regime=90, confirm=100, dur=50, iv=100-45=55, quality=80
        expected = 0.40 * 90 + 0.20 * 100 + 0.15 * 50 + 0.15 * 55 + 0.10 * 80
        assert abs(s.long_score - round(expected, 1)) < 0.2

    def test_iv_data_available_flag_set_correctly(self):
        """iv_data_available=True when iv_rank is set, False when None."""
        scorer = Scorer(threshold=0)

        r_with_iv = self._result(iv_rank=40.0)
        r_no_iv   = self._result(iv_rank=None)

        scored = scorer.score([r_with_iv, r_no_iv])
        by_ticker = {s.ticker + str(s.iv_rank): s for s in scored}

        s_with = next(s for s in scored if s.iv_rank is not None)
        s_none = next(s for s in scored if s.iv_rank is None)

        assert s_with.iv_data_available is True
        assert s_none.iv_data_available is False

    def test_iv_data_unavailable_appears_in_exclusions_breakdown(self, tmp_path):
        """Markdown exclusions section includes the IV data unavailable informational row."""
        import datetime
        deploy_file = tmp_path / "deployment_date.txt"
        deploy_file.write_text(
            (datetime.date.today() - datetime.timedelta(days=31)).isoformat(),
            encoding="utf-8",
        )
        excl = {"iv_data_unavailable": 12}
        reporter = Reporter(logs_dir=tmp_path)
        _, md_path = reporter.write([], {}, exclusion_counts=excl)
        assert "IV data unavailable" in md_path.read_text(encoding="utf-8")

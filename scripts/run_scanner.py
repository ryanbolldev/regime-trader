"""
scripts/run_scanner.py
-----------------------
Nightly HMM scanner pipeline — entry point for `make scan` and the Docker
cron service.

Pipeline:
  1. UniverseManager  — filter SP500+Nasdaq100 by volume, price, earnings
  2. Fetch OHLCV      — daily bars for each ticker via AlpacaClient
  3. BatchTrainer     — parallel HMM fit + regime classification
  4. OptionsEnricher  — attach IV rank + ATM spread (low-liquidity flag)
  5. Scorer           — composite LONG/SHORT scoring
  6. Reporter         — write JSON + Markdown; fire Telegram alert

Usage:
  python scripts/run_scanner.py
  make scan
"""

from __future__ import annotations

import datetime
import logging
import ssl
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when run directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config.settings import SCANNER_DATA_FEED, SCANNER_MAX_WORKERS, SCANNER_TRAIN_BARS
from core.scanner.batch_trainer import BatchTrainer
from core.scanner.options_enricher import OptionsEnricher
from core.scanner.reporter import Reporter
from core.scanner.scorer import Scorer
from core.scanner.universe import UniverseManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scanner")


def fetch_ohlcv(client, tickers: list[str], train_bars: int) -> dict[str, pd.DataFrame]:
    """Fetch daily bars for all tickers; return {ticker: DataFrame}."""
    end   = datetime.datetime.now(datetime.timezone.utc)
    # Fetch ~1.5x bars to account for weekends/holidays
    start = end - datetime.timedelta(days=int(train_bars * 1.6))

    ohlcv_map: dict[str, pd.DataFrame] = {}
    log.info("Fetching OHLCV for %d tickers (%d bars)…", len(tickers), train_bars)

    # Batch fetch in groups of 50 to stay within URL-length limits
    chunk_size = 50
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        try:
            resp = client._stocks.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols=chunk,
                    timeframe=TimeFrame.Day,
                    start=start,
                    end=end,
                    feed=SCANNER_DATA_FEED,
                )
            )
            for ticker in chunk:
                try:
                    bars = list(resp[ticker])
                except KeyError:
                    bars = []
                if len(bars) < 30:
                    log.debug("OHLCV: insufficient bars for %s (%d)", ticker, len(bars))
                    continue
                df = pd.DataFrame(
                    {
                        "open":   [b.open   for b in bars],
                        "high":   [b.high   for b in bars],
                        "low":    [b.low    for b in bars],
                        "close":  [b.close  for b in bars],
                        "volume": [b.volume for b in bars],
                    },
                    index=pd.DatetimeIndex([b.timestamp for b in bars]),
                )
                ohlcv_map[ticker] = df
        except Exception as exc:
            log.warning("OHLCV batch fetch failed for chunk %s: %s", chunk[:3], exc)

    log.info("OHLCV fetched for %d / %d tickers", len(ohlcv_map), len(tickers))
    return ohlcv_map


def main() -> None:
    t0 = time.monotonic()
    log.info("=== Regime Trader Nightly Scanner ===")
    log.info("Date: %s", datetime.date.today().isoformat())

    # ── 1. Initialise broker client ────────────────────────────────────
    try:
        from broker.alpaca_client import AlpacaClient
        client = AlpacaClient()
    except Exception as exc:
        log.critical("AlpacaClient init failed: %s", exc)
        sys.exit(1)

    try:
        # ── 2. Universe filter ─────────────────────────────────────────────
        universe_mgr = UniverseManager(client=client)
        tickers      = universe_mgr.get_tradeable()
        log.info("Universe after filters: %d tickers", len(tickers))

        # ── 3. Fetch OHLCV data ────────────────────────────────────────────
        ohlcv_map = fetch_ohlcv(client, tickers, train_bars=SCANNER_TRAIN_BARS)
        if not ohlcv_map:
            log.critical("No OHLCV data retrieved — aborting.")
            sys.exit(1)

        # ── 4. Batch HMM training ──────────────────────────────────────────
        trainer = BatchTrainer(max_workers=SCANNER_MAX_WORKERS, train_bars=SCANNER_TRAIN_BARS)
        results = trainer.run(list(ohlcv_map.keys()), ohlcv_map)

    except (ssl.SSLError, ConnectionResetError) as exc:
        log.error(
            "[Scanner] SSL/network error — scanner must be run inside Docker on the VPS, "
            "not locally on Windows: %s", exc
        )
        sys.exit(1)

    # ── 5. Options enrichment ──────────────────────────────────────────
    enricher = OptionsEnricher(client=client, max_workers=SCANNER_MAX_WORKERS)
    results  = enricher.enrich(results)

    # ── 6. Composite scoring ───────────────────────────────────────────
    scorer = Scorer()
    scored = scorer.score(results)

    runtime = round(time.monotonic() - t0, 1)
    metadata = {
        "universe_size": len(tickers),
        "trained":       len([r for r in results if not r.fit_failed]),
        "qualified":     len(scored),
        "runtime_secs":  runtime,
        "date":          datetime.date.today().isoformat(),
    }

    # ── 7. Aggregate exclusion counts ─────────────────────────────────
    exclusion_counts: dict[str, int] = dict(universe_mgr.exclusion_counts)
    for r in results:
        if r.fit_failed:
            key = "rate_limit_exhausted" if r.error_message == "rate_limit_exhausted" else "fit_failed"
            exclusion_counts[key] = exclusion_counts.get(key, 0) + 1
    exclusion_counts["low_liquidity_options"] = sum(
        1 for r in results if r.low_liquidity_options
    )
    exclusion_counts["high_iv_event_risk"] = sum(
        1 for r in results if r.high_iv_event_risk
    )
    exclusion_counts["iv_data_unavailable"] = sum(
        1 for r in results if r.iv_rank is None
    )

    # ── 8. Report ──────────────────────────────────────────────────────
    reporter = Reporter()
    json_path, md_path = reporter.write(
        scored, metadata, scorer.last_distribution, exclusion_counts
    )
    reporter.send_alert(scored, metadata)

    # ── Stdout summary ─────────────────────────────────────────────────
    longs  = [s for s in scored if s.direction == "LONG"]
    shorts = [s for s in scored if s.direction == "SHORT"]
    print()
    print("=" * 60)
    print(f"SCANNER COMPLETE  [{runtime}s]")
    retries = trainer.total_retries
    print(
        f"Universe: {len(tickers)} | Trained: {metadata['trained']} | "
        f"Qualified: {len(scored)} | Rate-limit retries: {retries}"
    )
    print(f"LONG: {len(longs)}  SHORT: {len(shorts)}")
    print()

    if longs:
        print("-- TOP LONG CANDIDATES --")
        for s in longs[:10]:
            print(
                f"  {s.ticker:<6} regime={s.regime_name:<8} score={s.long_score:>5.0f}"
                f"  iv={_fmt(s.iv_rank)}  dur={s.regime_duration_bars:>3}d"
                f"  [{s.suggested_strategy}]"
            )
    if shorts:
        print("-- TOP SHORT CANDIDATES --")
        for s in shorts[:10]:
            print(
                f"  {s.ticker:<6} regime={s.regime_name:<8} score={s.short_score:>5.0f}"
                f"  iv={_fmt(s.iv_rank)}  dur={s.regime_duration_bars:>3}d"
                f"  [{s.suggested_strategy}]"
            )

    print()
    print(f"JSON:     {json_path}")
    print(f"Markdown: {md_path}")
    print("=" * 60)


def _fmt(v: float | None) -> str:
    return f"{v:>5.0f}" if v is not None else "  N/A"


if __name__ == "__main__":
    main()

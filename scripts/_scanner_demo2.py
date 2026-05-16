"""Synthetic scanner demo — all three hardening improvements visible."""
import sys, pathlib, datetime
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from core.scanner.batch_trainer import BatchTrainer
from core.scanner.options_enricher import OptionsEnricher
from core.scanner.scorer import Scorer
from core.scanner.reporter import Reporter


def make_ohlcv(n=300, seed=0, drift=0.0003):
    rng = np.random.default_rng(seed)
    lr  = rng.normal(drift, 0.012, n)
    c   = 100.0 * np.exp(np.cumsum(lr))
    noise = rng.uniform(0.001, 0.015, n)
    h   = c * (1 + noise)
    lo  = c * (1 - noise)
    o   = np.clip(c * (1 + rng.normal(0, 0.005, n)), lo, h)
    v   = rng.lognormal(14.0, 0.6, n).astype(int)
    idx = pd.bdate_range(end="2024-01-01", periods=n)
    return pd.DataFrame({"open":o,"high":h,"low":lo,"close":c,"volume":v}, index=idx)


class _MockClient:
    """Stub client — no network calls; spread fetch raises so enricher skips it."""
    def get_option_chain(self, ticker):
        raise RuntimeError("no options data in synthetic demo")


def main():
    tickers = ["AAPL","MSFT","NVDA","AMZN","META","JPM","BAC","XOM","TSLA",
               "GOOGL","NFLX","PFE","KO","DIS"]
    drifts  = [0.0006,0.0005,0.0008,0.0004,0.0007,
               0.0003,0.0002,-0.0002,-0.0005,
               0.0004,0.0003,-0.0004,0.0001,-0.0003]
    ohlcv_map = {t: make_ohlcv(300, seed=i, drift=d)
                 for i, (t, d) in enumerate(zip(tickers, drifts))}

    print("Training HMM for", len(tickers), "tickers...")
    trainer = BatchTrainer(max_workers=4, train_bars=252, batch_sleep=0)
    results = trainer.run(tickers, ohlcv_map)

    # Route through OptionsEnricher so vol_estimated=True and tilde prefix shows.
    # vix_series=None simulates the paper-account VIXY fetch failure.
    enricher = OptionsEnricher(
        client=_MockClient(),
        max_workers=4,
        vix_series=None,
        ohlcv_map=ohlcv_map,
    )
    results = enricher.enrich(results)
    for r in results:
        r.low_liquidity_options = (r.spread or 0) > 0.15

    scorer = Scorer(threshold=55)
    scored = scorer.score(results)

    # Synthetic exclusion counts (simulating upstream pipeline)
    excl = {
        "earnings_within_7_days":    2,
        "earnings_data_unavailable": 1,
        "low_volume":                3,
        "low_price":                 1,
        "fit_failed":                sum(1 for r in results if r.fit_failed),
        "rate_limit_exhausted":      0,
        "low_liquidity_options":     sum(1 for r in results if r.low_liquidity_options),
        "high_iv_event_risk":        sum(1 for r in results if r.high_iv_event_risk),
    }

    logs_dir = pathlib.Path("logs/scanner_demo")
    # Write deployment_date as today so paper banner shows
    logs_dir.mkdir(parents=True, exist_ok=True)
    deploy_file = logs_dir / "deployment_date.txt"
    deploy_file.write_text(datetime.date.today().isoformat(), encoding="utf-8")

    reporter = Reporter(logs_dir=logs_dir)
    meta = {
        "universe_size": len(tickers),
        "trained":       sum(1 for r in results if not r.fit_failed),
        "qualified":     len(scored),
        "runtime_secs":  "N/A (synthetic)",
    }
    json_path, md_path = reporter.write(scored, meta, scorer.last_distribution, excl)

    print(md_path.read_text(encoding="utf-8"))
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()

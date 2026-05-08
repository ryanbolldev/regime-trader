"""Synthetic scanner demo — all three hardening improvements visible."""
import sys, pathlib, datetime
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from core.scanner.batch_trainer import BatchTrainer
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


def main():
    tickers = ["AAPL","MSFT","NVDA","AMZN","META","JPM","BAC","XOM","CVX","TSLA",
               "GOOGL","NFLX","PFE","KO","DIS"]
    drifts  = [0.0006,0.0005,0.0008,0.0004,0.0007,
               0.0003,0.0002,-0.0002,-0.0001,-0.0005,
               0.0004,0.0003,-0.0004,0.0001,-0.0003]
    ohlcv_map = {t: make_ohlcv(300, seed=i, drift=d)
                 for i, (t, d) in enumerate(zip(tickers, drifts))}

    print("Training HMM for", len(tickers), "tickers...")
    trainer = BatchTrainer(max_workers=4, train_bars=252, batch_sleep=0)
    results = trainer.run(tickers, ohlcv_map)

    iv_map = {"AAPL":35,"MSFT":28,"NVDA":55,"AMZN":42,"META":38,
              "JPM":60,"BAC":65,"XOM":70,"CVX":68,"TSLA":72,
              "GOOGL":40,"NFLX":58,"PFE":30,"KO":22,"DIS":45}
    for r in results:
        r.iv_rank = float(iv_map.get(r.ticker, 50))
        r.spread  = 0.05 if r.iv_rank < 60 else 0.18
        r.low_liquidity_options = r.spread > 0.15

    scorer = Scorer(threshold=55)
    scored = scorer.score(results)

    # Synthetic exclusion counts (simulating upstream pipeline)
    excl = {
        "earnings_within_7_days":    2,
        "earnings_data_unavailable": 1,
        "low_volume":                3,
        "low_price":                 1,
        "fit_failed":                0,
        "rate_limit_exhausted":      0,
        "low_liquidity_options":     sum(1 for r in results if r.low_liquidity_options),
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

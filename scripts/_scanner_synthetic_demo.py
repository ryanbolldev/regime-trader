"""Synthetic scanner demo — runs the full pipeline without network calls."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from core.scanner.batch_trainer import BatchTrainer
from core.scanner.scorer import Scorer
from core.scanner.reporter import Reporter

_REGIME_NAMES = {0:"crash",1:"bear",2:"neutral",3:"bull",4:"euphoria"}


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
    tickers = ["AAPL","MSFT","NVDA","AMZN","META","JPM","BAC","XOM","CVX","TSLA"]
    drifts  = [0.0006, 0.0005, 0.0008, 0.0004, 0.0007,
               0.0003, 0.0002, -0.0002, -0.0001, -0.0005]
    ohlcv_map = {t: make_ohlcv(300, seed=i, drift=d)
                 for i, (t, d) in enumerate(zip(tickers, drifts))}

    print("Training HMM for", len(tickers), "tickers...")
    trainer = BatchTrainer(max_workers=4, train_bars=252)
    results = trainer.run(tickers, ohlcv_map)

    iv_map = {"AAPL":35,"MSFT":28,"NVDA":55,"AMZN":42,"META":38,
              "JPM":60,"BAC":65,"XOM":70,"CVX":68,"TSLA":72}
    for r in results:
        r.iv_rank = float(iv_map.get(r.ticker, 50))
        r.spread  = 0.05 if r.iv_rank < 60 else 0.15

    scorer = Scorer(threshold=55)
    scored = scorer.score(results)

    logs_dir = pathlib.Path("logs/scanner")
    reporter = Reporter(logs_dir=logs_dir)
    n_trained = sum(1 for r in results if not r.fit_failed)
    meta = {
        "universe_size": len(tickers),
        "trained":       n_trained,
        "qualified":     len(scored),
        "runtime_secs":  "N/A (synthetic)",
    }
    json_path, md_path = reporter.write(scored, meta)

    longs  = [s for s in scored if s.direction == "LONG"]
    shorts = [s for s in scored if s.direction == "SHORT"]

    print()
    print("=" * 60)
    print("SCANNER COMPLETE  [synthetic data — SSL unavailable on Windows]")
    print(f"Tickers: {len(tickers)} | Trained: {n_trained} | Qualified: {len(scored)}")
    print(f"LONG: {len(longs)}  SHORT: {len(shorts)}")
    print()
    if longs:
        print("-- TOP LONG CANDIDATES --")
        for s in longs:
            iv = f"{s.iv_rank:>4.0f}" if s.iv_rank is not None else " N/A"
            print(f"  {s.ticker:<6} regime={s.regime_name:<8} score={s.long_score:>5.0f}"
                  f"  iv={iv}  dur={s.regime_duration_bars:>3}d  [{s.suggested_strategy}]")
    if shorts:
        print("-- TOP SHORT CANDIDATES --")
        for s in shorts:
            iv = f"{s.iv_rank:>4.0f}" if s.iv_rank is not None else " N/A"
            print(f"  {s.ticker:<6} regime={s.regime_name:<8} score={s.short_score:>5.0f}"
                  f"  iv={iv}  dur={s.regime_duration_bars:>3}d  [{s.suggested_strategy}]")
    print()
    print("JSON:     ", json_path)
    print("Markdown: ", md_path)
    print("=" * 60)
    print()
    print("=== MARKDOWN WATCHLIST ===")
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

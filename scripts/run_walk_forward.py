"""
scripts/run_walk_forward.py
---------------------------
Entry point for `make walkforward`.

Generates synthetic OHLCV data and runs the walk-forward backtest framework.
HMM settings are reduced for demo speed (fewer restarts / iterations).
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# Reduce HMM complexity for a fast demo run
import config.settings as _s
_s.HMM_N_INIT    = 1
_s.HMM_N_ITER    = 100
_s.HMM_MIN_STATES = 3
_s.HMM_MAX_STATES = 4

from core.walk_forward import WalkForwardBacktester


def _make_ohlcv(n: int = 900, seed: int = 42) -> pd.DataFrame:
    rng       = np.random.default_rng(seed)
    log_rets  = rng.normal(0.0003, 0.012, n)
    close     = 100.0 * np.exp(np.cumsum(log_rets))
    noise     = rng.uniform(0.001, 0.015, n)
    high      = close * (1 + noise)
    low       = close * (1 - noise)
    open_     = np.clip(close * (1 + rng.normal(0, 0.005, n)), low, high)
    volume    = rng.lognormal(14.0, 0.6, n).astype(int)
    dates     = pd.bdate_range(end="2024-01-01", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def main() -> None:
    print("Generating synthetic OHLCV data (900 bars ≈ 3.5 years daily)...")
    ohlcv = _make_ohlcv(900)

    print("Running walk-forward backtest (n_train=504, n_test=126, step=63)...")
    wf     = WalkForwardBacktester()
    result = wf.run(ohlcv, n_train=504, n_test=126, step=63, min_folds=3)

    # Locate the output files
    from pathlib import Path
    import datetime, json
    today     = datetime.date.today().isoformat()
    logs_dir  = Path("logs") / "walk_forward"
    md_path   = logs_dir / f"walk_forward_{today}.md"
    json_path = logs_dir / f"walk_forward_{today}.json"

    print()
    print("=" * 70)
    print(md_path.read_text(encoding="utf-8"))
    print(f"JSON: {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

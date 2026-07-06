"""
wheel_scanner/scanner.py
-------------------------
Main scanner class and standalone entry point.

Usage (standalone):
    python -m wheel_scanner.scanner
    python -m wheel_scanner.scanner --regime 3
    python -m wheel_scanner.scanner --regime 3 --output-dir /tmp/wheel_out

Usage (programmatic):
    from wheel_scanner import WheelScanner
    candidates = WheelScanner(regime_label=3).run()

The regime_label parameter matches the existing HMM engine output:
    0=crash  1=bear  2=neutral  3=bull  4=euphoria

When regime_label is None (default), the scanner uses 2 (neutral) with
a logged warning. To wire in the live regime, call:
    from core.hmm_engine import HMMEngine
    regime_label = engine.predict_current(features_row)
    candidates = WheelScanner(regime_label=regime_label).run()

Integration note:
    Output is written to logs/wheel_scanner/wheel_candidates_YYYY-MM-DD.json
    and .csv. The live loop can read the latest file by globbing:
    sorted(Path("logs/wheel_scanner").glob("wheel_candidates_*.json"))[-1]
"""

from __future__ import annotations

import logging
import math
import pathlib
import time
from datetime import date
from typing import Optional

log = logging.getLogger(__name__)

# Minimum composite score to include in output
_SCORE_THRESHOLD = 30.0

# Regime name map (mirrors HMM engine _REGIME_NAMES)
_REGIME_NAMES: dict[int, str] = {
    0: "crash",
    1: "bear",
    2: "neutral",
    3: "bull",
    4: "euphoria",
}


class WheelScanner:
    """Wheel strategy candidate scanner.

    Parameters
    ----------
    regime_label : int | None
        Current HMM market regime (0–4). Defaults to 2 (neutral) when None.
    output_dir : Path | None
        Override the default logs/wheel_scanner/ output directory.
    score_threshold : float
        Minimum composite score to include in results. Default 30.0.
    """

    def __init__(
        self,
        regime_label:    Optional[int] = None,
        output_dir:      Optional[pathlib.Path] = None,
        score_threshold: float = _SCORE_THRESHOLD,
    ) -> None:
        if regime_label is None:
            log.warning(
                "WheelScanner: no regime_label supplied — defaulting to 2 (neutral). "
                "Pass the HMM engine's predict_current() output for accurate regime scoring."
            )
            regime_label = 2

        if regime_label not in _REGIME_NAMES:
            raise ValueError(
                f"regime_label must be 0–4, got {regime_label!r}. "
                f"Values: {_REGIME_NAMES}"
            )

        self.regime_label    = regime_label
        self.regime_name     = _REGIME_NAMES[regime_label]
        self.output_dir      = output_dir
        self.score_threshold = score_threshold

        self._options_client = None
        self._stock_client   = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list["WheelCandidate"]:
        """Execute the full scan pipeline and return ranked candidates.

        Pipeline:
          1. Build universe (S&P 400 from Wikipedia)
          2. Filter: price, ADV, market cap, security type, earnings, options
          3. Score each surviving candidate
          4. Rank by composite score, filter by threshold
          5. Write output files
        """
        from wheel_scanner.filters import (
            fetch_sp400_universe,
            filter_price_and_volume,
            filter_market_cap,
            filter_security_type,
            filter_options_available,
        )
        from wheel_scanner.earnings import batch_earnings_filter
        from wheel_scanner.output import write_results, print_summary

        t_start = time.monotonic()

        self._options_client = self._build_options_client()
        self._stock_client   = self._build_stock_client()

        log.info(
            "WheelScanner: starting scan (regime=%s [%d], threshold=%.0f)",
            self.regime_name, self.regime_label, self.score_threshold,
        )

        # ── Stage 1: Universe ───────────────────────────────────────────
        universe = fetch_sp400_universe()
        log.info("Stage 1 — universe: %d tickers", len(universe))
        excl_counts: dict[str, int] = {}

        # ── Stage 2: Price / ADV ────────────────────────────────────────
        universe, counts = filter_price_and_volume(universe, self._stock_client)
        _merge(excl_counts, counts)
        log.info("Stage 2 — price/ADV: %d tickers remaining", len(universe))

        # ── Stage 3: Market cap (yfinance; skipped if not installed) ────
        universe, counts = filter_market_cap(universe)
        _merge(excl_counts, counts)
        log.info("Stage 3 — market cap: %d tickers remaining", len(universe))

        # ── Stage 4: Security type (yfinance; skipped if not installed) ─
        universe, counts = filter_security_type(universe)
        _merge(excl_counts, counts)
        log.info("Stage 4 — security type: %d tickers remaining", len(universe))

        # ── Stage 5: Earnings proximity (yfinance; skipped if absent) ───
        universe, excluded_earnings = batch_earnings_filter(universe, exclusion_days=14)
        excl_counts["earnings_within_14d"] = len(excluded_earnings)
        log.info("Stage 5 — earnings: %d tickers remaining", len(universe))

        # ── Stage 6: Options availability ───────────────────────────────
        universe, counts = filter_options_available(universe, self._options_client)
        _merge(excl_counts, counts)
        log.info("Stage 6 — options availability: %d tickers remaining", len(universe))

        # ── Stage 7: Score each candidate ───────────────────────────────
        log.info("Stage 7 — scoring %d candidates…", len(universe))
        candidates: list[WheelCandidate] = []
        for ticker in universe:
            candidate = self._score_ticker(ticker)
            if candidate is not None:
                candidates.append(candidate)

        # ── Rank and threshold ──────────────────────────────────────────
        candidates.sort(key=lambda c: c.composite_score, reverse=True)
        candidates = [c for c in candidates if c.composite_score >= self.score_threshold]

        runtime = time.monotonic() - t_start
        log.info(
            "WheelScanner: complete — %d candidates in %.0fs",
            len(candidates), runtime,
        )

        # ── Output ─────────────────────────────────────────────────────
        write_results(candidates, logs_dir=self.output_dir)
        print_summary(candidates)

        return candidates

    # ------------------------------------------------------------------
    # Internal scoring pipeline per ticker
    # ------------------------------------------------------------------

    def _score_ticker(self, ticker: str) -> Optional["WheelCandidate"]:
        """Score a single ticker. Returns None if data is insufficient."""
        from wheel_scanner.options_provider import fetch_put_chain, compute_ivr
        from wheel_scanner.options_data import (
            find_target_put,
            mid_price,
            annualized_yield_pct,
            bid_ask_spread_pct,
            fetch_ohlcv_for_sma,
            compute_sma_trend,
        )
        from wheel_scanner.scoring import (
            score_ivr,
            score_put_premium,
            score_regime,
            score_trend,
            score_liquidity,
            composite_score,
            compute_flags,
            trend_label,
            REGIME_NAMES,
        )
        from wheel_scanner.earnings import days_until_earnings
        from wheel_scanner.output import WheelCandidate

        log.debug("Scoring %s…", ticker)

        # ── Options chain ────────────────────────────────────────────────
        put_legs = fetch_put_chain(ticker, self._options_client, min_dte=21, max_dte=45)
        if not put_legs:
            log.debug("%s: no puts in 21-45 DTE window — skipped", ticker)
            return None

        target_put = find_target_put(put_legs, target_delta=-0.30)
        if target_put is None:
            log.debug("%s: no put with delta data — skipped", ticker)
            return None

        put_mid = mid_price(target_put)
        if put_mid is None or put_mid <= 0:
            log.debug("%s: put has no mid price — skipped", ticker)
            return None

        spread_pct   = bid_ask_spread_pct(target_put)
        ann_yield    = annualized_yield_pct(put_mid, target_put.strike, target_put.dte)

        # ── IV Rank ──────────────────────────────────────────────────────
        ivr = compute_ivr(ticker, self._options_client, self._stock_client)

        # ── Price / SMA data ─────────────────────────────────────────────
        ohlcv_df = fetch_ohlcv_for_sma(ticker, self._stock_client)
        if ohlcv_df is None:
            log.debug("%s: insufficient OHLCV for SMA — skipped", ticker)
            return None

        price, sma50, sma200, sma50_declining = compute_sma_trend(ohlcv_df)

        sma50_valid  = not math.isnan(sma50)
        sma200_valid = not math.isnan(sma200)

        # Fall back to neutral trend score when SMA data is incomplete
        if not sma50_valid:
            s_trend    = 60.0
            trend_str  = "recovering"
        elif not sma200_valid:
            # Have 50d but not 200d — treat as neutral trend
            above_50 = price > sma50
            s_trend   = 60.0 if above_50 else 20.0
            trend_str = "recovering" if above_50 else "below_50sma"
        else:
            s_trend   = score_trend(price, sma50, sma200, sma50_declining)
            trend_str = trend_label(s_trend)

        # ── Market cap (yfinance) ────────────────────────────────────────
        market_cap_B = _fetch_market_cap_b(ticker)

        # ── Earnings check ───────────────────────────────────────────────
        days_to_earnings = days_until_earnings(ticker)
        within_14d       = days_to_earnings is not None and 0 <= days_to_earnings <= 14

        # ── Component scores ─────────────────────────────────────────────
        s_ivr       = score_ivr(ivr)
        s_premium   = score_put_premium(ann_yield if ann_yield > 0 else None)
        s_regime    = score_regime(self.regime_label)
        s_liquidity = score_liquidity(
            target_put.open_interest,
            spread_pct,
            target_put.volume_today,   # today's volume as ADV proxy; see TODO in options_data.py
        )

        total = composite_score(s_ivr, s_premium, s_regime, s_trend, s_liquidity)

        # ── Flags ────────────────────────────────────────────────────────
        flags = compute_flags(
            ivr                = ivr,
            bid_ask_spread_pct = spread_pct,
            open_interest      = target_put.open_interest,
            price              = price,
            sma50              = sma50 if sma50_valid else price + 1,  # suppress flag if no data
            days_to_earnings   = days_to_earnings,
        )

        return WheelCandidate(
            ticker               = ticker,
            composite_score      = total,
            market_cap_B         = market_cap_B,
            price                = round(price, 2),
            ivr                  = ivr,
            target_put_strike    = target_put.strike,
            target_expiry        = target_put.expiration.isoformat(),
            dte                  = target_put.dte,
            put_mid              = round(put_mid, 2),
            annualized_yield_pct = round(ann_yield, 1) if ann_yield else None,
            open_interest        = target_put.open_interest,
            bid_ask_spread_pct   = round(spread_pct, 1) if spread_pct is not None else None,
            regime               = self.regime_name,
            trend_score          = trend_str,
            earnings_within_14d  = within_14d,
            flags                = flags,
            score_ivr            = round(s_ivr, 1),
            score_premium        = round(s_premium, 1),
            score_regime         = round(s_regime, 1),
            score_trend          = round(s_trend, 1),
            score_liquidity      = round(s_liquidity, 1),
        )

    # ------------------------------------------------------------------
    # Client factories
    # ------------------------------------------------------------------

    @staticmethod
    def _build_options_client():
        from wheel_scanner.options_provider import build_options_client
        return build_options_client()

    @staticmethod
    def _build_stock_client():
        from alpaca.data.historical import StockHistoricalDataClient
        from config.credentials import load_credentials
        creds = load_credentials()
        return StockHistoricalDataClient(
            api_key    = creds.api_key,
            secret_key = creds.api_secret,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge(target: dict[str, int], source: dict[str, int]) -> None:
    for k, v in source.items():
        target[k] = target.get(k, 0) + v


def _fetch_market_cap_b(ticker: str) -> Optional[float]:
    """Fetch market cap in billions via yfinance. Returns None if unavailable."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        cap  = info.get("marketCap") or info.get("market_cap")
        if cap:
            return round(cap / 1e9, 2)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt = "%H:%M:%S",
        level   = level,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Wheel strategy candidate scanner for regime_trader",
    )
    parser.add_argument(
        "--regime",
        type    = int,
        default = None,
        choices = [0, 1, 2, 3, 4],
        help    = "HMM regime label (0=crash 1=bear 2=neutral 3=bull 4=euphoria). "
                  "Defaults to 2 (neutral) when not supplied.",
    )
    parser.add_argument(
        "--output-dir",
        type    = pathlib.Path,
        default = None,
        help    = "Override default output directory (logs/wheel_scanner/)",
    )
    parser.add_argument(
        "--threshold",
        type    = float,
        default = _SCORE_THRESHOLD,
        help    = f"Minimum composite score to include in output (default {_SCORE_THRESHOLD})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action  = "store_true",
        help    = "Enable DEBUG logging",
    )
    args = parser.parse_args()

    _configure_logging(verbose=args.verbose)

    scanner = WheelScanner(
        regime_label    = args.regime,
        output_dir      = args.output_dir,
        score_threshold = args.threshold,
    )
    candidates = scanner.run()

    if not candidates:
        print(f"\nNo candidates met the score threshold of {args.threshold}.")
    else:
        print(f"\nTop candidate: {candidates[0].ticker} "
              f"(score={candidates[0].composite_score}, "
              f"regime={candidates[0].regime})")

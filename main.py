"""
main.py
--------
Orchestrator: startup sequence, main trading loop, and graceful shutdown.

Startup sequence:
  1. Load credentials via AlpacaClient constructor (fails fast on bad .env)
  2. Check for lockfile — exit if present
  3. Connect to Alpaca, verify account active
  4. Check market hours and log status
  5. Fetch 2 years of daily SPY bars; train HMM engine
  6. Initialize RiskManager with current portfolio NAV
  7. Initialize position tracker from live broker positions
  8. Enter main loop

Main loop (every BAR_INTERVAL_SECS, default 300):
  - Fetch latest bars for each ticker
  - Run feature engineering
  - Get HMM regime prediction
  - Build signal via regime_strategies.get_signal()
  - Gate through risk_manager.approve()
  - Execute approved orders via order_executor
  - Update position tracker on fill
  - Fire alerts on regime change, circuit breaker, or daily P&L summary

Shutdown (SIGINT / SIGTERM or unhandled exception):
  - Optionally cancel all open orders
  - Remove lockfile
  - Fire shutdown alert
"""

from __future__ import annotations

import logging
import os
import signal as signal_module
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from broker.alpaca_client import AlpacaClient, BrokerUnavailableError
from config import settings
from config.credentials import ConfigurationError
from core import (
    alerts,
    feature_engineering,
    market_data,
    order_executor,
    position_tracker,
    regime_strategies,
)
from core.hmm_engine import HMMEngine
from core.risk_manager import RiskManager

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOCKFILE            = Path(__file__).parent / "trading.lock"
LOG_DIR             = Path(__file__).parent / "logs"
BAR_INTERVAL_SECS   = 300   # 5-minute bars
API_RETRY_WAIT_SECS = 60    # pause after broker outage before retry
DATA_RETRY_WAIT_SECS = 30   # pause between data-feed retries
DATA_MAX_RETRIES    = 3     # max attempts before skipping a ticker

_REGIME_NAMES = {-1: "unconfirmed", 0: "crash", 1: "bear",
                 2: "neutral", 3: "bull", 4: "euphoria"}


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        root.addHandler(ch)
    log_file = LOG_DIR / f"regime_trader_{datetime.now().strftime('%Y%m%d')}.log"
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root.addHandler(fh)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class RegimeTrader:
    """Ties all components together for the live trading session."""

    def __init__(
        self,
        *,
        client: AlpacaClient,
        hmm: HMMEngine,
        risk_manager: RiskManager,
        lockfile: Path = LOCKFILE,
        bar_interval: int = BAR_INTERVAL_SECS,
        close_positions_on_shutdown: bool = False,
    ) -> None:
        self._client       = client
        self._hmm          = hmm
        self._risk         = risk_manager
        self._lockfile     = lockfile
        self._bar_interval = bar_interval
        self._close_on_shutdown = close_positions_on_shutdown

        self._running:          bool = False
        self._shutdown_reason:  str  = ""
        self._current_regime:   int  = -1
        self._market_was_open:  bool = False

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Execute the 7-step startup sequence."""
        log.info("=== Regime Trader starting up ===")

        # Step 2: lockfile guard
        if self._lockfile.exists():
            msg = (
                f"Lockfile already exists at {self._lockfile}. "
                "Another instance may be running. Delete it to restart."
            )
            log.error(msg)
            try:
                alerts.send("lockfile_present", msg, "critical")
            except Exception:
                pass
            raise SystemExit(1)

        self._lockfile.write_text(
            f"pid={os.getpid()} "
            f"started={datetime.now(tz=timezone.utc).isoformat()}\n"
        )
        log.info("Lockfile written: %s", self._lockfile)
        alerts.send("lockfile_written", f"Lockfile created: {self._lockfile}", "info")

        # Step 3: verify account
        acct = self._client.get_account()
        if str(acct.status).upper() not in {"ACTIVE", "APPROVED"}:
            raise RuntimeError(
                f"Alpaca account not tradeable: status={acct.status!r}"
            )
        log.info(
            "Account verified: id=%s status=%s NAV=$%.2f",
            acct.account_id, acct.status, acct.portfolio_value,
        )

        # Step 4: market hours (informational; continue either way)
        is_open = self._client.is_market_open()
        self._market_was_open = is_open
        log.info("Market is currently: %s", "OPEN" if is_open else "CLOSED")

        # Step 5: train HMM on 2 years of daily SPY data
        log.info("Fetching 2 years of daily SPY bars for HMM training…")
        end   = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=730)
        ohlcv    = market_data.get_historical_bars("SPY", start, end, "1Day")
        features = feature_engineering.compute(ohlcv)
        self._hmm.fit(features)
        log.info("HMM trained successfully")

        # Step 6: initialize risk manager
        nav = float(acct.portfolio_value)
        self._risk.initialize(nav)
        log.info("RiskManager initialised: NAV=$%.2f", nav)

        # Step 7: reconcile position tracker
        positions = self._client.get_positions()
        log.info("Broker reports %d open position(s)", len(positions))
        try:
            open_positions = position_tracker.get_open_positions()
            log.info(
                "Position tracker reconciled: %d position(s) tracked",
                len(open_positions),
            )
        except Exception as exc:
            log.warning("Position tracker reconciliation skipped: %s", exc)

        alerts.send(
            "startup",
            f"Regime Trader started. NAV=${nav:,.2f}  "
            f"Market={'OPEN' if is_open else 'CLOSED'}",
            "info",
        )
        log.info("Startup complete — entering main loop")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Block until shutdown is requested, processing one bar per interval."""
        self._running = True
        log.info(
            "Main loop running (tickers=%s  interval=%ds)",
            settings.TICKERS, self._bar_interval,
        )

        while self._running:
            try:
                self._run_bar()
            except BrokerUnavailableError as exc:
                log.error(
                    "Broker unavailable: %s — waiting %ds before retry",
                    exc, API_RETRY_WAIT_SECS,
                )
                alerts.send("api_outage", f"Broker unreachable: {exc}", "warning")
                time.sleep(API_RETRY_WAIT_SECS)
                continue
            except Exception as exc:
                log.exception("Unhandled exception in main loop")
                alerts.send(
                    "critical_error",
                    f"Unhandled exception: {type(exc).__name__}: {exc}",
                    "critical",
                )
                self.shutdown(f"unhandled_exception: {type(exc).__name__}")
                break

            time.sleep(self._bar_interval)

    def _run_bar(self) -> None:
        """Process a single bar iteration: risk update + per-ticker pipeline."""
        # Detect market open→close transition for daily P&L summary
        is_open = self._client.is_market_open()
        if self._market_was_open and not is_open:
            self._fire_daily_pnl_summary()
        self._market_was_open = is_open

        # Per-ticker signal pipeline
        for ticker in settings.TICKERS:
            self._process_ticker(ticker)

        # Update risk manager with current NAV after all bars processed
        try:
            nav   = float(position_tracker.get_nav())
            fired = self._risk.update(nav)
            for breaker in fired:
                log.warning("Circuit breaker fired: %s", breaker)
                alerts.send(
                    "circuit_breaker",
                    f"Circuit breaker triggered: {breaker}",
                    "warning",
                )
        except Exception as exc:
            log.warning("Risk manager NAV update failed: %s", exc)

    def _process_ticker(self, ticker: str) -> None:
        """Run the full signal pipeline for one ticker."""
        ohlcv = self._fetch_bars_with_retry(ticker)
        if ohlcv is None:
            return  # feed dropped; alert already fired in _fetch_bars_with_retry

        # Feature engineering + HMM prediction
        try:
            features_row = feature_engineering.compute_latest(ohlcv)
            regime       = self._hmm.predict_current(features_row)
        except Exception as exc:
            log.warning(
                "HMM prediction failed for %s: %s — skipping bar", ticker, exc
            )
            return

        # Regime change notification
        if regime != -1 and regime != self._current_regime:
            prev = self._current_regime
            self._current_regime = regime
            regime_name = self._hmm.regime_name(regime)
            log.info(
                "Regime change: %s → %s (%s)",
                _REGIME_NAMES.get(prev, str(prev)), regime_name, ticker,
            )
            alerts.send(
                "regime_change",
                f"Regime change detected: {_REGIME_NAMES.get(prev, prev)} "
                f"→ {regime_name}",
                "info",
            )

        if regime == -1:
            log.debug("%s: no regime confirmed yet, skipping signal", ticker)
            return

        # Build signal
        try:
            nav = float(position_tracker.get_nav())
        except Exception:
            nav = float(self._client.get_account().portfolio_value)

        confidence = (
            0.8 if self._hmm.is_confirmed() and not self._hmm.is_uncertain()
            else 0.5
        )
        signal = regime_strategies.get_signal(
            regime            = regime,
            confidence        = confidence,
            portfolio_nav     = nav,
            current_allocation= 0.0,
            is_uncertain      = self._hmm.is_uncertain(),
        )

        # Risk gate
        approval = self._risk.approve(signal, nav)
        if not approval.approved:
            log.info(
                "%s: signal blocked — %s", ticker, approval.reason
            )
            return

        # Execute order
        try:
            size = signal.position_size_usd * approval.size_multiplier
            order_result = order_executor.submit(signal)
            log.info(
                "Trade placed: %s  side=buy  size=$%.2f  regime=%s  request_id=%s",
                ticker, size, signal.regime_name,
                getattr(order_result, "request_id", ""),
            )
            alerts.send(
                "trade_placed",
                f"Trade placed: {ticker}  side=buy  "
                f"size=${size:.2f}  regime={signal.regime_name}",
                "info",
            )
            position_tracker.on_fill(order_result)
        except Exception as exc:
            log.error("Order execution failed for %s: %s", ticker, exc)

    def _fetch_bars_with_retry(self, ticker: str) -> Optional[pd.DataFrame]:
        """Return latest OHLCV bars, retrying up to DATA_MAX_RETRIES times."""
        for attempt in range(1, DATA_MAX_RETRIES + 1):
            try:
                return market_data.get_latest_bars(ticker, settings.HMM_TRAIN_BARS)
            except Exception as exc:
                log.warning(
                    "Data fetch failed for %s (attempt %d/%d): %s",
                    ticker, attempt, DATA_MAX_RETRIES, exc,
                )
                if attempt < DATA_MAX_RETRIES:
                    time.sleep(DATA_RETRY_WAIT_SECS)
        log.error("Data feed drop for %s after %d attempts", ticker, DATA_MAX_RETRIES)
        alerts.send(
            "data_feed_drop",
            f"Data feed unavailable for {ticker} after {DATA_MAX_RETRIES} retries",
            "warning",
        )
        return None

    def _fire_daily_pnl_summary(self) -> None:
        """Fire end-of-day P&L alert at market close."""
        try:
            pnl = float(position_tracker.get_daily_pnl())
            msg = f"Daily P&L summary: ${pnl:+,.2f}"
            log.info(msg)
            alerts.send("daily_pnl", msg, "info")
        except Exception as exc:
            log.warning("Could not compute daily P&L: %s", exc)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self, reason: str = "manual") -> None:
        """Stop the loop, optionally cancel orders, remove lockfile, alert."""
        self._running       = False
        self._shutdown_reason = reason
        log.info("Shutting down: reason=%s", reason)

        if self._close_on_shutdown:
            try:
                cancelled = order_executor.cancel_all()
                log.info("Cancelled %d open order(s)", sum(bool(c) for c in cancelled))
            except Exception as exc:
                log.error("Failed to cancel orders on shutdown: %s", exc)

        try:
            if self._lockfile.exists():
                self._lockfile.unlink()
                log.info("Lockfile removed: %s", self._lockfile)
        except OSError as exc:
            log.error("Could not remove lockfile: %s", exc)

        try:
            alerts.send(
                "shutdown",
                f"Regime Trader stopped. Reason: {reason}",
                "info",
            )
        except Exception:
            pass  # never let alert failure block shutdown completion

        log.info("=== Regime Trader stopped ===")


# ---------------------------------------------------------------------------
# Signal handling and entry point
# ---------------------------------------------------------------------------

_trader: Optional[RegimeTrader] = None


def _signal_handler(signum: int, _frame) -> None:  # type: ignore[type-arg]
    name = signal_module.Signals(signum).name
    log.info("OS signal received: %s", name)
    if _trader is not None:
        _trader.shutdown(f"signal_{name}")
    sys.exit(0)


def main() -> None:
    _setup_logging()

    global _trader  # noqa: PLW0603

    try:
        client = AlpacaClient()
    except ConfigurationError as exc:
        logging.critical("Credential error: %s", exc)
        sys.exit(1)

    _trader = RegimeTrader(
        client       = client,
        hmm          = HMMEngine(),
        risk_manager = RiskManager(),
    )

    signal_module.signal(signal_module.SIGINT,  _signal_handler)
    signal_module.signal(signal_module.SIGTERM, _signal_handler)

    try:
        _trader.startup()
        _trader.run()
    except SystemExit:
        raise
    except Exception as exc:
        log.exception("Fatal error: %s", exc)
        try:
            alerts.send("critical_error", f"Fatal error: {exc}", "critical")
        except Exception:
            pass
        if _trader is not None:
            _trader.shutdown(f"fatal_error: {type(exc).__name__}")
        sys.exit(1)


if __name__ == "__main__":
    main()

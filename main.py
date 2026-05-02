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
        hmm: Optional[HMMEngine] = None,
        risk_manager: RiskManager,
        lockfile: Path = LOCKFILE,
        bar_interval: int = BAR_INTERVAL_SECS,
        close_positions_on_shutdown: bool = False,
    ) -> None:
        self._client       = client
        self._hmm          = hmm        # fallback used when _hmm_engines is empty (tests)
        self._hmm_engines: dict[str, HMMEngine] = {}  # populated in startup()
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
        # str() handles both plain strings and enum forms like "AccountStatus.ACTIVE";
        # split on "." and take the last segment so either form compares correctly.
        status_str = str(acct.status).split(".")[-1].upper()
        if status_str not in {"ACTIVE", "APPROVED"}:
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

        # Step 5: train one HMMEngine per ticker on 2 years of daily data
        end   = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=730)
        for ticker in settings.TICKERS:
            log.info("Fetching 2 years of daily %s bars for HMM training…", ticker)
            ohlcv    = market_data.get_historical_bars(ticker, start, end, "1Day")
            features = feature_engineering.compute(ohlcv)
            engine   = HMMEngine(ticker)
            engine.fit(features)
            self._hmm_engines[ticker] = engine
            log.info("HMM [%s] trained successfully", ticker)
        log.info("HMM engines ready: %s", list(self._hmm_engines))

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

        # Cancel any lingering open orders from a prior session
        try:
            cancelled = order_executor.cancel_all()
            n_cancelled = sum(bool(c) for c in cancelled)
            if n_cancelled:
                log.info("Startup: cancelled %d lingering open order(s)", n_cancelled)
        except Exception as exc:
            log.warning("Startup: could not cancel open orders: %s", exc)

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
        engine = self._hmm_engines.get(ticker, self._hmm)
        if engine is None:
            log.warning("No HMM engine for %s — skipping bar", ticker)
            return
        try:
            features_row = feature_engineering.compute_latest(ohlcv)
            regime       = engine.predict_current(features_row)
        except Exception as exc:
            log.warning(
                "HMM prediction failed for %s: %s — skipping bar", ticker, exc
            )
            return

        # Regime change notification
        if regime != -1 and regime != self._current_regime:
            prev = self._current_regime
            self._current_regime = regime
            regime_name = engine.regime_name(regime)
            log.info(
                "HMM [%s] regime change: %s → %s",
                ticker, _REGIME_NAMES.get(prev, str(prev)), regime_name,
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

        if ticker in settings.REFERENCE_TICKERS:
            log.debug("%s: reference ticker — regime context only, no trade", ticker)
            return

        # BTC uses its own cycle-aware strategy pipeline
        if ticker.upper() == "BTC":
            try:
                self._process_btc(ticker, ohlcv, regime, engine)
            except Exception as exc:
                log.warning("BTC pipeline error for %s: %s", ticker, exc)
            return

        # Equity-only: gate order execution to market hours
        if settings.IS_EQUITY_HOURS_ONLY and not self._client.is_market_open():
            log.debug("Market closed - skipping equity ticker %s", ticker)
            return

        # Build signal
        try:
            nav = float(position_tracker.get_nav())
        except Exception:
            nav = float(self._client.get_account().portfolio_value)

        confidence = (
            0.8 if engine.is_confirmed() and not engine.is_uncertain()
            else 0.5
        )
        signal = regime_strategies.get_signal(
            regime            = regime,
            confidence        = confidence,
            portfolio_nav     = nav,
            current_allocation= 0.0,
            is_uncertain      = engine.is_uncertain(),
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
            order_result = order_executor.submit(signal, symbol=ticker)
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

    def _process_btc(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        regime: int,
        engine,
    ) -> None:
        """Run the BTC-specific cycle-aware strategy pipeline."""
        from core.btc_strategy import BTCPosition, BTCStrategy
        from core.cycle_engine import CycleEngine

        try:
            current_price = float(ohlcv["close"].iloc[-1])
        except Exception as exc:
            log.warning("BTC: could not read current price: %s", exc)
            return

        try:
            cycle_eng    = CycleEngine(ticker)
            cycle_signal = cycle_eng.get_cycle_signal(ohlcv)
        except Exception as exc:
            log.warning("BTC cycle signal failed: %s — skipping BTC bar", exc)
            return

        try:
            nav = float(position_tracker.get_nav())
        except Exception:
            nav = float(self._client.get_account().portfolio_value)

        try:
            buying_power = float(self._client.get_account().buying_power)
        except Exception:
            buying_power = nav

        # Resolve current BTC position from broker
        current_position: Optional[BTCPosition] = None
        try:
            for pos in self._client.get_positions():
                if pos.symbol.upper() in ("BTC/USD", "BTC"):
                    cost = float(pos.avg_entry_price)
                    current_position = BTCPosition(
                        symbol=pos.symbol,
                        shares_held=float(pos.qty),
                        avg_cost=cost,
                        current_price=current_price,
                        unrealized_pnl=float(pos.unrealized_pl),
                        unrealized_pnl_pct=(
                            (current_price - cost) / cost if cost > 0 else 0.0
                        ),
                        entry_regime=regime,
                        entry_cycle_score=float(cycle_signal.composite_score),
                    )
                    break
        except Exception as exc:
            log.warning("BTC: could not fetch broker position: %s", exc)

        is_uncertain = engine.is_uncertain()
        confidence   = (
            0.8 if engine.is_confirmed() and not is_uncertain else 0.5
        )

        strategy     = BTCStrategy()
        target_alloc = strategy.get_target_allocation(regime, cycle_signal, is_uncertain)

        action = strategy.get_action(
            current_position  = current_position,
            target_allocation = target_alloc,
            portfolio_nav     = nav,
            buying_power      = buying_power,
            current_price     = current_price,
            regime            = regime,
            cycle_score       = float(cycle_signal.composite_score),
            confidence        = confidence,
        )

        if action.action == "HOLD":
            log.debug(
                "BTC: HOLD (target=%.1f%% within threshold)", target_alloc * 100
            )
            return

        log.info(
            "BTC action: %s  size=$%.2f  target=%.1f%%  regime=%s  cycle=%.2f",
            action.action,
            action.size_usd,
            target_alloc * 100,
            _REGIME_NAMES.get(regime, str(regime)),
            float(cycle_signal.composite_score),
        )

        btc_symbol = settings.BTC_TICKERS[0]  # "BTC/USD"
        try:
            if action.action == "BUY":
                result = order_executor.submit_crypto_order(
                    btc_symbol, "buy", action.size_usd, self._client
                )
            elif action.action in ("REDUCE", "EXIT", "SELL"):
                result = order_executor.submit_crypto_order(
                    btc_symbol, "sell", action.size_usd, self._client
                )
            else:
                result = None

            if result is not None:
                alerts.send_btc_trade_alert(
                    action,
                    regime_name=_REGIME_NAMES.get(regime, str(regime)),
                    cycle_score=float(cycle_signal.composite_score),
                )
                position_tracker.on_fill(result)
        except Exception as exc:
            log.error("BTC order execution failed: %s", exc)

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

        try:
            cancelled = order_executor.cancel_all()
            log.info("Cancelled %d open order(s) on shutdown", sum(bool(c) for c in cancelled))
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

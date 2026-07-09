"""
wheel_main.py
--------------
Wheel-only orchestrator (Phase 1 — scan-only).

A slimmed fork of main.py that runs ONLY the wheel pipeline:

    startup → train HMM on the market-proxy ticker for regime context
            → run a regime-aware wheel scan (nightly + once at startup)
            → alert on candidates + write logs/wheel_state.json

It deliberately excludes every non-wheel trading path from main.py — no BTC
cycle strategy, no regime equity trading, no exit/crash logic, and no order
execution. Nothing here places an order; wheel execution is a later phase.

main.py is left untouched: this is a parallel entry point selected by running
`python wheel_main.py` instead of `python main.py`.

Startup sequence:
  1. Lockfile guard (wheel_trading.lock — independent of main.py's lock)
  2. Verify Alpaca account is tradeable
  3. Train HMM on WHEEL_REGIME_TICKER (market regime for the scan)
  4. Initialise RiskManager NAV (dashboard + future sizing; not gating yet)
  5. Start the scanner scheduler with an in-process callback

Loop:
  - Run an initial scan at startup (immediate candidates)
  - The scheduler fires a fresh regime-aware scan once per weekday
  - Idle otherwise; shutdown cleanly on SIGINT/SIGTERM
"""

from __future__ import annotations

import json
import logging
import os
import signal as signal_module
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from broker.alpaca_client import AlpacaClient
from config import settings
from config.credentials import ConfigurationError, enable_os_trust_store
from core import alerts, feature_engineering, market_data
from core.hmm_engine import HMMEngine
from core.risk_manager import RiskManager
from core.scheduler import ScannerScheduler
from core.wheel_executor import WheelExecutor
from wheel_scanner import WheelScanner

log = logging.getLogger(__name__)

LOCKFILE           = Path(__file__).parent / "wheel_trading.lock"
LOG_DIR            = Path(__file__).parent / "logs"
IDLE_INTERVAL_SECS = 300

_REGIME_NAMES = {-1: "unconfirmed", 0: "crash", 1: "bear",
                 2: "neutral", 3: "bull", 4: "euphoria"}


def _setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Windows consoles/files default to cp1252 and crash on log lines containing
    # non-latin glyphs (e.g. the HMM's "state→regime" arrow). Force UTF-8.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
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
    fh = logging.FileHandler(
        LOG_DIR / f"wheel_trader_{datetime.now().strftime('%Y%m%d')}.log",
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)


class WheelTrader:
    """Wheel-only orchestrator: regime context + scheduled scans, no trading."""

    def __init__(
        self,
        *,
        client: AlpacaClient,
        risk_manager: RiskManager,
        lockfile: Path = LOCKFILE,
        scan_on_startup: bool = True,
    ) -> None:
        self._client          = client
        self._risk            = risk_manager
        self._lockfile        = lockfile
        self._scan_on_startup = scan_on_startup

        self._hmm: Optional[HMMEngine] = None
        self._scheduler: Optional[ScannerScheduler] = None
        self._executor: Optional[WheelExecutor] = None
        self._scan_lock  = threading.Lock()
        self._stop_event = threading.Event()
        self._running    = False

        self._last_regime:     int = -1
        self._last_candidates: list = []
        self._last_scan_at:    Optional[datetime] = None

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def startup(self) -> None:
        log.info("=== Wheel Trader starting up (scan-only) ===")

        # This host runs a TLS-inspection proxy whose root CA is in the OS store
        # but not certifi, so all outbound HTTPS (Alpaca, alerts, tastytrade)
        # fails verification without routing through the OS trust store.
        enable_os_trust_store()

        if self._lockfile.exists():
            msg = (
                f"Lockfile already exists at {self._lockfile}. "
                "Another wheel instance may be running. Delete it to restart."
            )
            log.error(msg)
            try:
                alerts.send("lockfile_present", msg, "critical")
            except Exception:
                pass
            raise SystemExit(1)

        self._lockfile.write_text(
            f"pid={os.getpid()} started={datetime.now(tz=timezone.utc).isoformat()}\n"
        )
        log.info("Lockfile written: %s", self._lockfile)

        acct = self._client.get_account()
        status_str = str(acct.status).split(".")[-1].upper()
        if status_str not in {"ACTIVE", "APPROVED"}:
            raise RuntimeError(f"Alpaca account not tradeable: status={acct.status!r}")
        log.info(
            "Account verified: id=%s status=%s NAV=$%.2f",
            acct.account_id, acct.status, acct.portfolio_value,
        )

        is_open = self._client.is_market_open()
        log.info("Market is currently: %s", "OPEN" if is_open else "CLOSED")

        # Train once at startup to fail fast on data/model problems.
        regime = self._train_and_predict_regime()
        log.info(
            "Initial market regime [%s]: %s",
            settings.WHEEL_REGIME_TICKER, _REGIME_NAMES.get(regime, str(regime)),
        )

        self._risk.initialize(float(acct.portfolio_value))

        self._scheduler = ScannerScheduler(on_fire=self._run_scan)
        self._scheduler.start()

        try:
            alerts.send(
                "wheel_startup",
                f"Wheel Trader started (scan-only). NAV=${acct.portfolio_value:,.2f}  "
                f"regime={_REGIME_NAMES.get(regime, regime)}",
                "info",
            )
        except Exception:
            pass
        log.info("Startup complete — entering wheel loop")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._running = True
        log.info(
            "Wheel loop running (scan fires %02d:%02d UTC Mon–Fri; regime ticker=%s)",
            settings.SCANNER_RUN_UTC_HOUR, settings.SCANNER_RUN_UTC_MINUTE,
            settings.WHEEL_REGIME_TICKER,
        )
        if self._scan_on_startup:
            self._run_scan()

        exec_on = settings.WHEEL_EXECUTION_ENABLED
        log.info("Wheel execution: %s", "ENABLED" if exec_on else "disabled (scan-only)")
        interval = settings.WHEEL_EXEC_INTERVAL_SECS if exec_on else IDLE_INTERVAL_SECS

        while self._running:
            if exec_on:
                self._run_wheel_execution()
            self._stop_event.wait(timeout=interval)

    # ------------------------------------------------------------------
    # Regime + scan
    # ------------------------------------------------------------------

    def _run_wheel_execution(self) -> None:
        """One cash-secured-put execution pass over scanner candidates + any
        open wheel positions. Only runs when WHEEL_EXECUTION_ENABLED."""
        if self._executor is None:
            self._executor = WheelExecutor(client=self._client)

        scan_tickers = [c.ticker for c in self._last_candidates]
        candidates = list(dict.fromkeys(
            [*scan_tickers, *settings.WHEEL_TICKERS, *self._executor.open_tickers()]
        ))
        if not candidates:
            return

        is_uncertain = self._hmm.is_uncertain() if self._hmm is not None else True
        log.info(
            "Wheel execution pass: %d ticker(s), regime=%s uncertain=%s",
            len(candidates), _REGIME_NAMES.get(self._last_regime, self._last_regime),
            is_uncertain,
        )
        try:
            self._executor.run_once(candidates, self._last_regime, is_uncertain)
        except Exception:
            log.exception("Wheel execution pass failed")

    def _train_and_predict_regime(self) -> int:
        """Train the HMM on a fresh 2 years of the market-proxy ticker and return
        the current confirmed regime. Retraining each cycle keeps the model
        non-stale.

        The regime is read by replaying recent bars through predict_current so
        the confirmation + flicker gates settle exactly as they would across the
        live loop's bars — a single one-shot predict can never reach
        CONFIRMATION_BARS and would always return -1 (unconfirmed).
        """
        ticker = settings.WHEEL_REGIME_TICKER
        end    = datetime.now(tz=timezone.utc)
        start  = end - timedelta(days=730)

        ohlcv    = market_data.get_historical_bars(ticker, start, end, "1Day")
        features = feature_engineering.compute(ohlcv)
        engine   = HMMEngine(ticker)
        engine.fit(features)
        self._hmm = engine

        warmup = settings.CONFIRMATION_BARS + settings.FLICKER_WINDOW
        regime = -1
        for _, row in features.iloc[-warmup:].iterrows():
            regime = engine.predict_current(row)
        self._last_regime = regime
        return regime

    def _run_scan(self) -> None:
        """Run one regime-aware wheel scan. Invoked at startup and by the
        scheduler. Guarded so two scans never overlap."""
        if not self._scan_lock.acquire(blocking=False):
            log.info("Wheel scan already in progress — skipping trigger")
            return
        try:
            regime = self._train_and_predict_regime()
            regime_label = regime if regime in (0, 1, 2, 3, 4) else None
            log.info(
                "Wheel scan starting (regime=%s)",
                _REGIME_NAMES.get(regime, str(regime)),
            )
            candidates = WheelScanner(regime_label=regime_label).run()
            self._last_candidates = candidates
            self._last_scan_at    = datetime.now(tz=timezone.utc)
            self._alert_candidates(candidates)
            self._write_wheel_state()
        except Exception as exc:
            log.exception("Wheel scan failed")
            try:
                alerts.send(
                    "wheel_scan_error",
                    f"Wheel scan failed: {type(exc).__name__}: {exc}",
                    "warning",
                )
            except Exception:
                pass
        finally:
            self._scan_lock.release()

    def _alert_candidates(self, candidates: list) -> None:
        regime_name = _REGIME_NAMES.get(self._last_regime, "unconfirmed")
        if not candidates:
            alerts.send(
                "wheel_scan",
                f"Wheel scan complete (regime={regime_name}): no candidates met threshold",
                "info",
            )
            return
        top   = candidates[:5]
        names = ", ".join(f"{c.ticker}({c.composite_score:.0f})" for c in top)
        alerts.send(
            "wheel_scan",
            f"Wheel scan complete (regime={regime_name}): "
            f"{len(candidates)} candidate(s). Top: {names}",
            "info",
        )

    # ------------------------------------------------------------------
    # Dashboard state
    # ------------------------------------------------------------------

    def _write_wheel_state(self) -> None:
        try:
            nav = float(self._client.get_account().portfolio_value)
        except Exception:
            nav = 0.0

        candidates = [
            {
                "ticker":            c.ticker,
                "composite_score":   c.composite_score,
                "ivr":               c.ivr,
                "target_put_strike": c.target_put_strike,
                "target_expiry":     c.target_expiry,
                "dte":               c.dte,
                "annualized_yield_pct": c.annualized_yield_pct,
            }
            for c in self._last_candidates[:10]
        ]

        state = {
            "mode":            "wheel_only",
            "regime":          _REGIME_NAMES.get(self._last_regime, "unconfirmed"),
            "regime_id":       self._last_regime,
            "nav":             nav,
            "last_scan_at":    self._last_scan_at.isoformat() if self._last_scan_at else None,
            "candidate_count": len(self._last_candidates),
            "candidates":      candidates,
            "updated_at":      datetime.now(tz=timezone.utc).isoformat(),
        }

        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            tmp = LOG_DIR / "wheel_state.json.tmp"
            tmp.write_text(json.dumps(state, default=str), encoding="utf-8")
            tmp.replace(LOG_DIR / "wheel_state.json")
        except Exception as exc:
            log.warning("Failed to write wheel state: %s", exc)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self, reason: str = "manual") -> None:
        self._running = False
        self._stop_event.set()
        log.info("Shutting down: reason=%s", reason)

        if self._scheduler is not None:
            self._scheduler.stop()

        try:
            if self._lockfile.exists():
                self._lockfile.unlink()
                log.info("Lockfile removed: %s", self._lockfile)
        except OSError as exc:
            log.error("Could not remove lockfile: %s", exc)

        try:
            alerts.send("wheel_shutdown", f"Wheel Trader stopped. Reason: {reason}", "info")
        except Exception:
            pass
        log.info("=== Wheel Trader stopped ===")


# ---------------------------------------------------------------------------
# Signal handling and entry point
# ---------------------------------------------------------------------------

_trader: Optional[WheelTrader] = None


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

    _trader = WheelTrader(client=client, risk_manager=RiskManager())

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

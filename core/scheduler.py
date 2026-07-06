"""
core/scheduler.py
-----------------
Background scheduler that fires run_scanner.py once per weekday at a
configured UTC hour/minute, running alongside main.py in the same container.

Usage:
    scheduler = ScannerScheduler()
    scheduler.start()   # non-blocking — launches a daemon thread
    ...
    scheduler.stop()    # signals the thread to exit cleanly
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from datetime import date, datetime, timezone
from pathlib import Path

from config.settings import SCANNER_RUN_UTC_HOUR, SCANNER_RUN_UTC_MINUTE

log = logging.getLogger(__name__)

_SCRIPT    = Path(__file__).parent.parent / "scripts" / "run_scanner.py"
_TICK_SECS = 60   # how often the thread wakes to check the clock


class ScannerScheduler:
    """Fires run_scanner.py once per trading day at the configured UTC time.

    Scanner failures (non-zero exit, timeout, launch error) are logged but
    never propagate to the main trading loop.

    Parameters
    ----------
    utc_hour   : UTC hour  (0–23). Defaults to SCANNER_RUN_UTC_HOUR  (11 = 6 AM ET).
    utc_minute : UTC minute (0–59). Defaults to SCANNER_RUN_UTC_MINUTE (0).
    script     : Override path to run_scanner.py — useful in tests.
    on_fire    : Optional callable invoked in-process at fire time instead of
                 launching run_scanner.py as a subprocess. Lets an in-process
                 owner (e.g. wheel_main) run the scan with the live HMM regime.
                 Callback exceptions are logged and never propagate.
    """

    def __init__(
        self,
        utc_hour:   int       = SCANNER_RUN_UTC_HOUR,
        utc_minute: int       = SCANNER_RUN_UTC_MINUTE,
        script:     Path | None = None,
        on_fire:    "callable | None" = None,
    ) -> None:
        self._utc_hour   = utc_hour
        self._utc_minute = utc_minute
        self._script     = script or _SCRIPT
        self._on_fire    = on_fire
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_run_date: date | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the background scheduler thread (daemon, non-blocking)."""
        if self._thread is not None and self._thread.is_alive():
            log.warning("ScannerScheduler: already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="scanner-scheduler",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "ScannerScheduler started — fires Mon–Fri at %02d:%02d UTC",
            self._utc_hour, self._utc_minute,
        )

    def stop(self) -> None:
        """Signal the scheduler thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        log.info("ScannerScheduler stopped")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            now = datetime.now(tz=timezone.utc)
            if self._should_fire(now):
                self._last_run_date = now.date()
                self._fire()
            # Wake early if stop() is called, else tick every minute
            self._stop_event.wait(timeout=_TICK_SECS)

    def _should_fire(self, now: datetime) -> bool:
        """Return True when all conditions are met to launch the scanner."""
        if now.weekday() >= 5:                                    # Sat=5, Sun=6
            return False
        if now.hour != self._utc_hour or now.minute != self._utc_minute:
            return False
        if self._last_run_date == now.date():                     # already ran today
            return False
        return True

    def _fire(self) -> None:
        """Dispatch to the in-process callback if provided, else subprocess."""
        if self._on_fire is not None:
            try:
                self._on_fire()
            except Exception as exc:
                log.error("ScannerScheduler: on_fire callback failed: %s", exc)
            return
        self._run_scanner()

    def _run_scanner(self) -> None:
        log.info("ScannerScheduler: launching %s", self._script)
        try:
            result = subprocess.run(
                [sys.executable, str(self._script)],
                capture_output=True,
                text=True,
                timeout=3600,   # 1-hour hard limit
            )
            if result.returncode == 0:
                log.info("ScannerScheduler: scanner completed successfully")
            else:
                log.error(
                    "ScannerScheduler: scanner exited with code %d\nstderr:\n%s",
                    result.returncode,
                    result.stderr[-2000:] if result.stderr else "",
                )
        except subprocess.TimeoutExpired:
            log.error("ScannerScheduler: scanner timed out after 1 hour — killed")
        except Exception as exc:
            log.error("ScannerScheduler: failed to launch scanner: %s", exc)

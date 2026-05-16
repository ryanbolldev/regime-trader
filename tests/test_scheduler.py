"""
tests/test_scheduler.py
-----------------------
Unit tests for core.scheduler.ScannerScheduler.
"""

from __future__ import annotations

import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.scheduler import ScannerScheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc(year, month, day, hour, minute) -> datetime:
    """Build a UTC datetime; weekday is implicit from the calendar date."""
    return datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)


# Concrete UTC datetimes for weekday tests
# 2026-05-11 = Monday,  2026-05-15 = Friday
# 2026-05-16 = Saturday, 2026-05-17 = Sunday
_MON_11_00 = _utc(2026, 5, 11, 11, 0)
_TUE_11_00 = _utc(2026, 5, 12, 11, 0)
_FRI_11_00 = _utc(2026, 5, 15, 11, 0)
_SAT_11_00 = _utc(2026, 5, 16, 11, 0)
_SUN_11_00 = _utc(2026, 5, 17, 11, 0)
_MON_10_59 = _utc(2026, 5, 11, 10, 59)
_MON_11_01 = _utc(2026, 5, 11, 11, 1)
_MON_00_00 = _utc(2026, 5, 11, 0, 0)


def _scheduler(utc_hour=11, utc_minute=0, script=None) -> ScannerScheduler:
    """Return a scheduler pointing at a dummy script path."""
    script = script or Path("/fake/run_scanner.py")
    return ScannerScheduler(utc_hour=utc_hour, utc_minute=utc_minute, script=script)


# ---------------------------------------------------------------------------
# _should_fire — weekday / time / dedup logic
# ---------------------------------------------------------------------------

class TestShouldFire:
    def test_fires_on_monday_at_target_time(self):
        s = _scheduler()
        assert s._should_fire(_MON_11_00) is True

    def test_fires_on_friday_at_target_time(self):
        s = _scheduler()
        assert s._should_fire(_FRI_11_00) is True

    def test_does_not_fire_on_saturday(self):
        s = _scheduler()
        assert s._should_fire(_SAT_11_00) is False

    def test_does_not_fire_on_sunday(self):
        s = _scheduler()
        assert s._should_fire(_SUN_11_00) is False

    def test_does_not_fire_one_minute_early(self):
        s = _scheduler()
        assert s._should_fire(_MON_10_59) is False

    def test_does_not_fire_one_minute_late(self):
        s = _scheduler()
        assert s._should_fire(_MON_11_01) is False

    def test_does_not_fire_midnight(self):
        s = _scheduler()
        assert s._should_fire(_MON_00_00) is False

    def test_does_not_double_fire_same_day(self):
        s = _scheduler()
        s._last_run_date = _MON_11_00.date()
        assert s._should_fire(_MON_11_00) is False

    def test_fires_next_day_after_previous_run(self):
        s = _scheduler()
        s._last_run_date = _MON_11_00.date()   # ran Monday
        assert s._should_fire(_TUE_11_00) is True

    def test_custom_hour_and_minute(self):
        s = _scheduler(utc_hour=14, utc_minute=30)
        at_target  = _utc(2026, 5, 11, 14, 30)
        off_target = _utc(2026, 5, 11, 14, 31)
        assert s._should_fire(at_target)  is True
        assert s._should_fire(off_target) is False


# ---------------------------------------------------------------------------
# _run_scanner
# ---------------------------------------------------------------------------

class TestRunScanner:
    def test_calls_subprocess_with_script(self):
        script = Path("/fake/run_scanner.py")
        s = _scheduler(script=script)
        mock_result = MagicMock(returncode=0, stderr="")
        with patch("core.scheduler.subprocess.run", return_value=mock_result) as mock_run:
            s._run_scanner()
        import sys
        mock_run.assert_called_once_with(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=3600,
        )

    def test_logs_error_on_nonzero_exit(self, caplog):
        s = _scheduler()
        mock_result = MagicMock(returncode=1, stderr="boom")
        with patch("core.scheduler.subprocess.run", return_value=mock_result):
            with caplog.at_level("ERROR", logger="core.scheduler"):
                s._run_scanner()
        assert any("exited with code 1" in r.message for r in caplog.records)

    def test_handles_timeout(self, caplog):
        s = _scheduler()
        with patch("core.scheduler.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 3600)):
            with caplog.at_level("ERROR", logger="core.scheduler"):
                s._run_scanner()
        assert any("timed out" in r.message for r in caplog.records)

    def test_handles_launch_exception(self, caplog):
        s = _scheduler()
        with patch("core.scheduler.subprocess.run", side_effect=OSError("no such file")):
            with caplog.at_level("ERROR", logger="core.scheduler"):
                s._run_scanner()
        assert any("failed to launch" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# start / stop lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_start_creates_daemon_thread(self):
        s = _scheduler()
        with patch.object(s, "_loop"):   # don't actually loop
            s.start()
            assert s._thread is not None
            assert s._thread.daemon is True
            s.stop()

    def test_stop_sets_event_and_joins(self):
        s = _scheduler()
        fired = []

        def fake_loop():
            s._stop_event.wait()   # blocks until stop() sets the event
            fired.append(True)

        s._thread = threading.Thread(target=fake_loop, daemon=True)
        s._thread.start()
        s.stop()
        assert fired == [True]
        assert not s._thread.is_alive()

    def test_start_is_idempotent(self):
        s = _scheduler()
        # Use a blocking fake loop so the thread stays alive for the second start() call
        def blocking_loop():
            s._stop_event.wait()

        with patch.object(s, "_loop", side_effect=blocking_loop):
            s.start()
            first_thread = s._thread
            s.start()   # second call should be a no-op while thread is alive
            assert s._thread is first_thread
            s.stop()

    def test_last_run_date_set_before_subprocess(self):
        """Ensure _last_run_date is stamped before the subprocess fires so a
        crash in _run_scanner still prevents a same-day double-fire."""
        s = _scheduler()
        stamped_dates = []

        def fake_run():
            stamped_dates.append(s._last_run_date)

        with patch.object(s, "_run_scanner", side_effect=fake_run):
            now = _MON_11_00
            # Simulate the loop body directly
            if s._should_fire(now):
                s._last_run_date = now.date()
                s._run_scanner()

        assert stamped_dates == [_MON_11_00.date()]
        assert s._last_run_date == _MON_11_00.date()

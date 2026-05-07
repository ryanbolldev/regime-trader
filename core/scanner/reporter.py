"""
core/scanner/reporter.py
-------------------------
Writes scanner output to JSON, Markdown, and fires a Telegram alert.

Output files (logs/scanner/):
  watchlist_YYYY-MM-DD.json   — machine-readable full results
  watchlist_YYYY-MM-DD.md     — human-readable Markdown briefing

Public interface:
  Reporter.write(scored_tickers, run_metadata, distribution) -> tuple[Path, Path]
  Reporter.send_alert(scored_tickers, run_metadata) -> None
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import asdict
from datetime import date
from typing import Any

from core.scanner.scorer import ScoredTicker

log = logging.getLogger(__name__)

_LOGS_DIR   = pathlib.Path("logs") / "scanner"
_BAR_WIDTH  = 12   # max █ characters for histogram bars
_BUCKETS    = ["80-100", "60-80", "40-60", "20-40", "0-20"]  # top-down display order


class Reporter:
    """Serialise and distribute scanner results.

    Parameters
    ----------
    logs_dir : override the default logs/scanner directory (useful in tests)
    """

    def __init__(self, logs_dir: pathlib.Path | None = None) -> None:
        self._logs_dir = logs_dir or _LOGS_DIR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(
        self,
        scored: list[ScoredTicker],
        metadata: dict[str, Any] | None = None,
        distribution: dict | None = None,
    ) -> tuple[pathlib.Path, pathlib.Path]:
        """Write JSON and Markdown watchlist files.

        Parameters
        ----------
        scored       : filtered (threshold-passing) ScoredTicker list
        metadata     : run stats dict
        distribution : output of build_score_distribution() — covers ALL scored tickers

        Returns (json_path, md_path).
        """
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        today    = date.today().isoformat()
        metadata = metadata or {}

        json_path = self._write_json(scored, today, metadata, distribution or {})
        md_path   = self._write_markdown(scored, today, metadata, distribution or {})
        return json_path, md_path

    def send_alert(
        self,
        scored: list[ScoredTicker],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Fire a scanner_briefing alert via core.alerts.send()."""
        metadata = metadata or {}
        longs  = [s for s in scored if s.direction == "LONG"]
        shorts = [s for s in scored if s.direction == "SHORT"]

        lines = [
            "SCANNER BRIEFING",
            f"Date: {date.today().isoformat()}",
            f"Universe: {metadata.get('universe_size', '?')} tickers scanned",
            f"Qualified: {len(scored)} | LONG: {len(longs)} | SHORT: {len(shorts)}",
            "",
        ]
        if longs:
            lines.append("Top LONG candidates:")
            for s in longs[:5]:
                lines.append(
                    f"  {s.ticker} | regime={s.regime_name} | "
                    f"score={s.long_score:.0f} | iv_rank={_fmt_iv(s.iv_rank)} | "
                    f"strategy={s.suggested_strategy}"
                )
        if shorts:
            lines.append("Top SHORT candidates:")
            for s in shorts[:5]:
                lines.append(
                    f"  {s.ticker} | regime={s.regime_name} | "
                    f"score={s.short_score:.0f} | iv_rank={_fmt_iv(s.iv_rank)} | "
                    f"strategy={s.suggested_strategy}"
                )

        msg = "\n".join(lines)
        try:
            from core import alerts
            alerts.send("scanner_briefing", msg, severity="info")
        except Exception as exc:
            log.warning("Reporter: alert send failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _write_json(
        self,
        scored: list[ScoredTicker],
        today: str,
        metadata: dict[str, Any],
        distribution: dict,
    ) -> pathlib.Path:
        payload = {
            "date":               today,
            "metadata":           metadata,
            "score_distribution": distribution,
            "tickers":            [asdict(s) for s in scored],
        }
        path = self._logs_dir / f"watchlist_{today}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("Reporter: JSON written to %s", path)
        return path

    def _write_markdown(
        self,
        scored: list[ScoredTicker],
        today: str,
        metadata: dict[str, Any],
        distribution: dict,
    ) -> pathlib.Path:
        longs  = [s for s in scored if s.direction == "LONG"]
        shorts = [s for s in scored if s.direction == "SHORT"]

        lines = [
            f"# Regime Trader Watchlist — {today}",
            "",
            f"**Universe scanned:** {metadata.get('universe_size', '?')} tickers  ",
            f"**Qualified:** {len(scored)}  |  **LONG:** {len(longs)}  |  **SHORT:** {len(shorts)}  ",
            f"**Runtime:** {metadata.get('runtime_secs', '?')}s  ",
            "",
        ]

        # Score distribution section
        if distribution:
            lines += _render_distribution_md(distribution)

        if longs:
            lines += [
                "## LONG Candidates",
                "",
                "| Ticker | Regime | Score | IV Rank | Duration | Strategy |",
                "|--------|--------|-------|---------|----------|----------|",
            ]
            for s in longs:
                lines.append(
                    f"| {s.ticker} | {s.regime_name} | {s.long_score:.0f} | "
                    f"{_fmt_iv(s.iv_rank)} | {s.regime_duration_bars}d | "
                    f"{s.suggested_strategy} |"
                )
            lines.append("")

        if shorts:
            lines += [
                "## SHORT Candidates",
                "",
                "| Ticker | Regime | Score | IV Rank | Duration | Strategy |",
                "|--------|--------|-------|---------|----------|----------|",
            ]
            for s in shorts:
                lines.append(
                    f"| {s.ticker} | {s.regime_name} | {s.short_score:.0f} | "
                    f"{_fmt_iv(s.iv_rank)} | {s.regime_duration_bars}d | "
                    f"{s.suggested_strategy} |"
                )
            lines.append("")

        if not scored:
            lines.append("*No tickers met the scoring threshold today.*")

        path = self._logs_dir / f"watchlist_{today}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        log.info("Reporter: Markdown written to %s", path)
        return path


# ---------------------------------------------------------------------------
# Distribution rendering helpers
# ---------------------------------------------------------------------------

def _render_distribution_md(distribution: dict) -> list[str]:
    """Render the score distribution section as Markdown lines."""
    lines = [
        "## Score Distribution",
        "",
        "```",
        "=== SCORE DISTRIBUTION ===",
    ]

    for direction in ("long", "short"):
        d     = distribution.get(direction, {})
        total = d.get("total", 0)
        mean  = d.get("mean", 0.0)
        bkts  = d.get("buckets", {})
        passd = d.get("passed_threshold", 0)

        lines.append(f"Total scored: {total} tickers  (mean {mean:.1f})")
        lines.append("")
        lines.append(f"{direction.upper()} DISTRIBUTION:")

        max_count = max(bkts.values(), default=1) or 1
        for bucket in _BUCKETS:
            count = bkts.get(bucket, 0)
            bar   = _bar(count, max_count)
            lines.append(f"  {bucket}:  {count:>4} tickers  {bar}")

        lines.append(f"  Passed threshold (>=60): {passd}/{total}")
        lines.append("")

    lines.append("==========================")
    lines.append("```")
    lines.append("")
    return lines


def _bar(count: int, max_count: int) -> str:
    """Render a simple block bar scaled to _BAR_WIDTH."""
    if max_count == 0:
        return ""
    n = round(count / max_count * _BAR_WIDTH)
    return "█" * n


def _fmt_iv(iv_rank: float | None) -> str:
    return f"{iv_rank:.0f}" if iv_rank is not None else "N/A"

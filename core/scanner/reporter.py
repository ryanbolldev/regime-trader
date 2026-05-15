"""
core/scanner/reporter.py
-------------------------
Writes scanner output to JSON, Markdown, and fires a Telegram alert.

Output files (logs/scanner/):
  watchlist_YYYY-MM-DD.json   — machine-readable full results
  watchlist_YYYY-MM-DD.md     — human-readable Markdown briefing
  deployment_date.txt         — written on first run; used for paper-period tracking

Paper validation period:
  For SCANNER_PAPER_ONLY_DAYS calendar days after first deployment, every
  Markdown file and Telegram alert includes a prominent warning banner urging
  that scanner output is for research only.

Public interface:
  Reporter.write(scored, metadata, distribution, exclusion_counts) -> tuple[Path, Path]
  Reporter.send_alert(scored, metadata) -> None
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import asdict
from datetime import date
from typing import Any

from config.settings import SCANNER_PAPER_ONLY_DAYS
from core.scanner.scorer import ScoredTicker

log = logging.getLogger(__name__)

_LOGS_DIR  = pathlib.Path("logs") / "scanner"
_BAR_WIDTH = 12
_BUCKETS   = ["80-100", "60-80", "40-60", "20-40", "0-20"]

# Human-readable labels for exclusion reason keys
_EXCLUSION_LABELS: dict[str, str] = {
    "low_volume":            "Low volume (<1M ADV)",
    "low_price":             "Price below $10",
    "fit_failed":            "HMM fit failure",
    "rate_limit_exhausted":  "Rate limit exhausted",
    "low_liquidity_options": "Low liquidity options",
    "high_iv_event_risk":    "High IV event risk (>70)",
    "iv_data_unavailable":   "IV data unavailable (not excluded)",
}
_EXCLUSION_KEY_ORDER = list(_EXCLUSION_LABELS)


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
        exclusion_counts: dict[str, int] | None = None,
    ) -> tuple[pathlib.Path, pathlib.Path]:
        """Write JSON and Markdown watchlist files.

        Also creates deployment_date.txt on first run.
        Returns (json_path, md_path).
        """
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        _ensure_deployment_date(self._deployment_file)

        today    = date.today().isoformat()
        metadata = metadata or {}
        dist     = distribution or {}
        excl     = exclusion_counts or {}

        json_path = self._write_json(scored, today, metadata, dist, excl)
        md_path   = self._write_markdown(scored, today, metadata, dist, excl)
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

        # Paper validation banner in alert
        banner = _paper_banner(self._deployment_file)
        if banner:
            lines = [banner, ""] + lines

        if longs:
            lines.append("Top LONG candidates:")
            for s in longs[:10]:
                lines.append(
                    f"  {s.ticker} | regime={s.regime_name} | "
                    f"score={s.long_score:.0f} | iv_rank={_fmt_iv(s.iv_rank)} | "
                    f"strategy={s.suggested_strategy}"
                )
        if shorts:
            lines.append("Top SHORT candidates:")
            for s in shorts[:10]:
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

    @property
    def _deployment_file(self) -> pathlib.Path:
        return self._logs_dir / "deployment_date.txt"

    def _write_json(
        self,
        scored: list[ScoredTicker],
        today: str,
        metadata: dict[str, Any],
        distribution: dict,
        exclusion_counts: dict[str, int],
    ) -> pathlib.Path:
        payload = {
            "date":               today,
            "metadata":           metadata,
            "score_distribution": distribution,
            "exclusion_counts":   exclusion_counts,
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
        exclusion_counts: dict[str, int],
    ) -> pathlib.Path:
        longs  = [s for s in scored if s.direction == "LONG"]
        shorts = [s for s in scored if s.direction == "SHORT"]

        lines: list[str] = []

        # Paper validation banner (above everything when active)
        banner = _paper_banner(self._deployment_file)
        if banner:
            lines += [banner, ""]

        lines += [
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

        # Exclusions breakdown section
        if exclusion_counts:
            lines += _render_exclusions_md(exclusion_counts)

        if longs:
            lines += [
                "## LONG Candidates",
                "",
                "| Ticker | Regime | Score | IV Rank | Duration | Strategy |",
                "|--------|--------|-------|---------|----------|----------|",
            ]
            for s in longs:
                iv_note = "" if s.iv_data_available else " \\[regime-only, no IV]"
                lines.append(
                    f"| {s.ticker} | {s.regime_name} | {s.long_score:.0f} | "
                    f"{_fmt_iv(s.iv_rank)} | {s.regime_duration_bars}d | "
                    f"{s.suggested_strategy}{iv_note} |"
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
                iv_note = "" if s.iv_data_available else " \\[regime-only, no IV]"
                lines.append(
                    f"| {s.ticker} | {s.regime_name} | {s.short_score:.0f} | "
                    f"{_fmt_iv(s.iv_rank)} | {s.regime_duration_bars}d | "
                    f"{s.suggested_strategy}{iv_note} |"
                )
            lines.append("")

        if not scored:
            lines.append("*No tickers met the scoring threshold today.*")

        path = self._logs_dir / f"watchlist_{today}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        log.info("Reporter: Markdown written to %s", path)
        return path


# ---------------------------------------------------------------------------
# Paper validation helpers
# ---------------------------------------------------------------------------

def _ensure_deployment_date(deployment_file: pathlib.Path) -> date:
    """Read deployment date, writing today's date on first run."""
    if deployment_file.exists():
        try:
            return date.fromisoformat(deployment_file.read_text(encoding="utf-8").strip())
        except Exception:
            pass
    # First run — create the file
    today = date.today()
    deployment_file.parent.mkdir(parents=True, exist_ok=True)
    deployment_file.write_text(today.isoformat(), encoding="utf-8")
    log.info("Reporter: deployment date set to %s (day 1/%d)", today, SCANNER_PAPER_ONLY_DAYS)
    return today


def _paper_banner(deployment_file: pathlib.Path) -> str:
    """Return the paper-validation banner string if still in the paper period, else ''."""
    if not deployment_file.exists():
        return ""
    try:
        deploy_date  = date.fromisoformat(deployment_file.read_text(encoding="utf-8").strip())
        days_elapsed = (date.today() - deploy_date).days + 1   # day 1 on deploy date
        if days_elapsed > SCANNER_PAPER_ONLY_DAYS:
            return ""
        return (
            f"⚠️  PAPER VALIDATION PERIOD — Day {days_elapsed}/{SCANNER_PAPER_ONLY_DAYS}\n"
            "Scanner output is for research only. Do not deploy real capital until\n"
            "the 30-day paper validation window closes and regime scores have been\n"
            "verified against actual price outcomes."
        )
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Distribution rendering helpers
# ---------------------------------------------------------------------------

def _render_distribution_md(distribution: dict) -> list[str]:
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
            bar   = "█" * round(count / max_count * _BAR_WIDTH)
            lines.append(f"  {bucket}:  {count:>4} tickers  {bar}")

        lines.append(f"  Passed threshold (>=60): {passd}/{total}")
        lines.append("")

    lines += ["==========================", "```", ""]
    return lines


# ---------------------------------------------------------------------------
# Exclusions breakdown rendering
# ---------------------------------------------------------------------------

def _render_exclusions_md(exclusion_counts: dict[str, int]) -> list[str]:
    """Render the EXCLUSIONS BREAKDOWN section."""
    total = sum(exclusion_counts.values())
    lines = [
        "## Exclusions Breakdown",
        "",
        "```",
        "EXCLUSIONS BREAKDOWN:",
    ]
    for key in _EXCLUSION_KEY_ORDER:
        label = _EXCLUSION_LABELS[key]
        count = exclusion_counts.get(key, 0)
        lines.append(f"  {label:<30} {count:>4} tickers")

    # Also include any unknown keys not in the canonical order
    for key, count in exclusion_counts.items():
        if key not in _EXCLUSION_LABELS:
            lines.append(f"  {key:<30} {count:>4} tickers")

    lines += [
        "  " + "─" * 36,
        f"  {'Total excluded':<30} {total:>4} tickers",
        "```",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _fmt_iv(iv_rank: float | None) -> str:
    return f"{iv_rank:.0f}" if iv_rank is not None else "N/A"

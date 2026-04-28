"""
dashboard/app.py
-----------------
Streamlit real-time monitoring dashboard for the regime trader.

Reads shared state from logs/dashboard_state.json (written by main.py).
Falls back to sensible defaults when the system is not running.

Run with:
    streamlit run dashboard/app.py --server.port 8501
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT      = Path(__file__).parent.parent
_STATE_FILE = _ROOT / "logs" / "dashboard_state.json"
_LOCKFILE   = _ROOT / "trading.lock"

# ---------------------------------------------------------------------------
# Default state (shown when system is not running)
# ---------------------------------------------------------------------------
_DEFAULT_STATE: dict[str, Any] = {
    "regime":          -1,
    "regime_name":     "unconfirmed",
    "regime_probs":    {},
    "flicker_count":   0,
    "is_confirmed":    False,
    "is_uncertain":    False,
    "nav":             0.0,
    "daily_pnl":       0.0,
    "daily_pnl_pct":   0.0,
    "equity_curve":    [],
    "circuit_breakers": [],
    "drawdown_pct":    0.0,
    "daily_drawdown":  0.0,
    "weekly_drawdown": 0.0,
    "signals":         [],
    "positions":       [],
    "performance": {
        "sharpe":       None,
        "max_drawdown": None,
        "win_rate":     None,
    },
    "last_updated": None,
}

# ---------------------------------------------------------------------------
# Regime colour palette
# ---------------------------------------------------------------------------
_REGIME_COLOR: dict[str, str] = {
    "crash":       "#FF4B4B",   # red
    "bear":        "#FF8C00",   # orange
    "neutral":     "#888888",   # gray
    "bull":        "#21C55D",   # green
    "euphoria":    "#FFD700",   # gold
    "unconfirmed": "#AAAAAA",   # light gray
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_state() -> dict[str, Any]:
    try:
        raw = _STATE_FILE.read_text(encoding="utf-8")
        data = json.loads(raw)
        return {**_DEFAULT_STATE, **data}
    except Exception:
        return dict(_DEFAULT_STATE)


def _color_for(regime_name: str) -> str:
    return _REGIME_COLOR.get(regime_name.lower(), "#AAAAAA")


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.2%}"


def _fmt_dollars(value: float | None) -> str:
    if value is None:
        return "—"
    return f"${value:,.2f}"


# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Regime Trader Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Auto-refresh every 30 seconds
# ---------------------------------------------------------------------------
try:
    import streamlit_autorefresh  # type: ignore
    streamlit_autorefresh.st_autorefresh(interval=30_000, key="autorefresh")
except ImportError:
    pass  # optional dependency

# ---------------------------------------------------------------------------
# Load state
# ---------------------------------------------------------------------------
state = _load_state()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Regime Trader — Live Dashboard")

last_upd = state.get("last_updated")
if last_upd:
    st.caption(f"Last updated: {last_upd}")
else:
    st.caption("System not running — showing defaults")

# ---------------------------------------------------------------------------
# Lockfile banner
# ---------------------------------------------------------------------------
if _LOCKFILE.exists():
    st.error(
        f"🔒 **PEAK DRAWDOWN LOCKOUT ACTIVE**  \n"
        f"Lockfile path: `{_LOCKFILE}`  \n"
        "Delete this file to resume trading after reviewing risk controls.",
        icon="🚨",
    )

# ============================================================
# Row 1 — Regime Status | Portfolio Value
# ============================================================
col1, col2 = st.columns(2)

# ── Panel 1: Regime Status ───────────────────────────────────
with col1:
    regime_name = state["regime_name"]
    color       = _color_for(regime_name)

    st.subheader("Current Regime")
    st.markdown(
        f"<h2 style='color:{color}; margin:0'>{regime_name.upper()}</h2>",
        unsafe_allow_html=True,
    )

    badges = []
    if state["is_confirmed"]:
        badges.append("✅ Confirmed")
    else:
        badges.append("⏳ Unconfirmed")
    if state["is_uncertain"]:
        badges.append("⚠️ Uncertain")
    st.markdown("  |  ".join(badges))

    st.metric("Flicker count (20-bar window)", state["flicker_count"])

    probs = state.get("regime_probs") or {}
    if probs:
        st.write("State probability distribution")
        prob_labels = list(probs.keys())
        prob_values = list(probs.values())
        st.bar_chart(dict(zip(prob_labels, prob_values)))

# ── Panel 2: Portfolio Value ─────────────────────────────────
with col2:
    st.subheader("Portfolio")
    nav        = state["nav"]
    daily_pnl  = state["daily_pnl"]
    pnl_pct    = state["daily_pnl_pct"]

    m1, m2, m3 = st.columns(3)
    m1.metric("NAV",       _fmt_dollars(nav))
    m2.metric("Daily P&L", _fmt_dollars(daily_pnl))
    m3.metric("Daily %",   _fmt_pct(pnl_pct))

    equity = state.get("equity_curve") or []
    if equity:
        import pandas as pd
        df = pd.DataFrame(equity, columns=["timestamp", "nav"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        st.line_chart(df["nav"])
    else:
        st.info("No equity curve data yet.")

# ============================================================
# Row 2 — Circuit Breakers | Signal Feed
# ============================================================
col3, col4 = st.columns(2)

# ── Panel 3: Circuit Breakers / Risk Status ──────────────────
with col3:
    st.subheader("Risk Status")

    dd  = state["drawdown_pct"]
    ddd = state["daily_drawdown"]
    wdd = state["weekly_drawdown"]

    def _drawdown_color(val: float, warn: float, halt: float) -> str:
        if val <= halt:
            return "red"
        if val <= warn:
            return "orange"
        return "green"

    r1, r2, r3 = st.columns(3)
    r1.metric("Peak Drawdown",   _fmt_pct(dd),  delta_color="inverse")
    r2.metric("Daily Drawdown",  _fmt_pct(ddd), delta_color="inverse")
    r3.metric("Weekly Drawdown", _fmt_pct(wdd), delta_color="inverse")

    breakers = state.get("circuit_breakers") or []
    if breakers:
        st.warning("Active circuit breakers: " + ", ".join(breakers))
    else:
        st.success("No circuit breakers active")

# ── Panel 4: Signal Feed ─────────────────────────────────────
with col4:
    st.subheader("Signal Feed (last 50)")
    signals = (state.get("signals") or [])[-50:]
    if signals:
        import pandas as pd
        df_sig = pd.DataFrame(signals)
        st.dataframe(df_sig, use_container_width=True, hide_index=True)
    else:
        st.info("No signals recorded yet.")

# ============================================================
# Row 3 — Open Positions | Performance
# ============================================================
col5, col6 = st.columns(2)

# ── Panel 5: Open Positions ───────────────────────────────────
with col5:
    st.subheader("Open Positions")
    positions = state.get("positions") or []
    if positions:
        import pandas as pd
        df_pos = pd.DataFrame(positions)
        st.dataframe(df_pos, use_container_width=True, hide_index=True)
    else:
        st.info("No open positions.")

# ── Panel 6: Performance Metrics ─────────────────────────────
with col6:
    st.subheader("Performance")
    perf = state.get("performance") or {}

    sharpe   = perf.get("sharpe")
    max_dd   = perf.get("max_drawdown")
    win_rate = perf.get("win_rate")

    p1, p2, p3 = st.columns(3)
    p1.metric("Sharpe Ratio",  f"{sharpe:.2f}"    if sharpe   is not None else "—")
    p2.metric("Max Drawdown",  _fmt_pct(max_dd)   if max_dd   is not None else "—")
    p3.metric("Win Rate",      _fmt_pct(win_rate)  if win_rate is not None else "—")

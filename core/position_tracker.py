"""
core/position_tracker.py
-------------------------
Maintains the authoritative in-process view of open positions and P&L.

Responsibilities:
  - Store and update position records: ticker, direction, entry price, quantity,
    current market value, unrealised P&L, cost basis
  - Listen for fill events from order_executor.py and update positions
    accordingly (open, add-to, reduce, close)
  - Compute real-time portfolio metrics: gross/net exposure, NAV, daily P&L,
    high-water mark, current drawdown from peak
  - Provide the correlation matrix over trailing bars needed by risk_manager.py
  - Persist position state to disk so it survives process restarts; re-hydrate
    from the broker's live position feed on startup to reconcile any drift
  - Expose a snapshot suitable for the Streamlit dashboard

Public interface (to be implemented):
  on_fill(fill_event) -> None
  get_open_positions() -> list[Position]
  get_portfolio_snapshot() -> PortfolioSnapshot
  get_nav() -> float
  get_daily_pnl() -> float
  get_drawdown_from_peak() -> float
  get_correlation_matrix(lookback_bars) -> pd.DataFrame
"""

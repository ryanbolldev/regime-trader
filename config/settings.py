"""
config/settings.py
------------------
Central repository for all tunable runtime parameters.

Covers:
  - Universe of tickers to trade
  - HMM regime thresholds and the number of states to test (3–7)
  - Regime stability filter: maximum flicker count allowed in the last 20 bars
  - Confirmation bars required before acting on a detected regime change
  - Risk limits: intraday drawdown stops, weekly drawdown stops, peak-drawdown
    lockout threshold, and per-trade capital-at-risk cap
  - Backtest windows: in-sample (252 days) and out-of-sample (~126 days)
  - Bar resolution for the main loop (default: 5-minute bars)
  - Broker / exchange connection parameters (no credentials here)
  - Alert channels configuration (email recipients, webhook URLs)

All values here are defaults; they may be overridden at runtime via environment
variables loaded in config/credentials.py, or through the Streamlit dashboard.
Never place API keys or secrets in this file.
"""

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
TICKERS = ["SPY", "QQQ", "IWM"]

# ---------------------------------------------------------------------------
# HMM model selection
# ---------------------------------------------------------------------------
HMM_MIN_STATES = 3
HMM_MAX_STATES = 7
HMM_COVARIANCE_TYPE = "full"
HMM_N_ITER = 100
HMM_TRAIN_BARS = 504          # ~2 years of daily bars

# ---------------------------------------------------------------------------
# Regime stability filters
# ---------------------------------------------------------------------------
CONFIRMATION_BARS = 3         # consecutive same-state bars before confirming
FLICKER_WINDOW = 20           # bars to look back for regime-change counting
FLICKER_THRESHOLD = 4         # max changes in FLICKER_WINDOW before suppressing

# ---------------------------------------------------------------------------
# Risk limits
# ---------------------------------------------------------------------------
INTRADAY_STOP_WARN  = -0.02   # -2 % intraday drawdown → warning
INTRADAY_STOP_HALT  = -0.03   # -3 % intraday drawdown → halt trading
WEEKLY_STOP         = -0.05   # -5 % weekly drawdown   → halt trading
PEAK_DRAWDOWN_LOCKOUT = -0.10 # -10 % peak drawdown    → lockout
PER_TRADE_RISK_CAP  = 0.01    # max 1 % of capital per trade
MAX_POSITIONS        = 5       # concurrent open positions cap
MAX_CORR_BUDGET      = 0.70   # max portfolio-level correlation allowed

# ---------------------------------------------------------------------------
# Strategy / regime allocation
# ---------------------------------------------------------------------------
REBALANCE_DRIFT_THRESHOLD   = 0.05   # rebalance only if allocation drifts > 5%
UNCERTAINTY_ALLOCATION_FACTOR = 0.60 # multiply target allocation when HMM is uncertain

# ---------------------------------------------------------------------------
# Backtest windows
# ---------------------------------------------------------------------------
BACKTEST_IN_SAMPLE_BARS  = 252
BACKTEST_OUT_SAMPLE_BARS = 126
BACKTEST_STEP_BARS       = 63   # quarterly re-training step

# ---------------------------------------------------------------------------
# Bar resolution
# ---------------------------------------------------------------------------
BAR_TIMEFRAME = "1Day"

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
RSI_PERIOD   = 14
VOL_WINDOW   = 20
VOLUME_WINDOW = 20
HL_NORM_WINDOW = 1            # single-bar feature, no rolling

# ---------------------------------------------------------------------------
# Broker connection (no credentials)
# ---------------------------------------------------------------------------
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
ALERT_EMAIL_RECIPIENTS = []
ALERT_WEBHOOK_URL      = ""
ALERT_COOLDOWN_SECONDS = 300

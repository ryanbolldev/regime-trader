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
TICKERS           = ["SPY", "MSTR", "BTC"]
REFERENCE_TICKERS = ["SPY"]   # run HMM for regime context but never trade

# ---------------------------------------------------------------------------
# HMM model selection
# ---------------------------------------------------------------------------
HMM_MIN_STATES = 3
HMM_MAX_STATES = 7
HMM_COVARIANCE_TYPE = "full"
HMM_N_ITER  = 500             # max EM iterations per candidate model
HMM_TOL     = 1e-5            # EM convergence tolerance (log-likelihood improvement)
HMM_N_INIT  = 5               # random restarts per candidate state count; best BIC wins
HMM_TRAIN_BARS = 504          # ~2 years of daily bars
HMM_STALENESS_ZSCORE_LIVE        = 2.0   # std-deviation threshold for live trading staleness
HMM_STALENESS_ZSCORE_WALKFORWARD = 2.5   # wider threshold for walk-forward contexts
HMM_STALENESS_MIN_SAMPLE_BARS    = 10    # minimum calibration bars; below this, disable detection
HMM_STALENESS_CALIBRATION_BARS   = 30    # tail window used to compute staleness mean/std

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

# Per-position equity exit thresholds (unrealized P&L as decimal fraction)
# Euphoria: close all longs immediately for profit-taking.
# Bear/neutral/bull: trailing stop — close if unrealized P&L falls below threshold.
EQUITY_EUPHORIA_FLATTEN = True    # set False to disable auto-flatten in euphoria
EQUITY_BEAR_STOP_PCT    = -0.05   # trailing stop in bear:    -5 % from entry
EQUITY_NEUTRAL_STOP_PCT = -0.06   # trailing stop in neutral: -6 % from entry
EQUITY_BULL_STOP_PCT    = -0.08   # trailing stop in bull:    -8 % from entry (wider)

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

# Slippage in basis points (applied once per side of a trade).
SLIPPAGE_BPS        = 5    # 0.05 % — equity and general assets
CRYPTO_SLIPPAGE_BPS = 10   # 0.10 % — crypto (wider spreads, 24/7 markets)

# ---------------------------------------------------------------------------
# Bar resolution
# ---------------------------------------------------------------------------
BAR_TIMEFRAME      = "1Day"
BAR_INTERVAL_SECS  = 86400   # seconds per bar — used by backtester for lag/timing maths
                              # (NOT the live-loop poll frequency; see POLL_INTERVAL_SECS in main.py)

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
ALERT_EMAIL_RECIPIENTS    = []
ALERT_WEBHOOK_URL         = ""
ALERT_COOLDOWN_SECONDS    = 300
ALERT_EMAIL_MIN_SEVERITY  = "warning"   # email only on "warning" or "critical"
ALERT_COOLDOWN_OVERRIDES: dict[str, int] = {
    "circuit_breaker":  0,   # critical — never suppress
    "critical_error":   0,   # critical — never suppress
    "lockfile_written": 0,   # startup/shutdown lifecycle
    "lockfile_present": 0,   # startup/shutdown lifecycle
}

# ---------------------------------------------------------------------------
# On-chain data
# ---------------------------------------------------------------------------
ONCHAIN_ENABLED       = True
ONCHAIN_CACHE_SECONDS = 300

# ---------------------------------------------------------------------------
# Cycle detection (60-day BTC cycle)
# ---------------------------------------------------------------------------
CYCLE_60D_CENTER           = 60
CYCLE_60D_STD              = 12
CYCLE_4Y_CENTER            = 1458    # ~4 years in days
CYCLE_4Y_STD               = 120
CYCLE_LOW_CONFIRMATION_PCT = 0.10    # price must rise >10% to confirm a low
CYCLE_COMPOSITE_THRESHOLD  = 0.65
CYCLE_QUALITY_LOOKBACK     = 3
CYCLE_DONCHIAN_WEIGHT      = 0.40
CYCLE_GAUSSIAN_WEIGHT      = 0.35
CYCLE_BOLLINGER_WEIGHT     = 0.25

# ---------------------------------------------------------------------------
# Market hours gate
# ---------------------------------------------------------------------------
IS_EQUITY_HOURS_ONLY = True   # block equity orders when market is closed

# ---------------------------------------------------------------------------
# Global trading master switch (main.py equity/BTC system)
# ---------------------------------------------------------------------------
# False = block all NEW equity + BTC entries in main.py. Protective exits
# (crash close, trailing stops, BTC REDUCE/EXIT) still fire so open positions
# can always be de-risked. The wheel strategy (wheel_main.py) is independent
# and unaffected by this flag.
TRADING_ENABLED            = False

# ---------------------------------------------------------------------------
# Live account mode
# ---------------------------------------------------------------------------
LIVE_ACCOUNT_MODE          = False   # Set True only for live deployment

LIVE_MAX_POSITION_PCT      = 0.20    # 20% of NAV per-trade hard cap (live)
LIVE_MAX_DEPLOYED_PCT      = 0.30    # 30% of NAV total deployment cap (live)
LIVE_INTRADAY_STOP_HALT    = -0.02   # daily halt at -2%  (paper: -3%)
LIVE_WEEKLY_STOP           = -0.03   # weekly resize at -3%  (paper: -5%)
LIVE_PEAK_DRAWDOWN_LOCKOUT = -0.05   # lockout at -5%  (paper: -10%)

# ---------------------------------------------------------------------------
# BTC spot trading
# ---------------------------------------------------------------------------
BTC_TICKERS             = ["BTCUSD"]
BTC_MAX_ALLOCATION      = 0.15
BTC_REBALANCE_THRESHOLD = 0.05
BTC_CYCLE_TIER_BOOST    = True

# MSTR carries ~2.5× BTC beta; add (mstr_value × MSTR_BTC_BETA) to effective
# BTC exposure before approving any new MSTR equity signal.
MSTR_BTC_BETA = 2.5

# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------
# Volume threshold is feed-dependent. IEX captures ~2-3% of total US equity
# volume (paper accounts only); SIP is the full consolidated tape (live accounts).
# Switch to 1_000_000 and set SCANNER_DATA_FEED='sip' when going live.
SCANNER_MIN_VOLUME          = 20_000      # avg daily volume filter (IEX feed)
SCANNER_MIN_PRICE           = 10.0        # price per share filter
SCANNER_MAX_WORKERS         = 5           # ThreadPoolExecutor parallelism (rate-limit safe)
SCANNER_BATCH_SLEEP_SECS    = 0.5         # sleep between worker batches to throttle API calls
SCANNER_MAX_RETRIES         = 3           # per-ticker retries on HTTP 429 before exclusion
SCANNER_SCORE_THRESHOLD     = 60          # minimum composite score to include
SCANNER_TRAIN_BARS          = 252         # bars used to fit HMM per ticker
SCANNER_DURATION_HOLDOUT_BARS = 40        # out-of-sample bars for regime duration counting
SCANNER_OPTIONS_SPREAD_MAX  = 0.20        # max ATM bid-ask spread ($) for liquidity
SCANNER_MAX_IV_RANK         = 70          # IV rank ceiling; tickers above this are excluded
SCANNER_PAPER_ONLY_DAYS     = 30          # paper-validation window after first deployment
SCANNER_DATA_FEED           = 'iex'       # data feed for bar requests (paper account cannot use SIP)
SCANNER_RUN_UTC_HOUR        = 11          # scheduler fire time — 11:00 UTC = 6 AM ET
SCANNER_RUN_UTC_MINUTE      = 0
WHEEL_REGIME_TICKER         = "SPY"       # market proxy trained by wheel_main for the scan's regime_label

# Volatility estimator weights (combined vol rank proxy — no options API required)
SCANNER_VOL_REALIZED_WEIGHT  = 0.50   # realized vol percentile rank weight
SCANNER_VOL_VIX_WEIGHT       = 0.30   # VIX percentile rank weight
SCANNER_VOL_TERM_WEIGHT      = 0.20   # vol term structure score weight

# Realized vol windows
SCANNER_VOL_WINDOW_SHORT     = 10     # bars — short-term realized vol
SCANNER_VOL_WINDOW_MID       = 20     # bars — medium-term realized vol (primary)
SCANNER_VOL_WINDOW_LONG      = 60     # bars — long-term realized vol
SCANNER_VOL_LOOKBACK         = 252    # bars — percentile rank lookback window

# VIX settings
SCANNER_VIX_SYMBOL           = 'VIXY' # Alpaca-tradeable VIX proxy
SCANNER_VIX_LOOKBACK         = 252    # bars for VIX percentile rank

# S&P 500 + Nasdaq 100 combined universe (~200 most liquid tickers)
SP500_NASDAQ100_UNIVERSE: list[str] = [
    # Mega-cap tech / growth
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO",
    "ADBE", "CRM", "NFLX", "INTU", "NOW", "PANW", "CRWD", "FTNT",
    # Semiconductors
    "AMD", "INTC", "QCOM", "TXN", "MU", "LRCX", "KLAC", "AMAT", "ADI",
    "MRVL", "CDNS", "SNPS", "ON", "MCHP",
    # Financials
    "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SPGI", "MCO", "AXP",
    "V", "MA", "PYPL", "FI", "COF", "USB", "PNC", "TFC",
    # Healthcare / biotech
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "BMY",
    "AMGN", "GILD", "VRTX", "REGN", "ISRG", "IDXX", "DXCM", "MRNA",
    "ELV", "HUM", "CI", "CVS", "SYK", "BSX", "MDT", "ZBH",
    # Consumer staples
    "PEP", "KO", "WMT", "COST", "PG", "MDLZ", "KHC", "MNST", "CL",
    "GIS", "K", "CAG", "MKC", "CHD", "CLX",
    # Consumer discretionary
    "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX", "ROST", "DLTR",
    "ORLY", "AZO", "EBAY", "BKNG", "ABNB", "MAR", "HLT",
    # Industrials
    "RTX", "HON", "CAT", "DE", "GE", "MMM", "ETN", "PH", "EMR", "ITW",
    "GD", "LMT", "NOC", "BA", "UNP", "CSX", "NSC", "FDX", "UPS",
    "CTAS", "FAST", "PAYX", "VRSK", "ODFL", "PCAR",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "DVN",
    "FANG", "OXY", "HAL", "BKR",
    # Utilities / Real estate
    "NEE", "SO", "DUK", "D", "AEP", "EXC", "PLD", "AMT", "CCI", "EQIX",
    # Communication
    "T", "VZ", "CMCSA", "DIS", "CHTR", "TMUS",
    # Materials
    "LIN", "APD", "SHW", "FCX", "NEM", "ALB", "CE",
    # Insurance / diversified
    "CB", "MMC", "AON", "AIG", "PRU", "MET", "AFL", "ALL",
    # Software / cloud
    "WDAY", "ZS", "OKTA", "DDOG", "ZM", "SNOW", "PLTR", "RBLX",
    # Other large-cap
    "IBM", "ORCL", "SAP", "ACN", "TDG", "ROP", "ANSS",
]

# ---------------------------------------------------------------------------
# Wheel strategy
# ---------------------------------------------------------------------------
WHEEL_TICKERS               = ["MSTR"]
WHEEL_PUT_DELTA_TARGET      = -0.28   # target delta for put selection
WHEEL_CALL_DELTA_TARGET     =  0.28   # target delta for call selection
WHEEL_MIN_DTE               =  30     # minimum days to expiration
WHEEL_MAX_DTE               =  45     # maximum days to expiration
WHEEL_EARLY_CLOSE_PROFIT_PCT =  0.50  # close at 50 % of max profit
WHEEL_EARLY_CLOSE_LOSS_PCT   =  2.00  # stop loss at 200 % of premium received
WHEEL_GAMMA_RISK_DTE         =  7     # close losing positions with < 7 DTE
WHEEL_MIN_IV_RANK            = 40     # minimum IV Rank (0–100) before new entry
WHEEL_IV_LOOKBACK_DAYS       = 252    # rolling window for IV Rank (1 trading year)

# ---------------------------------------------------------------------------
# Options data provider (wheel scanner)
# ---------------------------------------------------------------------------
OPTIONS_DATA_PROVIDER  = "tastytrade"  # "alpaca" | "tastytrade"; flipped 2026-07-07
                                       # after market-hours parity cleared + adds OI
TASTYTRADE_USE_SANDBOX = False         # data-only on live production (no orders)

# ---------------------------------------------------------------------------
# Wheel execution (Phase 2a — cash-secured puts). Gated OFF by default.
# ---------------------------------------------------------------------------
WHEEL_EXECUTION_ENABLED   = True    # master switch — True = wheel_main places option orders
WHEEL_EXEC_INTERVAL_SECS  = 900      # how often the execution pass runs (15 min)
MAX_WHEEL_POSITIONS       = 2        # max concurrent open wheel legs (limited capital)
WHEEL_MAX_COLLATERAL_PCT  = 0.35     # per-position collateral ceiling as fraction of NAV
WHEEL_TOTAL_DEPLOYED_PCT  = 0.70     # total wheel collateral ceiling as fraction of NAV
WHEEL_LIMIT_SLIPPAGE_PCT  = 0.00     # 0 = sell at mid; >0 shades the limit toward the bid
WHEEL_MAX_SPREAD_PCT      = 0.20     # skip a NEW entry if the option's bid-ask spread
                                     # exceeds this fraction of mid (execution liquidity sanity)

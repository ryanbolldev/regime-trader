# Regime Trader

A Python algorithmic trading bot that uses a Hidden Markov Model (HMM) to classify market regimes and adapts its strategy accordingly. Supports equities, Bitcoin spot, and options income via the wheel strategy — all on the Alpaca brokerage platform.

## Features

- **HMM regime classification** across 5 market states (crash → euphoria)
- **BTC spot trading** with 60-day cycle overlay and regime-tiered allocation
- **Wheel strategy** (cash-secured puts + covered calls) for income generation on MSTR and CVX
- **Multi-layer risk management**: per-trade caps, intraday/weekly/peak drawdown circuit breakers
- **Live account mode** with tighter caps, equity-only BTC trading, and mandatory second-validation logging
- **Alerts** via email (SMTP) and webhook (Slack, etc.)
- **695+ passing tests** with full mock coverage — no real HTTP calls in test suite

## Market Regimes

| Label | Regime   | Equity Allocation | Behaviour |
|-------|----------|-------------------|-----------|
| 0     | Crash    | 10% (defensive)   | Flatten longs, cash-heavy, no new entries |
| 1     | Bear     | 30%               | Reduced exposure, defensive sectors |
| 2     | Neutral  | 60%               | Mean-reversion entries, balanced |
| 3     | Bull     | 90% (1.1× lever)  | Momentum entries, full deployment |
| 4     | Euphoria | 70% (trimming)    | Profit-taking, tight stops, no new longs |

BTC allocation follows a separate regime table (0–75%) with a cycle-score tier boost.

## Quick Start

```bash
# 1. Clone and enter the project
git clone <repo-url> && cd regime_trader

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the credentials template and fill in your values
cp .env.example .env
# Edit .env — never commit it

# 5. Run the full test suite (all must pass before live use)
pytest tests/ -v

# 6. Start the bot (paper trading by default)
python main.py

# 7. Launch the dashboard (separate terminal)
streamlit run dashboard/app.py
```

## Credentials (.env)

```env
# Required
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxxxxxxxx
ALPACA_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ALPACA_BASE_URL=https://paper-api.alpaca.markets   # swap for live URL when ready

# Optional — alerts
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/...
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=alerts@example.com
SMTP_PASS=app-password
EMAIL_TO=you@example.com
```

## Safety Rules

- **Lockfile**: the bot creates `trading.lock` on startup. If it exits uncleanly and leaves the file, delete it manually after investigating. Never script its deletion — it is your last line of defence against a double-start.
- **Live account mode**: `LIVE_ACCOUNT_MODE` in `config/settings.py` defaults to `False` (paper). Set it to `True` only after reviewing all live-mode caps and running against paper trading successfully.
- **Credentials**: `.env` is in `.gitignore`. Never add API keys to code or commit them.
- **Circuit breakers**: the peak-drawdown lockout (`-10%` paper, `-5%` live) permanently halts the session once triggered. It will not reset mid-session.
- **Test before deploying**: run `pytest tests/ -v` after every configuration change.

## Project Structure

```
regime_trader/
├── config/
│   ├── settings.py            # All tunable parameters — start here
│   └── credentials.py         # .env loader (no hardcoded secrets)
├── core/
│   ├── hmm_engine.py          # HMM regime classifier (5 states, BIC selection)
│   ├── feature_engineering.py # Anti-lookahead OHLCV features
│   ├── regime_strategies.py   # Per-regime allocation & strategy logic
│   ├── btc_strategy.py        # BTC cycle-aware allocation & rebalancing
│   ├── cycle_engine.py        # 60-day BTC cycle detection & scoring
│   ├── wheel_strategy.py      # Covered call / cash-secured put cycle
│   ├── risk_manager.py        # Circuit breakers with absolute veto power
│   ├── order_executor.py      # Signal → broker order translation
│   ├── position_tracker.py    # Open positions, P&L, and wheel state
│   ├── market_data.py         # Historical and real-time OHLCV feeds
│   └── alerts.py              # Email / webhook notifications with cooldown
├── broker/
│   └── alpaca_client.py       # Alpaca REST API wrapper (paper & live)
├── dashboard/
│   └── app.py                 # Streamlit monitoring UI
├── tests/                     # One test file per module — 695+ tests
├── main.py                    # Orchestrator: startup, main loop, shutdown
├── requirements.txt
├── .env.example
└── .gitignore
```

## Key Configuration (config/settings.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TICKERS` | `["SPY","MSTR","CVX","BTC"]` | Full trading universe |
| `REFERENCE_TICKERS` | `["SPY"]` | HMM context only — never traded |
| `PER_TRADE_RISK_CAP` | `0.01` | Max 1% of NAV per trade |
| `MAX_POSITIONS` | `5` | Concurrent open positions cap |
| `INTRADAY_STOP_HALT` | `-0.03` | Daily halt at -3% drawdown |
| `WEEKLY_STOP` | `-0.05` | Weekly resize at -5% drawdown |
| `PEAK_DRAWDOWN_LOCKOUT` | `-0.10` | Full lockout at -10% from peak |
| `LIVE_ACCOUNT_MODE` | `False` | Enables tighter live-mode constraints |
| `IS_EQUITY_HOURS_ONLY` | `True` | Gate equity orders to market hours |
| `BTC_TICKERS` | `["BTCUSD"]` | Alpaca crypto symbol |
| `BTC_MAX_ALLOCATION` | `0.75` | Hard cap on BTC exposure |
| `CYCLE_COMPOSITE_THRESHOLD` | `0.65` | Cycle score required for tier boost |

For full technical documentation including signal detection, cycle logic, and safety protocols, see [regime_trader.md](regime_trader.md).

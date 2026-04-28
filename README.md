# regime_trader

A Python algorithmic trading bot that uses a Hidden Markov Model to classify
market regimes and adapts its strategy accordingly.

## Regimes

| Label | Regime   | Behaviour                                        |
|-------|----------|--------------------------------------------------|
| 0     | Crash    | Flat / cash, optional short via inverse ETF      |
| 1     | Bear     | Reduced long, short-biased, defensive sectors    |
| 2     | Neutral  | Moderate long, mean-reversion entries            |
| 3     | Bull     | Full long, momentum entries                      |
| 4     | Euphoria | Profit-taking, tight stops, no new longs         |

## Quick start

```bash
# 1. Clone and enter the project
git clone <repo-url> && cd regime_trader

# 2. Create a virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the credentials template and fill in your values
cp .env.example .env
# Edit .env — never commit it

# 5. Run tests (all must pass before live use)
pytest tests/ -v

# 6. Start the bot (paper trading recommended first)
python main.py

# 7. Launch the dashboard (separate terminal)
streamlit run dashboard/app.py
```

## Safety rules

- **Lockfile**: if the bot writes `TRADING_HALTED.lock`, delete it manually
  after investigating the drawdown.  Do not script its deletion.
- **Credentials**: `.env` is in `.gitignore`.  Never add credentials to code.
- **Backtesting**: always run `backtester.py` on a new configuration before
  enabling live trading.

## Project structure

```
regime_trader/
├── config/
│   ├── settings.py           # Tunable parameters
│   └── credentials.py        # .env loader (no hardcoded secrets)
├── core/
│   ├── hmm_engine.py         # HMM regime classifier
│   ├── feature_engineering.py
│   ├── regime_strategies.py  # Per-regime logic (main customization surface)
│   ├── risk_manager.py       # Circuit breakers, position sizing
│   ├── order_executor.py     # Broker order submission
│   ├── position_tracker.py   # Open positions and P&L
│   ├── market_data.py        # Real-time and historical feeds
│   ├── backtester.py         # Walk-forward backtest engine
│   ├── performance.py        # Metrics and reporting
│   └── alerts.py             # Email / webhook notifications
├── broker/
│   └── alpaca_client.py      # Alpaca REST + WebSocket wrapper
├── dashboard/
│   └── app.py                # Streamlit UI
├── tests/                    # One test file per core module
├── main.py                   # Orchestrator
├── requirements.txt
├── .env.example
└── .gitignore
```

## Development phases

1. **Skeleton** (current): file structure and docstrings only.
2. **Feature engineering + HMM**: implement and unit-test before proceeding.
3. **Backtester + performance**: validate no-lookahead guarantees.
4. **Risk manager**: all circuit breakers, lockfile logic.
5. **Broker integration**: Alpaca paper trading.
6. **Strategies**: per-regime logic, tuning.
7. **Dashboard**: Streamlit UI.
8. **Live trading**: after passing all phase tests.

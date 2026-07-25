"""
scripts/fetch_btc_1h.py
------------------------
Pull 1-hour BTC/USD OHLCV bars from Alpaca and write a clean btc_1h.csv for
backtesting. Uses existing Alpaca credentials (never hardcoded) via the repo's
load_credentials(), plus the OS-trust-store fix this host's TLS proxy needs.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, r"c:\Users\bollr\regime_trader")

from config.credentials import enable_os_trust_store, load_credentials
enable_os_trust_store()

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

SYMBOL   = "BTC/USD"
OUT_PATH = Path("btc_1h.csv")

creds  = load_credentials()
client = CryptoHistoricalDataClient(creds.api_key, creds.api_secret)

end   = datetime.now(timezone.utc)
start = end - timedelta(days=730)  # last 2 years (or max available if less)

resp = client.get_crypto_bars(
    CryptoBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame(1, TimeFrameUnit.Hour),
        start=start,
        end=end,
    )
)

df = resp.df
if df.empty:
    print("No bars returned — check credentials / symbol / date range.")
    sys.exit(1)

# resp.df is multi-indexed (symbol, timestamp) → flatten to a timestamp column.
df = df.reset_index()
df["date"] = df["timestamp"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")

df = df[["date", "open", "high", "low", "close", "volume"]]
df = df.sort_values("date", ascending=True)          # oldest first
df = df.drop_duplicates(subset="date", keep="first")  # no duplicate timestamps
df = df.reset_index(drop=True)

df.to_csv(OUT_PATH, index=False)

dupes = int(df["date"].duplicated().sum())
print(f"Saved {OUT_PATH.resolve()}")
print(f"Total bars : {len(df)}")
print(f"Date range : {df['date'].iloc[0]}  ->  {df['date'].iloc[-1]}")
print(f"Duplicate timestamps: {dupes}  ({'OK — none' if dupes == 0 else 'WARNING'})")
print("\nFirst 3 rows:")
print(df.head(3).to_string(index=False))
print("\nLast 3 rows:")
print(df.tail(3).to_string(index=False))

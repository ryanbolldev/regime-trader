"""
core/onchain_data.py
--------------------
Bitcoin on-chain metrics from three free APIs (no key required):
  - Blockchain.com stats
  - Mempool.space
  - CoinGecko free tier

Results are cached for ONCHAIN_CACHE_SECONDS (default 300 s).
On any fetch failure the last cached value is returned, or a neutral
default (on_chain_score=0.0) when no cache exists yet.

Public interface:
  get_onchain_features() -> OnChainFeatures
  fire_signal_if_threshold(score, prev_score) -> None
"""

from __future__ import annotations

import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import requests

from config.settings import ONCHAIN_CACHE_SECONDS

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API base URLs (module-level so tests can monkeypatch)
# ---------------------------------------------------------------------------

BLOCKCHAIN_INFO_URL = "https://api.blockchain.info/stats"
MEMPOOL_SPACE_BASE  = "https://mempool.space/api"
COINGECKO_BASE      = "https://api.coingecko.com/api/v3"

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OnChainFeatures:
    btc_price_usd:        float
    btc_24h_change_pct:   float
    btc_24h_volume_usd:   float
    btc_market_cap_usd:   float
    mempool_fee_pressure: float  # fastestFee / minimumFee
    hash_rate_3d:         float  # avg hash rate over last 3 days, H/s
    block_height:         int
    tx_volume_usd:        float  # estimated daily tx volume, USD
    on_chain_score:       float  # composite signal, clamped to [-1.0, +1.0]

# ---------------------------------------------------------------------------
# Module-level cache and prev-score state
# ---------------------------------------------------------------------------

_cache_lock:   threading.Lock            = threading.Lock()
_cache_result: Optional[OnChainFeatures] = None
_cache_time:   float                     = 0.0
_prev_score:   Optional[float]           = None

# ---------------------------------------------------------------------------
# Individual source fetchers
# ---------------------------------------------------------------------------

def _fetch_coingecko() -> dict:
    resp = requests.get(
        f"{COINGECKO_BASE}/simple/price",
        params={
            "ids":                 "bitcoin",
            "vs_currencies":       "usd",
            "include_24hr_change": "true",
            "include_24hr_vol":    "true",
            "include_market_cap":  "true",
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["bitcoin"]


def _fetch_mempool() -> dict:
    fees_resp = requests.get(f"{MEMPOOL_SPACE_BASE}/fees/recommended", timeout=10)
    fees_resp.raise_for_status()

    hash_resp = requests.get(f"{MEMPOOL_SPACE_BASE}/v1/mining/hashrate/3d", timeout=10)
    hash_resp.raise_for_status()

    height_resp = requests.get(f"{MEMPOOL_SPACE_BASE}/blocks/tip/height", timeout=10)
    height_resp.raise_for_status()

    return {
        "fees":     fees_resp.json(),
        "hashrate": hash_resp.json(),
        "height":   int(height_resp.text.strip()),
    }


def _fetch_blockchain_info() -> dict:
    resp = requests.get(BLOCKCHAIN_INFO_URL, timeout=10)
    resp.raise_for_status()
    return resp.json()

# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def _compute_score(
    price_change_pct: float,
    fee_pressure:     float,
    hash_rate:        float,
    tx_volume_usd:    float,
) -> float:
    """Composite on-chain signal in [-1.0, +1.0].

    Positive → bullish (rising hash rate, high tx volume, low fees, positive price).
    Negative → bearish (falling hash rate, low tx volume, high fees, negative price).
    """
    # Price momentum: tanh maps % changes smoothly; ±5 % → ±0.76
    price_score = math.tanh(price_change_pct / 5.0)

    # Fee pressure: ratio ≤ 1.5 is healthy (+0.75), ratio ≥ 9 is congested (−0.75)
    fee_score = max(-1.0, min(1.0, (3.0 - fee_pressure) / 4.0))

    # Hash rate relative to ~400 EH/s baseline (H/s)
    _HASH_BASELINE = 400e18
    hash_score = math.tanh((hash_rate - _HASH_BASELINE) / max(_HASH_BASELINE, 1.0))

    # Tx volume relative to ~$10 B/day baseline
    _VOL_BASELINE = 10e9
    vol_score = math.tanh((tx_volume_usd - _VOL_BASELINE) / max(_VOL_BASELINE, 1.0))

    composite = (
        0.40 * price_score
        + 0.25 * fee_score
        + 0.20 * vol_score
        + 0.15 * hash_score
    )
    return max(-1.0, min(1.0, composite))

# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------

def _last_hashrate_from_data(data: dict) -> float:
    """Extract the most recent hash rate from a mempool hashrate response."""
    current = data.get("currentHashrate")
    if current is not None:
        return float(current)
    hashrates = data.get("hashrates", [])
    if hashrates:
        return float(hashrates[-1].get("avgHashrate", 0.0))
    return 0.0


def _build_features(cg: dict, mp: dict, bc: dict) -> OnChainFeatures:
    price      = float(cg.get("usd", 0.0))
    change_pct = float(cg.get("usd_24h_change", 0.0))
    vol_24h    = float(cg.get("usd_24h_vol", 0.0))
    mkt_cap    = float(cg.get("usd_market_cap", 0.0))

    fees         = mp.get("fees", {})
    fastest_fee  = float(fees.get("fastestFee", 50))
    minimum_fee  = float(fees.get("minimumFee", 1) or 1)
    fee_pressure = fastest_fee / minimum_fee

    hash_rate    = _last_hashrate_from_data(mp.get("hashrate", {}))
    block_height = int(mp.get("height", 0))

    tx_vol = float(bc.get("estimated_transaction_volume_usd", 0.0))

    score = _compute_score(change_pct, fee_pressure, hash_rate, tx_vol)

    return OnChainFeatures(
        btc_price_usd        = price,
        btc_24h_change_pct   = change_pct,
        btc_24h_volume_usd   = vol_24h,
        btc_market_cap_usd   = mkt_cap,
        mempool_fee_pressure  = fee_pressure,
        hash_rate_3d         = hash_rate,
        block_height         = block_height,
        tx_volume_usd        = tx_vol,
        on_chain_score       = score,
    )

# ---------------------------------------------------------------------------
# Concurrent fetch
# ---------------------------------------------------------------------------

def _fetch_and_assemble() -> OnChainFeatures:
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_cg = ex.submit(_fetch_coingecko)
        f_mp = ex.submit(_fetch_mempool)
        f_bc = ex.submit(_fetch_blockchain_info)
        cg = f_cg.result(timeout=15)
        mp = f_mp.result(timeout=15)
        bc = f_bc.result(timeout=15)
    return _build_features(cg, mp, bc)

# ---------------------------------------------------------------------------
# Neutral default
# ---------------------------------------------------------------------------

def _neutral_default() -> OnChainFeatures:
    return OnChainFeatures(
        btc_price_usd        = 0.0,
        btc_24h_change_pct   = 0.0,
        btc_24h_volume_usd   = 0.0,
        btc_market_cap_usd   = 0.0,
        mempool_fee_pressure  = 1.0,
        hash_rate_3d         = 0.0,
        block_height         = 0,
        tx_volume_usd        = 0.0,
        on_chain_score       = 0.0,
    )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_BULLISH_THRESHOLD = 0.5
_BEARISH_THRESHOLD = -0.5


def get_onchain_features() -> OnChainFeatures:
    """Return current Bitcoin on-chain metrics, cached for ONCHAIN_CACHE_SECONDS.

    Never raises — returns stale cache or neutral default on any error.
    Fires an ONCHAIN_SIGNAL alert when the composite score crosses ±0.5.
    """
    global _cache_result, _cache_time, _prev_score

    with _cache_lock:
        now = time.monotonic()
        if _cache_result is not None and (now - _cache_time) < ONCHAIN_CACHE_SECONDS:
            log.debug("on-chain cache hit (age=%.0fs)", now - _cache_time)
            return _cache_result

    try:
        result = _fetch_and_assemble()
    except Exception as exc:
        log.warning("on-chain fetch failed: %s — returning cached/default", exc)
        with _cache_lock:
            if _cache_result is not None:
                return _cache_result
        return _neutral_default()

    # Cross-threshold alert check (only on fresh fetches)
    if _prev_score is not None:
        fire_signal_if_threshold(result.on_chain_score, _prev_score)
    _prev_score = result.on_chain_score

    with _cache_lock:
        _cache_result = result
        _cache_time   = time.monotonic()

    return result


def fire_signal_if_threshold(score: float, prev_score: float) -> None:
    """Fire an ONCHAIN_SIGNAL alert when score crosses ±0.5.

    Crossing above +0.5 → bullish signal.
    Crossing below -0.5 → bearish signal.
    """
    try:
        from core import alerts
    except Exception:
        return

    if prev_score < _BULLISH_THRESHOLD <= score:
        msg = f"On-chain signal: BULLISH (score={score:.3f})"
        log.info(msg)
        alerts.send("onchain_signal", msg, "info")
    elif prev_score > _BEARISH_THRESHOLD >= score:
        msg = f"On-chain signal: BEARISH (score={score:.3f})"
        log.info(msg)
        alerts.send("onchain_signal", msg, "warning")

"""``/strategies/crypto/*`` — crypto microstructure signal panel.

Read-only summary of the ``cryptostuff/`` engine, which is a separate process
the user runs locally (``cd cryptostuff && python run.py``) and which streams
Binance WebSocket trade + bookTicker for 10 pairs and computes 9 quant signals.

This router fetches Binance public REST snapshots (24h price/volume +
bookTicker) so the UI can show LIVE prices, bid-ask spreads, and OBI-ish
metrics WITHOUT needing the WS engine running. Aggressive caching (30s) so
the UI panel feels real-time without thrashing Binance.

Endpoints
---------
``GET /strategies/crypto/snapshot``  — live 10-pair quotes from Binance REST
``GET /strategies/crypto/signals``   — catalogue of the 9 microstructure signals
``GET /strategies/crypto/spec``      — how to run the WS engine locally
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/strategies/crypto", tags=["strategies-crypto"])

#: Same 10 pairs the ``cryptostuff`` engine streams.
PAIRS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "MATICUSDT",
    "DOGEUSDT",
    "LINKUSDT",
]

#: Cached snapshot. Refreshed at most every ``_TTL`` seconds.
_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_TTL: float = 30.0

_SIGNAL_CATALOG = [
    {
        "id": "vwap",
        "name": "VWAP (1m / 5m / 30m)",
        "method": "sum(price × qty) / sum(qty), rolling",
        "output": "Price level",
        "use": "Anchor for mean-reversion alerts.",
    },
    {
        "id": "vwap_z",
        "name": "VWAP Z-score",
        "method": "(price − vwap) / std(prices) over 30m",
        "output": "Mean-reversion alert when |z| > 2",
        "use": "Fade extreme moves.",
    },
    {
        "id": "rv",
        "name": "Realized Volatility",
        "method": "std(log returns) over 5m / 15m",
        "output": "σ_realised",
        "use": "Size positions inverse to RV; trigger regime alerts.",
    },
    {
        "id": "signed_volume",
        "name": "Signed Volume",
        "method": "+qty if buyer-taker, −qty if seller-taker",
        "output": "Order-flow pressure",
        "use": "Distinguish accumulation from distribution.",
    },
    {
        "id": "ofi",
        "name": "Order Flow Imbalance",
        "method": "signed_volume / total_volume over 1m",
        "output": "OFI ∈ [−1, +1]",
        "use": "Short-horizon directional signal.",
    },
    {
        "id": "obi",
        "name": "Order Book Imbalance",
        "method": "(bid_qty − ask_qty) / (bid_qty + ask_qty), top-N levels",
        "output": "OBI ∈ [−1, +1]",
        "use": "Microstructure pressure ahead of trade prints.",
    },
    {
        "id": "whale",
        "name": "Whale Detection",
        "method": "notional ≥ P99 threshold per symbol",
        "output": "Alert (side + magnitude)",
        "use": "Spot informed flow / large prints.",
    },
    {
        "id": "spread_bps",
        "name": "Spread (bps)",
        "method": "(ask − bid) / midprice × 10000",
        "output": "Liquidity measure",
        "use": "Gate trading: widen ⇒ avoid; tighten ⇒ engage.",
    },
    {
        "id": "midprice",
        "name": "Midprice",
        "method": "(bid + ask) / 2",
        "output": "Fair value",
        "use": "Reference for every other signal.",
    },
]


async def _fetch_pair(client: httpx.AsyncClient, symbol: str) -> dict[str, Any]:
    """Fetch 24h stats + bookTicker for one pair from Binance REST."""
    try:
        # Concurrent fetch — book + 24h ticker together.
        r_book, r_24h = await asyncio.gather(
            client.get(
                "https://api.binance.com/api/v3/ticker/bookTicker",
                params={"symbol": symbol},
                timeout=4.0,
            ),
            client.get(
                "https://api.binance.com/api/v3/ticker/24hr",
                params={"symbol": symbol},
                timeout=4.0,
            ),
        )
        if r_book.status_code != 200 or r_24h.status_code != 200:
            return {"symbol": symbol, "error": f"HTTP {r_book.status_code}/{r_24h.status_code}"}
        bt = r_book.json()
        st = r_24h.json()
        bid = float(bt["bidPrice"])
        ask = float(bt["askPrice"])
        bid_qty = float(bt["bidQty"])
        ask_qty = float(bt["askQty"])
        mid = (bid + ask) / 2.0
        spread_bps = ((ask - bid) / mid * 10_000) if mid else None
        obi_top1 = ((bid_qty - ask_qty) / (bid_qty + ask_qty)) if (bid_qty + ask_qty) > 0 else 0.0
        last_price = float(st["lastPrice"])
        chg_pct = float(st["priceChangePercent"])
        volume = float(st["quoteVolume"])
        return {
            "symbol": symbol,
            "last_price": last_price,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread_bps": spread_bps,
            "obi_top1": obi_top1,
            "bid_qty": bid_qty,
            "ask_qty": ask_qty,
            "change_24h_pct": chg_pct,
            "quote_volume_24h": volume,
        }
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        logger.warning("crypto snapshot: %s failed: %s", symbol, exc)
        return {"symbol": symbol, "error": str(exc)[:120]}


@router.get("/snapshot", summary="Live 10-pair microstructure snapshot (Binance REST)")
async def get_snapshot(request: Request) -> dict[str, Any]:
    """Return live midprice/spread/OBI/24h-change for all 10 pairs.

    Cached for 30s in-process so a packed UI panel polling every 5s only hits
    Binance ~2× per minute.

    Also opportunistically feeds the crypto5min spot-buffer for every pair —
    so calling ``/5min/predict/SOLUSDT`` works even when the dedicated
    background sampler (``PFM_CRYPTO_5MIN_ENABLED=1``) isn't running.
    """
    cached = _CACHE.get("snapshot")
    now = time.time()
    if cached and now - cached[0] < _TTL:
        return cached[1]
    client_obj = getattr(request.app.state, "async_http", None)
    if client_obj is None:
        # Stand-alone fallback so the endpoint still works when called outside
        # the configured FastAPI lifespan (e.g. one-off test client).
        async with httpx.AsyncClient(timeout=4.0) as tmp:
            results = await asyncio.gather(*(_fetch_pair(tmp, p) for p in PAIRS))
    else:
        results = await asyncio.gather(*(_fetch_pair(client_obj, p) for p in PAIRS))

    # Opportunistically record samples so the 10-pair spot-buffer stays warm
    # for ``/5min/predict`` calls on any pair even without the dedicated
    # background sampler running. Best-effort; never fails the response.
    try:
        from pfm.crypto5min.state import get_state as _get_5m_state

        _state = _get_5m_state()
        ts = float(now)
        for r in results:
            mid = r.get("mid") if isinstance(r, dict) else None
            if isinstance(mid, (int, float)) and mid > 0:
                _state.record_spot(r["symbol"], ts, float(mid))
    except Exception:
        pass

    # WS engine state — can be detected when running in-process via
    # ``crypto_events_engine.get_engine().is_running()``. Surfaces the same
    # bool the UI uses to swap the "engine off" banner for live event rows.
    engine_running = False
    try:
        from pfm.crypto_events_engine import get_engine as _crypto_get_engine

        engine_running = bool(_crypto_get_engine().is_running())
    except Exception:
        engine_running = False

    payload = {
        "as_of_unix": int(now),
        "pairs": results,
        "engine_running": engine_running,
    }
    _CACHE["snapshot"] = (now, payload)
    return payload


@router.get(
    "/signals", summary="Catalogue of the 9 microstructure signals computed by the WS engine"
)
def get_signals() -> dict[str, Any]:
    """Static reference card — the signal taxonomy the WS engine produces."""
    return {
        "n_signals": len(_SIGNAL_CATALOG),
        "n_pairs": len(PAIRS),
        "pairs": PAIRS,
        "signals": _SIGNAL_CATALOG,
    }


@router.get("/events", summary="Live whale + mean-reversion events from the WS engine (last N min)")
def get_events(
    request: Request,
    window_min: float = 5.0,
    symbol: str | None = None,
    kinds: str | None = None,
) -> dict[str, Any]:
    """Return event-class signals captured by the in-process WS engine.

    Source priority:
    1. **Local engine** when this worker is the leader (has the in-memory
       deque buffer from the WS stream).
    2. **Redis** when a different gunicorn worker is the leader — events
       are published to a sorted set; followers do a range query.
    3. ``engine_running=false`` payload when neither source is available.

    Events are *significant*: whales (notional ≥ P99 per symbol) and VWAP
    mean-reversion triggers (|z| > 2). Continuous signals like per-tick
    VWAP / OBI / spread live in ``/strategies/crypto/snapshot`` instead.
    """
    from pfm.crypto_events_engine import get_engine, read_events_from_redis

    kinds_set = {k.strip() for k in kinds.split(",") if k.strip()} if kinds else None
    eng = get_engine()
    status = eng.status()

    # 1) Local leader — read in-memory deque
    if status.get("running"):
        events = eng.events(symbol=symbol, window_min=window_min, kinds=kinds_set)
        return {
            "engine_running": True,
            "source": "local",
            "window_min": window_min,
            "n_events": len(events),
            "events": events,
            "status": status,
        }

    # 2) Follower — read from Redis (populated by the leader worker)
    try:
        cache = getattr(request.app.state, "cache", None)
        client = getattr(cache, "_client", None) if cache is not None else None
    except Exception:
        client = None
    if client is not None:
        events = read_events_from_redis(
            client,
            symbol=symbol,
            window_min=window_min,
            kinds=kinds_set,
        )
        # If Redis has events, the leader is alive somewhere; surface that
        # so the UI doesn't claim "engine off" when it's actually running
        # in another worker.
        if events:
            return {
                "engine_running": True,
                "source": "redis",
                "window_min": window_min,
                "n_events": len(events),
                "events": events,
                "status": status,
            }

    # 3) No engine + no Redis data — return the helpful empty payload.
    return {
        "engine_running": False,
        "source": "none",
        "events": [],
        "status": status,
        "hint": (
            "WS engine is off. Set `PFM_CRYPTO_WS_ENABLED=1` and restart "
            "to capture live whale / mean-reversion events from Binance. "
            "While off, /strategies/crypto/snapshot still works."
        ),
    }


async def _fetch_historical_sigma(
    client: httpx.AsyncClient,
    symbol: str,
    days: int = 30,
) -> float | None:
    """Annualized σ from Binance 1d klines (closes), last N days."""
    try:
        r = await client.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": "1d", "limit": days},
            timeout=4.0,
        )
        if r.status_code != 200:
            return None
        rows = r.json()
        if len(rows) < 10:
            return None
        import math

        closes = [float(row[4]) for row in rows]
        rets = [
            math.log(closes[i] / closes[i - 1])
            for i in range(1, len(closes))
            if closes[i] > 0 and closes[i - 1] > 0
        ]
        if len(rets) < 5:
            return None
        mu = sum(rets) / len(rets)
        var = sum((r - mu) ** 2 for r in rets) / max(1, len(rets) - 1)
        return (var**0.5) * (365**0.5)
    except (httpx.HTTPError, ValueError, KeyError):
        return None


_VOL_CACHE: dict[str, tuple[float, float]] = {}  # symbol -> (ts, sigma)
_VOL_TTL = 60 * 60  # 1h — daily-close vol doesn't change fast


@router.get(
    "/model-state/{symbol}",
    summary="Live cryptostuff signals + annualized σ for the GBM model-prob calc",
)
async def get_model_state(symbol: str, request: Request) -> dict[str, Any]:
    """Expose the engine's live state for one symbol.

    The frontend uses this to compute *model probability* for Polymarket
    BTC/ETH strike markets: ``P(S_T > K) = 1 − Φ((ln(K/S) − μT) / (σ√T))``
    with ``σ`` from the WS engine's realized vol (annualized) instead of a
    hard-coded guess, and ``μ`` from the order-flow imbalance.

    Returns ``{available: false}`` if the engine is off or the symbol hasn't
    seen enough trades yet (needs ~30s of warmup). Unknown symbols (not in
    the curated 10-pair list) return 404 instead of a misleading 200 with
    null fields — caught by the QA agent.
    """
    # Validate against the known pair registry. Case-insensitive so callers
    # can pass either ``btcusdt`` or ``BTCUSDT``.
    sym_upper = symbol.upper()
    if sym_upper not in PAIRS:
        raise HTTPException(
            status_code=404,
            detail=(f"unknown crypto pair {symbol!r}; supported pairs: " + ", ".join(PAIRS)),
        )
    symbol = sym_upper
    from pfm.crypto_events_engine import get_engine

    eng = get_engine()
    state: dict[str, Any] | None = None
    engine_running = eng.is_running()
    if engine_running:
        state = eng.model_state(symbol)
    # Historical σ from Binance daily closes — cached 1h. This is the σ the
    # frontend SHOULD use for multi-day GBM strike probabilities; cryptostuff's
    # tick-derived σ is informative for short-horizon vol but doesn't annualize
    # cleanly to a daily/weekly horizon.
    sym = symbol.upper()
    now = time.time()
    cached = _VOL_CACHE.get(sym)
    sigma_hist: float | None = None
    if cached and (now - cached[0]) < _VOL_TTL:
        sigma_hist = cached[1]
    else:
        client_obj = getattr(request.app.state, "async_http", None)
        if client_obj is None:
            async with httpx.AsyncClient(timeout=4.0) as tmp:
                sigma_hist = await _fetch_historical_sigma(tmp, sym)
        else:
            sigma_hist = await _fetch_historical_sigma(client_obj, sym)
        if sigma_hist is not None:
            _VOL_CACHE[sym] = (now, sigma_hist)
    if not engine_running and sigma_hist is None:
        return {
            "available": False,
            "reason": "WS engine off and Binance kline fetch failed.",
        }
    out: dict[str, Any] = {
        "available": True,
        "sigma_historical_annual": sigma_hist,
        "engine_running": engine_running,
    }
    if state:
        out["state"] = state
        # OFI from the live engine is the directional bias; we hand the
        # frontend the recommended sigma + mu split.
        ofi = float(state.get("ofi_1m") or 0.0)
        out["mu_drift_annual"] = ofi * 0.30  # ±30%/yr at extreme ±1 OFI
        out["sigma_used_annual"] = sigma_hist  # honest: use the kline-derived σ
        out["sigma_source"] = "binance-klines-30d"
    else:
        out["sigma_used_annual"] = sigma_hist
        out["mu_drift_annual"] = 0.0
        out["sigma_source"] = "binance-klines-30d"
    return out


@router.get("/spec", summary="How to launch the WS engine locally + what to expect")
def get_spec() -> dict[str, Any]:
    """Plain instructions the UI panel can render verbatim."""
    return {
        "engine_running": False,
        "launch_command": "cd cryptostuff && pip install -e . && python run.py",
        "default_pairs": PAIRS,
        "default_streams": ["trade", "depth", "bookTicker", "kline_1s"],
        "what_it_does": (
            "Connects to Binance WebSocket, streams trade + order-book data for the "
            "10 pairs, computes the 9 microstructure signals listed in /strategies/"
            "crypto/signals, and prints live alerts. Order-book derived signals require "
            "Redis (the engine streams depth deltas into OrderBookState)."
        ),
        "live_snapshot_endpoint": "/strategies/crypto/snapshot",
        "signal_catalog_endpoint": "/strategies/crypto/signals",
    }

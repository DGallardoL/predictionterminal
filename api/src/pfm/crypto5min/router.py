"""FastAPI router for ``/strategies/crypto/5min/*``.

Four endpoints make up the surface area:

* ``GET /compare``            — **always-on side-by-side**: returns one row
  per (asset, window) in ``{BTC, ETH} × {5m, 15m}`` with ``model_prob_up``
  always populated and ``market_prob_up`` populated when Polymarket has an
  open market for that combo. This is what the UI table renders.
* ``GET /markets``            — Polymarket-only view: lists exactly the open
  ``btc-updown-*`` / ``eth-updown-*`` 5m & 15m markets, each paired with our
  model probability and the resulting edge + signal. Doesn't include
  combos where Polymarket is silent. Use ``/compare`` for the UI; this one
  is for downstream consumers that only care about live markets.
* ``GET /predict/{symbol}``   — pure-model probability for the *next* 5m
  boundary (regardless of Polymarket availability). Useful for headless
  monitoring and the UI's "we'd bet up/down" badge even when Polymarket
  doesn't have an open market.
* ``GET /diag``               — internal diagnostics: how many spot samples
  per pair, last sample age, last anchor anchor data. Helpful when the
  UI shows nothing and you want to know why.

The router shares the existing app-level ``async_http`` client when the
FastAPI lifespan has set it up; otherwise it spins up a one-shot client per
request so the endpoints still work in isolation (TestClient, smoke checks).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query, Request

from pfm.crypto5min.comparator import (
    DEFAULT_EDGE_THRESHOLD,
    MARKET_ANCHOR_WEIGHT,
    anchor_to_market,
    compare_market_vs_model,
)
from pfm.crypto5min.confidence import build_confidence_result
from pfm.crypto5min.market_fetcher import (
    SUPPORTED_ASSETS,
    ActiveMarket,
    discover_active_markets,
    fetch_binance_mid,
    fetch_binance_price_at,
    fetch_clob_midpoint,
)
from pfm.crypto5min.predictor import PredictorInputs, predict_for_window
from pfm.crypto5min.state import CryptoFiveMinState, get_state
from pfm.crypto5min.strike_scraper import get_strike_for_market

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/strategies/crypto/5min", tags=["strategies-crypto-5min"])

#: Polled-market response cache TTL. We re-discover + re-price markets at
#: most this often to keep CLOB load proportional to (1 call / TTL / market).
_MARKETS_TTL_SECONDS: float = 4.0
_markets_cache: dict[str, tuple[float, dict[str, Any]]] = {}


async def _client_for(request: Request) -> tuple[httpx.AsyncClient, bool]:
    """Return ``(client, owned)`` — owned=True means the caller must close it."""
    shared = getattr(request.app.state, "async_http", None)
    if isinstance(shared, httpx.AsyncClient):
        return shared, False
    return httpx.AsyncClient(timeout=5.0), True


#: Lazy module-level Redis client used by ``_live_engine_state`` when this
#: gunicorn worker is a *follower* — i.e. the WS engine runs in a different
#: process so the local engine handle has no state to return. ``None``
#: means "not yet probed" or "Redis unreachable on the last probe"; we
#: set it to a sentinel ``_REDIS_DISABLED`` after a hard failure so we
#: don't pay the connect cost on every poll.
_REDIS_STATE_CLIENT: Any = None
_REDIS_DISABLED = object()


def _get_redis_for_state_reads() -> Any | None:
    """Return a redis.Redis instance, or None if Redis is unreachable.

    Reads ``REDIS_URL`` from the environment with the same default as
    ``pfm.config.Settings`` (``redis://redis:6379/0``). The connection is
    cached on first success. On the first hard failure we cache a sentinel
    so repeated polls don't repeatedly time-out trying to reconnect.
    """
    global _REDIS_STATE_CLIENT
    if _REDIS_STATE_CLIENT is _REDIS_DISABLED:
        return None
    if _REDIS_STATE_CLIENT is not None:
        return _REDIS_STATE_CLIENT
    try:
        import os

        import redis  # type: ignore

        url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
        client = redis.Redis.from_url(url, socket_timeout=1.0)
        client.ping()
        _REDIS_STATE_CLIENT = client
        return client
    except Exception as exc:
        logger.debug("crypto5min follower: Redis unreachable (%s)", exc)
        _REDIS_STATE_CLIENT = _REDIS_DISABLED
        return None


def _reset_redis_state_client() -> None:
    """Test hook — drop the cached Redis client so the next call rebuilds it."""
    global _REDIS_STATE_CLIENT
    _REDIS_STATE_CLIENT = None


def _live_engine_state(symbol: str) -> dict[str, Any] | None:
    """Best-effort pull of the live cryptostuff engine state for ``symbol``.

    Resolution order:
      1. **Leader-local** read — ``crypto_events_engine.get_engine().model_state``.
         Only the gunicorn worker that won the WS leader election can serve
         this; for the others it returns ``None`` (the engine handle exists
         but ``_engine_obj`` is ``None``).
      2. **Redis fallback** — read ``pfm:crypto_engine:state:{SYMBOL}`` from
         Redis. The leader publishes here at 1 Hz via the engine's state
         publisher loop, with a 5-second TTL so a dead leader's keys
         self-evict and the follower cleanly falls back to ``None``.

    Returns ``None`` when neither source has data; the predictor still
    works in that case using σ_long only.
    """
    try:
        from pfm.crypto_events_engine import (
            get_engine as _crypto_get_engine,
        )
        from pfm.crypto_events_engine import (
            read_model_state_from_redis as _read_state,
        )
    except ImportError:
        return None
    # 1) Leader-local fast path.
    try:
        eng = _crypto_get_engine()
        if eng.is_running():
            local = eng.model_state(symbol)
            if local is not None:
                return local
    except Exception as exc:
        logger.debug("live engine state fetch failed for %s: %s", symbol, exc)
    # 2) Follower fallback via Redis.
    redis_client = _get_redis_for_state_reads()
    if redis_client is None:
        return None
    try:
        return _read_state(redis_client, symbol)
    except Exception as exc:
        logger.debug("redis state read failed for %s: %s", symbol, exc)
        return None


_SIGMA_CACHE: dict[str, tuple[float, float]] = {}
_SIGMA_TTL_SECONDS: float = 60 * 60  # 1h — daily-close σ moves slow.


async def _historical_sigma(client: httpx.AsyncClient, symbol: str) -> float | None:
    """30-day daily-close annualized σ for one Binance symbol, cached 1h."""
    sym = symbol.upper()
    now = time.time()
    cached = _SIGMA_CACHE.get(sym)
    if cached and (now - cached[0]) < _SIGMA_TTL_SECONDS:
        return cached[1]
    try:
        r = await client.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": sym, "interval": "1d", "limit": 30},
            timeout=4.0,
        )
    except httpx.HTTPError as exc:
        logger.debug("kline σ fetch failed for %s: %s", sym, exc)
        return None
    if r.status_code != 200:
        return None
    try:
        rows = r.json()
    except ValueError:
        return None
    if not isinstance(rows, list) or len(rows) < 10:
        return None
    import math

    closes: list[float] = []
    for row in rows:
        try:
            closes.append(float(row[4]))
        except (IndexError, TypeError, ValueError):
            continue
    if len(closes) < 10:
        return None
    rets: list[float] = []
    for i in range(1, len(closes)):
        a, b = closes[i - 1], closes[i]
        if a > 0 and b > 0:
            rets.append(math.log(b / a))
    if len(rets) < 5:
        return None
    mu = sum(rets) / len(rets)
    var = sum((x - mu) ** 2 for x in rets) / max(1, len(rets) - 1)
    sigma = math.sqrt(var) * math.sqrt(365.0)
    _SIGMA_CACHE[sym] = (now, sigma)
    return sigma


def _record_sample(state: CryptoFiveMinState, symbol: str, mid: float | None) -> None:
    if mid is None or mid <= 0:
        return
    state.record_spot(symbol, time.time(), mid)


async def _ensure_recent_sample(
    client: httpx.AsyncClient,
    state: CryptoFiveMinState,
    binance_symbol: str,
    max_age_seconds: float = 5.0,
) -> None:
    """Top up the spot buffer with a fresh Binance REST mid if it's stale."""
    latest = state.latest(binance_symbol)
    if latest is not None and (time.time() - latest[0]) < max_age_seconds:
        return
    mid = await fetch_binance_mid(client, binance_symbol)
    _record_sample(state, binance_symbol, mid)


async def _build_prediction_for_symbol(
    client: httpx.AsyncClient,
    state: CryptoFiveMinState,
    binance_symbol: str,
    window_minutes: int,
    *,
    sigma_long_override: float | None = None,
) -> dict[str, Any]:
    """Run the predictor for one (symbol, window) and return a JSON-ready dict."""
    await _ensure_recent_sample(client, state, binance_symbol)
    anchor = state.anchor(binance_symbol, period_seconds=window_minutes * 60)
    if anchor is None:
        raise HTTPException(
            status_code=503,
            detail=f"no spot samples for {binance_symbol}; let the buffer warm up",
        )
    sigma_long = sigma_long_override
    if sigma_long is None:
        sigma_long = await _historical_sigma(client, binance_symbol)
    live = _live_engine_state(binance_symbol)
    sigma_short = live.get("sigma_annual") if live else None
    ofi = float(live.get("ofi_1m") or 0.0) if live else 0.0
    z = live.get("z_vwap_30m") if live else None
    asset_code = binance_symbol.replace("USDT", "").upper()
    pred = predict_for_window(
        spot_t=anchor.spot_now,
        spot_0=anchor.spot_at_start,
        seconds_remaining=anchor.seconds_remaining,
        sigma_long_annual=sigma_long,
        sigma_short_annual=sigma_short,
        ofi_1m=ofi,
        z_vwap=z,
        asset=asset_code,
    )
    return {
        "binance_symbol": binance_symbol,
        "window_minutes": window_minutes,
        "anchor": {
            "spot_at_start": anchor.spot_at_start,
            "spot_now": anchor.spot_now,
            "start_unix": anchor.start_unix,
            "end_unix": anchor.end_unix,
            "seconds_remaining": anchor.seconds_remaining,
        },
        "live_engine_used": live is not None,
        "engine_state": live,
        "prediction": pred.as_dict(),
    }


@router.get(
    "/predict/{symbol}",
    summary="Pure-model P(up by end of next 5m/15m window) for one Binance pair",
)
async def predict_symbol(
    symbol: str,
    request: Request,
    window_minutes: int = Query(5, ge=1, le=60),
) -> dict[str, Any]:
    """Return our model's up-probability for ``symbol`` for the *next* boundary.

    No Polymarket dependency — useful as a headless signal source even when
    no short-dated crypto market is open. Live cryptostuff engine state
    (OFI / σ_short / z-VWAP) is layered on when available.
    """
    sym = symbol.upper()
    if not sym.endswith("USDT"):
        sym = sym + "USDT"
    if sym not in {m["binance_symbol"] for m in SUPPORTED_ASSETS.values()}:
        raise HTTPException(status_code=400, detail=f"unsupported symbol {symbol!r}")
    client, owned = await _client_for(request)
    state = get_state()
    try:
        result = await _build_prediction_for_symbol(client, state, sym, window_minutes)
    finally:
        if owned:
            await client.aclose()
    return result


async def _build_one_comparison(
    client: httpx.AsyncClient,
    state: CryptoFiveMinState,
    market: ActiveMarket,
    *,
    sigma_long_override: float | None = None,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
) -> dict[str, Any]:
    """Score one ActiveMarket: model probability + CLOB midpoint + signal."""
    # Fan-out: kline-σ + Binance spot + CLOB midpoint in parallel.
    sigma_task = (
        asyncio.create_task(_historical_sigma(client, market.binance_symbol))
        if sigma_long_override is None
        else None
    )
    spot_task = asyncio.create_task(fetch_binance_mid(client, market.binance_symbol))
    market_task = asyncio.create_task(fetch_clob_midpoint(client, market.up_token_id))
    sigma_long = (
        sigma_long_override if sigma_long_override is not None else await sigma_task  # type: ignore[misc]
    )
    spot_now = await spot_task
    if spot_now is not None:
        _record_sample(state, market.binance_symbol, spot_now)
    market_mid = await market_task

    anchor = state.anchor(market.binance_symbol, period_seconds=market.window_minutes * 60)
    if anchor is None:
        return {
            **market.as_dict(),
            "error": "no_spot_samples",
            "model_prob_up": None,
            "market_prob_up": market_mid,
        }

    live = _live_engine_state(market.binance_symbol)
    sigma_short = live.get("sigma_annual") if live else None
    ofi = float(live.get("ofi_1m") or 0.0) if live else 0.0
    z = live.get("z_vwap_30m") if live else None
    pred = predict_for_window(
        spot_t=anchor.spot_now,
        spot_0=anchor.spot_at_start,
        seconds_remaining=market.seconds_remaining,
        sigma_long_annual=sigma_long,
        sigma_short_annual=sigma_short,
        ofi_1m=ofi,
        z_vwap=z,
        asset=market.asset,
    )

    if market_mid is None:
        return {
            **market.as_dict(),
            "error": "no_market_midpoint",
            "anchor": {
                "spot_at_start": anchor.spot_at_start,
                "spot_now": anchor.spot_now,
            },
            "prediction": pred.as_dict(),
            "model_prob_up": pred.prob_up,
            "market_prob_up": None,
            "live_engine_used": live is not None,
        }

    cmp_result = compare_market_vs_model(
        slug=market.slug,
        asset=market.asset,
        window_minutes=market.window_minutes,
        market_prob_up=market_mid,
        prediction=pred,
        edge_threshold=edge_threshold,
    )
    return {
        **market.as_dict(),
        "anchor": {
            "spot_at_start": anchor.spot_at_start,
            "spot_now": anchor.spot_now,
        },
        "live_engine_used": live is not None,
        "engine_state": live,
        **cmp_result.as_dict(),
    }


@router.get(
    "/markets",
    summary="Live model-vs-market table for every open 5m/15m crypto market",
)
async def list_markets(
    request: Request,
    edge_threshold: float = Query(DEFAULT_EDGE_THRESHOLD, ge=0.0, le=0.5),
    assets: str | None = Query(None, description="CSV of assets, e.g. 'BTC,ETH'"),
    window_minutes_csv: str = Query("5,15", description="CSV of window sizes"),
    use_cache: bool = Query(True, description="Skip the in-memory cache for fresh data"),
) -> dict[str, Any]:
    """Discover & price every open short-dated crypto market.

    The response is cached for ``_MARKETS_TTL_SECONDS`` per (assets, windows)
    key to keep upstream load proportional to user count. Pass
    ``use_cache=false`` to force a refresh.
    """
    asset_list = [a.strip().upper() for a in (assets or "BTC,ETH").split(",") if a.strip()]
    try:
        windows = [int(w.strip()) for w in window_minutes_csv.split(",") if w.strip()]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"bad window_minutes_csv: {exc}") from exc
    if not windows:
        raise HTTPException(
            status_code=400, detail="window_minutes_csv must list at least one window"
        )
    cache_key = f"assets={','.join(sorted(asset_list))}|windows={','.join(map(str, sorted(windows)))}|edge={edge_threshold:.3f}"
    now = time.time()
    cached = _markets_cache.get(cache_key)
    if use_cache and cached and (now - cached[0]) < _MARKETS_TTL_SECONDS:
        cached_payload = dict(cached[1])
        cached_payload["from_cache"] = True
        return cached_payload

    client, owned = await _client_for(request)
    state = get_state()
    try:
        markets = await discover_active_markets(
            client,
            assets=asset_list,
            window_minutes_list=windows,
        )
        # Fan out predictions in parallel.
        comparisons = await asyncio.gather(
            *(
                _build_one_comparison(client, state, m, edge_threshold=edge_threshold)
                for m in markets
            ),
            return_exceptions=True,
        )
    finally:
        if owned:
            await client.aclose()

    cleaned: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for raw in comparisons:
        if isinstance(raw, Exception):
            errors.append({"error": str(raw)[:200]})
            continue
        cleaned.append(raw)
    payload: dict[str, Any] = {
        "as_of_unix": int(now),
        "n_markets": len(cleaned),
        "edge_threshold": edge_threshold,
        "assets": asset_list,
        "window_minutes": sorted(set(windows)),
        "markets": cleaned,
        "errors": errors,
        "from_cache": False,
    }
    _markets_cache[cache_key] = (now, payload)
    return payload


async def _build_one_compare_row(
    client: httpx.AsyncClient,
    state: CryptoFiveMinState,
    *,
    asset: str,
    window_minutes: int,
    binance_symbol: str,
    active_by_key: dict[tuple[str, int], ActiveMarket],
    edge_threshold: float,
    redis_client: Any | None = None,
) -> dict[str, Any]:
    """One side-by-side row for (asset, window).

    Always returns ``model_prob_up`` (assuming we have a spot sample);
    ``market_prob_up`` is null when Polymarket doesn't have an open market
    for this combo.

    ``redis_client`` enables the follower path inside
    :func:`fetch_clob_midpoint`: a follower worker (no in-process WS
    subscriber) can still read the leader's published midpoint from Redis
    instead of paying the REST round-trip. The leader passes ``None``
    because its in-process cache always wins.
    """
    # Fan-out: kline-σ + spot + (optional) CLOB midpoint in parallel.
    sigma_task = asyncio.create_task(_historical_sigma(client, binance_symbol))
    spot_task = asyncio.create_task(fetch_binance_mid(client, binance_symbol))
    market_obj = active_by_key.get((asset, window_minutes))
    market_mid_task: asyncio.Task[float | None] | None = None
    if market_obj is not None:
        market_mid_task = asyncio.create_task(
            fetch_clob_midpoint(
                client,
                market_obj.up_token_id,
                redis_client=redis_client,
            )
        )
    sigma_long = await sigma_task
    spot_now = await spot_task
    if spot_now is not None:
        _record_sample(state, binance_symbol, spot_now)
    market_mid = await market_mid_task if market_mid_task is not None else None

    # Compute the absolute end-of-window timestamp. When Polymarket has an
    # open market we use its end_unix directly; otherwise we anchor on the
    # next natural boundary derived from the current spot buffer's anchor.
    end_unix: int
    if market_obj is not None:
        end_unix = market_obj.end_unix
    else:
        # state.anchor() returns end_unix; we need it before checking for None.
        peek_anchor = state.anchor(binance_symbol, period_seconds=window_minutes * 60)
        end_unix = peek_anchor.end_unix if peek_anchor is not None else 0

    anchor = state.anchor(binance_symbol, period_seconds=window_minutes * 60)
    if anchor is None:
        # No spot sample yet (process just booted). Return placeholders so
        # the UI still has a row instead of an empty grid.
        return {
            "asset": asset,
            "binance_symbol": binance_symbol,
            "window_minutes": window_minutes,
            "model_prob_up": None,
            "market_prob_up": market_mid,
            "edge": None,
            "signal": "WAIT",
            "end_unix": end_unix or None,
            "seconds_remaining": (market_obj.seconds_remaining if market_obj is not None else None),
            "slug": market_obj.slug if market_obj is not None else None,
            "error": "no_spot_samples",
        }

    live = _live_engine_state(binance_symbol)
    sigma_short = live.get("sigma_annual") if live else None
    ofi = float(live.get("ofi_1m") or 0.0) if live else 0.0
    z = live.get("z_vwap_30m") if live else None
    # Use the *fresh* server-now when computing seconds_remaining so the
    # model uses the most accurate horizon possible. Anything more than ~1s
    # stale is mostly harmless except in the last few seconds of the window
    # where the GBM collapses to a step function.
    seconds_remaining_fresh = (
        max(0.0, end_unix - time.time()) if end_unix else (anchor.seconds_remaining)
    )
    seconds_remaining = seconds_remaining_fresh

    # Strike resolution. Fallback chain:
    #   1) polymarket.com scrape of priceToBeat (exact Chainlink reference,
    #      lags ~5-15 min after the boundary so it only covers older events)
    #   2) Binance aggTrades at exactly start_unix (Chainlink proxy, accurate
    #      to within ~$5 for BTC since Chainlink aggregates across exchanges)
    #   3) state.anchor()'s spot_at_start from our rolling buffer (only useful
    #      when the process has been running since before the boundary)
    start_unix_for_strike = end_unix - (window_minutes * 60) if end_unix else None
    strike_price: float | None = None
    strike_source: str = "binance-buffer"
    if start_unix_for_strike is not None:
        # Race the polymarket scrape against the Binance aggTrades fallback so
        # the slower of the two doesn't serialize after the faster one — the
        # scrape is preferred but the aggTrades round-trip is independent.
        # We still apply priority order after both settle.
        scrape_task = asyncio.create_task(
            get_strike_for_market(
                client,
                asset=asset,
                window_minutes=window_minutes,
                start_unix=start_unix_for_strike,
            )
        )
        aggtrades_task = asyncio.create_task(
            fetch_binance_price_at(client, binance_symbol, start_unix_for_strike)
        )
        scrape_result, aggtrades_result = await asyncio.gather(
            scrape_task,
            aggtrades_task,
            return_exceptions=True,
        )
        if isinstance(scrape_result, BaseException):
            logger.debug(
                "strike scrape failed for %s %dm: %s",
                asset,
                window_minutes,
                scrape_result,
            )
        else:
            scrape_price, scrape_src = scrape_result
            if scrape_price is not None:
                strike_price = scrape_price
                strike_source = scrape_src
        if strike_price is None:
            if isinstance(aggtrades_result, BaseException):
                logger.debug(
                    "binance aggTrades fallback failed for %s: %s",
                    binance_symbol,
                    aggtrades_result,
                )
            elif aggtrades_result is not None:
                strike_price = aggtrades_result
                strike_source = "binance-aggtrades"
    if strike_price is None:
        strike_price = anchor.spot_at_start
        strike_source = "binance-buffer"

    pred_inputs = PredictorInputs(
        spot_t=anchor.spot_now,
        spot_0=strike_price,
        seconds_remaining=seconds_remaining,
        sigma_long_annual=sigma_long,
        sigma_short_annual=sigma_short,
        ofi_1m=ofi,
        z_vwap=z,
        asset=asset,
    )
    pred = predict_for_window(
        spot_t=anchor.spot_now,
        spot_0=strike_price,
        seconds_remaining=seconds_remaining,
        sigma_long_annual=sigma_long,
        sigma_short_annual=sigma_short,
        ofi_1m=ofi,
        z_vwap=z,
        asset=asset,
    )

    # σ-jackknife confidence + z-scores. Always computed — for the
    # no-market case z_edge is None but z_model + confidence still land.
    n_samples = state.n_samples(binance_symbol)
    conf = build_confidence_result(
        base_inputs=pred_inputs,
        base_model_prob=pred.prob_up,
        market_prob=market_mid,
        sigma_used_annual=pred.sigma_used_annual,
        mu_used_annual=pred.mu_used_annual,
        n_samples=n_samples,
        live_engine_used=live is not None,
        window_seconds=window_minutes * 60,
    )

    # Apply market-anchor to the model_prob the UI sees. Pure GBM with
    # realistic σ over-shoots the market for short crypto windows (market
    # prices in mean-reversion + Chainlink lag that GBM doesn't model). The
    # anchored model = 0.80·market + 0.20·gbm produces small realistic
    # tilts. Raw GBM is kept in ``model_prob_gbm_raw`` for diagnostics.
    gbm_raw = pred.prob_up
    anchored_model = anchor_to_market(gbm_raw, market_mid)
    base = {
        "asset": asset,
        "binance_symbol": binance_symbol,
        "window_minutes": window_minutes,
        "model_prob_up": anchored_model,
        "model_prob_gbm_raw": gbm_raw,
        "market_anchor_weight": MARKET_ANCHOR_WEIGHT if market_mid is not None else 0.0,
        "market_prob_up": market_mid,
        "end_unix": end_unix or None,
        "seconds_remaining": seconds_remaining,
        "spot_at_start": anchor.spot_at_start,
        "spot_now": anchor.spot_now,
        "strike_price": strike_price,
        "strike_source": strike_source,
        "sigma_used_annual": pred.sigma_used_annual,
        "mu_used_annual": pred.mu_used_annual,
        "live_engine_used": live is not None,
        "n_spot_samples": n_samples,
        "slug": market_obj.slug if market_obj is not None else None,
        "has_polymarket_market": market_obj is not None,
        "polymarket_available": market_mid is not None,
        **conf.as_dict(),
    }
    if market_mid is None:
        # No market price — return the model side filled in and edge null.
        base.update(
            {
                "edge": None,
                "signal": "WAIT",
                "kelly_fraction": 0.0,
            }
        )
        return base
    cmp = compare_market_vs_model(
        slug=market_obj.slug if market_obj is not None else "",
        asset=asset,
        window_minutes=window_minutes,
        market_prob_up=market_mid,
        prediction=pred,
        edge_threshold=edge_threshold,
    )
    base.update(
        {
            "edge": cmp.edge,
            "signal": cmp.signal,
            "kelly_fraction": cmp.kelly_fraction,
        }
    )
    return base


_COMPARE_TTL_SECONDS: float = 1.0
_compare_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _build_compare_cache_key(
    asset_list: list[str], windows: list[int], edge_threshold: float
) -> str:
    return (
        f"assets={','.join(sorted(asset_list))}"
        f"|windows={','.join(map(str, sorted(windows)))}"
        f"|edge={edge_threshold:.3f}"
    )


async def build_compare_payload(
    client: httpx.AsyncClient,
    state: CryptoFiveMinState,
    *,
    assets: list[str],
    windows: list[int],
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    redis_client: Any | None = None,
) -> dict[str, Any]:
    """Pure async builder for the ``/compare`` response.

    Doesn't touch the cache — returns a fresh payload every call. Used by
    both the HTTP route (after a cache miss) and the background prewarmer
    that keeps the cache hot. Caller is responsible for the httpx client
    lifecycle.

    ``redis_client`` is forwarded to :func:`fetch_clob_midpoint` so the
    follower workers get sub-second cross-process midpoints published by
    the leader's WebSocket subscriber. ``None`` is safe (REST fallback).
    """
    now = time.time()
    active = await discover_active_markets(
        client,
        assets=assets,
        window_minutes_list=windows,
    )
    active_by_key: dict[tuple[str, int], ActiveMarket] = {
        (m.asset, m.window_minutes): m for m in active
    }
    rows_coros = [
        _build_one_compare_row(
            client,
            state,
            asset=a,
            window_minutes=w,
            binance_symbol=SUPPORTED_ASSETS[a]["binance_symbol"],
            active_by_key=active_by_key,
            edge_threshold=edge_threshold,
            redis_client=redis_client,
        )
        for a in assets
        for w in windows
    ]
    rows = await asyncio.gather(*rows_coros, return_exceptions=True)
    cleaned: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for raw in rows:
        if isinstance(raw, Exception):
            errors.append({"error": str(raw)[:200]})
            continue
        cleaned.append(raw)
    n_with_market = sum(1 for r in cleaned if r.get("market_prob_up") is not None)
    return {
        "as_of_unix": int(now),
        "n_rows": len(cleaned),
        "n_polymarket_active": n_with_market,
        "edge_threshold": edge_threshold,
        "assets": assets,
        "window_minutes": sorted(set(windows)),
        "rows": cleaned,
        "errors": errors,
        "from_cache": False,
    }


@router.get(
    "/compare",
    summary="Side-by-side model vs market for every BTC/ETH × 5m/15m combo",
)
async def compare_all(
    request: Request,
    edge_threshold: float = Query(DEFAULT_EDGE_THRESHOLD, ge=0.0, le=0.5),
    assets: str | None = Query(None, description="CSV of assets, default 'BTC,ETH'"),
    window_minutes_csv: str = Query("5,15", description="CSV of window sizes"),
    use_cache: bool = Query(True, description="Skip the in-memory cache for fresh data"),
) -> dict[str, Any]:
    """Always-on table of (asset × window) rows with both probabilities.

    Unlike ``/markets``, this endpoint **always returns a row for every
    combo** in the requested ``assets × window_minutes_csv`` product —
    even when Polymarket has no open market for it. ``market_prob_up`` is
    null in that case; ``model_prob_up`` is still computed from the GBM
    plus microstructure overlay.
    """
    asset_list = [a.strip().upper() for a in (assets or "BTC,ETH").split(",") if a.strip()]
    try:
        windows = [int(w.strip()) for w in window_minutes_csv.split(",") if w.strip()]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"bad window_minutes_csv: {exc}") from exc
    if not windows:
        raise HTTPException(
            status_code=400, detail="window_minutes_csv must list at least one window"
        )
    valid_assets = set(SUPPORTED_ASSETS)
    asset_list = [a for a in asset_list if a in valid_assets]
    if not asset_list:
        raise HTTPException(status_code=400, detail="no supported assets in `assets`")

    cache_key = _build_compare_cache_key(asset_list, windows, edge_threshold)
    now = time.time()
    cached = _compare_cache.get(cache_key)
    if use_cache and cached and (now - cached[0]) < _COMPARE_TTL_SECONDS:
        cached_payload = dict(cached[1])
        cached_payload["from_cache"] = True
        return cached_payload

    client, owned = await _client_for(request)
    state = get_state()
    # Followers (workers that didn't win the SETNX leader election) read
    # the leader's published midpoints from Redis to skip the REST hop.
    # Leaders pass ``None`` because their in-process cache always wins.
    cache = getattr(request.app.state, "cache", None)
    redis_client = getattr(cache, "_client", None) if cache is not None else None
    try:
        payload = await build_compare_payload(
            client,
            state,
            assets=asset_list,
            windows=windows,
            edge_threshold=edge_threshold,
            redis_client=redis_client,
        )
    finally:
        if owned:
            await client.aclose()
    _compare_cache[cache_key] = (now, payload)
    return payload


@router.get("/diag", summary="Spot-buffer diagnostics for the 5min predictor")
def diag() -> dict[str, Any]:
    """Internal health surface: per-symbol sample count + age."""
    state = get_state()
    return {
        "symbols": [state.snapshot(sym) for sym in state.all_symbols()],
        "now_unix": int(time.time()),
    }


def _reset_caches() -> None:
    """Test hook — clear in-memory caches so tests stay isolated."""
    _markets_cache.clear()
    _compare_cache.clear()
    _SIGMA_CACHE.clear()

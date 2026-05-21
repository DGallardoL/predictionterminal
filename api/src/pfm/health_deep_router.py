"""``GET /health/deep`` — parallel upstream-readiness probe.

The simple ``/health`` (in :mod:`pfm.main`) is a fast liveness probe that
just confirms the FastAPI process is up. The richer ``/health/detail``
endpoint (:mod:`pfm.health_router`) reports Redis + git SHA + auth posture.

This module adds a third readiness endpoint that pings the four upstream
data sources we depend on (Polymarket Gamma, Kalshi, yfinance, GDELT)
**in parallel** under a strict 5-second total budget, plus a Redis PING
when a cache backend is configured. It is intentionally tolerant — a
single slow or 5xx upstream marks that source as ``ok: false`` but never
fails the whole endpoint. The overall status is derived from the count
of failing sources:

* ``ok``       — every source responded successfully
* ``degraded`` — 1–2 sources down, more than half ok
* ``down``     — half or more sources down

Per-source shape::

    {
        "latency_ms": 82,
        "ok": true,
        "last_error": null,
        "checked_at": "2026-05-16T09:33:00Z"
    }

Used by ops dashboards and on-call to triage "is the issue our app or
an upstream?" in one curl.

Integration note (when main.py:routes is unclaimed):
    from pfm.health_deep_router import router as _health_deep_router
    app.include_router(_health_deep_router)
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter

router = APIRouter()


# Module-load timestamp — ``uptime_s`` in the response is wall-clock seconds
# since this module was first imported (close enough to process start for
# the gunicorn worker model that this codebase uses; the simple ``/health``
# does the same thing in :mod:`pfm.health_router`).
_PROCESS_START = time.time()

# Total budget for the entire /health/deep handler. Individual probes get a
# slightly smaller per-call timeout so a single slow upstream cannot consume
# the whole budget and starve the others. ``asyncio.gather`` runs them
# concurrently so wall-clock latency ≈ max(probe_latency) + small overhead.
# Frontend's connection-status.js marks the whole site "offline" when
# /health/deep responds in >5s (SLOW_CEILING_MS). Keep the total budget
# under that ceiling so a stale-cache miss doesn't paint the badge red.
# Cached probes (60s TTL) return instantly so steady-state is sub-100ms;
# the budget only matters for a cold start.
_TOTAL_BUDGET_S = 4.0
_PER_CALL_TIMEOUT_S = 3.5

# Upstream URLs. Hard-coded rather than pulled from
# :class:`pfm.config.Settings` because the deep probe should describe the
# **canonical** upstream regardless of whether the deployment has shimmed it.
POLYMARKET_URL = "https://gamma-api.polymarket.com/markets?limit=1"
KALSHI_URL = "https://api.elections.kalshi.com/trade-api/v2/markets?limit=1"
# GDELT's `/api/v2/doc/doc` endpoint frequently exceeds 5s. The lighter
# `/api/v2/tv/tv` shape (TV transcripts) responds in ~600ms with the same
# semantics for a liveness check.
GDELT_URL = (
    "https://api.gdeltproject.org/api/v2/tv/tv"
    "?query=test&format=json&mode=clipgallery&datanorm=perc"
    "&last24=yes&maxrecords=1"
)
YFINANCE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/SPY?range=1d&interval=1d"
# Stooq is a free, non-rate-limited public stock data source. Yahoo's chart
# endpoint frequently returns 429 even with full browser headers, but the
# probe is conceptually "stock-price upstream alive?" so Stooq satisfies it.
STOOQ_FALLBACK_URL = "https://stooq.com/q/d/l/?s=spy.us&i=d"

# Yahoo blocks default Python/httpx UAs with 429. A real browser UA gets
# through reliably on the v8/finance/chart endpoint.
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36"
)

# Per-probe cache: avoid hammering upstreams every poll cycle.
# Each source caches its last result; the TTL depends on whether the last
# result was ok (long cache, the upstream is healthy) or failed (short cache,
# retry soon but don't slow-loop on the failure).
_CACHE_TTL_OK_S = 60.0
_CACHE_TTL_FAIL_S = 15.0
_probe_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _cache_get(name: str) -> dict[str, Any] | None:
    entry = _probe_cache.get(name)
    if entry is None:
        return None
    ts, payload = entry
    ttl = _CACHE_TTL_OK_S if payload.get("ok") else _CACHE_TTL_FAIL_S
    if time.time() - ts > ttl:
        return None
    return payload


def _cache_put(name: str, payload: dict[str, Any]) -> None:
    _probe_cache[name] = (time.time(), payload)


def _iso_now() -> str:
    """UTC ISO8601 with second precision (matches the rest of the API)."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


async def _http_probe(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Fetch ``url`` and return a per-source health dict. Never raises."""
    start = time.perf_counter()
    t = timeout if timeout is not None else _PER_CALL_TIMEOUT_S
    try:
        resp = await client.get(url, timeout=t, headers=headers or {})
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        if 200 <= resp.status_code < 300:
            return {
                "latency_ms": latency_ms,
                "ok": True,
                "last_error": None,
                "checked_at": _iso_now(),
            }
        return {
            "latency_ms": latency_ms,
            "ok": False,
            "last_error": f"{resp.status_code} {resp.reason_phrase or 'HTTP error'}".strip(),
            "checked_at": _iso_now(),
        }
    except (TimeoutError, httpx.TimeoutException):
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return {
            "latency_ms": latency_ms,
            "ok": False,
            "last_error": "timeout",
            "checked_at": _iso_now(),
        }
    except httpx.HTTPError as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return {
            "latency_ms": latency_ms,
            "ok": False,
            "last_error": f"{type(exc).__name__}: {exc}"[:200],
            "checked_at": _iso_now(),
        }
    except Exception as exc:  # pragma: no cover - defensive
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return {
            "latency_ms": latency_ms,
            "ok": False,
            "last_error": f"{type(exc).__name__}: {exc}"[:200],
            "checked_at": _iso_now(),
        }


async def _probe_polymarket(client: httpx.AsyncClient) -> dict[str, Any]:
    cached = _cache_get("polymarket")
    if cached is not None:
        return cached
    result = await _http_probe(client, POLYMARKET_URL)
    _cache_put("polymarket", result)
    return result


async def _probe_kalshi(client: httpx.AsyncClient) -> dict[str, Any]:
    cached = _cache_get("kalshi")
    if cached is not None:
        return cached
    result = await _http_probe(client, KALSHI_URL)
    _cache_put("kalshi", result)
    return result


async def _probe_gdelt(client: httpx.AsyncClient) -> dict[str, Any]:
    cached = _cache_get("gdelt")
    if cached is not None:
        return cached
    # GDELT's edge regularly takes 5-7s on this endpoint. 8s gives us headroom
    # so we don't flap between OK and timeout. Once it succeeds we cache 60s.
    # GDELT is slow but non-critical. Cap at 3s so it doesn't blow the total
    # budget (4s); when it fails we'll just report degraded for one cycle
    # and the next 60s of probes serve from cache.
    result = await _http_probe(client, GDELT_URL, timeout=3.0)
    _cache_put("gdelt", result)
    return result


async def _probe_yfinance(client: httpx.AsyncClient) -> dict[str, Any]:
    """yfinance has no dedicated health URL; we hit the Yahoo chart API
    for a tiny SPY series. Tests mock this same URL.

    Yahoo blocks the default Python UA with HTTP 429; a real browser UA
    passes reliably. Cached for 60s so we don't poke Yahoo every cycle.
    """
    cached = _cache_get("yfinance")
    if cached is not None:
        return cached
    # Try Yahoo first with full browser headers. Yahoo aggressively 429s any
    # non-browser traffic — if it blocks us we fall back to Stooq, which
    # serves the same kind of data and never rate-limits. The probe label
    # stays "yfinance" because conceptually it's "stock-price source alive".
    result = await _http_probe(
        client,
        YFINANCE_URL,
        headers={
            "User-Agent": _BROWSER_UA,
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://finance.yahoo.com/",
            "Origin": "https://finance.yahoo.com",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
        },
    )
    if not result.get("ok"):
        fallback = await _http_probe(
            client,
            STOOQ_FALLBACK_URL,
            headers={"User-Agent": _BROWSER_UA},
        )
        if fallback.get("ok"):
            fallback["note"] = "yahoo 429 — using Stooq fallback"
            result = fallback
    _cache_put("yfinance", result)
    return result


async def _probe_redis() -> dict[str, Any]:
    """PING the configured Redis if ``REDIS_URL`` is set. Otherwise return
    a "not configured" entry so dashboards don't paint a stale red.
    """
    cached = _cache_get("redis")
    if cached is not None:
        return cached
    redis_url = os.environ.get("REDIS_URL")
    checked_at = _iso_now()
    if not redis_url:
        result = {
            "latency_ms": None,
            "ok": True,
            "last_error": None,
            "checked_at": checked_at,
            "note": "REDIS_URL not set",
        }
        _cache_put("redis", result)
        return result
    start = time.perf_counter()
    try:
        # Lazy-import — many test environments don't have redis-py wired up,
        # and the probe should degrade gracefully there too.
        import redis  # type: ignore[import-not-found]

        client = redis.from_url(redis_url, socket_timeout=_PER_CALL_TIMEOUT_S)
        # Run the (blocking) ping in a worker thread so we don't stall the
        # asyncio event loop. ``asyncio.to_thread`` requires py>=3.9.
        await asyncio.wait_for(
            asyncio.to_thread(client.ping),
            timeout=_PER_CALL_TIMEOUT_S,
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        result = {
            "latency_ms": latency_ms,
            "ok": True,
            "last_error": None,
            "checked_at": checked_at,
        }
        _cache_put("redis", result)
        return result
    except TimeoutError:
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return {
            "latency_ms": latency_ms,
            "ok": False,
            "last_error": "timeout",
            "checked_at": checked_at,
        }
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return {
            "latency_ms": latency_ms,
            "ok": False,
            "last_error": f"{type(exc).__name__}: {exc}"[:200],
            "checked_at": checked_at,
        }


# Critical sources drive the arb + regression flows and MUST be up for the
# product to function. yfinance and GDELT are best-effort: yfinance is for
# regression-mode stock prices (frequently 429-throttled by Yahoo) and GDELT
# is news enrichment (frequently slow). Their failure shouldn't tank status.
_CRITICAL_SOURCES = {"polymarket", "kalshi", "redis"}


def _overall_status(sources: dict[str, dict[str, Any]]) -> str:
    """Map per-source health to a single ok/degraded/down badge.

    Count-based rules (match the module docstring contract):
      * ``ok``       — every source ok
      * ``degraded`` — 1 source down (or any minority below half)
      * ``down``     — half-or-more sources down

    Critical-source weighting was tried but produced surprising results
    (a single Redis blip flagged the whole product as "down" even when
    every upstream was healthy). The count-based rule keeps the badge
    semantics legible: one upstream wobble → degraded, many → down.
    """
    total = len(sources)
    if total == 0:
        return "ok"
    down_count = sum(1 for s in sources.values() if not s.get("ok"))
    if down_count * 2 >= total:
        return "down"
    return "degraded" if down_count > 0 else "ok"


@router.get(
    "/health/deep",
    summary="Parallel upstream-readiness probe (Polymarket, Kalshi, yfinance, Redis, GDELT)",
    tags=["health"],
)
async def health_deep() -> dict[str, Any]:
    """Run all configured upstream pings in parallel and return a status
    snapshot. Total wall-clock budget is 5 seconds — slow upstreams get
    marked ``ok: false, last_error: 'timeout'`` rather than failing the
    whole endpoint.
    """
    async with httpx.AsyncClient(timeout=_PER_CALL_TIMEOUT_S) as client:
        probes = {
            "polymarket": _probe_polymarket(client),
            "kalshi": _probe_kalshi(client),
            "yfinance": _probe_yfinance(client),
            "redis": _probe_redis(),
            "gdelt": _probe_gdelt(client),
        }
        # Each probe has its own timeout (set by _http_probe + cached probes
        # return instantly). gather without outer wait_for means total wall
        # time = max(probe time), which is bounded by GDELT's 3s cap — well
        # under the frontend's 8s SLOW_CEILING. The previous outer wait_for
        # was cancelling fast probes mid-flight when the slowest one ran long.
        results = await asyncio.gather(*probes.values(), return_exceptions=True)

    sources: dict[str, dict[str, Any]] = {}
    for name, result in zip(probes.keys(), results, strict=False):
        if isinstance(result, BaseException):
            sources[name] = {
                "latency_ms": None,
                "ok": False,
                "last_error": f"{type(result).__name__}: {result}"[:200],
                "checked_at": _iso_now(),
            }
        else:
            sources[name] = result

    ok_count = sum(1 for s in sources.values() if s.get("ok"))
    total = len(sources)
    status = _overall_status(sources)
    return {
        "status": status,
        "uptime_s": round(time.time() - _PROCESS_START, 2),
        "sources": sources,
        "summary": f"{total} sources checked, {ok_count} ok",
    }

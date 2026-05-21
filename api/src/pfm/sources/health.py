"""Source-health probes for the data backends used by the API.

Each probe makes a single light-weight call (HEAD or a tiny GET) and
returns ``{ok: bool, latency_ms: float | None, detail: str | None}``.

The probes are deliberately defensive: a failed probe must never bubble
an exception out of :func:`check_all_sources`. If a source is gated on
configuration (``configured: False``) we report that without making the
network call.

Probes run **in parallel** via :func:`asyncio.gather` so the wall-clock
of ``check_all_sources`` is bounded by ``DEFAULT_TIMEOUT_S`` (≈4 s)
rather than the sum of every probe's RTT. The synchronous ``check_*``
helpers and ``check_all_sources`` are preserved for backwards compat
(they internally drive an event loop).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from pfm.sources import stooq as stooq_src
from pfm.sources import tiingo as tiingo_src

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 4.0


def _timed(
    fn: Callable[[httpx.Client], httpx.Response],
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> tuple[bool, float | None, str | None]:
    """Run ``fn`` with a fresh client and return (ok, latency_ms, detail)."""
    start = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout) as cli:
            resp = fn(cli)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        ok = 200 <= resp.status_code < 400
        detail = None if ok else f"HTTP {resp.status_code}"
        return ok, latency_ms, detail
    except httpx.HTTPError as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return False, latency_ms, f"{type(e).__name__}: {e}"
    except Exception as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return False, latency_ms, f"{type(e).__name__}: {e}"


async def _atimed(
    coro_fn: Callable[[httpx.AsyncClient], Awaitable[httpx.Response]],
    *,
    timeout: float = DEFAULT_TIMEOUT_S,  # noqa: ASYNC109
) -> tuple[bool, float | None, str | None]:
    """Async variant of :func:`_timed` using ``httpx.AsyncClient``."""
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as cli:
            resp = await coro_fn(cli)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        ok = 200 <= resp.status_code < 400
        detail = None if ok else f"HTTP {resp.status_code}"
        return ok, latency_ms, detail
    except httpx.HTTPError as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return False, latency_ms, f"{type(e).__name__}: {e}"
    except Exception as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return False, latency_ms, f"{type(e).__name__}: {e}"


def check_yfinance() -> dict[str, Any]:
    """Probe Yahoo Finance via the public chart endpoint."""

    def _call(cli: httpx.Client) -> httpx.Response:
        return cli.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/SPY",
            params={"range": "1d", "interval": "1d"},
        )

    ok, latency, detail = _timed(_call)
    return {"ok": ok, "latency_ms": latency, "detail": detail, "configured": True}


def check_tiingo() -> dict[str, Any]:
    """Probe Tiingo. Reports ``configured=False`` when no API key is set."""
    api_key = os.environ.get("TIINGO_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "latency_ms": None,
            "detail": "TIINGO_API_KEY not set",
            "configured": False,
        }

    def _call(cli: httpx.Client) -> httpx.Response:
        return cli.get(
            f"{tiingo_src.TIINGO_BASE}/SPY/prices",
            params={"startDate": "2024-01-02", "endDate": "2024-01-03"},
            headers={"Authorization": f"Token {api_key}"},
        )

    ok, latency, detail = _timed(_call)
    return {"ok": ok, "latency_ms": latency, "detail": detail, "configured": True}


def check_stooq() -> dict[str, Any]:
    """Probe Stooq's CSV endpoint."""

    def _call(cli: httpx.Client) -> httpx.Response:
        return cli.get(
            stooq_src.STOOQ_BASE,
            params={"s": "spy.us", "i": "d", "d1": "20240102", "d2": "20240103"},
        )

    ok, latency, detail = _timed(_call)
    return {"ok": ok, "latency_ms": latency, "detail": detail, "configured": True}


def check_polymarket() -> dict[str, Any]:
    """Probe the Polymarket Gamma API root."""

    def _call(cli: httpx.Client) -> httpx.Response:
        return cli.get("https://gamma-api.polymarket.com/markets", params={"limit": 1})

    ok, latency, detail = _timed(_call)
    return {"ok": ok, "latency_ms": latency, "detail": detail, "configured": True}


def check_kalshi() -> dict[str, Any]:
    """Probe Kalshi's public events endpoint."""

    def _call(cli: httpx.Client) -> httpx.Response:
        return cli.get(
            "https://api.elections.kalshi.com/trade-api/v2/events",
            params={"limit": 1},
        )

    ok, latency, detail = _timed(_call)
    return {"ok": ok, "latency_ms": latency, "detail": detail, "configured": True}


def check_fred() -> dict[str, Any]:
    """Probe the FRED ``fredgraph.csv`` auth-free endpoint."""

    def _call(cli: httpx.Client) -> httpx.Response:
        return cli.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv",
            params={"id": "DFF", "cosd": "2024-01-01", "coed": "2024-01-05"},
        )

    ok, latency, detail = _timed(_call)
    return {"ok": ok, "latency_ms": latency, "detail": detail, "configured": True}


SOURCE_CHECKS: dict[str, Callable[[], dict[str, Any]]] = {
    "yfinance": check_yfinance,
    "tiingo": check_tiingo,
    "stooq": check_stooq,
    "polymarket": check_polymarket,
    "kalshi": check_kalshi,
    "fred": check_fred,
}


# ---------------------------------------------------------------------- async


async def acheck_yfinance() -> dict[str, Any]:
    async def _call(cli: httpx.AsyncClient) -> httpx.Response:
        return await cli.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/SPY",
            params={"range": "1d", "interval": "1d"},
        )

    ok, latency, detail = await _atimed(_call)
    return {"ok": ok, "latency_ms": latency, "detail": detail, "configured": True}


async def acheck_tiingo() -> dict[str, Any]:
    api_key = os.environ.get("TIINGO_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "latency_ms": None,
            "detail": "TIINGO_API_KEY not set",
            "configured": False,
        }

    async def _call(cli: httpx.AsyncClient) -> httpx.Response:
        return await cli.get(
            f"{tiingo_src.TIINGO_BASE}/SPY/prices",
            params={"startDate": "2024-01-02", "endDate": "2024-01-03"},
            headers={"Authorization": f"Token {api_key}"},
        )

    ok, latency, detail = await _atimed(_call)
    return {"ok": ok, "latency_ms": latency, "detail": detail, "configured": True}


async def acheck_stooq() -> dict[str, Any]:
    async def _call(cli: httpx.AsyncClient) -> httpx.Response:
        return await cli.get(
            stooq_src.STOOQ_BASE,
            params={"s": "spy.us", "i": "d", "d1": "20240102", "d2": "20240103"},
        )

    ok, latency, detail = await _atimed(_call)
    return {"ok": ok, "latency_ms": latency, "detail": detail, "configured": True}


async def acheck_polymarket() -> dict[str, Any]:
    async def _call(cli: httpx.AsyncClient) -> httpx.Response:
        return await cli.get("https://gamma-api.polymarket.com/markets", params={"limit": 1})

    ok, latency, detail = await _atimed(_call)
    return {"ok": ok, "latency_ms": latency, "detail": detail, "configured": True}


async def acheck_kalshi() -> dict[str, Any]:
    async def _call(cli: httpx.AsyncClient) -> httpx.Response:
        return await cli.get(
            "https://api.elections.kalshi.com/trade-api/v2/events",
            params={"limit": 1},
        )

    ok, latency, detail = await _atimed(_call)
    return {"ok": ok, "latency_ms": latency, "detail": detail, "configured": True}


async def acheck_fred() -> dict[str, Any]:
    async def _call(cli: httpx.AsyncClient) -> httpx.Response:
        return await cli.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv",
            params={"id": "DFF", "cosd": "2024-01-01", "coed": "2024-01-05"},
        )

    ok, latency, detail = await _atimed(_call)
    return {"ok": ok, "latency_ms": latency, "detail": detail, "configured": True}


ASYNC_SOURCE_CHECKS: dict[str, Callable[[], Awaitable[dict[str, Any]]]] = {
    "yfinance": acheck_yfinance,
    "tiingo": acheck_tiingo,
    "stooq": acheck_stooq,
    "polymarket": acheck_polymarket,
    "kalshi": acheck_kalshi,
    "fred": acheck_fred,
}


async def acheck_all_sources() -> dict[str, dict[str, Any]]:
    """Run every probe in parallel via ``asyncio.gather`` and aggregate.

    Wall-clock = ``max(probe_latency)`` rather than the sum, so 6 probes
    each capped at 4 s finish in <= ~4 s instead of ~24 s worst case.
    """
    names = list(ASYNC_SOURCE_CHECKS.keys())
    coros = [ASYNC_SOURCE_CHECKS[n]() for n in names]
    settled = await asyncio.gather(*coros, return_exceptions=True)
    out: dict[str, dict[str, Any]] = {}
    for name, res in zip(names, settled, strict=False):
        if isinstance(res, BaseException):
            out[name] = {
                "ok": False,
                "latency_ms": None,
                "detail": f"probe crashed: {type(res).__name__}: {res}",
                "configured": True,
            }
        else:
            out[name] = res
    return out


def check_all_sources() -> dict[str, dict[str, Any]]:
    """Sync entry-point. Internally runs :func:`acheck_all_sources` so the
    six probes execute in parallel even from synchronous callers.

    If invoked from inside a running event loop (e.g. a FastAPI sync
    handler called from async context), we fall back to the per-probe sync
    implementations to avoid ``asyncio.run`` failing with "loop is already
    running".
    """
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False
    if not in_loop:
        return asyncio.run(acheck_all_sources())
    out: dict[str, dict[str, Any]] = {}
    for name, fn in SOURCE_CHECKS.items():
        try:
            out[name] = fn()
        except Exception as e:
            out[name] = {
                "ok": False,
                "latency_ms": None,
                "detail": f"probe crashed: {type(e).__name__}: {e}",
                "configured": True,
            }
    return out


__all__ = [
    "ASYNC_SOURCE_CHECKS",
    "SOURCE_CHECKS",
    "acheck_all_sources",
    "acheck_fred",
    "acheck_kalshi",
    "acheck_polymarket",
    "acheck_stooq",
    "acheck_tiingo",
    "acheck_yfinance",
    "check_all_sources",
    "check_fred",
    "check_kalshi",
    "check_polymarket",
    "check_stooq",
    "check_tiingo",
    "check_yfinance",
]

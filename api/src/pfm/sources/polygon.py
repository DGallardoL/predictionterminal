"""Polygon.io client for live earnings consensus + earnings calendar.

This source replaces the hardcoded ``CONSENSUS_EPS`` snapshot in
``pfm.earnings_whisper`` with a live feed when ``POLYGON_API_KEY`` is
configured. Polygon's free tier allows 5 calls/minute; we therefore
serialise outgoing requests behind an :class:`asyncio.Semaphore` and
sleep ``_RATE_LIMIT_SLEEP`` seconds between them.

Endpoints used (all under ``https://api.polygon.io``):

  - ``GET /v3/reference/financials?ticker=...`` — fundamentals incl.
    diluted EPS by period; we treat the most recent reported quarter as
    the "consensus benchmark".
  - ``GET /v3/reference/financials?ticker=...&period_of_report_date.gte=...``
    — same shape, filtered to a date window for the calendar fallback.

Polygon's true "earnings consensus" stream is a paid endpoint; on the
free tier we approximate the consensus number from the most recent
diluted EPS (typically what the street prints first), and surface the
``last_updated`` and ``surprise_history`` so callers can see the
provenance. When the key is unset, every public method returns
``None`` and logs a single warning.

The module exposes both the :class:`PolygonClient` (for tests / manual
calls) and a small set of convenience module-level helpers used by
:mod:`pfm.earnings_whisper`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import date, timedelta
from typing import Any

import httpx

from pfm.cache_utils import get_cache

logger = logging.getLogger(__name__)

POLYGON_BASE_URL: str = "https://api.polygon.io"

# Free-tier safety: 5 requests/minute → 12s gap is safe; we use 13s so
# bursts that interleave with other calls still fit the bucket.
_RATE_LIMIT_SLEEP: float = 13.0
_RATE_LIMIT_CONCURRENCY: int = 1

# 12-hour cache is the spec for consensus EPS — slow-moving by design.
_CONSENSUS_TTL_S: int = 43_200
_CALENDAR_TTL_S: int = 43_200
_CONSENSUS_CACHE = get_cache("polygon_consensus", ttl=_CONSENSUS_TTL_S)
_CALENDAR_CACHE = get_cache("polygon_calendar", ttl=_CALENDAR_TTL_S)

_API_KEY_ENV: str = "POLYGON_API_KEY"

# Process-wide one-warning latch so we don't spam logs every fallback.
_KEY_MISSING_WARNED: bool = False

# Process-wide gate so all callers share the 5/min budget, not just calls
# inside a single client instance.
#
# We create the Semaphore lazily, keyed by the running event loop, because
# asyncio primitives bind to a loop on first use. A module-level Semaphore
# instantiated at import time gets bound to whatever loop touches it first;
# in tests that run multiple ``asyncio.run(...)`` invocations each one
# spawns a fresh loop, and a Semaphore from a closed loop either deadlocks
# or silently no-ops, both of which break the retry-counting assertions in
# the polygon test suite.
_RATE_GATES: dict[int, asyncio.Semaphore] = {}


def _rate_gate() -> asyncio.Semaphore:
    """Return the rate-limit gate bound to the currently-running loop."""
    loop = asyncio.get_event_loop()
    key = id(loop)
    gate = _RATE_GATES.get(key)
    if gate is None:
        gate = asyncio.Semaphore(_RATE_LIMIT_CONCURRENCY)
        _RATE_GATES[key] = gate
    return gate


class PolygonError(RuntimeError):
    """Raised when Polygon returns a non-recoverable response."""


def _api_key() -> str | None:
    """Return ``POLYGON_API_KEY`` or ``None`` if unset (logged once)."""
    global _KEY_MISSING_WARNED
    key = os.environ.get(_API_KEY_ENV)
    if key:
        return key
    if not _KEY_MISSING_WARNED:
        logger.warning(
            "polygon: %s not set; earnings whisper will fall back to hardcoded snapshot",
            _API_KEY_ENV,
        )
        _KEY_MISSING_WARNED = True
    return None


def _reset_warning_latch() -> None:
    """Test helper: clear the one-shot warning flag so tests can reassert it."""
    global _KEY_MISSING_WARNED
    _KEY_MISSING_WARNED = False


class PolygonClient:
    """Async Polygon.io client for fundamentals and earnings calendar.

    Args:
        api_key: API token. Falls back to ``POLYGON_API_KEY`` env var.
        client: Injectable :class:`httpx.AsyncClient` (tests).
        base_url: Override the API root (tests).
        timeout: Per-request timeout in seconds.
        rate_limit_sleep: Seconds to wait after each successful call.
            Set to ``0`` in tests to skip the free-tier delay.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        base_url: str = POLYGON_BASE_URL,
        timeout: float = 15.0,
        rate_limit_sleep: float | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get(_API_KEY_ENV)
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None
        self._rate_limit_sleep = (
            float(rate_limit_sleep) if rate_limit_sleep is not None else _RATE_LIMIT_SLEEP
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> PolygonClient:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.close()

    # ---- low-level fetch ---------------------------------------------------

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        retries: int = 1,
    ) -> dict[str, Any]:
        """GET ``path`` with the API key appended; retry once on 429.

        429 is the free-tier rate-limit signal. We back off for the
        configured sleep, retry once, then raise :class:`PolygonError`
        so callers can fall back to the hardcoded snapshot.
        """
        if not self.api_key:
            raise PolygonError("POLYGON_API_KEY is required")
        url = f"{self.base_url}{path}"
        merged: dict[str, Any] = dict(params or {})
        merged["apiKey"] = self.api_key

        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            async with _rate_gate():
                try:
                    resp = await self._client.get(url, params=merged)
                except httpx.HTTPError as exc:
                    last_exc = exc
                    logger.info("polygon: transport error on %s: %s", path, exc)
                    if attempt >= retries:
                        raise PolygonError(f"polygon transport error: {exc}") from exc
                    await asyncio.sleep(self._rate_limit_sleep)
                    continue

                if resp.status_code == 429:
                    logger.info("polygon: 429 rate-limited on %s, attempt=%d", path, attempt)
                    if attempt >= retries:
                        raise PolygonError(
                            f"polygon rate-limited (429) after {attempt + 1} attempts"
                        )
                    await asyncio.sleep(self._rate_limit_sleep)
                    continue
                if resp.status_code >= 500:
                    logger.info("polygon: 5xx on %s status=%d", path, resp.status_code)
                    if attempt >= retries:
                        raise PolygonError(f"polygon HTTP {resp.status_code}: {resp.text[:200]}")
                    await asyncio.sleep(self._rate_limit_sleep)
                    continue
                if resp.status_code != 200:
                    raise PolygonError(
                        f"polygon HTTP {resp.status_code} on {path}: {resp.text[:200]}"
                    )
                # Success path — observe the rate-limit gap before releasing
                # the gate so the next caller gets a clean slot.
                if self._rate_limit_sleep > 0:
                    await asyncio.sleep(self._rate_limit_sleep)
                try:
                    return resp.json()
                except ValueError as exc:
                    raise PolygonError(f"polygon non-JSON response on {path}") from exc
        # Defensive — loop above always returns or raises.
        raise PolygonError(f"polygon: exhausted retries on {path}: {last_exc}")

    # ---- public methods ----------------------------------------------------

    async def fetch_consensus_eps(self, ticker: str) -> dict[str, Any]:
        """Return consensus-EPS metadata for ``ticker``.

        Output shape::

            {
                "ticker": "NVDA",
                "current_estimate": 0.84,
                "n_analysts": 0,         # 0 on free tier (paid feature)
                "last_updated": "2026-04-30",
                "surprise_history": [
                    {"period": "2026-Q1", "actual_eps": 0.84,
                     "estimated_eps": 0.80, "surprise_pct": 5.0},
                    ...
                ],
            }

        Caches in ``polygon_consensus`` for 12 hours.
        """
        tk = ticker.upper()
        cache_key = ("consensus", tk)
        cached = _CONSENSUS_CACHE.get(cache_key)
        if cached is not None:
            return cached

        payload = await self._get(
            "/vX/reference/financials",
            params={"ticker": tk, "limit": 8, "order": "desc"},
        )
        results = payload.get("results") or []
        if not results:
            raise PolygonError(f"polygon: no financials for {tk!r}")

        history = self._build_surprise_history(results)
        latest = results[0]
        latest_eps = self._extract_diluted_eps(latest)
        last_updated = (
            latest.get("filing_date")
            or latest.get("end_date")
            or latest.get("period_of_report_date")
        )

        out: dict[str, Any] = {
            "ticker": tk,
            "current_estimate": latest_eps,
            "n_analysts": int(latest.get("n_analysts") or 0),
            "last_updated": last_updated,
            "surprise_history": history,
        }
        _CONSENSUS_CACHE.set(cache_key, out, ttl=_CONSENSUS_TTL_S)
        return out

    async def fetch_eps_history(
        self,
        ticker: str,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """Return up to ``limit`` past EPS prints for ``ticker``."""
        consensus = await self.fetch_consensus_eps(ticker)
        return list(consensus.get("surprise_history", []))[:limit]

    async def fetch_earnings_calendar(
        self,
        start_date: date,
        end_date: date,
        ticker: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return earnings entries with ``period_of_report_date`` in window.

        Each entry::

            {"ticker": "...", "earnings_date": "YYYY-MM-DD",
             "consensus_eps": float | None, "n_analysts": int}
        """
        cache_key = (
            "calendar",
            start_date.isoformat(),
            end_date.isoformat(),
            (ticker or "").upper(),
        )
        cached = _CALENDAR_CACHE.get(cache_key)
        if cached is not None:
            return cached

        params: dict[str, Any] = {
            "period_of_report_date.gte": start_date.isoformat(),
            "period_of_report_date.lte": end_date.isoformat(),
            "limit": 100,
            "order": "asc",
        }
        if ticker:
            params["ticker"] = ticker.upper()

        payload = await self._get("/vX/reference/financials", params=params)
        results = payload.get("results") or []
        out: list[dict[str, Any]] = []
        for r in results:
            tickers = r.get("tickers") or [r.get("ticker")]
            tk = tickers[0] if tickers else None
            if not tk:
                continue
            ed = r.get("period_of_report_date") or r.get("end_date")
            if not ed:
                continue
            out.append(
                {
                    "ticker": tk.upper(),
                    "earnings_date": ed,
                    "consensus_eps": self._extract_diluted_eps(r),
                    "n_analysts": int(r.get("n_analysts") or 0),
                }
            )
        _CALENDAR_CACHE.set(cache_key, out, ttl=_CALENDAR_TTL_S)
        return out

    # ---- parsing helpers ---------------------------------------------------

    @staticmethod
    def _extract_diluted_eps(record: dict[str, Any]) -> float | None:
        """Pull diluted-EPS from a Polygon financials record.

        Polygon's vX shape nests metrics under
        ``financials.income_statement.diluted_earnings_per_share.value``.
        We also accept a flat ``diluted_eps`` for forward compatibility
        with paid tiers that surface the analyst-consensus directly.
        """
        if "diluted_eps" in record:
            try:
                v = record.get("diluted_eps")
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None
        fin = record.get("financials") or {}
        inc = fin.get("income_statement") or {}
        node = inc.get("diluted_earnings_per_share") or {}
        v = node.get("value")
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _build_surprise_history(
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for r in results:
            actual = PolygonClient._extract_diluted_eps(r)
            est = r.get("estimated_eps")
            try:
                est_f = float(est) if est is not None else None
            except (TypeError, ValueError):
                est_f = None
            surprise_pct: float | None = None
            if actual is not None and est_f not in (None, 0.0):
                surprise_pct = round((actual - est_f) / est_f * 100.0, 4)  # type: ignore[operator]
            period = r.get("fiscal_period") or r.get("period_of_report_date") or r.get("end_date")
            out.append(
                {
                    "period": period,
                    "actual_eps": actual,
                    "estimated_eps": est_f,
                    "surprise_pct": surprise_pct,
                }
            )
        return out


# ---------------------------------------------------------------------------
# Module-level convenience helpers (used by pfm.earnings_whisper)
# ---------------------------------------------------------------------------


async def fetch_consensus_eps_or_none(ticker: str) -> dict[str, Any] | None:
    """Return Polygon consensus for ``ticker`` or ``None`` on any failure.

    Designed to never raise — callers fall back to a hardcoded snapshot.
    """
    if _api_key() is None:
        return None
    try:
        async with PolygonClient() as cli:
            return await cli.fetch_consensus_eps(ticker)
    except PolygonError as exc:
        logger.info("polygon: consensus lookup failed for %s: %s", ticker, exc)
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("polygon: unexpected error for %s: %s", ticker, exc)
        return None


async def fetch_earnings_calendar_or_empty(
    start_date: date,
    end_date: date,
    ticker: str | None = None,
) -> list[dict[str, Any]]:
    """Return Polygon calendar entries or ``[]`` on any failure."""
    if _api_key() is None:
        return []
    try:
        async with PolygonClient() as cli:
            return await cli.fetch_earnings_calendar(start_date, end_date, ticker=ticker)
    except PolygonError as exc:
        logger.info("polygon: calendar lookup failed: %s", exc)
        return []
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("polygon: unexpected calendar error: %s", exc)
        return []


def is_configured() -> bool:
    """Return ``True`` when ``POLYGON_API_KEY`` is set in the environment."""
    return bool(os.environ.get(_API_KEY_ENV))


# Default look-ahead window for the calendar (kept here so the endpoint
# stays a thin shim).
DEFAULT_CALENDAR_DAYS: int = 30


def default_calendar_window(today: date | None = None) -> tuple[date, date]:
    today = today or date.today()
    return today, today + timedelta(days=DEFAULT_CALENDAR_DAYS)


__all__ = [
    "DEFAULT_CALENDAR_DAYS",
    "POLYGON_BASE_URL",
    "PolygonClient",
    "PolygonError",
    "default_calendar_window",
    "fetch_consensus_eps_or_none",
    "fetch_earnings_calendar_or_empty",
    "is_configured",
]

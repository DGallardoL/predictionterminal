"""Shared httpx.AsyncClient pool for Polymarket Gamma and CLOB.

Polymarket exposes two public, no-auth HTTPS endpoints:

  - ``https://gamma.polymarket.com``  — market metadata (slugs, token ids, …)
  - ``https://clob.polymarket.com``    — order book + ``/prices-history``

Today, many call sites construct a fresh ``httpx.AsyncClient()`` per request
(or per coroutine). That's wasteful for three reasons:

  1. **TLS handshake repeat** — every new client re-runs the TLS handshake
     (~50-150 ms RTT per host on cold pools). With keep-alive, the same TCP
     socket can be reused across hundreds of requests.
  2. **No HTTP/2 multiplexing** — single-shot clients fall back to HTTP/1.1
     and serialize requests on a connection.
  3. **Connection limits ignored** — without a shared pool, a burst of 50
     concurrent Terminal-mode opens spawns 50 sockets to the same host.

This module provides a process-wide singleton (``PolymarketHTTPPool``) that
maintains exactly one ``httpx.AsyncClient`` per host, HTTP/2-enabled, with
sensible defaults. Callers retrieve clients via ``.gamma_client`` and
``.clob_client`` properties.

Usage
-----

.. code-block:: python

    from pfm.sources.polymarket_pool import PolymarketHTTPPool

    pool = PolymarketHTTPPool.instance()
    resp = await pool.gamma_client.get("/markets", params={"slug": slug})
    resp.raise_for_status()
    market = resp.json()

The pool is cooperative with FastAPI lifespan: call
``await PolymarketHTTPPool.instance().aclose()`` on shutdown.

Retries
-------

Retries are **not** built into the pool; this keeps the contract explicit.
Callers that need them should wrap calls with ``tenacity``::

    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5))
    async def fetch():
        resp = await pool.gamma_client.get(...)
        resp.raise_for_status()
        return resp.json()

Migration guide
---------------

Files that currently spin up per-call ``httpx.AsyncClient()`` against
gamma/clob and should be migrated in a follow-up task (T-migrate-pool):

  - ``api/src/pfm/sources/polymarket.py`` — ``get_market_metadata`` /
    ``get_prices_history`` create clients inside each function.
  - ``api/src/pfm/sources/polymarket_gamma.py`` (if/when split out from
    ``polymarket.py``) — Gamma-only metadata helpers.
  - ``api/src/pfm/sources/polymarket_clob.py`` (if/when split out) — CLOB
    price-history + order-book helpers.
  - ``api/src/pfm/terminal/quote.py``, ``terminal/compare.py``,
    ``terminal/bulk_export.py``, ``terminal/orderbook.py``,
    ``terminal/quality_score.py``, ``terminal/watchlist.py``,
    ``terminal/live_stream.py`` — Terminal panes that currently take an
    ``http: httpx.AsyncClient`` parameter; swap the caller to pass
    ``PolymarketHTTPPool.instance().gamma_client`` / ``.clob_client``.
  - ``api/src/pfm/strategies_arb_router.py``, ``arb_scanner.py``,
    ``decay_monitor.py``, ``embed.py``, ``live_signals_job.py``,
    ``alpha_tier_regen.py``.
  - ``api/src/pfm/archive/polymarket_archive.py``,
    ``archive/cross_venue_archive.py``.
  - ``api/src/pfm/crypto5min/market_fetcher.py`` — CLOB REST polls
    (the WS subscriber path is independent).
  - ``api/src/pfm/realtime/pollers.py`` — CLOB token pollers.

Migration pattern::

    # Before
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get("https://gamma.polymarket.com/markets", params=...)

    # After
    from pfm.sources.polymarket_pool import PolymarketHTTPPool

    pool = PolymarketHTTPPool.instance()
    r = await pool.gamma_client.get("/markets", params=...)

Note the relative path: the pooled client has ``base_url`` set, so callers
should drop the ``https://...`` prefix.
"""

from __future__ import annotations

import asyncio
import socket
from typing import ClassVar

import httpx

__all__ = [
    "CLOB_BASE_URL",
    "DEFAULT_LIMITS",
    "DEFAULT_TIMEOUT",
    "GAMMA_BASE_URL",
    "PolymarketHTTPPool",
]

GAMMA_BASE_URL: str = "https://gamma.polymarket.com"
CLOB_BASE_URL: str = "https://clob.polymarket.com"

# Tuned for Terminal-mode burst: 50 concurrent market opens × 2 hosts ⇒ 100
# total sockets, but keepalive=20 means most reuse an existing connection.
DEFAULT_LIMITS: httpx.Limits = httpx.Limits(
    max_connections=50,
    max_keepalive_connections=20,
)

# Connect should be fast (TLS handshake); reads can be slower for
# /prices-history when the market has months of daily buckets.
DEFAULT_TIMEOUT: httpx.Timeout = httpx.Timeout(10.0, connect=3.0)


def _user_agent() -> str:
    """Return ``prediction-terminal/1.0 (<hostname>)`` for upstream logs."""
    try:
        host = socket.gethostname() or "unknown-host"
    except Exception:  # pragma: no cover - extremely unlikely
        host = "unknown-host"
    return f"prediction-terminal/1.0 ({host})"


class PolymarketHTTPPool:
    """Process-wide singleton pool of httpx.AsyncClient(s) for Polymarket.

    Holds exactly one client per host (Gamma + CLOB), each configured with
    HTTP/2, conservative connection limits, and a default 10s/3s timeout.

    Construction is lazy: the underlying clients are created on first
    property access, not in ``__init__``. This keeps import-time cheap
    (tests can import the module without opening sockets).

    Thread-safety note: ``httpx.AsyncClient`` is asyncio-bound. Do not
    share a single pool instance across distinct event loops — fork-style
    workers must call :meth:`reset_for_testing` (or re-construct) in each
    child process / loop.
    """

    _instance: ClassVar[PolymarketHTTPPool | None] = None
    _instance_lock: ClassVar[asyncio.Lock | None] = None

    def __init__(self) -> None:
        self._gamma_client: httpx.AsyncClient | None = None
        self._clob_client: httpx.AsyncClient | None = None
        self._closed: bool = False
        self._user_agent: str = _user_agent()

    # ------------------------------------------------------------------
    # Singleton management
    # ------------------------------------------------------------------

    @classmethod
    def instance(cls) -> PolymarketHTTPPool:
        """Return the process-wide singleton, constructing on first call.

        Subsequent calls return the same object. The clients themselves are
        still lazily constructed on first property access.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_for_testing(cls) -> None:
        """Drop the singleton without closing it.

        Tests that need a fresh pool (e.g. to assert constructor kwargs)
        should call this *after* awaiting :meth:`aclose` on the previous
        instance. Safe to call when no instance exists.
        """
        cls._instance = None

    # ------------------------------------------------------------------
    # Client factories
    # ------------------------------------------------------------------

    def _build_client(self, base_url: str) -> httpx.AsyncClient:
        """Construct an HTTP/2-enabled client for one host.

        Factored out so tests can patch ``httpx.AsyncClient`` and assert the
        exact kwargs without poking pool internals.
        """
        return httpx.AsyncClient(
            base_url=base_url,
            http2=True,
            limits=DEFAULT_LIMITS,
            timeout=DEFAULT_TIMEOUT,
            headers={"User-Agent": self._user_agent},
        )

    @property
    def gamma_client(self) -> httpx.AsyncClient:
        """Return the shared Gamma client (``gamma.polymarket.com``).

        Constructed lazily on first access. Reusing this client across
        coroutines is the *point* — that's how keepalive + HTTP/2 multiplex.
        """
        if self._closed:
            raise RuntimeError("PolymarketHTTPPool is closed")
        if self._gamma_client is None:
            self._gamma_client = self._build_client(GAMMA_BASE_URL)
        return self._gamma_client

    @property
    def clob_client(self) -> httpx.AsyncClient:
        """Return the shared CLOB client (``clob.polymarket.com``)."""
        if self._closed:
            raise RuntimeError("PolymarketHTTPPool is closed")
        if self._clob_client is None:
            self._clob_client = self._build_client(CLOB_BASE_URL)
        return self._clob_client

    @property
    def is_closed(self) -> bool:
        """True if :meth:`aclose` has been called."""
        return self._closed

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Close both clients (idempotent).

        Safe to call multiple times. After close, property access raises
        ``RuntimeError`` — re-acquire via :meth:`instance` after first
        calling :meth:`reset_for_testing` if you need a fresh pool.
        """
        self._closed = True
        gamma = self._gamma_client
        clob = self._clob_client
        self._gamma_client = None
        self._clob_client = None
        # Close concurrently; swallow exceptions so one failure doesn't
        # mask the other.
        coros = []
        if gamma is not None:
            coros.append(gamma.aclose())
        if clob is not None:
            coros.append(clob.aclose())
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)

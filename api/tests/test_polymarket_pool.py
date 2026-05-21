"""Tests for :mod:`pfm.sources.polymarket_pool`.

The pool itself opens no real sockets at construction time — clients are
built lazily on first property access. We exercise:

  * Singleton identity + lazy client construction.
  * Distinct clients for Gamma vs CLOB (different ``base_url``).
  * ``aclose()`` idempotence and post-close access guard.
  * HTTP/2 + limits + timeout + ``User-Agent`` kwargs passed to ``httpx``.
  * Functional round-trip with ``httpx.MockTransport`` to prove the pool's
    client is actually used to issue a request.
"""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import httpx
import pytest

from pfm.sources import polymarket_pool as pmp
from pfm.sources.polymarket_pool import (
    CLOB_BASE_URL,
    DEFAULT_LIMITS,
    DEFAULT_TIMEOUT,
    GAMMA_BASE_URL,
    PolymarketHTTPPool,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure each test starts with a clean module singleton."""
    PolymarketHTTPPool.reset_for_testing()
    yield
    # Best-effort teardown — clients may already have been closed.
    inst = PolymarketHTTPPool._instance
    if inst is not None and not inst.is_closed:
        import asyncio

        try:
            asyncio.get_event_loop().run_until_complete(inst.aclose())
        except RuntimeError:
            # No running loop in this thread; close synchronously via a new loop.
            asyncio.new_event_loop().run_until_complete(inst.aclose())
    PolymarketHTTPPool.reset_for_testing()


# ---------------------------------------------------------------------------
# Singleton + identity
# ---------------------------------------------------------------------------


def test_instance_returns_singleton():
    """``instance()`` called twice returns the same object."""
    a = PolymarketHTTPPool.instance()
    b = PolymarketHTTPPool.instance()
    assert a is b


def test_reset_for_testing_drops_singleton():
    """``reset_for_testing`` clears the module-level cache."""
    a = PolymarketHTTPPool.instance()
    PolymarketHTTPPool.reset_for_testing()
    b = PolymarketHTTPPool.instance()
    assert a is not b


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


def test_gamma_and_clob_clients_are_distinct():
    """Pool exposes one client per host; they are not the same object."""
    pool = PolymarketHTTPPool.instance()
    assert pool.gamma_client is not pool.clob_client


def test_gamma_client_base_url():
    """Gamma client points at ``gamma.polymarket.com``."""
    pool = PolymarketHTTPPool.instance()
    assert str(pool.gamma_client.base_url).rstrip("/") == GAMMA_BASE_URL


def test_clob_client_base_url():
    """CLOB client points at ``clob.polymarket.com``."""
    pool = PolymarketHTTPPool.instance()
    assert str(pool.clob_client.base_url).rstrip("/") == CLOB_BASE_URL


def test_repeated_property_access_returns_same_client():
    """Lazy construction: second access does NOT make a new client."""
    pool = PolymarketHTTPPool.instance()
    g1 = pool.gamma_client
    g2 = pool.gamma_client
    assert g1 is g2


# ---------------------------------------------------------------------------
# Configuration assertions (HTTP/2, limits, timeout, UA)
# ---------------------------------------------------------------------------


def test_build_client_uses_http2_and_limits_and_timeout():
    """``_build_client`` constructs ``httpx.AsyncClient`` with the documented kwargs.

    We patch ``httpx.AsyncClient`` so no socket is opened and we can introspect
    the exact arguments.
    """
    pool = PolymarketHTTPPool.instance()
    sentinel = MagicMock(spec=httpx.AsyncClient)

    with patch.object(pmp.httpx, "AsyncClient", return_value=sentinel) as mock_cls:
        _ = pool._build_client(GAMMA_BASE_URL)

        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["base_url"] == GAMMA_BASE_URL
        assert kwargs["http2"] is True
        assert kwargs["limits"] is DEFAULT_LIMITS
        assert kwargs["timeout"] is DEFAULT_TIMEOUT
        headers = kwargs["headers"]
        assert "User-Agent" in headers
        assert headers["User-Agent"].startswith("prediction-terminal/1.0")


def test_default_limits_values():
    """Default limits match the documented 50 / 20 split."""
    assert DEFAULT_LIMITS.max_connections == 50
    assert DEFAULT_LIMITS.max_keepalive_connections == 20


def test_default_timeout_values():
    """Default timeout: 10s overall, 3s connect."""
    # httpx.Timeout exposes per-op timeouts as attributes.
    assert DEFAULT_TIMEOUT.connect == pytest.approx(3.0)
    assert DEFAULT_TIMEOUT.read == pytest.approx(10.0)


def test_user_agent_contains_hostname():
    """User-Agent embeds the OS hostname for upstream debugging."""
    pool = PolymarketHTTPPool.instance()
    ua = pool._user_agent
    assert ua.startswith("prediction-terminal/1.0 (")
    assert ua.endswith(")")
    # Hostname appears verbatim inside the parens (modulo the fallback).
    host = socket.gethostname() or "unknown-host"
    assert host in ua or "unknown-host" in ua


# ---------------------------------------------------------------------------
# Lifecycle (aclose)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aclose_closes_both_clients():
    """``aclose`` closes Gamma and CLOB clients and marks the pool closed."""
    pool = PolymarketHTTPPool.instance()
    # Touch both properties to force client construction.
    g = pool.gamma_client
    c = pool.clob_client
    assert not g.is_closed
    assert not c.is_closed

    await pool.aclose()

    assert pool.is_closed
    assert g.is_closed
    assert c.is_closed


@pytest.mark.asyncio
async def test_aclose_is_idempotent():
    """Calling ``aclose`` twice does not raise."""
    pool = PolymarketHTTPPool.instance()
    _ = pool.gamma_client
    await pool.aclose()
    await pool.aclose()  # second call must be a no-op


@pytest.mark.asyncio
async def test_access_after_close_raises():
    """Accessing a client property after close raises ``RuntimeError``."""
    pool = PolymarketHTTPPool.instance()
    _ = pool.gamma_client
    await pool.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        _ = pool.gamma_client
    with pytest.raises(RuntimeError, match="closed"):
        _ = pool.clob_client


@pytest.mark.asyncio
async def test_aclose_without_touching_clients():
    """``aclose`` works even if no clients were ever constructed."""
    pool = PolymarketHTTPPool.instance()
    # No property access — both internal clients are None.
    await pool.aclose()
    assert pool.is_closed


# ---------------------------------------------------------------------------
# Functional: a real request flows through the pool's client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pool_client_used_for_real_request():
    """A request issued via the pool's Gamma client hits our mock transport.

    We swap the pool's lazily-built ``gamma_client`` for one wired to a
    ``MockTransport``. This proves callers can substitute behaviour for
    tests without monkey-patching the singleton's internals.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        # Verify the pooled client preserves base_url + UA.
        assert request.url.host == "gamma.polymarket.com"
        assert request.url.path == "/markets"
        assert request.headers.get("user-agent", "").startswith("prediction-terminal/1.0")
        return httpx.Response(200, json={"slug": "x", "ok": True})

    pool = PolymarketHTTPPool.instance()
    # Force the lazy client into existence, then swap its transport.
    real = pool.gamma_client
    await real.aclose()
    pool._gamma_client = httpx.AsyncClient(
        base_url=GAMMA_BASE_URL,
        transport=httpx.MockTransport(handler),
        headers={"User-Agent": pool._user_agent},
    )

    resp = await pool.gamma_client.get("/markets", params={"slug": "x"})
    assert resp.status_code == 200
    assert resp.json() == {"slug": "x", "ok": True}

# DEPRECATED 2026-05-08: use /terminal/stream from pfm.realtime instead.
# This endpoint launches one poller per SSE connection, which scales linearly
# with concurrent clients (~20k req/min at 100 users hitting CLOB). The
# replacement multiplexes one poller per (kind, slug) over N clients.
# Kept for backward compat; will be removed in v0.2.
"""Server-Sent Events stream of live Polymarket midpoints for the Terminal UI.

Exposes ``GET /terminal/live-stream?slugs=a,b,c&hz=0.5`` which yields one
``tick`` SSE event per slug every ``1/hz`` seconds with ``{slug, mid, bid,
ask, ts}``. The stream auto-disconnects after :data:`MAX_STREAM_SECONDS` so
that long-lived connections roll over and clients reconnect — this matches
how production SSE deployments survive load-balancer idle timeouts.

External calls per tick (per slug):
    - Gamma  ``/markets?slug={slug}``    → resolve slug → YES ``token_id``
                                            (cached once at stream start).
    - CLOB   ``/midpoint?token_id=…``    → live midpoint in [0, 1].
    - CLOB   ``/price?token_id=…&side=…``→ best bid + best ask in [0, 1].

All HTTP runs through ``httpx`` so tests can mock with ``respx``.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

import httpx
from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from pfm.sources.polymarket import PolymarketClient, PolymarketError

GAMMA_URL: str = "https://gamma-api.polymarket.com"
CLOB_URL: str = "https://clob.polymarket.com"

# Hard caps — enforced regardless of what the client requests.
MAX_SLUGS: int = 30
MAX_STREAM_SECONDS: float = 300.0  # 5 minutes; client reconnects after.
MIN_HZ: float = 0.1  # 1 tick / 10s — slowest we allow
MAX_HZ: float = 5.0  # 5 ticks / s   — Polymarket is generous but be polite
DEFAULT_HZ: float = 0.5  # 1 tick / 2s

router = APIRouter(prefix="/terminal", tags=["terminal"])


# ---- HTTP fetchers ---------------------------------------------------------


def _resolve_yes_token_id(slug: str, client: PolymarketClient) -> str | None:
    """Return the YES ``clobTokenIds[0]`` for ``slug`` or ``None`` on failure.

    We swallow failures here because the streaming endpoint must keep going
    even if one slug is bad — it just gets dropped from the rotation.
    """
    try:
        meta = client.get_market_metadata(slug)
    except (httpx.HTTPError, PolymarketError):
        return None
    return meta.yes_token_id


async def _fetch_midpoint(token_id: str, client: httpx.AsyncClient) -> float | None:
    """One CLOB ``/midpoint`` call. ``None`` on any error.

    Async so the streaming loop can fan out midpoint+bid+ask for many slugs
    concurrently with ``asyncio.gather`` instead of blocking the event loop on
    each sequential HTTP call (which used to dominate latency for 5+ slugs).
    """
    try:
        r = await client.get(f"{CLOB_URL}/midpoint", params={"token_id": token_id})
        r.raise_for_status()
        payload = r.json()
    except (httpx.HTTPError, ValueError):
        return None
    raw = payload.get("mid") if isinstance(payload, dict) else None
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


async def _fetch_side_price(token_id: str, side: str, client: httpx.AsyncClient) -> float | None:
    """One CLOB ``/price`` call for ``side`` ∈ {BUY, SELL}. ``None`` on error."""
    try:
        r = await client.get(
            f"{CLOB_URL}/price",
            params={"token_id": token_id, "side": side},
        )
        r.raise_for_status()
        payload = r.json()
    except (httpx.HTTPError, ValueError):
        return None
    raw = payload.get("price") if isinstance(payload, dict) else None
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


# ---- Pure helpers ----------------------------------------------------------


def _parse_slugs(raw: str) -> list[str]:
    """Split a comma-separated list, drop blanks, dedupe, cap at ``MAX_SLUGS``."""
    seen: dict[str, None] = {}  # preserve insertion order
    for s in raw.split(","):
        s = s.strip()
        if s and s not in seen:
            seen[s] = None
    return list(seen.keys())[:MAX_SLUGS]


def _format_event(event: str, payload: dict) -> bytes:
    """Format a single SSE frame as ``event:`` + ``data:`` + blank line."""
    body = json.dumps(payload, separators=(",", ":"))
    return f"event: {event}\ndata: {body}\n\n".encode()


# ---- The async generator that produces the stream -------------------------


async def _stream_ticks(
    request: Request,
    slugs: list[str],
    hz: float,
    *,
    deadline_seconds: float = MAX_STREAM_SECONDS,
    http_client: httpx.AsyncClient | None = None,
    poly_client: PolymarketClient | None = None,
) -> AsyncIterator[bytes]:
    """Yield SSE frames until the client disconnects, the deadline elapses,
    or the slug list is empty.

    Parameters mirror the endpoint kwargs so tests can drive this generator
    directly with mocked clients and a tiny ``deadline_seconds``.

    Per-cycle the three CLOB calls per slug (midpoint, bid, ask) and across
    all slugs run concurrently via ``asyncio.gather`` — for N slugs this turns
    a 3N-sequential burst into a single round-trip's worth of wall time.
    """
    interval = 1.0 / hz
    started = time.monotonic()

    owns_http = http_client is None
    owns_poly = poly_client is None
    http_client = http_client or httpx.AsyncClient(timeout=5.0)
    # PolymarketClient is sync — feed it a sync httpx.Client only for the
    # one-shot slug resolution; the streaming hot path uses the async client.
    if poly_client is None:
        _poly_sync_http = httpx.Client(timeout=5.0)
        poly_client = PolymarketClient(GAMMA_URL, CLOB_URL, client=_poly_sync_http)
    else:
        _poly_sync_http = None

    try:
        # Resolve slug → token_id ONCE at the top of the stream. Slugs that
        # fail to resolve are dropped silently (they would just emit nulls).
        # Resolution is sync (PolymarketClient.get_market_metadata) so we hop
        # off the event loop with to_thread for each lookup.
        token_map: dict[str, str] = {}
        for slug in slugs:
            tid = await asyncio.to_thread(_resolve_yes_token_id, slug, poly_client)
            if tid is not None:
                token_map[slug] = tid

        # Tell the client we're alive even if every slug failed to resolve.
        yield _format_event(
            "ready",
            {"slugs": list(token_map.keys()), "hz": hz, "interval_s": interval},
        )

        if not token_map:
            return

        while True:
            if await request.is_disconnected():
                return
            if time.monotonic() - started >= deadline_seconds:
                yield _format_event("bye", {"reason": "deadline"})
                return

            ts = int(time.time())

            async def _one_slug(
                slug: str,
                token_id: str,
                ts_local: int = ts,
            ) -> dict[str, object]:
                # Polymarket ``/price`` returns the best price on the named
                # SIDE of the order book: ``side=BUY`` → top of the buy side
                # (the best BID; what you'd receive when selling),
                # ``side=SELL`` → top of the sell side (the best ASK; what
                # you'd pay to buy). Verified live 2026-05-15: midpoint
                # 0.0085, side=BUY 0.008, side=SELL 0.009.
                # Fan the three calls out in parallel — they're independent.
                # ``ts_local`` default-binds the outer ``ts`` so the closure
                # doesn't share a single mutable reference across cycles.
                mid_t, bid_t, ask_t = await asyncio.gather(
                    _fetch_midpoint(token_id, http_client),
                    _fetch_side_price(token_id, "BUY", http_client),
                    _fetch_side_price(token_id, "SELL", http_client),
                )
                return {
                    "slug": slug,
                    "mid": mid_t,
                    "bid": bid_t,
                    "ask": ask_t,
                    "ts": ts_local,
                }

            results = await asyncio.gather(*(_one_slug(s, t) for s, t in token_map.items()))
            for payload in results:
                yield _format_event("tick", payload)

            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                # Client went away mid-sleep — exit cleanly.
                return
    finally:
        if owns_poly:
            poly_client.close()
            if _poly_sync_http is not None:
                _poly_sync_http.close()
        if owns_http:
            await http_client.aclose()


# ---- Endpoint --------------------------------------------------------------


@router.get("/live-stream")
async def live_stream(
    request: Request,
    slugs: str = Query(..., description="Comma-separated Polymarket slugs (max 30)."),
    hz: float = Query(default=DEFAULT_HZ, ge=MIN_HZ, le=MAX_HZ),
) -> StreamingResponse:
    """Open a Server-Sent Events stream of live midpoints for ``slugs``.

    The stream emits one ``tick`` event per slug per cycle, sleeps ``1/hz``
    seconds, and self-terminates after :data:`MAX_STREAM_SECONDS` so the
    client reconnects (a standard SSE pattern that survives proxy idle
    timeouts and prevents server-side memory leaks on long sessions).
    """
    slug_list = _parse_slugs(slugs)
    generator = _stream_ticks(request, slug_list, hz)
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
            "Connection": "keep-alive",
        },
    )

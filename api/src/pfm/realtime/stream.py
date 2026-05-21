"""SSE multiplexing endpoint — ``GET /terminal/stream``.

Subscribes one client to N ``(kind, slug)`` channels via the shared
:class:`pfm.realtime.hub.RealtimeHub`. The hub does the polling; this
module's only job is to translate hub events into SSE frames and to
send a periodic heartbeat that survives proxy idle timeouts.

Query string format::

    GET /terminal/stream?subs=book:slug-a,tape:slug-a,tick:slug-b

Each comma-separated entry is ``kind:slug``. Unknown kinds and missing
slugs return 400. Exceeding :data:`hub.MAX_SUBS_PER_CLIENT` returns 400.
At capacity the endpoint returns 503 with ``Retry-After: 30``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from pfm.realtime.hub import MAX_SUBS_PER_CLIENT, RealtimeHub
from pfm.realtime.pollers import SUPPORTED_KINDS

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_S: float = 10.0
QUEUE_GET_TIMEOUT_S: float = 1.0  # how long to wait before re-checking heartbeat

router = APIRouter(prefix="/terminal", tags=["terminal-realtime"])


def get_hub(request: Request) -> RealtimeHub:
    """FastAPI dependency — pull the per-app hub off ``app.state``."""
    hub = getattr(request.app.state, "hub", None)
    if hub is None:
        raise HTTPException(status_code=503, detail="realtime hub not initialized")
    return hub


# ---- subs parsing ----------------------------------------------------------


def parse_subs(raw: str) -> list[tuple[str, str]]:
    """Parse ``"book:slug-a,tape:slug-a"`` into ``[("book","slug-a"), ...]``.

    Empty entries are skipped, duplicates deduped (preserving order).
    Raises ``ValueError`` on malformed entries or unknown kinds.
    """
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"bad subscription {chunk!r}: expected 'kind:slug'")
        kind, _, slug = chunk.partition(":")
        kind = kind.strip().lower()
        slug = slug.strip()
        if not kind or not slug:
            raise ValueError(f"bad subscription {chunk!r}: empty kind or slug")
        if kind not in SUPPORTED_KINDS:
            raise ValueError(f"unsupported kind {kind!r}; supported={sorted(SUPPORTED_KINDS)}")
        key = (kind, slug)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


# ---- frame formatting ------------------------------------------------------


def format_event(event: str, payload: dict) -> bytes:
    """Encode one SSE frame. Keep this in sync with the legacy formatter."""
    body = json.dumps(payload, separators=(",", ":"))
    return f"event: {event}\ndata: {body}\n\n".encode()


# ---- the streaming generator ----------------------------------------------


async def _generate(
    request: Request,
    hub: RealtimeHub,
    cid: str,
    subs: list[tuple[str, str]],
) -> AsyncIterator[bytes]:
    """Drain the client's queue + emit heartbeats until disconnect."""
    session = hub.clients.get(cid)
    if session is None:
        return

    # Initial ack — confirm we're alive and what we subscribed to.
    yield format_event(
        "ready",
        {
            "cid": cid,
            "subs": [{"kind": k, "slug": s} for k, s in subs],
            "ts": int(time.time()),
        },
    )

    last_hb = time.monotonic()
    try:
        while True:
            if await request.is_disconnected() or session.closed:
                break
            try:
                evt = await asyncio.wait_for(
                    session.queue.get(),
                    timeout=QUEUE_GET_TIMEOUT_S,
                )
            except TimeoutError:
                evt = None

            now = time.monotonic()
            if evt is not None:
                event_type = str(evt.get("type") or "msg")
                yield format_event(event_type, evt)
                if event_type == "bye":
                    # Hub asked us to close (slow client).
                    break
            if now - last_hb >= HEARTBEAT_INTERVAL_S:
                yield format_event("hb", {"ts": int(time.time())})
                last_hb = now
    except asyncio.CancelledError:
        # Client went away mid-await; nothing to do — the finally in the
        # endpoint cleans up subscriptions.
        return


# ---- endpoint -------------------------------------------------------------


@router.get("/stream")
async def stream(
    request: Request,
    hub: Annotated[RealtimeHub, Depends(get_hub)],
    subs: str = Query(..., description="Comma-separated 'kind:slug' subscriptions."),
) -> StreamingResponse:
    """Multiplexed SSE stream over the realtime hub.

    Example::

        GET /terminal/stream?subs=book:slug-a,tape:slug-a,tick:slug-b
    """
    try:
        sub_list = parse_subs(subs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not sub_list:
        raise HTTPException(status_code=400, detail="no subscriptions provided")
    if len(sub_list) > MAX_SUBS_PER_CLIENT:
        raise HTTPException(
            status_code=400,
            detail=f"too many subs: {len(sub_list)} > {MAX_SUBS_PER_CLIENT}",
        )

    cid = uuid.uuid4().hex
    try:
        await hub.create_client(cid)
    except RuntimeError as e:
        if str(e) == "hub_full":
            raise HTTPException(
                status_code=503,
                detail="hub at capacity",
                headers={"Retry-After": "30"},
            ) from e
        raise HTTPException(status_code=500, detail=str(e)) from e

    try:
        for kind, slug in sub_list:
            await hub.subscribe(cid, kind, slug)
    except RuntimeError as e:
        await hub.remove_client(cid)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        await hub.remove_client(cid)
        raise HTTPException(status_code=400, detail=str(e)) from e

    async def _wrapped() -> AsyncIterator[bytes]:
        try:
            async for frame in _generate(request, hub, cid, sub_list):
                yield frame
        finally:
            await hub.remove_client(cid)

    return StreamingResponse(
        _wrapped(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


__all__ = ["format_event", "get_hub", "parse_subs", "router", "stream"]

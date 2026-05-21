"""In-process pub/sub hub for SSE multiplexing.

Design:
    - One :class:`asyncio.Task` per ``(kind, slug)`` regardless of how many
      clients are subscribed. The task polls Polymarket on a fixed cadence
      and fans the result out to every subscribed client's queue.
    - Each client has a bounded :class:`asyncio.Queue` (size 256). When a
      client falls behind, we apply two protections:
        1. *Coalescing* — if the queue still has a pending event for the
           same ``(kind, slug)``, the older event is silently replaced.
        2. *Drop-and-disconnect* — if coalescing fails (different keys
           queued) and ``put_nowait`` would raise ``QueueFull``, the
           ``dropped`` counter increments. Once a client crosses 50 drops
           inside a 30-second window the hub closes the session with an
           ``event: bye`` carrying ``{"reason":"slow_client"}``.
    - Global limits: 500 concurrent SSE connections, 60 subscriptions per
      client. Both are enforced at :meth:`RealtimeHub.create_client` and
      :meth:`RealtimeHub.subscribe` respectively.
    - Pollers are reference-counted by subscriber count: the last
      unsubscriber tears the poller task down so we don't keep hitting
      Polymarket for a slug nobody's watching.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Final

import httpx

from pfm.realtime.pollers import POLLERS, SUPPORTED_KINDS

logger = logging.getLogger(__name__)

# Hard limits — see module docstring.
QUEUE_MAXSIZE: Final[int] = 256
MAX_CLIENTS: Final[int] = 500
MAX_SUBS_PER_CLIENT: Final[int] = 60
DROPPED_THRESHOLD: Final[int] = 50
DROPPED_WINDOW_S: Final[float] = 30.0
DEFAULT_POLL_INTERVAL_S: Final[float] = 2.0  # one CLOB hit per slug per 2s


# Type alias — ``(slug, http) -> dict | None``.
PollerFn = Callable[[str, httpx.AsyncClient], Awaitable[dict | None]]


@dataclass
class ClientSession:
    """One SSE connection's bookkeeping.

    The hub writes events into ``queue`` and the streaming endpoint reads
    them. ``dropped_at`` records timestamps of recent backpressure drops so
    we can detect a sustained slow consumer (see module docstring).
    """

    cid: str
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=QUEUE_MAXSIZE))
    dropped: int = 0
    dropped_at: deque = field(default_factory=deque)
    last_active: float = field(default_factory=time.monotonic)
    subs: set[tuple[str, str]] = field(default_factory=set)
    closed: bool = False


class RealtimeHub:
    """In-process pub/sub hub. One instance per FastAPI app.

    Lifecycle: created at app startup, drained at shutdown via
    :meth:`shutdown`. Every async method is safe to call from any task.
    """

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient | None = None,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        pollers: dict[str, PollerFn] | None = None,
    ) -> None:
        self.clients: dict[str, ClientSession] = {}
        self.slug_subs: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.pollers: dict[tuple[str, str], asyncio.Task] = {}
        self.last_value: dict[tuple[str, str], dict] = {}
        self._lock = asyncio.Lock()
        self._http = http_client
        self._owns_http = http_client is None
        self._poll_interval_s = poll_interval_s
        # Test hook — override the kind→poller registry per-hub.
        self._poller_fns: dict[str, PollerFn] = pollers or dict(POLLERS)

    # ---- HTTP client ------------------------------------------------------

    def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=5.0)
        return self._http

    # ---- client lifecycle -------------------------------------------------

    async def create_client(self, cid: str) -> ClientSession:
        """Register a new SSE session. Raises ``RuntimeError`` if full."""
        async with self._lock:
            if len(self.clients) >= MAX_CLIENTS:
                raise RuntimeError("hub_full")
            if cid in self.clients:
                raise RuntimeError(f"duplicate cid {cid!r}")
            session = ClientSession(cid=cid)
            self.clients[cid] = session
            return session

    async def remove_client(self, cid: str) -> None:
        """Unsubscribe ``cid`` from everything and drop the session."""
        async with self._lock:
            session = self.clients.pop(cid, None)
            if session is None:
                return
            session.closed = True
            keys = list(session.subs)
        # Unsubscribe outside the lock — _maybe_stop_poller takes it again.
        for kind, slug in keys:
            await self.unsubscribe(cid, kind, slug)

    # ---- subscriptions ----------------------------------------------------

    async def subscribe(self, cid: str, kind: str, slug: str) -> None:
        """Add ``(kind, slug)`` to ``cid``'s subscriptions, starting the
        poller if it isn't already running.

        Raises ``ValueError`` for unknown ``kind``, ``RuntimeError`` if
        the client already has :data:`MAX_SUBS_PER_CLIENT` subs.
        """
        if kind not in self._poller_fns:
            raise ValueError(f"unsupported kind: {kind!r}")
        key = (kind, slug)
        async with self._lock:
            session = self.clients.get(cid)
            if session is None or session.closed:
                raise RuntimeError(f"no client {cid!r}")
            if key in session.subs:
                return  # idempotent
            if len(session.subs) >= MAX_SUBS_PER_CLIENT:
                raise RuntimeError("too_many_subs")
            session.subs.add(key)
            self.slug_subs[key].add(cid)
            # Start poller exactly once per (kind, slug).
            if key not in self.pollers:
                task = asyncio.create_task(
                    self._run_poller(kind, slug),
                    name=f"poller:{kind}:{slug}",
                )
                self.pollers[key] = task
            # If we already have a cached value, deliver it immediately so
            # the new subscriber doesn't have to wait a full poll cycle.
            cached = self.last_value.get(key)
        if cached is not None:
            self._enqueue(session, cached)

    async def unsubscribe(self, cid: str, kind: str, slug: str) -> None:
        """Remove ``(kind, slug)`` from ``cid``. Stops the poller when the
        last subscriber leaves."""
        key = (kind, slug)
        task_to_cancel: asyncio.Task | None = None
        async with self._lock:
            session = self.clients.get(cid)
            if session is not None:
                session.subs.discard(key)
            subs = self.slug_subs.get(key)
            if subs is not None:
                subs.discard(cid)
                if not subs:
                    self.slug_subs.pop(key, None)
                    task_to_cancel = self.pollers.pop(key, None)
                    self.last_value.pop(key, None)
        if task_to_cancel is not None:
            task_to_cancel.cancel()
            # Pollers swallow their own exceptions, but be defensive.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task_to_cancel

    # ---- poller coroutine -------------------------------------------------

    async def _run_poller(self, kind: str, slug: str) -> None:
        """Fixed-cadence poll of one ``(kind, slug)``. Cancelled by
        :meth:`unsubscribe` when the last subscriber leaves or by
        :meth:`shutdown`."""
        key = (kind, slug)
        fn = self._poller_fns[kind]
        http = self._get_http()
        try:
            while True:
                try:
                    evt = await fn(slug, http)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # Pollers must be resilient — never let a transient
                    # upstream error kill the task.
                    logger.exception("poller %s failed", key)
                    evt = None
                if evt is not None:
                    self.last_value[key] = evt
                    await self._fanout(key, evt)
                try:
                    await asyncio.sleep(self._poll_interval_s)
                except asyncio.CancelledError:
                    return
        except asyncio.CancelledError:
            return

    # ---- fanout -----------------------------------------------------------

    async def _fanout(self, key: tuple[str, str], evt: dict) -> None:
        """Push ``evt`` to every subscriber's queue, applying backpressure."""
        async with self._lock:
            cids = list(self.slug_subs.get(key, ()))
            sessions = [self.clients[c] for c in cids if c in self.clients]
        for session in sessions:
            self._enqueue(session, evt)

    def _enqueue(self, session: ClientSession, evt: dict) -> None:
        """Coalesce + drop-on-overflow.

        If the queue already holds an event for the same ``(kind, slug)``
        we replace it in-place (the consumer hasn't drained yet — they'll
        only ever get the latest). If coalescing fails we drop and count
        toward the slow-client threshold.
        """
        if session.closed:
            return
        # Coalesce: scan the underlying deque (O(queue_size) but bounded).
        # asyncio.Queue exposes its internal deque as `_queue`.
        target_key = (evt.get("type"), evt.get("slug"))
        try:
            internal: deque = session.queue._queue  # type: ignore[attr-defined]
            for i, existing in enumerate(internal):
                if (existing.get("type"), existing.get("slug")) == target_key:
                    internal[i] = evt
                    return
        except AttributeError:
            # Defensive: if asyncio internals change, fall through to put.
            pass
        try:
            session.queue.put_nowait(evt)
        except asyncio.QueueFull:
            self._record_drop(session)

    def _record_drop(self, session: ClientSession) -> None:
        now = time.monotonic()
        session.dropped += 1
        session.dropped_at.append(now)
        # Trim out-of-window timestamps.
        while session.dropped_at and session.dropped_at[0] < now - DROPPED_WINDOW_S:
            session.dropped_at.popleft()
        if len(session.dropped_at) >= DROPPED_THRESHOLD:
            # Mark closed and try to enqueue a final bye event.
            session.closed = True
            bye = {
                "type": "bye",
                "slug": "",
                "data": {"reason": "slow_client"},
                "ts": int(time.time()),
            }
            # The queue is full; rotate one slot for the bye.
            with contextlib.suppress(asyncio.QueueEmpty):
                session.queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                session.queue.put_nowait(bye)

    # ---- shutdown ---------------------------------------------------------

    async def shutdown(self) -> None:
        """Cancel every poller and close any owned HTTP client."""
        async with self._lock:
            tasks = list(self.pollers.values())
            self.pollers.clear()
            for session in self.clients.values():
                session.closed = True
            self.clients.clear()
            self.slug_subs.clear()
            self.last_value.clear()
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None


__all__ = [
    "DEFAULT_POLL_INTERVAL_S",
    "DROPPED_THRESHOLD",
    "DROPPED_WINDOW_S",
    "MAX_CLIENTS",
    "MAX_SUBS_PER_CLIENT",
    "QUEUE_MAXSIZE",
    "SUPPORTED_KINDS",
    "ClientSession",
    "RealtimeHub",
]

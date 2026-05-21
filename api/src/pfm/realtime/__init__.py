"""Realtime SSE multiplexing — one poller per (kind, slug), N clients fan out.

The legacy ``pfm.terminal_live_stream`` endpoint launches one poller per
SSE connection, which scales linearly with concurrent users (~20k req/min
at 100 users). This package replaces that pattern with a pub/sub hub:
each ``(kind, slug)`` gets at most one poller regardless of how many
clients have subscribed to it.

Public surface:

- :class:`pfm.realtime.hub.RealtimeHub`        — the in-process hub.
- :func:`pfm.realtime.stream.get_hub`          — FastAPI dependency.
- ``router`` from :mod:`pfm.realtime.stream`   — exposes ``GET /terminal/stream``.
"""

from __future__ import annotations

from pfm.realtime.hub import ClientSession, RealtimeHub
from pfm.realtime.stream import get_hub, router

__all__ = ["ClientSession", "RealtimeHub", "get_hub", "router"]

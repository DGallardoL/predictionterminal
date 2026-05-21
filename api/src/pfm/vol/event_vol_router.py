"""FastAPI router for the event-driven EM signal (module B3).

Mounted by ``pfm.main`` **only when** ``PFM_VOL_EVENT_ENABLED=1``. The
default-off gate keeps the new endpoints out of the OpenAPI spec for
the standard test suite (which spins up the app repeatedly through
``TestClient``) while letting us flip them on for live demos or staging.

Endpoints
---------
* ``GET /vol/event/calendar``           — curated upcoming events
* ``GET /vol/event/{event_id}/signal``  — live EM signal for one event
* ``GET /vol/event/signals``            — all upcoming signals in one shot
* ``GET /vol/event/kinds``              — supported ``event_kind`` taxonomy

The router reads the singleton Polymarket / Kalshi clients via the
shared DI helpers in :mod:`pfm.dependencies` so it doesn't depend on
``pfm.main`` at import time.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from pfm.cache_utils import get_cache
from pfm.dependencies import get_kalshi_client, get_polymarket_client
from pfm.sources.kalshi import KalshiClient
from pfm.sources.polymarket import PolymarketClient
from pfm.vol.event_calendar import CALENDAR, EventEntry, get_event, list_upcoming
from pfm.vol.event_signal import (
    EventSignal,
    compute_all_upcoming_signals,
    compute_event_signal,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vol/event", tags=["vol-event"])

# 5-minute TTL on per-event signals. Live midpoints don't move fast at
# this granularity and we don't want to hammer Gamma / Kalshi when
# many UI panels open the same event simultaneously.
_CACHE_TTL_S = 300
_CACHE = get_cache("event_signal", ttl=_CACHE_TTL_S)


# All ``event_kind`` literals the engine understands. Kept here so a UI
# can fetch the taxonomy without parsing the engine module's typing.
_SUPPORTED_KINDS: tuple[str, ...] = (
    "fomc",
    "cpi",
    "nfp",
    "election",
    "opec",
    "geopolitical",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_lookahead(lookahead_days: int) -> int:
    """422 if lookahead is outside ``[1, 180]``."""
    if lookahead_days < 1 or lookahead_days > 180:
        raise HTTPException(
            status_code=422,
            detail=f"lookahead_days must be in [1, 180], got {lookahead_days}",
        )
    return lookahead_days


def _asof_minute_bucket() -> int:
    """Quantise ``now`` into 5-minute buckets for cache keying."""
    now = datetime.now(tz=UTC)
    return int(now.timestamp() // _CACHE_TTL_S)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/calendar", response_model=list[EventEntry])
def get_calendar(
    lookahead_days: int = Query(default=30, ge=1, le=180),
    kind: str | None = Query(default=None, min_length=1, max_length=32),
) -> list[EventEntry]:
    """List curated upcoming events within the rolling window.

    Args:
        lookahead_days: Forward window length (1–180). Defaults to 30.
        kind: Optional ``event_kind`` filter (e.g. ``"fomc"``).
    """
    _validate_lookahead(lookahead_days)
    now = datetime.now(tz=UTC)
    upcoming = list_upcoming(now, lookahead_days=lookahead_days)
    if kind is not None:
        kind_lc = kind.strip().lower()
        upcoming = [e for e in upcoming if e.event_kind == kind_lc]
    return upcoming


@router.get("/kinds")
def get_kinds() -> dict[str, list[str]]:
    """Return the list of supported ``event_kind`` literals.

    Surfaced so a future UI can populate a "filter by event kind"
    dropdown without parsing the engine's ``Literal`` type.
    """
    # Also report which kinds are *actually present* in the curated
    # calendar — this is a strict subset of the supported taxonomy and
    # is more useful to a UI.
    present = sorted({entry.event_kind for entry in CALENDAR})
    return {
        "supported": list(_SUPPORTED_KINDS),
        "present": present,
    }


@router.get("/signals", response_model=list[EventSignal])
def get_signals(
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    kalshi: Annotated[KalshiClient, Depends(get_kalshi_client)],
    lookahead_days: int = Query(default=30, ge=1, le=180),
    kind: str | None = Query(default=None, min_length=1, max_length=32),
) -> list[EventSignal]:
    """Compute live EM signals for every upcoming event in the window.

    Partial failures (a single event's slugs all failing, etc.) are
    silently dropped — the returned list contains only signals that
    succeeded. Use the per-event endpoint to see why a specific event
    is missing.
    """
    _validate_lookahead(lookahead_days)
    now = datetime.now(tz=UTC)
    cache_key = ("signals", lookahead_days, kind or "", _asof_minute_bucket())
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return [EventSignal.model_validate(item) for item in cached]

    signals = compute_all_upcoming_signals(
        now_utc=now,
        lookahead_days=lookahead_days,
        polymarket_client=poly,
        kalshi_client=kalshi,
    )
    if kind is not None:
        kind_lc = kind.strip().lower()
        signals = [s for s in signals if s.event_kind == kind_lc]

    _CACHE.set(cache_key, [s.model_dump(mode="json") for s in signals], ttl=_CACHE_TTL_S)
    return signals


@router.get("/{event_id}/signal", response_model=EventSignal)
def get_event_signal(
    event_id: str,
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
    kalshi: Annotated[KalshiClient, Depends(get_kalshi_client)],
) -> EventSignal:
    """Compute the live EM signal for one event.

    Returns:
        :class:`EventSignal` with the live distribution, EM forecast,
        and fetch-completeness diagnostic.

    Raises:
        HTTPException 404: ``event_id`` is not in the curated calendar.
        HTTPException 502: every outcome-slug fetch failed.
    """
    if get_event(event_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown event_id: {event_id!r}")

    cache_key = ("signal", event_id, _asof_minute_bucket())
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return EventSignal.model_validate(cached)

    try:
        signal = compute_event_signal(
            event_id,
            polymarket_client=poly,
            kalshi_client=kalshi,
        )
    except KeyError as exc:
        # Belt-and-braces; the pre-check above already covers this.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"event {event_id!r}: all source fetches failed ({exc})",
        ) from exc

    _CACHE.set(cache_key, signal.model_dump(mode="json"), ttl=_CACHE_TTL_S)
    return signal


# Stable export for ``main.py`` to mount under the feature flag.
__all__ = ["router"]


# Re-export the kind literal alias so callers can introspect statically.
EventKindLiteral = Literal["fomc", "cpi", "nfp", "election", "opec", "geopolitical"]

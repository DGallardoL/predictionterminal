"""FastAPI router for the public Alpha Graveyard.

Endpoints
---------
* ``GET /alpha-hub/graveyard`` — list all dead / downgraded alphas, optionally
  filtered by ``cause``.
* ``GET /alpha-hub/graveyard/{pair_id}`` — fetch a single death certificate.

The router is *not* auto-mounted in ``main.py`` — Damian wires it himself.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from pfm.alpha_graveyard import (
    GraveyardCauseFilter,
    GraveyardEntry,
    GraveyardResponse,
    filter_by_cause,
    load_graveyard,
)

router = APIRouter(prefix="/alpha-hub", tags=["alpha-graveyard"])


@router.get(
    "/graveyard",
    response_model=GraveyardResponse,
    summary="List dead / downgraded alpha strategies",
)
def get_graveyard(
    cause: Annotated[
        GraveyardCauseFilter,
        Query(description="Filter by failure mode; 'all' returns the full list."),
    ] = "all",
) -> GraveyardResponse:
    """Return all entries in the alpha graveyard, optionally filtered by cause.

    The response is *intentionally* not paginated — the graveyard is small by
    design (one entry per killed alpha), and the front-end renders the full
    list as a scrollable cemetery view.
    """
    raw = load_graveyard()
    filtered = filter_by_cause(raw, cause)
    entries = [GraveyardEntry.model_validate(e) for e in filtered]
    return GraveyardResponse(n_entries=len(entries), cause_filter=cause, entries=entries)


@router.get(
    "/graveyard/{pair_id}",
    response_model=GraveyardEntry,
    summary="Fetch a single death certificate",
)
def get_graveyard_entry(pair_id: str) -> GraveyardEntry:
    """Return the death certificate for a single ``pair_id``.

    Raises HTTP 404 if the ``pair_id`` is not in the graveyard.
    """
    raw = load_graveyard()
    for e in raw:
        if e.get("pair_id") == pair_id:
            return GraveyardEntry.model_validate(e)
    raise HTTPException(status_code=404, detail=f"pair_id '{pair_id}' not found in graveyard")

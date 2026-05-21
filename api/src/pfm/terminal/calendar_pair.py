"""Calendar-pair-surface endpoint for the Terminal panel.

A *calendar pair* is two (or more) prediction-market contracts that resolve
the **same event** at **different deadlines** — e.g.::

    "Trump out as President by Jun 30 2026"   (deadline = 2026-06-30)
    "Trump out as President before 2027"       (deadline = 2026-12-31)

The two contracts share a single underlying event-token but trade at
different prices because the longer deadline gives more time for the event
to occur. Under a constant-hazard prior the implied hazard rate

    λ = -ln(1 - p) / T

should be **the same** at every deadline. Large dispersion in λ across
deadlines flags a Strategy-24 calendar-arbitrage opportunity (see
``/tmp/strat28_calendar_revalid.json`` for the full backtest).

Endpoint
--------

``GET /terminal/calendar-pair/{slug}``

* If the slug is part of a calendar pair → return the term-structure
  surface (sorted by deadline) plus the cross-deadline λ-ratio and the
  Strategy-24 trade-eligibility flag.
* Otherwise → return ``null``.

Routing note: this module owns its :class:`fastapi.APIRouter`, mirroring
``terminal_equity`` / ``terminal_trades``, so ``main.py`` is left untouched.
``main.py`` only needs::

    from pfm.terminal_calendar_pair import router as terminal_calendar_pair_router
    app.include_router(terminal_calendar_pair_router)
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException
from fastapi import Path as FPath
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache

logger = logging.getLogger(__name__)

# Shared cache for the (slug→event, event→legs, n_pairs) triple. Keyed by
# the resolved (strat28_path, strat2_path) pair so tests that swap paths
# get a fresh entry. 30-minute TTL is generous for a file-backed lookup
# rebuilt on demand by ``reload_lookup``.
_CAL_PAIR_CACHE = get_cache("calendar_pair", ttl=1800)

# --- constants --------------------------------------------------------------

# Path to the revalidated calendar-pair backtest. Overridable from tests
# via ``terminal_calendar_pair.STRAT28_PATH = Path(...)`` followed by
# :func:`reload_lookup`.
STRAT28_PATH: Path = Path("/tmp/strat28_calendar_revalid.json")
# The strat-2 file is used to recover ``id → slug`` aliases so the endpoint
# can answer queries by either form.
STRAT2_PATH: Path = Path("/tmp/strat2_calendar_arb.json")

# Strategy-24 trade-eligibility threshold on |log(λ_far / λ_near)|.
LOG_LAMBDA_RATIO_THRESHOLD: float = 0.50

# Reference "today" used to infer end-dates from days-to-resolution when
# the underlying file does not record an absolute deadline. Mirrors
# ``meta.today`` of the backtest file.
DEFAULT_TODAY: str = "2026-05-02"


# --- schemas ----------------------------------------------------------------


class CalendarLeg(BaseModel):
    """One contract on the calendar surface."""

    slug: str = Field(..., description="Polymarket slug or synthetic id-slug.")
    deadline: str = Field(..., description="ISO-8601 deadline date (UTC).")
    current_p: float = Field(..., ge=0.0, le=1.0)
    days_to_resolution: int = Field(..., ge=0)
    implied_lambda: float = Field(
        ...,
        description="Constant-hazard rate λ = -ln(1 - p) / T (per day).",
    )


class CalendarPairResponse(BaseModel):
    """Term-structure surface for a calendar pair."""

    slug: str
    event_token: str
    surface: list[CalendarLeg] = Field(
        ...,
        description="All legs sorted by deadline ascending.",
    )
    lambda_near: float
    lambda_far: float
    log_lambda_ratio: float = Field(
        ...,
        description="ln(λ_far / λ_near) — positive ⇒ far deadline cheaper "
        "in hazard terms (i.e. expensive in price terms).",
    )
    trade_eligible: bool = Field(
        ...,
        description=(
            "True when |log(λ_far / λ_near)| ≥ "
            f"{LOG_LAMBDA_RATIO_THRESHOLD} (Strategy-24 threshold)."
        ),
    )


# --- lookup -----------------------------------------------------------------


def _id_to_synthetic_slug(member_id: str) -> str:
    """Turn ``trump_out_2027`` into ``trump-out-2027``.

    Used as a fallback for IDs that don't appear in the strat-2 ``id → slug``
    table (the strat-28 file is keyed by id only).
    """
    return member_id.replace("_", "-")


def _infer_deadline(dtr: int, today: str = DEFAULT_TODAY) -> str:
    """Convert *days-to-resolution* into an ISO date relative to ``today``."""
    base = datetime.fromisoformat(today).date()
    return (base.fromordinal(base.toordinal() + int(dtr))).isoformat()


def _implied_lambda(p: float, days: int) -> float:
    """Constant-hazard rate λ such that 1 - exp(-λ T) = p.

    Returns ``0.0`` for degenerate inputs (p ≤ 0, T ≤ 0). Clips ``p`` just
    below 1 to keep the logarithm finite.
    """
    if days <= 0 or p <= 0.0:
        return 0.0
    p_clipped = min(p, 0.999_999)
    return -math.log(1.0 - p_clipped) / float(days)


def _build_lookup(
    strat28_path: Path | None = None,
    strat2_path: Path | None = None,
) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]], int]:
    """Load both backtest files and return ``(slug→event, event→legs, n_pairs)``.

    Defaults are resolved from the module-level :data:`STRAT28_PATH` /
    :data:`STRAT2_PATH` at *call* time — not at function-def time — so
    tests that rebind those names via ``monkeypatch.setattr`` take effect.

    The slug index resolves both the canonical Polymarket slug (from
    strat-2) and the synthetic id-slug (from strat-28) so the endpoint
    answers either form.
    """
    if strat28_path is None:
        strat28_path = STRAT28_PATH
    if strat2_path is None:
        strat2_path = STRAT2_PATH
    if not strat28_path.exists():
        logger.warning("strat28 file missing at %s — calendar lookup empty", strat28_path)
        return {}, {}, 0

    with strat28_path.open() as f:
        strat28 = json.load(f)

    today = strat28.get("meta", {}).get("today", DEFAULT_TODAY)
    n_pairs = int(strat28.get("meta", {}).get("n_pairs", 0))

    # --- recover id → slug from strat-2 (best-effort; file may be absent) ---
    id_to_slug: dict[str, str] = {}
    id_to_end_date: dict[str, str] = {}
    if strat2_path.exists():
        with strat2_path.open() as f:
            strat2 = json.load(f)
        for cluster in strat2.get("clusters", []):
            for member in cluster.get("members", []):
                mid_id = member.get("id")
                if not mid_id:
                    continue
                if member.get("slug"):
                    id_to_slug[mid_id] = member["slug"]
                if member.get("end_date"):
                    id_to_end_date[mid_id] = member["end_date"]

    # --- accumulate legs from strat-28 pair data ----------------------------
    # event_token → { member_id → leg-dict } (dedupe via the inner dict)
    event_legs: dict[str, dict[str, dict[str, Any]]] = {}

    pair_records: list[dict[str, Any]] = []
    pair_records.extend(strat28.get("pairs_sample", []))

    for pair in pair_records:
        event = pair.get("event")
        if not event:
            continue
        for side_key in ("short", "long"):
            side = pair.get(side_key)
            if not side:
                continue
            mid_id = side["id"]
            slug = id_to_slug.get(mid_id, _id_to_synthetic_slug(mid_id))
            dtr = int(side.get("dtr", 0))
            deadline = id_to_end_date.get(mid_id) or _infer_deadline(dtr, today)
            mid = float(side.get("mid", 0.0))
            event_legs.setdefault(event, {})[mid_id] = {
                "slug": slug,
                "deadline": deadline,
                "current_p": mid,
                "days_to_resolution": dtr,
                "implied_lambda": _implied_lambda(mid, dtr),
            }

    # Add any strat-2 cluster members the strat-28 file didn't enumerate so
    # canonical slugs (xi-jinping-out-by, trump-out-..., …) all resolve.
    if strat2_path.exists():
        with strat2_path.open() as f:
            strat2 = json.load(f)
        ref_today = datetime.fromisoformat(today).date()
        for cluster in strat2.get("clusters", []):
            event = cluster.get("signature")
            if not event:
                continue
            for member in cluster.get("members", []):
                mid_id = member.get("id")
                slug = member.get("slug") or (_id_to_synthetic_slug(mid_id) if mid_id else None)
                if not slug:
                    continue
                end_date = member.get("end_date")
                dtr = max(0, (date.fromisoformat(end_date) - ref_today).days) if end_date else 0
                mid = float(member.get("mid", 0.0))
                key = mid_id or slug
                event_legs.setdefault(event, {}).setdefault(
                    key,
                    {
                        "slug": slug,
                        "deadline": end_date or _infer_deadline(dtr, today),
                        "current_p": mid,
                        "days_to_resolution": dtr,
                        "implied_lambda": _implied_lambda(mid, dtr),
                    },
                )

    # --- collapse and build slug → event index ------------------------------
    event_to_legs: dict[str, list[dict[str, Any]]] = {}
    slug_to_event: dict[str, str] = {}
    for event, legs in event_legs.items():
        if len(legs) < 2:
            # A "pair" requires ≥ 2 legs at different deadlines.
            continue
        sorted_legs = sorted(legs.values(), key=lambda d: d["deadline"])
        event_to_legs[event] = sorted_legs
        for leg in sorted_legs:
            slug_to_event[leg["slug"]] = event

    return slug_to_event, event_to_legs, n_pairs


# Module-level cache populated at import time. Tests that rebind
# ``STRAT28_PATH`` should call :func:`reload_lookup`.
_SLUG_TO_EVENT: dict[str, str]
_EVENT_TO_LEGS: dict[str, list[dict[str, Any]]]
_N_PAIRS: int


def reload_lookup() -> int:
    """(Re)build the in-memory lookup. Returns the ``meta.n_pairs`` count.

    Routes the rebuilt triple through the shared ``calendar_pair`` cache
    so other modules can introspect / clear it via the
    :mod:`pfm.cache_utils` namespace.
    """
    global _SLUG_TO_EVENT, _EVENT_TO_LEGS, _N_PAIRS
    triple = _build_lookup()
    _CAL_PAIR_CACHE.clear()
    _CAL_PAIR_CACHE.set((str(STRAT28_PATH), str(STRAT2_PATH)), triple)
    _SLUG_TO_EVENT, _EVENT_TO_LEGS, _N_PAIRS = triple
    return _N_PAIRS


reload_lookup()


# --- router -----------------------------------------------------------------

router = APIRouter(prefix="/terminal", tags=["terminal-calendar-pair"])


@router.get("/calendar-pair/{slug}")
def get_calendar_pair(
    slug: Annotated[str, FPath(min_length=1, max_length=200)],
) -> CalendarPairResponse | None:
    """Return the calendar-pair surface for ``slug`` or ``null``.

    Response shape on a hit::

        {
          "slug": "trump-out-as-president-by-june-30",
          "event_token": "out president trump",
          "surface": [
            {"slug": "...", "deadline": "2026-06-30", "current_p": 0.0235,
             "days_to_resolution": 59, "implied_lambda": 4.03e-4},
            {"slug": "...", "deadline": "2026-12-31", ...}
          ],
          "lambda_near": 4.03e-4,
          "lambda_far": 5.94e-4,
          "log_lambda_ratio": 0.388,
          "trade_eligible": false
        }

    On a miss the endpoint returns ``null`` (HTTP 200) — the frontend
    distinguishes "no pair" from "unknown slug" via this convention.
    """
    event = _SLUG_TO_EVENT.get(slug)
    if event is None:
        return None

    legs = _EVENT_TO_LEGS.get(event, [])
    if len(legs) < 2:
        return None

    # Defensive sort — the cache is already sorted by deadline.
    surface = sorted(legs, key=lambda d: d["deadline"])
    near = surface[0]
    far = surface[-1]

    lam_near = float(near["implied_lambda"])
    lam_far = float(far["implied_lambda"])

    # Degenerate (any leg with p=0 ⇒ λ undefined): surface the term-structure
    # anyway, but report a 0-ratio so the trade-eligibility flag stays False.
    log_ratio = 0.0 if (lam_near <= 0.0 or lam_far <= 0.0) else math.log(lam_far / lam_near)

    trade_eligible = abs(log_ratio) >= LOG_LAMBDA_RATIO_THRESHOLD

    try:
        return CalendarPairResponse(
            slug=slug,
            event_token=event,
            surface=[CalendarLeg(**leg) for leg in surface],
            lambda_near=lam_near,
            lambda_far=lam_far,
            log_lambda_ratio=log_ratio,
            trade_eligible=trade_eligible,
        )
    except Exception as e:  # pragma: no cover - schema construction is total
        raise HTTPException(
            status_code=500,
            detail=f"failed to build calendar-pair response: {e}",
        ) from e

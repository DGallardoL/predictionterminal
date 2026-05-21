"""Terminal watchlist + alerts backend.

Backs a previously localStorage-only UI with a small server-side store so a
user's watched markets follow them across browsers/devices. Storage is a
single JSON file per user under ``/tmp/watchlists/{user_id}.json`` — no
database, no auth; a string ``user_id`` (default ``"default"``) is enough
for the POC.

Endpoints
---------
* ``POST   /terminal/watchlist``                  add a slug (idempotent)
* ``DELETE /terminal/watchlist/{user_id}/{slug}``  remove a slug
* ``GET    /terminal/watchlist/{user_id}``         list with current price + z-score
* ``GET    /terminal/watchlist/{user_id}/alerts``  currently-triggered z-score alerts

The z-score is computed as ``(current_p - mean) / std`` over the most recent
``WINDOW`` daily closes from Polymarket's ``/prices-history`` endpoint.
``alert_z`` is an absolute threshold (e.g. ``2.0`` ⇒ alert if ``|z| >= 2``).

External I/O is wrapped in two small seams (``_fetch_current_price`` and
``_fetch_price_history``) so tests can monkeypatch them and avoid live
HTTP calls.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/terminal/watchlist", tags=["terminal"])


GAMMA_URL: str = "https://gamma-api.polymarket.com"
CLOB_URL: str = "https://clob.polymarket.com"

# Where per-user watchlist JSONs live.
WATCHLIST_DIR: Path = Path("/tmp/watchlists")

# Z-score lookback (daily closes). 30 is a reasonable POC default — long
# enough to be statistically meaningful, short enough to react to regime
# changes within ~a month.
ZSCORE_WINDOW: int = 30

DEFAULT_USER_ID: str = "default"


# ──────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────


class WatchlistAddRequest(BaseModel):
    """Body for ``POST /terminal/watchlist``."""

    user_id: str = Field(default=DEFAULT_USER_ID, min_length=1, max_length=128)
    slug: str = Field(..., min_length=1, max_length=256)
    alert_z: float | None = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Absolute z-score threshold; alert fires when |z| >= alert_z.",
    )


class WatchlistEntry(BaseModel):
    """One row in a user's watchlist (storage representation)."""

    slug: str
    alert_z: float | None = None


class WatchlistItem(BaseModel):
    """One row in the GET response: storage row enriched with live data."""

    slug: str
    alert_z: float | None = None
    current_p: float | None = Field(
        default=None, description="Current YES probability from Gamma; null on fetch failure."
    )
    z_score: float | None = Field(
        default=None,
        description=f"(current_p - mean) / std over the last {ZSCORE_WINDOW} daily closes.",
    )
    alert_triggered: bool = Field(
        default=False, description="True iff alert_z is set and |z_score| >= alert_z."
    )


class WatchlistResponse(BaseModel):
    user_id: str
    n_items: int
    items: list[WatchlistItem]


class WatchlistQuoteItem(BaseModel):
    """One row in the ``/terminal/watchlist/quotes`` response.

    Surfaced by the sidebar watchlist widget — independent of any stored
    per-user list. The widget passes a CSV of slugs and we look each up in
    the gamma prewarm cache (no network round-trip per row) so the panel
    renders instantly.
    """

    slug: str
    name: str | None = None
    theme: str | None = None
    price: float | None = None
    change_24h: float | None = None
    volume_24h: float | None = None


class WatchlistQuotesResponse(BaseModel):
    n_items: int
    items: list[WatchlistQuoteItem]


class AlertsResponse(BaseModel):
    user_id: str
    n_alerts: int
    alerts: list[WatchlistItem]


class WatchlistAddResponse(BaseModel):
    user_id: str
    slug: str
    alert_z: float | None = None
    added: bool = Field(..., description="False if the slug was already present (idempotent).")


class WatchlistDeleteResponse(BaseModel):
    user_id: str
    slug: str
    removed: bool


# ──────────────────────────────────────────────────────────────────────────
# Storage — JSON-per-user
# ──────────────────────────────────────────────────────────────────────────


def _user_path(user_id: str) -> Path:
    """Return the on-disk path for a given ``user_id`` (creates parent dir)."""
    safe = user_id.strip()
    if not safe or "/" in safe or ".." in safe:
        raise HTTPException(status_code=400, detail=f"invalid user_id: {user_id!r}")
    WATCHLIST_DIR.mkdir(parents=True, exist_ok=True)
    return WATCHLIST_DIR / f"{safe}.json"


def _load(user_id: str) -> list[WatchlistEntry]:
    """Load a user's watchlist; empty list if no file or unreadable."""
    p = _user_path(user_id)
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("watchlist read failed for user=%s: %s", user_id, exc)
        return []
    if not isinstance(raw, list):
        return []
    out: list[WatchlistEntry] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        slug = row.get("slug")
        if not isinstance(slug, str) or not slug:
            continue
        alert_z = row.get("alert_z")
        if alert_z is not None:
            try:
                alert_z = float(alert_z)
            except (TypeError, ValueError):
                alert_z = None
        out.append(WatchlistEntry(slug=slug, alert_z=alert_z))
    return out


def _save(user_id: str, entries: list[WatchlistEntry]) -> None:
    """Persist a user's watchlist atomically (write-tmp + rename)."""
    p = _user_path(user_id)
    tmp = p.with_suffix(".json.tmp")
    payload = [e.model_dump() for e in entries]
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(p)


# ──────────────────────────────────────────────────────────────────────────
# External fetch seams (tests monkeypatch these)
# ──────────────────────────────────────────────────────────────────────────


def _fetch_current_price(slug: str, client: httpx.Client) -> float | None:
    """Get the current YES probability for ``slug`` via Gamma; ``None`` on failure."""
    try:
        r = client.get(f"{GAMMA_URL}/markets", params={"slug": slug})
        r.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("gamma fetch failed for slug=%s: %s", slug, exc)
        return None
    payload = r.json()
    if not isinstance(payload, list) or not payload:
        return None
    market = payload[0]
    last = market.get("lastTradePrice")
    if last is not None:
        try:
            return max(0.0, min(1.0, float(last)))
        except (TypeError, ValueError):
            pass
    raw = market.get("outcomePrices")
    if isinstance(raw, str):
        try:
            arr = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(arr, list) and arr:
            try:
                return max(0.0, min(1.0, float(arr[0])))
            except (TypeError, ValueError):
                return None
    return None


def _fetch_price_history(slug: str, client: httpx.Client) -> list[float]:
    """Get recent daily YES closes for ``slug`` via the CLOB; empty on failure.

    We hit Gamma first to extract the YES ``clobTokenId`` (which arrives as a
    JSON-encoded string inside the JSON response — the classic Polymarket
    quirk), then call ``/prices-history?market=<token>&fidelity=1440``.
    """
    try:
        meta = client.get(f"{GAMMA_URL}/markets", params={"slug": slug})
        meta.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("gamma meta fetch failed for slug=%s: %s", slug, exc)
        return []
    payload = meta.json()
    if not isinstance(payload, list) or not payload:
        return []
    raw_ids = payload[0].get("clobTokenIds")
    token_id: str | None = None
    if isinstance(raw_ids, str):
        try:
            ids = json.loads(raw_ids)
        except json.JSONDecodeError:
            ids = None
        if isinstance(ids, list) and ids:
            token_id = str(ids[0])
    if token_id is None:
        return []

    try:
        r = client.get(
            f"{CLOB_URL}/prices-history",
            params={"market": token_id, "fidelity": 1440},
        )
        r.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("clob price-history fetch failed for slug=%s: %s", slug, exc)
        return []
    body = r.json()
    history = body.get("history") if isinstance(body, dict) else None
    if not isinstance(history, list):
        return []
    out: list[float] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        p = row.get("p")
        try:
            out.append(float(p))
        except (TypeError, ValueError):
            continue
    return out


def _new_http_client(timeout: float = 10.0) -> httpx.Client:
    return httpx.Client(timeout=timeout)


# ──────────────────────────────────────────────────────────────────────────
# Pure helpers — z-score + alert evaluation
# ──────────────────────────────────────────────────────────────────────────


def compute_z_score(
    current: float, history: list[float], window: int = ZSCORE_WINDOW
) -> float | None:
    """Return ``(current - mean) / std`` over the last ``window`` history values.

    Returns ``None`` if the history is too short or has zero variance — both
    cases mean the z-score is undefined / not actionable.
    """
    if not history or len(history) < 2:
        return None
    sample = history[-window:] if len(history) > window else list(history)
    if len(sample) < 2:
        return None
    n = len(sample)
    mean = sum(sample) / n
    # Sample standard deviation (ddof=1) — matches numpy/pandas default for z-scores.
    var = sum((x - mean) ** 2 for x in sample) / (n - 1)
    if var <= 0.0:
        return None
    std = math.sqrt(var)
    return (current - mean) / std


def is_alert_triggered(z_score: float | None, alert_z: float | None) -> bool:
    """``True`` iff both z and threshold are set and ``|z| >= alert_z``."""
    if z_score is None or alert_z is None:
        return False
    return abs(z_score) >= alert_z


def _enrich(entries: list[WatchlistEntry]) -> list[WatchlistItem]:
    """Pull current price + price history for each entry and attach z-score / alert flag."""
    items: list[WatchlistItem] = []
    if not entries:
        return items
    with _new_http_client() as client:
        for e in entries:
            cur = _fetch_current_price(e.slug, client)
            hist = _fetch_price_history(e.slug, client) if cur is not None else []
            z = compute_z_score(cur, hist) if cur is not None else None
            items.append(
                WatchlistItem(
                    slug=e.slug,
                    alert_z=e.alert_z,
                    current_p=cur,
                    z_score=z,
                    alert_triggered=is_alert_triggered(z, e.alert_z),
                )
            )
    return items


# ──────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────


@router.post("", response_model=WatchlistAddResponse)
def add_to_watchlist(req: WatchlistAddRequest) -> WatchlistAddResponse:
    """Add ``slug`` to ``user_id``'s watchlist (idempotent).

    If the slug is already present, ``alert_z`` is updated in-place and
    ``added`` returns ``False``.
    """
    entries = _load(req.user_id)
    existing = next((e for e in entries if e.slug == req.slug), None)
    if existing is not None:
        existing.alert_z = req.alert_z
        _save(req.user_id, entries)
        return WatchlistAddResponse(
            user_id=req.user_id, slug=req.slug, alert_z=req.alert_z, added=False
        )
    entries.append(WatchlistEntry(slug=req.slug, alert_z=req.alert_z))
    _save(req.user_id, entries)
    return WatchlistAddResponse(user_id=req.user_id, slug=req.slug, alert_z=req.alert_z, added=True)


@router.delete("/{user_id}/{slug}", response_model=WatchlistDeleteResponse)
def remove_from_watchlist(user_id: str, slug: str) -> WatchlistDeleteResponse:
    """Remove ``slug`` from ``user_id``'s watchlist; ``removed=False`` if it wasn't there."""
    entries = _load(user_id)
    n_before = len(entries)
    entries = [e for e in entries if e.slug != slug]
    if len(entries) == n_before:
        return WatchlistDeleteResponse(user_id=user_id, slug=slug, removed=False)
    _save(user_id, entries)
    return WatchlistDeleteResponse(user_id=user_id, slug=slug, removed=True)


@router.get("/quotes", response_model=WatchlistQuotesResponse)
def watchlist_quotes(
    request: Request,
    slugs: str = Query(
        ...,
        max_length=4000,
        description="Comma-separated Polymarket slugs (max 50).",
    ),
) -> WatchlistQuotesResponse:
    """Bulk-quote a CSV of slugs from the sidebar watchlist widget.

    Reads from the in-memory ``gamma_prices``/``gamma_volumes`` cache that
    the price-prewarm task refreshes every 60 s, so this responds in <1 ms
    and never hits Polymarket directly.

    .. note::
       This route is declared **before** the ``/{user_id}`` route on purpose:
       FastAPI dispatches in registration order and ``/quotes`` would
       otherwise be matched as ``user_id="quotes"`` and silently return an
       empty list (the original UX-audit bug).
    """
    requested = [s.strip() for s in (slugs or "").split(",") if s.strip()]
    if not requested:
        return WatchlistQuotesResponse(n_items=0, items=[])
    if len(requested) > 50:
        raise HTTPException(status_code=400, detail="max 50 slugs per request")

    state = request.app.state
    prices: dict[str, float] = getattr(state, "gamma_prices", None) or {}
    volumes: dict[str, float] = getattr(state, "gamma_volumes", None) or {}
    # Factor catalogue carries the friendly ``name``/``theme`` for slugs that
    # we curate. Missing slugs degrade gracefully (price-only row).
    factors = getattr(state, "factors", None) or {}
    slug_to_meta: dict[str, tuple[str | None, str | None]] = {}
    for fc in factors.values():
        slug_to_meta[fc.slug] = (fc.name, fc.theme)

    items: list[WatchlistQuoteItem] = []
    for slug in requested:
        name, theme = slug_to_meta.get(slug, (None, None))
        items.append(
            WatchlistQuoteItem(
                slug=slug,
                name=name,
                theme=theme,
                price=prices.get(slug),
                # 24h price change is not in the prewarm payload; left as
                # None — the widget shows a dash. Wiring this would require
                # a second prewarm cache, out of scope for the audit fix.
                change_24h=None,
                volume_24h=volumes.get(slug),
            )
        )
    return WatchlistQuotesResponse(n_items=len(items), items=items)


@router.get("/{user_id}", response_model=WatchlistResponse)
def list_watchlist(user_id: str) -> WatchlistResponse:
    """List ``user_id``'s watchlist with current prices and z-scores."""
    entries = _load(user_id)
    items = _enrich(entries)
    return WatchlistResponse(user_id=user_id, n_items=len(items), items=items)


@router.get("/{user_id}/alerts", response_model=AlertsResponse)
def list_triggered_alerts(user_id: str) -> AlertsResponse:
    """Return only the watchlist rows whose z-score has breached their ``alert_z``."""
    entries = _load(user_id)
    items = _enrich(entries)
    triggered = [it for it in items if it.alert_triggered]
    return AlertsResponse(user_id=user_id, n_alerts=len(triggered), alerts=triggered)


# Re-export marker for type-checkers / static tools
__all__ = [
    "DEFAULT_USER_ID",
    "WATCHLIST_DIR",
    "ZSCORE_WINDOW",
    "AlertsResponse",
    "WatchlistAddRequest",
    "WatchlistEntry",
    "WatchlistItem",
    "WatchlistResponse",
    "compute_z_score",
    "is_alert_triggered",
    "router",
]


def _ignore_unused(*args: Any) -> None:  # pragma: no cover
    _ = args

"""Polymarket whale tracker — surface large positions and recent large trades.

Two endpoints, both backed by the public ``data-api.polymarket.com`` service:

  - ``GET /terminal/whales/{slug}`` — aggregate large *positions* per
    address for a single market. Useful for "who is long?" questions and
    for spotting net directional skew in the smart-money cohort.

  - ``GET /terminal/whales/recent-large-trades`` — large *trades* across
    a single market in the last N hours. Useful as an
    informed-flow-like signal ("X just bought $50k YES").

The endpoints are deliberately read-only and lightweight: no caching, no
persistence, no auth. They simply fan out to data-api, filter, aggregate,
and return JSON.

A few notes on the data model:

  - Polymarket's positions endpoint returns one row per (address, asset)
    where ``asset`` is a CLOB token id. Each market has two tokens
    (YES + NO) so a single whale may appear in two rows. We aggregate
    per address into ``position_yes_usd`` / ``position_no_usd`` and a
    ``net_usd = yes - no`` directional summary.

  - YES vs NO is identified by the ``outcome`` field on the position
    row when present (``"Yes"`` / ``"No"``); otherwise we fall back to
    matching the asset id against the market's ``clobTokenIds``.

  - ``net_directional_skew`` is YES-share of gross notional:
    ``Σ yes / (Σ yes + Σ no)``. A value > 0.5 means the whale cohort is
    net long YES on the market.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated

import httpx
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

from pfm.sources.polymarket import PolymarketClient

logger = logging.getLogger(__name__)

POLY_DATA_API: str = "https://data-api.polymarket.com"
POSITIONS_PAGE_CAP: int = 200  # data-api caps at ~500; 200 is plenty for whales.

router = APIRouter(prefix="/terminal", tags=["terminal"])


# --- schemas ----------------------------------------------------------------


class Whale(BaseModel):
    """One aggregated whale position in a single market."""

    address: str = Field(..., description="Wallet address (0x…).")
    position_yes_usd: float = Field(..., ge=0.0, description="Notional held in YES.")
    position_no_usd: float = Field(..., ge=0.0, description="Notional held in NO.")
    net_usd: float = Field(..., description="position_yes_usd minus position_no_usd.")
    last_active_iso: str | None = Field(
        None, description="ISO-8601 UTC timestamp of last activity, if known."
    )
    n_trades_24h: int = Field(0, ge=0, description="Trades by this address in last 24h.")


class WhalesResponse(BaseModel):
    slug: str
    condition_id: str
    n_whales: int = Field(..., ge=0)
    total_whale_notional_usd: float = Field(..., ge=0.0)
    whales: list[Whale]
    net_directional_skew: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Σ YES notional / (Σ YES + Σ NO). 0.5 = balanced; >0.5 = YES-leaning.",
    )
    interpretation: str


class LargeTrade(BaseModel):
    """One recent large trade across the whale cohort."""

    timestamp_iso: str
    address: str | None = None
    side: str | None = Field(None, description="BUY / SELL if reported by data-api.")
    outcome: str | None = Field(None, description="YES / NO if reported.")
    price: float = Field(..., ge=0.0, le=1.0)
    size_shares: float = Field(..., ge=0.0)
    size_usd: float = Field(..., ge=0.0)


class RecentLargeTradesResponse(BaseModel):
    slug: str
    condition_id: str
    hours: int
    min_size_usd: float
    n_trades: int
    total_notional_usd: float
    trades: list[LargeTrade]


# --- dependencies -----------------------------------------------------------


def get_polymarket_client(request: Request) -> PolymarketClient:
    """Resolve the shared ``PolymarketClient`` from app state.

    Mirrors the convention used by the rest of the terminal routers — keeps
    this module decoupled from ``main.py`` so it can be wired up by a single
    ``include_router`` call without circular imports.
    """
    poly = getattr(request.app.state, "poly", None)
    if poly is None:
        raise HTTPException(status_code=503, detail="polymarket client not initialized")
    return poly


# --- core aggregation -------------------------------------------------------


def aggregate_whales(
    raw_positions: list[dict],
    yes_token_id: str | None,
    no_token_id: str | None,
    min_position_usd: float,
    limit: int,
) -> list[Whale]:
    """Aggregate raw position rows into per-address Whale records.

    Each input row should look roughly like::

        {
            "proxyWallet": "0xabc…",
            "asset": "<clobTokenId>",
            "outcome": "Yes" | "No",
            "size": <shares>,
            "currentValue": <usd notional>,
            "avgPrice": <0..1>,
        }

    Behavior:
      - YES/NO is identified preferentially by the ``outcome`` field, falling
        back to matching ``asset`` against the supplied token ids.
      - Per-row notional is taken from ``currentValue`` if present, else
        ``size * avgPrice``.
      - The address-level filter applies to *gross* notional
        (yes + no), so a whale long $9k YES + $9k NO ($18k gross) clears a
        $10k threshold.
    """
    by_addr: dict[str, dict[str, float]] = {}
    for row in raw_positions:
        addr = _coalesce_str(row, ("proxyWallet", "user", "address", "owner", "wallet"))
        if not addr:
            continue
        side_yes = _classify_yes_no(row, yes_token_id, no_token_id)
        if side_yes is None:
            # Couldn't tell YES from NO — skip rather than miscount.
            continue
        notional = _row_notional_usd(row)
        if notional <= 0.0:
            continue
        bucket = by_addr.setdefault(addr, {"yes": 0.0, "no": 0.0})
        bucket["yes" if side_yes else "no"] += notional

    whales: list[Whale] = []
    for addr, sides in by_addr.items():
        gross = sides["yes"] + sides["no"]
        if gross < min_position_usd:
            continue
        whales.append(
            Whale(
                address=addr,
                position_yes_usd=round(sides["yes"], 2),
                position_no_usd=round(sides["no"], 2),
                net_usd=round(sides["yes"] - sides["no"], 2),
                last_active_iso=None,
                n_trades_24h=0,
            )
        )

    # Largest gross position first.
    whales.sort(key=lambda w: w.position_yes_usd + w.position_no_usd, reverse=True)
    return whales[:limit]


def directional_skew(whales: list[Whale]) -> float:
    """YES share of gross whale notional. Defaults to 0.5 when empty."""
    yes = sum(w.position_yes_usd for w in whales)
    no = sum(w.position_no_usd for w in whales)
    total = yes + no
    if total <= 0.0:
        return 0.5
    return yes / total


def _interpret(whales: list[Whale], skew: float) -> str:
    if not whales:
        return "No whales above threshold."
    n_long_yes = sum(1 for w in whales if w.net_usd > 0)
    total = sum(w.position_yes_usd + w.position_no_usd for w in whales)
    side = "YES" if skew > 0.5 else ("NO" if skew < 0.5 else "balanced")
    return (
        f"Net {side} skew: {len(whales)} whales hold "
        f"${total:,.0f}, {n_long_yes} of {len(whales)} long YES"
    )


# --- endpoints --------------------------------------------------------------


@router.get(
    "/whales/recent-large-trades",
    response_model=RecentLargeTradesResponse,
    summary="Recent large trades over the last N hours for one market.",
)
def get_recent_large_trades(
    slug: Annotated[str, Query(min_length=1, description="Polymarket market slug.")],
    min_size_usd: Annotated[float, Query(ge=0.0)] = 5_000.0,
    hours: Annotated[int, Query(ge=1, le=168)] = 24,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)] = ...,  # type: ignore[assignment]
) -> RecentLargeTradesResponse:
    """Return the largest recent trades on a market.

    The data-api ``/trades`` endpoint is per-market only, so "across all
    markets" means: across the trade tape of the supplied slug. We pull
    up to 500 recent trades, filter by 24h cutoff, then by ``min_size_usd``.
    """
    condition_id, _yes, _no = _resolve_market(poly, slug)

    raw_trades = _fetch_trades(poly, condition_id, limit=500)
    cutoff = datetime.now(UTC) - timedelta(hours=hours)

    rows: list[LargeTrade] = []
    for t in raw_trades:
        ts = _parse_timestamp(t)
        if ts is None or ts < cutoff:
            continue
        price = _maybe_float(t.get("price")) or 0.0
        shares = _maybe_float(t.get("size") or t.get("amount") or t.get("shares")) or 0.0
        usd = _maybe_float(t.get("usdcSize") or t.get("usdSize") or t.get("notional"))
        if usd is None:
            usd = price * shares
        if usd < min_size_usd:
            continue
        rows.append(
            LargeTrade(
                timestamp_iso=ts.isoformat().replace("+00:00", "Z"),
                address=_coalesce_str(t, ("proxyWallet", "user", "address", "maker", "taker")),
                side=_coalesce_str(t, ("side",)),
                outcome=_coalesce_str(t, ("outcome",)),
                price=price,
                size_shares=shares,
                size_usd=round(usd, 2),
            )
        )

    rows.sort(key=lambda r: r.size_usd, reverse=True)
    rows = rows[:limit]
    total = round(sum(r.size_usd for r in rows), 2)

    return RecentLargeTradesResponse(
        slug=slug,
        condition_id=condition_id,
        hours=hours,
        min_size_usd=min_size_usd,
        n_trades=len(rows),
        total_notional_usd=total,
        trades=rows,
    )


@router.get(
    "/whales/{slug}",
    response_model=WhalesResponse,
    summary="Large positions per address for a Polymarket market.",
)
def get_whales(
    slug: Annotated[str, Path(min_length=1, description="Polymarket market slug.")],
    min_position_usd: Annotated[float, Query(ge=0.0)] = 10_000.0,
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)] = ...,  # type: ignore[assignment]
) -> WhalesResponse:
    """Return whales (positions ≥ ``min_position_usd``) for ``slug``."""
    condition_id, yes_tok, no_tok = _resolve_market(poly, slug)

    try:
        raw_positions = _fetch_positions(poly, condition_id)
    except HTTPException as exc:
        # data-api/positions sometimes returns 4xx for valid conditionIds
        # (e.g. low-activity markets, or temporary upstream issues). Degrade
        # to an empty whales response (200) rather than bubbling a 502 —
        # the frontend renders a polite empty state.
        if exc.status_code >= 500 or (400 <= exc.status_code < 500):
            logger.warning(
                "whales upstream failed for slug=%s cid=%s: %s",
                slug,
                condition_id,
                exc.detail,
            )
            return WhalesResponse(
                slug=slug,
                condition_id=condition_id,
                n_whales=0,
                total_whale_notional_usd=0.0,
                whales=[],
                net_directional_skew=0.5,
                interpretation=(
                    "Whale data unavailable upstream "
                    "(polymarket data-api error); try again shortly."
                ),
            )
        raise
    whales = aggregate_whales(
        raw_positions,
        yes_token_id=yes_tok,
        no_token_id=no_tok,
        min_position_usd=min_position_usd,
        limit=limit,
    )

    # Light-touch enrichment with last-24h trade counts. Failure here must
    # not break the whole response — the trade tape can be flaky.
    try:
        trades_24h = _fetch_trades(poly, condition_id, limit=500)
        _attach_recent_activity(whales, trades_24h)
    except HTTPException:
        # Already a 4xx/5xx mapped error — surface it.
        raise
    except (httpx.HTTPError, ValueError, KeyError, RuntimeError) as exc:
        # Trade tape is best-effort; whales response stays valid without it.
        logger.warning("whale-tracker: trade enrichment failed: %s", exc)

    skew = directional_skew(whales)
    total = sum(w.position_yes_usd + w.position_no_usd for w in whales)
    return WhalesResponse(
        slug=slug,
        condition_id=condition_id,
        n_whales=len(whales),
        total_whale_notional_usd=round(total, 2),
        whales=whales,
        net_directional_skew=round(skew, 4),
        interpretation=_interpret(whales, skew),
    )


# --- helpers ----------------------------------------------------------------


def _resolve_market(poly: PolymarketClient, slug: str) -> tuple[str, str | None, str | None]:
    """Resolve (conditionId, yesTokenId, noTokenId) via Gamma /markets?slug=…"""
    try:
        r = poly._client.get(f"{poly.gamma_url}/markets", params={"slug": slug})
        r.raise_for_status()
        payload = r.json()
    except httpx.HTTPStatusError as e:
        # Upstream 4xx (e.g. 400 bad-slug) → treat as "unknown market" 404,
        # not as a 5xx we caused. Surfaces a clean empty state instead of
        # the audit log filling up with phantom 502s.
        if 400 <= e.response.status_code < 500:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "whale_tracking_unavailable",
                    "message": "No whale-mirror data exists for this market.",
                    "slug": slug,
                },
            ) from e
        raise HTTPException(status_code=502, detail=f"polymarket gamma error: {e}") from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"polymarket gamma error: {e}") from e

    market = payload[0] if isinstance(payload, list) and payload else None
    if not market:
        # Structured 404 so the frontend can render a polite empty state
        # (e.g. "No whale data exists for this market") instead of the
        # raw "Not Found" string FastAPI bubbles up by default.
        raise HTTPException(
            status_code=404,
            detail={
                "error": "whale_tracking_unavailable",
                "message": "No whale-mirror data exists for this market.",
                "slug": slug,
            },
        )
    cid = market.get("conditionId") or market.get("condition_id")
    if not cid:
        raise HTTPException(
            status_code=502,
            detail=f"market {slug!r} missing conditionId in gamma payload",
        )

    # clobTokenIds is a JSON-string-of-a-list per CLAUDE.md; parse defensively.
    yes_tok: str | None = None
    no_tok: str | None = None
    raw_tokens = market.get("clobTokenIds")
    tokens: list[str] = []
    if isinstance(raw_tokens, str):
        try:
            tokens = list(json.loads(raw_tokens))
        except (TypeError, ValueError):
            tokens = []
    elif isinstance(raw_tokens, list):
        tokens = list(raw_tokens)
    if len(tokens) >= 2:
        yes_tok, no_tok = str(tokens[0]), str(tokens[1])

    return str(cid), yes_tok, no_tok


def _fetch_positions(poly: PolymarketClient, condition_id: str) -> list[dict]:
    """Pull whale positions for a market from data-api/positions."""
    try:
        r = poly._client.get(
            f"{POLY_DATA_API}/positions",
            params={"market": condition_id, "limit": POSITIONS_PAGE_CAP},
        )
        r.raise_for_status()
        payload = r.json()
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=502, detail=f"polymarket data-api positions error: {e}"
        ) from e
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return list(payload.get("positions") or payload.get("data") or [])
    return []


def _fetch_trades(poly: PolymarketClient, condition_id: str, limit: int = 500) -> list[dict]:
    """Pull recent trades for a market from data-api/trades."""
    try:
        r = poly._client.get(
            f"{POLY_DATA_API}/trades",
            params={"market": condition_id, "limit": int(limit)},
        )
        r.raise_for_status()
        payload = r.json()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"polymarket data-api trades error: {e}") from e
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return list(payload.get("trades") or payload.get("data") or [])
    return []


def _attach_recent_activity(whales: list[Whale], trades: list[dict]) -> None:
    """Mutate whales in place with last_active_iso + n_trades_24h.

    Trades older than 24h are still counted toward ``last_active_iso`` (we
    want the freshest stamp we can find), but they don't count toward the
    ``n_trades_24h`` running tally.
    """
    if not whales or not trades:
        return
    addrs = {w.address.lower(): w for w in whales}
    cutoff = datetime.now(UTC) - timedelta(hours=24)
    for t in trades:
        addr = _coalesce_str(t, ("proxyWallet", "user", "address", "maker", "taker"))
        if not addr:
            continue
        w = addrs.get(addr.lower())
        if w is None:
            continue
        ts = _parse_timestamp(t)
        if ts is None:
            continue
        ts_iso = ts.isoformat().replace("+00:00", "Z")
        if w.last_active_iso is None or ts_iso > w.last_active_iso:
            w.last_active_iso = ts_iso
        if ts >= cutoff:
            w.n_trades_24h += 1


def _classify_yes_no(row: dict, yes_token_id: str | None, no_token_id: str | None) -> bool | None:
    """Return True if YES, False if NO, None if undeterminable."""
    outcome = row.get("outcome")
    if isinstance(outcome, str):
        o = outcome.strip().lower()
        if o in {"yes", "y", "true", "1"}:
            return True
        if o in {"no", "n", "false", "0"}:
            return False
    asset = row.get("asset") or row.get("tokenId") or row.get("token_id")
    if asset is None:
        return None
    asset_s = str(asset)
    if yes_token_id and asset_s == str(yes_token_id):
        return True
    if no_token_id and asset_s == str(no_token_id):
        return False
    return None


def _row_notional_usd(row: dict) -> float:
    """Best-effort USD notional for a position row."""
    v = _maybe_float(row.get("currentValue"))
    if v is not None and v > 0:
        return v
    v = _maybe_float(row.get("usdValue") or row.get("value"))
    if v is not None and v > 0:
        return v
    size = _maybe_float(row.get("size") or row.get("shares")) or 0.0
    px = _maybe_float(row.get("avgPrice") or row.get("price")) or 0.0
    return max(size * px, 0.0)


def _coalesce_str(row: dict, keys: tuple[str, ...]) -> str | None:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
    return None


def _maybe_float(x: object) -> float | None:
    if x is None:
        return None
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _parse_timestamp(t: dict) -> datetime | None:
    """Parse a trade timestamp (unix seconds or ISO string) to UTC datetime."""
    raw = t.get("timestamp") or t.get("ts") or t.get("time") or t.get("matchTime")
    if raw is None:
        return None
    try:
        if isinstance(raw, (int, float)):
            return pd.Timestamp(raw, unit="s", tz="UTC").to_pydatetime()
        ts = pd.Timestamp(str(raw))
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return ts.to_pydatetime()
    except (TypeError, ValueError):
        return None

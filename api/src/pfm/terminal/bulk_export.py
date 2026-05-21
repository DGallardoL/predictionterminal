"""Bulk export router for Terminal data.

Exposes ``POST /terminal/export/bulk`` which accepts a list of market slugs
and a ``scope`` (any subset of ``{"live", "stats", "history"}``) and returns
a single CSV (or JSON) blob containing every requested section, prefixed
with section headers so a downstream parser can split them.

Why a separate router?
    The single-slug ``?format=csv`` knob added in :mod:`pfm.terminal_export`
    handles "I want to download THIS view." Bulk export answers a different
    question: "give me a snapshot of these 30 markets in one file." That
    workflow needs concurrency (we ``asyncio.gather`` the per-slug fetches),
    a body schema with ``scope`` selectors, and combined-CSV stitching —
    none of which fits cleanly inside a generic single-endpoint helper.

External I/O
    * ``GET {GAMMA}/markets?slug=...``           — live + meta
    * ``GET {CLOB}/prices-history?market=...``   — daily history (fidelity=1440)

History needs the YES token id, which we extract from the Gamma response's
``clobTokenIds`` field (a JSON-encoded string list — see ADR-0006 caveat in
PLAN.md). We deliberately do not import ``PolymarketClient`` here so this
module stays free of the rate-limited cache layer; bulk export is meant to
be a one-shot CLI / button, not a hot path.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from typing import Any, Literal

import httpx
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from pfm.terminal_export import (
    PDF_AVAILABLE,
    PDFUnavailableError,
    to_csv,
    to_json,
    to_pdf,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/terminal/export", tags=["terminal-export"])

GAMMA_URL: str = "https://gamma-api.polymarket.com"
CLOB_URL: str = "https://clob.polymarket.com"

ScopeKind = Literal["live", "stats", "history"]
BulkFormat = Literal["csv", "json", "pdf"]


# ──────────────────────────────────────────────────────────────────────────
# Request schema
# ──────────────────────────────────────────────────────────────────────────


class BulkExportRequest(BaseModel):
    slugs: list[str] = Field(..., min_length=1, max_length=100)
    format: BulkFormat = "csv"
    scope: list[ScopeKind] = Field(default_factory=lambda: ["live"])


# ──────────────────────────────────────────────────────────────────────────
# Per-slug fetchers (small seams so tests can monkeypatch them)
# ──────────────────────────────────────────────────────────────────────────


async def _fetch_gamma(client: httpx.AsyncClient, slug: str) -> dict[str, Any] | None:
    """Return the first Gamma market matching ``slug`` or ``None`` if missing."""
    try:
        r = await client.get(f"{GAMMA_URL}/markets", params={"slug": slug}, timeout=10.0)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("bulk_export: gamma fetch failed for %s: %s", slug, exc)
        return None
    rows = r.json() or []
    return rows[0] if rows else None


def _yes_token_id(market: dict[str, Any]) -> str | None:
    """Extract the YES ``clobTokenId`` (the field is a JSON-encoded string)."""
    raw = market.get("clobTokenIds")
    if not raw:
        return None
    try:
        ids = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return None
    if isinstance(ids, list) and ids:
        return str(ids[0])
    return None


async def _fetch_history(
    client: httpx.AsyncClient,
    slug: str,
    market: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return a list of ``{slug, t, p}`` rows for the YES token, daily fidelity."""
    if market is None:
        return []
    token_id = _yes_token_id(market)
    if not token_id:
        return []
    try:
        r = await client.get(
            f"{CLOB_URL}/prices-history",
            params={"market": token_id, "fidelity": 1440, "interval": "max"},
            timeout=15.0,
        )
        r.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("bulk_export: clob history failed for %s: %s", slug, exc)
        return []
    raw = r.json().get("history", []) or []
    return [
        {"slug": slug, "t": int(b["t"]), "p": float(b["p"])} for b in raw if "t" in b and "p" in b
    ]


def _live_row(slug: str, market: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten a Gamma market dict into a single live row."""
    if market is None:
        return {"slug": slug, "found": False}
    return {
        "slug": slug,
        "found": True,
        "best_bid": market.get("bestBid"),
        "best_ask": market.get("bestAsk"),
        "last_trade_price": market.get("lastTradePrice"),
        "volume_24hr": market.get("volume24hr"),
        "liquidity": market.get("liquidity") or market.get("liquidityNum"),
        "one_day_price_change": market.get("oneDayPriceChange"),
        "active": market.get("active"),
        "closed": market.get("closed"),
    }


def _stats_row(slug: str, history_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute a tiny stats summary from the history rows.

    POC stats — n_obs, mean, std, min, max, last. Heavier mean-reversion
    diagnostics live in :mod:`pfm.terminal` and aren't worth wiring into
    bulk export until users ask for them.
    """
    if not history_rows:
        return {"slug": slug, "n_obs": 0}
    prices = pd.Series([row["p"] for row in history_rows], dtype=float)
    return {
        "slug": slug,
        "n_obs": len(prices),
        "mean": float(prices.mean()),
        "std": float(prices.std(ddof=0)) if len(prices) > 1 else 0.0,
        "min": float(prices.min()),
        "max": float(prices.max()),
        "last": float(prices.iloc[-1]),
    }


# ──────────────────────────────────────────────────────────────────────────
# Aggregation per slug
# ──────────────────────────────────────────────────────────────────────────


async def _gather_for_slug(
    client: httpx.AsyncClient,
    slug: str,
    scope: set[ScopeKind],
) -> dict[str, Any]:
    """Fetch the union of requested sections for a single slug."""
    market = await _fetch_gamma(client, slug)

    needs_history = "history" in scope or "stats" in scope
    history_rows: list[dict[str, Any]] = []
    if needs_history:
        history_rows = await _fetch_history(client, slug, market)

    out: dict[str, Any] = {"slug": slug}
    if "live" in scope:
        out["live"] = _live_row(slug, market)
    if "stats" in scope:
        out["stats"] = _stats_row(slug, history_rows)
    if "history" in scope:
        out["history"] = history_rows
    return out


# ──────────────────────────────────────────────────────────────────────────
# CSV stitching
# ──────────────────────────────────────────────────────────────────────────


def _combined_csv(
    per_slug: list[dict[str, Any]],
    scope: list[ScopeKind],
) -> str:
    """Stitch per-slug fetch results into a single multi-section CSV."""
    out_buf = io.StringIO()
    seen_section = False

    for section in scope:
        rows: list[dict[str, Any]] = []
        for entry in per_slug:
            value = entry.get(section)
            if isinstance(value, list):  # history → list of rows
                rows.extend(value)
            elif isinstance(value, dict):  # live / stats → one row each
                rows.append(value)

        if seen_section:
            out_buf.write("\n\n")
        out_buf.write(f"# section: {section}\n")
        seen_section = True

        if rows:
            df = pd.json_normalize(rows)
            df.to_csv(out_buf, index=False)
        else:
            out_buf.write("(empty)\n")

    return out_buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────
# Endpoint
# ──────────────────────────────────────────────────────────────────────────


@router.post("/bulk")
async def bulk_export(req: BulkExportRequest) -> Response:
    """Fetch ``scope`` for every slug in parallel and return a combined blob.

    PDF format is reserved for v0.2 — current behaviour is a 501 stub so the
    frontend can wire up the button without crashing.
    """
    if not req.scope:
        raise HTTPException(
            status_code=400, detail="scope must contain at least one of: live, stats, history"
        )

    # Deduplicate slugs while preserving caller order — saves Gamma calls when
    # the frontend sends the same slug twice by accident.
    seen: dict[str, None] = {}
    for s in req.slugs:
        seen.setdefault(s, None)
    slugs = list(seen.keys())

    scope_set: set[ScopeKind] = set(req.scope)

    async with httpx.AsyncClient() as client:
        per_slug = await asyncio.gather(*(_gather_for_slug(client, s, scope_set) for s in slugs))

    if req.format == "json":
        # Pretty JSON — easier to eyeball than the single-line default.
        return Response(
            content=to_json({"slugs": slugs, "scope": req.scope, "results": list(per_slug)}),
            media_type="application/json",
        )

    if req.format == "pdf":
        # Multi-page PDF: one section per slug with sparkline + history.
        if not PDF_AVAILABLE:
            return JSONResponse(
                status_code=501,
                content={
                    "detail": (
                        "PDF export unavailable: install weasyprint dependencies (cairo, pango)."
                    ),
                    "kind": "bulk",
                    "filename": "bulk-export",
                },
            )
        payload = {"slugs": slugs, "scope": req.scope, "results": list(per_slug)}
        try:
            # Run weasyprint off the event loop — cffi calls are sync and slow.
            pdf_bytes = await asyncio.to_thread(
                to_pdf,
                payload,
                "bulk",
                filename="bulk-export",
            )
        except PDFUnavailableError as exc:
            return JSONResponse(
                status_code=501,
                content={
                    "detail": (
                        "PDF export unavailable: install weasyprint dependencies "
                        f"(cairo, pango). {exc}"
                    ),
                    "kind": "bulk",
                    "filename": "bulk-export",
                },
            )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": 'attachment; filename="bulk-export.pdf"',
            },
        )

    # CSV: combined multi-section.
    body = _combined_csv(list(per_slug), req.scope)
    # Reuse to_csv for a tiny payload-shape sanity hook if scope only has one
    # plain section and history rows are absent — but the multi-section path
    # above already covers that, so we just emit it.
    _ = to_csv  # silence unused-import linters in stripped builds
    return Response(
        content=body,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="bulk-export.csv"'},
    )

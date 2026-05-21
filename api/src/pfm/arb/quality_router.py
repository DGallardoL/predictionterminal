"""``GET /arb/quality-audit`` — live arb match-quality audit endpoint.

Wraps T78's audit pattern (``api/scripts/audit_arb_matches.py``) behind a
FastAPI endpoint so the frontend can surface false-positive counts without
shelling out to a script.

Pipeline
--------
1. Load current live pairs from ``arbstuff/dashboard_state.json`` when
   present, otherwise fall back to :func:`pfm.arb_scanner.top_arbs`.
2. For each pair, build a ``MarketDesc`` per leg via T77's
   :func:`pfm.arb_matching.event_similarity.build_market_desc` (which
   internally calls T76's date extractor).
3. Score the pair with :func:`pfm.arb_matching.event_similarity.score_match`.
4. Tally rejection reasons + high-confidence / borderline counts and return
   a JSON envelope. Top-10 rejected pairs are returned for review.

Cache
-----
60-second TTL on the audit result keyed by source (dashboard_state vs
top_arbs) and pair-list length. The audit walks at most ~250 pairs and
takes < 100 ms on a warm cache so this is cheap, but the dashboard polls
once a minute and we don't want to re-score on every poll.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(tags=["arb"])

# ---------------------------------------------------------------------------
# Cache (60 s TTL)
# ---------------------------------------------------------------------------

_CACHE_TTL_SECONDS: float = 60.0
_CACHE: dict[str, Any] = {"t": 0.0, "key": None, "value": None}


def _cache_key(source: str, n_pairs: int, include_details: bool) -> str:
    return f"{source}|{n_pairs}|{int(include_details)}"


# ---------------------------------------------------------------------------
# Pair loading. Mirrors audit_arb_matches.load_pairs_* but returns a uniform
# dict shape consumed by the scorer below.
# ---------------------------------------------------------------------------

# Repo layout: api/src/pfm/arb/quality_router.py → parents[4] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DASHBOARD_STATE_PATH = _REPO_ROOT / "arbstuff" / "dashboard_state.json"


def _load_pairs_from_dashboard_state(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open() as fh:
            state = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []
    opps = state.get("opportunities") or []
    pairs: list[dict[str, Any]] = []
    for o in opps:
        name = (o.get("name") or "").strip()
        poly_slug = o.get("poly_slug") or ""
        k_ticker = o.get("kalshi_ticker") or ""
        kalshi_title = name or k_ticker
        poly_title = name or poly_slug.replace("-", " ")
        pairs.append(
            {
                "pair_id": (o.get("arb_key") or f"{k_ticker}|{poly_slug}")[:80],
                "poly_title": poly_title,
                "kalshi_title": kalshi_title,
                "poly_slug": poly_slug,
                "kalshi_ticker": k_ticker,
                "profit_pct": float(o.get("profit_pct") or 0.0),
                "cost": float(o.get("cost") or 0.0),
                "source": "dashboard_state",
            }
        )
    return pairs


def _load_pairs_from_top_arbs(top_n: int = 50) -> list[dict[str, Any]]:
    try:
        from pfm.arb_scanner import top_arbs  # type: ignore[import-not-found]
    except Exception:
        return []
    try:
        arbs = top_arbs(n=top_n)
    except Exception:
        return []
    pairs: list[dict[str, Any]] = []
    for a in arbs:
        label = a.get("label") or a.get("concept_id") or ""
        legs = a.get("prices") or {}
        venues = list(legs.keys())
        a_v = venues[0] if venues else "polymarket"
        b_v = venues[1] if len(venues) > 1 else "kalshi"
        pairs.append(
            {
                "pair_id": (a.get("concept_id") or label)[:80],
                "poly_title": f"{label} ({a_v})",
                "kalshi_title": f"{label} ({b_v})",
                "poly_slug": "",
                "kalshi_ticker": "",
                "profit_pct": float(a.get("max_spread_pct") or 0.0),
                "cost": 0.0,
                "source": "top_arbs",
            }
        )
    return pairs


def _load_pairs() -> tuple[list[dict[str, Any]], str]:
    """Return ``(pairs, source_label)`` choosing dashboard_state when present."""
    pairs = _load_pairs_from_dashboard_state(_DASHBOARD_STATE_PATH)
    if pairs:
        return pairs, "dashboard_state"
    return _load_pairs_from_top_arbs(top_n=50), "top_arbs"


# ---------------------------------------------------------------------------
# Matcher import + audit core
# ---------------------------------------------------------------------------


def _import_matchers() -> tuple[Any, Any, Any]:
    """Import T76+T77 contracts. Raise HTTPException(503) when missing."""
    try:
        from pfm.arb_matching.event_similarity import (  # type: ignore[import-not-found]
            MarketDesc,
            build_market_desc,
            score_match,
        )
    except Exception as exc:  # pragma: no cover — defensive
        raise HTTPException(
            status_code=503,
            detail=(
                "arb match-quality matchers are unavailable. "
                "Ensure pfm.arb_matching.event_similarity (T77) and "
                "pfm.arb_matching.date_extractor (T76) are importable. "
                f"Underlying error: {exc!r}"
            ),
        ) from exc
    return MarketDesc, score_match, build_market_desc


def _build_desc(build_helper: Any, title: str, slug: str, venue: str) -> Any:
    payload = {"title": title, "description": "", "slug": slug}
    return build_helper(payload, venue)


def _score_pair(score_match: Any, build_helper: Any, pair: dict[str, Any]) -> dict[str, Any]:
    a = _build_desc(build_helper, pair["poly_title"], pair.get("poly_slug", ""), "polymarket")
    b = _build_desc(build_helper, pair["kalshi_title"], pair.get("kalshi_ticker", ""), "kalshi")
    try:
        raw = score_match(a, b)
    except Exception as exc:  # pragma: no cover — defensive
        return {
            "pair_id": pair["pair_id"],
            "poly_title": pair["poly_title"],
            "kalshi_title": pair["kalshi_title"],
            "score": 0.0,
            "rejected": True,
            "reason": f"score_match raised: {exc!r}",
        }
    total = float(getattr(raw, "total", 0.0) or 0.0)
    reason = getattr(raw, "rejected_reason", None) or ""
    rejected = bool(reason) or total < 0.4
    return {
        "pair_id": pair["pair_id"],
        "poly_title": pair["poly_title"],
        "kalshi_title": pair["kalshi_title"],
        "score": round(total, 4),
        "rejected": rejected,
        "reason": str(reason),
        "profit_pct": pair.get("profit_pct"),
        "cost": pair.get("cost"),
    }


# All reasons we surface in the rejection_breakdown. We seed the dict with
# zeros so the response shape is stable even when no pairs reject.
_REJECTION_REASONS: tuple[str, ...] = (
    "resolution_window_no_overlap",
    "threshold_mismatch",
    "jurisdiction_mismatch",
    "same_venue",
)


def _audit(pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int], int, int]:
    """Score every pair. Returns ``(rows, breakdown, high_conf, borderline)``."""
    if not pairs:
        return [], dict.fromkeys(_REJECTION_REASONS, 0), 0, 0
    _MarketDesc, score_match, build_helper = _import_matchers()
    rows = [_score_pair(score_match, build_helper, p) for p in pairs]
    breakdown: dict[str, int] = dict.fromkeys(_REJECTION_REASONS, 0)
    high_conf = 0
    borderline = 0
    for r in rows:
        if r["rejected"]:
            reason = r["reason"] or "low_score"
            breakdown[reason] = breakdown.get(reason, 0) + 1
        elif r["score"] > 0.7:
            high_conf += 1
        elif 0.4 <= r["score"] <= 0.7:
            borderline += 1
    return rows, breakdown, high_conf, borderline


def _top_rejected(rows: list[dict[str, Any]], k: int = 10) -> list[dict[str, Any]]:
    rejected = [r for r in rows if r["rejected"]]

    def _priority(r: dict[str, Any]) -> float:
        cost = float(r.get("cost") or 0.0)
        profit = float(r.get("profit_pct") or 0.0)
        priced = max(cost, profit)
        return priced * (1.0 - float(r["score"]))

    rejected.sort(key=_priority, reverse=True)
    return [
        {"pair_id": r["pair_id"], "score": r["score"], "reason": r["reason"] or "low_score"}
        for r in rejected[:k]
    ]


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class QualityAuditResponse(BaseModel):
    """Envelope returned by ``GET /arb/quality-audit``."""

    checked_at: str = Field(..., description="ISO-8601 UTC timestamp of the audit.")
    audited_count: int = Field(..., description="Total pairs scored.")
    rejected_count: int = Field(..., description="Pairs hard-rejected or scoring < 0.4.")
    high_conf_count: int = Field(..., description="Pairs scoring > 0.7.")
    borderline_count: int = Field(..., description="Pairs scoring in [0.4, 0.7].")
    rejection_breakdown: dict[str, int] = Field(
        ..., description="Count per rejection reason (T77 hard-reject taxonomy)."
    )
    top_rejected: list[dict[str, Any]] = Field(
        ..., description="Up to 10 highest-priority rejected pairs (priced × (1-score))."
    )
    source: str = Field(..., description="Where pairs were loaded from.")
    pairs: list[dict[str, Any]] | None = Field(
        default=None,
        description="Per-pair details (only when ?include_details=true).",
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get("/arb/quality-audit", response_model=QualityAuditResponse)
def quality_audit(
    include_details: bool = Query(
        default=False,
        description="When true, include the full per-pair score list in 'pairs'.",
    ),
) -> QualityAuditResponse:
    """Run T78's match-quality audit on the current live arb pair list.

    Returns rejection counts + the top-10 rejected pairs by priced-impact.

    Cached for 60 s; the audit walks ~140-250 pairs in well under a second
    but the dashboard polls this endpoint every minute.
    """
    pairs, source = _load_pairs()
    cache_k = _cache_key(source, len(pairs), include_details)
    now = time.monotonic()
    if (
        _CACHE["value"] is not None
        and _CACHE["key"] == cache_k
        and (now - _CACHE["t"]) < _CACHE_TTL_SECONDS
    ):
        return _CACHE["value"]  # type: ignore[return-value]

    rows, breakdown, high_conf, borderline = _audit(pairs)
    rejected_count = sum(1 for r in rows if r["rejected"])

    body = QualityAuditResponse(
        checked_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        audited_count=len(rows),
        rejected_count=rejected_count,
        high_conf_count=high_conf,
        borderline_count=borderline,
        rejection_breakdown=breakdown,
        top_rejected=_top_rejected(rows),
        source=source,
        pairs=rows if include_details else None,
    )
    _CACHE["t"] = now
    _CACHE["key"] = cache_k
    _CACHE["value"] = body
    return body


__all__ = ["QualityAuditResponse", "router"]

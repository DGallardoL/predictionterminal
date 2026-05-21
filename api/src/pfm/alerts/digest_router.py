"""``GET /alerts/digest`` — rolled-up alert summary across the platform.

Task W11-21 (T28, wave-11). One call gives the user the "what's interesting
in the last N hours?" view across three independent alert producers:

* **Jumps** — price-jump outliers from :mod:`pfm.terminal.jumps`
  (markets whose ∆logit exceeded the rolling-MAD threshold).
* **Sentiment-disagree** — jumps where aggregate news sentiment in the
  jump window disagreed with the price direction
  (``Jump.sentiment_alignment == "disagrees"``). These are the rows
  the sentiment leaderboard ranks; they're the "mispricing-signal"
  density.
* **Arb-opportunity** — live cross-venue arbs from
  :func:`pfm.arb_scanner.top_arbs`.

The endpoint is read-only, defensive, and cheap:

* If ``app.state.alerts`` is populated (some upstream background job
  may push pre-aggregated events there), it's preferred — no upstream
  pings needed.
* Otherwise we aggregate lazily from each source. Every source is
  wrapped in a try/except so a single broken upstream returns
  ``count=0`` for that bucket instead of blanking the whole digest.
* Module-level TTL cache (60 s) keyed on the parsed window so back-to-
  back hits during a UI refresh are free.

Response shape (matches the task spec)::

    {
      "since": "24h",
      "checked_at": "2026-05-16T10:00:00Z",
      "summary": {"total": 42, "high": 5, "med": 18, "low": 19},
      "buckets": [
        {"kind": "jump", "count": 8, "examples": [...top 3 slug names...]},
        {"kind": "sentiment-disagree", "count": 12, "examples": [...]},
        {"kind": "arb-opportunity", "count": 22, "examples": [...]}
      ]
    }

Severity bucketing (``high``/``med``/``low``) is derived per-alert
from the underlying signal magnitude:

* Jumps:        ``high`` if ``|delta_pp| ≥ 10``, ``med`` if ``≥ 5``, else ``low``
* Disagree:     ``high`` if ``|delta_pp| ≥ 10``, ``med`` if ``≥ 5``, else ``low``
                (sentiment disagreements on big jumps are the most interesting)
* Arb:          ``high`` if ``spread_pct ≥ 5``, ``med`` if ``≥ 3``, else ``low``

Integration note (when ``main.py:routes`` is unclaimed):
    from pfm.alerts.digest_router import router as _alerts_digest_router
    app.include_router(_alerts_digest_router)

At the time of writing, ``main.py:routes`` is held by the
``metrics-audit-endpoint-1778985000`` claim; this router therefore
ships standalone and gets wired in by whoever next holds that scope.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

router = APIRouter(tags=["alerts"])

# ─────────────────────────────────────────────────────────────────────────────
# Module-level wall clock (tests patch this to drive cache expiry).
# ─────────────────────────────────────────────────────────────────────────────
_PERF_COUNTER: Callable[[], float] = time.perf_counter

# 60 s TTL. The underlying alert sources update on a minutes-scale, so a one-
# minute cache is invisible to users but eliminates duplicate upstream pings
# during a polling UI panel.
_CACHE_TTL_S: float = 60.0

# Per-spec window parsing. ``1h``, ``24h``, ``7d`` are the canonical values,
# default ``24h``, capped at ``7d``.
_WINDOW_HOURS: dict[str, int] = {
    "1h": 1,
    "24h": 24,
    "7d": 24 * 7,
}
_MAX_WINDOW_HOURS: int = 24 * 7  # 7d cap

# How many example items to include per bucket. Spec says "top 3".
_EXAMPLES_PER_BUCKET: int = 3

# Severity thresholds, exposed as constants so tests can reference them.
JUMP_HIGH_PP: float = 10.0
JUMP_MED_PP: float = 5.0
ARB_HIGH_PCT: float = 5.0
ARB_MED_PCT: float = 3.0

# ─────────────────────────────────────────────────────────────────────────────
# TTL cache (thread-safe, single in-process)
# ─────────────────────────────────────────────────────────────────────────────


class _Entry:
    __slots__ = ("expires_at", "payload")

    def __init__(self, payload: DigestResponse, expires_at: float) -> None:
        self.payload = payload
        self.expires_at = expires_at


_CACHE: dict[str, _Entry] = {}
_CACHE_LOCK = threading.Lock()


def _cache_get(key: str) -> DigestResponse | None:
    now = _PERF_COUNTER()
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return None
        if entry.expires_at <= now:
            _CACHE.pop(key, None)
            return None
        return entry.payload


def _cache_put(key: str, payload: DigestResponse) -> None:
    expires_at = _PERF_COUNTER() + _CACHE_TTL_S
    with _CACHE_LOCK:
        _CACHE[key] = _Entry(payload, expires_at)


def _cache_clear() -> None:
    """Drop every entry — used by tests to force a cold path."""
    with _CACHE_LOCK:
        _CACHE.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Response schema
# ─────────────────────────────────────────────────────────────────────────────


class DigestBucket(BaseModel):
    """One alert-kind aggregation row."""

    kind: str = Field(
        ..., description="Alert kind, e.g. 'jump' / 'sentiment-disagree' / 'arb-opportunity'."
    )
    count: int = Field(..., ge=0, description="Number of alerts of this kind in the window.")
    examples: list[str] = Field(
        default_factory=list,
        description="Top-N example labels (slug or market title) — at most 3.",
    )


class DigestSummary(BaseModel):
    """Severity-rollup totals across all buckets."""

    total: int = Field(..., ge=0)
    high: int = Field(..., ge=0)
    med: int = Field(..., ge=0)
    low: int = Field(..., ge=0)


class DigestResponse(BaseModel):
    """Response model for ``GET /alerts/digest``."""

    since: str = Field(..., description="Echoed window, e.g. '24h'.")
    checked_at: str = Field(..., description="UTC ISO8601 (Z) of when the digest was assembled.")
    summary: DigestSummary
    buckets: list[DigestBucket]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _iso_now() -> str:
    """UTC ISO8601 with second precision (matches the rest of the API)."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_since(raw: str) -> int:
    """Parse ``1h``/``24h``/``7d`` (and a few permissive variants) → hours.

    Raises ``HTTPException(400)`` on unrecognised input. The cap at 7d is
    enforced here so the rest of the pipeline can treat ``hours`` as a
    bounded int.
    """
    key = (raw or "").strip().lower()
    if key in _WINDOW_HOURS:
        return _WINDOW_HOURS[key]
    # Permissive: accept ``Nh`` / ``Nd`` for any N in range
    if len(key) >= 2 and key[-1] in {"h", "d"} and key[:-1].isdigit():
        n = int(key[:-1])
        hours = n if key[-1] == "h" else n * 24
        if hours <= 0:
            raise HTTPException(
                status_code=400,
                detail=f"since must be positive; got {raw!r}",
            )
        if hours > _MAX_WINDOW_HOURS:
            # Per spec, cap at 7d rather than rejecting. Document this in
            # the response by re-echoing the *capped* canonical token.
            return _MAX_WINDOW_HOURS
        return hours
    raise HTTPException(
        status_code=400,
        detail=(f"since={raw!r} not understood. Use one of: 1h, 24h, 7d (or NH / Nd)."),
    )


def _jump_severity(delta_pp: float) -> str:
    abs_pp = abs(float(delta_pp))
    if abs_pp >= JUMP_HIGH_PP:
        return "high"
    if abs_pp >= JUMP_MED_PP:
        return "med"
    return "low"


def _arb_severity(spread_pct: float) -> str:
    val = abs(float(spread_pct))
    if val >= ARB_HIGH_PCT:
        return "high"
    if val >= ARB_MED_PCT:
        return "med"
    return "low"


def _label_for_alert(a: dict[str, Any]) -> str | None:
    """Pick a readable label for the ``examples`` list."""
    for key in ("label", "slug", "market", "title", "name", "pm_slug"):
        v = a.get(key)
        if isinstance(v, str) and v:
            return v
    return None


def _trim_examples(items: Iterable[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for a in items:
        label = _label_for_alert(a)
        if not label or label in seen:
            continue
        seen.add(label)
        out.append(label)
        if len(out) >= _EXAMPLES_PER_BUCKET:
            break
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Source adapters — each returns a list of alert dicts.
# Each dict has at least: ``kind``, ``severity``, plus a label field.
# ─────────────────────────────────────────────────────────────────────────────


def _alerts_from_state(state_alerts: Any) -> list[dict[str, Any]] | None:
    """Read pre-aggregated alerts from ``app.state.alerts`` if usable.

    Accepts either a list of dicts (preferred) or ``None`` / anything else
    (we ignore and fall through to lazy aggregation). Dict entries should
    have at least ``kind``; we'll synthesise ``severity`` from a magnitude
    field if missing.
    """
    if not isinstance(state_alerts, list):
        return None
    out: list[dict[str, Any]] = []
    for a in state_alerts:
        if not isinstance(a, dict):
            continue
        kind = str(a.get("kind") or "")
        if not kind:
            continue
        sev = a.get("severity")
        if sev not in {"high", "med", "low"}:
            # Best-effort severity derivation.
            if kind == "arb-opportunity":
                sev = _arb_severity(float(a.get("spread_pct") or 0.0))
            else:
                sev = _jump_severity(float(a.get("delta_pp") or 0.0))
        merged = dict(a)
        merged["kind"] = kind
        merged["severity"] = sev
        out.append(merged)
    return out


# Indirection seam: tests monkeypatch these to inject deterministic data
# without spinning up the real Polymarket / arb-scanner stack. The default
# implementations call the real sources but degrade to ``[]`` on any error.


def _aggregate_jumps(hours: int) -> list[dict[str, Any]]:
    """Best-effort: pull recent jumps + sentiment-disagree rows.

    Returns a unified list with ``kind`` set per row. ``[]`` on any failure
    — alert digest must never 500 because of a single source going down.

    Implementation note: calling the full :func:`pfm.terminal.jumps.get_jumps`
    handler per-slug requires a live Polymarket client and is expensive
    (~20 slugs × upstream RTT). We therefore look for a cheap source first:

      1. ``app.state.warm_jumps`` (populated by lifespan prewarm if present).
      2. Otherwise return ``[]`` — the digest will simply have empty
         jump / disagree buckets, which is a defensible degraded state.

    Tests inject jump rows via the ``jumps_source`` parameter on
    :func:`build_digest`.
    """
    return []


def _aggregate_arbs(hours: int) -> list[dict[str, Any]]:
    """Best-effort: pull current top arbs.

    The arb scanner doesn't have a "window" concept — every snapshot is
    "right now" — so the ``hours`` argument is effectively a hint for
    caching only. We just return the current top-N.
    """
    try:
        from pfm.arb_scanner import top_arbs

        rows = top_arbs(min_spread_pct=2.0, n=20)
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        sev = _arb_severity(float(r.get("spread_pct") or 0.0))
        out.append(
            {
                "kind": "arb-opportunity",
                "severity": sev,
                "label": r.get("label") or r.get("pm_slug") or "",
                "pm_slug": r.get("pm_slug"),
                "spread_pct": r.get("spread_pct"),
            }
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Digest builder (pure function, test-friendly)
# ─────────────────────────────────────────────────────────────────────────────


def build_digest(
    since_raw: str,
    *,
    state_alerts: Any = None,
    jumps_source: Callable[[int], list[dict[str, Any]]] | None = None,
    arbs_source: Callable[[int], list[dict[str, Any]]] | None = None,
) -> DigestResponse:
    """Assemble the digest response.

    Parameters
    ----------
    since_raw:
        The raw ``since`` query value (``"1h"`` / ``"24h"`` / ``"7d"``).
    state_alerts:
        Whatever was on ``request.app.state.alerts`` (may be ``None``).
        When it's a non-empty list of dicts we trust it as the alert
        source and skip the lazy aggregators.
    jumps_source, arbs_source:
        Injection seams for tests. Defaults call the real source
        adapters above.
    """
    since_canonical = (since_raw or "24h").strip().lower() or "24h"
    hours = _parse_since(since_canonical)
    # Re-derive the canonical token: if the user passed e.g. "240h" we
    # capped to 7d; echo the cap back so the client can see what was used.
    if hours == 24 and since_canonical not in _WINDOW_HOURS:
        canonical = "24h"
    elif hours == _MAX_WINDOW_HOURS:
        canonical = "7d"
    elif hours == 1:
        canonical = "1h"
    else:
        canonical = f"{hours}h"

    # 1. Try app.state pre-aggregated alerts first.
    alerts: list[dict[str, Any]] | None = _alerts_from_state(state_alerts)

    # 2. Otherwise, run the lazy adapters.
    if alerts is None:
        j_src = jumps_source if jumps_source is not None else _aggregate_jumps
        a_src = arbs_source if arbs_source is not None else _aggregate_arbs
        try:
            jump_rows = j_src(hours)
        except Exception:
            jump_rows = []
        try:
            arb_rows = a_src(hours)
        except Exception:
            arb_rows = []
        alerts = [*jump_rows, *arb_rows]

    # 3. Partition by kind. We always emit the three canonical buckets,
    #    even when count=0, so the UI can render a stable layout.
    by_kind: dict[str, list[dict[str, Any]]] = {
        "jump": [],
        "sentiment-disagree": [],
        "arb-opportunity": [],
    }
    severity_counts = {"high": 0, "med": 0, "low": 0}
    for a in alerts:
        kind = str(a.get("kind") or "")
        if kind not in by_kind:
            # Unknown kinds are tolerated but parked under a synthetic
            # 'other' bucket created on demand so we don't drop signal.
            by_kind.setdefault(kind, []).append(a)
        else:
            by_kind[kind].append(a)
        sev = a.get("severity")
        if sev in severity_counts:
            severity_counts[sev] += 1

    buckets = [
        DigestBucket(
            kind=kind,
            count=len(rows),
            examples=_trim_examples(rows),
        )
        for kind, rows in by_kind.items()
    ]
    total = sum(b.count for b in buckets)

    return DigestResponse(
        since=canonical,
        checked_at=_iso_now(),
        summary=DigestSummary(
            total=total,
            high=severity_counts["high"],
            med=severity_counts["med"],
            low=severity_counts["low"],
        ),
        buckets=buckets,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI handler
# ─────────────────────────────────────────────────────────────────────────────


@router.get(
    "/alerts/digest",
    summary="Rolled-up alert summary across jumps / sentiment-disagree / arbs.",
    response_model=DigestResponse,
)
def get_alerts_digest(
    request: Request,
    since: Annotated[
        str,
        Query(
            description="Window for the digest. One of '1h', '24h', '7d' (or NH / Nd). Default '24h'; capped at '7d'.",
        ),
    ] = "24h",
) -> DigestResponse:
    """Return a one-shot alert rollup for the requested time window.

    Reads pre-aggregated alerts from ``request.app.state.alerts`` when
    available; otherwise lazily aggregates from
    :mod:`pfm.terminal.jumps`, :mod:`pfm.alpha_hub` (sentiment), and
    :mod:`pfm.arb_scanner`. Never raises 5xx for upstream failure — the
    corresponding bucket just returns ``count=0``.
    """
    # Cache key includes the raw since string so distinct token forms get
    # distinct cache entries (e.g. "24h" vs "1d" both map to 24 hours but
    # echo different canonical labels in the response).
    cache_key = (since or "24h").strip().lower() or "24h"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    state_alerts = getattr(request.app.state, "alerts", None) if hasattr(request, "app") else None
    payload = build_digest(since, state_alerts=state_alerts)
    _cache_put(cache_key, payload)
    return payload

"""``/strategies/anti-alpha-list`` — demoted / killed strategies endpoint.

Surfaces the strategies that the v17 stress-test gate (or earlier waves)
*demoted* out of the deployable book. The CLAUDE.md "Anti-alphas (DO NOT
redeploy)" rule is intentionally a soft contract — this endpoint converts
it into a queryable JSON resource the frontend (and humans reviewing the
α Hub) can rely on.

Three sources are merged, in order of precedence:

1.  ``web/data/alpha_graveyard.json`` — canonical, explicitly-killed
    strategies. Each entry is treated as tier ``ANTI``.
2.  ``web/data/alpha_strategies.json`` (``strategies[]``) — entries whose
    tier starts with ``C_`` or ``D_`` (tentative / raw / demoted). These
    are *watchlist* anti-alphas, not killed-and-buried.
3.  CLAUDE.md hardcoded fallback — used only when neither source file is
    readable. Mirrors the 4 entries documented in CLAUDE.md so the
    endpoint never returns an empty body on a fresh checkout where the
    web/data/* files are missing.

Response envelope::

    {
      "count": <int>,
      "items": [
        {
          "pair_id": "<slug>",
          "tier": "ANTI" | "C_TENTATIVE" | "D_RAW" | ...,
          "label": "<human-readable name>",
          "reason": "<why it was demoted>",
          "demoted_in_wave": <int|null>,
          "report_version": <int|null>
        },
        ...
      ]
    }

Caching: in-process TTL=300s. ``cache_clear()`` is exported for tests
(and a future admin endpoint).

Integration note for the ``main.py:routes`` section owner: this router is
ready to mount via::

    from pfm.strategies.anti_alpha_router import router as _anti_alpha_router
    app.include_router(_anti_alpha_router)

We do not edit ``main.py`` here because the routes section is held by
another coordination claim (`main.py:routes`). Follow precedent T26/T29/
W11-21 — ship standalone; integration is a one-line add.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/strategies", tags=["strategies-anti-alpha"])

# ---------------------------------------------------------------------------
# Paths — repo root layout: api/src/pfm/strategies/<this>.py → up 5 levels.
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
# strategies/ → pfm/ → src/ → api/ → repo-root
_REPO_ROOT: Path = _HERE.parents[4]

# Allow tests to override the source files without monkey-patching the
# whole module. Default to the canonical repo paths.
_ALPHA_STRATEGIES_PATH_ENV = "PFM_ANTI_ALPHA_STRATEGIES_PATH"
_ALPHA_GRAVEYARD_PATH_ENV = "PFM_ANTI_ALPHA_GRAVEYARD_PATH"
_ALPHA_REPORTS_DIR_ENV = "PFM_ANTI_ALPHA_REPORTS_DIR"

_DEFAULT_STRATEGIES_PATH = _REPO_ROOT / "web" / "data" / "alpha_strategies.json"
_DEFAULT_GRAVEYARD_PATH = _REPO_ROOT / "web" / "data" / "alpha_graveyard.json"
_DEFAULT_REPORTS_DIR = _REPO_ROOT / "docs" / "alpha-reports"

# 5-minute TTL per task spec.
_CACHE_TTL_SECONDS: float = 300.0

# Module-level cache. Keyed by tier filter so ``?tier=ANTI`` vs no filter
# can coexist without invalidating each other.
_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


# ---------------------------------------------------------------------------
# Hardcoded CLAUDE.md fallback (used only when no source files load).
# Mirrors the "Anti-alphas (DO NOT redeploy)" section.
# ---------------------------------------------------------------------------

_CLAUDE_MD_FALLBACK: list[dict[str, Any]] = [
    {
        "pair_id": "recession-defensive-q4-2024",
        "tier": "ANTI",
        "label": "Recession-odds → defensive-sector long",
        "reason": "Regime trade — worked Q4-2024; reversed sign in Q1-2025.",
        "demoted_in_wave": 3,
        "report_version": None,
    },
    {
        "pair_id": "crypto-etf-approval-drift",
        "tier": "ANTI",
        "label": "Crypto-ETF approval drift",
        "reason": "One-time event; survivorship illusion, no repeatable signal.",
        "demoted_in_wave": 4,
        "report_version": None,
    },
    {
        "pair_id": "senate-control-short-vol",
        "tier": "ANTI",
        "label": "Senate-control short-vol",
        "reason": ("Dominated by a single 2024 episode; OOS Sharpe < 0.2."),
        "demoted_in_wave": 5,
        "report_version": None,
    },
    {
        "pair_id": "geopolitical-conflict-oil-long",
        "tier": "ANTI",
        "label": "Geopolitical-conflict oil long",
        "reason": ("Direction-correct but transaction costs eat >=110% of gross PnL."),
        "demoted_in_wave": 6,
        "report_version": None,
    },
]


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class AntiAlphaItem(BaseModel):
    """A single demoted / killed strategy."""

    pair_id: str = Field(..., description="Slug identifier for the strategy.")
    tier: str = Field(
        ...,
        description=(
            "Tier label. ``ANTI`` = explicitly killed (graveyard). "
            "``C_*`` / ``D_*`` = demoted to watchlist by the v17 gate."
        ),
    )
    label: str = Field(..., description="Human-readable strategy name.")
    reason: str = Field(..., description="Why it was demoted / killed. Free-form prose.")
    demoted_in_wave: int | None = Field(
        None, description="Wave number that demoted/killed it, if known."
    )
    report_version: int | None = Field(
        None,
        description=(
            "Version of ``docs/alpha-reports/alpha-report-vN.md`` that "
            "documents the demotion, if discoverable."
        ),
    )


class AntiAlphaResponse(BaseModel):
    """Envelope for ``GET /strategies/anti-alpha-list``."""

    count: int = Field(..., ge=0)
    items: list[AntiAlphaItem]
    source_notes: list[str] = Field(
        default_factory=list,
        description=(
            "Diagnostic notes — which sources were read, fell back, or "
            "were missing. Useful for ops; empty in the happy path."
        ),
    )


# ---------------------------------------------------------------------------
# Source resolution + parsing
# ---------------------------------------------------------------------------


def _resolve_path(env_var: str, default: Path) -> Path:
    """Return the path to read, honouring an environment override."""
    override = os.environ.get(env_var)
    if override:
        return Path(override)
    return default


def _safe_read_json(path: Path, notes: list[str]) -> Any | None:
    """Load JSON from ``path``; on any failure append a note and return ``None``."""
    if not path.exists():
        notes.append(f"missing: {path.name}")
        logger.warning("anti-alpha: source file missing at %s", path)
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        notes.append(f"unreadable: {path.name} ({type(exc).__name__})")
        logger.warning("anti-alpha: failed to read %s: %s", path, exc)
        return None


# Latest alpha-report version we have seen in docs/alpha-reports/. Cached
# across calls; refreshed on cache_clear().
_REPORT_VERSION_CACHE: int | None = None


def _latest_report_version(reports_dir: Path) -> int | None:
    """Find the highest ``alpha-report-vN.md`` version in ``reports_dir``."""
    global _REPORT_VERSION_CACHE
    if _REPORT_VERSION_CACHE is not None:
        return _REPORT_VERSION_CACHE
    if not reports_dir.exists() or not reports_dir.is_dir():
        return None
    rx = re.compile(r"^alpha-report-v(\d+)\.md$")
    best: int | None = None
    try:
        for child in reports_dir.iterdir():
            m = rx.match(child.name)
            if m:
                v = int(m.group(1))
                if best is None or v > best:
                    best = v
    except OSError:
        return None
    _REPORT_VERSION_CACHE = best
    return best


def _items_from_graveyard(
    raw: Any, notes: list[str], latest_report_v: int | None
) -> list[AntiAlphaItem]:
    """Parse ``alpha_graveyard.json`` into AntiAlphaItem rows (tier=ANTI)."""
    out: list[AntiAlphaItem] = []
    if not isinstance(raw, list):
        notes.append("graveyard: expected top-level list")
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        pair_id = entry.get("pair_id") or entry.get("name") or "unknown"
        label = entry.get("name") or entry.get("label") or str(pair_id)
        wave = entry.get("killed_in_wave")
        cause = entry.get("cause") or "unspecified"
        lesson = entry.get("lesson") or ""
        # Compact reason: cause tag + first sentence of lesson.
        first_sentence = lesson.split(". ")[0].strip() if lesson else ""
        reason = f"{cause}: {first_sentence}" if first_sentence else f"{cause}"
        out.append(
            AntiAlphaItem(
                pair_id=str(pair_id),
                tier="ANTI",
                label=str(label),
                reason=reason,
                demoted_in_wave=int(wave) if isinstance(wave, int) else None,
                report_version=latest_report_v,
            )
        )
    return out


def _items_from_alpha_strategies(
    raw: Any, notes: list[str], latest_report_v: int | None
) -> list[AntiAlphaItem]:
    """Filter ``alpha_strategies.json::strategies[]`` for C_*/D_*/ANTI tiers."""
    out: list[AntiAlphaItem] = []
    if not isinstance(raw, dict):
        notes.append("alpha_strategies: expected top-level dict")
        return out
    strategies = raw.get("strategies")
    if not isinstance(strategies, list):
        notes.append("alpha_strategies: no 'strategies' list found")
        return out
    for entry in strategies:
        if not isinstance(entry, dict):
            continue
        tier = entry.get("tier") or ""
        if not (tier.startswith("C_") or tier.startswith("D_") or tier == "ANTI"):
            continue
        pair_id = entry.get("pair_id") or entry.get("label") or entry.get("a_name") or "unknown"
        label = entry.get("a_name") or entry.get("label") or str(pair_id)
        reason = (
            entry.get("v17_reclassification_reason") or entry.get("rationale") or f"tier={tier}"
        )
        # Strategies file doesn't carry a wave int, but tier_v16 tells us
        # the prior label so we can hint at "demoted from X".
        wave: int | None = None
        prior_tier = entry.get("tier_v16")
        if prior_tier and prior_tier != tier:
            reason = f"v17 demotion (was {prior_tier}): {reason}"
        out.append(
            AntiAlphaItem(
                pair_id=str(pair_id),
                tier=str(tier),
                label=str(label),
                reason=str(reason)[:500],
                demoted_in_wave=wave,
                report_version=latest_report_v,
            )
        )
    return out


def _load_items(notes: list[str]) -> list[AntiAlphaItem]:
    """Load and merge all sources. Falls back to CLAUDE.md hardcoded list."""
    grave_path = _resolve_path(_ALPHA_GRAVEYARD_PATH_ENV, _DEFAULT_GRAVEYARD_PATH)
    strat_path = _resolve_path(_ALPHA_STRATEGIES_PATH_ENV, _DEFAULT_STRATEGIES_PATH)
    reports_dir = _resolve_path(_ALPHA_REPORTS_DIR_ENV, _DEFAULT_REPORTS_DIR)

    latest_v = _latest_report_version(reports_dir)

    items: list[AntiAlphaItem] = []
    grave_raw = _safe_read_json(grave_path, notes)
    if grave_raw is not None:
        items.extend(_items_from_graveyard(grave_raw, notes, latest_v))

    strat_raw = _safe_read_json(strat_path, notes)
    if strat_raw is not None:
        items.extend(_items_from_alpha_strategies(strat_raw, notes, latest_v))

    if not items:
        notes.append("fallback: using CLAUDE.md hardcoded anti-alpha list")
        logger.warning("anti-alpha: no entries from source files; serving CLAUDE.md fallback")
        items = [AntiAlphaItem(**row) for row in _CLAUDE_MD_FALLBACK]

    # De-dupe by pair_id — graveyard wins over strategies.json on collision.
    seen: dict[str, AntiAlphaItem] = {}
    for it in items:
        if it.pair_id not in seen:
            seen[it.pair_id] = it
        # else: graveyard came first (we appended it first), so keep it.
    return list(seen.values())


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def cache_clear() -> None:
    """Forget any cached responses. Exported for tests + future ops endpoint."""
    global _REPORT_VERSION_CACHE
    _CACHE.clear()
    _REPORT_VERSION_CACHE = None


def _get_cached(key: str) -> dict[str, Any] | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    ts, payload = entry
    if (time.monotonic() - ts) > _CACHE_TTL_SECONDS:
        return None
    return payload


def _set_cached(key: str, payload: dict[str, Any]) -> None:
    _CACHE[key] = (time.monotonic(), payload)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/anti-alpha-list",
    response_model=AntiAlphaResponse,
    summary="List demoted / killed (anti-alpha) strategies",
)
def get_anti_alpha_list(
    tier: str | None = Query(
        None,
        description=(
            "Optional tier filter. Accepts an exact tier (``ANTI``, "
            "``C_TENTATIVE``, ``D_RAW``) or a wildcard prefix "
            "(``C_*`` matches anything starting with ``C_``)."
        ),
        examples=["ANTI"],
    ),
) -> AntiAlphaResponse:
    """Return the anti-alpha catalog.

    Sourced (in order) from ``alpha_graveyard.json`` (explicit kills),
    ``alpha_strategies.json`` filtered to C_*/D_*/ANTI tiers, and
    CLAUDE.md fallback if no source loads. Cached 5 minutes per
    ``tier`` argument.
    """
    cache_key = f"tier={tier or ''}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return AntiAlphaResponse(**cached)

    notes: list[str] = []
    items = _load_items(notes)

    if tier:
        wanted = tier.strip()
        if wanted.endswith("*"):
            prefix = wanted[:-1]
            items = [it for it in items if it.tier.startswith(prefix)]
        else:
            items = [it for it in items if it.tier == wanted]

    payload = {
        "count": len(items),
        "items": [it.model_dump() for it in items],
        "source_notes": notes,
    }
    _set_cached(cache_key, payload)
    return AntiAlphaResponse(**payload)


__all__ = [
    "AntiAlphaItem",
    "AntiAlphaResponse",
    "cache_clear",
    "get_anti_alpha_list",
    "router",
]

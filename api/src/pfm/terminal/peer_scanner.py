"""Peer-scanner endpoint for the Terminal panel.

Given any market slug, surface ALL cointegrated peers that the
alpha-hunter sweep flagged as ``REAL_ALPHA``, ranked by out-of-sample
Sharpe ratio. The peer list is enriched with display names from
``factors.yml``, theme metadata, and tier classifications from the
gauntlet-validated ``alpha_strategies.json`` file.

Inputs are read once at module import time and cached in process memory:

- ``/tmp/ah_sweeps/all_unique_hits.json`` — 697 cointegrated pairs
  produced by 8 prior alpha-hunter sweeps (politics, macro, crypto,
  sports, ai, geopolitics, pop_health_legal, energy_equity_other).
- ``factors.yml`` — display names + theme tags per factor id.
- ``web/data/alpha_strategies.json`` — tier breakdown
  (``A_STRUCTURAL``, ``A_GOLD``, ``B_VALIDATED``, ``B_FDR_ONLY``,
  ``C_TENTATIVE``, ``D_RAW``) for the curated subset of 88 pairs.

Routing
-------
This module owns its own :class:`fastapi.APIRouter`. Per project
convention ``main.py`` is left untouched — wire it explicitly via::

    from pfm.terminal_peer_scanner import router as terminal_peer_router
    app.include_router(terminal_peer_router)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from fastapi import APIRouter, Query
from fastapi import Path as FPath
from fastapi.responses import Response as FastAPIResponse

from pfm.cache_utils import get_cache
from pfm.terminal_export import respond as _export_respond

# A peer in the degraded fall-back path scores by theme + name overlap,
# not cointegration — labelled as such so callers can show the right
# warning in the UI.
_DEGRADED_FALLBACK_LIMIT: int = 50

logger = logging.getLogger(__name__)


# --- file-location constants ------------------------------------------------
# These point at locations that exist in the dev environment. Tests
# monkeypatch the cache loader to avoid file IO.
DEFAULT_HITS_PATH: Path = Path("/tmp/ah_sweeps/all_unique_hits.json")
# 2026-05 refactor: module moved into ``pfm/terminal/``; bump parent depth by
# one to keep pointing at the same on-disk files.
DEFAULT_FACTORS_PATH: Path = Path(__file__).resolve().parents[1] / "factors.yml"
DEFAULT_STRATEGIES_PATH: Path = (
    Path(__file__).resolve().parents[4] / "web" / "data" / "alpha_strategies.json"
)


# --- shared cache (1h TTL ≈ "until process restart" for the hits sweep) -----
# All three loaders share one namespaced TerminalCache instance with three
# distinct keys: "hits", "factors", "tiers". Hour-long TTL effectively
# preserves the original "load once per process" memoisation semantics
# while letting tests force a refresh via clear_cache().
_PEERS_CACHE = get_cache("peers", ttl=3600)


def _load_hits(path: Path = DEFAULT_HITS_PATH) -> list[dict[str, Any]]:
    """Load the 697 alpha-hunter pair-hits from disk (memoised)."""
    cached = _PEERS_CACHE.get("hits")
    if cached is not None:
        return cached
    if not path.exists():
        logger.warning("peer-scanner: hits file missing at %s", path)
        _PEERS_CACHE.set("hits", [])
        return []
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"expected list at {path}, got {type(raw).__name__}")
    _PEERS_CACHE.set("hits", raw)
    return raw


def _load_factors(path: Path = DEFAULT_FACTORS_PATH) -> dict[str, dict[str, str]]:
    """Load id → {name, theme, slug} from ``factors.yml`` (memoised)."""
    cached = _PEERS_CACHE.get("factors")
    if cached is not None:
        return cached
    if not path.exists():
        logger.warning("peer-scanner: factors file missing at %s", path)
        _PEERS_CACHE.set("factors", {})
        return {}
    with path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    out: dict[str, dict[str, str]] = {}
    for f_def in doc.get("factors", []) or []:
        fid = f_def.get("id")
        if not fid:
            continue
        out[fid] = {
            "name": f_def.get("name") or fid,
            "theme": f_def.get("theme") or "unknown",
            "slug": f_def.get("slug") or "",
        }
    _PEERS_CACHE.set("factors", out)
    return out


def _load_tiers(path: Path = DEFAULT_STRATEGIES_PATH) -> dict[str, str]:
    """Load pair_id → tier from ``alpha_strategies.json`` (memoised).

    The pair_id key is constructed by sorting (a_id, b_id) lexically and
    joining with ``__`` so the lookup is symmetric regardless of which
    side of the pair the user queried.
    """
    cached = _PEERS_CACHE.get("tiers")
    if cached is not None:
        return cached
    if not path.exists():
        logger.warning("peer-scanner: strategies file missing at %s", path)
        _PEERS_CACHE.set("tiers", {})
        return {}
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    out: dict[str, str] = {}
    for s in doc.get("strategies", []) or []:
        a, b, tier = s.get("a_id"), s.get("b_id"), s.get("tier")
        if not (a and b and tier):
            continue
        key = "__".join(sorted([a, b]))
        out[key] = tier
    _PEERS_CACHE.set("tiers", out)
    return out


def clear_cache() -> None:
    """Test/utility helper — drop all in-process caches."""
    _PEERS_CACHE.clear()


# --- core logic -------------------------------------------------------------


def _pair_key(a: str, b: str) -> str:
    return "__".join(sorted([a, b]))


def _enrich_peer(
    pair: dict[str, Any],
    input_slug: str,
    factors: dict[str, dict[str, str]],
    tiers: dict[str, str],
) -> dict[str, Any] | None:
    """Convert a raw alpha-hunter pair row into a peer record.

    Returns ``None`` if ``input_slug`` isn't one of the legs.
    """
    a, b = pair.get("a_id"), pair.get("b_id")
    if a is None or b is None:
        return None
    if input_slug == a:
        peer_id = b
    elif input_slug == b:
        peer_id = a
    else:
        return None

    peer_meta = factors.get(peer_id, {})
    self_meta = factors.get(input_slug, {})

    # Themes — fall back to the sweep label if factors.yml lacks the id.
    sweep_theme = pair.get("sweep") or "unknown"
    theme_self = self_meta.get("theme") or sweep_theme
    theme_peer = peer_meta.get("theme") or sweep_theme

    # Lookup tier; default to D_RAW if not in the curated subset (every
    # pair in the hits file IS REAL_ALPHA so the floor is D_RAW).
    tier = tiers.get(_pair_key(a, b), "D_RAW")

    return {
        "peer_slug": peer_id,
        "peer_name": peer_meta.get("name") or peer_id,
        "oos_sharpe": float(pair.get("oos_sharpe", 0.0)),
        "perm_p": float(pair.get("perm_p", 1.0)),
        "half_life_days": float(pair.get("half_life_days", 0.0)),
        "beta_hedge": float(pair.get("beta_hedge", 0.0)),
        "theme_a": theme_self,
        "theme_b": theme_peer,
        "verdict": pair.get("verdict") or "UNKNOWN",
        "tier": tier,
        "n_obs": int(pair.get("n_obs", 0)),
        "adf_pvalue": float(pair.get("adf_pvalue", 1.0)),
        "sweep": pair.get("sweep") or "unknown",
    }


def find_basic_peers(
    slug: str,
    factors: dict[str, dict[str, str]],
    *,
    top: int = 20,
) -> list[dict[str, Any]]:
    """Theme-only fallback when the alpha-hunter sweep cache is empty.

    Picks every other factor sharing the input's theme and emits a degraded
    peer record (no Sharpe, no cointegration evidence). The score is a
    simple token-overlap on the display name so the most plausibly-related
    peers float to the top. Used by ``get_peers`` when ``_load_hits`` is
    empty so the endpoint can return 200 + degraded data instead of 404.
    """
    self_meta = factors.get(slug, {})
    self_theme = self_meta.get("theme") or "unknown"
    self_name = (self_meta.get("name") or slug).lower()
    self_tokens = set(re.findall(r"[a-z0-9]+", self_name))

    candidates: list[tuple[int, dict[str, Any]]] = []
    for fid, meta in factors.items():
        if fid == slug:
            continue
        peer_theme = meta.get("theme") or "unknown"
        if peer_theme != self_theme or self_theme == "unknown":
            continue
        peer_name = (meta.get("name") or fid).lower()
        peer_tokens = set(re.findall(r"[a-z0-9]+", peer_name))
        score = len(self_tokens & peer_tokens)
        candidates.append(
            (
                score,
                {
                    "peer_slug": fid,
                    "peer_name": meta.get("name") or fid,
                    "oos_sharpe": None,
                    "perm_p": None,
                    "half_life_days": None,
                    "beta_hedge": None,
                    "theme_a": self_theme,
                    "theme_b": peer_theme,
                    "verdict": "DEGRADED_THEME_MATCH",
                    "tier": "UNRANKED",
                    "n_obs": 0,
                    "adf_pvalue": None,
                    "sweep": "fallback",
                },
            )
        )

    # Sort by overlap score desc, name asc for stable pagination.
    candidates.sort(key=lambda t: (-t[0], t[1]["peer_slug"]))
    return [c[1] for c in candidates[: max(0, int(top))]]


def find_peers(
    slug: str,
    *,
    top: int = 20,
    min_sharpe: float = 0.5,
) -> dict[str, Any]:
    """Return cointegrated peers for ``slug`` ranked by OOS Sharpe.

    Args:
        slug: Factor id (matches ``a_id`` / ``b_id`` in the hits file).
        top: Maximum number of peers in the response (post-sort).
        min_sharpe: Discard peers below this OOS Sharpe.

    Returns:
        Response dict with ``peers``, ``n_peers``, ``cross_theme_count``,
        ``tier_summary``, and ``best_peer``.
    """
    hits = _load_hits()
    factors = _load_factors()
    tiers = _load_tiers()

    enriched: list[dict[str, Any]] = []
    for pair in hits:
        rec = _enrich_peer(pair, slug, factors, tiers)
        if rec is None:
            continue
        if rec["oos_sharpe"] < min_sharpe:
            continue
        enriched.append(rec)

    # Sort by OOS Sharpe desc, take top N.
    enriched.sort(key=lambda d: d["oos_sharpe"], reverse=True)
    top_n = enriched[: max(0, int(top))]

    # Cross-theme counter — peers whose theme differs from the input's.
    # Compute on the truncated set so the metric matches the visible list.
    cross_theme_count = sum(1 for p in top_n if p["theme_a"] != p["theme_b"])

    # Tier summary on the same truncated set.
    tier_summary: dict[str, int] = {}
    for p in top_n:
        tier_summary[p["tier"]] = tier_summary.get(p["tier"], 0) + 1

    best_peer = top_n[0] if top_n else None

    return {
        "slug": slug,
        "n_peers": len(top_n),
        "peers": top_n,
        "cross_theme_count": cross_theme_count,
        "tier_summary": tier_summary,
        "best_peer": best_peer,
    }


# --- router -----------------------------------------------------------------

router = APIRouter(prefix="/terminal", tags=["terminal-peer-scanner"])


@router.get("/peers/{slug}", response_model=None)
def get_peers(
    slug: Annotated[str, FPath(min_length=1, max_length=120)],
    top: Annotated[int, Query(ge=1, le=200)] = 20,
    min_sharpe: Annotated[float, Query(ge=-10.0, le=20.0)] = 0.5,
    format: Annotated[Literal["json", "csv", "pdf"], Query()] = "json",
) -> dict[str, Any] | FastAPIResponse:
    """Cointegrated-peer lookup for a factor slug.

    Searches the alpha-hunter sweep cache for every pair where ``slug``
    is one of the two legs and returns the top ``top`` peers by
    out-of-sample Sharpe (filtered by ``min_sharpe``). Each peer is
    annotated with display name and theme from ``factors.yml`` and a
    tier classification from the curated strategies file.

    When the alpha-hunter sweep cache is empty (or missing on disk), the
    endpoint degrades gracefully: it returns 200 with ``degraded_mode=true``
    and a basic theme-matched peer list derived from ``factors.yml`` so the
    UI panel still has something to render. ``reason`` carries the cause
    (``alpha_hunter_cache_unavailable``). An empty peer list with
    ``n_peers == 0`` and ``degraded_mode == false`` is also a valid response
    — it means the slug has no cointegrated counterparts in a populated
    sweep set.
    """
    # Cache the compute output. The underlying alpha-hunter sweep cache is
    # rarely-changing (regenerated nightly); 5-minute TTL keeps the response
    # snappy across all gunicorn workers via Redis L2.
    from pfm import terminal as _term_mod  # lazy import to avoid cycle

    cache_key = f"peers::{slug}::{top}::{min_sharpe:.2f}"
    cached = _term_mod.TERMINAL_CACHE.get(cache_key)
    if cached is not None and format == "json":
        return cached

    hits = _load_hits()
    if not hits:
        factors = _load_factors()
        basic = find_basic_peers(slug, factors, top=top)
        cross = sum(1 for p in basic if p["theme_a"] != p["theme_b"])
        payload: dict[str, Any] = {
            "slug": slug,
            "n_peers": len(basic),
            "peers": basic,
            "cross_theme_count": cross,
            "tier_summary": {"UNRANKED": len(basic)} if basic else {},
            "best_peer": basic[0] if basic else None,
            "degraded_mode": True,
            "reason": "alpha_hunter_cache_unavailable",
        }
        # Cache degraded payloads briefly too — same key shape so a sweep
        # backfill auto-supersedes via TTL expiry.
        if format == "json":
            _term_mod.TERMINAL_CACHE.set(cache_key, payload, 60)
            return payload
        return _export_respond(payload, format, filename=f"peers-{slug}", kind="peers")

    payload = find_peers(slug, top=top, min_sharpe=min_sharpe)
    payload["degraded_mode"] = False
    payload["reason"] = None
    # When the sweep cache IS populated but this specific slug has no peers,
    # surface that explicitly. Otherwise the UI sees ``n_peers=0`` and
    # ``degraded_mode=false`` and can't tell whether the slug genuinely
    # has no cointegrated counterparts or the panel just failed silently.
    if payload.get("n_peers", 0) == 0:
        payload["degraded_mode"] = True
        payload["reason"] = (
            f"no peers found for slug={slug!r} above min_sharpe={min_sharpe} "
            "in the alpha-hunter sweep — the slug may be too new, too thin, "
            "or genuinely uncorrelated with the rest of the catalog."
        )
    if format == "json":
        _term_mod.TERMINAL_CACHE.set(cache_key, payload, 300)
        return payload
    return _export_respond(payload, format, filename=f"peers-{slug}", kind="peers")


__all__ = [
    "DEFAULT_FACTORS_PATH",
    "DEFAULT_HITS_PATH",
    "DEFAULT_STRATEGIES_PATH",
    "clear_cache",
    "find_basic_peers",
    "find_peers",
    "get_peers",
    "router",
]

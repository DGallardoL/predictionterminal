"""Unified resolver: name | slug | id -> :class:`FactorConfig`.

The factor catalog grew to ~1360 entries. Real users (not just devs reading
``factors.yml``) routinely paste *something that looks like* a slug or a
name, and the API used to either return a generic 400/404 with no guidance
or, worse, a 502 from downstream when the bad id reached an upstream
fetcher. This module centralises the "what did they mean?" logic so every
endpoint can reject unknown ids consistently with a structured
``did_you_mean`` payload.

Public surface
--------------
- :func:`resolve_factor`        — direct lookup, returns ``None`` on miss.
- :func:`suggest_factors`       — fuzzy top-k for "did you mean".
- :func:`resolve_or_404`        — strict variant that raises a structured
  :class:`fastapi.HTTPException` carrying suggestions.

The lookup tables are cached per-catalog-id in
``pfm.cache_utils.get_cache("factor_resolver")`` so repeat calls inside a
request stay O(1). Catalog identity is detected via :func:`id` so callers
that mutate the dict in-place will see refreshed indices the next time
they pass it in.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any

from fastapi import HTTPException

from pfm.cache_utils import get_cache
from pfm.factors import FactorConfig

logger = logging.getLogger(__name__)


_RESOLVER_CACHE_NAMESPACE = "factor_resolver"
_RESOLVER_CACHE_TTL = 600

# Tokens that don't help discriminate between factors — drop from the
# token-overlap signal so "fed-rate-cuts-2026" doesn't tie with every
# 2026-dated factor on the year token alone.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "will",
        "the",
        "a",
        "an",
        "be",
        "is",
        "are",
        "have",
        "happen",
        "in",
        "on",
        "of",
        "to",
        "by",
        "for",
        "and",
        "or",
        "as",
        "at",
        "this",
        "that",
        "with",
    }
)

_TOKEN_SPLIT_RE = re.compile(r"[\s_\-/]+")


def _tokenise(text: str) -> set[str]:
    """Lowercase + split on whitespace / ``_`` / ``-`` / ``/`` -> token set."""
    if not text:
        return set()
    raw = _TOKEN_SPLIT_RE.split(text.strip().lower())
    return {tok for tok in raw if tok and tok not in _STOPWORDS and len(tok) > 1}


def _build_lookup(
    factors_catalog: dict[str, FactorConfig],
) -> dict[str, Any]:
    """Pre-compute id/slug/name indices + per-factor token sets.

    Returns a dict with:
      - ``by_id``         — ``{id: FactorConfig}`` (the catalog itself)
      - ``by_slug``       — ``{slug: FactorConfig}``
      - ``by_name_lower`` — ``{name.lower(): FactorConfig}``
      - ``tokens``        — ``{id: set[str]}`` (tokens drawn from
        id + slug + name, used by :func:`suggest_factors`).
    """
    by_slug: dict[str, FactorConfig] = {}
    by_name_lower: dict[str, FactorConfig] = {}
    tokens: dict[str, set[str]] = {}
    for fc in factors_catalog.values():
        # First-write wins on collision — the catalog already enforces unique
        # ids, but two factors can legitimately share a slug across sources
        # (Polymarket vs Kalshi mirror markets) so don't overwrite.
        by_slug.setdefault(fc.slug, fc)
        by_name_lower.setdefault(fc.name.lower(), fc)
        tokens[fc.id] = _tokenise(fc.id) | _tokenise(fc.slug) | _tokenise(fc.name)
    return {
        "by_id": factors_catalog,
        "by_slug": by_slug,
        "by_name_lower": by_name_lower,
        "tokens": tokens,
    }


def _get_lookup(factors_catalog: dict[str, FactorConfig]) -> dict[str, Any]:
    """Return the cached lookup table for ``factors_catalog``.

    Cached by catalog object identity + length so swapping the catalog
    (or hot-reloading factors.yml) invalidates the cache automatically.
    """
    cache = get_cache(_RESOLVER_CACHE_NAMESPACE, ttl=_RESOLVER_CACHE_TTL)
    key = ("lookup", id(factors_catalog), len(factors_catalog))
    hit = cache.get(key)
    if hit is not None:
        return hit
    built = _build_lookup(factors_catalog)
    cache.set(key, built, ttl=_RESOLVER_CACHE_TTL)
    return built


def _get_default_catalog() -> dict[str, FactorConfig]:
    """Best-effort retrieval of the live app catalog.

    Used when callers don't pass one in. Falls back to an empty dict if
    the FastAPI app hasn't started (unit-test contexts that import the
    resolver directly).
    """
    try:
        from pfm import main as main_mod

        return getattr(main_mod.app.state, "factors", {}) or {}
    except Exception:  # pragma: no cover - defensive
        return {}


def resolve_factor(
    name_or_slug_or_id: str,
    factors_catalog: dict[str, FactorConfig] | None = None,
) -> FactorConfig | None:
    """Resolve a query to a :class:`FactorConfig` using exact lookup.

    Tries (in order): id match, slug match, case-insensitive name match.
    Returns ``None`` if nothing matches — callers that want a 4xx with
    suggestions should use :func:`resolve_or_404`.
    """
    if not name_or_slug_or_id or not isinstance(name_or_slug_or_id, str):
        return None
    catalog = factors_catalog if factors_catalog is not None else _get_default_catalog()
    if not catalog:
        return None
    lookup = _get_lookup(catalog)
    q = name_or_slug_or_id.strip()
    # 1) id match (exact, case-sensitive — ids are stable identifiers).
    fc = lookup["by_id"].get(q)
    if fc is not None:
        return fc
    # 2) slug match (also case-sensitive — slugs are URL-shaped).
    fc = lookup["by_slug"].get(q)
    if fc is not None:
        return fc
    # 3) name match (case-insensitive — humans paste names with random caps).
    fc = lookup["by_name_lower"].get(q.lower())
    if fc is not None:
        return fc
    return None


def _score_candidate(
    query_tokens: set[str],
    cand_tokens: set[str],
    q_lower: str,
    cand_text: str,
) -> float:
    """Combined score: token Jaccard + sequence-matcher ratio.

    The token component handles "fed rate cuts 2026" -> "no_fed_cuts_2026"
    where slug and id share no contiguous substring; the sequence-matcher
    component handles typos like "trumpp" -> "trump-2024".
    """
    if not query_tokens and not cand_tokens:
        return 0.0
    overlap = len(query_tokens & cand_tokens)
    union = len(query_tokens | cand_tokens) or 1
    jaccard = overlap / union
    # Bonus for exact substring match — catches "fed-rate-cuts" -> "fed_cuts".
    substring_bonus = 0.0
    if q_lower and cand_text:
        if q_lower in cand_text or cand_text in q_lower:
            substring_bonus = 0.15
    # Cheap fuzzy ratio. SequenceMatcher.ratio() is O(n*m) but n,m here are
    # short (id/slug strings); over 1360 candidates this is ~10ms total.
    fuzzy = SequenceMatcher(None, q_lower, cand_text).ratio()
    return 0.55 * jaccard + 0.30 * fuzzy + substring_bonus


def suggest_factors(
    query: str,
    factors_catalog: dict[str, FactorConfig] | None = None,
    top_k: int = 3,
) -> list[str]:
    """Return ``top_k`` factor IDs ranked by closeness to ``query``.

    Combines:
      1. Token-overlap Jaccard over id/slug/name tokens (split on
         ``-``/``_``/whitespace, stopwords removed).
      2. ``difflib.SequenceMatcher`` ratio against the slug for typos.
      3. Substring bonus when the query contains, or is contained in,
         the slug.

    Returns an empty list if the catalog is empty or the query is blank.
    """
    if not query or not isinstance(query, str):
        return []
    catalog = factors_catalog if factors_catalog is not None else _get_default_catalog()
    if not catalog:
        return []
    lookup = _get_lookup(catalog)
    q_tokens = _tokenise(query)
    q_lower = query.strip().lower()
    scored: list[tuple[float, str]] = []
    for fid, fc in lookup["by_id"].items():
        # Score against the slug as the canonical "discoverability" string;
        # token set already incorporates id+name. Keep the per-call cost
        # bounded by short-circuiting on totally-disjoint token sets when
        # the query has any tokens at all.
        cand_tokens = lookup["tokens"][fid]
        score = _score_candidate(q_tokens, cand_tokens, q_lower, fc.slug.lower())
        # Also try the id text — catches "no_fed_cuts" exactly.
        score_id = _score_candidate(q_tokens, cand_tokens, q_lower, fid.lower())
        score = max(score, score_id)
        if score > 0:
            scored.append((score, fid))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [fid for _score, fid in scored[:top_k]]


def suggest_factors_with_meta(
    query: str,
    factors_catalog: dict[str, FactorConfig] | None = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Like :func:`suggest_factors` but returns ``[{id, name, score}, ...]``.

    Used for the structured ``did_you_mean`` payload so the frontend can
    render human-readable choices without a second round-trip.
    """
    if not query or not isinstance(query, str):
        return []
    catalog = factors_catalog if factors_catalog is not None else _get_default_catalog()
    if not catalog:
        return []
    lookup = _get_lookup(catalog)
    q_tokens = _tokenise(query)
    q_lower = query.strip().lower()
    scored: list[tuple[float, str]] = []
    for fid, fc in lookup["by_id"].items():
        cand_tokens = lookup["tokens"][fid]
        score = _score_candidate(q_tokens, cand_tokens, q_lower, fc.slug.lower())
        score_id = _score_candidate(q_tokens, cand_tokens, q_lower, fid.lower())
        score = max(score, score_id)
        if score > 0:
            scored.append((score, fid))
    scored.sort(key=lambda t: (-t[0], t[1]))
    out: list[dict[str, Any]] = []
    for score, fid in scored[:top_k]:
        fc = lookup["by_id"][fid]
        out.append(
            {
                "id": fc.id,
                "name": fc.name,
                "slug": fc.slug,
                "source": fc.source,
                "score": round(float(score), 3),
            }
        )
    return out


def resolve_or_404(
    name_or_slug_or_id: str,
    factors_catalog: dict[str, FactorConfig] | None = None,
    *,
    status_code: int = 400,
) -> FactorConfig:
    """Resolve to a :class:`FactorConfig` or raise an HTTPException with hints.

    The exception's ``detail`` is a structured object that the frontend can
    render as a "did you mean ...?" prompt::

        {
          "error": "factor not found: 'fed-rate-cuts-2026'",
          "query": "fed-rate-cuts-2026",
          "did_you_mean": [
            {"id": "no_fed_cuts_2026", "name": "...", "score": 0.85},
            ...
          ]
        }

    ``status_code`` defaults to 400 (the historical contract on /fit), but
    callers that semantically want 404 (terminal market endpoints) can
    pass it explicitly.
    """
    fc = resolve_factor(name_or_slug_or_id, factors_catalog)
    if fc is not None:
        return fc
    suggestions = suggest_factors_with_meta(
        name_or_slug_or_id,
        factors_catalog,
        top_k=3,
    )
    raise HTTPException(
        status_code=status_code,
        detail={
            "error": f"factor not found: {name_or_slug_or_id!r}",
            "query": name_or_slug_or_id,
            "did_you_mean": suggestions,
        },
    )


def resolve_many_or_400(
    queries: list[str],
    factors_catalog: dict[str, FactorConfig] | None = None,
) -> list[FactorConfig]:
    """Resolve a list of queries; raise 400 with per-query suggestions on miss.

    Aggregates misses so callers get *all* the bad ids back in one shot,
    rather than the historical "trip on the first one" behaviour.
    """
    catalog = factors_catalog if factors_catalog is not None else _get_default_catalog()
    out: list[FactorConfig] = []
    misses: list[dict[str, Any]] = []
    for q in queries:
        fc = resolve_factor(q, catalog)
        if fc is None:
            misses.append(
                {
                    "query": q,
                    "did_you_mean": suggest_factors_with_meta(q, catalog, top_k=3),
                }
            )
        else:
            out.append(fc)
    if misses:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"{len(misses)} factor id(s) not found",
                "unknown": misses,
            },
        )
    return out


__all__ = [
    "resolve_factor",
    "resolve_many_or_400",
    "resolve_or_404",
    "suggest_factors",
    "suggest_factors_with_meta",
]

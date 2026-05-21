"""Recall-plus-precision matcher for full-universe cross-venue arb discovery.

The cross-venue arb scanner (:mod:`pfm.arb_scanner`) currently pairs Kalshi
and Polymarket markets with a *weak* similarity (keyword Jaccard + date +
theme). That is fine for the curated, capacity-limited universe it scans
today, but the moment discovery is scaled to the **full** market universe
(thousands of markets per venue, unlimited), the weak matcher produces a
flood of false positives: any two markets that share a topic word ("Senate",
"Bitcoin", "win") get paired regardless of jurisdiction, threshold, or
resolution window.

This module combines two stages:

1. **Recall** — a cheap, O(N+M) *prefilter* (token / shared-entity overlap)
   that produces a small set of plausible candidate pairs without running the
   (relatively expensive) strict scorer on the full O(N×M) cross-product.

2. **Precision** — the strict
   :func:`pfm.arb_matching.event_similarity.score_match` gates
   (same-venue / jurisdiction-mismatch / threshold-mismatch > 5 % /
   resolution-window non-overlap) run *only* on the prefiltered pairs. We do
   **not** reinvent these gates; we reuse ``score_match`` verbatim.

The public surface is :class:`Candidate`, :func:`match_one`,
:func:`match_markets` and :func:`summarize`.

Normalization
-------------
``score_match`` consumes :class:`~pfm.arb_matching.event_similarity.MarketDesc`
objects, built from venue-native payloads via ``build_market_desc(payload,
venue)``. We mirror exactly how ``pfm.arb.quality_router._score_pair``
normalizes its inputs: a payload dict ``{"title", "description", "slug"}`` for
the Polymarket leg and ``{"title", "description", "ticker"}`` for the Kalshi
leg, then ``build_market_desc(payload, venue)``.

Tier thresholds are aligned with the ``/arb/quality-audit`` endpoint:

- ``"high"``       — score ``>= 0.7`` and not rejected.
- ``"borderline"`` — score in ``[0.4, 0.7)`` and not rejected.
- ``"reject"``     — hard-rejected, or score ``< 0.4``.

``score_match`` is pure / offline (regex + set Jaccard + date arithmetic); it
never downloads a model, so this module is safe to drive from no-network
tests.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from pfm.arb_matching.event_similarity import (
    MarketDesc,
    build_market_desc,
    score_match,
)

__all__ = [
    "Candidate",
    "match_markets",
    "match_one",
    "summarize",
]


# ---------------------------------------------------------------------------
# Tier thresholds — kept in sync with pfm.arb.quality_router.
# ---------------------------------------------------------------------------

TIER_HIGH = "high"
TIER_BORDERLINE = "borderline"
TIER_REJECT = "reject"

#: Score at/above which a non-rejected pair is "high" confidence.
HIGH_SCORE = 0.7
#: Score below which a pair is treated as a reject even without a hard gate.
MIN_SCORE = 0.4


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """A scored cross-venue arb candidate pair.

    Attributes:
        kalshi_ticker: Kalshi market identifier (ticker), best-effort.
        kalshi_title: Kalshi market title.
        poly_slug: Polymarket market slug, best-effort.
        poly_title: Polymarket market title.
        score: ``score_match`` total in ``[0, 1]`` (``0.0`` when rejected).
        components: Per-component soft-score breakdown from ``score_match``.
        rejected: ``True`` when hard-rejected OR ``score < MIN_SCORE``.
        reject_reason: One of the ``score_match`` reject reasons, ``"low_score"``
            for sub-threshold pairs, or ``None`` when the pair is kept.
        tier: ``"high"`` / ``"borderline"`` / ``"reject"`` (see module docstring).
        new_side: Which venue side was freshly listed (``"kalshi"`` / ``"poly"``
            / ``"both"``), or ``None`` when freshness is not relevant. Only the
            ``mode="new"`` discovery path sets this; everywhere else it stays
            ``None``. Additive / backward-compatible.
    """

    kalshi_ticker: str
    kalshi_title: str
    poly_slug: str
    poly_title: str
    score: float
    components: dict[str, float]
    rejected: bool
    reject_reason: str | None
    tier: str
    new_side: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a plain-dict view (handy for JSON responses)."""

        return {
            "kalshi_ticker": self.kalshi_ticker,
            "kalshi_title": self.kalshi_title,
            "poly_slug": self.poly_slug,
            "poly_title": self.poly_title,
            "score": self.score,
            "components": dict(self.components),
            "rejected": self.rejected,
            "reject_reason": self.reject_reason,
            "tier": self.tier,
            "new_side": self.new_side,
        }


# ---------------------------------------------------------------------------
# Input normalization — mirrors pfm.arb.quality_router._score_pair.
# ---------------------------------------------------------------------------


def _poly_desc(poly_item: dict[str, Any]) -> MarketDesc:
    """Normalize a Polymarket-native payload into a ``MarketDesc``."""

    payload = {
        "title": _first(poly_item, "title", "question", "name"),
        "description": _first(poly_item, "description", "subtitle", "rules_primary"),
        "slug": _first(poly_item, "slug", "poly_slug"),
    }
    return build_market_desc(payload, "polymarket")


def _kalshi_desc(kalshi_item: dict[str, Any]) -> MarketDesc:
    """Normalize a Kalshi-native payload into a ``MarketDesc``."""

    payload = {
        "title": _first(kalshi_item, "title", "name"),
        "description": _first(kalshi_item, "description", "subtitle", "rules_primary"),
        "ticker": _first(kalshi_item, "ticker", "kalshi_ticker"),
    }
    return build_market_desc(payload, "kalshi")


def _first(item: dict[str, Any], *keys: str) -> str:
    """Return the first non-empty string among ``keys`` in ``item``."""

    for key in keys:
        val = item.get(key)
        if val:
            return str(val).strip()
    return ""


def _tier_for(score: float, rejected: bool) -> str:
    """Bucket a (score, rejected) pair into a tier label."""

    if rejected:
        return TIER_REJECT
    if score >= HIGH_SCORE:
        return TIER_HIGH
    if score >= MIN_SCORE:
        return TIER_BORDERLINE
    return TIER_REJECT


# ---------------------------------------------------------------------------
# Single-pair scoring
# ---------------------------------------------------------------------------


def match_one(kalshi_item: dict[str, Any], poly_item: dict[str, Any]) -> Candidate:
    """Score one Kalshi/Polymarket pair through the strict ``score_match`` gates.

    Args:
        kalshi_item: Kalshi-native market payload (``title``/``ticker``/...).
        poly_item: Polymarket-native market payload (``question``/``slug``/...).

    Returns:
        A :class:`Candidate`. A pair is ``rejected`` when ``score_match`` fires
        a hard-reject gate OR when the soft score is below :data:`MIN_SCORE`
        (``reject_reason="low_score"`` in the latter case).
    """

    poly = _poly_desc(poly_item)
    kalshi = _kalshi_desc(kalshi_item)
    return _score_pair(kalshi_item, poly_item, kalshi, poly)


def _score_pair(
    kalshi_item: dict[str, Any],
    poly_item: dict[str, Any],
    kalshi: MarketDesc,
    poly: MarketDesc,
) -> Candidate:
    """Score one pair from *prebuilt* ``MarketDesc`` objects.

    Identical semantics to :func:`match_one` but lets the bulk matcher build
    each ``MarketDesc`` exactly once instead of re-extracting on every pair.
    """

    result = score_match(poly, kalshi)
    total = float(result.total or 0.0)
    components = dict(result.components)
    hard_reason = result.rejected_reason or None

    if hard_reason is not None:
        rejected = True
        reason: str | None = hard_reason
        # Hard rejects force total to 0.0 inside score_match; keep that.
        total = 0.0
    elif total < MIN_SCORE:
        rejected = True
        reason = "low_score"
    else:
        rejected = False
        reason = None

    return Candidate(
        kalshi_ticker=_first(kalshi_item, "ticker", "kalshi_ticker"),
        kalshi_title=_first(kalshi_item, "title", "name"),
        poly_slug=_first(poly_item, "slug", "poly_slug"),
        poly_title=_first(poly_item, "title", "question", "name"),
        score=round(total, 4),
        components=components,
        rejected=rejected,
        reject_reason=reason,
        tier=_tier_for(total, rejected),
    )


# ---------------------------------------------------------------------------
# Prefilter (recall stage)
# ---------------------------------------------------------------------------

_PREFILTER_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'\-]*")
_BARE_YEAR_RE = re.compile(r"^(19|20)\d{2}$")

# Generic words that almost every market shares; using them as a bridge
# between two markets produces noise, so they don't count as a discriminative
# token for blocking purposes. Bare 4-digit years are dropped separately via
# ``_BARE_YEAR_RE`` (any year, not just an enumerated handful).
_PREFILTER_STOPWORDS: frozenset[str] = frozenset(
    {
        # connectives / function words
        "will",
        "the",
        "by",
        "in",
        "on",
        "at",
        "of",
        "to",
        "a",
        "an",
        "and",
        "or",
        "for",
        "with",
        "be",
        "is",
        "are",
        "was",
        "were",
        "than",
        "this",
        "that",
        "it",
        "its",
        "as",
        "from",
        "up",
        "down",
        # comparison / threshold words
        "above",
        "below",
        "over",
        "under",
        "before",
        "after",
        "reach",
        "reaches",
        "hit",
        "hits",
        "trade",
        "trades",
        "close",
        "closes",
        # ultra-generic market vocabulary
        "win",
        "wins",
        "winner",
        "won",
        "lose",
        "loses",
        "market",
        "markets",
        "price",
        "prices",
        "yes",
        "no",
        "event",
        "events",
        "race",
        "odds",
        "bet",
        "contract",
        "question",
    }
)

#: A discriminative token must appear in at most this fraction of the poly
#: corpus to be indexed (an idf-style document-frequency cap). Tokens above
#: this cap are too common to discriminate between markets, so indexing on
#: them would re-introduce the "everything shares 'bitcoin'" blow-up.
_DF_CAP_FRACTION = 0.15

#: Absolute floor on the document-frequency cap. Below this corpus size the
#: 15% fraction rounds to a tiny integer that would prune almost everything
#: (e.g. on 5 identical markets ``int(0.15*5)==0``), so we never prune a key
#: that appears in fewer than this many markets. At full-universe scale the
#: fraction dominates this floor and does the real work.
_DF_CAP_FLOOR = 50


def _df_cap(n_poly: int) -> int:
    """Document-frequency cap for the inverted index given the corpus size."""

    return max(_DF_CAP_FLOOR, int(_DF_CAP_FRACTION * n_poly))


def _content_tokens(item: dict[str, Any]) -> set[str]:
    """Cheap discriminative content-token set from title + description.

    Lowercases, drops short tokens, stopwords and bare years. This is the
    coarse signal used to *bridge* two markets in the blocking prefilter; it
    deliberately does not try to be exhaustive.
    """

    text = " ".join(
        str(item.get(k, "") or "") for k in ("title", "question", "name", "description", "subtitle")
    ).lower()
    toks: set[str] = set()
    for m in _PREFILTER_TOKEN_RE.findall(text):
        if len(m) <= 2 or m in _PREFILTER_STOPWORDS or _BARE_YEAR_RE.match(m):
            continue
        toks.add(m)
    return toks


def _blocking_keys(item: dict[str, Any], desc: MarketDesc) -> set[str]:
    """Discriminative blocking keys for a market.

    Combines the cheap content tokens (:func:`_content_tokens`) with the
    NER-style entity tokens already extracted by ``build_market_desc`` (people,
    tickers, countries). Entities are namespaced (``"ent:"`` prefix) so a rare
    entity and a same-spelled common word don't accidentally collapse, and so
    an entity match is a *distinct* bridge from a plain content-token match.

    Args:
        item: venue-native market payload.
        desc: the ``MarketDesc`` already built for ``item`` (reused to avoid
            re-running the comparatively expensive extraction in ``score_match``).

    Returns:
        The set of discriminative keys that index/lookup this market.
    """

    keys = _content_tokens(item)
    for ent in desc.entities:
        # entities are already lowercase-normalised and stop-filtered upstream.
        if _BARE_YEAR_RE.match(ent):
            continue
        keys.add("ent:" + ent)
    return keys


# ---------------------------------------------------------------------------
# Bulk matching (recall + precision)
# ---------------------------------------------------------------------------


def match_markets(
    kalshi_items: list[dict[str, Any]],
    poly_items: list[dict[str, Any]],
    *,
    min_score: float = MIN_SCORE,
    max_candidates_per_kalshi: int = 5,
    block_size: int = 20,
    prefilter: bool = True,
    keep_soft_rejects: bool = False,
) -> list[Candidate]:
    """Match a full universe of Kalshi markets against Polymarket markets.

    Two-stage pipeline:

    1. **Recall / blocking** (``prefilter=True``, the default): an inverted
       index is built from *discriminative* keys (rare content tokens +
       extracted entities) → Polymarket markets. "Discriminative" means a
       stoplist of connectives / generic market vocabulary and bare years are
       dropped, and any token whose document frequency exceeds
       :data:`_DF_CAP_FRACTION` of the poly corpus is dropped (an idf-style
       cap), so only rare/entity keys are indexed. For each Kalshi market we
       union the postings of its discriminative keys, rank candidates by the
       count of shared discriminative keys, and keep only the top
       ``block_size`` *before* invoking the strict scorer. This makes total
       ``score_match`` calls ``≈ n_kalshi × block_size`` instead of
       ``n_kalshi × n_poly``.

       With ``prefilter=False`` every pair is scored exhaustively (fine for
       small inputs / tests that want the full cross-product).

    2. **Precision / scoring**: each surviving pair is scored through the
       strict ``score_match`` gates. Only non-rejected candidates with
       ``score >= min_score`` are returned.

    Results are sorted best-first within each Kalshi market and capped at
    ``max_candidates_per_kalshi`` per Kalshi market, then the whole list is
    returned sorted best-first overall.

    Args:
        kalshi_items: Kalshi-native market payloads.
        poly_items: Polymarket-native market payloads.
        min_score: Minimum soft score for a kept candidate (default ``0.4``).
        max_candidates_per_kalshi: Cap on accepted matches per Kalshi market.
        block_size: Max poly candidates per Kalshi market that survive blocking
            and reach ``score_match`` (default ``20``). Ignored when
            ``prefilter=False``.
        prefilter: Enable the inverted-index blocking recall stage.
        keep_soft_rejects: When ``True``, **recall-first** mode: candidates that
            cleared every *hard* gate (same-venue / jurisdiction / threshold /
            window) but whose soft score is below ``min_score`` are RETAINED
            (their ``low_score`` reject is a soft reject, tagged but not dropped).
            Only genuine hard-gate rejects are dropped. ``min_score`` then acts
            purely as a floor on which soft candidates survive (callers pass a low
            ``recall_floor`` here). Hard-rejected pairs are always dropped — they
            are cross-venue impossibilities, not low-confidence guesses.

    Returns:
        Candidates sorted best-first. By default only non-rejected pairs with
        ``score >= min_score``; with ``keep_soft_rejects=True`` also the
        soft-rejected (``low_score``) pairs whose score clears ``min_score``.
    """

    if max_candidates_per_kalshi < 0:
        raise ValueError("max_candidates_per_kalshi must be >= 0")
    if block_size < 0:
        raise ValueError("block_size must be >= 0")

    # Build each MarketDesc exactly once and reuse it for both blocking-key
    # extraction and scoring (score_match's extraction is the expensive part).
    poly_descs = [_poly_desc(p) for p in poly_items]
    kalshi_descs = [_kalshi_desc(k) for k in kalshi_items]

    if not prefilter:
        return _match_exhaustive(
            kalshi_items,
            poly_items,
            kalshi_descs,
            poly_descs,
            min_score=min_score,
            max_candidates_per_kalshi=max_candidates_per_kalshi,
            keep_soft_rejects=keep_soft_rejects,
        )

    n_poly = len(poly_items)
    poly_keys = [_blocking_keys(p, d) for p, d in zip(poly_items, poly_descs, strict=True)]

    # Document frequency per key, then drop keys above the idf-style cap so
    # ultra-common keys never index (they'd union back to ~the whole corpus).
    df: Counter[str] = Counter()
    for keys in poly_keys:
        df.update(keys)
    df_cap = _df_cap(n_poly) if n_poly else 0

    index: dict[str, list[int]] = defaultdict(list)
    for idx, keys in enumerate(poly_keys):
        for key in keys:
            if df[key] <= df_cap:
                index[key].append(idx)

    accepted: list[Candidate] = []
    for k_item, k_desc in zip(kalshi_items, kalshi_descs, strict=True):
        k_keys = _blocking_keys(k_item, k_desc)

        # Rank poly candidates by shared discriminative-key count.
        shared: Counter[int] = Counter()
        for key in k_keys:
            if df.get(key, 0) > df_cap:
                continue
            for p_idx in index.get(key, ()):  # type: ignore[arg-type]
                shared[p_idx] += 1

        if not shared:
            continue

        top = [p_idx for p_idx, _ in shared.most_common(block_size)]

        per_kalshi: list[Candidate] = []
        for p_idx in top:
            cand = _score_pair(k_item, poly_items[p_idx], k_desc, poly_descs[p_idx])
            if not _keep_candidate(cand, min_score, keep_soft_rejects):
                continue
            per_kalshi.append(cand)

        per_kalshi.sort(key=lambda c: c.score, reverse=True)
        accepted.extend(per_kalshi[:max_candidates_per_kalshi])

    accepted.sort(key=lambda c: c.score, reverse=True)
    return accepted


def _keep_candidate(cand: Candidate, min_score: float, keep_soft_rejects: bool) -> bool:
    """Decide whether a scored candidate survives the matcher filter.

    Default (precision) mode keeps only non-rejected pairs at/above ``min_score``.
    Recall-first mode (``keep_soft_rejects=True``) additionally keeps *soft*
    rejects — those whose only reject reason is ``low_score`` — as long as the
    score clears ``min_score`` (the recall floor). Hard-gate rejects
    (jurisdiction / threshold / window / same-venue) are always dropped: they are
    cross-venue impossibilities, never low-confidence guesses.
    """

    if cand.score < min_score:
        return False
    if not cand.rejected:
        return True
    return keep_soft_rejects and cand.reject_reason == "low_score"


def _match_exhaustive(
    kalshi_items: list[dict[str, Any]],
    poly_items: list[dict[str, Any]],
    kalshi_descs: list[MarketDesc],
    poly_descs: list[MarketDesc],
    *,
    min_score: float,
    max_candidates_per_kalshi: int,
    keep_soft_rejects: bool = False,
) -> list[Candidate]:
    """Score the full Kalshi×Poly cross-product (``prefilter=False`` path)."""

    accepted: list[Candidate] = []
    for k_item, k_desc in zip(kalshi_items, kalshi_descs, strict=True):
        per_kalshi: list[Candidate] = []
        for p_item, p_desc in zip(poly_items, poly_descs, strict=True):
            cand = _score_pair(k_item, p_item, k_desc, p_desc)
            if not _keep_candidate(cand, min_score, keep_soft_rejects):
                continue
            per_kalshi.append(cand)

        per_kalshi.sort(key=lambda c: c.score, reverse=True)
        accepted.extend(per_kalshi[:max_candidates_per_kalshi])

    accepted.sort(key=lambda c: c.score, reverse=True)
    return accepted


# ---------------------------------------------------------------------------
# Summary (for the discovery UI)
# ---------------------------------------------------------------------------


def summarize(cands: list[Candidate]) -> dict[str, Any]:
    """Summarize a candidate list for the discovery UI.

    Args:
        cands: Candidates (typically the raw, *unfiltered* output of scoring
            every pair, including rejects — so the reject histogram is useful).

    Returns:
        A dict with::

            {
                "total": <int>,
                "by_tier": {"high": n, "borderline": n, "reject": n},
                "reject_reasons": {<reason>: count, ...},
            }
    """

    by_tier: Counter[str] = Counter()
    reject_reasons: Counter[str] = Counter()
    for c in cands:
        by_tier[c.tier] += 1
        if c.rejected and c.reject_reason:
            reject_reasons[c.reject_reason] += 1

    return {
        "total": len(cands),
        "by_tier": {
            TIER_HIGH: by_tier.get(TIER_HIGH, 0),
            TIER_BORDERLINE: by_tier.get(TIER_BORDERLINE, 0),
            TIER_REJECT: by_tier.get(TIER_REJECT, 0),
        },
        "reject_reasons": dict(reject_reasons),
    }

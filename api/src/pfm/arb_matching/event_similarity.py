"""Cross-venue event similarity scorer for prediction-market arb pairs.

The current Polymarket↔Kalshi pair matcher in :mod:`pfm.arb_scanner` keys off
title-keyword overlap alone, which produces false positives whenever two
markets share a topic but resolve on different dates or are about subtly
different events ("Trump wins 2024" vs "Trump wins 2028", "BTC > $80k" vs
"BTC > $90k", "US Senate D majority" vs "Florida State Senate D majority").

This module provides a richer pair scorer combining:

1. **Hard rejects** — disqualifying mismatches that we never want to score,
   regardless of how similar the surface text looks:

   - Non-overlapping resolution windows (date mismatch).
   - Numeric thresholds that differ by > 5 % (e.g. "$80k" vs "$90k").
   - Jurisdictions that conflict (US-Senate vs FL-State-Senate).
   - Same-venue pairs (arb requires *cross*-venue listings).

2. **Soft score** — a weighted blend of:

   - Title token jaccard (weight ``0.30``)
   - Entity jaccard (weight ``0.35``)
   - Topic-taxonomy clue overlap (weight ``0.15``)
   - Resolution-window center distance (weight ``0.20``)

The public surface is :func:`score_match` and :func:`build_market_desc`,
plus the two frozen dataclasses :class:`MarketDesc` and
:class:`SimilarityScore`.

Depends on :mod:`pfm.arb_matching.date_extractor` (T76) for
``ResolutionWindow`` and the ``windows_overlap`` predicate. We import these
lazily inside helpers so that this file remains importable even before T76
lands — but in normal operation the import resolves at module load.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# T76 dependency. The package __init__ already re-exports these symbols,
# so we route through the package to insulate ourselves from internal
# module renames inside arb_matching/.
try:  # pragma: no cover - import contract verified by tests
    from pfm.arb_matching import ResolutionWindow, windows_overlap
except Exception:  # pragma: no cover - extremely defensive fallback
    ResolutionWindow = None  # type: ignore[assignment]

    def windows_overlap(  # type: ignore[no-redef]
        a: ResolutionWindow | None, b: ResolutionWindow | None
    ) -> bool:
        """Fallback stub if T76's helper is not yet importable.

        Returns True when either side is missing (cannot disqualify) and
        otherwise performs a naive [earliest, latest] interval test.
        """

        if a is None or b is None:
            return True
        return not (a.latest < b.earliest or b.latest < a.earliest)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketDesc:
    """Normalized description of a prediction-market listing for scoring."""

    title: str
    description: str
    venue: str  # "polymarket" | "kalshi" | etc.
    resolution_window: Any | None  # ResolutionWindow from T76
    threshold: float | None  # e.g. price threshold "above $80k"
    entities: tuple[str, ...]  # extracted NER tokens (normalized lowercase)
    jurisdiction: str | None  # "US-Senate", "FL-Gov", "Global"
    raw_topic_clues: tuple[str, ...] = field(default_factory=tuple)
    """Optional taxonomy clues attached by ``build_market_desc``.

    These are persisted on the dataclass so that ``score_match`` does not have
    to re-derive them from the (possibly empty) title at score time.
    """


@dataclass(frozen=True)
class SimilarityScore:
    """Composite similarity output for a candidate arb pair."""

    total: float  # 0..1
    components: dict[str, float]
    rejected_reason: str | None  # set when the pair is hard-rejected


# ---------------------------------------------------------------------------
# Hard-reject taxonomy. These are the *reasons* exposed to callers and
# logged into the audit script in T78. Keep stable.
# ---------------------------------------------------------------------------

REJECT_SAME_VENUE = "same_venue"
REJECT_WINDOW_NO_OVERLAP = "resolution_window_no_overlap"
REJECT_THRESHOLD_MISMATCH = "threshold_mismatch"
REJECT_JURISDICTION_MISMATCH = "jurisdiction_mismatch"

REJECT_REASONS = (
    REJECT_SAME_VENUE,
    REJECT_WINDOW_NO_OVERLAP,
    REJECT_THRESHOLD_MISMATCH,
    REJECT_JURISDICTION_MISMATCH,
)

# Threshold tolerance: thresholds within ±5 % (relative) of one another are
# treated as the same target. "above $80k" vs "above $80,500" should not
# be a hard-reject, but "above $80k" vs "above $90k" must be.
_THRESHOLD_REL_TOL = 0.05


# ---------------------------------------------------------------------------
# Topic taxonomy. Small, deliberately coarse-grained. Each clue maps to a
# canonical topic label; the *overlap* of topics across the two market
# descriptions feeds the soft score.
# ---------------------------------------------------------------------------

_TOPIC_TAXONOMY: dict[str, tuple[str, ...]] = {
    "election": (
        "election",
        "elections",
        "president",
        "presidential",
        "senate",
        "house",
        "governor",
        "gop",
        "dem",
        "democrat",
        "democrats",
        "republican",
        "republicans",
        "incumbent",
        "ballot",
        "vote",
        "votes",
        "primary",
        "caucus",
        "nominee",
        "candidate",
        "win",
        "wins",
        "winner",
    ),
    "sports": (
        "nba",
        "nfl",
        "mlb",
        "nhl",
        "soccer",
        "football",
        "basketball",
        "baseball",
        "hockey",
        "tennis",
        "golf",
        "f1",
        "ufc",
        "boxing",
        "olympics",
        "champion",
        "championship",
        "playoffs",
        "world cup",
        "super bowl",
        "world series",
        "stanley cup",
        "finals",
    ),
    "crypto": (
        "btc",
        "bitcoin",
        "eth",
        "ethereum",
        "sol",
        "solana",
        "doge",
        "dogecoin",
        "xrp",
        "ada",
        "cardano",
        "altcoin",
        "stablecoin",
        "halving",
        "etf",
        "spot etf",
        "crypto",
        "blockchain",
    ),
    "macro": (
        "fed",
        "fomc",
        "rate",
        "rates",
        "hike",
        "cut",
        "cpi",
        "inflation",
        "recession",
        "gdp",
        "unemployment",
        "payrolls",
        "yield",
        "treasury",
        "powell",
        "bps",
        "basis points",
        "ecb",
        "boj",
    ),
    "geopolitics": (
        "war",
        "ceasefire",
        "treaty",
        "sanctions",
        "invasion",
        "putin",
        "zelensky",
        "xi",
        "kim",
        "korea",
        "ukraine",
        "russia",
        "israel",
        "gaza",
        "iran",
        "nato",
    ),
    "tech": (
        "ai",
        "openai",
        "anthropic",
        "google",
        "alphabet",
        "meta",
        "apple",
        "nvidia",
        "tsla",
        "tesla",
        "msft",
        "microsoft",
        "amzn",
        "amazon",
        "ipo",
        "merger",
        "acquisition",
    ),
    "weather": (
        "hurricane",
        "storm",
        "cyclone",
        "typhoon",
        "earthquake",
        "wildfire",
        "drought",
        "tornado",
    ),
}


# ---------------------------------------------------------------------------
# Token utilities. Deliberately conservative — punctuation stripped, common
# stopwords removed, lowercased. We do NOT stem, since "win" / "wins" are
# both already in the taxonomy lists and over-stemming hurts entity match.
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "be",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "and",
        "or",
        "but",
        "not",
        "as",
        "from",
        "into",
        "this",
        "that",
        "these",
        "those",
        "will",
        "would",
        "should",
        "can",
        "could",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "it",
        "its",
        "their",
        "there",
        "than",
        "then",
        "any",
        "all",
        "more",
        "less",
        "above",
        "below",
        "over",
        "under",
        "before",
        "after",
        "between",
        "vs",
        "v",
        "via",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")


def _tokenize(text: str) -> list[str]:
    """Tokenize ``text`` into normalized words, dropping stopwords."""

    if not text:
        return []
    out: list[str] = []
    for match in _TOKEN_RE.findall(text.lower()):
        if match in _STOPWORDS:
            continue
        if len(match) <= 1:
            continue
        out.append(match)
    return out


def _jaccard(a: set[str] | tuple[str, ...], b: set[str] | tuple[str, ...]) -> float:
    """Standard set Jaccard, returning ``0.0`` when both sides are empty."""

    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _extract_topic_clues(text: str) -> set[str]:
    """Return the set of taxonomy *labels* (e.g. ``"election"``) hit by ``text``."""

    if not text:
        return set()
    lc = text.lower()
    hits: set[str] = set()
    for topic, vocab in _TOPIC_TAXONOMY.items():
        for term in vocab:
            # multi-word terms need a substring check; single tokens use
            # word boundaries to avoid spurious "ai" inside "rain".
            if " " in term:
                if term in lc:
                    hits.add(topic)
                    break
            # Use a regex with word boundaries for single tokens.
            elif re.search(rf"\b{re.escape(term)}\b", lc):
                hits.add(topic)
                break
    return hits


# ---------------------------------------------------------------------------
# Threshold extraction. We only care about *numeric* thresholds anchored to
# a comparator ("above", "over", "≥", ">", "at least"). Currency symbols and
# k/m/b suffixes are normalised to a single float.
# ---------------------------------------------------------------------------

_THRESHOLD_RE = re.compile(
    r"""
    (?:above|over|>=|>|≥|at\ least|reach|reaches|hit|hits|exceed[s]?|
       cross[es]?|surpass[es]?|below|under|<=|<|≤|less\ than|no\ more\ than)
    \s*
    \$?\s*                                  # optional currency
    (?P<num>\d{1,3}(?:[,]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    (?P<suffix>[kKmMbB])?                   # k/m/b suffix MUST be glued to the number
    \b                                      # require a word break after the number/suffix
    """,
    re.VERBOSE,
)


def _extract_threshold(text: str) -> float | None:
    """Return the first numeric threshold mentioned in ``text``, or None.

    The match is anchored to a comparator word/symbol so that bare numerics
    like dates ("2024", "Nov 5") are NOT treated as thresholds.
    """

    if not text:
        return None
    m = _THRESHOLD_RE.search(text)
    if not m:
        return None
    raw = m.group("num").replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    suffix = (m.group("suffix") or "").lower()
    if suffix == "k":
        value *= 1_000
    elif suffix == "m":
        value *= 1_000_000
    elif suffix == "b":
        value *= 1_000_000_000
    return value


# ---------------------------------------------------------------------------
# Jurisdiction extraction. Very tight whitelist — we'd rather mark
# jurisdiction unknown than guess wrong. Conflicts only fire when *both*
# sides have a non-None jurisdiction and they disagree.
# ---------------------------------------------------------------------------

_US_STATE_NAMES: dict[str, str] = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
}

_OFFICE_TOKENS: dict[str, str] = {
    "senate": "Senate",
    "house": "House",
    "governor": "Gov",
    "gubernatorial": "Gov",
    "president": "Pres",
    "presidential": "Pres",
    "mayor": "Mayor",
    "attorney general": "AG",
    "secretary of state": "SoS",
}


def _extract_jurisdiction(text: str) -> str | None:
    """Heuristic jurisdiction extractor.

    Returns a canonical ``"<region>-<office>"`` token when both region and
    office are confidently present, otherwise just ``"<region>"`` or just
    ``"<office>"``-prefixed-with-US when only one side resolves. Returns
    ``None`` when nothing matches.
    """

    if not text:
        return None
    lc = text.lower()

    # Region resolution: prefer most specific match (state > country > none).
    region: str | None = None
    for name, abbr in _US_STATE_NAMES.items():
        if re.search(rf"\b{re.escape(name)}\b", lc):
            region = abbr
            break
    if region is None:
        # Common federal/US-wide cues
        if re.search(r"\b(us|u\.s\.|united states|america|federal)\b", lc):
            region = "US"

    # Office resolution
    office: str | None = None
    # Multi-word offices need explicit substring search before single tokens
    # to avoid "secretary of state" being shadowed by a bare "state".
    for term, canonical in _OFFICE_TOKENS.items():
        if " " in term:
            if term in lc:
                office = canonical
                break
    if office is None:
        for term, canonical in _OFFICE_TOKENS.items():
            if " " not in term and re.search(rf"\b{re.escape(term)}\b", lc):
                office = canonical
                break

    # "State Senate" / "State House" cue — when the literal phrase appears
    # AND we matched a US state, the jurisdiction is the state legislature,
    # not the US Senate/House. This is the FL-Senate vs US-Senate case
    # called out in the user-flagged false positives.
    if region and region != "US" and office in {"Senate", "House"}:
        if re.search(rf"\bstate\s+{office.lower()}\b", lc):
            return f"{region}-State-{office}"
        # Without an explicit "state X" prefix BUT with a state name AND a
        # legislative office, default to the state legislature anyway —
        # users say "Florida Senate" to mean Tallahassee, not D.C.
        return f"{region}-State-{office}"

    if region and office:
        return f"{region}-{office}"
    if region:
        return region
    if office:
        return f"US-{office}"
    return None


# ---------------------------------------------------------------------------
# Entity extraction. We use a small list of well-known proper-noun tokens
# plus a capitalised-token heuristic. This is intentionally lightweight —
# the goal is to catch "Trump", "BTC", "NVDA", "Fed" reliably, not to
# replicate spaCy.
# ---------------------------------------------------------------------------

_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-zA-Z]+|[A-Z]{2,5})\b")

# Tokens that look like proper nouns but are syntactic / connective noise.
# Excluding these reduces false matches on entity jaccard.
_PROPER_NOUN_STOPWORDS: frozenset[str] = frozenset(
    {
        "I",
        "A",
        "An",
        "The",
        "Will",
        "Should",
        "Would",
        "Could",
        "Can",
        "Be",
        "By",
        "In",
        "On",
        "At",
        "Of",
        "To",
        "And",
        "Or",
        "But",
        "Not",
        "For",
        "From",
        "With",
        "As",
        "Up",
        "Down",
        "Over",
        "Under",
        "Above",
        "Below",
        "Before",
        "After",
        "When",
        "Who",
        "What",
        "Where",
        "Why",
        "How",
        "Yes",
        "No",
        "Eoy",
        "EOY",
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    }
)


def _extract_entities(text: str) -> tuple[str, ...]:
    """Extract a deduplicated tuple of entity tokens from ``text``.

    Entities are lowercase-normalised so that ``"BTC"`` and ``"btc"`` collide.
    Numeric-only tokens are dropped (they are handled by threshold extraction).
    """

    if not text:
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for m in _PROPER_NOUN_RE.findall(text):
        if m in _PROPER_NOUN_STOPWORDS:
            continue
        if m.lower() in _STOPWORDS:
            continue
        norm = m.lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return tuple(out)


# ---------------------------------------------------------------------------
# Window center distance. Returns a similarity score in [0, 1] where 1.0
# means the windows are centered on the same day, 0.0 means they are
# more than 365 days apart.
# ---------------------------------------------------------------------------


def _window_center_similarity(a: Any | None, b: Any | None) -> float:
    """Center-distance similarity between two ResolutionWindow-like objects.

    Returns 0.5 when either side is missing (neutral — we cannot distinguish).
    """

    if a is None or b is None:
        return 0.5

    try:
        a_lo = _as_datetime(a.earliest)
        a_hi = _as_datetime(a.latest)
        b_lo = _as_datetime(b.earliest)
        b_hi = _as_datetime(b.latest)
    except Exception:
        return 0.5

    a_mid = a_lo + (a_hi - a_lo) / 2
    b_mid = b_lo + (b_hi - b_lo) / 2
    delta_days = abs((a_mid - b_mid).total_seconds()) / 86400.0
    # 0d → 1.0, 365d → 0.0, linear in between. Beyond 365d we clamp.
    if delta_days >= 365.0:
        return 0.0
    return 1.0 - (delta_days / 365.0)


def _as_datetime(value: Any) -> datetime:
    """Coerce a date / datetime / iso-string to a tz-aware UTC datetime."""

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        # plain date
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if isinstance(value, str):
        # Try ISO-8601
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except ValueError:
            pass
    raise TypeError(f"cannot coerce {value!r} to datetime")


# ---------------------------------------------------------------------------
# Threshold rejection: numeric thresholds disagree by > 5 % relative.
# When either side has no threshold, we cannot say they disagree → False.
# ---------------------------------------------------------------------------


def _thresholds_disagree(a: float | None, b: float | None) -> bool:
    """True iff both thresholds are present AND differ by more than 5 %."""

    if a is None or b is None:
        return False
    # If both are exactly zero, treat as agreement.
    if a == 0 and b == 0:
        return False
    base = max(abs(a), abs(b))
    if base == 0:
        return False
    return abs(a - b) / base > _THRESHOLD_REL_TOL


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# Weights for the soft-score blend. Sum to 1.0 by construction.
_W_TITLE_TOKENS = 0.30
_W_ENTITIES = 0.35
_W_TOPIC = 0.15
_W_WINDOW = 0.20


def score_match(a: MarketDesc, b: MarketDesc) -> SimilarityScore:
    """Score how likely two market descriptions are about the *same* event.

    Returns a :class:`SimilarityScore`. ``total`` is in ``[0, 1]``. When any
    hard-reject criterion fires, ``total`` is forced to ``0.0`` and
    ``rejected_reason`` is set to one of :data:`REJECT_REASONS`.
    """

    components: dict[str, float] = {
        "title_jaccard": 0.0,
        "entity_jaccard": 0.0,
        "topic_overlap": 0.0,
        "window_center": 0.0,
    }

    # --- Hard rejects ------------------------------------------------------
    # 1. Same venue. Arb requires cross-venue. We check this FIRST so that
    #    audit logs surface the cheapest reason.
    if a.venue and b.venue and a.venue.strip().lower() == b.venue.strip().lower():
        return SimilarityScore(total=0.0, components=components, rejected_reason=REJECT_SAME_VENUE)

    # 2. Jurisdiction conflict.
    if (
        a.jurisdiction is not None
        and b.jurisdiction is not None
        and a.jurisdiction != b.jurisdiction
    ):
        return SimilarityScore(
            total=0.0, components=components, rejected_reason=REJECT_JURISDICTION_MISMATCH
        )

    # 3. Threshold mismatch.
    if _thresholds_disagree(a.threshold, b.threshold):
        return SimilarityScore(
            total=0.0, components=components, rejected_reason=REJECT_THRESHOLD_MISMATCH
        )

    # 4. Resolution-window non-overlap.
    if (
        a.resolution_window is not None
        and b.resolution_window is not None
        and not windows_overlap(a.resolution_window, b.resolution_window)
    ):
        return SimilarityScore(
            total=0.0, components=components, rejected_reason=REJECT_WINDOW_NO_OVERLAP
        )

    # --- Soft score --------------------------------------------------------
    title_tokens_a = set(_tokenize(a.title))
    title_tokens_b = set(_tokenize(b.title))
    components["title_jaccard"] = _jaccard(title_tokens_a, title_tokens_b)

    components["entity_jaccard"] = _jaccard(set(a.entities), set(b.entities))

    topic_a = (
        set(a.raw_topic_clues)
        if a.raw_topic_clues
        else _extract_topic_clues(f"{a.title}\n{a.description}")
    )
    topic_b = (
        set(b.raw_topic_clues)
        if b.raw_topic_clues
        else _extract_topic_clues(f"{b.title}\n{b.description}")
    )
    components["topic_overlap"] = _jaccard(topic_a, topic_b)

    components["window_center"] = _window_center_similarity(
        a.resolution_window, b.resolution_window
    )

    total = (
        _W_TITLE_TOKENS * components["title_jaccard"]
        + _W_ENTITIES * components["entity_jaccard"]
        + _W_TOPIC * components["topic_overlap"]
        + _W_WINDOW * components["window_center"]
    )
    # Clamp to [0, 1] defensively against rounding.
    total = max(0.0, min(1.0, total))
    # Numerical normalisation: if total is NaN for any reason, treat as 0.
    if math.isnan(total):
        total = 0.0

    return SimilarityScore(total=total, components=components, rejected_reason=None)


def build_market_desc(raw_payload: dict, venue: str) -> MarketDesc:
    """Build a :class:`MarketDesc` from a venue-native market payload.

    The payload shapes we accept:

    - Polymarket Gamma: keys ``question``, ``slug``, ``description`` (optional),
      ``endDate`` (ISO string).
    - Kalshi: keys ``title``, ``subtitle`` (optional), ``ticker``,
      ``close_time`` / ``expiration_time``.
    - Arbitrary: keys ``title``, ``description`` — both optional.

    Any field that cannot be resolved falls back to ``""`` / ``None`` /
    empty-tuple so this function is total over reasonable inputs.
    """

    if not isinstance(raw_payload, dict):
        raw_payload = {}
    venue_norm = (venue or "").strip().lower()

    title = raw_payload.get("title") or raw_payload.get("question") or raw_payload.get("name") or ""
    title = str(title).strip()

    description = (
        raw_payload.get("description")
        or raw_payload.get("subtitle")
        or raw_payload.get("rules_primary")
        or ""
    )
    description = str(description).strip()

    combined = f"{title}\n{description}"

    # Resolution window: prefer an explicit ResolutionWindow attached by
    # callers (e.g. the audit script in T78), otherwise try to extract one
    # from the text via T76's helper. If T76 is missing, we degrade to None
    # rather than crashing.
    rw = raw_payload.get("resolution_window")
    if rw is None:
        try:
            from pfm.arb_matching import extract_resolution_window

            rw = extract_resolution_window(combined)
        except Exception:
            rw = None

    threshold = _extract_threshold(combined)
    entities = _extract_entities(combined)
    jurisdiction = _extract_jurisdiction(combined)
    topic_clues = tuple(sorted(_extract_topic_clues(combined)))

    return MarketDesc(
        title=title,
        description=description,
        venue=venue_norm,
        resolution_window=rw,
        threshold=threshold,
        entities=entities,
        jurisdiction=jurisdiction,
        raw_topic_clues=topic_clues,
    )


__all__ = [
    "REJECT_JURISDICTION_MISMATCH",
    "REJECT_REASONS",
    "REJECT_SAME_VENUE",
    "REJECT_THRESHOLD_MISMATCH",
    "REJECT_WINDOW_NO_OVERLAP",
    "MarketDesc",
    "SimilarityScore",
    "build_market_desc",
    "score_match",
]

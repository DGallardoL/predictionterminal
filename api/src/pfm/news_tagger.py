"""Sentiment NER tagger and news -> factor auto-linker.

A lightweight, dependency-free named-entity recognizer designed for
financial and political news headlines. Pure regex + dictionary lookups —
no spaCy / NLTK — because the POC does not justify the install footprint
and the entity universe is small and well-known.

Pipeline
--------
1. ``extract_entities(text)`` — finds tickers, politicians, countries,
   event keywords, commodities. Each entity type is a separate key in the
   returned dict so callers can score them independently.
2. ``score_factor_match(text, factor)`` — hybrid score combining
   keyword overlap, entity overlap and theme match. Output ∈ [0, 1].
3. ``tag_news_to_factors(items, catalog)`` — runs (1) + (2) for each
   ``(item, factor)`` pair, returns the items enriched with their
   matched factors above a configurable threshold.
4. ``enhanced_sentiment(text)`` — wraps :func:`pfm.sentiment_lexicon.score_sentiment`
   and adds aspect-based sentiment per detected entity (the sentence
   containing the entity drives that entity's sub-score).

Routing
-------
This module owns its :class:`fastapi.APIRouter`; ``main.py`` is left
untouched (per CLAUDE.md). Wire-up::

    from pfm.news_tagger import router as news_tagger_router
    app.include_router(news_tagger_router)
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi import Path as PathParam
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache
from pfm.factors import FactorConfig, load_factors
from pfm.sentiment_lexicon import (
    _NEG_SET,
    _POS_SET,
    _tokenize,
    score_sentiment,
)

logger = logging.getLogger(__name__)


CACHE_TTL_SECONDS: int = 600
NAMESPACE: str = "news_tagger"

# Score >= this threshold qualifies as a factor match.
DEFAULT_THRESHOLD: float = 0.3

# Path to the curated entity -> factor map shipped with the package.
_ENTITY_MAP_PATH: Path = Path(__file__).resolve().parent / "data" / "entity_factor_map.json"

# Default factors.yml path (only used when callers don't pass a catalog).
_DEFAULT_FACTORS_FILE: Path = Path(__file__).resolve().parent / "factors.yml"


# ---------------------------------------------------------------------------
# Entity dictionaries (hardcoded; spaCy-free NER)
# ---------------------------------------------------------------------------

POLITICIANS: frozenset[str] = frozenset(
    {
        "trump",
        "harris",
        "biden",
        "vance",
        "obama",
        "clinton",
        "putin",
        "zelensky",
        "zelenskyy",
        "xi",
        "modi",
        "netanyahu",
        "powell",
        "lagarde",
        "yellen",
        "bessent",
        "musk",
        "kennedy",
        "rfk",
        "desantis",
        "newsom",
    }
)

# Multi-word politician names — matched after lower-casing the headline.
POLITICIAN_PHRASES: dict[str, str] = {
    "donald trump": "Trump",
    "kamala harris": "Harris",
    "joe biden": "Biden",
    "jd vance": "Vance",
    "vladimir putin": "Putin",
    "xi jinping": "Xi",
    "jerome powell": "Powell",
    "christine lagarde": "Lagarde",
    "elon musk": "Musk",
    "robert kennedy": "Kennedy",
    "ron desantis": "DeSantis",
}

# Politician canonical-form lookup (lower -> Title display).
_POLITICIAN_DISPLAY: dict[str, str] = {
    "trump": "Trump",
    "harris": "Harris",
    "biden": "Biden",
    "vance": "Vance",
    "obama": "Obama",
    "clinton": "Clinton",
    "putin": "Putin",
    "zelensky": "Zelensky",
    "zelenskyy": "Zelensky",
    "xi": "Xi",
    "modi": "Modi",
    "netanyahu": "Netanyahu",
    "powell": "Powell",
    "lagarde": "Lagarde",
    "yellen": "Yellen",
    "bessent": "Bessent",
    "musk": "Musk",
    "kennedy": "Kennedy",
    "rfk": "Kennedy",
    "desantis": "DeSantis",
    "newsom": "Newsom",
}

COUNTRIES: dict[str, str] = {
    # lower -> Title display (handles common variants/aliases).
    "usa": "USA",
    "us": "USA",
    "america": "USA",
    "united states": "USA",
    "china": "China",
    "russia": "Russia",
    "iran": "Iran",
    "israel": "Israel",
    "ukraine": "Ukraine",
    "taiwan": "Taiwan",
    "japan": "Japan",
    "germany": "Germany",
    "france": "France",
    "uk": "UK",
    "united kingdom": "UK",
    "britain": "UK",
    "india": "India",
    "korea": "Korea",
    "north korea": "NorthKorea",
    "south korea": "SouthKorea",
    "saudi arabia": "SaudiArabia",
    "saudi": "SaudiArabia",
    "venezuela": "Venezuela",
    "mexico": "Mexico",
    "brazil": "Brazil",
    "canada": "Canada",
    "turkey": "Turkey",
    "syria": "Syria",
    "yemen": "Yemen",
    "lebanon": "Lebanon",
}

EVENT_KEYWORDS: dict[str, str] = {
    # lower -> canonical event tag.
    "fomc": "FOMC",
    "cpi": "CPI",
    "ppi": "PPI",
    "nfp": "NFP",
    "payrolls": "NFP",
    "non-farm": "NFP",
    "election": "Election",
    "elections": "Election",
    "primary": "Election",
    "midterm": "Election",
    "midterms": "Election",
    "recession": "Recession",
    "depression": "Recession",
    "war": "War",
    "warfare": "War",
    "invasion": "War",
    "ceasefire": "Ceasefire",
    "truce": "Ceasefire",
    "sanction": "Sanction",
    "sanctions": "Sanction",
    "embargo": "Sanction",
    "tariff": "Tariff",
    "tariffs": "Tariff",
    "earnings": "Earnings",
    "guidance": "Earnings",
    "ipo": "IPO",
    "merger": "Merger",
    "acquisition": "Merger",
    "ipcc": "ClimateReport",
    "summit": "Summit",
    "indictment": "Indictment",
    "impeach": "Impeachment",
    "impeachment": "Impeachment",
    "executive order": "ExecutiveOrder",
    "rate cut": "RateCut",
    "rate-cut": "RateCut",
    "cuts": "RateCut",
    "rate hike": "RateHike",
    "rate-hike": "RateHike",
    "hike": "RateHike",
    "default": "Default",
    "shutdown": "Shutdown",
}

# Multi-word event phrases that benefit from substring lookup.
_EVENT_PHRASES: list[tuple[str, str]] = [
    ("executive order", "ExecutiveOrder"),
    ("rate cut", "RateCut"),
    ("rate hike", "RateHike"),
    ("non-farm", "NFP"),
    ("non farm", "NFP"),
]

COMMODITIES: dict[str, str] = {
    "oil": "Oil",
    "crude": "Oil",
    "wti": "Oil",
    "brent": "Oil",
    "gold": "Gold",
    "silver": "Silver",
    "copper": "Copper",
    "btc": "BTC",
    "bitcoin": "BTC",
    "eth": "ETH",
    "ethereum": "ETH",
    "wheat": "Wheat",
    "corn": "Corn",
    "soy": "Soy",
    "soybeans": "Soy",
    "gas": "NatGas",
    "natural gas": "NatGas",
    "lng": "NatGas",
    "uranium": "Uranium",
    "lithium": "Lithium",
}

# Curated US/EU equity tickers we care about. The regex-only approach
# matches *any* 2-5 letter uppercase token, so we filter against this set
# to avoid tagging e.g. "US", "ETF", "CEO".
KNOWN_TICKERS: frozenset[str] = frozenset(
    {
        # Mega/large cap
        "AAPL",
        "MSFT",
        "GOOGL",
        "GOOG",
        "AMZN",
        "META",
        "TSLA",
        "NVDA",
        "NFLX",
        "ORCL",
        "ADBE",
        "CRM",
        "AMD",
        "INTC",
        "AVGO",
        "QCOM",
        "TSM",
        "IBM",
        "CSCO",
        "TXN",
        "MU",
        # Finance
        "JPM",
        "BAC",
        "WFC",
        "GS",
        "MS",
        "C",
        "BLK",
        "BX",
        "AXP",
        "V",
        "MA",
        "PYPL",
        "SQ",
        "COIN",
        # Energy
        "XOM",
        "CVX",
        "COP",
        "OXY",
        "SLB",
        "EOG",
        "BP",
        "SHEL",
        # Health
        "JNJ",
        "UNH",
        "PFE",
        "MRK",
        "LLY",
        "ABBV",
        "TMO",
        "ABT",
        "BMY",
        # Consumer
        "WMT",
        "COST",
        "HD",
        "MCD",
        "NKE",
        "DIS",
        "SBUX",
        "PG",
        "KO",
        "PEP",
        # Auto/Industrial
        "F",
        "GM",
        "RIVN",
        "LCID",
        "BA",
        "CAT",
        "DE",
        "GE",
        "HON",
        # ETFs & indices
        "SPY",
        "QQQ",
        "DIA",
        "IWM",
        "VTI",
        "VOO",
        "GLD",
        "SLV",
        "USO",
        "TLT",
        "HYG",
        "VIX",
        "VXX",
        "UVXY",
        "TQQQ",
        "SQQQ",
        # Crypto-adjacent
        "MSTR",
        "MARA",
        "RIOT",
        "HUT",
        "BITO",
        "GBTC",
        "IBIT",
        # AI / new tech
        "PLTR",
        "SNOW",
        "CRWD",
        "NET",
        "DDOG",
        "ZS",
        "ANET",
        # Other notable
        "DJT",
        "TRUMP",
        "HOOD",
    }
)

# Tickers that are also common English words — only tag them when they
# appear in $TICKER form or with explicit ticker context.
_AMBIGUOUS_TICKERS: frozenset[str] = frozenset({"F", "C", "V", "T", "GE", "DE", "MA", "GS"})


_TICKER_RE: re.Pattern[str] = re.compile(r"\$?\b([A-Z]{2,5})\b")
# Dollar form allows single-letter tickers (e.g. "$F" for Ford) since the
# leading $ disambiguates from English words / pronouns.
_DOLLAR_TICKER_RE: re.Pattern[str] = re.compile(r"\$([A-Z]{1,5})\b")
_WORD_RE: re.Pattern[str] = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")
_SENTENCE_RE: re.Pattern[str] = re.compile(r"[^.!?\n]+[.!?\n]?")


# ---------------------------------------------------------------------------
# Entity -> factor map (curated JSON, lazy-loaded)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_entity_factor_map() -> dict[str, list[str]]:
    """Read ``data/entity_factor_map.json`` once per process.

    Returns an empty dict (with a warning) if the file is missing or
    malformed; the tagger still works via keyword overlap alone.
    """
    if not _ENTITY_MAP_PATH.exists():
        logger.warning("entity_factor_map.json missing at %s", _ENTITY_MAP_PATH)
        return {}
    try:
        with _ENTITY_MAP_PATH.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("failed to load entity_factor_map.json: %s", e)
        return {}
    out: dict[str, list[str]] = {}
    for entity, slugs in raw.items():
        if entity.startswith("_"):
            continue
        if not isinstance(slugs, list):
            continue
        out[entity] = [str(s).lower() for s in slugs if isinstance(s, str)]
    return out


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


def extract_entities(text: str) -> dict[str, list[str]]:
    """Extract named entities from ``text``.

    Pure regex + dictionary lookups — no spaCy / NLTK. Output keys::

        tickers      : list[str]   uppercase, deduped, in encounter order
        politicians  : list[str]   canonicalized (e.g. "Trump")
        countries    : list[str]   canonicalized (e.g. "USA")
        events       : list[str]   canonicalized event tags
        commodities  : list[str]   canonicalized (e.g. "Oil")

    Empty list for any category with no hits. Order within a list reflects
    first occurrence.
    """
    if not text:
        return {
            "tickers": [],
            "politicians": [],
            "countries": [],
            "events": [],
            "commodities": [],
        }

    lowered = text.lower()

    # Tickers: collect $TICKER first (always tag), then bare uppercase
    # restricted to KNOWN_TICKERS minus ambiguous-only-with-$.
    tickers: list[str] = []
    seen_tickers: set[str] = set()
    for m in _DOLLAR_TICKER_RE.finditer(text):
        tk = m.group(1)
        if tk in KNOWN_TICKERS and tk not in seen_tickers:
            tickers.append(tk)
            seen_tickers.add(tk)
    for m in _TICKER_RE.finditer(text):
        tk = m.group(1)
        if tk not in KNOWN_TICKERS:
            continue
        if tk in _AMBIGUOUS_TICKERS:
            continue
        if tk in seen_tickers:
            continue
        tickers.append(tk)
        seen_tickers.add(tk)

    # Politicians: phrases first (so "Donald Trump" → Trump once), then
    # single-token lookup.
    politicians: list[str] = []
    seen_pol: set[str] = set()
    for phrase, display in POLITICIAN_PHRASES.items():
        if phrase in lowered and display not in seen_pol:
            politicians.append(display)
            seen_pol.add(display)
    for tok in _WORD_RE.findall(lowered):
        if tok in POLITICIANS:
            display = _POLITICIAN_DISPLAY.get(tok, tok.title())
            if display not in seen_pol:
                politicians.append(display)
                seen_pol.add(display)

    # Countries: phrase scan first (multi-word), then single-token.
    countries: list[str] = []
    seen_country: set[str] = set()
    for phrase, display in COUNTRIES.items():
        if " " not in phrase:
            continue
        if phrase in lowered and display not in seen_country:
            countries.append(display)
            seen_country.add(display)
    for tok in _WORD_RE.findall(lowered):
        if tok in COUNTRIES and " " not in tok:
            display = COUNTRIES[tok]
            if display not in seen_country:
                countries.append(display)
                seen_country.add(display)

    # Events: phrases first, then single-token.
    events: list[str] = []
    seen_event: set[str] = set()
    for phrase, tag in _EVENT_PHRASES:
        if phrase in lowered and tag not in seen_event:
            events.append(tag)
            seen_event.add(tag)
    for tok in _WORD_RE.findall(lowered):
        if tok in EVENT_KEYWORDS:
            tag = EVENT_KEYWORDS[tok]
            if tag not in seen_event:
                events.append(tag)
                seen_event.add(tag)

    # Commodities: phrases first ("natural gas"), then single-token.
    commodities: list[str] = []
    seen_comm: set[str] = set()
    for phrase, tag in COMMODITIES.items():
        if " " not in phrase:
            continue
        if phrase in lowered and tag not in seen_comm:
            commodities.append(tag)
            seen_comm.add(tag)
    for tok in _WORD_RE.findall(lowered):
        if tok in COMMODITIES and " " not in tok:
            tag = COMMODITIES[tok]
            if tag not in seen_comm:
                commodities.append(tag)
                seen_comm.add(tag)

    return {
        "tickers": tickers,
        "politicians": politicians,
        "countries": countries,
        "events": events,
        "commodities": commodities,
    }


def all_entities(extracted: dict[str, list[str]]) -> list[str]:
    """Flatten an :func:`extract_entities` output to a single list."""
    out: list[str] = []
    for key in ("tickers", "politicians", "countries", "events", "commodities"):
        out.extend(extracted.get(key, []))
    return out


# ---------------------------------------------------------------------------
# Factor scoring
# ---------------------------------------------------------------------------


_TOKEN_SPLIT_RE: re.Pattern[str] = re.compile(r"[^A-Za-z0-9]+")
_FACTOR_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "or",
        "of",
        "in",
        "on",
        "for",
        "to",
        "by",
        "with",
        "is",
        "are",
        "be",
        "as",
        "an",
        "a",
        "at",
        "from",
        "that",
        "will",
        "would",
        "this",
        "these",
        "those",
        "than",
        "ever",
        "any",
        "all",
        "no",
        "not",
        "yes",
    }
)


def _factor_tokens(factor: dict[str, Any]) -> set[str]:
    """Tokenize a factor dict's id / name / theme / keywords into a set.

    Lowercases, drops tokens shorter than 3 chars, drops stop-words.
    """
    parts: list[str] = []
    for key in ("id", "name", "slug", "theme", "description"):
        val = factor.get(key)
        if isinstance(val, str):
            parts.append(val)
    kws = factor.get("keywords")
    if isinstance(kws, list):
        parts.extend(str(k) for k in kws if isinstance(k, str))
    blob = " ".join(parts).lower()
    raw = _TOKEN_SPLIT_RE.split(blob)
    return {t for t in raw if len(t) >= 3 and t not in _FACTOR_STOP_WORDS}


def _entity_map_hits(entities: list[str], factor_id: str, factor_slug: str) -> int:
    """Count how many entities map (via curated JSON) to this factor.

    Each hit means: the curated map says "headlines mentioning entity E
    are likely about factors whose id/slug contains substring S", and the
    given factor's id or slug actually contains S.
    """
    if not entities:
        return 0
    emap = load_entity_factor_map()
    haystack = f"{factor_id} {factor_slug}".lower()
    hits = 0
    for ent in entities:
        for sub in emap.get(ent, []):
            if sub and sub in haystack:
                hits += 1
                break
    return hits


def score_factor_match(news_text: str, factor: dict[str, Any]) -> float:
    """Score how strongly ``news_text`` is "about" ``factor``.

    Returns a float in [0, 1]. Three signals are combined:

    1. **Token overlap** between news tokens and factor tokens
       (id/name/theme/keywords). Jaccard-style with a soft denominator.
    2. **Entity overlap** — entities found in the news that map (via
       :func:`load_entity_factor_map`) to the factor's id/slug.
    3. **Theme match** — boost if any extracted entity category aligns
       with the factor's ``theme`` (e.g. "Putin" + theme="geopolitics").

    The weights are deliberately blunt — this is a POC, not learned-to-rank.
    """
    if not news_text:
        return 0.0
    factor_id = str(factor.get("id", ""))
    factor_slug = str(factor.get("slug", factor_id))
    theme = str(factor.get("theme", "")).lower()

    # Tokens
    news_tokens = {
        t
        for t in _TOKEN_SPLIT_RE.split(news_text.lower())
        if len(t) >= 3 and t not in _FACTOR_STOP_WORDS
    }
    if not news_tokens:
        return 0.0
    factor_tokens = _factor_tokens(factor)
    if not factor_tokens:
        # Fall back to using just the slug as keywords.
        factor_tokens = {
            t
            for t in _TOKEN_SPLIT_RE.split(factor_slug.lower())
            if len(t) >= 3 and t not in _FACTOR_STOP_WORDS
        }
    overlap = news_tokens & factor_tokens
    # Soft Jaccard: |A∩B| / sqrt(|A|*|B|) — penalises tiny factor lex less
    # than full Jaccard would.
    denom = max(1.0, (len(news_tokens) * len(factor_tokens)) ** 0.5)
    token_score = min(1.0, len(overlap) / denom * 2.0)

    # Entity score
    entities = all_entities(extract_entities(news_text))
    entity_hits = _entity_map_hits(entities, factor_id, factor_slug)
    entity_score = min(1.0, entity_hits / 2.0) if entity_hits else 0.0

    # Theme bonus: e.g. theme=macro and we see a Fed/CPI event.
    extracted = extract_entities(news_text)
    theme_bonus = 0.0
    if theme:
        macro_evt = {"FOMC", "CPI", "PPI", "NFP", "RateCut", "RateHike", "Recession"}
        geo_evt = {"War", "Sanction", "Ceasefire", "Tariff", "Election"}
        if (
            (
                theme == "macro"
                and (
                    set(extracted["events"]) & macro_evt
                    or any(p in {"Powell", "Lagarde", "Yellen"} for p in extracted["politicians"])
                )
            )
            or (
                theme == "geopolitics"
                and (set(extracted["events"]) & geo_evt or extracted["countries"])
            )
            or (theme == "crypto" and any(c in {"BTC", "ETH"} for c in extracted["commodities"]))
            or (theme == "equities" and extracted["tickers"])
        ):
            theme_bonus = 0.15

    # Weighted combine. Token overlap is the dominant signal; entities and
    # theme are tie-breakers / disambiguators.
    score = 0.6 * token_score + 0.3 * entity_score + theme_bonus
    return float(min(1.0, max(0.0, score)))


# ---------------------------------------------------------------------------
# Tag a list of news against a factor catalog
# ---------------------------------------------------------------------------


def _factor_to_dict(fc: FactorConfig) -> dict[str, Any]:
    """Convert a :class:`FactorConfig` to a plain dict (for the public API)."""
    return {
        "id": fc.id,
        "name": fc.name,
        "slug": fc.slug,
        "theme": fc.theme,
        "description": fc.description,
    }


@lru_cache(maxsize=1)
def _default_catalog() -> list[dict[str, Any]]:
    """Load and cache the catalog from the project's ``factors.yml``.

    Returns an empty list (with a warning) on any error so callers don't
    crash if the file disappears in tests.
    """
    if not _DEFAULT_FACTORS_FILE.exists():
        logger.warning("default factors.yml missing at %s", _DEFAULT_FACTORS_FILE)
        return []
    try:
        configs = load_factors(_DEFAULT_FACTORS_FILE)
    except (OSError, ValueError) as e:
        logger.warning("failed to load default factors.yml: %s", e)
        return []
    return [_factor_to_dict(fc) for fc in configs.values()]


def tag_news_to_factors(
    news_items: list[dict[str, Any]],
    factor_catalog: list[dict[str, Any]] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict[str, Any]]:
    """For each ``news_items`` entry, return matched factors above ``threshold``.

    Args:
        news_items: List of dict-shaped news items. Each item is read for
            ``title`` and ``description`` (concatenated for scoring).
        factor_catalog: List of factor dicts (id, name, slug, theme,
            description, optional keywords). When ``None``, the project's
            ``factors.yml`` is used.
        threshold: Minimum score to qualify as a match. Default 0.3.

    Returns:
        List of dicts shaped::

            {"news_item": <item>,
             "matched_factors": [
                {"factor_id": ..., "factor_name": ..., "match_score": 0.42},
                ...
             ]}

        ``matched_factors`` is sorted by descending score.
    """
    catalog = factor_catalog if factor_catalog is not None else _default_catalog()
    out: list[dict[str, Any]] = []
    for item in news_items:
        text = " ".join(str(item.get(k, "")) for k in ("title", "description") if item.get(k))
        if not text:
            out.append({"news_item": item, "matched_factors": []})
            continue
        scored: list[dict[str, Any]] = []
        for fac in catalog:
            score = score_factor_match(text, fac)
            if score >= threshold:
                scored.append(
                    {
                        "factor_id": fac.get("id", ""),
                        "factor_name": fac.get("name", fac.get("id", "")),
                        "match_score": round(score, 4),
                    }
                )
        scored.sort(key=lambda d: d["match_score"], reverse=True)
        out.append({"news_item": item, "matched_factors": scored})
    return out


# ---------------------------------------------------------------------------
# Enhanced sentiment (overall + per entity)
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """Split on ., !, ?, newlines. Keeps non-empty fragments only."""
    parts = [p.strip() for p in _SENTENCE_RE.findall(text)]
    return [p for p in parts if p]


def _entity_in_sentence(entity: str, sentence: str) -> bool:
    return entity.lower() in sentence.lower()


def enhanced_sentiment(text: str) -> dict[str, Any]:
    """Sentiment + per-entity (aspect-based) sentiment.

    Reuses :func:`pfm.sentiment_lexicon.score_sentiment` for the overall
    score. For each detected entity, we average the overall sentiment of
    the sentences containing that entity to get a sub-score. If the entity
    appears in no sentence with polarity, sub-score is 0 (neutral).

    Returns::

        {
          "overall_sentiment": float in [-1, 1],
          "dominant": "positive" | "negative" | "neutral",
          "confidence": float in [0, 1],
          "sentiment_per_entity": {"Trump": 0.42, "China": -0.31, ...},
        }
    """
    overall = score_sentiment(text or "")
    entities = all_entities(extract_entities(text or ""))
    sentences = _split_sentences(text or "")

    per_entity: dict[str, float] = {}
    for ent in entities:
        scores: list[float] = []
        for sent in sentences:
            if _entity_in_sentence(ent, sent):
                scores.append(float(score_sentiment(sent)["score"]))
        if scores:
            per_entity[ent] = round(sum(scores) / len(scores), 4)
        else:
            # Polarity tokens but entity occurs only in punctuation-free
            # one-liner: fall back to overall.
            per_entity[ent] = float(overall["score"])

    return {
        "overall_sentiment": float(overall["score"]),
        "dominant": str(overall["dominant"]),
        "confidence": float(overall["confidence"]),
        "sentiment_per_entity": per_entity,
        # Cheap polarity-density signal callers might want for ranking.
        "n_positive_tokens": int(overall["n_positive"]),
        "n_negative_tokens": int(overall["n_negative"]),
    }


# Helper used by tests / external callers that want to know whether a
# given token is recognised by the sentiment lexicon.
def is_polarity_token(token: str) -> bool:
    return token.lower() in _POS_SET or token.lower() in _NEG_SET


def _has_any_polarity(text: str) -> bool:
    return any(is_polarity_token(t) for t in _tokenize(text))


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


Sentiment = Literal["positive", "negative", "neutral"]


class TagRequest(BaseModel):
    news_text: str = Field(..., min_length=1, max_length=5000)
    factor_ids: list[str] | None = Field(
        None,
        description=(
            "Optional restriction to a subset of factor ids. When None, "
            "the full factors.yml catalog is used."
        ),
    )
    threshold: float = Field(DEFAULT_THRESHOLD, ge=0.0, le=1.0)


class MatchedFactor(BaseModel):
    factor_id: str
    factor_name: str
    match_score: float


class EntityBundle(BaseModel):
    tickers: list[str]
    politicians: list[str]
    countries: list[str]
    events: list[str]
    commodities: list[str]


class TagResponse(BaseModel):
    news_text: str
    entities: EntityBundle
    matched_factors: list[MatchedFactor]
    sentiment: dict[str, Any]


class TagBatchItem(BaseModel):
    title: str = Field(..., min_length=1)
    description: str = Field("")
    ts: str = Field("")
    url: str = Field("")
    source: str = Field("")


class TagBatchRequest(BaseModel):
    news_items: list[TagBatchItem] = Field(..., min_length=1, max_length=200)
    factor_ids: list[str] | None = None
    threshold: float = Field(DEFAULT_THRESHOLD, ge=0.0, le=1.0)


class TagBatchResponseItem(BaseModel):
    news_item: TagBatchItem
    entities: EntityBundle
    matched_factors: list[MatchedFactor]
    sentiment: dict[str, Any]


class TagBatchResponse(BaseModel):
    n_items: int
    n_with_matches: int
    results: list[TagBatchResponseItem]


class FactorRecentResponse(BaseModel):
    factor_id: str
    hours: int
    n_returned: int
    items: list[dict[str, Any]]


class EntityFactorsResponse(BaseModel):
    entity: str
    n_returned: int
    factors: list[MatchedFactor]


# ---------------------------------------------------------------------------
# Catalog filtering helper
# ---------------------------------------------------------------------------


def _filter_catalog(
    factor_ids: list[str] | None,
    base: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    cat = base if base is not None else _default_catalog()
    if not factor_ids:
        return cat
    wanted = set(factor_ids)
    return [f for f in cat if f.get("id") in wanted]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(prefix="/news", tags=["news-tagger"])


@router.post(
    "/tag",
    response_model=TagResponse,
    summary="Tag a single news headline -> entities + matched factors + sentiment.",
)
def post_tag(body: Annotated[TagRequest, Body()]) -> TagResponse:
    cache = get_cache(NAMESPACE, ttl=CACHE_TTL_SECONDS)
    cache_key = ("tag", body.news_text, tuple(body.factor_ids or ()), body.threshold)
    cached = cache.get(cache_key)
    if cached is not None:
        return TagResponse(**cached)

    catalog = _filter_catalog(body.factor_ids)
    tagged = tag_news_to_factors([{"title": body.news_text}], catalog, threshold=body.threshold)
    matched = tagged[0]["matched_factors"] if tagged else []
    ents = extract_entities(body.news_text)
    sentiment = enhanced_sentiment(body.news_text)

    payload = TagResponse(
        news_text=body.news_text,
        entities=EntityBundle(**ents),
        matched_factors=[MatchedFactor(**m) for m in matched],
        sentiment=sentiment,
    )
    cache.set(cache_key, payload.model_dump())
    return payload


@router.post(
    "/tag-batch",
    response_model=TagBatchResponse,
    summary="Bulk-tag a list of news items.",
)
def post_tag_batch(body: Annotated[TagBatchRequest, Body()]) -> TagBatchResponse:
    cache = get_cache(NAMESPACE, ttl=CACHE_TTL_SECONDS)
    cache_key = (
        "tag-batch",
        tuple((it.title, it.ts) for it in body.news_items),
        tuple(body.factor_ids or ()),
        body.threshold,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return TagBatchResponse(**cached)

    catalog = _filter_catalog(body.factor_ids)
    items_as_dicts = [it.model_dump() for it in body.news_items]
    tagged = tag_news_to_factors(items_as_dicts, catalog, threshold=body.threshold)

    results: list[TagBatchResponseItem] = []
    n_with_matches = 0
    for raw_item, t in zip(body.news_items, tagged, strict=False):
        text = f"{raw_item.title} {raw_item.description}".strip()
        ents = extract_entities(text)
        sentiment = enhanced_sentiment(text)
        matched = [MatchedFactor(**m) for m in t["matched_factors"]]
        if matched:
            n_with_matches += 1
        results.append(
            TagBatchResponseItem(
                news_item=raw_item,
                entities=EntityBundle(**ents),
                matched_factors=matched,
                sentiment=sentiment,
            )
        )

    payload = TagBatchResponse(
        n_items=len(results),
        n_with_matches=n_with_matches,
        results=results,
    )
    cache.set(cache_key, payload.model_dump())
    return payload


# In-memory store of recently tagged items, populated by callers (e.g.
# the GDELT/RSS routers via :func:`record_tagged_items`). The endpoint
# below is the *read* side; population is a soft contract — when no items
# have been recorded the endpoint returns an empty list rather than 404,
# because the typical Terminal flow is "open the panel, see what we have,
# subscribe for fresher data".
_RECENT_BY_FACTOR: dict[str, list[dict[str, Any]]] = {}
_RECENT_MAX_PER_FACTOR: int = 200


def record_tagged_items(items_with_matches: list[dict[str, Any]]) -> int:
    """Append tagged items to the per-factor in-memory ring.

    ``items_with_matches`` follows the shape returned by
    :func:`tag_news_to_factors`. Items with at least one matched factor
    are pushed into ``_RECENT_BY_FACTOR[factor_id]`` (capped at
    :data:`_RECENT_MAX_PER_FACTOR`). Returns the number of (item, factor)
    pairs recorded. Idempotent on (url, factor_id).
    """
    n_recorded = 0
    for entry in items_with_matches:
        item = entry.get("news_item")
        if not isinstance(item, dict):
            continue
        for mf in entry.get("matched_factors", []):
            fid = mf.get("factor_id")
            if not fid:
                continue
            bucket = _RECENT_BY_FACTOR.setdefault(fid, [])
            url = item.get("url", "")
            if url and any(b.get("url") == url for b in bucket):
                continue
            bucket.append(
                {
                    **item,
                    "match_score": mf.get("match_score", 0.0),
                }
            )
            n_recorded += 1
            if len(bucket) > _RECENT_MAX_PER_FACTOR:
                del bucket[0 : len(bucket) - _RECENT_MAX_PER_FACTOR]
    return n_recorded


def clear_recent_tagged() -> None:
    """Test helper — drop every recorded recent item."""
    _RECENT_BY_FACTOR.clear()


@router.get(
    "/factor/{factor_id}/recent",
    response_model=FactorRecentResponse,
    summary="Recently tagged news items for a factor.",
)
def get_factor_recent(
    factor_id: Annotated[str, PathParam(min_length=1, max_length=200)],
    hours: Annotated[int, Query(ge=1, le=24 * 14)] = 24,
    n: Annotated[int, Query(ge=1, le=100)] = 20,
) -> FactorRecentResponse:
    cache = get_cache(NAMESPACE, ttl=CACHE_TTL_SECONDS)
    cache_key = ("factor-recent", factor_id, hours, n)
    cached = cache.get(cache_key)
    if cached is not None:
        return FactorRecentResponse(**cached)

    bucket = list(_RECENT_BY_FACTOR.get(factor_id, []))
    # Sort newest first when ts is present; items without ts go last.
    bucket.sort(key=lambda d: d.get("ts", ""), reverse=True)
    items = bucket[: int(n)]
    payload = FactorRecentResponse(
        factor_id=factor_id, hours=int(hours), n_returned=len(items), items=items
    )
    cache.set(cache_key, payload.model_dump())
    return payload


@router.get(
    "/entity/{entity}/factors",
    response_model=EntityFactorsResponse,
    summary="Top factors associated with a named entity.",
)
def get_entity_factors(
    entity: Annotated[str, PathParam(min_length=1, max_length=80)],
    n: Annotated[int, Query(ge=1, le=100)] = 10,
) -> EntityFactorsResponse:
    """Return the top-N factors most associated with ``entity``.

    Uses the curated entity-factor map plus a keyword score against the
    catalog, so even un-curated entities still produce results.
    """
    cache = get_cache(NAMESPACE, ttl=CACHE_TTL_SECONDS)
    cache_key = ("entity-factors", entity.lower(), n)
    cached = cache.get(cache_key)
    if cached is not None:
        return EntityFactorsResponse(**cached)

    catalog = _default_catalog()
    if not catalog:
        raise HTTPException(status_code=503, detail="factor catalog unavailable")

    # Build a synthetic news string so the same scorer applies.
    synthetic = f"{entity} news headline"
    matched: list[MatchedFactor] = []
    for fac in catalog:
        s = score_factor_match(synthetic, fac)
        # Boost when the curated map associates this entity with the factor.
        emap = load_entity_factor_map()
        haystack = f"{fac.get('id', '')} {fac.get('slug', '')}".lower()
        for sub in emap.get(entity, []):
            if sub and sub in haystack:
                s = min(1.0, s + 0.4)
                break
        if s > 0.0:
            matched.append(
                MatchedFactor(
                    factor_id=str(fac.get("id", "")),
                    factor_name=str(fac.get("name", fac.get("id", ""))),
                    match_score=round(s, 4),
                )
            )
    matched.sort(key=lambda m: m.match_score, reverse=True)
    matched = matched[: int(n)]

    payload = EntityFactorsResponse(entity=entity, n_returned=len(matched), factors=matched)
    cache.set(cache_key, payload.model_dump())
    return payload


__all__ = [
    "CACHE_TTL_SECONDS",
    "DEFAULT_THRESHOLD",
    "EntityBundle",
    "FactorRecentResponse",
    "MatchedFactor",
    "TagBatchRequest",
    "TagBatchResponse",
    "TagBatchResponseItem",
    "TagRequest",
    "TagResponse",
    "all_entities",
    "clear_recent_tagged",
    "enhanced_sentiment",
    "extract_entities",
    "load_entity_factor_map",
    "record_tagged_items",
    "router",
    "score_factor_match",
    "tag_news_to_factors",
]

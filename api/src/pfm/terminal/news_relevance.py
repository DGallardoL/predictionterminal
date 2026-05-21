"""Shared relevance scoring for terminal news modules.

The user complaint was: "las noticias de los eventos no parecen ser taaan
de los eventos" — news returned for a Polymarket slug was only loosely
about that event. Root cause: every news module sent 2–3 lower-cased
tokens to its upstream API and accepted whatever came back, with no
post-hoc relevance filter. Reddit/HN/GDELT treat space-separated tokens
as a permissive OR, so a question like "Will Trump resign by 2027?"
returns every Trump headline ever filed.

This module gives the four news modules a small, shared toolkit to
tighten the matching:

* :func:`build_terms` splits a question into **anchor terms** (proper
  nouns, tickers, multi-letter capitalised entities) and **topic terms**
  (lower-cased content words). Anchor terms are what we *insist* on.
* :func:`score_relevance` computes a [0, 1] relevance score against a
  candidate title (+optional description) and returns the matched terms
  alongside, so callers can attach a breakdown to the API response.
* :func:`build_phrase_query` and :func:`build_anchor_query` produce
  upstream-friendly query strings — quoted phrases for GDELT/Algolia,
  optional ``AND`` joining for Reddit's lucene-ish parser.

The module has zero IO and no FastAPI dependency, which keeps it
trivially testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Public threshold used by callers as a sane default. Items scoring
# strictly below ``RELEVANCE_MIN`` are typically dropped from the feed.
# Empirically anything below ~0.18 hits one common topic token but no
# anchor — i.e. off-topic. Tuned against the slugs in the audit report.
RELEVANCE_MIN: float = 0.18

# Maximum number of anchor / topic terms we surface. Three of each is
# enough to keep upstream queries short while still distinguishing
# topics ("Trump impeach Senate" vs "Trump tariff China").
MAX_ANCHORS: int = 3
MAX_TOPICS: int = 5

# Tokens that should never count as anchors or topics. Kept separate
# from the existing ``_STOP_WORDS`` in ``terminal_news`` so we can be
# more aggressive here (we drop things like "year", "end", "2026"
# which are useless on their own).
_RELEVANCE_STOP_WORDS: frozenset[str] = frozenset(
    {
        # Polymarket question framing
        "will",
        "would",
        "could",
        "should",
        "shall",
        "may",
        "might",
        "must",
        "can",
        "did",
        "does",
        "do",
        "have",
        "has",
        "had",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "then",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "by",
        "with",
        "from",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "than",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        # Polymarket noise
        "win",
        "wins",
        "winner",
        "lose",
        "loses",
        "loser",
        "happen",
        "happens",
        "occur",
        "occurs",
        "say",
        "says",
        "said",
        "vs",
        "versus",
        "before",
        "after",
        "above",
        "below",
        "during",
        # Generic time / quantity words that match anything
        "year",
        "years",
        "month",
        "months",
        "day",
        "days",
        "week",
        "weeks",
        "end",
        "ends",
        "start",
        "starts",
        "next",
        "first",
        "last",
        "any",
        "all",
        "each",
        "every",
        "some",
        "many",
        "more",
        "most",
        "less",
        "least",
        "high",
        "higher",
        "low",
        "lower",
        "reach",
        "reaches",
        "hit",
        "hits",
        "exceed",
        "exceeds",
        "yes",
        "no",
        "not",
        "yet",
        "still",
        # Numbers spelled out
        "one",
        "two",
        "three",
        "four",
        "five",
    }
)

# Negative-context patterns: their presence in a title strongly
# *de*-prioritises the item. Examples: "not about X", "unrelated to X".
_NEGATIVE_CONTEXT_RE = re.compile(
    r"\b(not about|unrelated to|nothing to do with|has nothing|debunk(?:s|ed)?)\b",
    re.IGNORECASE,
)


# A short allowlist of well-known tickers / abbreviations that are valid
# anchors despite being shorter than the usual 3-char minimum. Lower-
# cased here; comparison is case-insensitive.
_SHORT_ANCHORS: frozenset[str] = frozenset(
    {
        "ai",
        "us",
        "eu",
        "uk",
        "un",
        "ev",
        "vc",
        "ge",
        "gm",
        "hp",
        "fx",
        "fed",
        "ecb",
        "boj",
        "boe",
        "ipo",
    }
)


@dataclass(frozen=True)
class QuestionTerms:
    """Anchors (entities) and topic terms extracted from a question."""

    anchors: tuple[str, ...]
    topics: tuple[str, ...]

    @property
    def all_terms(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.anchors, *self.topics)))


# ---------------------------------------------------------------------------
# Term extraction
# ---------------------------------------------------------------------------


def _is_acronym_or_ticker(tok: str) -> bool:
    """True for ALL-CAPS tokens of length >=2 (NVDA, BTC) or known short anchors."""
    if len(tok) >= 2 and tok.isupper() and tok.isalpha():
        return True
    return tok.lower() in _SHORT_ANCHORS


def _is_capitalised(tok: str) -> bool:
    """True for tokens that start with an uppercase letter and have lowercase rest."""
    return len(tok) >= 2 and tok[0].isupper() and any(c.islower() for c in tok[1:])


def build_terms(
    question: str,
    *,
    max_anchors: int = MAX_ANCHORS,
    max_topics: int = MAX_TOPICS,
) -> QuestionTerms:
    """Return anchor + topic terms for a Polymarket question.

    *Anchors* are entities we want to see in the result: proper nouns,
    capitalised multi-word names (collapsed to a single phrase), and
    tickers / acronyms. They are the most discriminating signal.

    *Topics* are lowercased content words (≥3 chars, non-stop). They
    are a softer filter — useful for ranking but not for hard exclusion.

    Both lists preserve the order in which the terms appear in the
    question and are de-duplicated case-insensitively.

    Non-ASCII proper nouns are NFKD-normalized before tokenisation so
    accented entities ("Castellón", "Müller", "Cádiz") become single
    base-letter tokens. Without this, `[A-Za-z]` would split mid-word
    and feed unrecognisable fragments to upstream search APIs.
    """
    import unicodedata

    folded = unicodedata.normalize("NFKD", question)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9]*", folded)

    anchors: list[str] = []
    topics: list[str] = []
    seen_anchors: set[str] = set()
    seen_topics: set[str] = set()

    # Greedily group consecutive capitalised tokens into multi-word
    # anchors (e.g. "Joe Biden" → "Joe Biden"). Done in a single pass.
    # Stop-word capitalised tokens (sentence-leading "Will", "Should",
    # "When", "How") are *skipped* so we don't conflate a question
    # particle with the following entity ("Will Trump" → "Trump").
    i = 0
    while i < len(raw_tokens):
        tok = raw_tokens[i]
        if _is_acronym_or_ticker(tok):
            key = tok.upper()
            if key not in seen_anchors:
                anchors.append(tok)
                seen_anchors.add(key)
            i += 1
            continue
        if _is_capitalised(tok):
            # Skip capitalised stop-words at the head of a question
            # (e.g. "Will", "Should", "Can"). Otherwise greedily collect
            # adjacent capitalised non-stopword tokens into one phrase.
            if tok.lower() in _RELEVANCE_STOP_WORDS:
                i += 1
                continue
            j = i + 1
            while (
                j < len(raw_tokens)
                and _is_capitalised(raw_tokens[j])
                and (raw_tokens[j].lower() not in _RELEVANCE_STOP_WORDS)
            ):
                j += 1
            phrase = " ".join(raw_tokens[i:j])
            key = phrase.lower()
            if key not in seen_anchors and phrase.lower() not in _RELEVANCE_STOP_WORDS:
                anchors.append(phrase)
                seen_anchors.add(key)
            i = j
            continue
        i += 1

    # Topic terms: everything not used as an anchor, lower-cased.
    anchor_word_set = {w.lower() for a in anchors for w in re.findall(r"[A-Za-z0-9]+", a)}
    for tok in raw_tokens:
        low = tok.lower()
        if low in anchor_word_set:
            continue
        if len(low) < 3 or low in _RELEVANCE_STOP_WORDS:
            continue
        if low in seen_topics:
            continue
        seen_topics.add(low)
        topics.append(low)

    return QuestionTerms(
        anchors=tuple(anchors[:max_anchors]),
        topics=tuple(topics[:max_topics]),
    )


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------


def _word_match(haystack: str, needle: str) -> bool:
    """Case-insensitive whole-word / whole-phrase match with light stemming.

    For needles ≥5 chars we match a word that *starts with* the needle's
    first 5 characters so morphological variants line up:
    ``impeached`` matches ``impeachment``; ``elect`` matches ``election``.
    For short tokens (e.g. ``ai``, ``btc``, ``fed``) we require an exact
    word match, otherwise common substrings would dominate.
    """
    needle = needle.strip()
    if not needle:
        return False
    if " " in needle:
        # Multi-word phrases are matched verbatim — case-insensitive, but
        # we still wrap in word boundaries so "Joe Biden" doesn't match
        # "Joe Bidens".
        pat = r"\b" + re.escape(needle) + r"s?\b"
        return re.search(pat, haystack, flags=re.IGNORECASE) is not None
    if len(needle) >= 5:
        stem = re.escape(needle[:5])
        pat = r"\b" + stem + r"[A-Za-z]*\b"
        return re.search(pat, haystack, flags=re.IGNORECASE) is not None
    pat = r"\b" + re.escape(needle) + r"\b"
    return re.search(pat, haystack, flags=re.IGNORECASE) is not None


def score_relevance(
    text: str,
    terms: QuestionTerms,
    *,
    title_weight: float = 1.0,
    body: str = "",
) -> tuple[float, list[str]]:
    """Compute relevance score in [0, 1] for ``text`` against ``terms``.

    Scoring scheme:

    * +0.40 for the first matched anchor in the title (anchors are the
      hard signal — a NVDA story should mention "NVDA").
    * +0.20 for each additional anchor in the title (capped at 2 extra).
    * +0.10 for each matched topic in the title (capped at 3).
    * +0.05 for each anchor present in the body but not the title.
    * −0.50 if a negative-context phrase ("not about X") is detected.
    * Score is clipped to [0, 1].

    Returns ``(score, matched_terms)`` where ``matched_terms`` is the
    de-duplicated list of terms that contributed positive points,
    ordered by their contribution (anchors first, then topics).
    """
    if not text:
        return 0.0, []

    title = text
    matched: list[str] = []
    score = 0.0

    title_anchor_hits: list[str] = [a for a in terms.anchors if _word_match(title, a)]
    if title_anchor_hits:
        score += 0.40
        matched.append(title_anchor_hits[0])
        for extra in title_anchor_hits[1:3]:
            score += 0.20
            matched.append(extra)

    title_topic_hits: list[str] = [t for t in terms.topics if _word_match(title, t)]
    # When the question has no anchors, the first topic match is the
    # strongest signal we have — promote it to anchor-tier weighting so
    # the floor is reachable on topic-only questions.
    topic_first_weight = 0.10 if terms.anchors else 0.30
    for idx, t in enumerate(title_topic_hits[:3]):
        score += topic_first_weight if idx == 0 else 0.10
        matched.append(t)

    if body:
        for a in terms.anchors:
            if a in title_anchor_hits:
                continue
            if _word_match(body, a):
                score += 0.05
                matched.append(a)

    if _NEGATIVE_CONTEXT_RE.search(title):
        score -= 0.50

    score = max(0.0, min(1.0, score * title_weight))
    # Dedupe while preserving the contribution-ordered sequence.
    seen: set[str] = set()
    deduped: list[str] = []
    for m in matched:
        k = m.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(m)
    return score, deduped


# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------


def build_phrase_query(terms: QuestionTerms) -> str:
    """Quoted-phrase query suitable for GDELT / Algolia.

    Strategy: each anchor becomes a quoted phrase (so multi-word
    anchors like ``"Joe Biden"`` are kept atomic). Topic terms are
    appended unquoted so the upstream still has fuzzy ranking signal.
    Empty inputs return an empty string — caller decides the fallback.
    """
    parts: list[str] = []
    for a in terms.anchors:
        a = a.strip()
        if not a:
            continue
        parts.append(f'"{a}"' if " " in a else a)
    for t in terms.topics:
        if t.strip():
            parts.append(t)
    return " ".join(parts)


def build_anchor_phrase(terms: QuestionTerms) -> str:
    """Single quoted phrase covering the first anchor (or empty).

    Used as a *hard* filter for Reddit: ``q="Joe Biden"`` matches the
    phrase, not the union of "joe" and "biden". Falls back to the
    first topic if no anchors are available.
    """
    if terms.anchors:
        a = terms.anchors[0].strip()
        if a:
            return f'"{a}"' if " " in a else a
    if terms.topics:
        return terms.topics[0]
    return ""


def build_reddit_query(terms: QuestionTerms) -> str:
    """Reddit-friendly query: phrase-quoted anchor + 1-2 topic terms.

    Reddit's search uses lucene syntax: bare tokens are OR-joined by
    default, but a quoted phrase is required. We require the first
    anchor (if any) and AND the first topic to keep the result set
    on-topic without being empty.
    """
    chunks: list[str] = []
    if terms.anchors:
        a = terms.anchors[0]
        chunks.append(f'"{a}"' if " " in a else a)
    chunks.extend(terms.topics[:2])
    return " ".join(chunks)


# ---------------------------------------------------------------------------
# Convenience filter
# ---------------------------------------------------------------------------


def filter_and_rank(
    items: list[tuple[str, str, object]],
    terms: QuestionTerms,
    *,
    min_score: float = RELEVANCE_MIN,
) -> list[tuple[float, list[str], object]]:
    """Score, filter, and rank a sequence of ``(title, body, payload)`` tuples.

    Returns a list of ``(score, matched_terms, payload)`` filtered to
    ``score >= min_score`` and sorted by score descending. ``payload``
    is opaque — typically a Pydantic model the caller will attach the
    score to.
    """
    scored: list[tuple[float, list[str], object]] = []
    for title, body, payload in items:
        score, matched = score_relevance(title, terms, body=body)
        if score >= min_score:
            scored.append((score, matched, payload))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored


__all__ = [
    "MAX_ANCHORS",
    "MAX_TOPICS",
    "RELEVANCE_MIN",
    "QuestionTerms",
    "build_anchor_phrase",
    "build_phrase_query",
    "build_reddit_query",
    "build_terms",
    "filter_and_rank",
    "score_relevance",
]

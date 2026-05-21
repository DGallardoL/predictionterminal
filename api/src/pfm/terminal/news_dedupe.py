"""Across-source news dedupe via SimHash on title.

Multi-source ingestion (GDELT + Reddit + HN + RSS) emits near-duplicates
because the same headline reappears across feeds with minor punctuation /
wording shifts. This module produces a 64-bit SimHash signature per title
and merges items whose Hamming distance falls within a configurable
threshold.

Public API
----------
- ``simhash(text, *, bits=64) -> int``
- ``hamming(a, b) -> int``
- ``NewsItem`` dataclass
- ``dedupe_news(items, *, threshold_bits=4) -> list[NewsItem]``

The dedupe pass is O(n^2) on the input length; that is fine for the
typical batch sizes (<1000 items per query). For larger inputs swap in
LSH bands -- left as a follow-up task.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime

__all__ = [
    "NewsItem",
    "dedupe_news",
    "hamming",
    "simhash",
    "tokenize",
]


# Minimal English stopword set tuned for news headlines.  Kept small on
# purpose: aggressive stopword removal would collapse headlines that
# genuinely differ on a function word.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, drop stopwords.

    Returns the surviving token list in order.  Numbers are kept (e.g.
    ``25bps`` -> ``25bps``).  Whitespace and punctuation are folded by
    the regex.
    """

    if not text:
        return []
    lowered = text.lower()
    return [tok for tok in _TOKEN_RE.findall(lowered) if tok not in _STOPWORDS]


def _hash_feature(feature: str, bits: int) -> int:
    """Hash a single token to a ``bits``-wide unsigned integer.

    Uses BLAKE2b truncated to the requested width.  BLAKE2b is fast and
    avoids the legacy biases of MD5/SHA1 short prefixes.
    """

    byte_width = max(1, (bits + 7) // 8)
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=byte_width).digest()
    value = int.from_bytes(digest, "big")
    mask = (1 << bits) - 1
    return value & mask


def simhash(text: str, *, bits: int = 64) -> int:
    """Compute the SimHash signature of ``text`` as an unsigned integer.

    Algorithm:
    1. Tokenise into features (see :func:`tokenize`).
    2. Hash each feature to a ``bits``-wide value.
    3. For each bit position, accumulate ``+weight`` if that bit is set
       in the feature hash else ``-weight``.  Weight is the feature
       frequency.
    4. Final signature bit is 1 iff the accumulator is positive.

    Empty input returns 0.  Bits must be a positive multiple of 8 (we
    rely on byte-aligned BLAKE2b digests).
    """

    if bits <= 0 or bits % 8 != 0:
        raise ValueError("bits must be a positive multiple of 8")
    tokens = tokenize(text)
    if not tokens:
        return 0
    # Feature weights = token frequency.
    weights: dict[str, int] = {}
    for tok in tokens:
        weights[tok] = weights.get(tok, 0) + 1
    accum = [0] * bits
    for feature, weight in weights.items():
        h = _hash_feature(feature, bits)
        for i in range(bits):
            if (h >> i) & 1:
                accum[i] += weight
            else:
                accum[i] -= weight
    sig = 0
    for i in range(bits):
        if accum[i] > 0:
            sig |= 1 << i
    return sig


def hamming(a: int, b: int) -> int:
    """Hamming distance between two unsigned integers (bit-difference count)."""

    if a < 0 or b < 0:
        raise ValueError("hamming requires non-negative integers")
    return (a ^ b).bit_count()


@dataclass
class NewsItem:
    """A single news headline.

    ``sources`` / ``urls`` are populated only after :func:`dedupe_news`
    has merged duplicate items.  They are absent on freshly ingested
    items (use the singular ``source`` / ``url`` fields then).
    """

    title: str
    url: str
    source: str
    published_at: datetime
    tone: float | None = None
    # Populated only on merged items returned by ``dedupe_news``.
    sources: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)


def dedupe_news(
    items: list[NewsItem],
    *,
    threshold_bits: int = 4,
) -> list[NewsItem]:
    """Dedupe a list of NewsItem via SimHash on ``title``.

    Two items collide when ``hamming(sig_a, sig_b) <= threshold_bits``.
    On collision the *earliest* ``published_at`` wins and the merged
    item accumulates the per-input ``source`` / ``url`` values into the
    ``sources`` and ``urls`` lists (deduplicated, order-preserving).

    Returns a new list ordered by the earliest occurrence of each
    cluster in the input.  The input is not mutated.
    """

    if not items:
        return []
    if threshold_bits < 0:
        raise ValueError("threshold_bits must be non-negative")

    sigs: list[int] = [simhash(it.title) for it in items]
    n = len(items)
    cluster_of: list[int] = [-1] * n
    clusters: list[list[int]] = []

    for i in range(n):
        if cluster_of[i] != -1:
            continue
        cluster_id = len(clusters)
        cluster_of[i] = cluster_id
        members = [i]
        for j in range(i + 1, n):
            if cluster_of[j] != -1:
                continue
            if hamming(sigs[i], sigs[j]) <= threshold_bits:
                cluster_of[j] = cluster_id
                members.append(j)
        clusters.append(members)

    out: list[NewsItem] = []
    for members in clusters:
        # Earliest published_at wins; ties broken by original order.
        winner_idx = min(members, key=lambda k: (items[k].published_at, k))
        winner = items[winner_idx]
        # Merge sources/urls in the order encountered.
        merged_sources: list[str] = []
        merged_urls: list[str] = []
        seen_sources: set[str] = set()
        seen_urls: set[str] = set()
        for k in members:
            it = items[k]
            for src in (*(it.sources or []), it.source):
                if src and src not in seen_sources:
                    seen_sources.add(src)
                    merged_sources.append(src)
            for url in (*(it.urls or []), it.url):
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    merged_urls.append(url)
        merged = NewsItem(
            title=winner.title,
            url=winner.url,
            source=winner.source,
            published_at=winner.published_at,
            tone=winner.tone,
            sources=merged_sources,
            urls=merged_urls,
        )
        out.append(merged)
    return out

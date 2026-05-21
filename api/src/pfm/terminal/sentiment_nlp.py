"""Hybrid NLP sentiment scorer for news headlines.

Why hybrid: vanilla VADER (tuned for social-media language) scores
"Bitcoin surges to new all-time high" as 0.00 — it doesn't know
"surges" is bullish in finance. We boost it with a hand-curated
financial-domain lexicon (Loughran-McDonald-inspired but compact)
and, when present, an external "tone" signal from GDELT.

Output contract
---------------

* ``score_headline(text, *, external_tone=None) -> tuple[float, str]``
  returns ``(compound_score, label)`` where:
    - ``compound_score ∈ [-1.0, +1.0]``  (signed)
    - ``label ∈ {"positive", "negative", "neutral"}``  (threshold at ±0.15)

* ``aggregate_sentiment(scores) -> tuple[float, str, str]``
  takes a list of compound scores from articles in one jump window
  and returns ``(mean_score, dominant_label, jump_alignment)`` where
  ``jump_alignment`` is one of:
    - ``"agrees"``    — news sentiment matches the jump direction
    - ``"disagrees"`` — news bullish but jump down (or vice versa)
    - ``"neutral"``   — either side is too close to zero to call

Why a separate module
---------------------
Used by both ``terminal/jumps.py`` and (eventually) ``terminal/news.py``.
Keeping it pure (no IO, no FastAPI imports) makes it cheap to unit-test
and lets callers cache the scorer instance — the underlying VADER
analyzer caches its lexicon and is moderately expensive to construct.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _VADER_AVAILABLE = True
except ImportError:  # pragma: no cover — fallback path
    SentimentIntensityAnalyzer = None  # type: ignore[assignment]
    _VADER_AVAILABLE = False


Label = Literal["positive", "negative", "neutral"]


# Loughran-McDonald-inspired financial lexicon. Tiny vs the full list
# (~250 words each side) but covers the high-frequency vocabulary that
# vanilla VADER misses on financial headlines.
# Each entry contributes ``±FIN_TOKEN_WEIGHT`` to the compound score.
FIN_TOKEN_WEIGHT = 0.18


_FIN_POSITIVE: frozenset[str] = frozenset(
    {
        # price/move-up vocabulary
        "surge",
        "surges",
        "surged",
        "surging",
        "soar",
        "soars",
        "soared",
        "rally",
        "rallies",
        "rallied",
        "rallying",
        "spike",
        "spikes",
        "spiked",
        "jump",
        "jumps",
        "jumped",
        "jumping",
        "climb",
        "climbs",
        "climbed",
        "gain",
        "gains",
        "gained",
        "gainer",
        "gainers",
        "rise",
        "rises",
        "risen",
        "rising",
        "rose",
        "advance",
        "advances",
        "advanced",
        "advancing",
        "breakout",
        "breakouts",
        "rebound",
        "rebounds",
        "rebounded",
        "recovery",
        "recovers",
        "recovered",
        "outperform",
        "outperforms",
        "outperformed",
        # event/news vocabulary
        "approved",
        "approves",
        "approval",
        "approve",
        "deal",
        "deals",
        "agreement",
        "settled",
        "settle",
        "resolution",
        "resolved",
        "wins",
        "win",
        "won",
        "winner",
        "winning",
        "victory",
        "boost",
        "boosts",
        "boosted",
        "upgrade",
        "upgrades",
        "upgraded",
        "beat",
        "beats",
        "beating",
        "exceeded",
        "exceeds",
        "stronger",
        "strongest",
        "best",
        "better",
        "improved",
        "improves",
        "improving",
        "robust",
        "solid",
        "strong",
        "strength",
        "positive",
        "optimism",
        "optimistic",
        "bullish",
        "bull",
        "expansion",
        "expand",
        "expands",
        "expanding",
        # crypto/tech specific bullish
        "moon",
        "moons",
        "halving",
        "adoption",
        "etf",
        "etfs",
    }
)


_FIN_NEGATIVE: frozenset[str] = frozenset(
    {
        # price/move-down vocabulary
        "crash",
        "crashes",
        "crashed",
        "crashing",
        "plunge",
        "plunges",
        "plunged",
        "plunging",
        "plummet",
        "plummets",
        "plummeted",
        "plummeting",
        "tumble",
        "tumbles",
        "tumbled",
        "tumbling",
        "slump",
        "slumps",
        "slumped",
        "slide",
        "slides",
        "slid",
        "sliding",
        "fall",
        "falls",
        "fell",
        "falling",
        "drop",
        "drops",
        "dropped",
        "dropping",
        "sink",
        "sinks",
        "sank",
        "sinking",
        "dip",
        "dips",
        "dipped",
        "dipping",
        "decline",
        "declines",
        "declined",
        "declining",
        "selloff",
        "sell-off",
        "rout",
        "routs",
        "routed",
        "correction",
        "corrections",
        "bearish",
        "bear",
        "downgrade",
        "downgrades",
        "downgraded",
        # event/news vocabulary
        "miss",
        "missed",
        "misses",
        "missing",
        "loss",
        "losses",
        "losing",
        "lose",
        "lost",
        "loser",
        "losers",
        "default",
        "defaults",
        "defaulted",
        "bankrupt",
        "bankruptcy",
        "bankrupted",
        "weak",
        "weakness",
        "weaker",
        "weakest",
        "worst",
        "worse",
        "deteriorating",
        "deteriorate",
        "deteriorated",
        "deterioration",
        "concern",
        "concerns",
        "concerned",
        "concerning",
        "fear",
        "fears",
        "feared",
        "fearful",
        "worry",
        "worries",
        "worried",
        "uncertain",
        "uncertainty",
        "risk",
        "risks",
        "risky",
        "warning",
        "warned",
        "warns",
        "warn",
        "negative",
        "pessimism",
        "pessimistic",
        "recession",
        "recessions",
        "depression",
        "crisis",
        "crises",
        "panic",
        "panicked",
        "stress",
        "stressed",
        "stressful",
        "turmoil",
        "volatile",
        "volatility",
        "shock",
        "shocks",
        "shocked",
        # event-specific bearish
        "halted",
        "halts",
        "halt",
        "suspended",
        "suspends",
        "suspend",
        "investigation",
        "investigated",
        "investigates",
        "lawsuit",
        "lawsuits",
        "hack",
        "hacked",
        "hacks",
        "exploit",
        "exploits",
        "exploited",
        "scam",
        "scams",
        "fraud",
        "rejected",
        "rejects",
        "reject",
    }
)


# Negators within ±2 words flip the polarity of an adjacent financial token.
_NEGATORS: frozenset[str] = frozenset(
    {
        "no",
        "not",
        "never",
        "without",
        "lacks",
        "lacking",
        "fails",
        "failed",
        "failing",
        "fail",
        "doesn't",
        "don't",
        "won't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "cannot",
        "can't",
    }
)


@lru_cache(maxsize=1)
def _vader_analyzer():
    """Cache the analyzer — its lexicon construction is non-trivial."""
    if not _VADER_AVAILABLE:
        return None
    return SentimentIntensityAnalyzer()


def _financial_score(text: str) -> float:
    """Add up financial-token weights with a basic negator flip.

    Tokens are lowercased and matched word-by-word; a negator within
    two positions to the left flips the contributing sign. Returns a
    score in roughly ``[-1.5, +1.5]`` (we soft-clip downstream).
    """
    toks = re.findall(r"[A-Za-z']+", text.lower())
    score = 0.0
    for i, tok in enumerate(toks):
        is_pos = tok in _FIN_POSITIVE
        is_neg = tok in _FIN_NEGATIVE
        if not (is_pos or is_neg):
            continue
        weight = FIN_TOKEN_WEIGHT * (1.0 if is_pos else -1.0)
        # Negator look-back (window of 2 preceding tokens)
        for back in range(1, 3):
            if i - back < 0:
                break
            if toks[i - back] in _NEGATORS:
                weight = -weight
                break
        score += weight
    # Soft clip so a string with many bullish words doesn't dominate.
    return max(-1.0, min(1.0, score))


def _label_for(score: float, threshold: float = 0.15) -> Label:
    """Bucket a compound score into a label with a deadband around 0."""
    if score >= threshold:
        return "positive"
    if score <= -threshold:
        return "negative"
    return "neutral"


@lru_cache(maxsize=8192)
def _score_headline_cached(
    text: str,
    external_tone: float | None,
) -> tuple[float, Label]:
    """Cached core of ``score_headline``.

    Pure function of ``(text, external_tone)`` — safe to memoise.
    Both arguments are hashable (``str`` and ``Optional[float]``), so we
    use a plain ``functools.lru_cache`` keyed positionally. ``None`` and
    ``0.0`` are treated as **distinct** keys (Python's default hash/eq):
    callers passing ``external_tone=0.0`` will compute a fresh entry even
    though the resulting score is identical to ``external_tone=None``.
    That's fine — ``0.0`` is rarely passed explicitly (GDELT either omits
    tone or returns a real number); the duplication cost is negligible.
    """
    if not text:
        return 0.0, "neutral"

    components: list[tuple[float, float]] = []  # (weight, value)

    if _VADER_AVAILABLE:
        v = _vader_analyzer().polarity_scores(text)["compound"]
        if abs(v) > 1e-6:
            components.append((0.50, float(v)))

    f = _financial_score(text)
    if abs(f) > 1e-6:
        components.append((0.40, float(f)))

    if external_tone is not None and abs(external_tone) > 1e-6:
        ext = max(-1.0, min(1.0, float(external_tone) / 10.0))
        components.append((0.10, ext))

    if not components:
        return 0.0, "neutral"

    total_w = sum(w for w, _ in components)
    compound = sum(w * v for w, v in components) / total_w
    compound = max(-1.0, min(1.0, compound))
    return round(compound, 4), _label_for(compound)


def score_headline(
    text: str,
    *,
    external_tone: float | None = None,
) -> tuple[float, Label]:
    """Score a single headline. Returns ``(compound, label)``.

    Combines (with these weights):
      - VADER's ``compound`` (50%)
      - Domain financial lexicon (40%)
      - ``external_tone / 10`` mapped to [-1, +1] (10%) when GDELT provides it

    The weights are weighted-mean style — components with zero contribution
    are dropped from the denominator so a fully-zero VADER doesn't drag
    a strong financial score down.

    Identical ``(text, external_tone)`` calls are memoised with a
    ``functools.lru_cache(maxsize=8192)`` — see ``_score_headline_cached``.
    Use ``score_headline.cache_info()`` / ``score_headline.cache_clear()``
    to inspect or reset it (delegated to the underlying cached function).

    Args:
        text: headline text (English).
        external_tone: optional external sentiment in roughly [-10, +10]
            (GDELT convention). Will be normalised to [-1, +1].
    """
    return _score_headline_cached(text, external_tone)


# Expose cache controls on the public function so callers can do
# ``score_headline.cache_info()`` and ``score_headline.cache_clear()``
# the same way they would with a directly-decorated function.
score_headline.cache_info = _score_headline_cached.cache_info  # type: ignore[attr-defined]
score_headline.cache_clear = _score_headline_cached.cache_clear  # type: ignore[attr-defined]


def aggregate_sentiment(
    scores: list[float],
    *,
    jump_direction: Literal["up", "down", "flat"] = "flat",
    label_threshold: float = 0.15,
    alignment_threshold: float = 0.10,
) -> tuple[float, Label, Literal["agrees", "disagrees", "neutral"]]:
    """Aggregate per-article sentiments for a single jump window.

    Returns ``(mean_score, dominant_label, jump_alignment)``.

    ``jump_alignment``:
      - ``"agrees"``    when both move in the same direction by ≥
                        ``alignment_threshold``
      - ``"disagrees"`` when both clear ``alignment_threshold`` but in
                        opposite directions
      - ``"neutral"``   when either is too close to 0 to call

    Empty input → ``(0.0, "neutral", "neutral")``.
    """
    if not scores:
        return 0.0, "neutral", "neutral"
    mean = sum(scores) / len(scores)
    label = _label_for(mean, threshold=label_threshold)
    if jump_direction == "flat" or abs(mean) < alignment_threshold:
        align: Literal["agrees", "disagrees", "neutral"] = "neutral"
    elif jump_direction == "up":
        align = "agrees" if mean > 0 else "disagrees"
    elif jump_direction == "down":
        align = "agrees" if mean < 0 else "disagrees"
    else:  # pragma: no cover - exhaustive
        align = "neutral"
    return round(mean, 4), label, align


# Public alias: CLAUDE.md and some external docs reference ``score_text``.
# The canonical implementation is ``score_headline``; this preserves the
# documented name without renaming the original.
score_text = score_headline


__all__ = [
    "FIN_TOKEN_WEIGHT",
    "_VADER_AVAILABLE",
    "Label",
    "aggregate_sentiment",
    "score_headline",
    "score_text",
]

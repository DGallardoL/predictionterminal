"""Financial-domain sentiment lexicon and scoring utilities.

A lightweight, dependency-free sentiment scorer tailored to financial news
headlines and short articles. Lexicons are curated for finance/macro/markets
contexts, where words like "rally", "sanction", or "downgrade" carry specific
polarity that general-purpose lexicons (e.g., VADER) miss or get backwards.

Scoring algorithm:
    1. Tokenize the input on whitespace and basic punctuation.
    2. Walk tokens left-to-right, tracking a 2-gram window for negation/amplifier.
    3. For each polarity word, contribution = sign * amp_factor * neg_factor.
    4. Aggregate, then normalize by the count of polarity-bearing tokens to
       produce a score in [-1, +1]. Confidence is the polarity density, capped
       by total tokens.
"""

from __future__ import annotations

import re
from typing import Final

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

POSITIVE_WORDS: Final[list[str]] = [
    "rally",
    "rallies",
    "rallied",
    "surge",
    "surges",
    "surged",
    "surging",
    "soar",
    "soars",
    "soared",
    "soaring",
    "jump",
    "jumps",
    "jumped",
    "climb",
    "climbs",
    "climbed",
    "rise",
    "rises",
    "rose",
    "rising",
    "gain",
    "gains",
    "gained",
    "gaining",
    "advance",
    "advances",
    "advanced",
    "bullish",
    "bull",
    "outperform",
    "outperforms",
    "outperformed",
    "upgrade",
    "upgrades",
    "upgraded",
    "beat",
    "beats",
    "exceed",
    "exceeds",
    "exceeded",
    "surpass",
    "surpassed",
    "boost",
    "boosts",
    "boosted",
    "boom",
    "booming",
    "expand",
    "expands",
    "expanded",
    "expanding",
    "growth",
    "grow",
    "grew",
    "growing",
    "strong",
    "stronger",
    "strongest",
    "strength",
    "robust",
    "solid",
    "healthy",
    "stable",
    "stability",
    "improve",
    "improves",
    "improved",
    "improving",
    "improvement",
    "recover",
    "recovers",
    "recovered",
    "recovery",
    "rebound",
    "rebounds",
    "rebounded",
    "breakthrough",
    "breakthroughs",
    "innovation",
    "innovative",
    "accord",
    "accords",
    "agreement",
    "agreements",
    "deal",
    "deals",
    "partnership",
    "partnerships",
    "alliance",
    "alliances",
    "treaty",
    "treaties",
    "ceasefire",
    "truce",
    "peace",
    "peaceful",
    "resolve",
    "resolved",
    "resolution",
    "approve",
    "approves",
    "approved",
    "approval",
    "endorse",
    "endorsed",
    "support",
    "supports",
    "supported",
    "supporting",
    "supportive",
    "optimistic",
    "optimism",
    "confident",
    "confidence",
    "positive",
    "positively",
    "favorable",
    "favourable",
    "promising",
    "encouraging",
    "upbeat",
    "buoyant",
    "profit",
    "profits",
    "profitable",
    "profitability",
    "earnings",
    "revenue",
    "dividend",
    "dividends",
    "buyback",
    "buybacks",
    "win",
    "wins",
    "winning",
    "success",
    "successful",
    "successfully",
    "achievement",
    "achievements",
    "milestone",
    "milestones",
    "record",
    "records",
    "all-time-high",
    "ath",
    "stimulus",
    "easing",
    "dovish",
    "accommodative",
    "lowered",
    "cut",
    "cuts",
    "expansion",
    "expansionary",
    "tailwind",
    "tailwinds",
    "leadership",
    "leader",
    "leading",
    "premium",
    "outpace",
    "outpaced",
    "thrive",
    "thrives",
    "thriving",
    "flourish",
    "flourishing",
    "momentum",
    "accelerate",
    "accelerates",
    "accelerated",
    "accelerating",
    "breakout",
    "breakouts",
    "uptrend",
    "reform",
    "reforms",
    "stabilize",
    "stabilizes",
    "stabilized",
    "settle",
    "settled",
    "settlement",
    "settlements",
    "resilient",
    "resilience",
    "secure",
    "secures",
    "secured",
    "safe",
    "safer",
    "safest",
]

NEGATIVE_WORDS: Final[list[str]] = [
    "crash",
    "crashes",
    "crashed",
    "crashing",
    "plunge",
    "plunges",
    "plunged",
    "plunging",
    "tumble",
    "tumbles",
    "tumbled",
    "tumbling",
    "slump",
    "slumps",
    "slumped",
    "slumping",
    "fall",
    "falls",
    "fell",
    "falling",
    "drop",
    "drops",
    "dropped",
    "dropping",
    "decline",
    "declines",
    "declined",
    "declining",
    "sink",
    "sinks",
    "sank",
    "sinking",
    "slide",
    "slides",
    "slid",
    "sliding",
    "bearish",
    "bear",
    "underperform",
    "underperforms",
    "underperformed",
    "downgrade",
    "downgrades",
    "downgraded",
    "miss",
    "misses",
    "missed",
    "shortfall",
    "shortfalls",
    "weak",
    "weaker",
    "weakest",
    "weakness",
    "fragile",
    "vulnerable",
    "vulnerability",
    "deteriorate",
    "deteriorates",
    "deteriorated",
    "deteriorating",
    "deterioration",
    "worsen",
    "worsens",
    "worsened",
    "worsening",
    "war",
    "wars",
    "warfare",
    "conflict",
    "conflicts",
    "invasion",
    "invasions",
    "attack",
    "attacks",
    "attacked",
    "strike",
    "strikes",
    "sanction",
    "sanctions",
    "sanctioned",
    "tariff",
    "tariffs",
    "embargo",
    "embargoes",
    "ban",
    "bans",
    "banned",
    "blockade",
    "blockades",
    "recession",
    "recessions",
    "recessionary",
    "depression",
    "downturn",
    "downturns",
    "contraction",
    "contractions",
    "contracting",
    "shrink",
    "shrinks",
    "shrank",
    "shrinking",
    "stagnation",
    "stagflation",
    "inflation",
    "inflationary",
    "hawkish",
    "tightening",
    "hike",
    "hikes",
    "hiked",
    "raise",
    "raises",
    "raised",
    "default",
    "defaults",
    "defaulted",
    "bankruptcy",
    "bankrupt",
    "insolvent",
    "insolvency",
    "loss",
    "losses",
    "loses",
    "lost",
    "losing",
    "deficit",
    "deficits",
    "debt",
    "burden",
    "burdens",
    "fail",
    "fails",
    "failed",
    "failing",
    "failure",
    "failures",
    "collapse",
    "collapses",
    "collapsed",
    "collapsing",
    "panic",
    "panicked",
    "fear",
    "fears",
    "feared",
    "afraid",
    "anxiety",
    "anxious",
    "worry",
    "worries",
    "worried",
    "concern",
    "concerns",
    "concerning",
    "alarming",
    "warning",
    "warnings",
    "threat",
    "threats",
    "threatened",
    "threatening",
    "risk",
    "risks",
    "risky",
    "uncertain",
    "uncertainty",
    "uncertainties",
    "volatile",
    "volatility",
    "turmoil",
    "chaos",
    "chaotic",
    "crisis",
    "crises",
    "scandal",
    "scandals",
    "fraud",
    "fraudulent",
    "investigation",
    "lawsuit",
    "lawsuits",
    "fine",
    "fines",
    "fined",
    "penalty",
    "penalties",
    "headwind",
    "headwinds",
    "downside",
    "selloff",
    "sell-off",
    "rout",
    "routs",
    "bloodbath",
    "meltdown",
    "freeze",
    "frozen",
    "halt",
    "halted",
    "suspend",
    "suspended",
    "suspension",
    "delay",
    "delays",
    "delayed",
    "shortage",
    "shortages",
    "scarce",
    "scarcity",
    "negative",
    "negatively",
    "pessimistic",
    "pessimism",
    "gloomy",
    "bleak",
    "dire",
    "severe",
    "harsh",
    "punish",
    "punishes",
    "punished",
    "reject",
    "rejects",
    "rejected",
    "rejection",
    "refuse",
    "refused",
    "refuses",
    "oppose",
    "opposes",
    "opposed",
    "opposition",
    "protest",
    "protests",
    "unrest",
    "instability",
    "unstable",
]

AMPLIFIERS: Final[list[str]] = [
    "very",
    "extremely",
    "highly",
    "incredibly",
    "remarkably",
    "exceptionally",
    "tremendously",
    "massively",
    "hugely",
    "enormously",
    "significantly",
    "substantially",
    "dramatically",
    "sharply",
    "steeply",
    "deeply",
    "heavily",
    "strongly",
    "severely",
    "seriously",
    "drastically",
    "vastly",
    "greatly",
    "particularly",
    "notably",
    "decidedly",
    "absolutely",
    "completely",
    "totally",
    "utterly",
    "fully",
    "thoroughly",
    "intensely",
    "profoundly",
    "soaring",
    "skyrocketing",
    "plummeting",
    "record-high",
    "record-low",
]

NEGATORS: Final[list[str]] = [
    "not",
    "no",
    "never",
    "none",
    "nothing",
    "neither",
    "nor",
    "without",
    "lacks",
    "lacking",
    "lack",
    "absent",
    "denies",
    "denied",
    "deny",
    "fail",
    "fails",
    "failed",
    "failing",  # also negative; double-counts intentionally
    "refuse",
    "refused",
    "refuses",
    "cannot",
    "can't",
    "won't",
    "wouldn't",
    "shouldn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "didn't",
    "doesn't",
    "don't",
    "hasn't",
    "haven't",
    "hadn't",
]

# Multi-word negator phrases handled via regex pre-pass (lowercased input).
NEGATOR_PHRASES: Final[list[str]] = [
    "fail to",
    "failed to",
    "fails to",
    "failing to",
    "refuse to",
    "refused to",
    "refuses to",
    "unable to",
    "decline to",
    "declines to",
    "declined to",
    "no longer",
    "by no means",
]

# ---------------------------------------------------------------------------
# Topic categorization
# ---------------------------------------------------------------------------

TOPICS: Final[dict[str, list[str]]] = {
    "macro": [
        "fed",
        "fomc",
        "ecb",
        "boj",
        "boe",
        "central bank",
        "cpi",
        "ppi",
        "inflation",
        "deflation",
        "gdp",
        "unemployment",
        "payrolls",
        "nfp",
        "rate",
        "rates",
        "yield",
        "yields",
        "treasury",
        "treasuries",
        "recession",
        "stagflation",
        "stimulus",
        "fiscal",
        "monetary",
        "powell",
        "lagarde",
        "tariff",
        "trade balance",
        "consumer",
        "retail sales",
    ],
    "geopolitics": [
        "war",
        "ceasefire",
        "sanction",
        "sanctions",
        "treaty",
        "election",
        "elections",
        "putin",
        "trump",
        "biden",
        "xi",
        "nato",
        "un",
        "eu",
        "ukraine",
        "russia",
        "china",
        "taiwan",
        "iran",
        "israel",
        "gaza",
        "korea",
        "diplomacy",
        "summit",
        "embargo",
        "invasion",
        "coup",
        "protest",
        "unrest",
        "regime",
        "geopolitical",
    ],
    "crypto": [
        "bitcoin",
        "btc",
        "ethereum",
        "eth",
        "crypto",
        "cryptocurrency",
        "blockchain",
        "stablecoin",
        "defi",
        "nft",
        "halving",
        "mining",
        "miner",
        "wallet",
        "exchange",
        "binance",
        "coinbase",
        "tether",
        "usdt",
        "usdc",
        "altcoin",
        "token",
        "etf approval",
    ],
    "equities": [
        "earnings",
        "revenue",
        "guidance",
        "ipo",
        "buyback",
        "dividend",
        "merger",
        "acquisition",
        "spinoff",
        "split",
        "stock",
        "shares",
        "ceo",
        "cfo",
        "board",
        "shareholder",
        "10-k",
        "10-q",
        "filing",
    ],
    "energy": [
        "oil",
        "crude",
        "wti",
        "brent",
        "opec",
        "gas",
        "natural gas",
        "lng",
        "refinery",
        "barrel",
        "pipeline",
        "saudi",
        "drilling",
        "energy",
        "renewable",
        "solar",
        "wind",
    ],
    "tech": [
        "ai",
        "artificial intelligence",
        "chip",
        "chips",
        "semiconductor",
        "nvidia",
        "openai",
        "anthropic",
        "google",
        "microsoft",
        "apple",
        "meta",
        "amazon",
        "cloud",
        "cybersecurity",
        "data center",
        "quantum",
        "robotics",
    ],
    "regulation": [
        "sec",
        "cftc",
        "fda",
        "doj",
        "ftc",
        "regulator",
        "regulation",
        "antitrust",
        "lawsuit",
        "settlement",
        "fine",
        "investigation",
        "compliance",
        "subpoena",
        "indictment",
    ],
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")

_POS_SET: Final[frozenset[str]] = frozenset(w.lower() for w in POSITIVE_WORDS)
_NEG_SET: Final[frozenset[str]] = frozenset(w.lower() for w in NEGATIVE_WORDS)
_AMP_SET: Final[frozenset[str]] = frozenset(w.lower() for w in AMPLIFIERS)
_NEGATOR_SET: Final[frozenset[str]] = frozenset(w.lower() for w in NEGATORS)

AMPLIFIER_FACTOR: Final[float] = 1.3
NEGATION_WINDOW: Final[int] = 3  # tokens of look-back for negation/amplifier


def _preprocess(text: str) -> str:
    """Lowercase and substitute multi-word negator phrases with a single token."""
    lowered = text.lower()
    for phrase in NEGATOR_PHRASES:
        lowered = lowered.replace(phrase, "__NEG__")
    return lowered


def _tokenize(text: str) -> list[str]:
    pre = _preprocess(text)
    # Keep our placeholder token intact alongside word tokens.
    raw = re.findall(r"__NEG__|[A-Za-z][A-Za-z'\-]*", pre)
    return raw


def score_sentiment(text: str) -> dict:
    """Compute a financial sentiment score for ``text``.

    Args:
        text: Free-form input (e.g., a news headline).

    Returns:
        A dict with keys ``score`` (float in [-1, 1]), ``n_positive``,
        ``n_negative``, ``n_neutral`` (token counts), ``dominant`` (one of
        "positive"/"negative"/"neutral"), and ``confidence`` (float in [0, 1]).
    """
    if not text or not text.strip():
        return {
            "score": 0.0,
            "n_positive": 0,
            "n_negative": 0,
            "n_neutral": 0,
            "dominant": "neutral",
            "confidence": 0.0,
        }

    tokens = _tokenize(text)
    n_total = len(tokens)
    n_pos = 0
    n_neg = 0
    weighted = 0.0

    for i, tok in enumerate(tokens):
        if tok not in _POS_SET and tok not in _NEG_SET:
            continue

        sign = 1.0 if tok in _POS_SET else -1.0
        amp = 1.0
        negate = False

        # Look back up to NEGATION_WINDOW tokens for amplifiers/negators.
        start = max(0, i - NEGATION_WINDOW)
        for j in range(start, i):
            prev = tokens[j]
            if prev == "__NEG__" or prev in _NEGATOR_SET:
                negate = not negate
            if prev in _AMP_SET:
                amp *= AMPLIFIER_FACTOR

        contribution = sign * amp
        if negate:
            contribution = -contribution

        if contribution > 0:
            n_pos += 1
        else:
            n_neg += 1
        weighted += contribution

    n_polarity = n_pos + n_neg
    n_neutral = max(0, n_total - n_polarity)

    if n_polarity == 0:
        score = 0.0
        dominant = "neutral"
        confidence = 0.0
    else:
        # Normalize by the sum of absolute possible contributions: each polarity
        # token contributes at most AMPLIFIER_FACTOR. Using n_polarity * AMP as
        # the denominator keeps the score in [-1, +1].
        score = weighted / (n_polarity * AMPLIFIER_FACTOR)
        score = max(-1.0, min(1.0, score))

        if score > 0.05:
            dominant = "positive"
        elif score < -0.05:
            dominant = "negative"
        else:
            dominant = "neutral"

        # Confidence: polarity density, with a soft floor for very short text.
        confidence = min(1.0, n_polarity / max(1, n_total))

    return {
        "score": round(score, 4),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_neutral": n_neutral,
        "dominant": dominant,
        "confidence": round(confidence, 4),
    }


def categorize(text: str) -> dict[str, list[str]]:
    """Return a mapping of topic -> matched keywords found in ``text``.

    Only topics with at least one match are included.
    """
    if not text:
        return {}
    lowered = text.lower()
    out: dict[str, list[str]] = {}
    for topic, keywords in TOPICS.items():
        hits = [kw for kw in keywords if kw in lowered]
        if hits:
            out[topic] = hits
    return out


__all__ = [
    "AMPLIFIERS",
    "NEGATIVE_WORDS",
    "NEGATORS",
    "POSITIVE_WORDS",
    "TOPICS",
    "categorize",
    "score_sentiment",
]

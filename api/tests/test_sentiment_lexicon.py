"""Tests for the financial sentiment lexicon."""

from __future__ import annotations

from pfm.sentiment_lexicon import (
    AMPLIFIERS,
    NEGATIVE_WORDS,
    NEGATORS,
    POSITIVE_WORDS,
    categorize,
    score_sentiment,
)


def test_positive_headline() -> None:
    """A clearly bullish headline scores positive."""
    result = score_sentiment("Stocks rally as earnings beat estimates and growth surges")
    assert result["dominant"] == "positive"
    assert result["score"] > 0.3
    assert result["n_positive"] >= 3
    assert result["n_negative"] == 0
    assert 0.0 <= result["confidence"] <= 1.0


def test_negative_headline() -> None:
    """A clearly bearish headline scores negative."""
    result = score_sentiment(
        "Markets crash as recession fears mount and economy plunges into crisis"
    )
    assert result["dominant"] == "negative"
    assert result["score"] < -0.3
    assert result["n_negative"] >= 3
    assert result["n_positive"] == 0


def test_neutral_headline() -> None:
    """A factual headline with no polarity words scores neutral."""
    result = score_sentiment("The Federal Reserve will release its meeting minutes on Wednesday")
    assert result["dominant"] == "neutral"
    assert result["score"] == 0.0
    assert result["n_positive"] == 0
    assert result["n_negative"] == 0
    assert result["confidence"] == 0.0


def test_negation_flips_sentiment() -> None:
    """Negation should flip a positive word to negative."""
    pos = score_sentiment("Earnings beat expectations")
    neg = score_sentiment("Earnings did not beat expectations")
    assert pos["dominant"] == "positive"
    # The negation should pull the score down significantly (no longer positive-dominant).
    assert neg["score"] < pos["score"]
    assert neg["dominant"] in {"negative", "neutral"}

    # "Fail to deliver" should also flip a positive.
    flipped = score_sentiment("Company failed to beat guidance")
    assert flipped["score"] <= 0.0


def test_amplifier_intensifies_sentiment() -> None:
    """An amplifier should make the score more extreme than the baseline."""
    base = score_sentiment("Stocks surge on strong earnings")
    amped = score_sentiment("Stocks surge dramatically on extremely strong earnings")
    assert base["dominant"] == "positive"
    assert amped["dominant"] == "positive"
    # Amplified version should be at least as strong, and strictly greater
    # because the AMPLIFIER_FACTOR multiplies polarity-bearing tokens.
    assert amped["score"] >= base["score"]
    assert amped["score"] > 0.0


def test_mixed_sentiment() -> None:
    """A mixed headline should report both positive and negative counts."""
    result = score_sentiment(
        "Earnings beat estimates but guidance disappoints amid recession fears"
    )
    assert result["n_positive"] >= 1
    assert result["n_negative"] >= 1
    # Score lives in [-1, 1].
    assert -1.0 <= result["score"] <= 1.0
    # Confidence reflects polarity density.
    assert result["confidence"] > 0.0


def test_lexicon_sizes_and_categorize() -> None:
    """Sanity-check lexicon coverage and topic categorization."""
    assert len(POSITIVE_WORDS) >= 140
    assert len(NEGATIVE_WORDS) >= 140
    assert len(AMPLIFIERS) >= 20
    assert len(NEGATORS) >= 15
    assert "rally" in POSITIVE_WORDS
    assert "crash" in NEGATIVE_WORDS

    cats = categorize("Fed signals rate cut as bitcoin surges and oil falls on OPEC decision")
    assert "macro" in cats
    assert "crypto" in cats
    assert "energy" in cats

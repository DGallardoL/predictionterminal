"""Tests for the hybrid sentiment scorer (VADER + financial lexicon).

The whole point of the financial lexicon is to push scores in the right
direction on headlines VADER fails on (e.g. "Bitcoin surges to new ATH"
which vanilla VADER scores 0.00). These tests pin the contract.
"""

from __future__ import annotations

import pytest

from pfm.terminal.sentiment_nlp import (
    _financial_score,
    _label_for,
    aggregate_sentiment,
    score_headline,
)

# ---------------------------------------------------------------------------
# Financial lexicon
# ---------------------------------------------------------------------------


def test_financial_score_is_positive_for_bull_words() -> None:
    assert _financial_score("Bitcoin surges to new high") > 0
    assert _financial_score("Crude oil rallies on supply cut") > 0
    assert _financial_score("ETF approved — bullish breakout") > 0


def test_financial_score_is_negative_for_bear_words() -> None:
    assert _financial_score("Bitcoin crashes amid macro rout") < 0
    assert _financial_score("Crude oil plummets on demand fears") < 0
    assert _financial_score("Banks face downgrade after lawsuit") < 0


def test_financial_score_neutral_for_irrelevant_text() -> None:
    assert _financial_score("Bitcoin trades sideways through quiet session") == 0


def test_financial_score_negator_flips_polarity() -> None:
    # "not crashing" should flip the negative "crashing" → effectively positive
    flipped = _financial_score("Bitcoin is not crashing despite the noise")
    plain = _financial_score("Bitcoin is crashing despite the noise")
    assert flipped > 0
    assert plain < 0


def test_financial_score_caps_at_one() -> None:
    # Loaded headline with many bull words should still cap at +1.0
    s = _financial_score(
        "Surge rally jump climb gain rise advance breakout rebound recovery boost win victory"
    )
    assert s <= 1.0
    s_neg = _financial_score(
        "Crash plunge plummet tumble slump slide fall drop sink dip decline selloff rout"
    )
    assert s_neg >= -1.0


# ---------------------------------------------------------------------------
# score_headline (hybrid)
# ---------------------------------------------------------------------------


def test_score_headline_returns_score_label_tuple() -> None:
    s, lab = score_headline("Bitcoin surges to all-time high")
    assert -1.0 <= s <= 1.0
    assert lab in {"positive", "negative", "neutral"}


def test_score_headline_recovers_bullish_headline_vader_misses() -> None:
    """Vanilla VADER scores this as 0; financial lexicon must push it positive."""
    s, lab = score_headline("Bitcoin surges to new all-time high on ETF inflows")
    assert s > 0.15
    assert lab == "positive"


def test_score_headline_recovers_bearish_headline_vader_misses() -> None:
    s, lab = score_headline("Bitcoin crashes amid macro rout")
    assert s < -0.15
    assert lab == "negative"


def test_score_headline_uses_external_tone_when_provided() -> None:
    """A headline with neutral text + strongly-negative GDELT tone should
    lean negative (the GDELT signal is real)."""
    # Plain text where neither VADER nor financial lex fires
    s_no_ext, _ = score_headline("The president signed legislation today")
    s_with_ext, lab_with_ext = score_headline(
        "The president signed legislation today", external_tone=-7.0
    )
    assert s_with_ext < s_no_ext
    assert lab_with_ext == "negative"


def test_score_headline_empty_input_is_neutral() -> None:
    s, lab = score_headline("")
    assert s == 0.0
    assert lab == "neutral"


def test_score_headline_all_zero_components_returns_neutral() -> None:
    """A text VADER scores 0 and that has no financial tokens → 0 / neutral."""
    s, lab = score_headline("the cat sat on the mat")
    assert abs(s) < 0.15
    assert lab == "neutral"


# ---------------------------------------------------------------------------
# Label thresholds
# ---------------------------------------------------------------------------


def test_label_threshold_uses_deadband() -> None:
    assert _label_for(0.16) == "positive"
    assert _label_for(0.14) == "neutral"
    assert _label_for(-0.14) == "neutral"
    assert _label_for(-0.16) == "negative"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_empty_input() -> None:
    assert aggregate_sentiment([]) == (0.0, "neutral", "neutral")


def test_aggregate_uniform_positive_with_up_jump_agrees() -> None:
    mean, lab, align = aggregate_sentiment([0.6, 0.4, 0.5], jump_direction="up")
    assert mean > 0
    assert lab == "positive"
    assert align == "agrees"


def test_aggregate_positive_news_with_down_jump_disagrees() -> None:
    mean, lab, align = aggregate_sentiment([0.6, 0.4, 0.5], jump_direction="down")
    assert mean > 0
    assert lab == "positive"
    assert align == "disagrees"


def test_aggregate_uniform_negative_with_down_jump_agrees() -> None:
    mean, lab, align = aggregate_sentiment([-0.6, -0.4, -0.5], jump_direction="down")
    assert mean < 0
    assert lab == "negative"
    assert align == "agrees"


def test_aggregate_neutral_news_no_alignment() -> None:
    # Mean near zero → alignment "neutral" regardless of jump direction
    _mean, lab, align = aggregate_sentiment([0.05, -0.04, 0.02], jump_direction="up")
    assert lab == "neutral"
    assert align == "neutral"


def test_aggregate_flat_jump_no_alignment() -> None:
    _mean, lab, align = aggregate_sentiment([0.6, 0.5], jump_direction="flat")
    assert lab == "positive"
    assert align == "neutral"


# ---------------------------------------------------------------------------
# Real-world headline corpus (regression — verifies common cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "headline,expected_label",
    [
        ("Bitcoin surges to new all-time high on ETF inflows", "positive"),
        ("Bitcoin crashes amid macro rout", "negative"),
        ("Crude Oil Prices Plummet on Optimism of US-Iran War to End", "negative"),
        # "Closure tightens supplies" reads bearish (supply-shock) in plain
        # English even though it's bullish for an oil long — context-dependent
        # headlines like this should fall in the neutral deadband.
        ("Oil prices supported as Hormuz tensions rise", "neutral"),
        ("Trump wins re-election by narrow margin", "positive"),
        ("Federal Reserve fears recession, signals rate cuts", "negative"),
        ("Apple beats earnings, stock rallies", "positive"),
        ("Tesla halted on regulatory investigation", "negative"),
        ("Stocks trade flat amid mixed signals", "neutral"),
    ],
)
def test_real_headline_labels(headline: str, expected_label: str) -> None:
    _s, label = score_headline(headline)
    assert label == expected_label, f"expected {expected_label} for {headline!r}"


# ---------------------------------------------------------------------------
# LRU cache behaviour
# ---------------------------------------------------------------------------


def test_score_headline_lru_cache_hits_on_repeat_input() -> None:
    """Same ``(text, external_tone)`` twice → second call is a cache hit.

    News feeds frequently re-publish identical Reuters/AP headlines across
    Yahoo, FT, etc. — deduping via lru_cache saves ~0.2 ms of VADER per dup.
    """
    score_headline.cache_clear()
    txt = "Bitcoin surges to new all-time high on ETF inflows"
    s1, lab1 = score_headline(txt)
    info_after_first = score_headline.cache_info()
    assert info_after_first.hits == 0
    assert info_after_first.misses == 1

    s2, lab2 = score_headline(txt)
    info_after_second = score_headline.cache_info()
    assert info_after_second.hits == 1
    assert info_after_second.misses == 1
    # Cached value must match the original (return contract preserved).
    assert (s1, lab1) == (s2, lab2)


def test_score_headline_lru_cache_distinguishes_different_inputs() -> None:
    """Different texts must NOT collide on the same cache entry."""
    score_headline.cache_clear()
    s_a, lab_a = score_headline("Bitcoin surges to new all-time high")
    s_b, lab_b = score_headline("Bitcoin crashes amid macro rout")
    info = score_headline.cache_info()
    # Both calls miss (distinct keys), no hits.
    assert info.misses == 2
    assert info.hits == 0
    assert info.currsize == 2
    # Sanity: results are actually different.
    assert lab_a == "positive"
    assert lab_b == "negative"
    assert s_a > 0 > s_b


def test_score_headline_lru_cache_external_tone_is_part_of_key() -> None:
    """``external_tone`` participates in the cache key.

    Same text with different tones → distinct cache entries; same text
    with the same tone (including both being ``None``) → cache hit.
    """
    score_headline.cache_clear()
    txt = "The president signed legislation today"

    s_none_1, _ = score_headline(txt)  # external_tone=None
    s_none_2, _ = score_headline(txt)  # repeat → hit
    s_neg, _ = score_headline(txt, external_tone=-7.0)  # distinct key → miss
    s_neg_repeat, _ = score_headline(txt, external_tone=-7.0)  # hit

    info = score_headline.cache_info()
    # 2 distinct keys: (txt, None) and (txt, -7.0). 4 calls total → 2 hits, 2 misses.
    assert info.misses == 2
    assert info.hits == 2
    # Sanity: tone actually moves the score.
    assert s_none_1 == s_none_2
    assert s_neg == s_neg_repeat
    assert s_neg < s_none_1


def test_score_headline_cache_clear_resets_counters() -> None:
    """``cache_clear()`` must be exposed on the public function and work."""
    score_headline("Apple beats earnings, stock rallies")
    score_headline.cache_clear()
    info = score_headline.cache_info()
    assert info.hits == 0
    assert info.misses == 0
    assert info.currsize == 0
    # And maxsize is the documented 8192.
    assert info.maxsize == 8192


# ---------------------------------------------------------------------------
# Additional edge cases for aggregate_sentiment (filling coverage gaps)
# ---------------------------------------------------------------------------


def test_aggregate_all_zero_scores_returns_neutral() -> None:
    """A list of strictly-zero scores must hit the alignment_threshold gate
    and yield ``("neutral","neutral")`` regardless of jump direction."""
    for direction in ("up", "down", "flat"):
        mean, lab, align = aggregate_sentiment([0.0, 0.0, 0.0], jump_direction=direction)
        assert mean == 0.0
        assert lab == "neutral"
        assert align == "neutral"


def test_aggregate_mixed_signs_cancel_to_zero() -> None:
    """+0.5 and -0.5 average to 0 → no actionable alignment even with an up jump.
    Pins the "deadband first" semantics: small absolute means take priority over
    the direction signal."""
    mean, lab, align = aggregate_sentiment([0.5, -0.5], jump_direction="up")
    assert mean == 0.0
    assert lab == "neutral"
    assert align == "neutral"


def test_aggregate_single_score_is_passed_through() -> None:
    """One-element list should yield mean == that element (no off-by-one)."""
    mean, lab, align = aggregate_sentiment([0.42], jump_direction="up")
    assert mean == pytest.approx(0.42, abs=1e-4)
    assert lab == "positive"
    assert align == "agrees"


def test_aggregate_negative_news_with_up_jump_disagrees() -> None:
    """Symmetric to the existing positive/down disagrees test."""
    mean, lab, align = aggregate_sentiment([-0.6, -0.4, -0.5], jump_direction="up")
    assert mean < 0
    assert lab == "negative"
    assert align == "disagrees"


def test_aggregate_custom_label_threshold_lets_caller_widen_deadband() -> None:
    """A mean of 0.10 should be 'positive' under default threshold (0.15→neutral
    because 0.10<0.15, actually neutral) — pin the threshold-override behavior."""
    # Default threshold 0.15 → 0.10 falls in deadband.
    _mean, lab_default, _align = aggregate_sentiment([0.10, 0.10], jump_direction="flat")
    assert lab_default == "neutral"
    # Lower threshold to 0.05 → same mean reads as positive.
    _mean2, lab_low, _align2 = aggregate_sentiment(
        [0.10, 0.10],
        jump_direction="flat",
        label_threshold=0.05,
    )
    assert lab_low == "positive"


def test_aggregate_custom_alignment_threshold_changes_action() -> None:
    """A weak +0.05 mean with up jump: at default 0.10 alignment threshold
    → neutral; lower the threshold to 0.01 → agrees."""
    _m, _l, a_default = aggregate_sentiment([0.05, 0.05], jump_direction="up")
    assert a_default == "neutral"
    _m2, _l2, a_low = aggregate_sentiment(
        [0.05, 0.05],
        jump_direction="up",
        alignment_threshold=0.01,
    )
    assert a_low == "agrees"

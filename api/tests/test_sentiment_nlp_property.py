"""Hypothesis property-based tests for ``pfm.terminal.sentiment_nlp``.

Validates structural invariants of ``score_headline`` (the public scorer)
across random inputs:

1. Output compound score is always inside [-1.0, 1.0].
2. Empty / whitespace-only strings score 0.0 with label "neutral".
3. ``None`` input is rejected gracefully (no segfault / crash).
4. Symmetry-ish: ``score("X") + score("not X")`` is near zero for a
   small library of single-token financial phrases (negator handling).
5. Length invariance: scoring a phrase is robust to trailing
   whitespace / non-alphanumeric punctuation appended.
6. LRU cache: scoring the same text twice returns exactly the same tuple.
7. Unicode safety: arbitrary unicode strings don't raise.
8. Idempotency: scoring already-scored phrase shape returns same answer
   when called repeatedly.

Skipped entirely if ``hypothesis`` isn't installed (it currently is, but
guard the file so it stays optional).
"""

from __future__ import annotations

import math

import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pfm.terminal.sentiment_nlp import score_headline

# Common settings: 200 examples each, suppress slow-data and
# function-scoped-fixture health checks (we don't use fixtures here).
PROPERTY_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


# ---------------------------------------------------------------------------
# Property 1: output bounded in [-1, 1]
# ---------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(text=st.text(max_size=400))
def test_output_in_unit_range(text: str) -> None:
    compound, label = score_headline(text)
    assert isinstance(compound, float)
    assert -1.0 <= compound <= 1.0
    assert label in {"positive", "negative", "neutral"}
    assert math.isfinite(compound)


# ---------------------------------------------------------------------------
# Property 2: empty / whitespace -> 0.0 neutral
# ---------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(
    pad=st.text(
        alphabet=st.sampled_from([" ", "\t", "\n", "\r", "\f", "\v"]),
        max_size=20,
    )
)
def test_whitespace_only_is_neutral(pad: str) -> None:
    """Pure-whitespace and empty strings produce no signal.

    Note: ``score_headline`` does not strip — but a string containing only
    whitespace has no alphabetic tokens for the financial scorer and
    yields a VADER compound of 0, so both code paths short-circuit to
    the neutral fallback.
    """
    compound, label = score_headline(pad)
    assert compound == 0.0
    assert label == "neutral"


def test_empty_string_explicit() -> None:
    compound, label = score_headline("")
    assert compound == 0.0
    assert label == "neutral"


# ---------------------------------------------------------------------------
# Property 3: None safety
# ---------------------------------------------------------------------------


def test_none_input_does_not_segfault() -> None:
    """Passing ``None`` is not a supported input — but it must not crash
    the Python interpreter. The current implementation raises a
    ``TypeError`` from ``lru_cache`` / ``re.findall`` paths, which is the
    acceptable defensive behaviour. We just assert *something* sane
    happens (exception, or a neutral score).
    """
    try:
        compound, label = score_headline(None)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        # Expected: lru_cache or downstream string ops reject None.
        return
    # If it didn't raise, it must have returned a valid neutral-shaped tuple.
    assert -1.0 <= compound <= 1.0
    assert label in {"positive", "negative", "neutral"}


# ---------------------------------------------------------------------------
# Property 4: negation symmetry (limited library)
# ---------------------------------------------------------------------------


# Each base phrase has a clear positive or negative connotation in the
# financial lexicon. ``"not " + phrase`` should land closer to zero than
# the original. Strict cancellation isn't guaranteed (VADER's intensifier
# weights aren't symmetric), so we assert a magnitude-reduction property
# rather than exact algebraic symmetry.
_NEGATION_PAIRS = [
    "bullish",
    "bearish",
    "rally",
    "crash",
    "surges",
    "plunges",
    "beat",
    "miss",
    "strong",
    "weak",
]


@PROPERTY_SETTINGS
@given(phrase=st.sampled_from(_NEGATION_PAIRS))
def test_negation_reduces_magnitude(phrase: str) -> None:
    """``not <phrase>`` should be closer to zero than ``<phrase>`` alone
    OR flip sign. Either is acceptable evidence that the negator
    handling is doing *something* — both are improvements over leaving
    the polarity unchanged.
    """
    pos_score, _ = score_headline(phrase)
    neg_score, _ = score_headline(f"not {phrase}")

    # If the original had non-trivial signal, negation must either flip
    # sign or attenuate the absolute value.
    if abs(pos_score) > 0.05:
        assert (pos_score * neg_score <= 0) or (abs(neg_score) < abs(pos_score))


def test_pairwise_negation_sum_near_zero() -> None:
    """For a small curated set, ``score(x) + score("not " + x)`` should
    sit inside a fairly tight band around zero.

    Threshold is loose (0.6) because VADER baseline + financial-lex
    blend is not algebraically symmetric — but it should never *amplify*
    on average.
    """
    sums = []
    for phrase in _NEGATION_PAIRS:
        a, _ = score_headline(phrase)
        b, _ = score_headline(f"not {phrase}")
        sums.append(a + b)
    mean_sum = sum(sums) / len(sums)
    assert abs(mean_sum) < 0.6, f"mean negation sum {mean_sum} too large"


# ---------------------------------------------------------------------------
# Property 5: length / punctuation invariance
# ---------------------------------------------------------------------------


# Punctuation that does NOT contain alphabetic chars and shouldn't move
# the financial-lexicon needle. VADER reacts to emoticons formed by
# combinations like ``:)`` / ``:(`` (parens + colon/semicolon/dash), so
# we restrict to plain separators that can't form a smiley pattern.
_NEUTRAL_PUNCT = st.text(
    alphabet=st.sampled_from([".", ",", " "]),
    max_size=15,
)


@PROPERTY_SETTINGS
@given(suffix=_NEUTRAL_PUNCT)
def test_neutral_punctuation_does_not_change_score(suffix: str) -> None:
    """Appending neutral, non-emphatic punctuation should not perturb
    the compound score by more than a small tolerance.
    """
    base = "fed rate cut bullish"
    a, _ = score_headline(base)
    b, _ = score_headline(base + suffix)
    assert abs(a - b) < 1e-3


# ---------------------------------------------------------------------------
# Property 6: LRU cache exactness — same input -> identical output
# ---------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(text=st.text(max_size=200))
def test_cache_returns_identical_tuple(text: str) -> None:
    first = score_headline(text)
    second = score_headline(text)
    # Compound rounded to 4 dp inside the cached path -> bitwise equal.
    assert first == second
    assert first[0] == second[0]
    assert first[1] == second[1]


# ---------------------------------------------------------------------------
# Property 7: Unicode safety
# ---------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(
    text=st.text(
        # Wide unicode block, including emoji, CJK, RTL marks, etc.
        alphabet=st.characters(
            blacklist_categories=("Cs",),  # surrogates are invalid in str
        ),
        max_size=200,
    )
)
def test_unicode_does_not_crash(text: str) -> None:
    """Random unicode input — including emoji, control chars, mixed RTL
    — must produce a valid (score, label) tuple without raising.
    """
    compound, label = score_headline(text)
    assert -1.0 <= compound <= 1.0
    assert label in {"positive", "negative", "neutral"}


# ---------------------------------------------------------------------------
# Property 8: idempotency
# ---------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(text=st.text(max_size=200))
def test_idempotent_repeated_scoring(text: str) -> None:
    """Calling ``score_headline`` 3+ times on the same input is stable —
    nothing in the function should be stateful in a way that drifts the
    output.
    """
    r1 = score_headline(text)
    r2 = score_headline(text)
    r3 = score_headline(text)
    assert r1 == r2 == r3


# ---------------------------------------------------------------------------
# Property 9 (bonus): with-tone path stays bounded
# ---------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(
    text=st.text(max_size=200),
    tone=st.one_of(
        st.none(),
        st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    ),
)
def test_external_tone_does_not_break_bounds(text: str, tone: float | None) -> None:
    """The ``external_tone`` keyword may receive arbitrary floats from
    GDELT. Even out-of-spec ranges (|tone| > 10) must not push the final
    compound outside [-1, 1].
    """
    compound, label = score_headline(text, external_tone=tone)
    assert -1.0 <= compound <= 1.0
    assert label in {"positive", "negative", "neutral"}
    assert math.isfinite(compound)

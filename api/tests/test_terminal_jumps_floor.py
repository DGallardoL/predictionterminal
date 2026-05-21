"""Boundary tests for ``_articles_for_jump_with_floor``.

The function lives in :mod:`pfm.terminal.jumps` and enforces the user
rule "estrictamente no pongas news before el evento empieza": no
article published before ``market_start_ts`` may ever be attributed to
a price jump, even if it falls inside the ``[jump - 2h, jump + 1h]``
proximity window.

These tests exercise the *boundary* conditions of that floor — the
single-second margins, exact-equality cases, ``None`` floor, mixed
timezones, and duck-typed article shims — without touching any
network or Polymarket dependency.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from pfm.terminal.jumps import _articles_for_jump_with_floor
from pfm.terminal_gdelt_news import GDELTArticle

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_article(
    ts: Any, headline: str = "Test", *, source: str = "test.com", tone: float = 0.0
) -> GDELTArticle:
    """Build a minimal :class:`GDELTArticle` for floor-boundary assertions.

    The matcher in ``_articles_for_jump_with_floor`` only reads
    ``.ts``, ``.title``, ``.source``, ``.url`` and ``.tone`` — anything
    else can be left at the model defaults.
    """
    return GDELTArticle(
        ts=str(ts),
        title=headline,
        source=source,
        country="us",
        language="english",
        tone=tone,
        url=f"https://{source}/{abs(hash(headline)) % 10_000}",
    )


# ---------------------------------------------------------------------------
# Boundary tests
# ---------------------------------------------------------------------------


def test_article_one_second_before_market_start_is_dropped() -> None:
    """T34-1: An article 1 s pre-market must be excluded.

    The floor is half-open: ``art_ts < market_start_ts`` is strictly
    excluded; equality is admitted (see T34-2 below).
    """
    jump_ts = pd.Timestamp("2026-05-15T12:00:00Z")
    market_start = pd.Timestamp("2026-05-15T10:00:00Z")
    scored = [
        (make_article("2026-05-15T09:59:59Z", "one-second-pre"), 0.9, ["term"]),
        # Sanity post-market control so we can tell drop-everything from
        # drop-the-pre-market article.
        (make_article("2026-05-15T11:30:00Z", "post-market"), 0.8, ["term"]),
    ]
    picked, n_window = _articles_for_jump_with_floor(jump_ts, scored, market_start_ts=market_start)
    headlines = {p.headline for p in picked}
    assert "one-second-pre" not in headlines
    assert "post-market" in headlines
    assert n_window == 1


def test_article_exactly_at_market_start_is_kept() -> None:
    """T34-2: ``art_ts == market_start_ts`` is admitted (inclusive floor)."""
    jump_ts = pd.Timestamp("2026-05-15T12:00:00Z")
    market_start = pd.Timestamp("2026-05-15T10:00:00Z")
    scored = [
        (make_article("2026-05-15T10:00:00Z", "exact-at-start"), 0.9, ["term"]),
    ]
    picked, n_window = _articles_for_jump_with_floor(jump_ts, scored, market_start_ts=market_start)
    assert n_window == 1
    assert len(picked) == 1
    assert picked[0].headline == "exact-at-start"


def test_article_one_second_after_market_start_is_kept() -> None:
    """T34-3: An article 1 s post-market falls inside the floor."""
    jump_ts = pd.Timestamp("2026-05-15T12:00:00Z")
    market_start = pd.Timestamp("2026-05-15T10:00:00Z")
    scored = [
        (make_article("2026-05-15T10:00:01Z", "one-second-post"), 0.9, ["term"]),
    ]
    picked, n_window = _articles_for_jump_with_floor(jump_ts, scored, market_start_ts=market_start)
    assert n_window == 1
    assert len(picked) == 1
    assert picked[0].headline == "one-second-post"


def test_article_one_hour_after_start_and_one_second_before_jump_is_kept() -> None:
    """T34-4: Well inside both bounds — must always be admitted.

    This is the canonical "good" case: post-market, pre-jump, near
    enough that proximity decay still gives it a positive rank.
    """
    jump_ts = pd.Timestamp("2026-05-15T12:00:00Z")
    market_start = pd.Timestamp("2026-05-15T10:00:00Z")
    scored = [
        (
            make_article("2026-05-15T11:59:59Z", "just-before-jump"),
            0.95,
            ["term"],
        ),
    ]
    picked, n_window = _articles_for_jump_with_floor(jump_ts, scored, market_start_ts=market_start)
    assert n_window == 1
    assert picked[0].headline == "just-before-jump"
    # Negative seconds_from_jump => article precedes the jump.
    assert picked[0].seconds_from_jump == -1


def test_article_after_lookahead_window_is_dropped() -> None:
    """T34-5: Articles past ``jump_ts + LOOKAHEAD_HOURS`` are dropped.

    The default lookahead is 1 h. An article 2 h after the jump lies
    outside the window even with no floor in play.
    """
    jump_ts = pd.Timestamp("2026-05-15T12:00:00Z")
    market_start = pd.Timestamp("2026-05-15T08:00:00Z")
    scored = [
        (make_article("2026-05-15T14:00:00Z", "too-late"), 0.9, ["term"]),
        # Control: an in-window article so we know the function is
        # otherwise working.
        (make_article("2026-05-15T12:30:00Z", "in-window"), 0.6, ["term"]),
    ]
    picked, n_window = _articles_for_jump_with_floor(jump_ts, scored, market_start_ts=market_start)
    assert n_window == 1
    assert {p.headline for p in picked} == {"in-window"}


def test_market_start_ts_none_disables_floor() -> None:
    """T34-6: ``market_start_ts=None`` ⇒ no floor; only the proximity window applies."""
    jump_ts = pd.Timestamp("2026-05-15T12:00:00Z")
    scored = [
        # Inside lookback (2 h before jump).
        (make_article("2026-05-15T11:00:00Z", "in-window"), 0.5, []),
        # Inside lookahead.
        (make_article("2026-05-15T12:30:00Z", "post-jump"), 0.5, []),
    ]
    picked, n_window = _articles_for_jump_with_floor(jump_ts, scored, market_start_ts=None)
    assert n_window == 2
    assert {p.headline for p in picked} == {"in-window", "post-jump"}


def test_empty_scored_list_returns_empty() -> None:
    """T34-7: An empty input list short-circuits to ``([], 0)``."""
    jump_ts = pd.Timestamp("2026-05-15T12:00:00Z")
    picked, n_window = _articles_for_jump_with_floor(
        jump_ts, [], market_start_ts=pd.Timestamp("2026-05-15T08:00:00Z")
    )
    assert picked == []
    assert n_window == 0


def test_all_articles_pre_market_returns_empty() -> None:
    """T34-8: When every candidate predates the floor, return ``([], 0)``."""
    jump_ts = pd.Timestamp("2026-05-15T12:00:00Z")
    market_start = pd.Timestamp("2026-05-15T11:00:00Z")
    scored = [
        (make_article("2026-05-15T10:00:00Z", "stale-1"), 0.95, ["term"]),
        (make_article("2026-05-15T10:30:00Z", "stale-2"), 0.90, ["term"]),
        (make_article("2026-05-15T10:59:59Z", "stale-3"), 0.99, ["term"]),
    ]
    picked, n_window = _articles_for_jump_with_floor(jump_ts, scored, market_start_ts=market_start)
    assert picked == []
    assert n_window == 0


def test_mixed_timezones_naive_and_utc_normalize_correctly() -> None:
    """T34-9: Both naive and UTC-labelled article ts strings normalize to UTC.

    The matcher calls ``tz_localize('UTC')`` on naive timestamps and
    ``tz_convert('UTC')`` on already-aware ones, so a naive
    ``"2026-05-15T11:30:00"`` should be treated identically to its
    ``"...Z"`` cousin.
    """
    jump_ts = pd.Timestamp("2026-05-15T12:00:00Z")
    market_start = pd.Timestamp("2026-05-15T08:00:00Z")
    scored = [
        # Naive (no tz suffix) — must be assumed UTC.
        (make_article("2026-05-15T11:30:00", "naive"), 0.9, ["term"]),
        # Aware Z-suffix.
        (make_article("2026-05-15T11:30:00Z", "aware-utc"), 0.8, ["term"]),
        # Aware non-UTC offset that maps to 11:30 UTC.
        (make_article("2026-05-15T13:30:00+02:00", "aware-cest"), 0.7, ["term"]),
    ]
    picked, n_window = _articles_for_jump_with_floor(jump_ts, scored, market_start_ts=market_start)
    assert n_window == 3
    headlines = {p.headline for p in picked}
    assert headlines == {"naive", "aware-utc", "aware-cest"}
    # All three articles map to 11:30 UTC ⇒ 30 min before the jump ⇒
    # ``seconds_from_jump == -1800``.
    assert all(p.seconds_from_jump == -1800 for p in picked)


def test_iso_string_ts_with_fractional_seconds_is_accepted() -> None:
    """T34-10a: ISO-8601 ``.ts`` strings with fractional seconds parse correctly.

    The contract is that ``.ts`` is an ISO-8601 *string* — both the
    upstream :class:`GDELTArticle` validator and the downstream
    :class:`pfm.terminal.jumps.JumpArticle` model require ``str``.
    Within that contract, microsecond precision must round-trip
    through the matcher.
    """
    jump_ts = pd.Timestamp("2026-05-15T12:00:00Z")
    market_start = pd.Timestamp("2026-05-15T08:00:00Z")
    scored = [
        (make_article("2026-05-15T11:30:00.123456Z", "with-fractional"), 0.9, ["term"]),
    ]
    picked, n_window = _articles_for_jump_with_floor(jump_ts, scored, market_start_ts=market_start)
    assert n_window == 1
    assert picked[0].headline == "with-fractional"
    # Sub-second precision must still place the article 30 min before
    # the jump (rounding to int seconds).
    assert picked[0].seconds_from_jump == -1799


def test_unix_seconds_must_be_pre_formatted_to_iso_by_callers() -> None:
    """T34-10b: Unix-seconds ``.ts`` (raw int or numeric str) is rejected upstream.

    Polymarket's price API returns unix seconds; if a caller forwards
    them verbatim, :class:`GDELTArticle` (which constrains ``ts: str``)
    refuses raw ints, and a numeric *string* like ``"1747311000"`` is
    parsed by ``pd.Timestamp`` as the year (out of range). The
    documented wire format is ISO-8601: callers must convert
    upstream. This test pins that contract so a future refactor that
    silently changes ts-type assumptions surfaces here first.
    """
    # Raw int rejected by the Pydantic model.
    with pytest.raises(Exception):
        GDELTArticle(
            ts=1_747_310_400,  # type: ignore[arg-type]
            title="raw-int",
            source="x.com",
            country="us",
            language="english",
            url="https://x.com/y",
        )

    # Numeric *string* is accepted by the model but unparseable by
    # pd.Timestamp ⇒ the matcher silently skips it (no crash).
    jump_ts = pd.Timestamp("2026-05-15T12:00:00Z")
    market_start = pd.Timestamp("2026-05-15T08:00:00Z")
    scored = [
        (make_article("1747311000", "raw-unix-str"), 0.9, ["term"]),
        (make_article("2026-05-15T11:30:00Z", "iso-control"), 0.8, ["term"]),
    ]
    picked, n_window = _articles_for_jump_with_floor(jump_ts, scored, market_start_ts=market_start)
    assert n_window == 1
    assert {p.headline for p in picked} == {"iso-control"}


def test_unparseable_ts_is_silently_skipped() -> None:
    """T34-bonus: Unparseable ``.ts`` strings are skipped, not raised.

    The matcher wraps ``pd.Timestamp(art.ts)`` in a try/except so a
    single malformed article cannot poison the whole jump→news join.
    """
    jump_ts = pd.Timestamp("2026-05-15T12:00:00Z")
    market_start = pd.Timestamp("2026-05-15T08:00:00Z")
    scored = [
        (make_article("not-a-real-date", "garbage"), 0.9, ["term"]),
        (make_article("2026-05-15T11:30:00Z", "good"), 0.8, ["term"]),
    ]
    picked, n_window = _articles_for_jump_with_floor(jump_ts, scored, market_start_ts=market_start)
    assert n_window == 1
    assert {p.headline for p in picked} == {"good"}

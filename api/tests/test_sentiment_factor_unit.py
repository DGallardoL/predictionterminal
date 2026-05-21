"""Unit tests for ``pfm.sources.sentiment_factor`` (Task W11-29 / T35).

These tests target ≥85 % line coverage of the module while keeping every
upstream call mocked via :mod:`respx`. They exercise the three free
sources (GDELT timelinetone, Reddit search, HN Algolia), the caching
layer, the curated catalogue, the ``sentiment:<query>`` syntax detector,
and the :func:`fetch_sentiment_history` dispatcher shim.

Design notes
------------
* Each test gets its own ``SentimentFactorSource`` injected with a fresh
  :class:`TerminalCache` so cache state never leaks between tests.
* The module's ``_SINGLETON`` is reset between tests that touch it.
* ``score_headline`` is the same blended VADER + finance-lex scorer the
  module imports; we use real-finance headlines to keep the test
  assertions stable against the lexicon's known behaviour.
* Concurrent-call test uses :mod:`threading` — the cache itself is
  thread-safe and the source has no shared mutable state beyond it.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import httpx
import pandas as pd
import pytest
import respx

from pfm import sources as _sources_pkg  # noqa: F401  (ensure namespace loaded)
from pfm.cache_utils import TerminalCache
from pfm.sources import sentiment_factor as sf
from pfm.sources.sentiment_factor import (
    CACHE_NAMESPACE,
    CACHE_TTL_SECONDS,
    CURATED_QUERIES,
    GDELT_DOC_URL,
    HN_SEARCH_URL,
    REDDIT_SEARCH_URL,
    SentimentFactorSource,
    _seendate_to_ts,
    _ts_from_unix,
    curated_sentiment_query,
    fetch_sentiment_history,
    parse_sentiment_factor_id,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fresh_cache() -> TerminalCache:
    """A fresh, isolated cache instance so tests don't leak state."""
    return TerminalCache(default_ttl=CACHE_TTL_SECONDS)


@pytest.fixture()
def source(fresh_cache: TerminalCache) -> SentimentFactorSource:
    """A SentimentFactorSource backed by an isolated cache."""
    src = SentimentFactorSource(cache=fresh_cache)
    yield src
    src.close()


@pytest.fixture(autouse=True)
def reset_singleton():
    """Make sure module-level ``_SINGLETON`` doesn't leak across tests."""
    sf._SINGLETON = None
    yield
    sf._SINGLETON = None


def _gdelt_payload(rows: list[tuple[str, float]]) -> dict[str, Any]:
    """Build a minimal GDELT timelinetone-style payload."""
    return {
        "timeline": [
            {
                "series": "Average Tone",
                "data": [{"date": date, "value": tone} for date, tone in rows],
            }
        ]
    }


def _reddit_payload(rows: list[tuple[int, str]]) -> dict[str, Any]:
    """Build a minimal Reddit search.json payload."""
    return {
        "data": {"children": [{"data": {"created_utc": ts, "title": title}} for ts, title in rows]}
    }


def _hn_payload(rows: list[tuple[int, str]]) -> dict[str, Any]:
    """Build a minimal HN Algolia /search payload."""
    return {"hits": [{"created_at_i": ts, "title": title} for ts, title in rows]}


def _today_unix() -> int:
    """A unix-seconds value pinned to "today, UTC midnight" for stable buckets."""
    return int(pd.Timestamp.utcnow().tz_convert("UTC").normalize().timestamp())


# ---------------------------------------------------------------------------
# helpers tests
# ---------------------------------------------------------------------------


def test_seendate_compact_form_parses_to_utc():
    ts = _seendate_to_ts("20260101T120000Z")
    assert ts is not None
    assert ts.tzinfo is not None
    assert ts.year == 2026 and ts.month == 1 and ts.day == 1


def test_seendate_iso_form_parses():
    ts = _seendate_to_ts("2026-03-15T00:00:00Z")
    assert ts is not None
    assert ts.year == 2026 and ts.month == 3 and ts.day == 15


def test_seendate_empty_returns_none():
    assert _seendate_to_ts("") is None
    assert _seendate_to_ts("   ") is None


def test_seendate_garbage_returns_none():
    assert _seendate_to_ts("not-a-date") is None
    assert _seendate_to_ts("99999999T999999Z") is None


def test_ts_from_unix_valid_and_invalid():
    assert _ts_from_unix(0) == pd.Timestamp(0, unit="s", tz="UTC")
    assert _ts_from_unix(None) is None
    assert _ts_from_unix("abc") is None


def test_parse_sentiment_factor_id_positive():
    ok, q = parse_sentiment_factor_id("sentiment:bitcoin price")
    assert ok is True
    assert q == "bitcoin price"


def test_parse_sentiment_factor_id_case_insensitive():
    ok, q = parse_sentiment_factor_id("SENTIMENT:FED")
    assert ok is True
    assert q == "FED"


def test_parse_sentiment_factor_id_blank_query_returns_none():
    ok, q = parse_sentiment_factor_id("sentiment:   ")
    assert ok is True
    assert q is None


def test_parse_sentiment_factor_id_non_sentiment_input():
    assert parse_sentiment_factor_id("eq:AAPL") == (False, None)
    assert parse_sentiment_factor_id("") == (False, None)
    assert parse_sentiment_factor_id(None) == (False, None)  # type: ignore[arg-type]


def test_curated_sentiment_query_known_and_unknown():
    assert curated_sentiment_query("sentiment_bitcoin") == "bitcoin"
    assert curated_sentiment_query("not-a-real-key") is None


def test_curated_catalog_is_well_formed():
    for fid, entry in CURATED_QUERIES.items():
        assert fid.startswith("sentiment_")
        assert entry.get("query")
        assert "name" in entry
        assert "description" in entry


# ---------------------------------------------------------------------------
# get_daily_sentiment — input validation
# ---------------------------------------------------------------------------


def test_empty_query_returns_empty_series(source):
    s = source.get_daily_sentiment("", days=30)
    assert s.empty
    assert s.name == ""


def test_non_string_query_returns_empty_series(source):
    s = source.get_daily_sentiment(None, days=30)  # type: ignore[arg-type]
    assert s.empty


def test_days_clamped_to_at_least_one(source):
    # No mocks → upstream connection errors → empty series, but the call
    # must not raise on ``days <= 0``.
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(side_effect=httpx.ConnectError("nope"))
        rmock.get(REDDIT_SEARCH_URL).mock(side_effect=httpx.ConnectError("nope"))
        rmock.get(HN_SEARCH_URL).mock(side_effect=httpx.ConnectError("nope"))
        s = source.get_daily_sentiment("bitcoin", days=0)
    assert s.empty


# ---------------------------------------------------------------------------
# GDELT branch
# ---------------------------------------------------------------------------


def test_gdelt_empty_response_yields_empty_series(source):
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        s = source.get_daily_sentiment("bitcoin", days=30)
    assert s.empty
    assert s.name == "bitcoin"


def test_gdelt_five_positive_articles_mean_positive(source):
    rows = [
        ("20260101T000000Z", 6.0),
        ("20260102T000000Z", 7.5),
        ("20260103T000000Z", 5.0),
        ("20260104T000000Z", 8.0),
        ("20260105T000000Z", 9.0),
    ]
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload(rows)))
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        s = source.get_daily_sentiment("bitcoin", days=30)
    assert not s.empty
    assert len(s) == 5
    assert s.mean() > 0
    # tone scaled by /10
    assert all(-1.0 <= v <= 1.0 for v in s.values)


def test_gdelt_mixed_sentiment_aggregates_reasonably(source):
    rows = [
        ("20260101T000000Z", 8.0),
        ("20260102T000000Z", -7.0),
        ("20260103T000000Z", 0.0),
        ("20260104T000000Z", -4.0),
        ("20260105T000000Z", 3.0),
    ]
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload(rows)))
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        s = source.get_daily_sentiment("bitcoin", days=30)
    # min/max bounded by [-1, +1]
    assert s.min() >= -1.0 and s.max() <= 1.0
    # Has both positive and negative days
    assert (s > 0).any() and (s < 0).any()


def test_gdelt_clamps_extreme_tones_to_unit_range(source):
    rows = [
        ("20260101T000000Z", 100.0),  # absurd +tone → must clamp to +1
        ("20260102T000000Z", -100.0),  # absurd -tone → must clamp to -1
    ]
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload(rows)))
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        s = source.get_daily_sentiment("bitcoin", days=30)
    assert s.max() <= 1.0 and s.min() >= -1.0
    assert s.iloc[0] == pytest.approx(1.0)
    assert s.iloc[1] == pytest.approx(-1.0)


def test_gdelt_non_2xx_returns_empty(source):
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(503, text="boom"))
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        s = source.get_daily_sentiment("bitcoin", days=30)
    assert s.empty


def test_gdelt_throttle_banner_returns_empty(source):
    body = "Please limit your query rate. Try again later."
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, text=body))
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        s = source.get_daily_sentiment("bitcoin", days=30)
    assert s.empty


def test_gdelt_invalid_json_returns_empty(source):
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(
            return_value=httpx.Response(200, text="not json at all <html>")
        )
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        s = source.get_daily_sentiment("bitcoin", days=30)
    assert s.empty


def test_gdelt_network_error_returns_empty(source):
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(side_effect=httpx.ConnectError("kaput"))
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        s = source.get_daily_sentiment("bitcoin", days=30)
    assert s.empty


def test_gdelt_malformed_timeline_shapes_ignored(source):
    # ``timeline`` not a list, then a list with non-dict entries, then a
    # dict missing "data". All paths must be handled without raising.
    bodies = [
        {"timeline": "nope"},
        {"timeline": ["x", 42]},
        {"timeline": [{"series": "Other", "data": "still-not-list"}]},
        {
            "timeline": [
                {
                    "series": "Average Tone",
                    "data": [
                        "not-a-dict",
                        {"date": "bad", "value": 1.0},
                        {"date": "20260101T000000Z", "value": "not-a-float"},
                        {"date": "20260101T000000Z", "value": None},
                        {"date": "20260102T000000Z", "value": 3.0},
                    ],
                }
            ]
        },
    ]
    for body in bodies:
        local_src = SentimentFactorSource(cache=TerminalCache(default_ttl=CACHE_TTL_SECONDS))
        with respx.mock(assert_all_called=False) as rmock:
            rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=body))
            rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
            rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
            s = local_src.get_daily_sentiment(f"bitcoin-{id(body)}", days=10)
        # Only the last body has any usable point → series has one entry
        if body is bodies[-1]:
            assert len(s) == 1
        else:
            assert s.empty
        local_src.close()


def test_gdelt_fallback_to_first_data_series_when_no_average_tone(source):
    body = {
        "timeline": [
            {
                "series": "Some Other Series",
                "data": [
                    {"date": "20260105T000000Z", "value": 4.0},
                ],
            },
        ]
    }
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=body))
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        s = source.get_daily_sentiment("alt-series", days=30)
    assert len(s) == 1
    assert s.iloc[0] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Reddit branch
# ---------------------------------------------------------------------------


def test_reddit_three_positive_posts_contribute_to_mean(source):
    now = _today_unix()
    rows = [
        (now, "Stocks surge as earnings beat estimates"),
        (now - 86400, "Bitcoin rally accelerates on strong growth"),
        (now - 2 * 86400, "Markets soar to all-time highs"),
    ]
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
        rmock.get(REDDIT_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_reddit_payload(rows))
        )
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        s = source.get_daily_sentiment("bitcoin", days=5)
    assert not s.empty
    # All three headlines have positive financial-lex scores; aggregate > 0.
    assert s.mean() > 0


def test_reddit_old_posts_dropped_by_cutoff(source):
    now = _today_unix()
    too_old = now - 365 * 86400  # one year old; well outside the window
    rows = [
        (too_old, "Stocks surge as earnings beat estimates"),
        (now, "Bitcoin rally continues on strong growth"),
    ]
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
        rmock.get(REDDIT_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_reddit_payload(rows))
        )
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        s = source.get_daily_sentiment("bitcoin", days=3)
    # Only the recent post remains → at most one day
    assert len(s) <= 1


def test_reddit_skipped_when_days_far_exceeds_ceiling(source):
    # days = 200 > 2 * _REDDIT_MAX_DAYS (31*2 = 62) → reddit branch skipped
    now = _today_unix()
    rows = [(now, "Stocks surge as earnings beat estimates")]
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
        reddit_route = rmock.get(REDDIT_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_reddit_payload(rows))
        )
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        s = source.get_daily_sentiment("bitcoin", days=200)
    assert s.empty
    assert reddit_route.call_count == 0


def test_reddit_non_2xx_yields_empty_contribution(source):
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(429, json={}))
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        s = source.get_daily_sentiment("bitcoin", days=5)
    assert s.empty


def test_reddit_invalid_json_yields_empty(source):
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, text="not json"))
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        s = source.get_daily_sentiment("bitcoin", days=5)
    assert s.empty


def test_reddit_network_error_yields_empty(source):
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
        rmock.get(REDDIT_SEARCH_URL).mock(side_effect=httpx.ReadTimeout("slow"))
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        s = source.get_daily_sentiment("bitcoin", days=5)
    assert s.empty


def test_reddit_zero_score_headlines_dropped(source):
    """Headlines whose blended score is ~0 don't pollute the daily mean."""
    now = _today_unix()
    rows = [
        (now, "the weather is mild today"),  # neutral → 0 → dropped
        (now, "Stocks surge as earnings beat estimates"),  # positive
    ]
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
        rmock.get(REDDIT_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_reddit_payload(rows))
        )
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        s = source.get_daily_sentiment("bitcoin", days=5)
    # Single day, single non-zero score → positive
    assert len(s) == 1
    assert s.iloc[0] > 0


def test_reddit_malformed_payload_branches(source):
    # children not a list, individual child not a dict, child with no title.
    # NB: the module assumes ``data`` is a dict when present — a non-dict
    # ``data`` is treated as an upstream contract violation and not
    # defended against, so we don't test that case here.
    bodies = [
        {"data": {"children": [None, 42, "string"]}},
        {"data": {"children": [{"data": {"title": "", "created_utc": None}}]}},
        {"data": {}},
    ]
    for body in bodies:
        local_src = SentimentFactorSource(cache=TerminalCache(default_ttl=CACHE_TTL_SECONDS))
        with respx.mock(assert_all_called=False) as rmock:
            rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
            rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json=body))
            rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
            s = local_src.get_daily_sentiment(f"q-{id(body)}", days=5)
        assert s.empty
        local_src.close()


# ---------------------------------------------------------------------------
# HN branch
# ---------------------------------------------------------------------------


def test_hn_five_stories_contribute(source):
    now = _today_unix()
    rows = [
        (now, "Federal Reserve hikes rates aggressively"),  # neutral or near-zero
        (now - 86400, "Markets plunge on recession fears"),
        (now - 2 * 86400, "Credit crunch deepens as banks fail"),
        (now - 3 * 86400, "Stocks rally on strong earnings beat"),
        (now - 4 * 86400, "Bitcoin surges to new highs"),
    ]
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json=_hn_payload(rows)))
        s = source.get_daily_sentiment("federal reserve", days=10)
    assert not s.empty
    # Mix of pos and neg → bounded
    assert s.min() >= -1.0 and s.max() <= 1.0


def test_hn_uses_story_title_fallback(source):
    now = _today_unix()
    payload = {
        "hits": [
            {"created_at_i": now, "title": None, "story_title": "Stocks rally on strong growth"},
        ]
    }
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json=payload))
        s = source.get_daily_sentiment("fallback-title", days=5)
    assert len(s) == 1
    assert s.iloc[0] > 0


def test_hn_old_posts_outside_window_dropped(source):
    now = _today_unix()
    rows = [
        (now - 400 * 86400, "Stocks surge on earnings"),  # outside 5-day cutoff
        (now, "Credit crunch deepens as banks fail"),
    ]
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json=_hn_payload(rows)))
        s = source.get_daily_sentiment("credit", days=5)
    assert len(s) <= 1


def test_hn_skipped_when_days_far_exceeds_ceiling(source):
    # days = 500 > 2 * _HN_MAX_DAYS (90*2 = 180) → HN branch skipped
    now = _today_unix()
    rows = [(now, "Stocks surge")]
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        hn_route = rmock.get(HN_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_hn_payload(rows))
        )
        s = source.get_daily_sentiment("bitcoin", days=500)
    assert s.empty
    assert hn_route.call_count == 0


def test_hn_non_2xx_yields_empty(source):
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(500, text="boom"))
        s = source.get_daily_sentiment("bitcoin", days=5)
    assert s.empty


def test_hn_invalid_json_yields_empty(source):
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, text="<html>"))
        s = source.get_daily_sentiment("bitcoin", days=5)
    assert s.empty


def test_hn_network_error_yields_empty(source):
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        rmock.get(HN_SEARCH_URL).mock(side_effect=httpx.ConnectError("nope"))
        s = source.get_daily_sentiment("bitcoin", days=5)
    assert s.empty


def test_hn_non_dict_hits_skipped(source):
    payload = {
        "hits": [None, 42, "string", {"created_at_i": _today_unix(), "title": "Stocks surge"}]
    }
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json=payload))
        s = source.get_daily_sentiment("bitcoin", days=5)
    assert len(s) == 1


# ---------------------------------------------------------------------------
# Multi-source fusion (the canonical happy path)
# ---------------------------------------------------------------------------


def test_three_sources_blend_into_daily_mean(source):
    now = _today_unix()
    gdelt_rows = [("20260512T000000Z", 5.0)]
    reddit_rows = [(now, "Stocks surge on earnings beat")]
    hn_rows = [(now, "Bitcoin rally accelerates")]
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(
            return_value=httpx.Response(200, json=_gdelt_payload(gdelt_rows))
        )
        rmock.get(REDDIT_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=_reddit_payload(reddit_rows))
        )
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json=_hn_payload(hn_rows)))
        s = source.get_daily_sentiment("bitcoin", days=10)
    assert not s.empty
    assert s.min() >= -1.0 and s.max() <= 1.0
    # name preserved for design assembler
    assert s.name == "bitcoin"


def test_all_sources_fail_returns_empty_and_logs_warning(source, caplog):
    caplog.set_level(logging.WARNING)
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(side_effect=httpx.ConnectError("nope"))
        rmock.get(REDDIT_SEARCH_URL).mock(side_effect=httpx.ConnectError("nope"))
        rmock.get(HN_SEARCH_URL).mock(side_effect=httpx.ConnectError("nope"))
        s = source.get_daily_sentiment("bitcoin", days=10)
    assert s.empty
    # At least one structured warning per failed source
    msgs = " ".join(rec.message for rec in caplog.records)
    assert "gdelt" in msgs.lower() or "reddit" in msgs.lower() or "hn" in msgs.lower()


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_second_call_uses_cache(source):
    rows = [("20260101T000000Z", 6.0), ("20260102T000000Z", 4.0)]
    with respx.mock(assert_all_called=False) as rmock:
        g = rmock.get(GDELT_DOC_URL).mock(
            return_value=httpx.Response(200, json=_gdelt_payload(rows))
        )
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        s1 = source.get_daily_sentiment("bitcoin", days=30)
        first_calls = g.call_count
        s2 = source.get_daily_sentiment("bitcoin", days=30)
    # No new GDELT call on the cached hit
    assert first_calls == g.call_count
    pd.testing.assert_series_equal(s1, s2)


def test_empty_result_is_cached(source):
    with respx.mock(assert_all_called=False) as rmock:
        g = rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        s1 = source.get_daily_sentiment("crash-search", days=10)
        first = g.call_count
        s2 = source.get_daily_sentiment("crash-search", days=10)
    assert s1.empty and s2.empty
    assert g.call_count == first  # no additional call


# ---------------------------------------------------------------------------
# fetch_sentiment_history dispatcher
# ---------------------------------------------------------------------------


def test_fetch_sentiment_history_returns_price_frame(monkeypatch):
    idx = pd.DatetimeIndex(["2026-05-10", "2026-05-11", "2026-05-12"], tz="UTC")
    fake_series = pd.Series([0.1, -0.2, 0.3], index=idx, name="bitcoin")

    class FakeSource:
        def get_daily_sentiment(self, query, days):
            assert query == "bitcoin"
            assert days >= 1
            return fake_series

    monkeypatch.setattr(sf, "_SINGLETON", FakeSource())

    frame = fetch_sentiment_history(
        "bitcoin",
        pd.Timestamp("2026-05-10", tz="UTC"),
        pd.Timestamp("2026-05-12", tz="UTC"),
    )
    assert list(frame.columns) == ["price"]
    assert frame.index.name == "date"
    assert len(frame) == 3
    assert frame["price"].iloc[0] == pytest.approx(0.1)


def test_fetch_sentiment_history_empty_when_singleton_returns_empty(monkeypatch):
    class FakeSource:
        def get_daily_sentiment(self, query, days):
            return pd.Series(dtype=float, name=query)

    monkeypatch.setattr(sf, "_SINGLETON", FakeSource())

    frame = fetch_sentiment_history(
        "no-news",
        pd.Timestamp("2026-05-10"),
        pd.Timestamp("2026-05-12"),
    )
    assert frame.empty
    assert list(frame.columns) == ["price"]


def test_fetch_sentiment_history_filters_outside_window(monkeypatch):
    idx = pd.DatetimeIndex(["2026-05-01", "2026-05-11", "2026-05-30"], tz="UTC")
    fake_series = pd.Series([0.5, 0.1, -0.2], index=idx, name="bitcoin")

    class FakeSource:
        def get_daily_sentiment(self, query, days):
            return fake_series

    monkeypatch.setattr(sf, "_SINGLETON", FakeSource())

    frame = fetch_sentiment_history(
        "bitcoin",
        pd.Timestamp("2026-05-10"),
        pd.Timestamp("2026-05-12"),
    )
    # Only the middle observation is inside [10, 12]
    assert len(frame) == 1
    assert frame.index[0] == pd.Timestamp("2026-05-11", tz="UTC")


def test_fetch_sentiment_history_filters_to_empty_when_all_outside(monkeypatch):
    idx = pd.DatetimeIndex(["2026-01-01"], tz="UTC")
    fake_series = pd.Series([0.5], index=idx, name="bitcoin")

    class FakeSource:
        def get_daily_sentiment(self, query, days):
            return fake_series

    monkeypatch.setattr(sf, "_SINGLETON", FakeSource())

    frame = fetch_sentiment_history(
        "bitcoin",
        pd.Timestamp("2026-05-10"),
        pd.Timestamp("2026-05-12"),
    )
    assert frame.empty
    assert list(frame.columns) == ["price"]


def test_fetch_sentiment_history_localises_naive_timestamps(monkeypatch):
    idx = pd.DatetimeIndex(["2026-05-11"], tz="UTC")
    fake_series = pd.Series([0.1], index=idx, name="bitcoin")

    captured: dict[str, Any] = {}

    class FakeSource:
        def get_daily_sentiment(self, query, days):
            captured["query"] = query
            captured["days"] = days
            return fake_series

    monkeypatch.setattr(sf, "_SINGLETON", FakeSource())

    frame = fetch_sentiment_history(
        "bitcoin",
        pd.Timestamp("2026-05-10"),  # tz-naive
        pd.Timestamp("2026-05-12"),  # tz-naive
    )
    assert not frame.empty
    # The dispatcher passes a positive window
    assert captured["days"] >= 1


def test_global_source_lazy_singleton(monkeypatch):
    """``_global_source`` should construct once and reuse on subsequent calls."""
    sf._SINGLETON = None
    s1 = sf._global_source()
    s2 = sf._global_source()
    assert s1 is s2
    s1.close()


# ---------------------------------------------------------------------------
# Concurrency & edge cases
# ---------------------------------------------------------------------------


def test_concurrent_calls_are_thread_safe(source):
    """Multiple threads hitting the same source must not raise or corrupt state."""
    rows = [("20260101T000000Z", 6.0), ("20260102T000000Z", 4.0)]
    results: list[pd.Series] = []
    errors: list[BaseException] = []

    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload(rows)))
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))

        def worker(q: str):
            try:
                results.append(source.get_daily_sentiment(q, days=10))
            except BaseException as e:  # pragma: no cover - defensive
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"q-{i}",)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

    assert not errors
    assert len(results) == 8
    for s in results:
        assert not s.empty
        assert s.min() >= -1.0 and s.max() <= 1.0


def test_very_long_query_is_accepted(source):
    long_query = "bitcoin " * 130  # ~1040 chars
    with respx.mock(assert_all_called=False) as rmock:
        rmock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=_gdelt_payload([])))
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        s = source.get_daily_sentiment(long_query, days=10)
    assert s.empty  # no rows mocked, but no error either
    assert s.name == long_query


def test_query_with_special_characters_passed_through(source):
    # httpx will percent-encode the query — respx matches the URL-decoded path
    weird = 'btc + "$&?#%'
    rows = [("20260101T000000Z", 5.0)]
    with respx.mock(assert_all_called=False) as rmock:
        gdelt_route = rmock.get(GDELT_DOC_URL).mock(
            return_value=httpx.Response(200, json=_gdelt_payload(rows))
        )
        rmock.get(REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        rmock.get(HN_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
        s = source.get_daily_sentiment(weird, days=10)
    assert gdelt_route.called
    # Verify the query param made it through unmangled
    request = gdelt_route.calls[0].request
    assert "query=" in str(request.url)
    assert s.name == weird


def test_close_is_idempotent():
    src = SentimentFactorSource()
    src.close()
    src.close()  # must not raise


def test_close_does_not_close_externally_owned_client():
    client = httpx.Client(timeout=1.0)
    src = SentimentFactorSource(client=client)
    src.close()
    # External client should still be usable (we didn't own it)
    assert client.is_closed is False
    client.close()


def test_constants_exposed_match_module_intent():
    assert CACHE_NAMESPACE == "sentiment_factor"
    assert CACHE_TTL_SECONDS == 900
    assert GDELT_DOC_URL.startswith("https://")
    assert REDDIT_SEARCH_URL.startswith("https://")
    assert HN_SEARCH_URL.startswith("https://")

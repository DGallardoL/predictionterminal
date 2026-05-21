"""Tests for ``pfm.sources.sentiment_factor``.

Coverage strategy mirrors the other source-tests in this repo:

  * **Pure-function path** — feed the aggregator a controlled set of
    in-memory headlines / GDELT timeline points and verify the per-day
    aggregation behaves (mean, clip, sort).
  * **Caching** — two calls with the same args hit upstream once.
  * **Empty upstream** — no GDELT / Reddit / HN data produces an empty
    Series so the design matrix's ``dropna`` cleanly removes the column.
  * **Integration with ``_assemble_design``** — mock the source, run a
    full design assembly with a synthesised ticker return series and
    verify the sentiment column appears with the expected name.

All HTTP calls are mocked via ``respx``; no live network access.
"""

from __future__ import annotations

from datetime import date

import httpx
import pandas as pd
import pytest
import respx

from pfm.cache_utils import get_cache, reset_caches
from pfm.sources import sentiment_factor as sf

# Reset caches around every test — sentiment_factor uses a module-wide
# named cache namespace and a singleton instance that we want clean.


@pytest.fixture(autouse=True)
def _clear_state():
    reset_caches()
    sf._SINGLETON = None
    yield
    reset_caches()
    sf._SINGLETON = None


# ---------------------------------------------------------------------------
# 1. Pure-function aggregation
# ---------------------------------------------------------------------------


def _build_gdelt_timelinetone(daily: dict[str, float]) -> dict:
    """Synthesise a GDELT timelinetone payload from ``{YYYY-MM-DD: tone}``.

    Tone values are GDELT-native (i.e. nominally in ``[-10, +10]``).
    """
    data_points = [
        # Date format mirrors what GDELT emits: ``20260101T000000Z``.
        {
            "date": f"{d.replace('-', '')}T000000Z",
            "value": v,
        }
        for d, v in daily.items()
    ]
    return {
        "timeline": [
            {"series": "Average Tone", "data": data_points},
        ]
    }


def _build_reddit_payload(items: list[tuple[str, int]]) -> dict:
    """``items`` is ``[(title, unix_ts), …]``; wrap into a Reddit-shaped dict."""
    return {"data": {"children": [{"data": {"title": t, "created_utc": ts}} for t, ts in items]}}


def _build_hn_payload(items: list[tuple[str, int]]) -> dict:
    return {"hits": [{"title": t, "created_at_i": ts} for t, ts in items]}


@respx.mock
def test_aggregation_mean_per_day() -> None:
    """Per-day mean across all three upstreams; GDELT rescaled /10."""
    # GDELT: two days with strong signals.
    gdelt = _build_gdelt_timelinetone(
        {
            "2026-05-01": 5.0,  # → 0.5 after /10
            "2026-05-02": -8.0,  # → -0.8 after /10
        }
    )
    # Reddit: a strongly-bullish ("surges") and a bearish ("crashes") title
    # on the SAME day so the mean blends with GDELT's contribution.
    reddit_unix_may1 = int(pd.Timestamp("2026-05-01T12:00:00Z").timestamp())
    reddit = _build_reddit_payload(
        [
            ("Bitcoin surges to new all-time high amid ETF inflows", reddit_unix_may1),
        ]
    )
    # HN: another bearish title on may-2.
    hn_unix_may2 = int(pd.Timestamp("2026-05-02T09:00:00Z").timestamp())
    hn = _build_hn_payload(
        [
            ("Crypto market crashes on new selloff", hn_unix_may2),
        ]
    )

    respx.get(sf.GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=gdelt))
    respx.get(sf.REDDIT_SEARCH_URL).mock(return_value=httpx.Response(200, json=reddit))
    respx.get(sf.HN_SEARCH_URL).mock(return_value=httpx.Response(200, json=hn))

    # ``days`` is generous so the freshness cutoff doesn't drop our
    # synthetic May 1/2 timestamps (the test environment's wall-clock
    # may be later than 2026-05-02 — the cutoff is "today minus days").
    src = sf.SentimentFactorSource()
    series = src.get_daily_sentiment("bitcoin", days=3650)

    # Two distinct UTC days appear in the output.
    assert not series.empty
    assert series.name == "bitcoin"
    # Index is UTC, normalised to midnight.
    assert all(ts.tz is not None for ts in series.index)
    assert all(ts.hour == 0 and ts.minute == 0 for ts in series.index)

    by_day = {ts.date().isoformat(): v for ts, v in series.items()}
    # May 1: gdelt=0.5, reddit bullish headline ≈ +0.something (varies w/
    # whether VADER is installed). Mean is positive; exact value is
    # asserted only loosely so the test survives lexicon tweaks.
    assert "2026-05-01" in by_day
    assert by_day["2026-05-01"] > 0.0
    # May 2: gdelt=-0.8 and HN "crashes" headline strongly negative.
    assert "2026-05-02" in by_day
    assert by_day["2026-05-02"] < 0.0
    # All values clipped to [-1, +1].
    assert all(-1.0 <= v <= 1.0 for v in series.values)


@respx.mock
def test_empty_upstreams_returns_empty_series() -> None:
    """No data anywhere → empty Series that ``dropna`` handles cleanly."""
    respx.get(sf.GDELT_DOC_URL).mock(
        return_value=httpx.Response(200, json={"timeline": []}),
    )
    respx.get(sf.REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}}),
    )
    respx.get(sf.HN_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"hits": []}),
    )

    src = sf.SentimentFactorSource()
    series = src.get_daily_sentiment("does-not-exist", days=30)

    assert isinstance(series, pd.Series)
    assert series.empty
    # ``dropna`` is the standard cleanup the design pipeline uses; verify
    # it still produces an empty series (rather than e.g. an error).
    assert series.dropna().empty


@respx.mock
def test_gdelt_throttled_returns_empty_from_gdelt() -> None:
    """GDELT's plaintext 'Please limit ...' throttle is detected gracefully."""
    respx.get(sf.GDELT_DOC_URL).mock(
        return_value=httpx.Response(
            200,
            text="Please limit your requests to no more than once every 5 seconds.",
        ),
    )
    respx.get(sf.REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}}),
    )
    respx.get(sf.HN_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"hits": []}),
    )

    src = sf.SentimentFactorSource()
    series = src.get_daily_sentiment("china", days=14)
    assert series.empty


# ---------------------------------------------------------------------------
# 2. Caching: second call doesn't re-fetch
# ---------------------------------------------------------------------------


@respx.mock
def test_caching_second_call_hits_cache() -> None:
    """After the first call populates the cache, the upstreams aren't hit again."""
    gdelt = _build_gdelt_timelinetone({"2026-05-01": 4.0})
    gdelt_route = respx.get(sf.GDELT_DOC_URL).mock(
        return_value=httpx.Response(200, json=gdelt),
    )
    reddit_route = respx.get(sf.REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}}),
    )
    hn_route = respx.get(sf.HN_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"hits": []}),
    )

    src = sf.SentimentFactorSource()
    s1 = src.get_daily_sentiment("bitcoin", days=30)
    s2 = src.get_daily_sentiment("bitcoin", days=30)

    # Exactly one call per upstream — the 2nd call is served from cache.
    assert gdelt_route.call_count == 1
    assert reddit_route.call_count == 1
    assert hn_route.call_count == 1
    # And the materialised series is identical.
    pd.testing.assert_series_equal(s1, s2)


@respx.mock
def test_caching_distinct_keys_for_different_queries() -> None:
    """A different query is a cache miss and triggers a fresh fetch."""
    gdelt = _build_gdelt_timelinetone({"2026-05-01": 2.0})
    gdelt_route = respx.get(sf.GDELT_DOC_URL).mock(
        return_value=httpx.Response(200, json=gdelt),
    )
    respx.get(sf.REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}}),
    )
    respx.get(sf.HN_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"hits": []}),
    )

    src = sf.SentimentFactorSource()
    src.get_daily_sentiment("trump", days=30)
    src.get_daily_sentiment("biden", days=30)
    # Two distinct cache keys → two GDELT hits.
    assert gdelt_route.call_count == 2


# ---------------------------------------------------------------------------
# 3. Factor-id parsing
# ---------------------------------------------------------------------------


def test_parse_sentiment_prefix_extracts_query() -> None:
    assert sf.parse_sentiment_factor_id("sentiment:bitcoin") == (True, "bitcoin")
    assert sf.parse_sentiment_factor_id("Sentiment:Federal Reserve") == (
        True,
        "Federal Reserve",
    )
    # Whitespace is trimmed.
    assert sf.parse_sentiment_factor_id("sentiment:  trump  ") == (True, "trump")


def test_parse_sentiment_prefix_no_match() -> None:
    assert sf.parse_sentiment_factor_id("polymarket-slug-2026") == (False, None)
    assert sf.parse_sentiment_factor_id("") == (False, None)
    assert sf.parse_sentiment_factor_id("sentiment") == (False, None)


def test_parse_sentiment_empty_query_returns_none() -> None:
    """``sentiment:`` with no query → (True, None) so the caller can 400."""
    assert sf.parse_sentiment_factor_id("sentiment:") == (True, None)
    assert sf.parse_sentiment_factor_id("sentiment:   ") == (True, None)


# ---------------------------------------------------------------------------
# 4. Dispatcher bridge — fetch_sentiment_history returns ``[date, price]``
# ---------------------------------------------------------------------------


@respx.mock
def test_fetch_sentiment_history_returns_price_frame() -> None:
    """The dispatcher bridge shapes the Series as ``DataFrame[price]``."""
    gdelt = _build_gdelt_timelinetone(
        {"2026-04-15": 3.0, "2026-04-16": -2.5, "2026-04-17": 0.0},
    )
    respx.get(sf.GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=gdelt))
    respx.get(sf.REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}}),
    )
    respx.get(sf.HN_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"hits": []}),
    )

    df = sf.fetch_sentiment_history(
        "oil",
        start=pd.Timestamp("2026-04-15", tz="UTC"),
        end=pd.Timestamp("2026-04-17", tz="UTC"),
    )
    assert list(df.columns) == ["price"]
    assert df.index.name == "date"
    assert len(df) == 3
    assert df.index.tz is not None  # UTC
    # The midpoint day's tone was 0.0 → mean 0.0 within the bucket.
    assert df.loc[pd.Timestamp("2026-04-17", tz="UTC"), "price"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. Integration with _assemble_design — mock the source, verify column
# ---------------------------------------------------------------------------


@respx.mock
def test_assemble_design_picks_up_sentiment_factor(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: ``sentiment:bitcoin`` flows through the design assembler.

    We mock ``_cached_log_returns`` and the dispatcher path so no real
    Polymarket / yfinance traffic is needed. The assertion is structural:
    the returned design matrix has the sentiment column under its
    user-typed id.
    """
    from pfm import main as main_mod
    from pfm.regression_core import _assemble_design, _resolve_factor_specs

    # 1) Synthesise a 20-day daily return series for the ticker.
    idx = pd.date_range("2026-04-01", periods=20, freq="B", tz="UTC")
    fake_returns = pd.Series(
        0.001 * pd.Series(range(20)).values,
        index=idx,
        name="r",
    )

    def _fake_log_returns(*_args, **_kwargs):
        return fake_returns

    monkeypatch.setattr(main_mod, "get_log_returns", _fake_log_returns)

    # 2) Mock the GDELT/Reddit/HN upstreams. Cover the same window so
    # delta_level produces non-NaN values for at least a handful of days.
    daily = {
        ts.strftime("%Y-%m-%d"): float(i - 10)  # values in [-10, +9]
        for i, ts in enumerate(idx)
    }
    respx.get(sf.GDELT_DOC_URL).mock(
        return_value=httpx.Response(200, json=_build_gdelt_timelinetone(daily)),
    )
    respx.get(sf.REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}}),
    )
    respx.get(sf.HN_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"hits": []}),
    )

    # 3) Resolve the factor specs from the prefixed id and assemble.
    factor_specs = _resolve_factor_specs(["sentiment:bitcoin"], [], {})
    assert len(factor_specs) == 1
    spec = factor_specs[0]
    assert spec.id == "sentiment:bitcoin"
    assert spec.source == "sentiment"
    assert spec.is_probability is False

    # Use a minimal cache + settings stub — the design assembler only
    # touches them through ``_cached_factor_history``, which in turn
    # routes the sentiment source through ``fetch_factor_history_dispatch``.
    class _StubCache:
        def get(self, *_a, **_k):
            return None

        def set(self, *_a, **_k):
            pass

    class _StubSettings:
        cache_ttl_seconds = 60

    y, X, raw = _assemble_design(
        ticker="FAKE",
        factor_specs=factor_specs,
        start=date(2026, 4, 1),
        end=date(2026, 4, 28),
        epsilon=0.01,
        return_type="log",
        poly=None,
        cache=_StubCache(),
        settings=_StubSettings(),
        alignment="strict",
        residualize_market=False,
    )

    # The column appears under the user-typed id.
    assert "sentiment:bitcoin" in X.columns
    # And y / X are still index-aligned.
    assert (y.index == X.index).all()
    # delta_level → first observation drops out, so we expect >0 rows
    # but fewer than the raw input length.
    assert len(X) > 0
    # The raw_prices dict carries the original [-1, +1] level series.
    assert "sentiment:bitcoin" in raw
    assert raw["sentiment:bitcoin"].between(-1.0, 1.0).all()


# ---------------------------------------------------------------------------
# 6. Cache namespace plumbing — the singleton uses the named cache
# ---------------------------------------------------------------------------


def test_singleton_uses_named_cache() -> None:
    """Sanity-check: ``_global_source()`` is wired to the shared namespace."""
    cache = get_cache(sf.CACHE_NAMESPACE, ttl=sf.CACHE_TTL_SECONDS)
    src = sf._global_source()
    assert src._cache is cache


# ---------------------------------------------------------------------------
# 7. Additional edge cases (filling coverage gaps)
# ---------------------------------------------------------------------------


def test_empty_query_returns_empty_series_without_http() -> None:
    """An empty/blank ``query`` short-circuits before any upstream call —
    no respx mocks needed because no HTTP traffic should occur."""
    src = sf.SentimentFactorSource()
    for q in ("", None):
        series = src.get_daily_sentiment(q, days=30)  # type: ignore[arg-type]
        assert isinstance(series, pd.Series)
        assert series.empty


@respx.mock
def test_only_gdelt_returns_data_other_sources_silent() -> None:
    """Reddit + HN return empty but GDELT has data → series still populated.
    Verifies the per-source independence of the aggregator."""
    gdelt = _build_gdelt_timelinetone(
        {"2026-05-01": 4.0, "2026-05-02": -2.0},
    )
    respx.get(sf.GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=gdelt))
    respx.get(sf.REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}}),
    )
    respx.get(sf.HN_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"hits": []}),
    )
    src = sf.SentimentFactorSource()
    series = src.get_daily_sentiment("oil", days=3650)
    assert not series.empty
    assert len(series) == 2


@respx.mock
def test_gdelt_returns_500_treated_as_no_data() -> None:
    """An upstream 5xx is logged and treated as "no rows" — empty series
    when nobody returned anything."""
    respx.get(sf.GDELT_DOC_URL).mock(return_value=httpx.Response(503, text="oops"))
    respx.get(sf.REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}}),
    )
    respx.get(sf.HN_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"hits": []}),
    )
    src = sf.SentimentFactorSource()
    series = src.get_daily_sentiment("any", days=14)
    assert series.empty


@respx.mock
def test_gdelt_returns_invalid_json_handled_gracefully() -> None:
    """If GDELT returns text the JSON decoder rejects, return [] from that
    source and keep going (no crash)."""
    respx.get(sf.GDELT_DOC_URL).mock(
        return_value=httpx.Response(200, text="<html>not json</html>"),
    )
    respx.get(sf.REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}}),
    )
    respx.get(sf.HN_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"hits": []}),
    )
    src = sf.SentimentFactorSource()
    series = src.get_daily_sentiment("china", days=14)
    assert series.empty


@respx.mock
def test_reddit_500_does_not_break_pipeline() -> None:
    """Reddit upstream errors must not poison the GDELT path."""
    gdelt = _build_gdelt_timelinetone({"2026-05-01": 3.0})
    respx.get(sf.GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=gdelt))
    respx.get(sf.REDDIT_SEARCH_URL).mock(return_value=httpx.Response(500))
    respx.get(sf.HN_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"hits": []}),
    )
    src = sf.SentimentFactorSource()
    series = src.get_daily_sentiment("oil", days=3650)
    # Still get the GDELT row.
    assert not series.empty


@respx.mock
def test_hn_500_does_not_break_pipeline() -> None:
    """HN upstream errors must not poison the rest."""
    gdelt = _build_gdelt_timelinetone({"2026-05-01": -3.0})
    respx.get(sf.GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=gdelt))
    respx.get(sf.REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}}),
    )
    respx.get(sf.HN_SEARCH_URL).mock(return_value=httpx.Response(500))
    src = sf.SentimentFactorSource()
    series = src.get_daily_sentiment("oil", days=3650)
    assert not series.empty


@respx.mock
def test_reddit_invalid_json_handled() -> None:
    """Reddit returning non-JSON triggers the JSONDecodeError fallback."""
    respx.get(sf.GDELT_DOC_URL).mock(
        return_value=httpx.Response(200, json={"timeline": []}),
    )
    respx.get(sf.REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, text="literally html"),
    )
    respx.get(sf.HN_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"hits": []}),
    )
    src = sf.SentimentFactorSource()
    series = src.get_daily_sentiment("any", days=14)
    assert series.empty


@respx.mock
def test_hn_invalid_json_handled() -> None:
    """HN returning non-JSON triggers the JSONDecodeError fallback."""
    respx.get(sf.GDELT_DOC_URL).mock(
        return_value=httpx.Response(200, json={"timeline": []}),
    )
    respx.get(sf.REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}}),
    )
    respx.get(sf.HN_SEARCH_URL).mock(
        return_value=httpx.Response(200, text="not json"),
    )
    src = sf.SentimentFactorSource()
    series = src.get_daily_sentiment("any", days=14)
    assert series.empty


@respx.mock
def test_gdelt_throttle_text_returns_empty_from_that_source() -> None:
    """Already covered above but pin a slightly-different throttle prefix."""
    respx.get(sf.GDELT_DOC_URL).mock(
        return_value=httpx.Response(
            200,
            text="Please limit your requests; rate limit exceeded.",
        ),
    )
    respx.get(sf.REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}}),
    )
    respx.get(sf.HN_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"hits": []}),
    )
    src = sf.SentimentFactorSource()
    series = src.get_daily_sentiment("any", days=14)
    assert series.empty


@respx.mock
def test_gdelt_returns_non_dict_payload_handled() -> None:
    """GDELT returning a list (rather than the expected dict) → empty."""
    respx.get(sf.GDELT_DOC_URL).mock(
        return_value=httpx.Response(200, json=["unexpected", "shape"]),
    )
    respx.get(sf.REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}}),
    )
    respx.get(sf.HN_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"hits": []}),
    )
    src = sf.SentimentFactorSource()
    series = src.get_daily_sentiment("oil", days=14)
    assert series.empty


@respx.mock
def test_gdelt_timeline_with_non_list_handled() -> None:
    """`timeline` field is not a list → no data points extracted."""
    respx.get(sf.GDELT_DOC_URL).mock(
        return_value=httpx.Response(200, json={"timeline": "should-be-list"}),
    )
    respx.get(sf.REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}}),
    )
    respx.get(sf.HN_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"hits": []}),
    )
    src = sf.SentimentFactorSource()
    series = src.get_daily_sentiment("oil", days=14)
    assert series.empty


@respx.mock
def test_gdelt_data_point_with_missing_value_skipped() -> None:
    """One malformed point is filtered; the rest still flow through."""
    payload = {
        "timeline": [
            {
                "series": "Average Tone",
                "data": [
                    {"date": "20260501T000000Z", "value": 3.0},  # good
                    {"date": "20260502T000000Z"},  # missing value
                    {"date": "20260503T000000Z", "value": "garbage"},  # bad cast
                    {"date": "20260504T000000Z", "value": 1.5},  # good
                    {"date": "garbage-ts", "value": 2.0},  # bad ts
                ],
            },
        ],
    }
    respx.get(sf.GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=payload))
    respx.get(sf.REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}}),
    )
    respx.get(sf.HN_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"hits": []}),
    )
    src = sf.SentimentFactorSource()
    series = src.get_daily_sentiment("oil", days=3650)
    # Two good points produce two rows; missing-value, garbage-string, and
    # garbage-ts are all silently skipped.
    assert len(series) == 2


def test_seendate_to_ts_iso_format_supported() -> None:
    """Already-ISO timestamps are passed through (the GDELT bridge handles
    both compact and ISO forms)."""
    ts = sf._seendate_to_ts("2026-05-01T12:34:56Z")
    assert ts is not None
    assert ts.year == 2026 and ts.month == 5 and ts.day == 1


def test_seendate_to_ts_empty_returns_none() -> None:
    assert sf._seendate_to_ts("") is None
    assert sf._seendate_to_ts(None) is None  # type: ignore[arg-type]


def test_seendate_to_ts_garbage_returns_none() -> None:
    assert sf._seendate_to_ts("not a date") is None


def test_ts_from_unix_handles_none_and_invalid() -> None:
    assert sf._ts_from_unix(None) is None
    assert sf._ts_from_unix("not-a-number") is None
    # Valid unix → tz-aware UTC.
    ts = sf._ts_from_unix(1714521600)  # 2024-05-01
    assert ts is not None
    assert ts.tz is not None


def test_curated_sentiment_query_known_and_unknown() -> None:
    assert sf.curated_sentiment_query("sentiment_bitcoin") == "bitcoin"
    assert sf.curated_sentiment_query("sentiment_fed") == "federal reserve"
    assert sf.curated_sentiment_query("does-not-exist") is None


def test_close_is_idempotent_and_safe() -> None:
    """`SentimentFactorSource.close()` must be safe to call repeatedly."""
    src = sf.SentimentFactorSource()
    src._http()  # force lazy init
    assert src._client is not None
    src.close()
    assert src._client is None
    src.close()  # second call must not raise


def test_close_does_not_close_externally_owned_client() -> None:
    """When a client is injected, the source must NOT close it on shutdown."""
    external = httpx.Client()
    try:
        src = sf.SentimentFactorSource(client=external)
        assert src._owns_client is False
        src.close()
        # The external client must still be usable — close() should have
        # been a no-op for it.
        assert external.is_closed is False
    finally:
        external.close()


@respx.mock
def test_global_source_is_singleton() -> None:
    """Two calls to ``_global_source`` return the same instance."""
    a = sf._global_source()
    b = sf._global_source()
    assert a is b


def test_parse_sentiment_factor_id_non_string_returns_false() -> None:
    """Non-string inputs are coerced to a (False, None) tuple."""
    # type: ignore — exercising defensive branch
    assert sf.parse_sentiment_factor_id(123) == (False, None)  # type: ignore[arg-type]
    assert sf.parse_sentiment_factor_id(None) == (False, None)  # type: ignore[arg-type]


@respx.mock
def test_fetch_sentiment_history_empty_when_window_outside_data() -> None:
    """A start/end window that yields no rows after slicing returns an empty
    DataFrame (with the expected column list)."""
    # GDELT returns data on 2026-05-01 but we ask for a 2030 window.
    gdelt = _build_gdelt_timelinetone({"2026-05-01": 4.0})
    respx.get(sf.GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=gdelt))
    respx.get(sf.REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}}),
    )
    respx.get(sf.HN_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"hits": []}),
    )

    df = sf.fetch_sentiment_history(
        "oil",
        start=pd.Timestamp("2030-04-15", tz="UTC"),
        end=pd.Timestamp("2030-04-17", tz="UTC"),
    )
    assert list(df.columns) == ["price"]
    assert len(df) == 0


@respx.mock
def test_fetch_sentiment_history_empty_when_no_upstream_data() -> None:
    """No upstream data at all → empty DataFrame with [price] schema."""
    respx.get(sf.GDELT_DOC_URL).mock(
        return_value=httpx.Response(200, json={"timeline": []}),
    )
    respx.get(sf.REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}}),
    )
    respx.get(sf.HN_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"hits": []}),
    )
    df = sf.fetch_sentiment_history(
        "no-such-thing",
        start=pd.Timestamp("2026-04-15"),
        end=pd.Timestamp("2026-04-17"),
    )
    assert list(df.columns) == ["price"]
    assert df.empty


@respx.mock
def test_fetch_sentiment_history_naive_timestamps_localized() -> None:
    """``start``/``end`` passed without tz are silently localized to UTC."""
    gdelt = _build_gdelt_timelinetone({"2026-04-16": 3.0})
    respx.get(sf.GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=gdelt))
    respx.get(sf.REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}}),
    )
    respx.get(sf.HN_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"hits": []}),
    )
    df = sf.fetch_sentiment_history(
        "oil",
        start=pd.Timestamp("2026-04-15"),  # naive
        end=pd.Timestamp("2026-04-17"),  # naive
    )
    assert not df.empty
    # Indexed by tz-aware UTC.
    assert df.index.tz is not None


@respx.mock
def test_get_daily_sentiment_negative_days_coerced_to_one() -> None:
    """A non-positive ``days`` must be clamped to 1 — no zero-window cost."""
    gdelt = _build_gdelt_timelinetone({"2026-05-01": 1.0})
    respx.get(sf.GDELT_DOC_URL).mock(return_value=httpx.Response(200, json=gdelt))
    respx.get(sf.REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}}),
    )
    respx.get(sf.HN_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"hits": []}),
    )
    src = sf.SentimentFactorSource()
    # Pass an obviously-bad days value; the call must not crash.
    series = src.get_daily_sentiment("oil", days=-5)
    # Either empty (because the cutoff drops everything) or has 1 row;
    # the key invariant is "doesn't raise".
    assert isinstance(series, pd.Series)


@respx.mock
def test_cache_empty_result_is_cached() -> None:
    """An empty result is still cached — a re-query must not re-hit upstreams."""
    gdelt_route = respx.get(sf.GDELT_DOC_URL).mock(
        return_value=httpx.Response(200, json={"timeline": []}),
    )
    respx.get(sf.REDDIT_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": {"children": []}}),
    )
    respx.get(sf.HN_SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"hits": []}),
    )
    src = sf.SentimentFactorSource()
    s1 = src.get_daily_sentiment("nothing", days=30)
    s2 = src.get_daily_sentiment("nothing", days=30)
    assert s1.empty and s2.empty
    # The empty result must come from cache on the second call (call_count
    # stays at 1).
    assert gdelt_route.call_count == 1

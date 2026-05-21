"""Tests for ``pfm.terminal_sentiment_trend``. All HTTP is mocked via respx."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pandas as pd
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.sources.polymarket import PolymarketClient
from pfm.terminal_sentiment_trend import (
    _CACHE,
    _best_lag_correlation,
    _pearson,
)
from pfm.terminal_sentiment_trend import router as sentiment_trend_router

GAMMA = "https://gamma-api.test"
CLOB = "https://clob.test"
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    _CACHE.clear()
    yield
    _CACHE.clear()


@pytest.fixture
def app_client() -> TestClient:
    app = FastAPI()
    app.state.poly = PolymarketClient(gamma_url=GAMMA, clob_url=CLOB, client=httpx.Client())
    app.include_router(sentiment_trend_router)
    return TestClient(app)


def _iso(d: datetime) -> str:
    return d.strftime("%Y%m%dT%H%M%SZ")


def _gdelt_articles_recent(now: datetime, n_days: int = 7) -> list[dict]:
    """Build a GDELT-style article list with one article per day for n_days back."""
    arts: list[dict] = []
    for offset in range(n_days):
        day = now - timedelta(days=offset)
        # Linear ramp -3 → +3 across days; alternate sources.
        tone = -3.0 + (n_days - 1 - offset) * (6.0 / max(n_days - 1, 1))
        arts.append(
            {
                "url": f"https://www.bbc.com/news/x{offset}",
                "title": f"Trump policy update day {offset}",
                "domain": "bbc.com",
                "sourcecountry": "United Kingdom",
                "language": "English",
                "seendate": _iso(day),
                "socialimage": "",
                "tone": tone,
            }
        )
    return arts


def _clob_price_history(now: datetime, n_days: int = 30) -> dict:
    """Build a CLOB ``/prices-history`` payload: one daily print per day."""
    history = []
    for offset in range(n_days, 0, -1):
        d = now - timedelta(days=offset)
        # Price inversely correlated with tone via a ramp; we don't need
        # tight dependence — just a non-trivial series.
        p = 0.30 + (n_days - offset) * (0.40 / max(n_days - 1, 1))
        history.append({"t": int(d.replace(tzinfo=UTC).timestamp()), "p": p})
    return {"history": history}


# --- pure helpers -----------------------------------------------------------


def test_pearson_and_best_lag_recover_known_offset() -> None:
    """Construct tone[t] = price[t-2] (i.e. tone leads price by k=+2)."""
    tone = [0.1, 0.4, 0.9, 1.6, 2.5, 3.6, 4.9, 6.4, 8.1, 10.0]
    # price[t] = tone[t-2] padded with leading zeros — i.e. tone leads price by 2.
    price = [0.0, 0.0, 0.1, 0.4, 0.9, 1.6, 2.5, 3.6, 4.9, 6.4]

    r0 = _pearson(tone, price)
    assert -1.0 <= r0 <= 1.0

    corr, lag = _best_lag_correlation(tone, price, max_lag=5)
    assert lag == 2
    assert corr > 0.99


# --- /sentiment-trend/{slug} ------------------------------------------------


@respx.mock
def test_sentiment_trend_per_slug_full_pipeline(app_client: TestClient) -> None:
    """Happy path: gamma + GDELT + CLOB all return data → assemble full response."""
    now = datetime.now(tz=UTC)

    respx.get(f"{GAMMA}/markets", params={"slug": "trump-out-by-2027"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "trump-out-by-2027",
                    "question": "Will Trump resign or be impeached by 2027?",
                    "clobTokenIds": '["111", "222"]',
                    "closed": False,
                    "active": True,
                }
            ],
        )
    )
    respx.get(GDELT_DOC_URL).mock(
        return_value=httpx.Response(200, json={"articles": _gdelt_articles_recent(now, n_days=7)})
    )
    respx.get(f"{CLOB}/prices-history").mock(
        return_value=httpx.Response(200, json=_clob_price_history(now, n_days=30))
    )

    resp = app_client.get("/terminal/sentiment-trend/trump-out-by-2027?days=30")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["slug"] == "trump-out-by-2027"
    # Tone series must have exactly `days` entries (dense daily buckets).
    assert len(body["tone_series"]) == 30
    # Each entry has the right schema.
    for pt in body["tone_series"]:
        assert set(pt.keys()) == {"date", "mean_tone", "n_articles", "dominant_topic"}
        assert -10.0 <= pt["mean_tone"] <= 10.0
        assert pt["n_articles"] >= 0

    # Regime is one of three known labels.
    assert body["sentiment_regime"] in {"bullish", "bearish", "neutral"}

    # Lag in [-7, +7].
    assert -7 <= body["lead_lag_days"] <= 7
    assert -1.0 <= body["correlation_with_price"] <= 1.0
    assert isinstance(body["interpretation"], str)
    assert body["interpretation"]


@respx.mock
def test_sentiment_trend_404_when_market_missing(app_client: TestClient) -> None:
    respx.get(f"{GAMMA}/markets", params={"slug": "ghost"}).mock(
        return_value=httpx.Response(200, json=[])
    )
    gdelt_route = respx.get(GDELT_DOC_URL).mock(
        return_value=httpx.Response(200, json={"articles": []})
    )
    resp = app_client.get("/terminal/sentiment-trend/ghost")
    assert resp.status_code == 404
    assert "no market" in resp.json()["detail"]
    assert gdelt_route.call_count == 0


@respx.mock
def test_sentiment_trend_handles_empty_gdelt_gracefully(app_client: TestClient) -> None:
    """No articles → regime=neutral, corr=0, lag=0, interpretation explains it."""
    now = datetime.now(tz=UTC)
    respx.get(f"{GAMMA}/markets", params={"slug": "btc-200k"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "btc-200k",
                    "question": "Will Bitcoin reach 200k by end of year?",
                    "clobTokenIds": '["aaa", "bbb"]',
                    "closed": False,
                    "active": True,
                }
            ],
        )
    )
    # GDELT throttle plaintext.
    respx.get(GDELT_DOC_URL).mock(
        return_value=httpx.Response(200, text="Please limit requests to one every 5 seconds...")
    )
    respx.get(f"{CLOB}/prices-history").mock(
        return_value=httpx.Response(200, json=_clob_price_history(now, n_days=15))
    )

    resp = app_client.get("/terminal/sentiment-trend/btc-200k?days=10")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["slug"] == "btc-200k"
    assert body["current_tone"] == 0.0
    # All buckets empty.
    assert all(p["n_articles"] == 0 for p in body["tone_series"])
    assert body["sentiment_regime"] == "neutral"
    assert body["correlation_with_price"] == 0.0
    assert body["lead_lag_days"] == 0
    assert "No news coverage" in body["interpretation"]


# --- /sentiment-trend/spike-alerts ------------------------------------------


@respx.mock
def test_sentiment_trend_per_day_uses_timelinetone_overlay(
    app_client: TestClient,
) -> None:
    """Production case: artlist returns articles WITHOUT tone, but timelinetone
    provides daily mean-tone samples. Every day in the returned ``tone_series``
    must carry a non-zero value pulled from (or forward/back-filled from) the
    timeline — not a flat 0.0 — and ``degraded_mode`` must stay False.
    """
    now = datetime.now(tz=UTC)

    respx.get(f"{GAMMA}/markets", params={"slug": "fed-cut-2026"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "fed-cut-2026",
                    "question": "Will the Fed cut rates in 2026?",
                    "clobTokenIds": '["t1", "t2"]',
                    "closed": False,
                    "active": True,
                }
            ],
        )
    )

    # Artlist payload: 5 articles WITHOUT a tone field (production reality —
    # GDELT artlist does not include per-article tone).
    arts: list[dict] = []
    for offset in range(5):
        day = now - timedelta(days=offset)
        arts.append(
            {
                "url": f"https://reuters.com/x{offset}",
                "title": f"Fed policy update day {offset}",
                "domain": "reuters.com",
                "sourcecountry": "United States",
                "language": "English",
                "seendate": _iso(day),
                "socialimage": "",
                # NO "tone" key — matches real GDELT artlist response.
            }
        )

    # Timelinetone payload: sparse hourly samples — covers only some days
    # in the 7-day window, mimicking a niche-query GDELT response. We expect
    # the implementation to forward-fill (and back-fill leading gaps) so
    # every per-day point carries a real tone number.
    timeline_data = [
        # 5 days ago: oldest sample → back-fills 6-days-ago and 7-days-ago
        {"date": _iso(now - timedelta(days=5)), "value": -1.5},
        # 2 days ago: jump
        {"date": _iso(now - timedelta(days=2)), "value": 2.0},
        # today
        {"date": _iso(now), "value": 4.0},
    ]

    def _gdelt_route(request: httpx.Request) -> httpx.Response:
        if "timelinetone" in str(request.url):
            return httpx.Response(
                200,
                json={"timeline": [{"series": "Average Tone", "data": timeline_data}]},
            )
        return httpx.Response(200, json={"articles": arts})

    respx.get(GDELT_DOC_URL).mock(side_effect=_gdelt_route)
    respx.get(f"{CLOB}/prices-history").mock(return_value=httpx.Response(200, json={"history": []}))

    resp = app_client.get("/terminal/sentiment-trend/fed-cut-2026?days=7")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # The aggregate (last-24h) already worked from the previous fix.
    assert body["current_tone"] == pytest.approx(4.0)
    # Degraded mode MUST be False — we have a real tone signal from the overlay.
    assert body["degraded_mode"] is False
    assert body["reason"] is None

    # Every day must carry a non-zero tone (the actual bug fix). Without
    # the forward/back-fill, days outside {5d-ago, 2d-ago, today} would be 0.0.
    series = body["tone_series"]
    assert len(series) == 7
    tones = [p["mean_tone"] for p in series]
    assert all(t != 0.0 for t in tones), f"per-day tones should be non-zero, got {tones}"

    # Spot-check the fill behaviour (series is oldest → newest, index 6 = today):
    # - Index 0 is 6 days ago (older than the first sample at 5d-ago)
    #   → back-filled from the first sample (-1.5).
    # - Index 1 is 5 days ago → exact sample (-1.5).
    # - Indices 2, 3 are 4/3 days ago → forward-filled from -1.5.
    # - Index 4 is 2 days ago → exact sample (2.0).
    # - Index 5 is 1 day ago → forward-filled from 2.0.
    # - Index 6 is today → exact sample (4.0).
    assert tones[0] == pytest.approx(-1.5)
    assert tones[1] == pytest.approx(-1.5)
    assert tones[2] == pytest.approx(-1.5)
    assert tones[3] == pytest.approx(-1.5)
    assert tones[4] == pytest.approx(2.0)
    assert tones[5] == pytest.approx(2.0)
    assert tones[6] == pytest.approx(4.0)


@respx.mock
def test_sentiment_trend_degrades_when_timelinetone_empty(
    app_client: TestClient,
) -> None:
    """When both artlist tone AND timelinetone are empty, we must surface
    ``degraded_mode=True`` rather than silently returning zeros."""
    now = datetime.now(tz=UTC)

    respx.get(f"{GAMMA}/markets", params={"slug": "obscure-market"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "obscure-market",
                    "question": "Will an obscure thing happen?",
                    "clobTokenIds": '["a", "b"]',
                    "closed": False,
                    "active": True,
                }
            ],
        )
    )

    # Artlist returns articles without tone; timelinetone returns empty payload.
    arts = [
        {
            "url": "https://x.com/y",
            "title": "Obscure update",
            "domain": "x.com",
            "sourcecountry": "United States",
            "language": "English",
            "seendate": _iso(now),
            "socialimage": "",
        }
    ]

    def _gdelt_route(request: httpx.Request) -> httpx.Response:
        if "timelinetone" in str(request.url):
            return httpx.Response(200, json={"timeline": []})
        return httpx.Response(200, json={"articles": arts})

    respx.get(GDELT_DOC_URL).mock(side_effect=_gdelt_route)
    respx.get(f"{CLOB}/prices-history").mock(return_value=httpx.Response(200, json={"history": []}))

    resp = app_client.get("/terminal/sentiment-trend/obscure-market?days=5")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # No usable tone anywhere → degraded mode true, all per-day tones 0.0.
    assert body["degraded_mode"] is True
    assert body["reason"] is not None
    assert all(p["mean_tone"] == 0.0 for p in body["tone_series"])


@respx.mock
def test_spike_alerts_flags_large_tone_shift(app_client: TestClient) -> None:
    """One discovered market whose tone shifts -4 → +4 across the window → 1 alert."""
    now = datetime.now(tz=UTC)

    # discover_markets walks pages of /markets ordered by volumeNum desc.
    discover_payload = [
        {
            "slug": "fed-cut-march",
            "question": "Will the Fed cut rates in March?",
            "volume": 5_000_000,
            "endDate": "2026-03-31T00:00:00Z",
            "active": True,
            "closed": False,
        }
    ]
    respx.get(f"{GAMMA}/markets").mock(return_value=httpx.Response(200, json=discover_payload))

    # Build a 7-day window where day 0..2 have tone ≈ -4 and day 4..6 have tone ≈ +4
    arts: list[dict] = []
    for offset in range(7):
        day = now - timedelta(days=offset)
        # offset 0 = today (latest half), offset 6 = oldest (first half).
        tone = -4.0 if offset >= 4 else 4.0
        arts.append(
            {
                "url": f"https://www.cnn.com/x{offset}",
                "title": f"Fed news day {offset}",
                "domain": "cnn.com",
                "sourcecountry": "United States",
                "language": "English",
                "seendate": _iso(day),
                "socialimage": "",
                "tone": tone,
            }
        )
    respx.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json={"articles": arts}))

    resp = app_client.get("/terminal/sentiment-trend/spike-alerts?days=7&min_n_articles=3")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["days"] == 7
    assert body["min_n_articles"] == 3
    assert body["n_alerts"] >= 1
    alert = body["alerts"][0]
    assert alert["slug"] == "fed-cut-march"
    assert abs(alert["tone_shift"]) > 3.0
    assert alert["direction"] in {"up", "down"}
    assert alert["n_articles"] >= 3


# ---------------------------------------------------------------------------
# Cache layering: L1 + Redis L2 (cross-worker)
# ---------------------------------------------------------------------------


class _FakeRedisL2:
    """Tiny stand-in for the Redis cache wrapper."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.enabled = True

    def get(self, key: str) -> bytes | None:
        return self.store.get(key)

    def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        self.store[key] = value


@respx.mock
def test_sentiment_trend_l2_redis_short_circuits_upstreams(
    app_client: TestClient,
) -> None:
    """A seeded L2 entry must serve the request without any upstream traffic.

    Verifies cross-worker promotion: worker A wrote to Redis on its
    cold-fill, worker B inherits the payload via the L2 read instead of
    paying the triple GDELT+CLOB round-trip again.
    """
    import json as _json

    fake_l2 = _FakeRedisL2()
    seeded = {
        "slug": "trump-out-by-2027",
        "current_tone": 1.5,
        "tone_series": [],
        "sentiment_regime": "neutral",
        "correlation_with_price": 0.3,
        "lead_lag_days": 0,
        "interpretation": "seeded-from-L2",
        "degraded_mode": False,
        "reason": None,
    }
    fake_l2.store["terminal_sentiment_trend:trend:trump-out-by-2027:30"] = _json.dumps(
        seeded
    ).encode("utf-8")
    app_client.app.state.cache = fake_l2

    gamma_route = respx.get(f"{GAMMA}/markets").mock(
        return_value=httpx.Response(500, json={"error": "should-not-be-called"})
    )
    gdelt_route = respx.get(GDELT_DOC_URL).mock(
        return_value=httpx.Response(500, json={"error": "should-not-be-called"})
    )
    clob_route = respx.get(f"{CLOB}/prices-history").mock(
        return_value=httpx.Response(500, json={"error": "should-not-be-called"})
    )

    resp = app_client.get("/terminal/sentiment-trend/trump-out-by-2027?days=30")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["interpretation"] == "seeded-from-L2"
    # Zero upstream traffic.
    assert gamma_route.call_count == 0
    assert gdelt_route.call_count == 0
    assert clob_route.call_count == 0


# ---------------------------------------------------------------------------
# Hybrid NLP fallback: headline scoring when GDELT tone signal is absent
# ---------------------------------------------------------------------------


@respx.mock
def test_sentiment_trend_falls_back_to_headline_nlp(app_client: TestClient) -> None:
    """When artlist has no tone AND timelinetone is empty, but headlines
    carry sentiment, the per-day tone series must show real non-zero values
    via the hybrid VADER + financial-lex headline scorer fallback.
    """
    now = datetime.now(tz=UTC)

    respx.get(f"{GAMMA}/markets", params={"slug": "nvda-earnings"}).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "slug": "nvda-earnings",
                    "question": "Will NVDA beat earnings?",
                    "clobTokenIds": '["t1", "t2"]',
                    "closed": False,
                    "active": True,
                }
            ],
        )
    )

    # Articles with sentiment-rich headlines but NO tone field. We pick
    # finance-lexicon hits so ``score_headline`` is guaranteed non-zero.
    headlines_by_offset: dict[int, str] = {
        0: "NVDA stock surges to record high on AI rally",
        1: "Analysts upgrade NVDA after strong outlook",
        3: "NVDA shares crash on regulatory concern",
        4: "Selloff continues as NVDA plummets further",
    }
    arts: list[dict] = []
    for offset, title in headlines_by_offset.items():
        day = now - timedelta(days=offset)
        arts.append(
            {
                "url": f"https://reuters.com/nvda{offset}",
                "title": title,
                "domain": "reuters.com",
                "sourcecountry": "United States",
                "language": "English",
                "seendate": _iso(day),
                "socialimage": "",
                # NO tone key.
            }
        )

    def _gdelt_route(request: httpx.Request) -> httpx.Response:
        if "timelinetone" in str(request.url):
            # Empty timeline → forces the headline-NLP fallback path.
            return httpx.Response(200, json={"timeline": []})
        return httpx.Response(200, json={"articles": arts})

    respx.get(GDELT_DOC_URL).mock(side_effect=_gdelt_route)
    respx.get(f"{CLOB}/prices-history").mock(return_value=httpx.Response(200, json={"history": []}))

    resp = app_client.get("/terminal/sentiment-trend/nvda-earnings?days=7")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # With non-neutral headlines we now have real signal — must NOT degrade.
    assert body["degraded_mode"] is False
    # At least one per-day point must be non-zero (the fallback bug fix).
    tones = [p["mean_tone"] for p in body["tone_series"]]
    assert any(t != 0.0 for t in tones), (
        f"expected at least one non-zero per-day tone from headline NLP fallback, got {tones}"
    )

    # Sign-check: the bullish day (offset=0, today) should produce a
    # positive per-day tone; the bearish days should be negative. Series
    # is oldest-first so today is the LAST entry.
    by_date = {p["date"]: p["mean_tone"] for p in body["tone_series"]}
    today_key = now.strftime("%Y-%m-%d")
    three_days_ago = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    assert by_date[today_key] > 0.0, (
        f"bullish headline today should yield positive tone, got {by_date[today_key]}"
    )
    assert by_date[three_days_ago] < 0.0, (
        f"bearish headline 3d ago should yield negative tone, got {by_date[three_days_ago]}"
    )


@respx.mock
def test_sentiment_trend_handles_timeline_keys_with_hour_suffix(
    app_client: TestClient,
) -> None:
    """Defensive: even if a future GDELT change starts emitting timelinetone
    keys with hour/minute suffixes (e.g. ``2026-05-10T12:00:00Z`` instead of
    bare ``20260510T120000Z``), key normalisation must still align per-day
    points to the dense series. Without the normalisation, ``d in timeline``
    fails and per-day tones silently stay at 0.0.
    """
    from pfm.terminal_sentiment_trend import _build_tone_series

    now = datetime.now(tz=UTC)
    end_ts = pd.Timestamp(now)
    # Timeline supplied with non-canonical keys (ISO with hours instead of
    # ``YYYY-MM-DD``) — must still resolve correctly after normalisation.
    iso_today = now.strftime("%Y-%m-%dT00:00:00Z")
    iso_yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%dT12:30:00Z")
    weird_timeline = {iso_today: 5.0, iso_yesterday: -2.0}

    series = _build_tone_series(
        articles=[],
        fallback_topic="x",
        days=3,
        end_date=end_ts,
        tone_timeline=weird_timeline,
    )
    # Series is oldest-first; today is the last entry.
    tones = [p.mean_tone for p in series]
    dates = [p.date for p in series]
    today_key = now.strftime("%Y-%m-%d")
    yesterday_key = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    assert today_key in dates
    assert yesterday_key in dates
    # Match position by date so we don't depend on slice indices.
    by_date = dict(zip(dates, tones, strict=True))
    assert by_date[today_key] == 5.0, f"timeline key normalisation failed, got {tones}"
    assert by_date[yesterday_key] == -2.0


def test_normalize_timeline_keys_handles_mixed_formats() -> None:
    """Unit test for the date-key alignment helper itself."""
    from pfm.terminal_sentiment_trend import _normalize_timeline_keys

    raw = {
        "2026-05-10": 1.5,  # already canonical
        "2026-05-11T12:00:00Z": 2.5,  # ISO with time
        "20260512T080000Z": 3.5,  # GDELT seendate format
        "garbage": 99.0,  # unparseable → dropped
    }
    out = _normalize_timeline_keys(raw)
    assert out["2026-05-10"] == 1.5
    assert out["2026-05-11"] == 2.5
    assert out["2026-05-12"] == 3.5
    assert "garbage" not in out
    assert _normalize_timeline_keys(None) == {}
    assert _normalize_timeline_keys({}) == {}

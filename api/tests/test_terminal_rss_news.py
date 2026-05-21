"""Tests for ``pfm.terminal_rss_news``. All HTTP mocked via respx."""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.sources.polymarket import PolymarketClient
from pfm.terminal_rss_news import (
    _CACHE,
    SOURCES,
    _parse_rss,
)
from pfm.terminal_rss_news import (
    router as rss_router,
)

GAMMA = "https://gamma-api.test"
CLOB = "https://clob.test"

# Minimal valid RSS-2.0 fixtures, one per source slug.
_RSS_BBC_WORLD = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>BBC World</title>
  <item>
    <title>UK PM addresses NATO summit</title>
    <link>https://bbc.test/world/1</link>
    <pubDate>Fri, 01 May 2026 09:00:00 GMT</pubDate>
    <description>Strong remarks at the alliance.</description>
  </item>
  <item>
    <title>Markets crash on tariff fears</title>
    <link>https://bbc.test/world/2</link>
    <pubDate>Fri, 02 May 2026 10:00:00 GMT</pubDate>
    <description>Investors panic as new tariffs are announced.</description>
  </item>
</channel></rss>
"""

_RSS_COINDESK = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>CoinDesk</title>
  <item>
    <title>Bitcoin rallies past $100k milestone</title>
    <link>https://coindesk.test/btc-rally</link>
    <pubDate>Fri, 02 May 2026 12:00:00 GMT</pubDate>
    <description>Crypto soars after dovish Fed signal.</description>
  </item>
</channel></rss>
"""

# A minimal Atom 1.0 feed (The Verge ships Atom).
_ATOM_VERGE = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>The Verge</title>
  <entry>
    <title>Apple unveils new chip</title>
    <link href="https://verge.test/apple-chip"/>
    <updated>2026-05-02T08:00:00Z</updated>
    <summary>Faster, cooler, cheaper.</summary>
  </entry>
</feed>
"""

# Used to emulate a "down" source (HTTP 503).


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    _CACHE.clear()
    yield
    _CACHE.clear()


@pytest.fixture
def app_client() -> TestClient:
    app = FastAPI()
    app.state.poly = PolymarketClient(gamma_url=GAMMA, clob_url=CLOB, client=httpx.Client())
    app.include_router(rss_router)
    return TestClient(app)


def _mock_all_sources(*, bbc_status: int = 200, others_empty: bool = True) -> None:
    """Default-mock every source so respx doesn't error on a stray real call."""
    empty_rss = b'<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>'
    for src in SOURCES:
        if src.slug == "bbc_world":
            respx.get(src.url).mock(return_value=httpx.Response(bbc_status, content=_RSS_BBC_WORLD))
        elif src.slug == "coindesk":
            respx.get(src.url).mock(return_value=httpx.Response(200, content=_RSS_COINDESK))
        elif src.slug == "verge":
            respx.get(src.url).mock(return_value=httpx.Response(200, content=_ATOM_VERGE))
        else:
            content = empty_rss if others_empty else _RSS_BBC_WORLD
            respx.get(src.url).mock(return_value=httpx.Response(200, content=content))


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_parse_rss_extracts_titles_and_strips_html() -> None:
    """RSS-2.0 parsing produces well-formed RssHeadline objects."""
    src = next(s for s in SOURCES if s.slug == "bbc_world")
    items = _parse_rss(_RSS_BBC_WORLD, src)
    assert len(items) == 2
    titles = [it.title for it in items]
    assert "Markets crash on tariff fears" in titles
    crash = next(it for it in items if "crash" in it.title.lower())
    assert crash.sentiment == "negative"
    assert crash.source == "bbc_world"
    assert crash.link == "https://bbc.test/world/2"
    # ISO-8601 normalisation: GMT -> Z-suffix.
    assert crash.pub_date.endswith("Z")


def test_parse_atom_feed_handles_namespaces() -> None:
    """Atom 1.0 feeds (The Verge) parse via the namespaced fallback path."""
    src = next(s for s in SOURCES if s.slug == "verge")
    items = _parse_rss(_ATOM_VERGE, src)
    assert len(items) == 1
    assert items[0].title == "Apple unveils new chip"
    assert items[0].link == "https://verge.test/apple-chip"
    assert items[0].pub_date == "2026-05-02T08:00:00Z"


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


@respx.mock
def test_headlines_aggregates_and_ranks_by_recency(app_client: TestClient) -> None:
    """``/headlines`` merges every source, dedupes on link, sorts newest-first."""
    _mock_all_sources()
    resp = app_client.get("/terminal/rss/headlines?limit=10")
    assert resp.status_code == 200
    body = resp.json()
    titles = [it["title"] for it in body["items"]]
    # BBC's "Markets crash on tariff fears" (2026-05-02 10:00) comes before
    # BBC's "UK PM addresses NATO summit" (2026-05-01 09:00).
    assert titles.index("Markets crash on tariff fears") < titles.index(
        "UK PM addresses NATO summit"
    )
    # Sentiment scoring wired through.
    crash = next(it for it in body["items"] if it["title"] == "Markets crash on tariff fears")
    assert crash["sentiment"] == "negative"
    # bbc_world, coindesk, verge contributed.
    assert "bbc_world" in body["sources_used"]
    assert "coindesk" in body["sources_used"]
    assert "verge" in body["sources_used"]


@respx.mock
def test_sources_endpoint_marks_failed_source_as_error(app_client: TestClient) -> None:
    """``/sources`` returns a per-source ok/error status, including HTTP 5xx."""
    _mock_all_sources(bbc_status=503)
    resp = app_client.get("/terminal/rss/sources")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_sources"] == len(SOURCES)
    bbc = next(s for s in body["sources"] if s["slug"] == "bbc_world")
    assert bbc["status"] == "error"
    assert "503" in bbc["error"]
    coindesk = next(s for s in body["sources"] if s["slug"] == "coindesk")
    assert coindesk["status"] == "ok"
    assert coindesk["n_items"] == 1


@respx.mock
def test_slug_endpoint_filters_by_keyword_overlap(app_client: TestClient) -> None:
    """``/{slug}`` resolves the question and ranks by keyword overlap."""
    respx.get(f"{GAMMA}/markets", params={"slug": "btc-100k"}).mock(
        return_value=httpx.Response(
            200,
            json=[{"slug": "btc-100k", "question": "Will Bitcoin reach 100k by year end?"}],
        )
    )
    _mock_all_sources()

    resp = app_client.get("/terminal/rss/btc-100k?limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["slug"] == "btc-100k"
    assert "bitcoin" in body["keywords"]
    # CoinDesk's "Bitcoin rallies past $100k milestone" must match the bitcoin/100k tokens.
    titles = [it["title"] for it in body["items"]]
    assert any("Bitcoin" in t for t in titles)
    # The unrelated NATO headline must NOT appear (zero token overlap).
    assert not any("NATO" in t for t in titles)


@respx.mock
def test_headlines_filters_by_category(app_client: TestClient) -> None:
    """category=crypto only returns crypto-tagged sources."""
    _mock_all_sources()
    resp = app_client.get("/terminal/rss/headlines?category=crypto&limit=10")
    assert resp.status_code == 200
    body = resp.json()
    assert body["category"] == "crypto"
    # Every returned item is from a crypto source (coindesk).
    assert all(it["category"] == "crypto" for it in body["items"])
    assert any(it["source"] == "coindesk" for it in body["items"])
    # No BBC world entries in a crypto-only feed.
    assert not any(it["source"] == "bbc_world" for it in body["items"])


@respx.mock
def test_headlines_swallows_timeouts_and_returns_partial(
    app_client: TestClient,
) -> None:
    """One feed timing out must NOT 502 the whole endpoint — we return a
    partial set + log a warning and the rest of the wires keep flowing.

    Simulates a wedged TLS handshake by having one source raise
    ``httpx.ConnectTimeout`` (which the inner ``_fetch_source`` already
    catches and turns into ``[], 'http error: ...'``); the other sources
    return normal data."""
    for src in SOURCES:
        if src.slug == "bbc_world":
            respx.get(src.url).mock(side_effect=httpx.ConnectTimeout("simulated TLS timeout"))
        elif src.slug == "coindesk":
            respx.get(src.url).mock(return_value=httpx.Response(200, content=_RSS_COINDESK))
        else:
            respx.get(src.url).mock(
                return_value=httpx.Response(
                    200,
                    content=b'<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>',
                )
            )

    resp = app_client.get("/terminal/rss/headlines?limit=20")
    assert resp.status_code == 200
    body = resp.json()
    # CoinDesk still surfaced — partial set, not a 502.
    assert any(it["source"] == "coindesk" for it in body["items"])
    # The dead source is NOT in sources_used.
    assert "bbc_world" not in body["sources_used"]

"""Tests for ``pfm.news_causal_chain``.

The module is largely pure Python — keyword overlap, logit math, β →
expected return — so most tests pin behaviour without any HTTP mocking.
A small respx-mocked test exercises the GDELT-hydration fallback path on
the ``/movers`` endpoint to ensure it degrades gracefully when the live
feed returns nothing.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.cache_utils import reset_caches
from pfm.news_causal_chain import (
    BETA_REGISTRY,
    SYNTHETIC_BETA_PLACEHOLDER,
    build_causal_chain,
    register_betas,
    top_news_movers,
)
from pfm.news_causal_chain import (
    router as news_causal_router,
)
from pfm.sources.polymarket import PolymarketClient
from pfm.terminal_gdelt_news import GDELT_DOC_URL

GAMMA = "https://gamma-api.test"
CLOB = "https://clob.test"


@pytest.fixture(autouse=True)
def _clear_state() -> None:
    BETA_REGISTRY.clear()
    reset_caches()
    yield
    BETA_REGISTRY.clear()
    reset_caches()


@pytest.fixture
def app_client() -> TestClient:
    app = FastAPI()
    app.state.poly = PolymarketClient(gamma_url=GAMMA, clob_url=CLOB, client=httpx.Client())
    app.include_router(news_causal_router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# build_causal_chain — synthetic items
# ---------------------------------------------------------------------------


def test_build_causal_chain_three_items_one_match() -> None:
    """3 items, 1 carries the factor's keyword and a price reaction.

    The matching item should produce a single-link chain whose tagged_factor
    equals the factor id, has a non-zero Δlogit, and yields per-ticker
    expected_return forecasts at sensible magnitudes (β=0.5 × Δlogit*100).
    """
    register_betas("trump-impeach-2027", {"DJT": 0.5, "SPY": -0.1})

    items = [
        {
            "title": "Senate to vote on Trump impeachment next week",
            "ts": "2026-04-01T12:00:00Z",
            "url": "https://bbc.com/a",
            "source": "bbc.com",
            "price_before": 0.30,
            "price_after": 0.45,  # +15 pp → big positive Δlogit
        },
        {
            "title": "Apple earnings beat expectations",
            "ts": "2026-04-01T13:00:00Z",
            "url": "https://wsj.com/b",
            "source": "wsj.com",
        },
        {
            "title": "Market closes flat on light volume",
            "ts": "2026-04-01T20:00:00Z",
            "url": "https://wsj.com/c",
            "source": "wsj.com",
        },
    ]

    out = build_causal_chain("trump-impeach-2027", items, lookback_hours=24)

    assert out["factor_id"] == "trump-impeach-2027"
    assert out["lookback_hours"] == 24
    assert out["n_items"] == 3
    assert out["n_tagged"] == 1

    # The Trump item is the only tagged one.
    chain = out["chain"]
    tagged = [c for c in chain if c["tagged_factor"] is not None]
    assert len(tagged) == 1
    link = tagged[0]
    assert link["tagged_factor"] == "trump-impeach-2027"
    assert link["delta_prob"] == pytest.approx(0.15, abs=1e-9)
    # logit(0.45) - logit(0.30) ≈ -0.201 - (-0.847) ≈ 0.646
    assert link["delta_logit"] == pytest.approx(0.646, abs=0.05)
    assert link["confidence"] in {"high", "medium"}

    # Two ticker impacts (one per registered β).
    impacts = {tk["ticker"]: tk for tk in link["affected_tickers"]}
    assert set(impacts) == {"DJT", "SPY"}
    djt = impacts["DJT"]
    assert djt["beta"] == pytest.approx(0.5)
    assert djt["beta_source"] == "regression"
    # β=0.5 × Δlogit ≈ 0.646 × 100 ≈ +32%
    assert djt["expected_return_pct"] == pytest.approx(32.3, abs=3.0)

    spy = impacts["SPY"]
    # Negative β → opposite sign.
    assert spy["expected_return_pct"] is not None
    assert spy["expected_return_pct"] < 0


def test_build_causal_chain_no_beta_registry_falls_back_to_synthetic() -> None:
    """Tagged item but no β registered → confidence=medium, synthetic placeholder."""
    items = [
        {
            "title": "Fed signals one rate cut at next FOMC meeting",
            "ts": "2026-04-01T18:00:00Z",
            "source": "reuters.com",
            "price_before": 0.40,
            "price_after": 0.55,
        }
    ]

    out = build_causal_chain("fed-rate-cut-2026", items)
    link = out["chain"][0]
    assert link["tagged_factor"] == "fed-rate-cut-2026"
    assert link["confidence"] == "medium"
    assert len(link["affected_tickers"]) == 1
    tk = link["affected_tickers"][0]
    assert tk["beta_source"] == "synthetic"
    assert tk["beta"] == pytest.approx(SYNTHETIC_BETA_PLACEHOLDER)
    assert tk["confidence"] == "medium"
    assert tk["expected_return_pct"] is not None


def test_build_causal_chain_no_factor_beta_yields_low_confidence() -> None:
    """β registry empty AND no price reaction → tagged but expected_return None."""
    items = [
        {
            "title": "Trump faces 2027 impeach vote, says senator",
            "source": "cnn.com",
            # No price_before/price_after provided.
        }
    ]

    out = build_causal_chain("trump-impeach-2027", items)
    link = out["chain"][0]
    assert link["tagged_factor"] == "trump-impeach-2027"
    assert link["delta_logit"] is None
    assert link["confidence"] == "low"
    # No β registered AND no Δlogit → no ticker rows emitted.
    assert link["affected_tickers"] == []


def test_build_causal_chain_no_keyword_match_skipped() -> None:
    """Item with zero keyword overlap → tagged_factor=None, confidence=low."""
    items = [
        {
            "title": "Local sports team wins championship",
            "source": "espn.com",
            "price_before": 0.50,
            "price_after": 0.60,
        }
    ]
    out = build_causal_chain("trump-impeach-2027", items)
    link = out["chain"][0]
    assert link["tagged_factor"] is None
    assert link["keyword_overlap"] == 0
    assert link["confidence"] == "low"
    assert link["affected_tickers"] == []


def test_build_causal_chain_explicit_beta_map_wins_over_registry() -> None:
    """``beta_map`` argument takes precedence over BETA_REGISTRY entries."""
    register_betas("trump-impeach-2027", {"DJT": 0.5})
    items = [
        {
            "title": "Trump indictment news",
            "price_before": 0.30,
            "price_after": 0.45,
        }
    ]
    out = build_causal_chain(
        "trump-impeach-2027",
        items,
        beta_map={"AAPL": -0.20, "TSLA": 0.30},
    )
    impacts = out["chain"][0]["affected_tickers"]
    assert {tk["ticker"] for tk in impacts} == {"AAPL", "TSLA"}


# ---------------------------------------------------------------------------
# top_news_movers — pure function path
# ---------------------------------------------------------------------------


def test_top_news_movers_ranks_by_absolute_impact() -> None:
    """Two factors, each with one tagged item → highest |impact| ranks first."""
    fetched = {
        "trump-impeach-2027": [
            {
                "title": "Trump impeachment moves forward",
                "ts": "2026-04-01T10:00:00Z",
                "source": "bbc.com",
                "price_before": 0.20,
                "price_after": 0.55,  # huge logit move
            }
        ],
        "fed-rate-cut-2026": [
            {
                "title": "Fed minutes hint at one rate cut",
                "ts": "2026-04-01T14:00:00Z",
                "source": "wsj.com",
                "price_before": 0.50,
                "price_after": 0.52,  # tiny move
            }
        ],
    }
    register_betas("trump-impeach-2027", {"DJT": 1.0})
    register_betas("fed-rate-cut-2026", {"TLT": 1.0})

    movers = top_news_movers(
        window_hours=24,
        n=10,
        min_impact_pct=0.0,
        fetched_items_by_factor=fetched,
    )
    assert len(movers) == 2
    # Highest-magnitude impact first.
    assert abs(movers[0]["expected_impact_pct"]) > abs(movers[1]["expected_impact_pct"])
    assert movers[0]["factor_id"] == "trump-impeach-2027"


def test_top_news_movers_drops_below_threshold() -> None:
    """``min_impact_pct`` filters tiny-impact items."""
    register_betas("fed-rate-cut-2026", {"TLT": 0.05})
    fetched = {
        "fed-rate-cut-2026": [
            {
                "title": "Fed officials weigh rate path",
                "price_before": 0.50,
                "price_after": 0.51,  # ~0.04 logit × β=0.05 × 100 ≈ 0.2%
            }
        ]
    }
    movers = top_news_movers(
        window_hours=24,
        n=10,
        min_impact_pct=5.0,  # 5 pct threshold
        fetched_items_by_factor=fetched,
    )
    assert movers == []


# ---------------------------------------------------------------------------
# Router endpoints
# ---------------------------------------------------------------------------


def test_post_causal_chain_with_inline_items(app_client: TestClient) -> None:
    """POST /news/causal-chain accepts inline items and skips hydration."""
    register_betas("ai-bubble-pop", {"NVDA": -0.30})
    body = {
        "factor_id": "ai-bubble-pop",
        "news_items": [
            {
                "title": "AI bubble concerns mount as bubble talk grows",
                "price_before": 0.20,
                "price_after": 0.30,
            }
        ],
        "lookback_hours": 24,
    }
    resp = app_client.post("/news/causal-chain", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["factor_id"] == "ai-bubble-pop"
    assert data["n_tagged"] >= 1
    link = data["chain"][0]
    assert link["affected_tickers"][0]["ticker"] == "NVDA"
    assert link["affected_tickers"][0]["expected_return_pct"] is not None


@respx.mock
def test_get_movers_handles_empty_gdelt_gracefully(app_client: TestClient) -> None:
    """GDELT returns no articles → /movers returns 200 with empty list."""
    register_betas("trump-impeach-2027", {"DJT": 0.5})
    respx.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json={"articles": []}))
    resp = app_client.get("/news/movers?hours=24&n=5&min_impact_pct=1.0")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["window_hours"] == 24
    assert body["n_returned"] == 0
    assert body["movers"] == []


def test_get_movers_with_no_registered_factors(app_client: TestClient) -> None:
    """Empty BETA_REGISTRY → endpoint returns 200 / 0 movers, no upstream calls."""
    resp = app_client.get("/news/movers")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_total"] == 0
    assert body["n_returned"] == 0

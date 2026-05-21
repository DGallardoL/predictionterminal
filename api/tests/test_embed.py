"""Tests for ``pfm.embed`` — embeddable widgets + OG images.

External HTTP (Polymarket Gamma + CLOB) is mocked via :mod:`respx`. The
matplotlib OG renderer is exercised once for real (it's pure-CPU and very
fast) and patched out for the rest so we don't hammer the renderer needlessly.
The router is mounted on a fresh :class:`FastAPI` app to bypass the heavy
``pfm.main`` lifespan.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import embed as embed_mod
from pfm import og_image as og_mod

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"


# --- helpers ----------------------------------------------------------------


def _build_app() -> TestClient:
    app = FastAPI()
    app.include_router(embed_mod.router)
    return TestClient(app)


def _gamma_market_payload(
    slug: str, *, token_id: str = "tok-x", base_price: float = 0.55
) -> dict[str, Any]:
    return {
        "slug": slug,
        "question": f"Will {slug.replace('-', ' ')} happen?",
        "description": "Test market",
        "clobTokenIds": json.dumps([token_id, f"{token_id}_no"]),
        "bestBid": base_price - 0.01,
        "bestAsk": base_price + 0.01,
        "lastTradePrice": base_price,
        "volume24hr": 50_000.0,
        "volumeNum": 1_000_000.0,
        "liquidityNum": 25_000.0,
        "oneDayPriceChange": 0.02,
        "oneWeekPriceChange": -0.05,
        "endDate": "2026-12-01T00:00:00Z",
        "startDate": "2025-01-01T00:00:00Z",
        "createdAt": "2025-01-01T00:00:00Z",
        "active": True,
        "closed": False,
    }


def _clob_history_payload(days: int = 30, *, base: float = 0.55) -> dict[str, Any]:
    history = []
    p = base
    for i in range(days):
        # Simple deterministic walk so tests don't depend on random seeds.
        p = max(0.05, min(0.95, p + (0.005 if i % 2 == 0 else -0.004)))
        ts = 1_700_000_000 + i * 86400
        history.append({"t": ts, "p": float(p)})
    return {"history": history}


def _mock_market(slug: str, *, token_id: str = "tok-x", base_price: float = 0.55) -> None:
    respx.get(f"{GAMMA_URL}/markets", params={"slug": slug}).mock(
        return_value=httpx.Response(
            200, json=[_gamma_market_payload(slug, token_id=token_id, base_price=base_price)]
        )
    )
    respx.get(f"{CLOB_URL}/prices-history", params={"market": token_id}).mock(
        return_value=httpx.Response(200, json=_clob_history_payload(base=base_price))
    )


# --- fixtures ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _drop_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reset the embed in-process cache + redirect disk paths to ``tmp_path``."""
    embed_mod._EMBED_CACHE.clear()

    # Per-test BEACON path under tmp_path so concurrent tests don't collide.
    monkeypatch.setattr(embed_mod, "BEACON_LOG_PATH", tmp_path / "beacons.jsonl")

    # Per-test OG cache dir (the renderer mkdirs it lazily).
    og_dir = tmp_path / "og_cache"
    monkeypatch.setattr(og_mod, "CACHE_DIR", og_dir)

    yield
    embed_mod._EMBED_CACHE.clear()


@pytest.fixture
def alpha_strategies_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a minimal alpha_strategies.json + live_signals.json pointing at it."""
    pair_id = "test_pair_alpha"
    strategies_doc = {
        "strategies": [
            {
                "pair_id": pair_id,
                "a_id": "factor_a",
                "b_id": "factor_b",
                "a_name": "Factor A",
                "b_name": "Factor B",
                "tier": "A_GOLD",
                "oos_sharpe": 2.45,
                "half_life_days": 1.7,
            }
        ]
    }
    signals_doc = {
        "signals": {
            pair_id: {
                "pair_id": pair_id,
                "current_z": 1.85,
                "action": "SHORT_SPREAD",
                "reason": "z above entry threshold",
            }
        }
    }
    sp = tmp_path / "alpha_strategies.json"
    lp = tmp_path / "live_signals.json"
    sp.write_text(json.dumps(strategies_doc))
    lp.write_text(json.dumps(signals_doc))
    monkeypatch.setattr(embed_mod, "ALPHA_STRATEGIES_PATH", sp)
    monkeypatch.setattr(embed_mod, "LIVE_SIGNALS_PATH", lp)
    return sp


# --- /embed/market ----------------------------------------------------------


class TestEmbedMarket:
    @respx.mock
    def test_returns_html_with_og_meta(self) -> None:
        _mock_market("will-x-happen")
        client = _build_app()
        r = client.get("/embed/market/will-x-happen")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        body = r.text
        # OG / Twitter meta tags present.
        assert '<meta property="og:title"' in body
        assert '<meta name="twitter:card" content="summary_large_image">' in body
        # Sparkline container + footer present.
        assert 'id="pfm-emb-chart"' in body
        assert "Prediction Factor Model" in body
        # Question rendered.
        assert "Will" in body

    @respx.mock
    def test_embed_headers_allow_framing(self) -> None:
        _mock_market("will-x-happen")
        client = _build_app()
        r = client.get("/embed/market/will-x-happen")
        assert r.headers.get("x-frame-options") == "ALLOWALL"
        csp = r.headers.get("content-security-policy", "")
        assert "frame-ancestors *" in csp
        assert "max-age" in r.headers.get("cache-control", "")

    @respx.mock
    def test_handles_missing_market_gracefully(self) -> None:
        respx.get(f"{GAMMA_URL}/markets", params={"slug": "ghost-market"}).mock(
            return_value=httpx.Response(200, json=[])
        )
        respx.get(
            f"{GAMMA_URL}/markets",
            params={"slug": "ghost-market", "closed": "true"},
        ).mock(return_value=httpx.Response(200, json=[]))
        client = _build_app()
        r = client.get("/embed/market/ghost-market")
        # Should still return a 200 placeholder rather than 5xx the host page.
        assert r.status_code == 200
        assert "Ghost Market" in r.text  # title-cased fallback question

    @respx.mock
    def test_dark_theme_renders(self) -> None:
        _mock_market("will-x-happen")
        client = _build_app()
        r = client.get("/embed/market/will-x-happen?theme=dark")
        assert r.status_code == 200
        # Dark background hex appears in the inline style block.
        assert "#0d1117" in r.text


# --- /embed/strategy --------------------------------------------------------


class TestEmbedStrategy:
    def test_renders_tier_badge_and_action(self, alpha_strategies_file: Path) -> None:
        client = _build_app()
        r = client.get("/embed/strategy/test_pair_alpha")
        assert r.status_code == 200
        body = r.text
        assert "A_GOLD" in body
        assert "SHORT_SPREAD" in body
        assert "2.45" in body  # OOS Sharpe rendered
        assert "Factor A" in body and "Factor B" in body
        assert r.headers["x-frame-options"] == "ALLOWALL"

    def test_unknown_pair_id_returns_placeholder(self) -> None:
        client = _build_app()
        r = client.get("/embed/strategy/this_pair_does_not_exist")
        assert r.status_code == 200
        assert "UNKNOWN" in r.text


# --- /embed/og/market/{slug}.png -------------------------------------------


class TestEmbedOGImage:
    @respx.mock
    def test_returns_png(self) -> None:
        _mock_market("will-x-happen")
        client = _build_app()
        r = client.get("/embed/og/market/will-x-happen.png")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        # PNG magic bytes \x89PNG\r\n\x1a\n.
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"

    @respx.mock
    def test_uses_disk_cache_on_second_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_market("will-x-happen")
        client = _build_app()
        calls = {"n": 0}
        original = og_mod.render_market_og

        def _spy(*args: Any, **kwargs: Any) -> bytes:
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(og_mod, "render_market_og", _spy)
        # First call — renders and writes to disk.
        r1 = client.get("/embed/og/market/will-x-happen.png")
        # Second call — should hit the on-disk cache.
        r2 = client.get("/embed/og/market/will-x-happen.png")
        assert r1.status_code == 200 and r2.status_code == 200
        assert r1.content == r2.content
        # Render called at most once thanks to the on-disk cache.
        assert calls["n"] <= 1


# --- /embed/compare ---------------------------------------------------------


class TestEmbedCompare:
    @respx.mock
    def test_renders_overlay(self) -> None:
        _mock_market("apple-up", token_id="tok-a", base_price=0.5)
        _mock_market("orange-up", token_id="tok-b", base_price=0.6)
        client = _build_app()
        r = client.get("/embed/compare?slugs=apple-up,orange-up&theme=dark")
        assert r.status_code == 200
        body = r.text
        assert "Compare" in body
        assert "apple-up" in body
        assert "orange-up" in body
        assert r.headers["x-frame-options"] == "ALLOWALL"

    def test_single_slug_rejected(self) -> None:
        client = _build_app()
        r = client.get("/embed/compare?slugs=onlyone")
        assert r.status_code == 400


# --- /embed/beacon ----------------------------------------------------------


class TestEmbedBeacon:
    def test_returns_204(self, tmp_path: Path) -> None:
        client = _build_app()
        r = client.post(
            "/embed/beacon",
            json={
                "slug": "will-x-happen",
                "referrer": "https://example.com/post",
                "utm_source": "twitter",
                "ts": "2026-05-08T12:00:00Z",
            },
        )
        assert r.status_code == 204
        assert r.content == b""

    def test_writes_jsonl_row(self) -> None:
        client = _build_app()
        client.post("/embed/beacon", json={"slug": "abc", "referrer": "https://x.com"})
        client.post("/embed/beacon", json={"pair_id": "fed_x__fed_y"})
        rows = embed_mod.BEACON_LOG_PATH.read_text().splitlines()
        assert len(rows) == 2
        first = json.loads(rows[0])
        assert first["slug"] == "abc"
        assert first["referrer"] == "https://x.com"
        second = json.loads(rows[1])
        assert second["pair_id"] == "fed_x__fed_y"

    def test_empty_payload_still_204(self) -> None:
        client = _build_app()
        r = client.post("/embed/beacon", json={})
        assert r.status_code == 204


# --- /embed (cross-cutting) -------------------------------------------------


class TestEmbedHeaders:
    @respx.mock
    def test_market_headers(self) -> None:
        _mock_market("will-x-happen")
        client = _build_app()
        r = client.get("/embed/market/will-x-happen")
        assert r.headers["x-frame-options"] == "ALLOWALL"
        assert "frame-ancestors *" in r.headers["content-security-policy"]

    def test_strategy_headers(self, alpha_strategies_file: Path) -> None:
        client = _build_app()
        r = client.get("/embed/strategy/test_pair_alpha")
        assert r.headers["x-frame-options"] == "ALLOWALL"
        assert "frame-ancestors *" in r.headers["content-security-policy"]

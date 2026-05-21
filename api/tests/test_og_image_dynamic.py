"""Tests for the dynamic Factor + Strategy OG image renderers.

These tests intentionally avoid any network: the factor endpoint relies on
``_fetch_market_history`` which is monkeypatched to return synthetic data, and
the strategy endpoint reads from a temp-file ``alpha_strategies.json`` that we
write inline. The PNG bytes are inspected for the standard PNG magic header
and a minimum size threshold so we can distinguish a real render from an
empty/fallback path.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import embed as embed_mod
from pfm import og_image as og_mod

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MIN_PNG_BYTES = 5000


# --- fixtures ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Redirect OG-image disk cache + embed in-process cache to test-local paths."""
    monkeypatch.setattr(og_mod, "CACHE_DIR", tmp_path / "og_cache")
    embed_mod._EMBED_CACHE.clear()
    yield
    embed_mod._EMBED_CACHE.clear()


@pytest.fixture
def app_client() -> TestClient:
    app = FastAPI()
    app.include_router(embed_mod.router)
    return TestClient(app)


@pytest.fixture
def synthetic_history() -> list[float]:
    """A deterministic 90-day probability series."""
    return [0.45 + 0.08 * math.sin(i / 7.0) for i in range(90)]


@pytest.fixture
def synthetic_equity_curve() -> list[float]:
    """A deterministic ~1-year-long equity curve with modest drift."""
    base = 1.0
    out = []
    for i in range(252):
        base *= 1.0 + 0.0005 + 0.005 * math.sin(i / 13.0)
        out.append(base)
    return out


# --- direct renderer tests --------------------------------------------------


class TestRenderFactorOG:
    def test_returns_valid_png(self, synthetic_history: list[float]) -> None:
        png = og_mod.render_factor_og(
            "us_election_2026_dem",
            name="Democrats win 2026 midterms",
            theme="politics",
            source="polymarket",
            history=synthetic_history,
            last_price=0.524,
        )
        assert png[:8] == PNG_MAGIC
        assert len(png) > MIN_PNG_BYTES

    def test_handles_missing_data_gracefully(self) -> None:
        png = og_mod.render_factor_og("unknown_factor_xyz")
        assert png[:8] == PNG_MAGIC
        assert len(png) > MIN_PNG_BYTES

    def test_kalshi_source_pill(self, synthetic_history: list[float]) -> None:
        # Just exercise the Kalshi branch of the source-pill colour map.
        png = og_mod.render_factor_og(
            "kalshi_factor_1",
            name="Kalshi factor",
            theme="macro",
            source="kalshi",
            history=synthetic_history,
            last_price=0.7,
        )
        assert png[:8] == PNG_MAGIC
        assert len(png) > MIN_PNG_BYTES

    def test_disk_cache_returns_identical_bytes(
        self, synthetic_history: list[float], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}
        original = og_mod.render_factor_og

        def _spy(*args: Any, **kwargs: Any) -> bytes:
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(og_mod, "render_factor_og", _spy)
        p1 = og_mod.get_or_render_factor_og(
            "fid",
            name="N",
            theme="t",
            source="polymarket",
            history=synthetic_history,
            last_price=0.5,
        )
        p2 = og_mod.get_or_render_factor_og(
            "fid",
            name="N",
            theme="t",
            source="polymarket",
            history=synthetic_history,
            last_price=0.5,
        )
        assert p1 == p2
        assert p1[:8] == PNG_MAGIC
        # Cache should mean we render exactly once across the two calls.
        assert calls["n"] == 1


class TestRenderStrategyOG:
    def test_returns_valid_png(self, synthetic_equity_curve: list[float]) -> None:
        png = og_mod.render_strategy_og(
            "fed_target_40_eoy__fed_target_45_eoy",
            name="Fed straddle 4.0 vs 4.5",
            description="Long the 4.5 leg when implied path inverts vs 4.0 leg.",
            tier="A_GOLD",
            sharpe=2.45,
            equity_curve=synthetic_equity_curve,
        )
        assert png[:8] == PNG_MAGIC
        assert len(png) > MIN_PNG_BYTES

    def test_negative_sharpe_renders(self, synthetic_equity_curve: list[float]) -> None:
        png = og_mod.render_strategy_og(
            "bad_strat",
            name="A bad strategy",
            description="It loses money.",
            tier="D_RAW",
            sharpe=-0.85,
            equity_curve=synthetic_equity_curve,
        )
        assert png[:8] == PNG_MAGIC
        assert len(png) > MIN_PNG_BYTES

    def test_handles_missing_data(self) -> None:
        png = og_mod.render_strategy_og("orphan_strategy")
        assert png[:8] == PNG_MAGIC
        assert len(png) > MIN_PNG_BYTES

    def test_content_hash_busts_cache_on_change(self, synthetic_equity_curve: list[float]) -> None:
        # Different sharpe → different content hash → different cache file →
        # both render fresh and return valid PNGs.
        a = og_mod.get_or_render_strategy_og(
            "sid",
            name="S",
            tier="A_GOLD",
            sharpe=1.0,
            equity_curve=synthetic_equity_curve,
        )
        b = og_mod.get_or_render_strategy_og(
            "sid",
            name="S",
            tier="A_GOLD",
            sharpe=2.0,
            equity_curve=synthetic_equity_curve,
        )
        assert a[:8] == PNG_MAGIC and b[:8] == PNG_MAGIC
        # Different sharpe should produce visually different bytes (not equal).
        assert a != b


# --- endpoint tests ---------------------------------------------------------


class TestEmbedOGFactorEndpoint:
    def test_returns_png_when_history_available(
        self,
        app_client: TestClient,
        synthetic_history: list[float],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Patch the network-touching fetcher to return synthetic data.
        def _fake_fetch(slug: str, *, days: int = 7) -> tuple[dict[str, Any] | None, list[float]]:
            market = {
                "question": "Will the proxy event happen?",
                "bestBid": 0.50,
                "bestAsk": 0.54,
                "lastTradePrice": 0.52,
                "volume24hr": 25_000.0,
                "oneWeekPriceChange": 0.03,
            }
            return market, synthetic_history[-days:] if days > 0 else synthetic_history

        monkeypatch.setattr(embed_mod, "_fetch_market_history", _fake_fetch)

        r = app_client.get("/embed/og/factor/test_factor_1")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "image/png"
        assert r.content[:8] == PNG_MAGIC
        assert len(r.content) > MIN_PNG_BYTES

    def test_returns_404_when_no_data(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _empty_fetch(slug: str, *, days: int = 7) -> tuple[None, list[float]]:
            return None, []

        monkeypatch.setattr(embed_mod, "_fetch_market_history", _empty_fetch)
        r = app_client.get("/embed/og/factor/no_such_factor")
        assert r.status_code == 404

    def test_factor_endpoint_caches_disk(
        self,
        app_client: TestClient,
        synthetic_history: list[float],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _fake_fetch(slug: str, *, days: int = 7) -> tuple[dict[str, Any] | None, list[float]]:
            return ({"question": "Q"}, synthetic_history[-days:])

        monkeypatch.setattr(embed_mod, "_fetch_market_history", _fake_fetch)

        calls = {"n": 0}
        original = og_mod.render_factor_og

        def _spy(*args: Any, **kwargs: Any) -> bytes:
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(og_mod, "render_factor_og", _spy)

        r1 = app_client.get("/embed/og/factor/cache_me")
        r2 = app_client.get("/embed/og/factor/cache_me")
        assert r1.status_code == 200 and r2.status_code == 200
        assert r1.content == r2.content
        assert calls["n"] <= 1


class TestEmbedOGStrategyEndpoint:
    @pytest.fixture
    def strategies_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        pair_id = "test_pair_alpha"
        doc = {
            "strategies": [
                {
                    "pair_id": pair_id,
                    "a_name": "Factor A",
                    "b_name": "Factor B",
                    "tier": "A_GOLD",
                    "oos_sharpe": 2.45,
                    "half_life_days": 1.7,
                    "n_obs": 200,
                    "rationale": "Long-short pair on macro factors when z-score expands.",
                }
            ]
        }
        sp = tmp_path / "alpha_strategies.json"
        sp.write_text(json.dumps(doc))
        monkeypatch.setattr(embed_mod, "ALPHA_STRATEGIES_PATH", sp)
        return sp

    def test_returns_png_for_known_strategy(
        self, app_client: TestClient, strategies_file: Path
    ) -> None:
        r = app_client.get("/embed/og/strategy/test_pair_alpha")
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "image/png"
        assert r.content[:8] == PNG_MAGIC
        assert len(r.content) > MIN_PNG_BYTES

    def test_returns_404_for_unknown_strategy(
        self, app_client: TestClient, strategies_file: Path
    ) -> None:
        r = app_client.get("/embed/og/strategy/this_pair_does_not_exist")
        assert r.status_code == 404

    def test_returns_404_when_strategies_file_missing(
        self, app_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Point at a non-existent file.
        monkeypatch.setattr(embed_mod, "ALPHA_STRATEGIES_PATH", tmp_path / "nope.json")
        r = app_client.get("/embed/og/strategy/anything")
        assert r.status_code == 404

    def test_strategy_endpoint_caches_disk(
        self,
        app_client: TestClient,
        strategies_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = {"n": 0}
        original = og_mod.render_strategy_og

        def _spy(*args: Any, **kwargs: Any) -> bytes:
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(og_mod, "render_strategy_og", _spy)

        r1 = app_client.get("/embed/og/strategy/test_pair_alpha")
        r2 = app_client.get("/embed/og/strategy/test_pair_alpha")
        assert r1.status_code == 200 and r2.status_code == 200
        assert r1.content == r2.content
        assert calls["n"] <= 1


# --- sample byte sizes for the verification report -------------------------


def test_record_sample_png_sizes(
    synthetic_history: list[float],
    synthetic_equity_curve: list[float],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Render both shapes and print sizes — handy for the verification report.

    This is also a final guarantee that both renderers stay above the 5KB
    floor under realistic inputs.
    """
    factor_png = og_mod.render_factor_og(
        "demo_factor",
        name="A long-titled prediction-market factor for testing",
        theme="macro",
        source="polymarket",
        history=synthetic_history,
        last_price=0.612,
    )
    strategy_png = og_mod.render_strategy_og(
        "demo_strat_a__demo_strat_b",
        name="Demo strategy A | Demo strategy B",
        description="Long the leading binary on resolution decay; capacity-limited.",
        tier="A_STRUCTURAL",
        sharpe=1.87,
        equity_curve=synthetic_equity_curve,
    )
    print(f"factor png size: {len(factor_png)}")
    print(f"strategy png size: {len(strategy_png)}")
    assert factor_png[:8] == PNG_MAGIC and len(factor_png) > MIN_PNG_BYTES
    assert strategy_png[:8] == PNG_MAGIC and len(strategy_png) > MIN_PNG_BYTES
    # Force pytest to keep stdout visible when running with -s.
    capsys.readouterr()

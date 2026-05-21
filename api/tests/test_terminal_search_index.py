"""Tests for ``pfm.terminal_search_index`` — /terminal/search-index."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import terminal_search_index
from pfm.terminal_search_index import clear_cache, router


def _build_app() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _drop_cache() -> Iterator[None]:
    clear_cache()
    yield
    clear_cache()


class TestSearchIndexShape:
    def test_returns_factor_index_with_required_fields(self) -> None:
        client = _build_app()
        r = client.get("/terminal/search-index")
        assert r.status_code == 200, r.text
        body = r.json()

        # Top-level envelope.
        assert "version" in body
        assert "factors" in body
        assert "strategies" in body
        assert "pages" in body
        assert "actions" in body

        # Production factors.yml carries 1090 entries; tests run against the
        # real file so we expect at least 100.
        assert len(body["factors"]) >= 100, f"expected >=100 factors, got {len(body['factors'])}"
        assert body["n_factors"] == len(body["factors"])

        # Each factor row has the four required short keys.
        for row in body["factors"][:25]:
            assert "i" in row
            assert "s" in row
            assert "n" in row
            assert "t" in row  # may be None for "other"-themed factors
            assert isinstance(row["i"], str) and row["i"]
            assert isinstance(row["s"], str)
            assert isinstance(row["n"], str) and row["n"]


class TestSearchIndexCustomFactors:
    def test_uses_factors_file_supplied(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        custom = tmp_path / "factors.yml"
        custom.write_text(
            """
factors:
  - id: alpha_one
    name: Alpha factor one
    slug: alpha-1
    source: polymarket
    theme: macro
    description: A
  - id: alpha_two
    name: Alpha factor two
    slug: alpha-2
    source: polymarket
    theme: politics
    description: B
"""
        )
        monkeypatch.setattr(terminal_search_index, "DEFAULT_FACTORS_PATH", custom)
        # Also patch _load_factors_yaml's default arg by re-routing the
        # module-level helper to the new path. Tests previously cleared
        # the cache so the next call recomputes.
        original = terminal_search_index._load_factors_yaml
        monkeypatch.setattr(
            terminal_search_index,
            "_load_factors_yaml",
            lambda path=custom: original(custom),
        )

        client = _build_app()
        r = client.get("/terminal/search-index")
        assert r.status_code == 200, r.text
        body = r.json()
        ids = {row["i"] for row in body["factors"]}
        assert "alpha_one" in ids
        assert "alpha_two" in ids
        # Theme survives.
        themes = {row["i"]: row["t"] for row in body["factors"]}
        assert themes["alpha_one"] == "macro"
        assert themes["alpha_two"] == "politics"


class TestSearchIndexCache:
    def test_second_request_served_from_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}
        real = terminal_search_index._build_index

        def _counted() -> object:
            calls["n"] += 1
            return real()

        monkeypatch.setattr(terminal_search_index, "_build_index", _counted)

        client = _build_app()
        r1 = client.get("/terminal/search-index")
        r2 = client.get("/terminal/search-index")
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert calls["n"] == 1, "second request should hit cache"


class TestSearchIndexStaticBlocks:
    def test_pages_and_actions_present(self) -> None:
        client = _build_app()
        r = client.get("/terminal/search-index")
        body = r.json()
        # Static lists are non-empty and shaped right.
        assert len(body["pages"]) > 0
        assert all({"i", "n", "u"}.issubset(p.keys()) for p in body["pages"])
        assert len(body["actions"]) > 0
        assert all({"i", "n", "k"}.issubset(a.keys()) for a in body["actions"])

"""Regression tests for ``pfm.terminal.search_index``.

The main scenario covered here is the ``'str' object has no attribute 'iloc'``
crash that took down ``/terminal/search-index/chunked`` (and any non-trivial
query-string variant of ``/terminal/search-index``) in prod on 2026-05-16.

Root cause: ``TERMINAL_CACHE``'s L2 Redis layer JSON-encodes values with
``default=str``, which stringifies every ``pd.Series`` in the factor-history
dict before persisting. On a cross-worker readback the dict comes back as
``dict[str, str]`` and the original ``ser.iloc[-1]`` access on a ``str``
crashed the request. The fix isolates the search-index builder via an
``isinstance(ser, pd.Series)`` gate; this test pins that behaviour.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.terminal import search_index as si_mod


@pytest.fixture(autouse=True)
def _drop_cache() -> Iterator[None]:
    si_mod.clear_cache()
    yield
    si_mod.clear_cache()


def _custom_factors(tmp_path: Path) -> Path:
    p = tmp_path / "factors.yml"
    p.write_text(
        """
factors:
  - id: alpha_one
    name: Alpha factor one
    slug: alpha-1
    source: polymarket
    theme: macro
  - id: alpha_two
    name: Alpha factor two
    slug: alpha-2
    source: polymarket
    theme: politics
  - id: alpha_three
    name: Alpha three
    slug: alpha-3
    source: polymarket
    theme: tech
""",
        encoding="utf-8",
    )
    return p


def _build_client() -> TestClient:
    app = FastAPI()
    app.include_router(si_mod.router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Regression: corrupted history (post-L2-Redis readback) must not 500.
# ---------------------------------------------------------------------------


def test_search_index_handles_str_in_history_dict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A history dict that contains plain strings (the corrupted shape we
    observed in prod after the L2 Redis ``default=str`` JSON round-trip)
    must not crash the endpoint; the affected rows simply lose ``p``/``h``."""
    custom = _custom_factors(tmp_path)
    monkeypatch.setattr(si_mod, "DEFAULT_FACTORS_PATH", custom)
    monkeypatch.setattr(
        si_mod,
        "_load_factors_yaml",
        lambda path=custom: (
            si_mod._load_factors_yaml.__wrapped__(custom)  # type: ignore[attr-defined]
            if hasattr(si_mod._load_factors_yaml, "__wrapped__")
            else _real_loader(custom)
        ),
    )

    # Stub the history loader to return the corrupted shape: a real Series
    # for one slug, raw strings (the post-readback shape) for the other two.
    ser = pd.Series([0.10, 0.20, 0.30], dtype=float)

    def _fake_history(path: Path) -> dict[str, object]:
        return {
            "alpha-1": ser,
            "alpha-2": "0    0.5\n1    0.6\nName: alpha-2, dtype: float64",
            "alpha-3": "garbage string from json-dumps-default-str",
        }

    monkeypatch.setattr(si_mod.terminal_mod, "_load_factor_history_cache", _fake_history)

    client = _build_client()
    r = client.get("/terminal/search-index")
    assert r.status_code == 200, r.text
    body = r.json()
    by_id = {row["i"]: row for row in body["factors"]}
    assert by_id["alpha_one"]["p"] == pytest.approx(0.30, abs=1e-9)
    assert by_id["alpha_one"]["h"] == [0.10, 0.20, 0.30]
    # Corrupted rows degrade to no-price-no-spark — never 500, never raise.
    assert by_id["alpha_two"]["p"] is None
    assert by_id["alpha_two"]["h"] is None
    assert by_id["alpha_three"]["p"] is None
    assert by_id["alpha_three"]["h"] is None


def _real_loader(path: Path) -> list[dict[str, object]]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    raw = doc.get("factors") or []
    return raw if isinstance(raw, list) else []


def test_chunked_handles_str_in_history_dict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same corruption, exercised through ``/terminal/search-index/chunked``."""
    custom = _custom_factors(tmp_path)
    monkeypatch.setattr(si_mod, "DEFAULT_FACTORS_PATH", custom)
    monkeypatch.setattr(si_mod, "_load_factors_yaml", lambda path=custom: _real_loader(custom))
    monkeypatch.setattr(
        si_mod.terminal_mod,
        "_load_factor_history_cache",
        lambda _path: {"alpha-1": "stringified-series"},
    )

    client = _build_client()
    r = client.get("/terminal/search-index/chunked?chunk=0&size=2")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["chunk"] == 0
    assert body["chunk_size"] == 2
    assert body["total_chunks"] == 2  # 3 factors / 2 per chunk = 2 chunks
    assert body["n_factors"] == 3
    # First chunk has 2 rows, both with p/h=None thanks to the defensive gate.
    assert len(body["factors"]) == 2
    for row in body["factors"]:
        assert row["p"] is None
        assert row["h"] is None


# ---------------------------------------------------------------------------
# Misc shape checks (defence-in-depth for the new isinstance gate).
# ---------------------------------------------------------------------------


def test_search_index_with_real_series_still_populates_price_and_spark(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sanity: the isinstance gate must NOT suppress legitimate Series data."""
    custom = _custom_factors(tmp_path)
    monkeypatch.setattr(si_mod, "DEFAULT_FACTORS_PATH", custom)
    monkeypatch.setattr(si_mod, "_load_factors_yaml", lambda path=custom: _real_loader(custom))

    s1 = pd.Series([0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.61], dtype=float)
    s2 = pd.Series([0.10, 0.11], dtype=float)
    monkeypatch.setattr(
        si_mod.terminal_mod,
        "_load_factor_history_cache",
        lambda _path: {"alpha-1": s1, "alpha-2": s2},
    )

    client = _build_client()
    body = client.get("/terminal/search-index").json()
    by_id = {row["i"]: row for row in body["factors"]}
    assert by_id["alpha_one"]["p"] == pytest.approx(0.61)
    # Spark is the LAST SPARKLINE_LENGTH values (7).
    assert by_id["alpha_one"]["h"] == [0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.61]
    assert by_id["alpha_two"]["p"] == pytest.approx(0.11)
    assert by_id["alpha_two"]["h"] == [0.10, 0.11]
    # alpha-3 has no entry in history → degrades to None cleanly.
    assert by_id["alpha_three"]["p"] is None
    assert by_id["alpha_three"]["h"] is None


def test_search_index_handles_none_and_empty_series(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``None`` values and zero-length Series should also degrade to None."""
    custom = _custom_factors(tmp_path)
    monkeypatch.setattr(si_mod, "DEFAULT_FACTORS_PATH", custom)
    monkeypatch.setattr(si_mod, "_load_factors_yaml", lambda path=custom: _real_loader(custom))
    monkeypatch.setattr(
        si_mod.terminal_mod,
        "_load_factor_history_cache",
        lambda _path: {
            "alpha-1": None,
            "alpha-2": pd.Series([], dtype=float),
            "alpha-3": [0.1, 0.2, 0.3],  # raw list — not a Series either
        },
    )

    client = _build_client()
    r = client.get("/terminal/search-index")
    assert r.status_code == 200
    body = r.json()
    by_id = {row["i"]: row for row in body["factors"]}
    for fid in ("alpha_one", "alpha_two", "alpha_three"):
        assert by_id[fid]["p"] is None
        assert by_id[fid]["h"] is None


def test_chunked_oob_chunk_returns_empty_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Out-of-range chunk index must return an empty ``factors`` list, not 500."""
    custom = _custom_factors(tmp_path)
    monkeypatch.setattr(si_mod, "DEFAULT_FACTORS_PATH", custom)
    monkeypatch.setattr(si_mod, "_load_factors_yaml", lambda path=custom: _real_loader(custom))
    monkeypatch.setattr(si_mod.terminal_mod, "_load_factor_history_cache", lambda _path: {})

    client = _build_client()
    r = client.get("/terminal/search-index/chunked?chunk=99&size=2")
    assert r.status_code == 200
    body = r.json()
    assert body["factors"] == []
    assert body["total_chunks"] == 2  # 3 / 2 = 2 chunks

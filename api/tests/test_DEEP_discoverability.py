"""Tests for the unified factor resolver and ``/factors`` pagination.

These cover the three discoverability problems uncovered by the live-server
client trial:

1.  Slug / name acceptance — :func:`resolve_factor` accepts id, slug *and*
    case-insensitive name.
2.  ``did_you_mean`` suggestions — :func:`suggest_factors` returns
    semantically-close ids for a near-miss query.
3.  ``/factors`` no longer returns 1 360 entries in one shot; pagination
    + ``theme`` / ``source`` / ``search`` filters land. ``/factors/all``
    keeps the full dump available for power users.
4.  Endpoints that historically rejected unknown ids with an opaque
    ``detail`` string now embed structured ``did_you_mean`` payloads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pfm.factor_resolver import (
    resolve_factor,
    resolve_or_404,
    suggest_factors,
    suggest_factors_with_meta,
)
from pfm.factors import FactorConfig, load_factors

# ---------------------------------------------------------------------------
# Resolver-level tests — pure-function, catalog passed explicitly.
# ---------------------------------------------------------------------------


@pytest.fixture
def discoverability_catalog(tmp_path: Path) -> dict[str, FactorConfig]:
    """A small fixture catalog with the exact ids/slugs/names the spec calls out."""
    p = tmp_path / "factors.yml"
    p.write_text(
        """
factors:
  - id: no_fed_cuts_2026
    name: Will no Fed rate cuts happen in 2026?
    slug: will-no-fed-rate-cuts-happen-in-2026
    source: polymarket
    description: Tail probability the Fed holds steady through 2026.
    theme: macro
  - id: twelve_plus_fed_cuts
    name: Will 12+ Fed rate cuts happen in 2026?
    slug: will-12-or-more-fed-rate-cuts-happen-in-2026
    source: polymarket
    description: Volcker-pivot tail.
    theme: macro
  - id: k_fed_jul_cut25
    name: Kalshi Fed July cut 25bps
    slug: KXFED-2026JUL-25BP
    source: kalshi
    description: Kalshi market for the July FOMC.
    theme: macro
  - id: trump_2024
    name: Donald Trump wins 2024 election
    slug: will-donald-trump-win-the-2024-presidential-election
    source: polymarket
    description: Headline 2024 presidential market.
    theme: politics
  - id: spx_5500
    name: S&P 500 above 5500 EOY
    slug: will-the-sp-500-close-above-5500-on-2025-12-31
    source: polymarket
    description: Equity index level bet.
    theme: equity
"""
    )
    return load_factors(p)


def test_resolve_factor_by_id(discoverability_catalog: dict[str, FactorConfig]) -> None:
    fc = resolve_factor("no_fed_cuts_2026", discoverability_catalog)
    assert fc is not None
    assert fc.id == "no_fed_cuts_2026"


def test_resolve_factor_by_slug(discoverability_catalog: dict[str, FactorConfig]) -> None:
    fc = resolve_factor("will-no-fed-rate-cuts-happen-in-2026", discoverability_catalog)
    assert fc is not None
    assert fc.id == "no_fed_cuts_2026"


def test_resolve_factor_by_name(discoverability_catalog: dict[str, FactorConfig]) -> None:
    # Exact name (case-sensitive original).
    fc = resolve_factor(
        "Will no Fed rate cuts happen in 2026?",
        discoverability_catalog,
    )
    assert fc is not None and fc.id == "no_fed_cuts_2026"
    # Case-insensitive variant — humans paste names with random caps.
    fc2 = resolve_factor(
        "will no fed rate cuts happen in 2026?",
        discoverability_catalog,
    )
    assert fc2 is not None and fc2.id == "no_fed_cuts_2026"


def test_resolve_factor_unknown_returns_none(
    discoverability_catalog: dict[str, FactorConfig],
) -> None:
    assert resolve_factor("nonexistent_factor", discoverability_catalog) is None
    assert resolve_factor("", discoverability_catalog) is None


def test_suggest_factors_top_k_for_fed_query(
    discoverability_catalog: dict[str, FactorConfig],
) -> None:
    suggestions = suggest_factors("fed-rate-cuts-2026", discoverability_catalog, top_k=3)
    assert "no_fed_cuts_2026" in suggestions
    # All three Fed-related entries should outrank Trump / SPX.
    assert "trump_2024" not in suggestions[:1]


def test_suggest_factors_handles_typo(
    discoverability_catalog: dict[str, FactorConfig],
) -> None:
    # "trumpp" should still surface trump-related ids via SequenceMatcher.
    suggestions = suggest_factors("trumpp", discoverability_catalog, top_k=3)
    assert "trump_2024" in suggestions


def test_suggest_factors_with_meta_returns_score(
    discoverability_catalog: dict[str, FactorConfig],
) -> None:
    rows = suggest_factors_with_meta(
        "fed-rate-cuts-2026",
        discoverability_catalog,
        top_k=3,
    )
    assert rows, "expected at least one suggestion"
    top = rows[0]
    assert {"id", "name", "score"} <= set(top.keys())
    assert isinstance(top["score"], float)
    assert top["score"] > 0.0


def test_resolve_or_404_payload_shape(
    discoverability_catalog: dict[str, FactorConfig],
) -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        resolve_or_404("fed-rate-cuts-2026", discoverability_catalog)
    detail = exc_info.value.detail
    assert isinstance(detail, dict)
    assert detail["query"] == "fed-rate-cuts-2026"
    assert "did_you_mean" in detail
    ids = [s["id"] for s in detail["did_you_mean"]]
    assert "no_fed_cuts_2026" in ids


# ---------------------------------------------------------------------------
# /factors pagination tests — exercise the live FastAPI app via TestClient.
# ---------------------------------------------------------------------------


def test_factors_default_returns_paginated_envelope(app_client: TestClient) -> None:
    """Default call (no params) is now paginated — total/limit/offset present."""
    r = app_client.get("/factors")
    assert r.status_code == 200
    body = r.json()
    assert {"factors", "total", "limit", "offset", "next_offset"} <= set(body.keys())
    # The fixture catalog only has 2 factors so total is 2 and next_offset=None.
    assert body["total"] == 2
    assert body["limit"] == 50
    assert body["next_offset"] is None
    assert {f["id"] for f in body["factors"]} == {"factor_a", "factor_b"}


def test_factors_limit_caps_response(app_client: TestClient) -> None:
    r = app_client.get("/factors?limit=1")
    assert r.status_code == 200
    body = r.json()
    assert body["limit"] == 1
    assert len(body["factors"]) == 1
    # next_offset present when there's more data.
    assert body["total"] == 2
    assert body["next_offset"] == 1


def test_factors_offset_pages_forward(app_client: TestClient) -> None:
    r = app_client.get("/factors?limit=1&offset=1")
    assert r.status_code == 200
    body = r.json()
    assert body["offset"] == 1
    assert len(body["factors"]) == 1
    assert body["next_offset"] is None  # last page


def test_factors_search_filters_by_substring(app_client: TestClient) -> None:
    r = app_client.get("/factors?search=factor_a")
    assert r.status_code == 200
    body = r.json()
    ids = {f["id"] for f in body["factors"]}
    assert ids == {"factor_a"}


def test_factors_source_filter(app_client: TestClient) -> None:
    # All test fixture factors are polymarket -> filter returns both.
    r = app_client.get("/factors?source=polymarket")
    assert r.status_code == 200
    assert r.json()["total"] == 2
    # Different source -> empty result, but envelope still present.
    r2 = app_client.get("/factors?source=kalshi")
    assert r2.status_code == 200
    assert r2.json()["total"] == 0
    assert r2.json()["factors"] == []


def test_factors_all_returns_full_dump_with_warning(app_client: TestClient) -> None:
    r = app_client.get("/factors/all")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert body["limit"] == 2
    # Header carries the explicit count and a Warning advising pagination.
    assert r.headers.get("x-factor-count") == "2"
    assert "Warning" in r.headers


def test_factors_limit_invalid_rejected(app_client: TestClient) -> None:
    r = app_client.get("/factors?limit=0")  # ge=1 enforced
    assert r.status_code == 422
    r2 = app_client.get("/factors?limit=10000")  # le=500 enforced
    assert r2.status_code == 422


# ---------------------------------------------------------------------------
# Endpoints now reject unknown ids with structured did_you_mean.
# ---------------------------------------------------------------------------


def test_fit_unknown_factor_returns_did_you_mean(app_client: TestClient) -> None:
    r = app_client.post(
        "/fit",
        json={
            "ticker": "TEST",
            "factors": ["factor_aa"],  # near-miss for factor_a
            "start": "2025-06-15",
            "end": "2025-12-15",
        },
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert isinstance(detail, dict)
    assert "unknown" in detail
    assert detail["unknown"][0]["query"] == "factor_aa"
    suggested_ids = [s["id"] for s in detail["unknown"][0]["did_you_mean"]]
    assert "factor_a" in suggested_ids


def test_fit_accepts_slug_in_factors(app_client: TestClient) -> None:
    """Slug input now resolves to canonical factor id (was a 400 before)."""
    r = app_client.post(
        "/fit",
        json={
            "ticker": "TEST",
            "factors": ["slug-a", "slug-b"],
            "start": "2025-06-15",
            "end": "2025-12-15",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert {f["id"] for f in body["factors"]} == {"factor_a", "factor_b"}


def test_fit_accepts_name_in_factors(app_client: TestClient) -> None:
    """Human-readable names (e.g. pasted from UI) now resolve."""
    r = app_client.post(
        "/fit",
        json={
            "ticker": "TEST",
            "factors": ["Factor A", "Factor B"],
            "start": "2025-06-15",
            "end": "2025-12-15",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert {f["id"] for f in body["factors"]} == {"factor_a", "factor_b"}


def test_attribution_unknown_factor_returns_did_you_mean(app_client: TestClient) -> None:
    r = app_client.post(
        "/attribution",
        json={
            "ticker": "TEST",
            "factors": ["unknown_xyz"],
            "start": "2025-06-15",
            "end": "2025-12-15",
            "date": "2025-09-01",
        },
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert isinstance(detail, dict)
    assert detail["unknown"][0]["query"] == "unknown_xyz"


def test_event_model_correlation_resolves_slug(app_client: TestClient) -> None:
    """Slug input on /event-model/correlation-matrix now succeeds (or 422 on
    the legitimate "no history" path) — it should never bail out as 502
    from the upstream simply because the slug wasn't keyed by canonical id.
    """
    r = app_client.post(
        "/event-model/correlation-matrix",
        json={
            "factor_ids": ["slug-a", "slug-b"],
            "start": "2025-06-15",
            "end": "2025-12-15",
        },
    )
    # Either succeeds (200) on the synthetic fixture or fails on a downstream
    # math problem (422) — but not 400 "unknown factor" or 502 from the
    # opaque-slug path.
    assert r.status_code in (200, 422), (r.status_code, r.text)

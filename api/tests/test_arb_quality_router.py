"""Tests for ``pfm.arb.quality_router`` — ``GET /arb/quality-audit``.

Strategy
--------
- Mount the router on a fresh ``FastAPI`` app per test via fixture.
- Monkeypatch ``_load_pairs`` so we don't touch real ``dashboard_state.json``
  or ``top_arbs()`` (network-dependent).
- Use real T76/T77 modules where available — the audit pipeline is fast and
  exercising it end-to-end is the whole point. A separate test forces the
  503 path by stubbing ``_import_matchers`` to raise.

All tests run under the standard pytest harness; ``PYTHONPATH=src`` is
already configured by ``api/pyproject.toml``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers + fixtures
# ---------------------------------------------------------------------------


def _make_pair(
    pair_id: str,
    poly_title: str,
    kalshi_title: str,
    *,
    poly_slug: str = "",
    kalshi_ticker: str = "",
    profit_pct: float = 1.0,
    cost: float = 100.0,
    source: str = "dashboard_state",
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "poly_title": poly_title,
        "kalshi_title": kalshi_title,
        "poly_slug": poly_slug,
        "kalshi_ticker": kalshi_ticker,
        "profit_pct": profit_pct,
        "cost": cost,
        "source": source,
    }


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    """Always start with a cold cache so tests don't see each other's results."""
    from pfm.arb import quality_router as qr

    qr._CACHE["t"] = 0.0
    qr._CACHE["key"] = None
    qr._CACHE["value"] = None


@pytest.fixture
def app_client() -> TestClient:
    from pfm.arb.quality_router import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _patch_pairs(
    monkeypatch: pytest.MonkeyPatch, pairs: list[dict[str, Any]], source: str = "dashboard_state"
) -> None:
    from pfm.arb import quality_router as qr

    monkeypatch.setattr(qr, "_load_pairs", lambda: (list(pairs), source))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_200_with_mocked_pairs(app_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy-path: a few realistic pairs return a 200 with the documented envelope."""
    pairs = [
        _make_pair("pair-elec", "Will Trump win the 2024 election", "Trump wins 2024 presidency"),
        _make_pair("pair-btc-same", "Will BTC exceed $80k by 2024", "Will BTC exceed $80k by 2024"),
    ]
    _patch_pairs(monkeypatch, pairs)

    r = app_client.get("/arb/quality-audit")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) >= {
        "checked_at",
        "audited_count",
        "rejected_count",
        "high_conf_count",
        "borderline_count",
        "rejection_breakdown",
        "top_rejected",
        "source",
    }
    assert body["audited_count"] == 2
    assert body["source"] == "dashboard_state"
    # checked_at is a UTC ISO timestamp
    assert body["checked_at"].endswith("Z")
    # rejection_breakdown always contains the four canonical reason keys
    assert set(body["rejection_breakdown"].keys()) >= {
        "resolution_window_no_overlap",
        "threshold_mismatch",
        "jurisdiction_mismatch",
        "same_venue",
    }


def test_all_rejection_reasons_appear(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Construct pairs that trip each of the four hard-reject reasons."""
    # We bypass real text-based detection by patching _score_pair to return
    # canned scoring rows mapping to each reason.
    from pfm.arb import quality_router as qr

    canned = [
        {
            "pair_id": "p1",
            "poly_title": "a",
            "kalshi_title": "b",
            "score": 0.0,
            "rejected": True,
            "reason": "resolution_window_no_overlap",
            "profit_pct": 2.0,
            "cost": 100.0,
        },
        {
            "pair_id": "p2",
            "poly_title": "a",
            "kalshi_title": "b",
            "score": 0.0,
            "rejected": True,
            "reason": "threshold_mismatch",
            "profit_pct": 3.0,
            "cost": 200.0,
        },
        {
            "pair_id": "p3",
            "poly_title": "a",
            "kalshi_title": "b",
            "score": 0.0,
            "rejected": True,
            "reason": "jurisdiction_mismatch",
            "profit_pct": 1.0,
            "cost": 50.0,
        },
        {
            "pair_id": "p4",
            "poly_title": "a",
            "kalshi_title": "b",
            "score": 0.0,
            "rejected": True,
            "reason": "same_venue",
            "profit_pct": 4.0,
            "cost": 80.0,
        },
        {
            "pair_id": "p5",
            "poly_title": "x",
            "kalshi_title": "y",
            "score": 0.85,
            "rejected": False,
            "reason": "",
            "profit_pct": 0.5,
            "cost": 10.0,
        },
    ]
    seq = iter(canned)

    def fake_score(_sm: Any, _bh: Any, _p: dict[str, Any]) -> dict[str, Any]:
        return next(seq)

    monkeypatch.setattr(qr, "_score_pair", fake_score)
    _patch_pairs(monkeypatch, [_make_pair(f"p{i}", "x", "y") for i in range(1, 6)])

    r = app_client.get("/arb/quality-audit")
    assert r.status_code == 200
    body = r.json()
    assert body["audited_count"] == 5
    assert body["rejected_count"] == 4
    assert body["high_conf_count"] == 1
    bd = body["rejection_breakdown"]
    assert bd["resolution_window_no_overlap"] == 1
    assert bd["threshold_mismatch"] == 1
    assert bd["jurisdiction_mismatch"] == 1
    assert bd["same_venue"] == 1


def test_top_rejected_ranked_by_priced_impact(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Highest cost × (1 - score) pair must lead ``top_rejected``."""
    from pfm.arb import quality_router as qr

    canned = [
        {
            "pair_id": "cheap",
            "poly_title": "a",
            "kalshi_title": "b",
            "score": 0.0,
            "rejected": True,
            "reason": "same_venue",
            "profit_pct": 0.1,
            "cost": 1.0,
        },
        {
            "pair_id": "expensive",
            "poly_title": "a",
            "kalshi_title": "b",
            "score": 0.0,
            "rejected": True,
            "reason": "threshold_mismatch",
            "profit_pct": 10.0,
            "cost": 5000.0,
        },
        {
            "pair_id": "medium",
            "poly_title": "a",
            "kalshi_title": "b",
            "score": 0.2,
            "rejected": True,
            "reason": "jurisdiction_mismatch",
            "profit_pct": 2.0,
            "cost": 200.0,
        },
    ]
    seq = iter(canned)
    monkeypatch.setattr(qr, "_score_pair", lambda *_a, **_k: next(seq))
    _patch_pairs(monkeypatch, [_make_pair(f"p{i}", "x", "y") for i in range(3)])

    r = app_client.get("/arb/quality-audit")
    assert r.status_code == 200
    body = r.json()
    assert len(body["top_rejected"]) == 3
    assert body["top_rejected"][0]["pair_id"] == "expensive"


def test_cache_hit_avoids_second_audit(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second call within TTL must not re-invoke ``_audit``."""
    from pfm.arb import quality_router as qr

    pairs = [_make_pair("p1", "Trump 2024 election", "Trump wins 2024")]
    _patch_pairs(monkeypatch, pairs)

    audit_calls = {"n": 0}
    real_audit = qr._audit

    def counting_audit(p: list[dict[str, Any]]):
        audit_calls["n"] += 1
        return real_audit(p)

    monkeypatch.setattr(qr, "_audit", counting_audit)

    r1 = app_client.get("/arb/quality-audit")
    r2 = app_client.get("/arb/quality-audit")
    assert r1.status_code == 200 and r2.status_code == 200
    # Same checked_at proves the cached object was reused.
    assert r1.json()["checked_at"] == r2.json()["checked_at"]
    assert audit_calls["n"] == 1


def test_cache_invalidated_by_include_details_flag(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache key includes ``include_details`` — toggling it re-audits."""
    from pfm.arb import quality_router as qr

    pairs = [_make_pair("p1", "x", "y")]
    _patch_pairs(monkeypatch, pairs)

    calls = {"n": 0}
    real_audit = qr._audit

    def counting_audit(p: list[dict[str, Any]]):
        calls["n"] += 1
        return real_audit(p)

    monkeypatch.setattr(qr, "_audit", counting_audit)

    r1 = app_client.get("/arb/quality-audit")
    r2 = app_client.get("/arb/quality-audit?include_details=true")
    assert r1.status_code == 200 and r2.status_code == 200
    # Different cache keys → two audits.
    assert calls["n"] == 2
    assert r1.json()["pairs"] is None
    assert isinstance(r2.json()["pairs"], list)


def test_empty_pairs_returns_zero_counts(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No live pairs → audited_count=0 and all reason buckets zeroed."""
    _patch_pairs(monkeypatch, [], source="top_arbs")

    r = app_client.get("/arb/quality-audit")
    assert r.status_code == 200
    body = r.json()
    assert body["audited_count"] == 0
    assert body["rejected_count"] == 0
    assert body["high_conf_count"] == 0
    assert body["borderline_count"] == 0
    assert body["top_rejected"] == []
    # All canonical reason keys present and zero.
    assert all(v == 0 for v in body["rejection_breakdown"].values())
    assert body["source"] == "top_arbs"


def test_missing_matchers_returns_503(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When T76/T77 are unavailable the endpoint must 503 with a helpful hint.

    We simulate the failure where it actually originates: at import time
    inside ``_import_matchers``. Monkeypatching ``sys.modules`` so the
    nested import raises lets us verify the real conversion path
    (ImportError → HTTPException(503, hint)).
    """
    import sys

    import pfm.arb_matching.event_similarity as es  # noqa: F401 — ensure imported

    # Force the next import attempt to fail by stashing None.
    monkeypatch.setitem(sys.modules, "pfm.arb_matching.event_similarity", None)
    _patch_pairs(monkeypatch, [_make_pair("p1", "x", "y")])

    r = app_client.get("/arb/quality-audit")
    assert r.status_code == 503
    detail = r.json().get("detail", "")
    # The detail must point the user at the missing modules.
    assert "arb_matching" in detail or "T76" in detail or "T77" in detail


def test_include_details_returns_full_pair_list(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``?include_details=true`` returns the full per-pair score rows."""
    pairs = [
        _make_pair("pair-a", "Trump 2024 election", "Trump wins 2024 presidential"),
        _make_pair("pair-b", "BTC above $80k by 2024", "BTC over $80k by 2024"),
    ]
    _patch_pairs(monkeypatch, pairs)

    r = app_client.get("/arb/quality-audit?include_details=true")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["pairs"], list)
    assert len(body["pairs"]) == 2
    sample = body["pairs"][0]
    assert {"pair_id", "score", "rejected", "reason"} <= set(sample.keys())


def test_default_omits_pairs_field(app_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default response sends ``pairs: null`` to keep the payload small."""
    _patch_pairs(monkeypatch, [_make_pair("pair-a", "x", "y")])

    r = app_client.get("/arb/quality-audit")
    assert r.status_code == 200
    assert r.json()["pairs"] is None


def test_loads_pairs_from_dashboard_state_file(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify the dashboard_state.json loader pulls real-world-shape opportunities."""
    from pfm.arb import quality_router as qr

    state = {
        "opportunities": [
            {
                "arb_key": "k1|poly-elec",
                "name": "Trump wins 2024",
                "poly_slug": "trump-wins-2024",
                "kalshi_ticker": "ELECTION-24-TRUMP",
                "profit_pct": 1.5,
                "cost": 250.0,
            },
            {
                "arb_key": "k2|poly-btc",
                "name": "BTC above $80k by EOY",
                "poly_slug": "btc-above-80k",
                "kalshi_ticker": "BTC-EOY-80",
                "profit_pct": 0.3,
                "cost": 1000.0,
            },
        ]
    }
    p = tmp_path / "dashboard_state.json"
    p.write_text(json.dumps(state))

    pairs = qr._load_pairs_from_dashboard_state(p)
    assert len(pairs) == 2
    assert pairs[0]["pair_id"] == "k1|poly-elec"
    assert pairs[0]["poly_title"] == "Trump wins 2024"
    assert pairs[0]["kalshi_title"] == "Trump wins 2024"
    assert pairs[0]["profit_pct"] == 1.5
    assert pairs[1]["cost"] == 1000.0


def test_dashboard_state_missing_file_returns_empty(tmp_path: Path) -> None:
    from pfm.arb import quality_router as qr

    pairs = qr._load_pairs_from_dashboard_state(tmp_path / "nope.json")
    assert pairs == []


def test_dashboard_state_corrupt_file_returns_empty(tmp_path: Path) -> None:
    from pfm.arb import quality_router as qr

    p = tmp_path / "corrupt.json"
    p.write_text("{not valid json")
    pairs = qr._load_pairs_from_dashboard_state(p)
    assert pairs == []


def test_high_conf_count_increments_for_strong_matches(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A canned score > 0.7 must bump high_conf_count and skip rejection."""
    from pfm.arb import quality_router as qr

    canned_iter = iter(
        [
            {
                "pair_id": "good",
                "poly_title": "a",
                "kalshi_title": "b",
                "score": 0.92,
                "rejected": False,
                "reason": "",
                "profit_pct": 1.0,
                "cost": 10.0,
            },
            {
                "pair_id": "borderline",
                "poly_title": "a",
                "kalshi_title": "b",
                "score": 0.55,
                "rejected": False,
                "reason": "",
                "profit_pct": 1.0,
                "cost": 10.0,
            },
        ]
    )
    monkeypatch.setattr(qr, "_score_pair", lambda *_a, **_k: next(canned_iter))
    _patch_pairs(monkeypatch, [_make_pair("good", "x", "y"), _make_pair("borderline", "x", "y")])

    r = app_client.get("/arb/quality-audit")
    body = r.json()
    assert body["high_conf_count"] == 1
    assert body["borderline_count"] == 1
    assert body["rejected_count"] == 0


def test_router_can_mount_on_app_without_errors() -> None:
    """Smoke: importing + mounting the router must not raise."""
    from pfm.arb.quality_router import router

    app = FastAPI()
    app.include_router(router)
    # The endpoint must be registered under the documented path.
    paths = [r.path for r in app.routes]
    assert "/arb/quality-audit" in paths

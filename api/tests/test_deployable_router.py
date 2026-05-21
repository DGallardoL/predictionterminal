"""Tests for ``pfm.strategies.deployable_router`` (W11-24).

Symmetrical to W11-23 (anti-alpha). Covers:

* 200 default response and schema shape
* Tier filter (``?tier=A_GOLD`` / ``?tier=A_STRUCTURAL`` / ``?tier=B_VALIDATED``)
* Fallback when ``alpha_strategies.json`` is missing
* Fallback when the JSON has zero A-tier / B_VALIDATED rows
* 5-minute cache hit and TTL expiry
* ``?min_sharpe=0.5`` floor filter
* Validation rejection for the closed-set tier vocab
* Source flag (``json`` vs ``fallback``)
* Item ordering (most conservative-sharpe first)
* Robustness envelope derivation
* Caveat / theory-ref fall-through

Run with ``pytest --noconftest`` to bypass the unrelated repo-wide
``pfm.portfolio.import_router`` bug; tests do not depend on conftest
fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.strategies import deployable_router as dr

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_cache():
    """Reset the module-level cache around every test."""
    dr.clear_cache()
    yield
    dr.clear_cache()


@pytest.fixture
def app() -> FastAPI:
    fastapi_app = FastAPI()
    fastapi_app.include_router(dr.router)
    return fastapi_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a deterministic ``alpha_strategies.json`` to a tmp path.

    Includes one row per tier of interest plus a couple of non-deployable
    tiers so the filter is exercised.
    """
    payload = {
        "generated": "2026-05-16",
        "strategies": [
            {
                "pair_id": "fed_target_40_eoy__fed_target_45_eoy",
                "tier": "A_STRUCTURAL",
                "a_name": "Fed funds upper bound = 4.0% at end-2026",
                "b_name": "Fed funds upper bound = 4.5% at end-2026",
                "oos_sharpe": 5.343,
                "full_sharpe": 3.285,
                "sharpe_ci_lo": 0.0,
                "sharpe_ci_hi": 9.68,
                "n_obs": 252,
                "rationale": "Strike-family structural cointegration.",
                "theory_reference": "Carr-Madan 1998",
            },
            {
                "pair_id": "bp_acquired__fannie_mae_ipo_before",
                "tier": "B_VALIDATED",
                "a_name": "BP acquired before 2027",
                "b_name": "Fannie Mae IPO before 2027",
                "oos_sharpe": 5.12,
                "full_sharpe": 2.769,
                "sharpe_ci_lo": 1.517,
                "sharpe_ci_hi": 10.12,
                "n_obs": 189,
                "rationale": "Cross-theme cointegration.",
                "theory_reference": "Schwartz 1997",
            },
            {
                "pair_id": "manuel_a__renan_b",
                "tier": "A_GOLD",
                "a_name": "Manuel A wins",
                "b_name": "Renan B wins",
                "oos_sharpe": 3.2,
                "full_sharpe": 2.1,
                "sharpe_ci_lo": 0.9,
                "sharpe_ci_hi": 6.5,
                "quarters_passed": 4,
                "deflated_sharpe": 1.45,
                "rationale": "Hand-tuned gold tier.",
                "theory_reference": "Bailey-Lopez de Prado 2014",
            },
            {
                "pair_id": "low_sharpe_b_val",
                "tier": "B_VALIDATED",
                "a_name": "Low Sharpe B",
                "b_name": "Pair",
                "oos_sharpe": 0.30,
                "full_sharpe": 0.20,
                "sharpe_ci_lo": 0.1,
                "n_obs": 63,
                "rationale": "Marginal alpha.",
                "theory_reference": "Engle-Granger 1987",
            },
            # Non-deployable tiers — must be filtered out.
            {
                "pair_id": "tentative_c_one",
                "tier": "C_TENTATIVE",
                "a_name": "Tentative",
                "b_name": "Pair",
                "oos_sharpe": 1.0,
                "sharpe_ci_lo": -0.2,
            },
            {
                "pair_id": "raw_d_one",
                "tier": "D_RAW",
                "a_name": "Raw",
                "b_name": "Pair",
                "oos_sharpe": 0.5,
            },
        ],
    }
    path = tmp_path / "alpha_strategies.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setenv("PFM_ALPHA_STRATEGIES_JSON", str(path))
    return path


# ---------------------------------------------------------------------------
# 1 — 200 default response
# ---------------------------------------------------------------------------


def test_200_default(client: TestClient, sample_json: Path):
    resp = client.get("/strategies/deployable-list")
    assert resp.status_code == 200
    data = resp.json()
    # Only the 4 deployable rows from the fixture make the cut.
    assert data["count"] == 4
    assert data["source"] == "json"
    assert len(data["items"]) == 4
    tiers = {item["tier"] for item in data["items"]}
    assert tiers == {"A_STRUCTURAL", "B_VALIDATED", "A_GOLD"}


# ---------------------------------------------------------------------------
# 2 — Schema shape
# ---------------------------------------------------------------------------


def test_response_schema_shape(client: TestClient, sample_json: Path):
    data = client.get("/strategies/deployable-list").json()
    item = data["items"][0]
    assert {"pair_id", "tier", "label", "caveat", "robustness", "theory_ref"} <= set(item)
    rb = item["robustness"]
    assert {"quarters_passed", "min_sharpe", "deflated_sharpe"} <= set(rb)
    assert isinstance(rb["quarters_passed"], int)
    assert isinstance(rb["min_sharpe"], (int, float))
    assert isinstance(rb["deflated_sharpe"], (int, float))


# ---------------------------------------------------------------------------
# 3 — Tier filter ?tier=A_GOLD
# ---------------------------------------------------------------------------


def test_filter_by_tier_a_gold(client: TestClient, sample_json: Path):
    data = client.get("/strategies/deployable-list?tier=A_GOLD").json()
    assert data["count"] == 1
    assert data["items"][0]["pair_id"] == "manuel_a__renan_b"
    assert data["items"][0]["tier"] == "A_GOLD"


def test_filter_by_tier_a_structural(client: TestClient, sample_json: Path):
    data = client.get("/strategies/deployable-list?tier=A_STRUCTURAL").json()
    assert data["count"] == 1
    assert data["items"][0]["tier"] == "A_STRUCTURAL"


def test_filter_by_tier_b_validated(client: TestClient, sample_json: Path):
    data = client.get("/strategies/deployable-list?tier=B_VALIDATED").json()
    assert data["count"] == 2  # bp_acquired + low_sharpe_b_val
    assert all(it["tier"] == "B_VALIDATED" for it in data["items"])


def test_invalid_tier_rejected(client: TestClient, sample_json: Path):
    resp = client.get("/strategies/deployable-list?tier=C_TENTATIVE")
    # Closed-set Literal rejects unknown tiers.
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 4 — Fallback paths
# ---------------------------------------------------------------------------


def test_fallback_when_json_missing(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("PFM_ALPHA_STRATEGIES_JSON", str(tmp_path / "does-not-exist.json"))
    data = client.get("/strategies/deployable-list").json()
    assert data["source"] == "fallback"
    assert data["count"] == 4
    pair_ids = {it["pair_id"] for it in data["items"]}
    assert pair_ids == {
        "election-binary-momentum",
        "fed-decision-straddle-proxy",
        "sports-event-mean-reversion",
        "earnings-surprise-odds-vs-iv",
    }


def test_fallback_when_json_has_no_deployable_rows(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"strategies": [{"pair_id": "x", "tier": "D_RAW"}]}))
    monkeypatch.setenv("PFM_ALPHA_STRATEGIES_JSON", str(path))
    data = client.get("/strategies/deployable-list").json()
    assert data["source"] == "fallback"
    assert data["count"] == 4


def test_fallback_when_json_is_malformed(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "broken.json"
    path.write_text("{not valid json")
    monkeypatch.setenv("PFM_ALPHA_STRATEGIES_JSON", str(path))
    data = client.get("/strategies/deployable-list").json()
    assert data["source"] == "fallback"
    assert data["count"] == 4


# ---------------------------------------------------------------------------
# 5 — Caching
# ---------------------------------------------------------------------------


def test_cache_hit_within_ttl(client: TestClient, sample_json: Path):
    """Second call returns the cached object — no JSON re-read."""
    first = client.get("/strategies/deployable-list").json()
    # Now mutate the file on disk; without caching the count would change.
    payload = json.loads(sample_json.read_text())
    payload["strategies"] = []
    sample_json.write_text(json.dumps(payload))
    second = client.get("/strategies/deployable-list").json()
    assert first == second  # cache survived the disk mutation


def test_cache_expires_after_ttl(
    client: TestClient, sample_json: Path, monkeypatch: pytest.MonkeyPatch
):
    """After ``_CACHE_TTL_SECONDS`` elapses, the response is recomputed."""
    fake_clock = {"t": 0.0}

    def _now() -> float:
        return fake_clock["t"]

    monkeypatch.setattr(dr, "_PERF_COUNTER", _now)
    dr.clear_cache()

    first = client.get("/strategies/deployable-list").json()
    assert first["count"] == 4
    assert first["source"] == "json"

    # Drop deployable rows from the JSON and skip past TTL.
    payload = json.loads(sample_json.read_text())
    payload["strategies"] = [
        s for s in payload["strategies"] if s["tier"] not in dr._DEPLOYABLE_TIERS
    ]
    sample_json.write_text(json.dumps(payload))
    fake_clock["t"] += dr._CACHE_TTL_SECONDS + 1.0

    second = client.get("/strategies/deployable-list").json()
    assert second["source"] == "fallback"


def test_cache_keyed_by_filters(client: TestClient, sample_json: Path):
    """Different (tier, min_sharpe) combos must not collide in cache."""
    a = client.get("/strategies/deployable-list?tier=A_GOLD").json()
    b = client.get("/strategies/deployable-list?tier=B_VALIDATED").json()
    assert a["count"] == 1
    assert b["count"] == 2
    # Hit the same keys again — must still be the right shape.
    a2 = client.get("/strategies/deployable-list?tier=A_GOLD").json()
    assert a == a2


# ---------------------------------------------------------------------------
# 6 — min_sharpe filter
# ---------------------------------------------------------------------------


def test_min_sharpe_filter_default(client: TestClient, sample_json: Path):
    data = client.get("/strategies/deployable-list?min_sharpe=0.5").json()
    # low_sharpe_b_val has min_sharpe=0.1 (sharpe_ci_lo) -> drops
    # A_STRUCTURAL row has sharpe_ci_lo=0.0 -> drops
    # A_GOLD row has 0.9, BP row has 1.517 -> survive
    assert data["count"] == 2
    survivors = {it["pair_id"] for it in data["items"]}
    assert survivors == {"manuel_a__renan_b", "bp_acquired__fannie_mae_ipo_before"}


def test_min_sharpe_filter_zero_keeps_all(client: TestClient, sample_json: Path):
    data = client.get("/strategies/deployable-list?min_sharpe=0").json()
    assert data["count"] == 4  # everyone has min_sharpe >= 0 in fixture


def test_min_sharpe_filter_too_high_returns_empty(client: TestClient, sample_json: Path):
    data = client.get("/strategies/deployable-list?min_sharpe=9.0").json()
    assert data["count"] == 0
    assert data["items"] == []


# ---------------------------------------------------------------------------
# 7 — Ordering, label derivation, defaults
# ---------------------------------------------------------------------------


def test_items_sorted_by_min_sharpe_desc(client: TestClient, sample_json: Path):
    items = client.get("/strategies/deployable-list").json()["items"]
    sharpes = [it["robustness"]["min_sharpe"] for it in items]
    assert sharpes == sorted(sharpes, reverse=True)


def test_label_is_a_name_pipe_b_name(client: TestClient, sample_json: Path):
    items = client.get("/strategies/deployable-list?tier=A_GOLD").json()["items"]
    assert items[0]["label"] == "Manuel A wins | Renan B wins"


def test_quarters_passed_explicit_field_wins_over_n_obs(client: TestClient, sample_json: Path):
    a_gold = next(
        it
        for it in client.get("/strategies/deployable-list").json()["items"]
        if it["pair_id"] == "manuel_a__renan_b"
    )
    assert a_gold["robustness"]["quarters_passed"] == 4


def test_robustness_envelope_falls_through_to_n_obs(client: TestClient, sample_json: Path):
    a_struct = next(
        it
        for it in client.get("/strategies/deployable-list").json()["items"]
        if it["tier"] == "A_STRUCTURAL"
    )
    # n_obs=252 -> 252 // 63 == 4 quarters
    assert a_struct["robustness"]["quarters_passed"] == 4


def test_theory_ref_passes_through_from_theory_reference(client: TestClient, sample_json: Path):
    bp = next(
        it
        for it in client.get("/strategies/deployable-list").json()["items"]
        if it["pair_id"] == "bp_acquired__fannie_mae_ipo_before"
    )
    assert "Schwartz 1997" in bp["theory_ref"]


def test_fallback_items_carry_caveats(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("PFM_ALPHA_STRATEGIES_JSON", str(tmp_path / "missing.json"))
    items = client.get("/strategies/deployable-list").json()["items"]
    election = next(it for it in items if it["pair_id"] == "election-binary-momentum")
    assert "Capacity-limited" in election["caveat"]
    assert election["robustness"]["quarters_passed"] == 4

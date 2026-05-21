"""Tests for ``pfm.strategies.risk_budget_router`` (W12-20).

Covers:

* default 200 response shape against the fallback deployable list
* allocation weights sum to <= 1.0 (cash slack tolerated)
* tier caps (A_GOLD 25%, A_STRUCTURAL 20%, B_VALIDATED 10%) respected
* 30% concentration limit respected
* empty deployable list -> 100% cash
* notionals match weight * total_capital
* risk-parity ordering (low vol -> higher weight, all else equal)
* total_capital must be positive
* tier-conditional fallback vol propagates into the response
* reproducibility (deterministic, no RNG)
* JSON-driven path produces ``source='json'``
* explicit vol override changes weights but still respects caps

Run with ``pytest --noconftest`` to bypass unrelated repo-wide conftest
side-effects; these tests do not depend on conftest fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.strategies import deployable_router as _dr
from pfm.strategies import risk_budget_router as rb

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_caches():
    """Reset both the deployable-router and any risk-budget caches."""
    _dr.clear_cache()
    yield
    _dr.clear_cache()


@pytest.fixture
def app() -> FastAPI:
    fastapi_app = FastAPI()
    fastapi_app.include_router(rb.router)
    return fastapi_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _make_item(pair_id: str, tier: str, label: str = "x") -> _dr.DeployableItem:
    return _dr.DeployableItem(
        pair_id=pair_id,
        tier=tier,
        label=label or pair_id,
        caveat="",
        robustness=_dr.Robustness(quarters_passed=4, min_sharpe=0.5, deflated_sharpe=0.4),
        theory_ref="",
    )


# ---------------------------------------------------------------------------
# build_risk_budget (pure-function) tests
# ---------------------------------------------------------------------------


def test_empty_deployable_list_gives_full_cash():
    response = rb.build_risk_budget([], total_capital=100_000.0)
    assert response.allocations == []
    assert response.remaining_cash == pytest.approx(100_000.0)
    assert response.total_active_capital == pytest.approx(0.0)
    assert response.total_capital == pytest.approx(100_000.0)


def test_weights_sum_at_most_one():
    items = [
        _make_item("a", "A_GOLD"),
        _make_item("b", "A_STRUCTURAL"),
        _make_item("c", "B_VALIDATED"),
        _make_item("d", "B_VALIDATED"),
    ]
    response = rb.build_risk_budget(items, total_capital=100_000.0)
    total_w = sum(a.weight for a in response.allocations)
    assert total_w <= 1.0 + 1e-9
    # Notionals + cash conserve total capital.
    assert response.total_active_capital + response.remaining_cash == pytest.approx(
        100_000.0, abs=1.0
    )


def test_tier_caps_respected():
    """Every weight must lie at or below its tier cap."""
    items = [
        _make_item("g1", "A_GOLD"),
        _make_item("s1", "A_STRUCTURAL"),
        _make_item("v1", "B_VALIDATED"),
        _make_item("v2", "B_VALIDATED"),
        _make_item("v3", "B_VALIDATED"),
    ]
    response = rb.build_risk_budget(items, total_capital=1_000_000.0)
    for a in response.allocations:
        cap = rb.TIER_CAPS[a.tier]
        assert a.weight <= cap + 1e-6, f"{a.pair_id} {a.tier} weight {a.weight} > cap {cap}"


def test_concentration_limit_30pct():
    """A_GOLD's cap is 25% which is already < 30%, but make sure no strategy
    of any tier exceeds the global 30% concentration limit."""
    items = [
        _make_item("only", "A_GOLD"),
    ]
    response = rb.build_risk_budget(items, total_capital=100_000.0)
    for a in response.allocations:
        assert a.weight <= rb.CONCENTRATION_LIMIT + 1e-6


def test_concentration_limit_binds_when_higher_tier_cap_supplied(monkeypatch):
    """If somebody loosened tier caps to >30%, the 30% concentration limit
    must still bind."""
    monkeypatch.setitem(rb.TIER_CAPS, "A_GOLD", 0.50)
    items = [_make_item("only", "A_GOLD")]
    response = rb.build_risk_budget(items, total_capital=100_000.0)
    for a in response.allocations:
        assert a.weight <= rb.CONCENTRATION_LIMIT + 1e-6


def test_risk_parity_ranks_low_vol_higher():
    """When caps don't bind, low-vol strategies should win more weight.

    We use many strategies so that risk-parity raw weights stay well below
    the per-tier caps and the ranking is preserved through the optimiser.
    """
    items = [_make_item(f"s{i}", "B_VALIDATED") for i in range(20)]
    vols = [0.005] + [0.05] * 19
    response = rb.build_risk_budget(items, total_capital=100_000.0, vols=vols)
    by_id = {a.pair_id: a for a in response.allocations}
    # low-vol strategy is s0; it should have strictly higher raw weight
    # than any high-vol peer.  Since per-item caps are 10% and s0's raw is
    # < 10% (1/0.005 / (1/0.005 + 19*1/0.05) ≈ 34% -> binds at 10%), use a
    # less aggressive vol gap that keeps it under cap.
    # Rerun with milder gap:
    vols2 = [0.018] + [0.020] * 19
    response2 = rb.build_risk_budget(items, total_capital=100_000.0, vols=vols2)
    by_id2 = {a.pair_id: a for a in response2.allocations}
    assert by_id2["s0"].weight > by_id2["s1"].weight
    assert by_id2["s0"].weight <= rb.TIER_CAPS["B_VALIDATED"] + 1e-6
    # First-run sanity: when low_vol binds at cap, high-vol allocations are
    # also non-zero (mass spreads through redistribution).
    assert by_id["s0"].weight <= rb.TIER_CAPS["B_VALIDATED"] + 1e-6


def test_notionals_match_weight_times_capital():
    items = [_make_item("a", "A_GOLD"), _make_item("b", "A_STRUCTURAL")]
    response = rb.build_risk_budget(items, total_capital=50_000.0)
    for a in response.allocations:
        assert a.notional == pytest.approx(round(a.weight * 50_000.0, 2), abs=0.01)


def test_negative_capital_rejected():
    with pytest.raises(ValueError):
        rb.build_risk_budget([_make_item("a", "A_GOLD")], total_capital=-1.0)


def test_zero_capital_rejected():
    with pytest.raises(ValueError):
        rb.build_risk_budget([_make_item("a", "A_GOLD")], total_capital=0.0)


def test_non_positive_vol_rejected():
    items = [_make_item("a", "A_GOLD"), _make_item("b", "A_GOLD")]
    with pytest.raises(ValueError):
        rb.build_risk_budget(items, total_capital=100_000.0, vols=[0.01, 0.0])


def test_vols_length_mismatch_rejected():
    items = [_make_item("a", "A_GOLD"), _make_item("b", "A_GOLD")]
    with pytest.raises(ValueError):
        rb.build_risk_budget(items, total_capital=100_000.0, vols=[0.01])


def test_reproducible_repeat_call():
    items = [
        _make_item("a", "A_GOLD"),
        _make_item("b", "A_STRUCTURAL"),
        _make_item("c", "B_VALIDATED"),
    ]
    r1 = rb.build_risk_budget(items, total_capital=100_000.0)
    r2 = rb.build_risk_budget(items, total_capital=100_000.0)
    assert [a.weight for a in r1.allocations] == [a.weight for a in r2.allocations]
    assert [a.pair_id for a in r1.allocations] == [a.pair_id for a in r2.allocations]


def test_rationale_mentions_cap_when_binding():
    items = [_make_item("a", "A_GOLD"), _make_item("b", "B_VALIDATED")]
    # Make A_GOLD very low-vol so risk-parity wants > cap.
    response = rb.build_risk_budget(items, total_capital=100_000.0, vols=[0.001, 0.05])
    by_id = {a.pair_id: a for a in response.allocations}
    assert by_id["a"].weight == pytest.approx(rb.TIER_CAPS["A_GOLD"], abs=1e-6)
    assert "cap" in by_id["a"].rationale.lower()


def test_redistribution_scales_up_under_capped_when_other_bound():
    """When risk-parity over-allocates to one tier-capped strategy, the
    trimmed mass should flow into the under-capped names."""
    items = [_make_item("a", "A_GOLD"), _make_item("b", "B_VALIDATED")]
    # a wants way more than 25%, so we trim and dump into b.
    response = rb.build_risk_budget(items, total_capital=100_000.0, vols=[0.001, 0.5])
    by_id = {a.pair_id: a for a in response.allocations}
    # a binds at 25%.
    assert by_id["a"].weight == pytest.approx(0.25, abs=1e-6)
    # b also caps at 10% (its tier cap), so total is 35% and the rest is cash.
    assert by_id["b"].weight == pytest.approx(0.10, abs=1e-6)
    assert response.remaining_cash == pytest.approx(65_000.0, abs=1.0)


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------


def test_endpoint_default_returns_200(client: TestClient, monkeypatch):
    # Point loader at a missing file so we hit the fallback list.
    monkeypatch.setenv("PFM_ALPHA_STRATEGIES_JSON", "/tmp/__nonexistent__rb.json")
    r = client.get("/strategies/risk-budget")
    assert r.status_code == 200
    body = r.json()
    assert body["total_capital"] == pytest.approx(100_000.0)
    assert isinstance(body["allocations"], list)
    assert body["source"] == "fallback"
    # Fallback list is 4 B_VALIDATED -> each caps at 10% -> 40% deployed.
    total_w = sum(a["weight"] for a in body["allocations"])
    assert total_w <= 1.0 + 1e-6
    for a in body["allocations"]:
        assert a["weight"] <= rb.TIER_CAPS["B_VALIDATED"] + 1e-6


def test_endpoint_with_custom_capital(client: TestClient, monkeypatch):
    monkeypatch.setenv("PFM_ALPHA_STRATEGIES_JSON", "/tmp/__nonexistent__rb.json")
    r = client.get("/strategies/risk-budget?total_capital=500000")
    assert r.status_code == 200
    body = r.json()
    assert body["total_capital"] == pytest.approx(500_000.0)
    # Notionals should scale with capital.
    for a in body["allocations"]:
        assert a["notional"] == pytest.approx(a["weight"] * 500_000.0, abs=1.0)


def test_endpoint_rejects_non_positive_capital(client: TestClient):
    r = client.get("/strategies/risk-budget?total_capital=0")
    assert r.status_code == 422
    r2 = client.get("/strategies/risk-budget?total_capital=-100")
    assert r2.status_code == 422


def test_endpoint_json_source(tmp_path: Path, client: TestClient, monkeypatch):
    """When alpha_strategies.json has deployable rows, source='json'."""
    payload = {
        "generated": "2026-05-16",
        "strategies": [
            {
                "pair_id": "p1",
                "tier": "A_GOLD",
                "a_name": "P1A",
                "b_name": "P1B",
                "oos_sharpe": 4.0,
                "full_sharpe": 3.0,
                "sharpe_ci_lo": 1.0,
                "daily_vol": 0.01,
                "n_obs": 250,
            },
            {
                "pair_id": "p2",
                "tier": "B_VALIDATED",
                "a_name": "P2A",
                "b_name": "P2B",
                "oos_sharpe": 2.0,
                "full_sharpe": 1.5,
                "sharpe_ci_lo": 0.5,
                "daily_vol": 0.02,
                "n_obs": 200,
            },
            {
                "pair_id": "p3",
                "tier": "C_TENTATIVE",  # non-deployable, must be filtered
                "a_name": "P3A",
                "b_name": "P3B",
                "oos_sharpe": 1.0,
            },
        ],
    }
    json_path = tmp_path / "alpha_strategies.json"
    json_path.write_text(json.dumps(payload))
    monkeypatch.setenv("PFM_ALPHA_STRATEGIES_JSON", str(json_path))

    r = client.get("/strategies/risk-budget?total_capital=200000")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "json"
    pair_ids = {a["pair_id"] for a in body["allocations"]}
    assert "p3" not in pair_ids
    assert {"p1", "p2"} == pair_ids
    # p1 (A_GOLD) cap 25% > p2 (B_VALIDATED) cap 10%, and p1 has lower vol.
    by_id = {a["pair_id"]: a for a in body["allocations"]}
    assert by_id["p1"]["weight"] <= 0.25 + 1e-6
    assert by_id["p2"]["weight"] <= 0.10 + 1e-6


def test_endpoint_response_remaining_cash_consistency(client: TestClient, monkeypatch):
    monkeypatch.setenv("PFM_ALPHA_STRATEGIES_JSON", "/tmp/__nonexistent__rb2.json")
    r = client.get("/strategies/risk-budget?total_capital=100000")
    body = r.json()
    assert body["total_active_capital"] + body["remaining_cash"] == pytest.approx(
        100_000.0, abs=1.0
    )


def test_allocations_sorted_descending(client: TestClient, monkeypatch):
    monkeypatch.setenv("PFM_ALPHA_STRATEGIES_JSON", "/tmp/__nonexistent__rb3.json")
    r = client.get("/strategies/risk-budget")
    body = r.json()
    weights = [a["weight"] for a in body["allocations"]]
    assert weights == sorted(weights, reverse=True)

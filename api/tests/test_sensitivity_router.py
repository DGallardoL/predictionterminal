"""Tests for ``pfm.strategies.sensitivity_router`` (W13-16).

Covers:

* pure-function ``compute_sensitivity`` shape + central-difference math
* synthetic-DGP recovery: known closed-form gradients are recovered to
  within finite-difference truncation error
* baselines absorbed from a JSON fixture (``alpha_strategies.json`` shape)
* override path: ad-hoc query params let callers evaluate strategies that
  are not in the JSON catalog
* 404 path: unknown pair_id with no overrides
* gradient-norm robustness ranking: flatter strategies report smaller norms
* perturbation parameter is honoured (size of ±step)
* invalid perturbation rejected
* deterministic / reproducible (no RNG)
* HTTP integration via TestClient
* proxy PnL function basic properties (monotone in z near peak, etc.)
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.strategies import sensitivity_router as sr

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_override():
    """Always start tests with the proxy PnL active (no override)."""
    sr.set_pnl_override(None)
    yield
    sr.set_pnl_override(None)


@pytest.fixture
def app() -> FastAPI:
    fast_app = FastAPI()
    fast_app.include_router(sr.router)
    return fast_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture
def json_strategies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a tiny alpha_strategies.json fixture and point the env var at it."""
    payload = {
        "strategies": [
            {
                "pair_id": "synthetic-alpha-1",
                "a_name": "Synthetic Alpha 1",
                "tier": "A_GOLD",
                "suggested_allocation": 0.20,
                "rule_entry_z": 1.5,
                "deployment_params": {"epsilon": 0.015},
            },
            {
                "pair_id": "synthetic-alpha-no-params",
                "a_name": "Synthetic Alpha (no params)",
                "tier": "B_VALIDATED",
                # No suggested_allocation, no rule_entry_z, no eps -> defaults.
            },
        ]
    }
    path = tmp_path / "alpha_strategies.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv(sr._STRATEGIES_PATH_ENV, str(path))
    return path


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_compute_sensitivity_shape_default_params():
    """All three parameters are perturbed and ordered consistently."""
    baseline = dict(sr.DEFAULT_BASELINES)
    rows, grad_norm, pnl_b = sr.compute_sensitivity(baseline)
    assert [r.name for r in rows] == list(sr.PARAM_NAMES)
    assert grad_norm >= 0.0
    assert math.isfinite(pnl_b)
    # Each row's perturbed_low/high straddle baseline.
    for r in rows:
        assert r.perturbed_low < r.baseline < r.perturbed_high
        # Central-difference local_gradient equals
        # (pnl_high - pnl_low) / (perturbed_high - perturbed_low).
        num = (r.pnl_delta_high + r.pnl_baseline) - (r.pnl_delta_low + r.pnl_baseline)
        denom = r.perturbed_high - r.perturbed_low
        assert r.local_gradient == pytest.approx(num / denom)


def test_synthetic_linear_strategy_recovers_known_gradient():
    """Plug in a linear PnL ``2 f + 3 z - 4 ε``; gradients must come out 2, 3, -4."""

    def linear_pnl(p: Mapping[str, float]) -> float:
        return 2.0 * p["kelly_cap"] + 3.0 * p["z_threshold"] - 4.0 * p["epsilon"]

    baseline = {"kelly_cap": 0.30, "z_threshold": 1.5, "epsilon": 0.02}
    rows, _, _ = sr.compute_sensitivity(baseline, pnl_fn=linear_pnl)
    by_name = {r.name: r for r in rows}
    assert by_name["kelly_cap"].local_gradient == pytest.approx(2.0)
    assert by_name["z_threshold"].local_gradient == pytest.approx(3.0)
    assert by_name["epsilon"].local_gradient == pytest.approx(-4.0)


def test_synthetic_quadratic_strategy_central_diff_at_baseline():
    """For PnL = f^2, central diff at f0 gives df = 2 f0 (exactly, since the
    error term is O(h^2 f''') = 0 for a pure quadratic.
    """

    def quad_pnl(p: Mapping[str, float]) -> float:
        return p["kelly_cap"] ** 2

    baseline = {"kelly_cap": 0.40}
    rows, _, pnl_b = sr.compute_sensitivity(baseline, pnl_fn=quad_pnl)
    assert pnl_b == pytest.approx(0.16)
    assert rows[0].local_gradient == pytest.approx(0.80, rel=1e-9)


def test_perturbation_size_scales_step_proportionally():
    """A 20% perturbation creates steps twice the size of the default 10%."""
    baseline = {"kelly_cap": 0.50, "z_threshold": 2.0, "epsilon": 0.01}
    r10, _, _ = sr.compute_sensitivity(baseline, perturbation=0.10)
    r20, _, _ = sr.compute_sensitivity(baseline, perturbation=0.20)
    for a, b in zip(r10, r20, strict=True):
        # |perturbed_high - baseline| at 20% is double that at 10%.
        d10 = a.perturbed_high - a.baseline
        d20 = b.perturbed_high - b.baseline
        assert d20 == pytest.approx(2.0 * d10, rel=1e-9)


def test_invalid_perturbation_rejected():
    baseline = dict(sr.DEFAULT_BASELINES)
    with pytest.raises(ValueError):
        sr.compute_sensitivity(baseline, perturbation=0.0)
    with pytest.raises(ValueError):
        sr.compute_sensitivity(baseline, perturbation=-0.05)
    with pytest.raises(ValueError):
        sr.compute_sensitivity(baseline, perturbation=1.5)


def test_gradient_norm_ranks_robustness():
    """Flat-PnL strategy reports a strictly smaller gradient_norm than a
    highly-sensitive one.
    """

    def flat_pnl(_: Mapping[str, float]) -> float:
        return 7.0

    def steep_pnl(p: Mapping[str, float]) -> float:
        return 100.0 * p["kelly_cap"] + 50.0 * p["z_threshold"]

    baseline = {"kelly_cap": 0.20, "z_threshold": 2.0, "epsilon": 0.01}
    _, norm_flat, _ = sr.compute_sensitivity(baseline, pnl_fn=flat_pnl)
    _, norm_steep, _ = sr.compute_sensitivity(baseline, pnl_fn=steep_pnl)
    assert norm_flat == pytest.approx(0.0, abs=1e-12)
    assert norm_steep > norm_flat


def test_reproducible_no_randomness():
    """Repeat calls with identical inputs return identical outputs."""
    baseline = dict(sr.DEFAULT_BASELINES)
    r1, n1, p1 = sr.compute_sensitivity(baseline)
    r2, n2, p2 = sr.compute_sensitivity(baseline)
    assert n1 == n2
    assert p1 == p2
    for a, b in zip(r1, r2, strict=True):
        assert a.model_dump() == b.model_dump()


def test_only_supplied_params_are_perturbed():
    """Baseline missing keys -> they are silently skipped from the rows."""
    baseline = {"kelly_cap": 0.25}  # missing z_threshold and epsilon
    rows, _, _ = sr.compute_sensitivity(baseline)
    assert [r.name for r in rows] == ["kelly_cap"]


def test_proxy_pnl_basic_properties():
    """The built-in proxy is C¹, positive in the operating region, and the
    z_score piece peaks near z ≈ 1.
    """
    base = {"kelly_cap": 0.25, "z_threshold": 1.0, "epsilon": 0.01}
    pnl_at_1 = sr.proxy_pnl(base)
    pnl_at_0p5 = sr.proxy_pnl({**base, "z_threshold": 0.5})
    pnl_at_3 = sr.proxy_pnl({**base, "z_threshold": 3.0})
    assert pnl_at_1 > 0.0
    # z=1 should beat z=3 (rapid Φ(-z) decay) and z=0.5 (weak per-trade edge).
    assert pnl_at_1 > pnl_at_3
    assert pnl_at_1 > pnl_at_0p5
    # Larger ε bleeds signal monotonically.
    assert sr.proxy_pnl({**base, "epsilon": 0.005}) > sr.proxy_pnl({**base, "epsilon": 0.05})


# ---------------------------------------------------------------------------
# HTTP / endpoint tests
# ---------------------------------------------------------------------------


def test_endpoint_returns_404_for_unknown_pair_no_overrides(
    client: TestClient, json_strategies: Path
):
    """Unknown strategy + no overrides → 404 (don't silently return defaults)."""
    resp = client.get("/strategies/does-not-exist/sensitivity")
    assert resp.status_code == 404


def test_endpoint_known_pair_uses_json_baselines(client: TestClient, json_strategies: Path):
    """A pair_id with suggested_allocation/rule_entry_z/eps wires those in."""
    resp = client.get("/strategies/synthetic-alpha-1/sensitivity")
    assert resp.status_code == 200
    body = resp.json()
    assert body["pair_id"] == "synthetic-alpha-1"
    assert body["source"] == "json"
    by_name = {r["name"]: r for r in body["params"]}
    assert by_name["kelly_cap"]["baseline"] == pytest.approx(0.20)
    assert by_name["z_threshold"]["baseline"] == pytest.approx(1.5)
    assert by_name["epsilon"]["baseline"] == pytest.approx(0.015)
    # ±10% by default.
    assert by_name["kelly_cap"]["perturbed_low"] == pytest.approx(0.18)
    assert by_name["kelly_cap"]["perturbed_high"] == pytest.approx(0.22)


def test_endpoint_falls_back_to_defaults_when_row_has_no_params(
    client: TestClient, json_strategies: Path
):
    """A row with no allocation/z/eps fields → DEFAULT_BASELINES, source='defaults'."""
    resp = client.get("/strategies/synthetic-alpha-no-params/sensitivity")
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "defaults"
    by_name = {r["name"]: r for r in body["params"]}
    assert by_name["kelly_cap"]["baseline"] == pytest.approx(sr.DEFAULT_BASELINES["kelly_cap"])
    assert by_name["z_threshold"]["baseline"] == pytest.approx(sr.DEFAULT_BASELINES["z_threshold"])
    assert by_name["epsilon"]["baseline"] == pytest.approx(sr.DEFAULT_BASELINES["epsilon"])


def test_endpoint_override_path_evaluates_ad_hoc_params(client: TestClient, json_strategies: Path):
    """Caller-supplied overrides win even when the row exists."""
    resp = client.get(
        "/strategies/synthetic-alpha-1/sensitivity",
        params={"kelly_cap": 0.10, "z_threshold": 2.5, "epsilon": 0.02},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "override"
    by_name = {r["name"]: r for r in body["params"]}
    assert by_name["kelly_cap"]["baseline"] == pytest.approx(0.10)
    assert by_name["z_threshold"]["baseline"] == pytest.approx(2.5)
    assert by_name["epsilon"]["baseline"] == pytest.approx(0.02)


def test_endpoint_override_for_unknown_pair_succeeds(client: TestClient, json_strategies: Path):
    """Unknown strategy + overrides → 200 with source='override'."""
    resp = client.get(
        "/strategies/does-not-exist/sensitivity",
        params={"kelly_cap": 0.25},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["pair_id"] == "does-not-exist"
    assert body["source"] == "override"


def test_endpoint_response_shape_matches_spec(client: TestClient, json_strategies: Path):
    """Response schema must match the W13-16 spec exactly."""
    resp = client.get("/strategies/synthetic-alpha-1/sensitivity")
    assert resp.status_code == 200
    body = resp.json()
    # Top-level keys.
    assert set(body.keys()) >= {
        "pair_id",
        "perturbation",
        "params",
        "gradient_norm",
        "pnl_baseline",
        "source",
    }
    # Per-param row keys per the spec.
    for row in body["params"]:
        assert set(row.keys()) >= {
            "name",
            "baseline",
            "perturbed_low",
            "perturbed_high",
            "pnl_delta_low",
            "pnl_delta_high",
        }


def test_endpoint_perturbation_query_param_honoured(client: TestClient, json_strategies: Path):
    """Setting ``?perturbation=0.20`` doubles the step size."""
    r10 = client.get("/strategies/synthetic-alpha-1/sensitivity").json()
    r20 = client.get(
        "/strategies/synthetic-alpha-1/sensitivity",
        params={"perturbation": 0.20},
    ).json()
    by10 = {r["name"]: r for r in r10["params"]}
    by20 = {r["name"]: r for r in r20["params"]}
    for name in by10:
        d10 = by10[name]["perturbed_high"] - by10[name]["baseline"]
        d20 = by20[name]["perturbed_high"] - by20[name]["baseline"]
        assert d20 == pytest.approx(2.0 * d10, rel=1e-9)


def test_endpoint_invalid_perturbation_rejected(client: TestClient, json_strategies: Path):
    """``perturbation > 1`` or ``<= 0`` must be rejected by FastAPI validation."""
    r = client.get(
        "/strategies/synthetic-alpha-1/sensitivity",
        params={"perturbation": 1.5},
    )
    assert r.status_code == 422
    r2 = client.get(
        "/strategies/synthetic-alpha-1/sensitivity",
        params={"perturbation": 0.0},
    )
    assert r2.status_code == 422


def test_endpoint_uses_override_pnl_hook(client: TestClient, json_strategies: Path):
    """The HTTP handler routes through ``_PNL_FN_OVERRIDE`` when set."""

    def constant_pnl(_: Mapping[str, float]) -> float:
        return 42.0

    sr.set_pnl_override(constant_pnl)
    try:
        resp = client.get("/strategies/synthetic-alpha-1/sensitivity")
        assert resp.status_code == 200
        body = resp.json()
        assert body["pnl_baseline"] == pytest.approx(42.0)
        # Constant PnL ⇒ zero gradient norm.
        assert body["gradient_norm"] == pytest.approx(0.0, abs=1e-12)
        for row in body["params"]:
            assert row["pnl_delta_low"] == pytest.approx(0.0, abs=1e-12)
            assert row["pnl_delta_high"] == pytest.approx(0.0, abs=1e-12)
    finally:
        sr.set_pnl_override(None)

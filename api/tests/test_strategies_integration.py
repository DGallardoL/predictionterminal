"""End-to-end smoke of every /strategies/* endpoint.

Uses the synthetic factor_a / factor_b fixture (see conftest.py). All IO
is mocked. The point is not to verify quant correctness (the per-module
unit tests do that) — it's to guarantee that every endpoint at least
returns 200 with a sane payload, so refactors can't silently break a
sub-tab.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

PAIR = {"a_id": "factor_a", "b_id": "factor_b", "start": "2025-06-15", "end": "2025-12-15"}


def _post(client: TestClient, path: str, body: dict) -> dict:
    r = client.post(path, json=body)
    assert r.status_code == 200, f"{path} → {r.status_code}: {r.text[:200]}"
    return r.json()


def test_implication(app_client: TestClient) -> None:
    body = {
        "antecedent_id": "factor_a",
        "consequent_id": "factor_b",
        "start": "2025-06-15",
        "end": "2025-12-15",
        "tolerance": 0.02,
    }
    out = _post(app_client, "/strategies/implication", body)
    assert out["verdict"] in {"consistent", "borderline", "violated"}
    assert out["n_obs"] > 100


def test_conditional(app_client: TestClient) -> None:
    body = {**PAIR, "hac_lag": 5}
    out = _post(app_client, "/strategies/conditional", body)
    assert "beta" in out and "beta_ci_lo" in out
    assert out["n_obs"] > 50


def test_bounds(app_client: TestClient) -> None:
    out = _post(app_client, "/strategies/bounds", PAIR)
    assert out["n_obs"] > 0
    for pt in out["series"]:
        assert pt["lower"] <= pt["upper"] + 1e-12


def test_cointegration(app_client: TestClient) -> None:
    out = _post(app_client, "/strategies/cointegration", PAIR)
    assert out["verdict"] in {"cointegrated", "not_cointegrated", "insufficient-data"}


def test_pairs_backtest(app_client: TestClient) -> None:
    body = {
        **PAIR,
        "window": 20,
        "entry_z": 2.0,
        "exit_z": 0.5,
        "stop_z": 4.0,
        "annualisation": 252.0,
        "oos_fraction": 0.30,
    }
    out = _post(app_client, "/strategies/pairs-backtest", body)
    assert "sharpe" in out and "sharpe_is" in out and "sharpe_oos" in out
    assert out["n_obs_is"] + out["n_obs_oos"] <= out["n_obs"]


def test_kalman_hedge(app_client: TestClient) -> None:
    body = {**PAIR, "delta": 1e-4}
    out = _post(app_client, "/strategies/kalman-hedge", body)
    assert out["n_obs"] > 50
    assert "beta_init" in out and "beta_final" in out
    assert len(out["series"]) == out["n_obs"]


def test_mean_reversion(app_client: TestClient) -> None:
    body = {"factor_id": "factor_a", "start": "2025-06-15", "end": "2025-12-15", "vr_q": 2}
    out = _post(app_client, "/strategies/mean-reversion", body)
    assert out["hurst_interpretation"] in {
        "mean_reverting",
        "random_walk",
        "trending",
        "insufficient-data",
    }
    assert out["vr_verdict"] in {
        "mean_reverting",
        "random_walk",
        "momentum",
        "insufficient-data",
    }


def test_ou_bands(app_client: TestClient) -> None:
    body = {**PAIR, "transaction_cost_sigma": 0.10}
    out = _post(app_client, "/strategies/ou-bands", body)
    # OU may report not-stationary on the short synthetic; the response
    # must still be well-formed.
    assert "cointegrated" in out


def test_granger(app_client: TestClient) -> None:
    body = {**PAIR, "max_lag": 5, "alpha": 0.05}
    out = _post(app_client, "/strategies/granger", body)
    assert out["direction"] in {
        "B_causes_A",
        "A_causes_B",
        "bidirectional",
        "neither",
    }


def test_event_model(app_client: TestClient) -> None:
    body = {
        "target_id": "factor_a",
        "factor_ids": ["factor_b"],
        "start": "2025-06-15",
        "end": "2025-12-15",
        "hac_lag": 5,
    }
    out = _post(app_client, "/strategies/event-model", body)
    assert len(out["coefficients"]) == 1
    assert out["coefficients"][0]["factor_id"] == "factor_b"


def test_basket_stat_arb(app_client: TestClient) -> None:
    body = {
        "factor_ids": ["factor_a", "factor_b"],
        "start": "2025-06-15",
        "end": "2025-12-15",
        "z_window": 20,
    }
    out = _post(app_client, "/strategies/basket-stat-arb", body)
    assert out["n_components_used"] >= 1


def test_scan(app_client: TestClient) -> None:
    body = {
        "mode": "all",
        "start": "2025-06-15",
        "end": "2025-12-15",
        "max_pairs": 50,
        "top_k_per_track": 10,
    }
    out = _post(app_client, "/strategies/scan", body)
    assert out["n_factors_scanned"] >= 2
    assert isinstance(out["implication"], list)
    assert isinstance(out["conditional"], list)
    assert isinstance(out["cointegration"], list)


def test_auto_backtest(app_client: TestClient) -> None:
    body = {"start": "2025-06-15", "end": "2025-12-15", "max_pairs": 50, "max_to_backtest": 5}
    out = _post(app_client, "/strategies/auto-backtest", body)
    assert out["n_factors_scanned"] >= 2
    assert "leaderboard" in out
    # Each row has IS/OOS Sharpe.
    for row in out["leaderboard"]:
        assert "sharpe_is" in row and "sharpe_oos" in row


def test_presets(app_client: TestClient) -> None:
    r = app_client.get("/strategies/presets")
    assert r.status_code == 200
    body = r.json()
    expected_keys = {
        "cointegration",
        "pairs",
        "pair_explorer",
        "event_model",
        "basket",
        "spot_vs_implied",
        "ou_bands",
        "granger",
    }
    assert expected_keys <= set(body.keys())
    # Every preset should be well-formed.
    for items in body.values():
        for item in items:
            assert "label" in item and "description" in item and "inputs" in item


def test_all_strategy_routes_in_openapi(app_client: TestClient) -> None:
    r = app_client.get("/openapi.json")
    paths = set(r.json()["paths"].keys())
    expected = {
        "/strategies/implication",
        "/strategies/conditional",
        "/strategies/bounds",
        "/strategies/cointegration",
        "/strategies/pairs-backtest",
        "/strategies/kalman-hedge",
        "/strategies/mean-reversion",
        "/strategies/ou-bands",
        "/strategies/granger",
        "/strategies/event-model",
        "/strategies/basket-stat-arb",
        "/strategies/scan",
        "/strategies/auto-backtest",
        "/strategies/spot-vs-implied",
        "/strategies/presets",
    }
    missing = expected - paths
    assert not missing, f"missing routes: {missing}"

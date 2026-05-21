"""End-to-end tests of the four endpoints with TestClient + mocked IO."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health(app_client: TestClient) -> None:
    r = app_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_factors_lists_loaded_factors(app_client: TestClient) -> None:
    r = app_client.get("/factors")
    assert r.status_code == 200
    ids = {f["id"] for f in r.json()["factors"]}
    assert ids == {"factor_a", "factor_b"}


def test_fit_returns_well_formed_response(app_client: TestClient) -> None:
    r = app_client.post(
        "/fit",
        json={
            "ticker": "TEST",
            "factors": ["factor_a", "factor_b"],
            "start": "2025-06-15",
            "end": "2025-12-15",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ticker"] == "TEST"
    assert body["n_obs"] > 10
    assert body["epsilon"] == 0.01
    assert {f["id"] for f in body["factors"]} == {"factor_a", "factor_b"}
    assert "vif" in body["diagnostics"]
    assert "hac_lag" in body["diagnostics"]


def test_fit_unknown_factor_400(app_client: TestClient) -> None:
    r = app_client.post(
        "/fit",
        json={
            "ticker": "TEST",
            "factors": ["does_not_exist"],
            "start": "2025-06-15",
            "end": "2025-12-15",
        },
    )
    assert r.status_code == 400
    # Detail is now a structured object with did_you_mean suggestions; the
    # legacy "unknown factor" string still appears in the error message.
    detail = r.json()["detail"]
    assert isinstance(detail, dict)
    assert "factor id(s) not found" in detail["error"]
    assert detail["unknown"][0]["query"] == "does_not_exist"


def test_fit_invalid_dates_400(app_client: TestClient) -> None:
    r = app_client.post(
        "/fit",
        json={
            "ticker": "TEST",
            "factors": ["factor_a"],
            "start": "2025-12-15",
            "end": "2025-06-15",
        },
    )
    assert r.status_code == 400


def test_fit_epsilon_query_param_changes_response(app_client: TestClient) -> None:
    payload = {
        "ticker": "TEST",
        "factors": ["factor_a"],
        "start": "2025-06-15",
        "end": "2025-12-15",
    }
    r1 = app_client.post("/fit?epsilon=0.01", json=payload)
    r2 = app_client.post("/fit?epsilon=0.2", json=payload)
    assert r1.status_code == 200 and r2.status_code == 200
    # Different clipping should give different βs in general.
    assert r1.json()["epsilon"] != r2.json()["epsilon"]


def test_attribution_returns_decomposition(app_client: TestClient) -> None:
    r = app_client.post(
        "/attribution",
        json={
            "ticker": "TEST",
            "factors": ["factor_a", "factor_b"],
            "start": "2025-06-15",
            "end": "2025-12-15",
            "date": "2025-09-01",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    ids = [c["id"] for c in body["contributions"]]
    assert ids[0] == "alpha"
    assert "factor_a" in ids and "factor_b" in ids
    # Predicted + residual should equal observed (within fp noise).
    assert abs(body["observed_return"] - (body["predicted_return"] + body["residual"])) < 1e-9


def test_attribution_unknown_date_404(app_client: TestClient) -> None:
    r = app_client.post(
        "/attribution",
        json={
            "ticker": "TEST",
            "factors": ["factor_a"],
            "start": "2025-06-15",
            "end": "2025-12-15",
            "date": "2099-01-01",
        },
    )
    assert r.status_code in (404, 422)


def test_openapi_lists_all_endpoints(app_client: TestClient) -> None:
    r = app_client.get("/openapi.json")
    assert r.status_code == 200
    paths = set(r.json()["paths"].keys())
    assert {"/health", "/factors", "/fit", "/fit/preview", "/attribution"} <= paths


def test_fit_preview_returns_coverage(app_client: TestClient) -> None:
    r = app_client.post(
        "/fit/preview",
        json={
            "ticker": "TEST",
            "factors": ["factor_a", "factor_b"],
            "start": "2025-06-15",
            "end": "2025-12-15",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ticker"] == "TEST"
    assert body["joint_n_obs"] > 10
    ids = {c["factor_id"] for c in body["factor_coverage"]}
    assert ids == {"factor_a", "factor_b"}
    for c in body["factor_coverage"]:
        assert c["n_obs"] > 0
        assert 0.0 <= c["coverage_pct"] <= 1.0
    assert body["min_recommended_obs"] == 30


def test_fit_preview_unknown_factor_400(app_client: TestClient) -> None:
    r = app_client.post(
        "/fit/preview",
        json={
            "ticker": "TEST",
            "factors": ["does_not_exist"],
            "start": "2025-06-15",
            "end": "2025-12-15",
        },
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert isinstance(detail, dict)
    assert "factor id(s) not found" in detail["error"]


def test_fit_preview_warns_on_short_window(app_client: TestClient) -> None:
    """Short windows should surface n<30 warning so the UI can flag pre-Run."""
    r = app_client.post(
        "/fit/preview",
        json={
            "ticker": "TEST",
            "factors": ["factor_a"],
            "start": "2025-11-01",
            "end": "2025-11-20",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["joint_n_obs"] < 30
    msgs = " ".join(body["warnings"])
    assert "joint observations" in msgs or "unreliable" in msgs


def test_fit_preview_no_factors_400(app_client: TestClient) -> None:
    r = app_client.post(
        "/fit/preview",
        json={
            "ticker": "TEST",
            "factors": [],
            "start": "2025-06-15",
            "end": "2025-12-15",
        },
    )
    assert r.status_code == 400


# ---- /strategies/* ---------------------------------------------------------


def test_strategies_implication(app_client: TestClient) -> None:
    """Using the synthetic factor_a (rising) and factor_b (falling) — the
    test isn't a true logical pair but exercises the endpoint plumbing."""
    r = app_client.post(
        "/strategies/implication",
        json={
            "antecedent_id": "factor_a",
            "consequent_id": "factor_b",
            "start": "2025-06-15",
            "end": "2025-12-15",
            "tolerance": 0.02,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["antecedent_id"] == "factor_a"
    assert body["consequent_id"] == "factor_b"
    assert body["n_obs"] > 100
    assert body["verdict"] in {"consistent", "borderline", "violated"}
    assert len(body["series"]) == body["n_obs"]
    pt = body["series"][0]
    assert "p_a" in pt and "p_b" in pt and "gap" in pt and "logit_gap" in pt


def test_strategies_implication_unknown_factor_400(app_client: TestClient) -> None:
    r = app_client.post(
        "/strategies/implication",
        json={
            "antecedent_id": "factor_a",
            "consequent_id": "does_not_exist",
            "start": "2025-06-15",
            "end": "2025-12-15",
        },
    )
    assert r.status_code == 400


def test_strategies_conditional(app_client: TestClient) -> None:
    r = app_client.post(
        "/strategies/conditional",
        json={
            "a_id": "factor_a",
            "b_id": "factor_b",
            "start": "2025-06-15",
            "end": "2025-12-15",
            "hac_lag": 5,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_obs"] > 100
    assert body["beta_ci_lo"] <= body["beta"] <= body["beta_ci_hi"]
    assert 0.0 <= body["r_squared"] <= 1.0
    assert body["n_b_high"] >= 0


def test_strategies_bounds(app_client: TestClient) -> None:
    r = app_client.post(
        "/strategies/bounds",
        json={
            "a_id": "factor_a",
            "b_id": "factor_b",
            "start": "2025-06-15",
            "end": "2025-12-15",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_obs"] > 0
    # Per-date: lower ≤ upper
    for pt in body["series"]:
        assert pt["lower"] <= pt["upper"] + 1e-12
        assert pt["lower"] >= 0.0
        assert pt["upper"] <= 1.0


def test_strategies_routes_in_openapi(app_client: TestClient) -> None:
    r = app_client.get("/openapi.json")
    paths = set(r.json()["paths"].keys())
    assert "/strategies/implication" in paths
    assert "/strategies/conditional" in paths
    assert "/strategies/bounds" in paths


# ---- /strategies/spot-vs-implied ------------------------------------------


def test_strategies_spot_vs_implied(monkeypatch, app_client: TestClient) -> None:
    """End-to-end test using a stubbed Binance client.

    We replace the BinanceClient with a fake whose ``get_klines`` returns
    a deterministic OHLCV DataFrame, then verify the endpoint returns a
    well-formed SpotVsImpliedResponse.
    """

    import numpy as np
    import pandas as pd

    # Build 100 days of synthetic OHLC ending today.
    today = pd.Timestamp.now(tz="UTC").normalize()
    n = 100
    rng = np.random.default_rng(0)
    daily_sig = 0.50 / np.sqrt(365.0)
    log_ret = rng.normal(0, daily_sig, size=n)
    closes = 100.0 * np.exp(np.cumsum(log_ret))
    opens = np.concatenate([[100.0], closes[:-1]])
    intraday = np.abs(rng.normal(0, daily_sig * 0.4, size=n))
    highs = np.maximum(opens, closes) * np.exp(intraday)
    lows = np.minimum(opens, closes) * np.exp(-intraday)
    idx = pd.date_range(today - pd.Timedelta(days=n - 1), periods=n, freq="D", tz="UTC")
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": np.ones(n)},
        index=idx,
    )

    class FakeBinance:
        def get_klines(self, symbol, *, start=None, end=None, limit=1000, interval="1d"):
            return df.copy()

    app_client.app.state.binance = FakeBinance()

    expiry = (today + pd.Timedelta(days=30)).date()
    r = app_client.post(
        "/strategies/spot-vs-implied",
        json={
            "symbol": "BTCUSDT",
            "strike": float(closes[-1]) * 1.10,
            "expiry": expiry.isoformat(),
            "geometry": "terminal",
            "market_prob": 0.30,
            "n_bootstrap": 40,
            "vol_window_days": 60,
            "seed": 1,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["symbol"] == "BTCUSDT"
    assert 0.0 <= body["model_prob"] <= 1.0
    assert body["ci_lo_95"] <= body["model_prob"] <= body["ci_hi_95"]
    assert body["sigma_method"] == "yang_zhang"
    assert body["sigma_used"] > 0
    assert abs(body["edge"] - (0.30 - body["model_prob"])) < 1e-9
    assert isinstance(body["edge_significant_95"], bool)
    assert body["n_bootstrap"] == 40


def test_strategies_spot_vs_implied_past_expiry_400(app_client: TestClient) -> None:
    r = app_client.post(
        "/strategies/spot-vs-implied",
        json={
            "symbol": "BTCUSDT",
            "strike": 50000.0,
            "expiry": "2020-01-01",
            "geometry": "terminal",
        },
    )
    assert r.status_code == 400


def test_strategies_spot_vs_implied_in_openapi(app_client: TestClient) -> None:
    r = app_client.get("/openapi.json")
    paths = set(r.json()["paths"].keys())
    assert "/strategies/spot-vs-implied" in paths


# ---- /strategies/cointegration / pairs-backtest / event-model / scan -------


def test_strategies_cointegration(app_client: TestClient) -> None:
    r = app_client.post(
        "/strategies/cointegration",
        json={
            "a_id": "factor_a",
            "b_id": "factor_b",
            "start": "2025-06-15",
            "end": "2025-12-15",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["a_id"] == "factor_a"
    assert body["verdict"] in {"cointegrated", "not_cointegrated", "insufficient-data"}
    assert isinstance(body["cointegrated"], bool)
    if body["cointegrated"]:
        assert body["adf_pvalue"] < 0.05
        assert body["beta_hedge"] is not None
    assert isinstance(body["series"], list)


def test_strategies_pairs_backtest(app_client: TestClient) -> None:
    r = app_client.post(
        "/strategies/pairs-backtest",
        json={
            "a_id": "factor_a",
            "b_id": "factor_b",
            "start": "2025-06-15",
            "end": "2025-12-15",
            "window": 20,
            "entry_z": 2.0,
            "exit_z": 0.5,
            "stop_z": 4.0,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["a_id"] == "factor_a"
    assert "sharpe" in body
    assert body["n_obs"] > 0
    assert isinstance(body["trades"], list)
    assert isinstance(body["series"], list)


def test_strategies_event_model(app_client: TestClient) -> None:
    r = app_client.post(
        "/strategies/event-model",
        json={
            "target_id": "factor_a",
            "factor_ids": ["factor_b"],
            "start": "2025-06-15",
            "end": "2025-12-15",
            "hac_lag": 5,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["target_id"] == "factor_a"
    assert body["factor_ids"] == ["factor_b"]
    assert body["n_obs"] > 50
    assert len(body["coefficients"]) == 1
    coef = body["coefficients"][0]
    assert coef["factor_id"] == "factor_b"
    assert coef["ci_lo"] <= coef["beta"] <= coef["ci_hi"]


def test_strategies_event_model_target_in_factors_400(app_client: TestClient) -> None:
    r = app_client.post(
        "/strategies/event-model",
        json={
            "target_id": "factor_a",
            "factor_ids": ["factor_a", "factor_b"],
            "start": "2025-06-15",
            "end": "2025-12-15",
        },
    )
    assert r.status_code == 400


def test_strategies_scan_basic(app_client: TestClient) -> None:
    r = app_client.post(
        "/strategies/scan",
        json={
            "mode": "all",
            "start": "2025-06-15",
            "end": "2025-12-15",
            "max_pairs": 50,
            "top_k_per_track": 10,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "all"
    assert body["n_factors_scanned"] >= 2
    assert isinstance(body["implication"], list)
    assert isinstance(body["conditional"], list)
    assert isinstance(body["cointegration"], list)


def test_new_strategy_routes_in_openapi(app_client: TestClient) -> None:
    r = app_client.get("/openapi.json")
    paths = set(r.json()["paths"].keys())
    for p in (
        "/strategies/cointegration",
        "/strategies/pairs-backtest",
        "/strategies/event-model",
        "/strategies/scan",
    ):
        assert p in paths, f"{p} missing"

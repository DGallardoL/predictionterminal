"""Coverage-focused tests for ``pfm.advanced_event_models_router``.

Hits the under-covered error / edge branches in the router:

* non-``polymarket`` factor source (line 142)
* ``PolymarketError`` and ``httpx.HTTPError`` from ``fetch_factor_history``
  (lines 148-151)
* empty / missing ``price`` column DataFrame (line 153)
* ``EquityFactorError`` from yfinance for both log-return and price paths
  (lines 168-169 and 183-184)
* ``ValueError`` from each of the six core fitters → 422 (lines 225-226,
  258-259, 291-292, 325-326, 357-358, 390-391)
* Pydantic input validation (each endpoint)
* invalid window guard
* cache hot path (second call hits cache)
* happy-path smoke for each of the six endpoints

External IO is mocked at the router module level — no network is touched.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixture: TestClient with router-level patches for factor + equity IO.
# ---------------------------------------------------------------------------


def _make_smooth_probs(n: int = 250) -> pd.DataFrame:
    idx = pd.date_range("2025-04-01", periods=n, freq="D", tz="UTC")
    t = np.arange(n) / n
    prob = (0.30 + 0.30 * np.sin(2 * np.pi * t * 1.2)).clip(0.05, 0.95)
    df = pd.DataFrame({"price": prob}, index=idx)
    df.index.name = "date"
    return df


def _make_price_series(n: int = 250, seed: int = 12345) -> pd.Series:
    idx = pd.date_range("2025-04-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(seed)
    base = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, n)))
    return pd.Series(base, index=idx, name="px")


@pytest.fixture
def cov_client(
    monkeypatch: pytest.MonkeyPatch,
    factors_file: Path,
) -> Iterator[tuple[TestClient, dict]]:
    """TestClient + mutable state dict.

    Yields ``(client, state)`` so tests can mutate the state (factor df /
    factor error / price series / price error) and have the patched
    fetchers reflect it on the next call. We can't stash the state on
    the TestClient itself because TestClient.state is a ``ClientState``
    namespace, not a dict.
    """
    import pfm.advanced_event_models_router as router_mod
    import pfm.config as cfg
    import pfm.main as main_mod
    from pfm.cache import NullCache

    monkeypatch.setenv("FACTORS_FILE", str(factors_file))
    cfg._settings = None

    state: dict = {
        "factor_df": _make_smooth_probs(),
        "factor_error": None,
        "price_series": _make_price_series(),
        "price_error": None,
    }

    def _fake_factor_history(_client, slug, start=None, end=None):
        if state["factor_error"] is not None:
            raise state["factor_error"]
        df = state["factor_df"]
        if df is None:
            return None
        # Don't try to slice an empty / non-datetime indexed frame.
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return df
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        return df

    def _fake_equity_history(ticker, start, end, **_):
        if state["price_error"] is not None:
            raise state["price_error"]
        s = state["price_series"].copy()
        s.name = ticker
        if start is not None:
            s = s[s.index >= start]
        if end is not None:
            s = s[s.index <= end]
        return s

    monkeypatch.setattr(router_mod, "fetch_factor_history", _fake_factor_history)
    monkeypatch.setattr(router_mod, "fetch_equity_history", _fake_equity_history)
    monkeypatch.setattr(main_mod, "RedisCache", lambda url: NullCache())

    # Reset shared cache between tests so coverage-of-cache assertions are
    # deterministic.
    from pfm.cache_utils import get_cache

    get_cache("advanced_event_models").clear()

    with TestClient(main_mod.app) as client:
        yield client, state

    get_cache("advanced_event_models").clear()


# Base body used by all happy-path / error-path POSTs.
BASE_BODY: dict[str, str | int | float | list[float]] = {
    "ticker": "TEST",
    "factor_id": "factor_a",
    "start": "2025-04-15",
    "end": "2025-12-01",
}

# All six endpoints + their endpoint-specific bodies.
ENDPOINTS: list[tuple[str, dict]] = [
    ("/advanced-model/conditional", {"conditioning_thresholds": [0.3, 0.7]}),
    ("/advanced-model/polynomial", {"degree": 2}),
    ("/advanced-model/regime-switching", {"n_regimes": 2}),
    ("/advanced-model/vecm", {"det_order": 0, "k_ar_diff": 1}),
    ("/advanced-model/garch-x", {}),
    ("/advanced-model/tail-dependence", {"quantile": 0.05}),
]


# ===========================================================================
# A) Happy path — each endpoint returns 200 (or 422 for regime-switching on
#    smooth synthetic data; statsmodels can fail to converge).
# ===========================================================================


class TestHappyPath:
    def test_conditional_happy(self, cov_client) -> None:
        client, _ = cov_client
        r = client.post(
            "/advanced-model/conditional",
            json={**BASE_BODY, "conditioning_thresholds": [0.3, 0.7]},
        )
        assert r.status_code == 200, r.text
        assert "buckets" in r.json()

    def test_polynomial_happy(self, cov_client) -> None:
        client, _ = cov_client
        r = client.post(
            "/advanced-model/polynomial",
            json={**BASE_BODY, "degree": 2},
        )
        assert r.status_code == 200, r.text
        assert r.json()["degree"] == 2

    def test_regime_switching_happy_or_422(self, cov_client) -> None:
        client, _ = cov_client
        r = client.post(
            "/advanced-model/regime-switching",
            json={**BASE_BODY, "n_regimes": 2},
        )
        assert r.status_code in (200, 422), r.text

    def test_vecm_happy(self, cov_client) -> None:
        client, _ = cov_client
        r = client.post(
            "/advanced-model/vecm",
            json={**BASE_BODY, "det_order": 0, "k_ar_diff": 1},
        )
        assert r.status_code == 200, r.text
        assert "is_cointegrated" in r.json()

    def test_garch_x_happy(self, cov_client) -> None:
        client, _ = cov_client
        r = client.post("/advanced-model/garch-x", json={**BASE_BODY})
        assert r.status_code == 200, r.text
        assert "persistence" in r.json()

    def test_tail_dependence_happy(self, cov_client) -> None:
        client, _ = cov_client
        r = client.post(
            "/advanced-model/tail-dependence",
            json={**BASE_BODY, "quantile": 0.05},
        )
        assert r.status_code == 200, r.text
        assert "lower_tail_dependence" in r.json()


# ===========================================================================
# B) Cache hot path — second identical call must short-circuit.
# ===========================================================================


class TestCacheHotPath:
    def test_second_call_hits_cache(self, cov_client) -> None:
        client, state = cov_client
        body = {**BASE_BODY, "degree": 2}
        r1 = client.post("/advanced-model/polynomial", json=body)
        assert r1.status_code == 200
        # Break the upstream after the first call. The cache layer should
        # short-circuit, so we still get a 200.
        state["factor_error"] = RuntimeError("must not be called again")
        r2 = client.post("/advanced-model/polynomial", json=body)
        assert r2.status_code == 200
        assert r2.json() == r1.json()


# ===========================================================================
# C) Non-polymarket source → 400 (line 142).
# ===========================================================================


class TestNonPolymarketSource:
    def test_kalshi_source_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Build a one-off factors file whose factor has a non-polymarket source.
        # We use ``bls`` because polymarket.yml only validates a fixed set of
        # source names and bls is widely accepted in the codebase.
        p = tmp_path / "factors.yml"
        p.write_text(
            """
factors:
  - id: factor_macro
    name: Macro CPI
    slug: CUUR0000SA0
    source: bls
    description: BLS CPI level.
"""
        )

        import pfm.advanced_event_models_router as router_mod
        import pfm.config as cfg
        import pfm.main as main_mod
        from pfm.cache import NullCache
        from pfm.cache_utils import get_cache

        monkeypatch.setenv("FACTORS_FILE", str(p))
        cfg._settings = None
        get_cache("advanced_event_models").clear()

        # Patch in dummy fetchers (they should not be called for non-poly source).
        def _explode(*_a, **_kw):  # pragma: no cover - must not be called
            raise AssertionError("fetcher should not run for non-polymarket source")

        monkeypatch.setattr(router_mod, "fetch_factor_history", _explode)
        monkeypatch.setattr(
            router_mod,
            "fetch_equity_history",
            lambda *_a, **_kw: _make_price_series(),
        )
        monkeypatch.setattr(main_mod, "RedisCache", lambda url: NullCache())

        with TestClient(main_mod.app) as client:
            r = client.post(
                "/advanced-model/polynomial",
                json={
                    "ticker": "TEST",
                    "factor_id": "factor_macro",
                    "start": "2025-04-15",
                    "end": "2025-12-01",
                    "degree": 2,
                },
            )
            assert r.status_code == 400, r.text
            assert "polymarket" in r.json()["detail"]
        get_cache("advanced_event_models").clear()


# ===========================================================================
# D) Polymarket error paths (lines 148-151, 153).
# ===========================================================================


class TestPolymarketErrors:
    def test_polymarket_error_502(self, cov_client) -> None:
        from pfm.sources.polymarket import PolymarketError

        client, state = cov_client
        state["factor_error"] = PolymarketError("upstream down")
        r = client.post(
            "/advanced-model/polynomial",
            json={**BASE_BODY, "degree": 2},
        )
        assert r.status_code == 502
        assert "polymarket error" in r.json()["detail"]

    def test_httpx_error_502(self, cov_client) -> None:
        client, state = cov_client
        state["factor_error"] = httpx.ConnectError("boom")
        r = client.post(
            "/advanced-model/polynomial",
            json={**BASE_BODY, "degree": 2},
        )
        assert r.status_code == 502
        assert "polymarket http error" in r.json()["detail"]

    def test_empty_dataframe_502(self, cov_client) -> None:
        client, state = cov_client
        state["factor_df"] = pd.DataFrame()
        r = client.post(
            "/advanced-model/polynomial",
            json={**BASE_BODY, "degree": 2},
        )
        assert r.status_code == 502
        assert "no probability history" in r.json()["detail"]

    def test_none_dataframe_502(self, cov_client) -> None:
        client, state = cov_client
        state["factor_df"] = None
        r = client.post(
            "/advanced-model/polynomial",
            json={**BASE_BODY, "degree": 2},
        )
        assert r.status_code == 502
        assert "no probability history" in r.json()["detail"]

    def test_missing_price_column_502(self, cov_client) -> None:
        client, state = cov_client
        n = 100
        idx = pd.date_range("2025-04-01", periods=n, freq="D", tz="UTC")
        bad = pd.DataFrame({"value": np.linspace(0.1, 0.9, n)}, index=idx)
        bad.index.name = "date"
        state["factor_df"] = bad
        r = client.post(
            "/advanced-model/polynomial",
            json={**BASE_BODY, "degree": 2},
        )
        assert r.status_code == 502


# ===========================================================================
# E) yfinance error paths (lines 168-169 for returns, 183-184 for prices).
# ===========================================================================


class TestEquityErrors:
    def test_yfinance_error_502_returns_path(self, cov_client) -> None:
        """Polynomial endpoint hits _fetch_equity_returns (lines 168-169)."""
        from pfm.equity_factors import EquityFactorError

        client, state = cov_client
        state["price_error"] = EquityFactorError("no data")
        r = client.post(
            "/advanced-model/polynomial",
            json={**BASE_BODY, "degree": 2},
        )
        assert r.status_code == 502
        assert "yfinance error" in r.json()["detail"]

    def test_yfinance_error_502_prices_path(self, cov_client) -> None:
        """VECM endpoint hits _fetch_equity_prices (lines 183-184)."""
        from pfm.equity_factors import EquityFactorError

        client, state = cov_client
        state["price_error"] = EquityFactorError("no data")
        r = client.post(
            "/advanced-model/vecm",
            json={**BASE_BODY, "det_order": 0, "k_ar_diff": 1},
        )
        assert r.status_code == 502
        assert "yfinance error" in r.json()["detail"]


# ===========================================================================
# F) ValueError → 422 (each endpoint).
# ===========================================================================


def _short_window_state(state: dict, n: int = 25) -> None:
    """Replace upstream fixtures with a too-short series so every fitter raises.

    ``n`` is small enough to break every core (each requires >50 obs after
    Δlogit/diff). We anchor the index inside the BASE_BODY window (start
    2025-04-15) so the fake fetcher's `df.index >= start` filter doesn't
    drop every row.
    """
    idx = pd.date_range("2025-04-15", periods=n, freq="D", tz="UTC")
    df = pd.DataFrame({"price": np.linspace(0.1, 0.9, n)}, index=idx)
    df.index.name = "date"
    state["factor_df"] = df
    state["price_series"] = pd.Series(np.linspace(100.0, 110.0, n), index=idx, name="px")


class TestValueErrorTo422:
    def test_conditional_422_on_short_window(self, cov_client) -> None:
        client, state = cov_client
        _short_window_state(state)
        r = client.post(
            "/advanced-model/conditional",
            json={**BASE_BODY, "conditioning_thresholds": [0.3, 0.7]},
        )
        assert r.status_code == 422, r.text

    def test_polynomial_422_on_short_window(self, cov_client) -> None:
        client, state = cov_client
        _short_window_state(state)
        r = client.post(
            "/advanced-model/polynomial",
            json={**BASE_BODY, "degree": 2},
        )
        assert r.status_code == 422, r.text

    def test_regime_422_on_short_window(self, cov_client) -> None:
        client, state = cov_client
        _short_window_state(state)
        r = client.post(
            "/advanced-model/regime-switching",
            json={**BASE_BODY, "n_regimes": 2},
        )
        assert r.status_code == 422, r.text

    def test_vecm_422_on_short_window(self, cov_client) -> None:
        client, state = cov_client
        _short_window_state(state)
        r = client.post(
            "/advanced-model/vecm",
            json={**BASE_BODY, "det_order": 0, "k_ar_diff": 1},
        )
        assert r.status_code == 422, r.text

    def test_garch_x_422_on_short_window(self, cov_client) -> None:
        client, state = cov_client
        _short_window_state(state)
        r = client.post(
            "/advanced-model/garch-x",
            json={**BASE_BODY},
        )
        assert r.status_code == 422, r.text

    def test_tail_dependence_422_on_short_window(self, cov_client) -> None:
        client, state = cov_client
        _short_window_state(state)
        r = client.post(
            "/advanced-model/tail-dependence",
            json={**BASE_BODY, "quantile": 0.05},
        )
        assert r.status_code == 422, r.text


# ===========================================================================
# G) Invalid window (start >= end) → 400 for every endpoint.
# ===========================================================================


class TestInvalidWindow:
    @pytest.mark.parametrize("path,extra", ENDPOINTS)
    def test_start_after_end_400(self, cov_client, path: str, extra: dict) -> None:
        client, _ = cov_client
        body = {
            **BASE_BODY,
            "start": "2025-12-01",
            "end": "2025-04-15",
            **extra,
        }
        r = client.post(path, json=body)
        assert r.status_code == 400, r.text
        assert "start must be < end" in r.json()["detail"]

    @pytest.mark.parametrize("path,extra", ENDPOINTS)
    def test_start_equals_end_400(self, cov_client, path: str, extra: dict) -> None:
        client, _ = cov_client
        body = {
            **BASE_BODY,
            "start": "2025-04-15",
            "end": "2025-04-15",
            **extra,
        }
        r = client.post(path, json=body)
        assert r.status_code == 400, r.text


# ===========================================================================
# H) Pydantic validation (each endpoint).
# ===========================================================================


class TestPydanticValidation:
    def test_empty_ticker_422(self, cov_client) -> None:
        client, _ = cov_client
        body = {**BASE_BODY, "ticker": "", "degree": 2}
        r = client.post("/advanced-model/polynomial", json=body)
        assert r.status_code == 422

    def test_ticker_too_long_422(self, cov_client) -> None:
        client, _ = cov_client
        body = {**BASE_BODY, "ticker": "X" * 21, "degree": 2}
        r = client.post("/advanced-model/polynomial", json=body)
        assert r.status_code == 422

    def test_empty_factor_id_422(self, cov_client) -> None:
        client, _ = cov_client
        body = {**BASE_BODY, "factor_id": "", "degree": 2}
        r = client.post("/advanced-model/polynomial", json=body)
        assert r.status_code == 422

    def test_factor_id_too_long_422(self, cov_client) -> None:
        client, _ = cov_client
        body = {**BASE_BODY, "factor_id": "x" * 121, "degree": 2}
        r = client.post("/advanced-model/polynomial", json=body)
        assert r.status_code == 422

    def test_epsilon_zero_422(self, cov_client) -> None:
        client, _ = cov_client
        body = {**BASE_BODY, "degree": 2, "epsilon": 0.0}
        r = client.post("/advanced-model/polynomial", json=body)
        assert r.status_code == 422

    def test_epsilon_too_large_422(self, cov_client) -> None:
        client, _ = cov_client
        body = {**BASE_BODY, "degree": 2, "epsilon": 0.6}
        r = client.post("/advanced-model/polynomial", json=body)
        assert r.status_code == 422

    def test_conditional_empty_thresholds_422(self, cov_client) -> None:
        client, _ = cov_client
        body = {**BASE_BODY, "conditioning_thresholds": []}
        r = client.post("/advanced-model/conditional", json=body)
        assert r.status_code == 422

    def test_conditional_too_many_thresholds_422(self, cov_client) -> None:
        client, _ = cov_client
        body = {
            **BASE_BODY,
            "conditioning_thresholds": [
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                0.95,
            ],
        }
        r = client.post("/advanced-model/conditional", json=body)
        assert r.status_code == 422

    def test_polynomial_degree_zero_422(self, cov_client) -> None:
        client, _ = cov_client
        body = {**BASE_BODY, "degree": 0}
        r = client.post("/advanced-model/polynomial", json=body)
        assert r.status_code == 422

    def test_polynomial_degree_too_high_422(self, cov_client) -> None:
        client, _ = cov_client
        body = {**BASE_BODY, "degree": 7}
        r = client.post("/advanced-model/polynomial", json=body)
        assert r.status_code == 422

    def test_regime_too_few_422(self, cov_client) -> None:
        client, _ = cov_client
        body = {**BASE_BODY, "n_regimes": 1}
        r = client.post("/advanced-model/regime-switching", json=body)
        assert r.status_code == 422

    def test_regime_too_many_422(self, cov_client) -> None:
        client, _ = cov_client
        body = {**BASE_BODY, "n_regimes": 5}
        r = client.post("/advanced-model/regime-switching", json=body)
        assert r.status_code == 422

    def test_vecm_det_order_low_422(self, cov_client) -> None:
        client, _ = cov_client
        body = {**BASE_BODY, "det_order": -2, "k_ar_diff": 1}
        r = client.post("/advanced-model/vecm", json=body)
        assert r.status_code == 422

    def test_vecm_det_order_high_422(self, cov_client) -> None:
        client, _ = cov_client
        body = {**BASE_BODY, "det_order": 2, "k_ar_diff": 1}
        r = client.post("/advanced-model/vecm", json=body)
        assert r.status_code == 422

    def test_vecm_k_ar_diff_low_422(self, cov_client) -> None:
        client, _ = cov_client
        body = {**BASE_BODY, "det_order": 0, "k_ar_diff": 0}
        r = client.post("/advanced-model/vecm", json=body)
        assert r.status_code == 422

    def test_vecm_k_ar_diff_high_422(self, cov_client) -> None:
        client, _ = cov_client
        body = {**BASE_BODY, "det_order": 0, "k_ar_diff": 6}
        r = client.post("/advanced-model/vecm", json=body)
        assert r.status_code == 422

    def test_tail_quantile_zero_422(self, cov_client) -> None:
        client, _ = cov_client
        body = {**BASE_BODY, "quantile": 0.0}
        r = client.post("/advanced-model/tail-dependence", json=body)
        assert r.status_code == 422

    def test_tail_quantile_too_large_422(self, cov_client) -> None:
        client, _ = cov_client
        body = {**BASE_BODY, "quantile": 0.7}
        r = client.post("/advanced-model/tail-dependence", json=body)
        assert r.status_code == 422


# ===========================================================================
# I) Unknown factor → 400 with did_you_mean payload.
# ===========================================================================


class TestUnknownFactor:
    def test_unknown_factor_400(self, cov_client) -> None:
        client, _ = cov_client
        body = {**BASE_BODY, "factor_id": "definitely_not_a_real_factor", "degree": 2}
        r = client.post("/advanced-model/polynomial", json=body)
        assert r.status_code == 400
        detail = r.json()["detail"]
        if isinstance(detail, dict):
            assert "did_you_mean" in detail
        else:
            assert "factor" in detail.lower()


# ===========================================================================
# J) Module-level smoke: importable + router prefix correct.
# ===========================================================================


def test_router_prefix() -> None:
    import pfm.advanced_event_models_router as router_mod

    assert router_mod.router.prefix == "/advanced-model"
    paths = {r.path for r in router_mod.router.routes}
    assert "/advanced-model/conditional" in paths
    assert "/advanced-model/polynomial" in paths
    assert "/advanced-model/regime-switching" in paths
    assert "/advanced-model/vecm" in paths
    assert "/advanced-model/garch-x" in paths
    assert "/advanced-model/tail-dependence" in paths


def test_cache_key_helper_uses_model_dump_json() -> None:
    from pfm.advanced_event_models_router import (
        PolynomialRequest,
        _cache_key,
    )

    body = PolynomialRequest(
        ticker="AAPL",
        factor_id="factor_a",
        start="2025-01-01",  # type: ignore[arg-type]
        end="2025-02-01",  # type: ignore[arg-type]
        degree=3,
    )
    key1 = _cache_key("poly", body)
    key2 = _cache_key("poly", body)
    assert key1 == key2
    assert key1.startswith("poly:")
    assert "AAPL" in key1

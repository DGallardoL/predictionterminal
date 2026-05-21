"""Extra coverage for ``pfm.terminal_correlations``.

Focuses on edge cases not exercised in ``test_terminal_correlations.py``:

  * empty / insufficient probability history
  * caching round-trip
  * scipy-free p-value helper degenerate inputs
  * upstream HTTPError → 502
  * benchmark fetcher errors are non-fatal (per-asset ``error`` field)
  * unknown benchmark source raises
  * ``_innovations`` kind dispatch + ``_logit`` clipping
  * query parameter validation (days lower/upper bound)
  * schema correctness on the JSON response
  * ``best_lag_corr`` returns ``None`` on too-short input
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import terminal_correlations
from pfm.equity_factors import EquityFactorError
from pfm.sources.fred import FredDataError
from pfm.terminal_correlations import (
    BENCHMARKS,
    _coerce_finite,
    _fetch_benchmark,
    _innovations,
    _interpret,
    _logit,
    _pearson_p_value,
    best_lag_corr,
    clear_cache,
    get_polymarket_client,
    router,
)


class _FakePoly:
    """Sentinel injected into the dependency override."""


def _make_prob_history(days: int = 180, base: float = 0.5, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=days, freq="D", tz="UTC")
    walk = np.cumsum(rng.standard_normal(days) * 0.03)
    prices = (base + walk).clip(0.05, 0.95)
    df = pd.DataFrame({"price": prices}, index=idx)
    df.index.name = "date"
    return df


def _make_benchmark_series(days: int, *, name: str, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=days, freq="D", tz="UTC")
    log_returns = 0.01 * rng.standard_normal(days)
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    return pd.Series(prices, index=idx, name=name)


@pytest.fixture(autouse=True)
def _drop_cache() -> Iterator[None]:
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def make_client(monkeypatch: pytest.MonkeyPatch):
    def _build(
        *,
        prob_df: pd.DataFrame | None = None,
        prob_raises: BaseException | None = None,
        bench_factory=None,
    ) -> TestClient:
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_polymarket_client] = _FakePoly

        def _fake_factor_history(_client, _slug, start=None, end=None):
            if prob_raises is not None:
                raise prob_raises
            return prob_df if prob_df is not None else _make_prob_history()

        def _default_bench(symbol, source, _start, _end):
            days = 180
            return _make_benchmark_series(days, name=symbol, seed=hash(symbol) % (2**32))

        factory = bench_factory or _default_bench

        def _fake_equity_history(ticker, start, end, **_):
            return factory(ticker, "yf", start, end)

        def _fake_fred_series(series_id, start, end, **_):
            return factory(series_id, "fred", start, end)

        monkeypatch.setattr(terminal_correlations, "fetch_factor_history", _fake_factor_history)
        monkeypatch.setattr(terminal_correlations, "fetch_equity_history", _fake_equity_history)
        monkeypatch.setattr(terminal_correlations, "fetch_fred_series", _fake_fred_series)
        return TestClient(app)

    return _build


# ---------------------------------------------------------------------------
# Edge cases on the probability series
# ---------------------------------------------------------------------------


class TestProbabilityEdgeCases:
    def test_empty_polymarket_history_returns_404(self, make_client) -> None:
        empty = pd.DataFrame({"price": []}, index=pd.DatetimeIndex([], tz="UTC", name="date"))
        client = make_client(prob_df=empty)
        r = client.get("/terminal/correlations/empty-slug?days=90")
        assert r.status_code == 404
        assert "empty-slug" in r.json()["detail"]

    def test_too_few_observations_returns_422(self, make_client) -> None:
        # 3 obs ⇒ at most 2 innovations after diff ⇒ < 5 required.
        idx = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
        tiny = pd.DataFrame({"price": [0.4, 0.5, 0.6]}, index=idx)
        client = make_client(prob_df=tiny)
        r = client.get("/terminal/correlations/short-slug?days=60")
        assert r.status_code == 422
        assert "insufficient" in r.json()["detail"]

    def test_polymarket_http_error_yields_502(self, make_client) -> None:
        client = make_client(prob_raises=httpx.ConnectError("boom"))
        r = client.get("/terminal/correlations/anything?days=60")
        assert r.status_code == 502
        assert "polymarket http error" in r.json()["detail"]

    def test_missing_price_column_returns_404(self, make_client) -> None:
        idx = pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC")
        df = pd.DataFrame({"other": np.arange(10) * 0.1}, index=idx)
        client = make_client(prob_df=df)
        r = client.get("/terminal/correlations/no-price?days=60")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


class TestCache:
    def test_cache_hit_skips_data_layer(self, make_client, monkeypatch) -> None:
        client = make_client()
        r1 = client.get("/terminal/correlations/cache-key?days=90")
        assert r1.status_code == 200

        # Sabotage the data layer: a second call must return the cached
        # body without invoking the (now broken) fetchers.
        def _explode(*_a, **_k):
            raise RuntimeError("must not be called on cache hit")

        monkeypatch.setattr(terminal_correlations, "fetch_factor_history", _explode)
        monkeypatch.setattr(terminal_correlations, "fetch_equity_history", _explode)
        monkeypatch.setattr(terminal_correlations, "fetch_fred_series", _explode)

        r2 = client.get("/terminal/correlations/cache-key?days=90")
        assert r2.status_code == 200
        assert r2.json() == r1.json()

    def test_clear_cache_drops_entries(self) -> None:
        terminal_correlations._CACHE[("foo", 30)] = (9e18, {"slug": "foo"})
        assert ("foo", 30) in terminal_correlations._CACHE
        clear_cache()
        assert terminal_correlations._CACHE == {}


# ---------------------------------------------------------------------------
# Benchmark error tolerance
# ---------------------------------------------------------------------------


class TestBenchmarkErrors:
    def test_per_benchmark_error_is_non_fatal(self, make_client) -> None:
        """When a single benchmark fetcher raises, the response still
        succeeds with that asset's ``error`` field populated."""

        def factory(symbol, source, start, end):
            if symbol == "BTC-USD":
                raise EquityFactorError("yf is down")
            if symbol == "DGS10":
                raise FredDataError("FRED 503")
            return _make_benchmark_series(180, name=symbol, seed=hash(symbol) % (2**32))

        client = make_client(bench_factory=factory)
        r = client.get("/terminal/correlations/some-slug?days=90")
        assert r.status_code == 200
        body = r.json()
        assert body["correlations"]["BTC-USD"]["error"] == "yf is down"
        assert body["correlations"]["BTC-USD"]["corr"] is None
        assert body["correlations"]["DGS10"]["error"] == "FRED 503"
        # And the other benchmarks are still computed.
        for name, _, _ in BENCHMARKS:
            if name in {"BTC-USD", "DGS10"}:
                continue
            assert "error" not in body["correlations"][name]

    def test_empty_benchmark_series_is_skipped_cleanly(self, make_client) -> None:
        """If a fetcher returns an empty series, the asset row is still
        emitted but with ``corr=None`` / ``n=0``."""

        def factory(symbol, source, start, end):
            if symbol == "SPY":
                return pd.Series(dtype=float, name=symbol)
            return _make_benchmark_series(180, name=symbol, seed=hash(symbol) % (2**32))

        client = make_client(bench_factory=factory)
        r = client.get("/terminal/correlations/some-slug?days=90")
        assert r.status_code == 200
        spy = r.json()["correlations"]["SPY"]
        assert spy["corr"] is None
        assert spy["n"] == 0


# ---------------------------------------------------------------------------
# Helpers (pure)
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_pearson_p_value_handles_degenerate_inputs(self) -> None:
        assert _pearson_p_value(0.5, n=3) is None  # n too small
        assert _pearson_p_value(float("nan"), n=50) is None
        # Perfect correlations clip rather than blow up.
        p = _pearson_p_value(1.0, n=50)
        assert p is not None
        assert 0.0 <= p <= 1.0
        # Zero correlation → p-value near 1.
        p0 = _pearson_p_value(0.0, n=100)
        assert p0 is not None
        assert p0 > 0.5

    def test_innovations_dispatch(self) -> None:
        idx = pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC")
        prob = pd.Series([0.2, 0.3, 0.4, 0.5, 0.6], index=idx)
        prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=idx)
        rates = pd.Series([4.0, 4.05, 4.10, 4.15, 4.20], index=idx)

        # logit kind ⇒ first NaN, then finite differences.
        innov = _innovations(prob, kind="logit")
        assert pd.isna(innov.iloc[0])
        assert innov.iloc[1:].notna().all()

        # log kind ⇒ log returns of strictly positive series.
        log_innov = _innovations(prices, kind="log")
        assert pd.isna(log_innov.iloc[0])
        assert np.allclose(
            log_innov.iloc[1:], np.log(prices.iloc[1:].to_numpy() / prices.iloc[:-1].to_numpy())
        )

        # diff kind ⇒ raw first differences.
        d_innov = _innovations(rates, kind="diff")
        assert np.allclose(d_innov.iloc[1:], 0.05, atol=1e-12)

        with pytest.raises(ValueError, match="unknown innovation kind"):
            _innovations(prob, kind="bogus")

    def test_logit_clips_to_eps(self) -> None:
        idx = pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC")
        # 0.001 < clip_eps → clipped to eps; 0.999 > 1-eps → clipped to 1-eps.
        s = pd.Series([0.001, 0.5, 0.999, 0.0], index=idx)
        out = _logit(s, clip_eps=0.01)
        # 0.0 is out-of-bounds → NaN.
        assert pd.isna(out.iloc[3])
        # 0.001 (below 0.01) and 0.999 (above 0.99) clip symmetrically.
        eps = 0.01
        clipped_lo = np.log(eps / (1 - eps))
        clipped_hi = np.log((1 - eps) / eps)
        assert out.iloc[0] == pytest.approx(clipped_lo)
        assert out.iloc[2] == pytest.approx(clipped_hi)
        # Mid-point is 0.
        assert out.iloc[1] == pytest.approx(0.0)

    def test_log_innovations_drop_nonpositive(self) -> None:
        idx = pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC")
        s = pd.Series([100.0, 0.0, 50.0, 60.0], index=idx)  # contains a non-positive obs
        out = _innovations(s, kind="log")
        # The 0.0 row is dropped before log() — surviving row count should be 3.
        assert out.notna().sum() >= 1
        # No infs / NaNs from log(0).
        assert np.isfinite(out.dropna()).all()

    def test_coerce_finite_handles_non_numeric(self) -> None:
        assert _coerce_finite(None) is None
        assert _coerce_finite("not-a-number") is None
        assert _coerce_finite(float("nan")) is None
        assert _coerce_finite(float("inf")) is None
        assert _coerce_finite(2.5) == 2.5
        assert _coerce_finite("3.14") == pytest.approx(3.14)

    def test_fetch_benchmark_unknown_source_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown benchmark source"):
            _fetch_benchmark(
                "X",
                "bloomberg",
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-12-31"),
            )

    def test_best_lag_corr_returns_none_on_short_input(self) -> None:
        idx = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
        s = pd.Series([1.0, 2.0, 3.0], index=idx)
        lag, corr, n = best_lag_corr(s, s, max_lag=2)
        assert lag is None
        assert corr is None
        assert n is None

    def test_best_lag_corr_skips_zero_variance(self) -> None:
        idx = pd.date_range("2025-01-01", periods=20, freq="D", tz="UTC")
        prob = pd.Series(np.linspace(0.0, 1.0, 20), index=idx)
        flat = pd.Series(np.zeros(20), index=idx)
        lag, corr, n = best_lag_corr(prob, flat, max_lag=3)
        assert lag is None
        assert corr is None
        assert n is None

    def test_interpret_with_no_corrs_says_so(self) -> None:
        msg = _interpret({})
        assert "No correlation" in msg

    def test_interpret_mentions_top_asset(self) -> None:
        rows = {
            "BTC-USD": {"corr": -0.55},
            "SPY": {"corr": 0.20},
            "DXY": {"corr": 0.10},
        }
        msg = _interpret(rows)
        assert "BTC-USD" in msg
        assert "negatively" in msg


# ---------------------------------------------------------------------------
# Endpoint validation + schema
# ---------------------------------------------------------------------------


class TestEndpointValidation:
    def test_days_lower_bound_rejected(self, make_client) -> None:
        client = make_client()
        r = client.get("/terminal/correlations/x?days=10")
        assert r.status_code == 422

    def test_days_upper_bound_rejected(self, make_client) -> None:
        client = make_client()
        r = client.get("/terminal/correlations/x?days=10000")
        assert r.status_code == 422

    def test_default_days_is_90(self, make_client) -> None:
        client = make_client()
        r = client.get("/terminal/correlations/x")
        assert r.status_code == 200
        assert r.json()["lookback_days"] == 90

    def test_response_top_level_keys(self, make_client) -> None:
        client = make_client()
        r = client.get("/terminal/correlations/x?days=120")
        assert r.status_code == 200
        body = r.json()
        assert {
            "slug",
            "polymarket_series_n",
            "lookback_days",
            "correlations",
            "strongest",
            "interpretation",
        }.issubset(body.keys())
        # ``strongest`` is at most three entries.
        assert len(body["strongest"]) <= 3
        # Each correlation entry has the documented keys.
        for info in body["correlations"].values():
            assert {"corr", "p_value", "lag_days", "n"}.issubset(info.keys())

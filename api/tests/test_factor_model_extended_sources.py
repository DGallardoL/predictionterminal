"""Extended-source factor-model tests.

Verifies the dispatch in :mod:`pfm.factors` and :func:`_assemble_design`
correctly handles Manifold / PredictIt / BLS / FRED in addition to the
original Polymarket / Kalshi / chain shapes. All upstream HTTP is
mocked with ``respx`` so the suite stays offline.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
import respx
from fastapi.testclient import TestClient

import pfm.main as main_mod
from pfm.cache_utils import get_cache as _get_cache
from pfm.factors import (
    KNOWN_SOURCES,
    LEVEL_SOURCES,
    PROBABILITY_SOURCES,
    FactorConfig,
    fetch_factor_history_dispatch,
    load_factors,
)
from pfm.model import delta_level, delta_logit
from pfm.sources.bls import BLS_API_BASE
from pfm.sources.fred import FREDGRAPH_BASE
from pfm.sources.manifold import MANIFOLD_BASE_URL
from pfm.sources.predictit import PREDICTIT_BASE_URL

# ---------------------------------------------------------------------------
# load_factors — new sources / fields
# ---------------------------------------------------------------------------


def test_load_factors_accepts_all_new_sources(tmp_path: Path) -> None:
    p = tmp_path / "f.yml"
    p.write_text(
        """
factors:
  - id: pm
    name: PM
    slug: x
    source: polymarket
    description: pm
  - id: mf
    name: MF
    slug: trump-2028-mf
    source: manifold
    description: manifold sample
    is_probability: true
  - id: pi
    name: PI
    slug: '8200'
    source: predictit
    description: predictit sample
    is_probability: true
  - id: bls
    name: BLS
    slug: ICSA
    series_id: ICSA
    source: bls
    description: bls sample
    is_probability: false
  - id: fred
    name: FRED
    slug: T10Y2Y
    series_id: T10Y2Y
    source: fred
    description: fred sample
    is_probability: false
"""
    )
    factors = load_factors(p)
    assert set(factors) == {"pm", "mf", "pi", "bls", "fred"}
    assert factors["pm"].is_probability is True
    assert factors["mf"].source == "manifold"
    assert factors["pi"].source == "predictit"
    assert factors["bls"].source == "bls"
    assert factors["bls"].is_probability is False
    assert factors["bls"].effective_series_id == "ICSA"
    assert factors["fred"].is_probability is False
    assert factors["fred"].effective_series_id == "T10Y2Y"


def test_load_factors_defaults_is_probability_for_level_sources(tmp_path: Path) -> None:
    """``source: bls`` with no ``is_probability`` key should default to ``False``."""
    p = tmp_path / "f.yml"
    p.write_text(
        """
factors:
  - id: bls_default
    name: BLS default
    slug: ICSA
    source: bls
    description: 'no is_probability set — defaults to False for level sources'
"""
    )
    f = load_factors(p)["bls_default"]
    assert f.is_probability is False


def test_factor_config_rejects_level_source_with_is_probability_true() -> None:
    with pytest.raises(ValueError, match="requires is_probability=false"):
        FactorConfig(
            id="bad",
            name="bad",
            slug="ICSA",
            source="bls",
            description="bad",
            is_probability=True,
        )


def test_factor_config_rejects_unknown_source() -> None:
    with pytest.raises(ValueError, match="source must be one of"):
        FactorConfig(
            id="bad",
            name="bad",
            slug="x",
            source="weibo",
            description="bad",
        )


def test_known_sources_includes_extension() -> None:
    assert {"polymarket", "kalshi", "manifold", "predictit", "bls", "fred"} <= KNOWN_SOURCES
    assert frozenset() == PROBABILITY_SOURCES & LEVEL_SOURCES


# ---------------------------------------------------------------------------
# delta_logit guardrail
# ---------------------------------------------------------------------------


def test_delta_logit_guardrail_warns_on_out_of_range_input() -> None:
    """A series with values clearly outside [0, 1] should fall back to plain diff."""
    # Bond yield-style series in [-1.5, 4.0] — definitely not probabilities.
    levels = pd.Series([-1.5, -1.4, -1.2, 0.5, 2.1, 4.0])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = delta_logit(levels)
        # At least one warning was emitted
        assert any(
            "falls back" in str(rec.message) or "outside [0, 1]" in str(rec.message) for rec in w
        )
    # First entry is NaN (no predecessor); the rest equal plain diff().
    plain = levels.astype(float).diff()
    pd.testing.assert_series_equal(out.iloc[1:], plain.iloc[1:], check_names=False)


def test_delta_level_simple_diff() -> None:
    s = pd.Series([100.0, 101.5, 99.5, 102.0])
    out = delta_level(s)
    assert pd.isna(out.iloc[0])
    np.testing.assert_allclose(out.iloc[1:].values, np.array([1.5, -2.0, 2.5]), rtol=1e-9)


# ---------------------------------------------------------------------------
# fetch_factor_history_dispatch — Manifold / PredictIt / BLS / FRED
# ---------------------------------------------------------------------------


def _mk_factor(source: str, slug: str, **kw: object) -> FactorConfig:
    return FactorConfig(
        id=f"f_{source}",
        name=source,
        slug=slug,
        source=source,
        description=f"{source} test factor",
        **kw,  # type: ignore[arg-type]
    )


@respx.mock
def test_dispatch_manifold_normalises_to_price_column() -> None:
    """Manifold returns ``[date, prob, volume]``; dispatcher renames prob → price."""
    market_id = "abc123"
    respx.get(f"{MANIFOLD_BASE_URL}/slug/trump-2028-mf").mock(
        return_value=httpx.Response(200, json={"id": market_id, "slug": "trump-2028-mf"})
    )
    # 5 daily bets; client aggregates to last-prob per UTC day.
    base_ts_ms = int(pd.Timestamp("2026-04-01", tz="UTC").timestamp() * 1000)
    bets = [
        {"createdTime": base_ts_ms + i * 86_400_000, "probAfter": 0.40 + 0.01 * i, "amount": 100}
        for i in range(5)
    ]
    respx.get(f"{MANIFOLD_BASE_URL}/bets").mock(return_value=httpx.Response(200, json=bets))

    fc = _mk_factor("manifold", "trump-2028-mf", is_probability=True)
    df = fetch_factor_history_dispatch(
        fc,
        start=pd.Timestamp("2026-04-01", tz="UTC"),
        end=pd.Timestamp("2026-04-30", tz="UTC"),
    )
    assert list(df.columns) == ["price"]
    assert df.index.name == "date"
    assert (df["price"] >= 0.0).all() and (df["price"] <= 1.0).all()
    assert len(df) == 5


@respx.mock
def test_dispatch_predictit_normalises_to_price_column() -> None:
    market_id = 8200
    payload = {
        "id": market_id,
        "name": "2028 Trump nomination",
        "totalSharesTraded": 250_000,
        "contracts": [
            {"id": 1, "name": "Trump", "lastTradePrice": 0.65},
            {"id": 2, "name": "Other", "lastTradePrice": 0.21},
        ],
    }
    # Cold cache → falls back to per-market endpoint.
    _get_cache("predictit_all").clear()
    respx.get(f"{PREDICTIT_BASE_URL}/marketdata/markets/{market_id}").mock(
        return_value=httpx.Response(200, json=payload)
    )

    fc = _mk_factor("predictit", str(market_id), is_probability=True)
    df = fetch_factor_history_dispatch(
        fc,
        start=pd.Timestamp("2026-04-01", tz="UTC"),
        end=pd.Timestamp("2026-04-30", tz="UTC"),
    )
    assert list(df.columns) == ["price"]
    assert df.index.name == "date"
    assert len(df) == 1
    assert df["price"].iloc[0] == pytest.approx(0.65)


def _bls_payload(series_id: str, monthly_values: list[tuple[int, int, float]]) -> dict:
    """Build a BLS-shaped success payload."""
    return {
        "status": "REQUEST_SUCCEEDED",
        "responseTime": 0,
        "message": [],
        "Results": {
            "series": [
                {
                    "seriesID": series_id,
                    "data": [
                        {
                            "year": str(y),
                            "period": f"M{m:02d}",
                            "periodName": "April",
                            "value": str(v),
                            "footnotes": [],
                        }
                        for y, m, v in monthly_values
                    ],
                }
            ]
        },
    }


@respx.mock
def test_dispatch_bls_returns_level_series_no_logit() -> None:
    """BLS ICSA is a count series; dispatcher must keep it un-logit-transformed."""
    series_id = "ICSA"
    rows = [
        (2025, 1, 230_000.0),
        (2025, 2, 235_000.0),
        (2025, 3, 240_000.0),
        (2025, 4, 250_000.0),
    ]
    respx.post(BLS_API_BASE).mock(
        return_value=httpx.Response(200, json=_bls_payload(series_id, rows))
    )
    # Force-clear the BLS cache so the mock is exercised.
    _get_cache("bls-series").clear()

    fc = _mk_factor("bls", series_id, series_id=series_id, is_probability=False)
    df = fetch_factor_history_dispatch(
        fc,
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-12-31", tz="UTC"),
    )
    assert list(df.columns) == ["price"]
    assert df.index.name == "date"
    # Levels are far outside [0, 1] — guarding against accidental logit.
    assert df["price"].min() > 1.0
    assert df["price"].max() == pytest.approx(250_000.0)


@respx.mock
def test_dispatch_fred_returns_level_series() -> None:
    """FRED T10Y2Y is a yield spread (level), not a probability."""
    series_id = "T10Y2Y"
    csv = "DATE,T10Y2Y\n2025-09-01,0.50\n2025-09-02,0.48\n2025-09-03,0.52\n"
    respx.get(FREDGRAPH_BASE).mock(return_value=httpx.Response(200, text=csv))
    _get_cache("fred-series").clear()

    fc = _mk_factor("fred", series_id, series_id=series_id, is_probability=False)
    df = fetch_factor_history_dispatch(
        fc,
        start=pd.Timestamp("2025-09-01", tz="UTC"),
        end=pd.Timestamp("2025-09-03", tz="UTC"),
    )
    assert list(df.columns) == ["price"]
    assert df.index.name == "date"
    assert df["price"].iloc[0] == pytest.approx(0.50)
    assert df["price"].iloc[-1] == pytest.approx(0.52)


def test_dispatch_rejects_chain_source() -> None:
    """Chained factors must go through the dedicated chain fetcher."""
    from datetime import date

    from pfm.factors import ChainSegment

    fc = FactorConfig(
        id="ch",
        name="ch",
        slug="ch",
        source="chain",
        description="chain",
        segments=(ChainSegment(source="polymarket", slug="a", end=date(2025, 12, 31)),),
    )
    with pytest.raises(ValueError, match="chain"):
        fetch_factor_history_dispatch(
            fc,
            start=pd.Timestamp("2025-01-01", tz="UTC"),
            end=pd.Timestamp("2025-12-31", tz="UTC"),
        )


# ---------------------------------------------------------------------------
# /fit endpoint accepts mixed-source factors
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_factors_file(tmp_path: Path) -> Path:
    p = tmp_path / "factors.yml"
    p.write_text(
        """
factors:
  - id: pm_a
    name: PM A
    slug: slug-a
    source: polymarket
    description: polymarket factor A
  - id: mf_b
    name: Manifold B
    slug: manifold-slug-b
    source: manifold
    description: manifold factor B
    is_probability: true
  - id: fred_c
    name: FRED C
    slug: T10Y2Y
    series_id: T10Y2Y
    source: fred
    description: fred T10Y2Y level factor
    is_probability: false
"""
    )
    return p


@pytest.fixture
def mixed_app_client(
    monkeypatch: pytest.MonkeyPatch,
    mixed_factors_file: Path,
    fake_log_returns,
) -> Iterator[TestClient]:
    """TestClient with multi-source ``_cached_factor_history`` mocked.

    We patch ``_cached_factor_history`` directly so we don't have to
    juggle four real upstream mocks — the dispatch logic itself is
    covered by the unit tests above. This isolates the model-level
    path: factor-source heterogeneity must round-trip through ``/fit``
    cleanly and return one beta per factor.
    """
    monkeypatch.setenv("FACTORS_FILE", str(mixed_factors_file))
    import pfm.config as cfg

    cfg._settings = None

    rng = pd.date_range("2025-06-01", "2025-12-31", freq="D", tz="UTC")
    n = len(rng)
    t = np.arange(n) / n

    # PM A: probability oscillator in [0.05, 0.95].
    pm_prices = pd.DataFrame(
        {"price": (0.30 + 0.30 * np.sin(2 * np.pi * t * 1.2)).clip(0.05, 0.95)},
        index=rng,
    )
    pm_prices.index.name = "date"
    # Manifold B: probability series in [0.05, 0.95].
    mf_prices = pd.DataFrame(
        {"price": (0.55 + 0.20 * np.cos(2 * np.pi * t * 0.8)).clip(0.05, 0.95)},
        index=rng,
    )
    mf_prices.index.name = "date"
    # FRED C: yield-spread-style level series with realistic magnitude.
    fred_prices = pd.DataFrame(
        {"price": 0.50 + 0.05 * np.sin(2 * np.pi * t * 0.5)},
        index=rng,
    )
    fred_prices.index.name = "date"

    bank = {"pm_a": pm_prices, "mf_b": mf_prices, "fred_c": fred_prices}

    def _cached(fc, start, end, poly, cache, settings):
        df = bank[fc.id]
        return df[(df.index >= start) & (df.index <= end)]

    monkeypatch.setattr(main_mod, "_cached_factor_history", _cached)
    monkeypatch.setattr(main_mod, "get_log_returns", fake_log_returns)
    from pfm.cache import NullCache

    monkeypatch.setattr(main_mod, "RedisCache", lambda url: NullCache())

    with TestClient(main_mod.app) as client:
        yield client


def test_fit_accepts_manifold_factor(mixed_app_client: TestClient) -> None:
    """``/fit`` must accept a Manifold factor id without error."""
    body = {
        "ticker": "NVDA",
        "factors": ["mf_b"],
        "start": "2025-06-01",
        "end": "2025-12-15",
    }
    r = mixed_app_client.post("/fit", json=body)
    assert r.status_code == 200, r.text
    payload = r.json()
    factor_ids = [f["id"] for f in payload["factors"]]
    assert factor_ids == ["mf_b"]


def test_fit_accepts_predictit_factor(tmp_path, monkeypatch, fake_log_returns) -> None:
    """``/fit`` must accept a PredictIt factor id without 4xx."""
    p = tmp_path / "factors.yml"
    p.write_text(
        """
factors:
  - id: pi_a
    name: PI A
    slug: '8200'
    source: predictit
    description: predictit sample
    is_probability: true
"""
    )
    monkeypatch.setenv("FACTORS_FILE", str(p))
    import pfm.config as cfg

    cfg._settings = None

    rng = pd.date_range("2025-06-01", "2025-12-31", freq="D", tz="UTC")
    t = np.arange(len(rng)) / len(rng)
    prices = pd.DataFrame(
        {"price": (0.40 + 0.20 * np.sin(2 * np.pi * t * 1.0)).clip(0.05, 0.95)},
        index=rng,
    )
    prices.index.name = "date"

    def _cached(fc, start, end, poly, cache, settings):
        return prices[(prices.index >= start) & (prices.index <= end)]

    monkeypatch.setattr(main_mod, "_cached_factor_history", _cached)
    monkeypatch.setattr(main_mod, "get_log_returns", fake_log_returns)
    from pfm.cache import NullCache

    monkeypatch.setattr(main_mod, "RedisCache", lambda url: NullCache())

    with TestClient(main_mod.app) as client:
        body = {
            "ticker": "AAPL",
            "factors": ["pi_a"],
            "start": "2025-06-01",
            "end": "2025-12-15",
        }
        r = client.post("/fit", json=body)
        assert r.status_code == 200, r.text


def test_fit_accepts_fred_factor(mixed_app_client: TestClient) -> None:
    """``/fit`` must accept a FRED-level factor and not crash on logit."""
    body = {
        "ticker": "AAPL",
        "factors": ["fred_c"],
        "start": "2025-06-01",
        "end": "2025-12-15",
    }
    r = mixed_app_client.post("/fit", json=body)
    assert r.status_code == 200, r.text


def test_fit_multi_source_regression(mixed_app_client: TestClient) -> None:
    """End-to-end: NVDA ~ [polymarket, manifold, fred] with synthetic data.

    Confirms the model returns one beta per factor and that mixing a
    probability source with a level source does not raise. The actual
    coefficient values are not asserted (synthetic returns aren't
    constructed to make them deterministic across all three factors at
    once); we only verify shape + finiteness + factor ordering.
    """
    body = {
        "ticker": "NVDA",
        "factors": ["pm_a", "mf_b", "fred_c"],
        "start": "2025-06-01",
        "end": "2025-12-15",
    }
    r = mixed_app_client.post("/fit", json=body)
    assert r.status_code == 200, r.text
    payload = r.json()

    # Order is preserved and one beta per factor.
    factor_ids = [f["id"] for f in payload["factors"]]
    assert factor_ids == ["pm_a", "mf_b", "fred_c"]
    for f in payload["factors"]:
        assert isinstance(f["beta"], (int, float))
        assert np.isfinite(f["beta"])

    # R² is finite.
    assert np.isfinite(payload["model"]["r_squared"])


def test_existing_polymarket_factors_still_work(mixed_app_client: TestClient) -> None:
    """Backward-compat smoke: a single-source PM regression still passes."""
    body = {
        "ticker": "AAPL",
        "factors": ["pm_a"],
        "start": "2025-06-01",
        "end": "2025-12-15",
    }
    r = mixed_app_client.post("/fit", json=body)
    assert r.status_code == 200, r.text

"""Tests for ``pfm.terminal_calendar_curated`` — /terminal/calendar-curated/*.

The router is mounted on a fresh :class:`FastAPI` app (no Redis / no
main-app lifespan) and Polymarket fetches are replaced with a
deterministic stub so the suite is hermetic.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import terminal_calendar_curated as tcc
from pfm.factors import load_factors

# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def factors_index(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Load the real factors.yml shipped with the package — the curated
    table is intentionally tied to that file, so we exercise the actual
    mapping rather than a hand-mocked one.

    Other suites (notably ``conftest.app_client``) point ``Settings.factors_file``
    at a tmp_path-based factors fixture and cache the result on
    ``pfm.config._settings``. That cached value is *not* invalidated when
    the producing test tears down, so by the time we run the cached
    Settings still references a deleted tmp path. Reset the cache and
    clear ``FACTORS_FILE`` so this fixture always sees the shipped YAML.
    """
    import pfm.config as cfg
    from pfm.config import get_settings

    monkeypatch.delenv("FACTORS_FILE", raising=False)
    monkeypatch.setattr(cfg, "_settings", None)
    return load_factors(get_settings().factors_file)


def _make_fetcher(
    price_table: dict[str, float],
) -> Callable[..., pd.DataFrame]:
    """Return a stub `fetch_factor_history` keyed by Polymarket slug.

    The stub emits a 30-bar daily series ending today at the given
    constant price so ``_latest_price`` yields the desired mid.
    """
    end = pd.Timestamp(tcc._today(), tz="UTC").normalize()
    idx = pd.date_range(end - pd.Timedelta(days=29), end, freq="D")

    def _fetch(_client, slug: str, start=None, end=None):
        if slug not in price_table:
            return pd.DataFrame(columns=["price"]).set_index(pd.DatetimeIndex([], name="date"))
        series = pd.Series(price_table[slug], index=idx, name="price")
        df = series.to_frame()
        df.index.name = "date"
        return df

    return _fetch


@pytest.fixture
def client_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[dict[str, float]], TestClient]:
    """Return a factory that wires a TestClient with the given price table."""

    def _make(price_table: dict[str, float]) -> TestClient:
        fetcher = _make_fetcher(price_table)
        monkeypatch.setattr(tcc, "fetch_factor_history", fetcher)

        # Bypass the live PolymarketClient by overriding the dependency
        # — the stub never touches the client object, so any sentinel
        # works.
        app = FastAPI()
        app.include_router(tcc.router)
        app.dependency_overrides[tcc.get_polymarket_client] = object
        return TestClient(app)

    return _make


# --- tests ------------------------------------------------------------------


def test_curated_factor_ids_all_resolve_in_factors_yml(
    factors_index: dict[str, object],
) -> None:
    """Every curated factor_id must exist in the shipped factors.yml.

    This is the audit guardrail — if a YAML prune removes a leg the
    curated table needs to be updated in lock-step.
    """
    audit = tcc._audit()
    missing = {cid: ids for cid, ids in audit.items() if ids}
    assert not missing, f"curated clusters reference factor_ids not in factors.yml: {missing}"

    # And every cluster has at least 2 legs after resolution (no point
    # in a calendar surface with one contract).
    mapping = tcc.curated_factor_ids()
    for cluster_id, ids in mapping.items():
        survivors = [fid for fid in ids if fid in factors_index]
        assert len(survivors) >= 2, f"cluster {cluster_id} has <2 surviving legs: {survivors}"


def test_clusters_endpoint_returns_all_curated_clusters_with_signal(
    client_factory: Callable[[dict[str, float]], TestClient],
    factors_index: dict[str, object],
) -> None:
    """``GET /clusters`` returns one entry per cluster with a valid signal.

    With every leg priced at the same midpoint the front λ is *higher*
    than the back λ (same p, smaller T), so the canonical answer is
    ``FLATTEN_CURVE``.
    """
    # Build a price table: every leg priced at p = 0.30.
    prices: dict[str, float] = {}
    for cluster in tcc._CURATED_CLUSTERS:
        for leg in cluster.legs:
            cfg = factors_index.get(leg.factor_id)
            if cfg is not None:
                prices[cfg.slug] = 0.30

    client = client_factory(prices)
    r = client.get("/terminal/calendar-curated/clusters")
    assert r.status_code == 200, r.text

    body = r.json()
    cluster_ids = {c["cluster_id"] for c in body}
    expected_ids = {c.cluster_id for c in tcc._CURATED_CLUSTERS}
    # Every curated cluster with ≥2 surviving legs should appear.
    assert expected_ids.issubset(cluster_ids), f"missing cluster ids: {expected_ids - cluster_ids}"

    valid_signals = {
        "FLATTEN_CURVE",
        "STEEPEN_CURVE",
        "HOLD",
        "INSUFFICIENT_DATA",
    }
    for entry in body:
        assert entry["trade_signal"] in valid_signals
        assert entry["n_legs"] >= 2
        assert len(entry["legs"]) == entry["n_legs"]
        for leg in entry["legs"]:
            assert leg["days_to_resolve"] >= 0

    # With identical mids and only T differing, the front-month λ is
    # mechanically higher → FLATTEN_CURVE on every multi-leg cluster.
    flat_signals = [c for c in body if c["trade_signal"] == "FLATTEN_CURVE"]
    assert flat_signals, "expected at least one FLATTEN_CURVE under identical-mid prior"


def test_cluster_detail_returns_90_day_history_and_classification(
    client_factory: Callable[[dict[str, float]], TestClient],
    factors_index: dict[str, object],
) -> None:
    """``GET /{cluster_id}`` returns the full surface plus 90 daily rows.

    Picks the Powell-tenure cluster because it has three legs (and so
    exercises the ``min``/``max``/``std`` aggregators non-trivially).
    """
    cluster = next(c for c in tcc._CURATED_CLUSTERS if c.cluster_id == "powell_tenure")

    # Front leg: cheap (low hazard); back leg: rich (high hazard) — the
    # ratio λ_front / λ_back is then small ⇒ STEEPEN_CURVE.
    prices: dict[str, float] = {}
    for i, leg in enumerate(cluster.legs):
        cfg = factors_index[leg.factor_id]
        prices[cfg.slug] = 0.01 if i == 0 else 0.40

    client = client_factory(prices)
    r = client.get("/terminal/calendar-curated/powell_tenure")
    assert r.status_code == 200, r.text

    body = r.json()
    assert body["cluster_id"] == "powell_tenure"
    assert body["n_legs"] == 3
    assert body["trade_signal"] == "STEEPEN_CURVE"
    # ratio < 1 because front is cheap.
    assert 0.0 < body["lambda_ratio_front_back"] < 1.0
    # Three priced legs ⇒ std is positive.
    assert body["lambda_std"] is not None and body["lambda_std"] > 0.0

    history = body["historical_ratio"]
    assert len(history) == tcc.HIST_DAYS
    # Each row has a valid date and at least the per-leg λ fields populated.
    seen_dates: set[str] = set()
    for row in history:
        assert row["date"] not in seen_dates, "history dates must be unique"
        seen_dates.add(row["date"])
        # When both lambdas are present the ratio must be finite.
        if row["lambda_front"] is not None and row["lambda_back"] not in (None, 0.0):
            assert row["ratio"] is not None
            assert row["ratio"] > 0.0


def test_unknown_cluster_id_returns_404_and_short_path_is_validated(
    client_factory: Callable[[dict[str, float]], TestClient],
) -> None:
    """Sanity: 404 for unknown ids; FPath-validation enforced on length."""
    client = client_factory({})

    r = client.get("/terminal/calendar-curated/nope_does_not_exist")
    assert r.status_code == 404
    assert "unknown curated cluster" in r.json()["detail"]

    long_id = "x" * 200
    r2 = client.get(f"/terminal/calendar-curated/{long_id}")
    assert r2.status_code == 422

"""Tests for the Polymarket-backed live-signals fetcher.

Covers:

* ``_polymarket_live_fetcher`` returns inner-joined Series for two legs.
* Inner-join correctness: 100 obs vs 80 obs → 80 aligned rows.
* Failure: leg-A Gamma 404 raises a descriptive ``LookupError``.
* Failure: missing slug in the catalog raises a descriptive ``LookupError``.
* ``run_once(fetcher_kind="polymarket")`` exercises the real path through
  to the on-disk ``live_signals.json``.
* ``run_once(fetcher_kind="synthetic")`` (default) keeps working — proves
  the new branch hasn't regressed existing behaviour.
* ``GET /signals/connectivity-check`` happy path returns ``ok=True``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.cache_utils import reset_caches
from pfm.live_signals_job import (
    CLOB_URL,
    GAMMA_URL,
    _polymarket_live_fetcher,
    run_once,
    verify_polymarket_connectivity,
)
from pfm.live_signals_job import router as live_signals_router

# --- fixtures ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_caches_each_test() -> Any:
    """Clear every named cache before/after each test.

    The Polymarket fetcher caches ``(pair_id, a_id, b_id)`` for 600 s; we
    don't want a hit from one test to mask a route-level mismatch in
    the next.
    """
    reset_caches()
    yield
    reset_caches()


def _gamma_market_response(slug: str, token_id: str) -> httpx.Response:
    """Build a minimal Gamma ``/markets?slug=`` body with one market."""
    return httpx.Response(
        200,
        json=[
            {
                "slug": slug,
                # clobTokenIds arrives as a JSON-encoded string inside JSON.
                "clobTokenIds": json.dumps([token_id, f"{token_id}-no"]),
            }
        ],
    )


def _clob_history_payload(prices: list[float], *, start_unix: int) -> dict[str, Any]:
    """Daily-spaced history payload: one bar per day, oldest → newest."""
    one_day = 86_400
    return {"history": [{"t": start_unix + i * one_day, "p": p} for i, p in enumerate(prices)]}


def _write_catalog(
    tmp_path: Path,
    *,
    pair_id: str = "aaa__bbb",
    a_id: str = "aaa",
    b_id: str = "bbb",
    a_slug: str = "slug-a",
    b_slug: str = "slug-b",
) -> Path:
    """Write a tiny ``alpha_strategies.json`` with one pair."""
    catalog = {
        "strategies": [
            {
                "pair_id": pair_id,
                "a_id": a_id,
                "b_id": b_id,
                "a_slug": a_slug,
                "b_slug": b_slug,
                "beta_hedge": 1.0,
                "rule_window": 10,
                "rule_entry_z": 2.0,
                "rule_exit_z": 0.5,
                "rule_stop_z": 4.0,
            }
        ]
    }
    p = tmp_path / "alpha_strategies.json"
    p.write_text(json.dumps(catalog))
    return p


# --- _polymarket_live_fetcher happy path -------------------------------------


@respx.mock
def test_polymarket_fetcher_returns_aligned_series(tmp_path: Path) -> None:
    """Both legs return aligned Series indexed by UTC dates."""
    catalog_path = _write_catalog(tmp_path)

    # Gamma: route on the slug query param so each leg gets the right token.
    def _gamma(req: httpx.Request) -> httpx.Response:
        slug = req.url.params["slug"]
        if slug == "slug-a":
            return _gamma_market_response(slug, "tok-a")
        if slug == "slug-b":
            return _gamma_market_response(slug, "tok-b")
        return httpx.Response(404, json=[])

    respx.get(f"{GAMMA_URL}/markets").mock(side_effect=_gamma)

    # CLOB: route on the market token param.
    start = 1_700_000_000  # arbitrary recent unix seconds
    a_prices = [0.40 + 0.001 * i for i in range(30)]
    b_prices = [0.55 - 0.001 * i for i in range(30)]

    def _clob(req: httpx.Request) -> httpx.Response:
        token = req.url.params["market"]
        if token == "tok-a":
            return httpx.Response(200, json=_clob_history_payload(a_prices, start_unix=start))
        if token == "tok-b":
            return httpx.Response(200, json=_clob_history_payload(b_prices, start_unix=start))
        return httpx.Response(404, json={"history": []})

    respx.get(f"{CLOB_URL}/prices-history").mock(side_effect=_clob)

    a_out, b_out = asyncio.run(
        _polymarket_live_fetcher("aaa__bbb", "aaa", "bbb", catalog_path=catalog_path)
    )
    assert isinstance(a_out, pd.Series)
    assert isinstance(b_out, pd.Series)
    assert len(a_out) == 30
    assert len(b_out) == 30
    # Both legs must share the same index after the inner join.
    assert list(a_out.index) == list(b_out.index)
    # And the index must be UTC-localised daily Timestamps.
    assert all(ts.tzinfo is not None for ts in a_out.index)


# --- inner-join correctness --------------------------------------------------


@respx.mock
def test_polymarket_fetcher_inner_join_truncates_to_overlap(tmp_path: Path) -> None:
    """leg_a 100 obs + leg_b 80 obs (offset start) → 80 aligned rows."""
    catalog_path = _write_catalog(tmp_path)
    start_a = 1_700_000_000
    one_day = 86_400
    # leg_b starts 20 days later, so the overlap is 80 days.
    start_b = start_a + 20 * one_day

    def _gamma(req: httpx.Request) -> httpx.Response:
        slug = req.url.params["slug"]
        token = "tok-a" if slug == "slug-a" else "tok-b"
        return _gamma_market_response(slug, token)

    respx.get(f"{GAMMA_URL}/markets").mock(side_effect=_gamma)

    a_prices = [0.50 + 0.0005 * i for i in range(100)]
    b_prices = [0.45 + 0.0005 * i for i in range(80)]

    def _clob(req: httpx.Request) -> httpx.Response:
        token = req.url.params["market"]
        if token == "tok-a":
            return httpx.Response(200, json=_clob_history_payload(a_prices, start_unix=start_a))
        return httpx.Response(200, json=_clob_history_payload(b_prices, start_unix=start_b))

    respx.get(f"{CLOB_URL}/prices-history").mock(side_effect=_clob)

    # window_days big enough to keep the full overlap.
    a_out, b_out = asyncio.run(
        _polymarket_live_fetcher(
            "aaa__bbb",
            "aaa",
            "bbb",
            window_days=200,
            catalog_path=catalog_path,
        )
    )
    assert len(a_out) == 80
    assert len(b_out) == 80
    assert list(a_out.index) == list(b_out.index)


# --- failure cases -----------------------------------------------------------


@respx.mock
def test_polymarket_fetcher_raises_on_gamma_404(tmp_path: Path) -> None:
    """A missing market on leg-A surfaces as a descriptive LookupError."""
    catalog_path = _write_catalog(tmp_path)

    def _gamma(req: httpx.Request) -> httpx.Response:
        slug = req.url.params["slug"]
        if slug == "slug-a":
            return httpx.Response(404, json=[])
        return _gamma_market_response(slug, "tok-b")

    respx.get(f"{GAMMA_URL}/markets").mock(side_effect=_gamma)
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json=_clob_history_payload([0.5], start_unix=1))
    )

    with pytest.raises(LookupError, match="slug-a"):
        asyncio.run(_polymarket_live_fetcher("aaa__bbb", "aaa", "bbb", catalog_path=catalog_path))


def test_polymarket_fetcher_raises_when_slug_missing_in_catalog(
    tmp_path: Path,
) -> None:
    """If ``a_slug`` is missing from the catalog the fetcher refuses early."""
    catalog = {
        "strategies": [
            {
                "pair_id": "p",
                "a_id": "aaa",
                "b_id": "bbb",
                # No a_slug at all.
                "b_slug": "slug-b",
            }
        ]
    }
    cat = tmp_path / "c.json"
    cat.write_text(json.dumps(catalog))
    with pytest.raises(LookupError, match="no a_slug"):
        asyncio.run(_polymarket_live_fetcher("p", "aaa", "bbb", catalog_path=cat))


# --- run_once integration ----------------------------------------------------


@respx.mock
def test_run_once_with_polymarket_fetcher_writes_signals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``fetcher_kind='polymarket'`` runs the real path end to end."""
    catalog_path = _write_catalog(tmp_path)
    monkeypatch.setenv("PFM_ALPHA_STRATEGIES_PATH", str(catalog_path))

    def _gamma(req: httpx.Request) -> httpx.Response:
        slug = req.url.params["slug"]
        token = "tok-a" if slug == "slug-a" else "tok-b"
        return _gamma_market_response(slug, token)

    respx.get(f"{GAMMA_URL}/markets").mock(side_effect=_gamma)

    start = 1_700_000_000
    a_prices = [0.30 + 0.002 * i for i in range(40)]
    b_prices = [0.60 - 0.002 * i for i in range(40)]

    def _clob(req: httpx.Request) -> httpx.Response:
        token = req.url.params["market"]
        prices = a_prices if token == "tok-a" else b_prices
        return httpx.Response(200, json=_clob_history_payload(prices, start_unix=start))

    respx.get(f"{CLOB_URL}/prices-history").mock(side_effect=_clob)

    write_path = tmp_path / "live.json"
    status_path = tmp_path / "status.json"
    summary = asyncio.run(
        run_once(
            write_path=write_path,
            strategies_path=catalog_path,
            status_path=status_path,
            fetcher_kind="polymarket",
        )
    )
    assert summary["n_alphas_total"] == 1
    assert summary["n_alphas_failed"] == 0
    assert summary["n_alphas_updated"] == 1

    payload = json.loads(write_path.read_text())
    assert "aaa__bbb" in payload["signals"]
    sig = payload["signals"]["aaa__bbb"]
    # The signal payload has the standard fields produced by the real
    # compute path — z-score finite, action one of the known labels.
    assert sig["n_obs"] == 40
    assert sig["current_z"] is not None
    assert sig["action"] in {
        "OPEN_LONG",
        "OPEN_SHORT",
        "HOLD",
        "CLOSE",
        "STOP_OUT",
        "FLAT",
    }


def test_run_once_default_synthetic_still_works(tmp_path: Path) -> None:
    """No-network synthetic path keeps behaving the same — no regression."""
    catalog_path = _write_catalog(tmp_path)
    write_path = tmp_path / "live.json"
    status_path = tmp_path / "status.json"
    summary = asyncio.run(
        run_once(
            write_path=write_path,
            strategies_path=catalog_path,
            status_path=status_path,
            # fetcher_kind defaults to "synthetic"; no fetcher / pair_fetcher.
        )
    )
    assert summary["n_alphas_total"] == 1
    assert summary["n_alphas_failed"] == 0
    payload = json.loads(write_path.read_text())
    assert "aaa__bbb" in payload["signals"]
    assert payload["signals"]["aaa__bbb"]["current_z"] is not None


# --- /signals/connectivity-check ---------------------------------------------


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(live_signals_router)
    return app


@respx.mock
def test_connectivity_check_endpoint_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probe returns ``ok=True`` with the expected sample size."""
    monkeypatch.delenv("PFM_ADMIN_TOKEN", raising=False)
    sample_slug = "test-sample-slug"
    monkeypatch.setenv("PFM_CONNECTIVITY_SAMPLE_SLUG", sample_slug)

    respx.get(f"{GAMMA_URL}/markets").mock(
        return_value=_gamma_market_response(sample_slug, "tok-sample")
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(
            200,
            json=_clob_history_payload(
                [0.50 + 0.001 * i for i in range(180)], start_unix=1_700_000_000
            ),
        )
    )

    app = _make_app()
    client = TestClient(app)
    r = client.get("/signals/connectivity-check")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["sample_size"] == 180
    assert body["error"] is None
    assert body["latency_ms"] >= 0.0
    assert body["slug"] == sample_slug


@respx.mock
def test_verify_connectivity_returns_error_on_404() -> None:
    """A missing market produces ``ok=False`` plus an ``error`` string."""
    respx.get(f"{GAMMA_URL}/markets").mock(return_value=httpx.Response(404, json=[]))
    out = asyncio.run(verify_polymarket_connectivity(sample_slug="missing-slug"))
    assert out["ok"] is False
    assert out["sample_size"] == 0
    assert out["error"] is not None
    assert "missing-slug" in out["error"] or "LookupError" in out["error"]
    assert out["slug"] == "missing-slug"

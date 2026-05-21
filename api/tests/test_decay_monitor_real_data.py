"""Tests for the real-data path in :mod:`pfm.decay_monitor`.

Scope:

* ``_load_real_returns`` reads ``live_signals.json`` when present and
  the entry carries a usable trail (``spread_history`` / ``z_history``).
* ``_load_real_returns`` falls through to a Polymarket history fetch
  when ``live_signals`` lacks a trail; we monkey-patch the leg fetcher
  to keep the test offline.
* Both real paths failing must degrade silently to the synthetic
  fallback while logging a warning.
* Endpoint telemetry (``/alpha/decay``) reports ``n_using_real_data``
  / ``n_using_synthetic_fallback`` / ``data_quality_warning`` correctly
  in the mixed case.
* ``GET /{pair_id}/rolling-sharpe`` carries ``source_used`` and
  ``data_quality_note``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import pfm.decay_monitor as dm
from pfm.cache_utils import get_cache
from pfm.decay_monitor import (
    _load_real_returns,
    _resolve_returns_for_endpoint,
    check_all_alphas,
)
from pfm.decay_monitor import router as decay_router

# --- shared fixtures --------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_decay_real_cache() -> Iterator[None]:
    """Drop the ``decay_real`` named cache between tests.

    Without this the first test that classifies ``pair_alpha`` would
    pin the result for the rest of the session and the mixed-mode tests
    further down would re-read a stale entry.
    """
    get_cache("decay_real", ttl=dm.DECAY_REAL_CACHE_TTL).clear()
    yield
    get_cache("decay_real", ttl=dm.DECAY_REAL_CACHE_TTL).clear()


@pytest.fixture
def alpha_catalog(tmp_path: Path) -> Path:
    """Tiny ``alpha_strategies.json`` covering both real and synth paths."""
    payload = {
        "generated": "2026-05-08",
        "strategies": [
            {
                "pair_id": "pair_alpha",
                "a_id": "alpha_a",
                "b_id": "alpha_b",
                "a_slug": "alpha-a-slug",
                "b_slug": "alpha-b-slug",
                "tier": "A_GOLD",
                "oos_sharpe": 2.0,
                "beta_hedge": 1.0,
            },
            {
                "pair_id": "pair_beta",
                "a_id": "beta_a",
                "b_id": "beta_b",
                "a_slug": "beta-a-slug",
                "b_slug": "beta-b-slug",
                "tier": "A_GOLD",
                "oos_sharpe": 1.5,
                "beta_hedge": 1.0,
            },
        ],
    }
    p = tmp_path / "alpha_strategies.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _live_signals_with_trail(pair_id: str, *, n: int = 30, drift: float = 0.0) -> dict[str, Any]:
    """Synthesise a ``live_signals.json``-shaped payload with a trail."""
    spread_trail = [drift * i + (i % 5) * 0.01 for i in range(n)]
    return {
        "as_of": "2026-05-08T00:00:00+00:00",
        "n_strategies": 1,
        "signals": {
            pair_id: {
                "pair_id": pair_id,
                "a_id": "x",
                "b_id": "y",
                "as_of": "2026-05-08T00:00:00+00:00",
                "n_obs": n,
                "current_spread": spread_trail[-1],
                "current_z": 0.0,
                "spread_history": spread_trail,
            }
        },
    }


@pytest.fixture
def live_signals_file(tmp_path: Path) -> Path:
    """``live_signals.json`` with a trail for ``pair_alpha`` only.

    ``pair_beta`` is intentionally absent so the mixed-mode test can
    exercise the fall-through path on a per-pair basis.
    """
    payload = _live_signals_with_trail("pair_alpha", n=40)
    p = tmp_path / "live_signals.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# --- _load_real_returns -----------------------------------------------------


def test_load_real_returns_uses_live_signals(alpha_catalog: Path) -> None:
    payload = _live_signals_with_trail("pair_alpha", n=25)
    alpha = json.loads(alpha_catalog.read_text())["strategies"][0]
    returns, source = _load_real_returns(
        "pair_alpha",
        alpha,
        live_signals=payload,
        allow_polymarket=False,
    )
    assert source == "live_signals"
    assert returns is not None
    assert isinstance(returns, pd.Series)
    assert len(returns) >= 5


def test_load_real_returns_polymarket_fallback(
    alpha_catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When live_signals is missing, _load_real_returns hits Polymarket.

    We monkey-patch the leg fetchers so no network call leaves the box.
    """
    alpha = json.loads(alpha_catalog.read_text())["strategies"][0]

    def _fake_token_ids(http: Any, slug: str, *, timeout: float = 5.0) -> tuple[str, str]:
        return (f"yes_{slug}", f"no_{slug}")

    def _fake_daily_prices(
        http: Any, token_id: str, *, days: int, timeout: float = 5.0
    ) -> pd.Series:
        idx = pd.date_range("2026-04-01", periods=30, freq="D", tz="UTC").normalize()
        # Differing patterns per token so the spread isn't degenerate.
        offset = 0.05 if "yes_alpha-a-slug" in token_id else 0.10
        values = [offset + 0.001 * i for i in range(30)]
        return pd.Series(values, index=idx, name="price")

    monkeypatch.setattr(dm, "_fetch_clob_token_ids", _fake_token_ids)
    monkeypatch.setattr(dm, "_fetch_clob_daily_prices", _fake_daily_prices)

    returns, source = _load_real_returns(
        "pair_alpha",
        alpha,
        live_signals=None,
        allow_polymarket=True,
    )
    assert source == "polymarket_history"
    assert returns is not None
    assert len(returns) >= 5


def test_load_real_returns_both_fail_returns_none(
    alpha_catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alpha = json.loads(alpha_catalog.read_text())["strategies"][0]

    def _no_tokens(http: Any, slug: str, *, timeout: float = 5.0) -> tuple[None, None]:
        return (None, None)

    monkeypatch.setattr(dm, "_fetch_clob_token_ids", _no_tokens)

    returns, source = _load_real_returns(
        "pair_alpha",
        alpha,
        live_signals=None,
        allow_polymarket=True,
    )
    assert returns is None
    assert source is None


def test_load_real_returns_skips_error_entry(alpha_catalog: Path) -> None:
    """An entry with ``error`` field must NOT count as live-signals data."""
    payload = {
        "signals": {
            "pair_alpha": {
                "pair_id": "pair_alpha",
                "error": "factor not found (alpha_a or alpha_b)",
            }
        }
    }
    alpha = json.loads(alpha_catalog.read_text())["strategies"][0]
    returns, source = _load_real_returns(
        "pair_alpha",
        alpha,
        live_signals=payload,
        allow_polymarket=False,
    )
    assert returns is None
    assert source is None


# --- check_all_alphas / endpoint counts -------------------------------------


def test_check_all_alphas_counts_real_and_synthetic(
    alpha_catalog: Path,
    live_signals_file: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Mixed: pair_alpha → live_signals; pair_beta → synthetic fallback."""
    caplog.set_level("WARNING", logger="pfm.decay_monitor")
    result = check_all_alphas(
        str(alpha_catalog),
        data_source="live_signals",
        live_signals_path=str(live_signals_file),
        allow_polymarket=False,
    )
    # Backward-compat: the mapping still exposes pair_id -> status.
    assert set(result.keys()) == {"pair_alpha", "pair_beta"}
    assert result.n_using_real_data == 1
    assert result.n_using_synthetic_fallback == 1
    assert result["pair_alpha"]["source_used"] == "live_signals"
    assert result["pair_beta"]["source_used"] == "synthetic_fallback"
    assert result["pair_beta"]["data_quality_note"] is not None


def test_check_all_alphas_synthetic_only(alpha_catalog: Path, tmp_path: Path) -> None:
    """No live_signals + polymarket disabled → everything is synthetic."""
    missing = tmp_path / "no_live_signals.json"
    result = check_all_alphas(
        str(alpha_catalog),
        data_source="live_signals",
        live_signals_path=str(missing),
        allow_polymarket=False,
    )
    assert result.n_using_real_data == 0
    assert result.n_using_synthetic_fallback == 2
    for status in result.values():
        assert status["source_used"] == "synthetic_fallback"


def test_check_all_alphas_explicit_synthetic_short_circuits(
    alpha_catalog: Path, live_signals_file: Path
) -> None:
    """``data_source='synthetic_fallback'`` must skip live_signals entirely."""
    result = check_all_alphas(
        str(alpha_catalog),
        data_source="synthetic_fallback",
        live_signals_path=str(live_signals_file),
    )
    assert result.n_using_real_data == 0
    assert result.n_using_synthetic_fallback == 2


# --- endpoint integration ---------------------------------------------------


@pytest.fixture
def decay_app_for_real(alpha_catalog: Path, live_signals_file: Path) -> TestClient:
    app = FastAPI()
    app.include_router(decay_router)
    app.state.alpha_catalog_path = str(alpha_catalog)
    app.state.live_signals_path = str(live_signals_file)
    return TestClient(app)


def test_endpoint_decay_reports_real_and_synth_counts(
    decay_app_for_real: TestClient,
) -> None:
    catalog = decay_app_for_real.app.state.alpha_catalog_path
    live = decay_app_for_real.app.state.live_signals_path
    resp = decay_app_for_real.get(
        "/alpha/decay",
        params={
            "alpha_strategies_path": catalog,
            "live_signals_path": live,
            "data_source": "live_signals",
            "allow_polymarket": False,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["n_total"] == 2
    assert body["n_using_real_data"] == 1
    assert body["n_using_synthetic_fallback"] == 1
    # 50% synthetic — at the boundary, no warning expected.
    assert body["data_quality_warning"] is None
    pairs = {it["pair_id"]: it for it in body["items"]}
    assert pairs["pair_alpha"]["source_used"] == "live_signals"
    assert pairs["pair_beta"]["source_used"] == "synthetic_fallback"


def test_endpoint_decay_data_quality_warning_when_majority_synth(
    alpha_catalog: Path, tmp_path: Path
) -> None:
    """All synthetic → warning fires (>50% threshold)."""
    app = FastAPI()
    app.include_router(decay_router)
    client = TestClient(app)

    missing = tmp_path / "no_live_signals.json"
    resp = client.get(
        "/alpha/decay",
        params={
            "alpha_strategies_path": str(alpha_catalog),
            "live_signals_path": str(missing),
            "data_source": "live_signals",
            "allow_polymarket": False,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["n_using_synthetic_fallback"] == 2
    assert body["data_quality_warning"] is not None
    assert "synthetic" in body["data_quality_warning"].lower()


def test_endpoint_rolling_sharpe_reports_source_used(
    decay_app_for_real: TestClient,
) -> None:
    catalog = decay_app_for_real.app.state.alpha_catalog_path
    live = decay_app_for_real.app.state.live_signals_path
    # pair_alpha has live_signals → source_used == live_signals.
    resp = decay_app_for_real.get(
        "/alpha/pair_alpha/rolling-sharpe",
        params={
            "window": 10,
            "alpha_strategies_path": catalog,
            "live_signals_path": live,
            "data_source": "live_signals",
            "allow_polymarket": False,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source_used"] == "live_signals"
    assert body["data_quality_note"] is None
    # pair_beta must fall through to synthetic and surface the note.
    resp2 = decay_app_for_real.get(
        "/alpha/pair_beta/rolling-sharpe",
        params={
            "window": 10,
            "alpha_strategies_path": catalog,
            "live_signals_path": live,
            "data_source": "live_signals",
            "allow_polymarket": False,
        },
    )
    assert resp2.status_code == 200, resp2.text
    body2 = resp2.json()
    assert body2["source_used"] == "synthetic_fallback"
    assert body2["data_quality_note"] is not None


def test_resolve_returns_for_endpoint_synthetic_branch(
    alpha_catalog: Path,
) -> None:
    """Direct unit-level coverage of the per-pair returns resolver."""
    alpha = json.loads(alpha_catalog.read_text())["strategies"][0]
    returns, src, note = _resolve_returns_for_endpoint(
        "pair_alpha",
        alpha,
        baseline=2.0,
        data_source="synthetic_fallback",
        live_signals_path="/does/not/exist",
        allow_polymarket=False,
    )
    assert src == "synthetic_fallback"
    assert note is not None
    assert len(returns) > 30


# --- run_once + status cache ------------------------------------------------


@pytest.mark.anyio("asyncio")
async def test_run_once_writes_status_cache(
    alpha_catalog: Path, live_signals_file: Path, tmp_path: Path
) -> None:
    """``run_once`` must persist a status JSON and report counts correctly."""
    status_path = tmp_path / "decay_status.json"
    payload = await dm.run_once(
        alpha_strategies_path=str(alpha_catalog),
        live_signals_path=str(live_signals_file),
        status_path=str(status_path),
        data_source="live_signals",
    )
    assert status_path.is_file()
    on_disk = json.loads(status_path.read_text())
    assert on_disk["n_total"] == 2
    assert on_disk["n_using_real_data"] == payload["n_using_real_data"]
    assert on_disk["n_using_synthetic_fallback"] == payload["n_using_synthetic_fallback"]
    assert "last_refreshed_iso" in on_disk

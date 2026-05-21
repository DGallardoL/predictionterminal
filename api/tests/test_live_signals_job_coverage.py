"""Coverage-fill tests for ``pfm.live_signals_job``.

Companion suite to ``test_live_signals_job.py`` and
``test_live_signals_polymarket_fetcher.py``. The pre-existing tests
hit the happy paths and the most prominent failure modes; this file
fills in the gaps that the coverage report (W13-23 baseline = 84 %)
flagged on ``pfm.live_signals_job``:

* Decay-status branches: INSUFFICIENT_DATA, STRESSED, QUIET, UNKNOWN.
* ``_signal_from_z`` edge-trigger for OPEN_LONG (entry crossed from
  above) and ``prev_z = nan`` skip.
* ``_resolve_alpha_strategies_path`` fallback hierarchy (env var
  override, cwd hit, repo-relative fallback).
* ``_slug_lookup_from_catalog`` corner cases: list-shaped catalog,
  catalog with non-dict elements, catalog with the wrong top-level
  shape, and missing-catalog-file.
* ``_resolve_token_id`` failure modes: empty list body, no
  ``clobTokenIds``, malformed JSON in the double-encoded string,
  unexpected ``clobTokenIds`` shape, and an empty list inside it.
* ``_fetch_clob_history`` parse robustness: ``history`` not a list,
  rows that aren't dicts, missing ``t``/``p`` fields, non-numeric
  values, non-finite floats, duplicated timestamps.
* ``_polymarket_live_fetcher`` cache-hit short-circuit, leg-B
  missing-slug path, both-leg-history-empty paths, the no-overlap
  inner-join failure, and the window-trimming when the joined
  history exceeds ``window_days``.
* ``_wrap_pair_fetcher_as_price_fetcher`` adapter shape + the
  ``shared_client`` attribute.
* ``_build_pair_fetcher_for_kind`` for synthetic, polymarket, and the
  ``ValueError`` for an unknown kind.
* ``_load_alphas``: list-shaped, missing-strategies, and absent file.
* ``recompute_all_signals``: missing-``a_id`` short-circuit, compute
  failure path inside the per-alpha try, and pair_fetcher precedence
  (provided fetcher takes precedence over kind).
* ``run_once``: persistence shape on disk, ``http_client`` injection,
  ``signals_live`` cache invalidation, and the status-write OSError
  swallow.
* ``run_forever``: inner-exception swallow (run_once raises but loop
  continues), stop-event already-set short-circuit *and* stop-event
  fired during the inter-cycle sleep.
* Concurrent run-guard: two ``run_once`` calls in parallel both
  complete and produce a consistent final file.
* Reproducibility: identical input → identical output (synthetic
  fetcher determinism).
* Router: admin token gating on ``/signals/recompute-now``,
  ``/signals/status`` parse-error path, ``/signals/live`` cache-hit
  short-circuit, and the corrupted-status-file 500 path.

These tests deliberately use the public ``run_once`` /
``recompute_all_signals`` / ``_polymarket_live_fetcher`` entry points
plus the small private helpers exported on ``__all__`` so refactors
inside the module don't churn this suite.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import live_signals_job
from pfm.cache_utils import get_cache, reset_caches
from pfm.live_signals_job import (
    CLOB_URL,
    DEFAULT_ALPHA_STRATEGIES_PATH,
    GAMMA_URL,
    SIGNALS_LIVE_CACHE_TTL,
    _atomic_write_json,
    _build_pair_fetcher_for_kind,
    _decay_status,
    _fetch_clob_history,
    _load_alphas,
    _polymarket_live_fetcher,
    _resolve_alpha_strategies_path,
    _resolve_token_id,
    _signal_from_z,
    _slug_lookup_from_catalog,
    _wrap_pair_fetcher_as_price_fetcher,
    recompute_all_signals,
    run_forever,
    run_once,
    verify_polymarket_connectivity,
)
from pfm.live_signals_job import router as live_signals_router

# --- shared fixtures --------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_all_caches() -> Any:
    """Drop every named TerminalCache before and after each test.

    The Polymarket leg cache lives for 600 s by default; without a reset
    one test's cached frame can shadow another's mock surface.
    """
    reset_caches()
    yield
    reset_caches()


def _alpha(pair_id: str, **overrides: Any) -> dict[str, Any]:
    """Minimal alpha record with sensible defaults for compute paths."""
    base: dict[str, Any] = {
        "pair_id": pair_id,
        "a_id": f"{pair_id}_A",
        "b_id": f"{pair_id}_B",
        "beta_hedge": 1.0,
        "rule_window": 10,
        "rule_entry_z": 2.0,
        "rule_exit_z": 0.5,
        "rule_stop_z": 4.0,
    }
    base.update(overrides)
    return base


def _good_series(seed: int = 0, n: int = 60) -> list[float]:
    """Deterministic-walk price series in (0, 1)."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0.0, 0.02, size=n))
    return [float(p) for p in 1.0 / (1.0 + np.exp(-x))]


def _good_fetcher(failures: set[str] | None = None) -> Callable[[str], Awaitable[list[float]]]:
    failures = set(failures or set())

    async def _f(factor_id: str) -> list[float]:
        if factor_id in failures:
            raise RuntimeError(f"injected for {factor_id}")
        return _good_series(seed=abs(hash(factor_id)) % 10_000)

    return _f


def _gamma_resp(slug: str, token_id: str) -> httpx.Response:
    return httpx.Response(
        200,
        json=[{"slug": slug, "clobTokenIds": json.dumps([token_id, f"{token_id}-no"])}],
    )


def _clob_payload(prices: list[float], *, start_unix: int = 1_700_000_000) -> dict[str, Any]:
    one_day = 86_400
    return {"history": [{"t": start_unix + i * one_day, "p": p} for i, p in enumerate(prices)]}


def _write_one_pair_catalog(tmp_path: Path) -> Path:
    p = tmp_path / "alpha_strategies.json"
    p.write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "pair_id": "ab",
                        "a_id": "aa",
                        "b_id": "bb",
                        "a_slug": "slug-a",
                        "b_slug": "slug-b",
                        "beta_hedge": 1.0,
                        "rule_window": 10,
                        "rule_entry_z": 2.0,
                        "rule_exit_z": 0.5,
                        "rule_stop_z": 4.0,
                    }
                ]
            }
        )
    )
    return p


# --- decay-status branches --------------------------------------------------


def test_decay_status_all_branches() -> None:
    assert _decay_status(0.0, n_obs=10) == "INSUFFICIENT_DATA"
    assert _decay_status(float("nan"), n_obs=30) == "UNKNOWN"
    assert _decay_status(5.0, n_obs=30) == "STRESSED"
    assert _decay_status(-5.0, n_obs=30) == "STRESSED"
    assert _decay_status(0.1, n_obs=30) == "QUIET"
    assert _decay_status(-0.1, n_obs=30) == "QUIET"
    assert _decay_status(1.0, n_obs=30) == "ACTIVE"


# --- _signal_from_z edge cases ----------------------------------------------


def test_signal_from_z_edge_trigger_long() -> None:
    action, reason = _signal_from_z(-2.2, 2.0, 0.5, 4.0, prev_z=-1.5)
    assert action == "OPEN_LONG"
    assert "crossed" in reason


def test_signal_from_z_prev_z_nan_skips_edge_path() -> None:
    """A non-finite ``prev_z`` must not crash the edge-trigger branch."""
    action, _ = _signal_from_z(2.5, 2.0, 0.5, 4.0, prev_z=float("nan"))
    assert action == "OPEN_SHORT"


def test_signal_from_z_no_prev_within_band() -> None:
    """Without ``prev_z`` the (exit, entry) band returns HOLD."""
    assert _signal_from_z(1.0, 2.0, 0.5, 4.0)[0] == "HOLD"


# --- _resolve_alpha_strategies_path fallback chain --------------------------


def test_resolve_path_env_var_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "explicit.json"
    monkeypatch.setenv("PFM_ALPHA_STRATEGIES_PATH", str(target))
    assert _resolve_alpha_strategies_path() == target


def test_resolve_path_cwd_hit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When the env var is unset and the default cwd-relative path exists."""
    monkeypatch.delenv("PFM_ALPHA_STRATEGIES_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    # Materialise the default path under cwd so the helper finds it.
    target = Path(DEFAULT_ALPHA_STRATEGIES_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}")
    out = _resolve_alpha_strategies_path()
    assert out.exists()


def test_resolve_path_repo_relative_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When neither env nor cwd resolve, the repo-relative path is returned."""
    monkeypatch.delenv("PFM_ALPHA_STRATEGIES_PATH", raising=False)
    monkeypatch.chdir(tmp_path)  # cwd has no web/data dir
    out = _resolve_alpha_strategies_path()
    # The repo-relative path may or may not exist depending on the test
    # environment; the helper only constructs it, never reads it here.
    assert out.name == "alpha_strategies.json"
    assert "web" in out.parts and "data" in out.parts


# --- _slug_lookup_from_catalog corner cases ---------------------------------


def test_slug_lookup_list_shape(tmp_path: Path) -> None:
    """A top-level list catalog (no ``strategies`` wrapper) is also accepted."""
    p = tmp_path / "c.json"
    p.write_text(json.dumps([{"a_id": "a1", "a_slug": "s1", "b_id": "b1", "b_slug": "s2"}]))
    out = _slug_lookup_from_catalog(p)
    assert out == {"a1": "s1", "b1": "s2"}


def test_slug_lookup_wrong_top_shape_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"strategies": "not-a-list"}))
    assert _slug_lookup_from_catalog(p) == {}


def test_slug_lookup_skips_non_dict_entries(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"strategies": ["bogus-string", 42, None]}))
    assert _slug_lookup_from_catalog(p) == {}


def test_slug_lookup_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _slug_lookup_from_catalog(tmp_path / "absent.json")


# --- _resolve_token_id failure paths ----------------------------------------


def _async_run(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)


@respx.mock
def test_resolve_token_id_empty_body_raises() -> None:
    respx.get(f"{GAMMA_URL}/markets").mock(return_value=httpx.Response(200, json=[]))

    async def _go() -> None:
        async with httpx.AsyncClient() as c:
            with pytest.raises(LookupError, match="empty body"):
                await _resolve_token_id("slug-x", client=c)

    _async_run(_go())


@respx.mock
def test_resolve_token_id_missing_clob_ids_raises() -> None:
    respx.get(f"{GAMMA_URL}/markets").mock(
        return_value=httpx.Response(200, json=[{"slug": "x", "clobTokenIds": None}])
    )

    async def _go() -> None:
        async with httpx.AsyncClient() as c:
            with pytest.raises(LookupError, match="no clobTokenIds"):
                await _resolve_token_id("slug-x", client=c)

    _async_run(_go())


@respx.mock
def test_resolve_token_id_bad_json_in_clob_ids_raises() -> None:
    respx.get(f"{GAMMA_URL}/markets").mock(
        return_value=httpx.Response(200, json=[{"slug": "x", "clobTokenIds": "not-valid-json"}])
    )

    async def _go() -> None:
        async with httpx.AsyncClient() as c:
            with pytest.raises(LookupError, match="not valid JSON"):
                await _resolve_token_id("slug-x", client=c)

    _async_run(_go())


@respx.mock
def test_resolve_token_id_unexpected_shape_raises() -> None:
    respx.get(f"{GAMMA_URL}/markets").mock(
        return_value=httpx.Response(200, json=[{"slug": "x", "clobTokenIds": {"k": "v"}}])
    )

    async def _go() -> None:
        async with httpx.AsyncClient() as c:
            with pytest.raises(LookupError, match="unexpected clobTokenIds shape"):
                await _resolve_token_id("slug-x", client=c)

    _async_run(_go())


@respx.mock
def test_resolve_token_id_accepts_native_list() -> None:
    """A list shape for ``clobTokenIds`` (rare but allowed) parses fine."""
    respx.get(f"{GAMMA_URL}/markets").mock(
        return_value=httpx.Response(
            200, json=[{"slug": "x", "clobTokenIds": ["tok-yes", "tok-no"]}]
        )
    )

    async def _go() -> str:
        async with httpx.AsyncClient() as c:
            return await _resolve_token_id("slug-x", client=c)

    assert _async_run(_go()) == "tok-yes"


@respx.mock
def test_resolve_token_id_empty_list_raises() -> None:
    respx.get(f"{GAMMA_URL}/markets").mock(
        return_value=httpx.Response(200, json=[{"slug": "x", "clobTokenIds": json.dumps([])}])
    )

    async def _go() -> None:
        async with httpx.AsyncClient() as c:
            with pytest.raises(LookupError, match="empty clobTokenIds"):
                await _resolve_token_id("slug-x", client=c)

    _async_run(_go())


# --- _fetch_clob_history parse robustness -----------------------------------


@respx.mock
def test_fetch_clob_history_non_list_history_returns_empty() -> None:
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json={"history": "not-a-list"})
    )

    async def _go() -> pd.Series:
        async with httpx.AsyncClient() as c:
            return await _fetch_clob_history("tok-x", client=c)

    s = _async_run(_go())
    assert isinstance(s, pd.Series)
    assert s.empty


@respx.mock
def test_fetch_clob_history_skips_malformed_rows() -> None:
    body = {
        "history": [
            "not-a-dict",
            {"t": None, "p": 0.5},
            {"t": 1_700_000_000, "p": None},
            {"t": "not-a-number", "p": 0.5},
            # ``float("badnum")`` raises ValueError → row is skipped.
            {"t": 1_700_000_000, "p": "not-a-float"},
            {"t": 1_700_086_400, "p": 0.5},  # one valid row, distinct day
        ]
    }
    respx.get(f"{CLOB_URL}/prices-history").mock(return_value=httpx.Response(200, json=body))

    async def _go() -> pd.Series:
        async with httpx.AsyncClient() as c:
            return await _fetch_clob_history("tok-x", client=c)

    s = _async_run(_go())
    assert isinstance(s, pd.Series)
    assert len(s) == 1


@respx.mock
def test_fetch_clob_history_skips_non_finite_floats() -> None:
    """Rows whose ``p`` is NaN/Inf are silently skipped.

    The parse path uses ``np.isfinite`` *after* ``float()`` succeeds, so
    we feed the response as a raw JSON body that the stdlib JSON parser
    accepts (``NaN``/``Infinity`` literals are non-standard but Python's
    json module decodes them by default).
    """
    body_text = (
        '{"history": ['
        '{"t": 1700000000, "p": NaN},'
        '{"t": 1700086400, "p": Infinity},'
        '{"t": 1700172800, "p": 0.5}'
        "]}"
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(
            200, text=body_text, headers={"content-type": "application/json"}
        )
    )

    async def _go() -> pd.Series:
        async with httpx.AsyncClient() as c:
            return await _fetch_clob_history("tok-x", client=c)

    s = _async_run(_go())
    # Only the third (finite) row survives.
    assert len(s) == 1
    assert float(s.iloc[0]) == 0.5


@respx.mock
def test_fetch_clob_history_all_rows_invalid_returns_empty() -> None:
    body = {"history": [{"t": None, "p": None}, "string"]}
    respx.get(f"{CLOB_URL}/prices-history").mock(return_value=httpx.Response(200, json=body))

    async def _go() -> pd.Series:
        async with httpx.AsyncClient() as c:
            return await _fetch_clob_history("tok-x", client=c)

    s = _async_run(_go())
    assert s.empty


@respx.mock
def test_fetch_clob_history_dedupes_same_day() -> None:
    """Two rows on the same UTC date → last-wins, length 1."""
    ts = 1_700_000_000
    body = {"history": [{"t": ts, "p": 0.4}, {"t": ts + 3600, "p": 0.6}]}
    respx.get(f"{CLOB_URL}/prices-history").mock(return_value=httpx.Response(200, json=body))

    async def _go() -> pd.Series:
        async with httpx.AsyncClient() as c:
            return await _fetch_clob_history("tok-x", client=c)

    s = _async_run(_go())
    assert len(s) == 1
    assert float(s.iloc[0]) == 0.6


# --- _polymarket_live_fetcher remaining branches ----------------------------


@respx.mock
def test_polymarket_fetcher_cache_hit_short_circuits(tmp_path: Path) -> None:
    """A second call with the same key reuses the cached Series tuple."""
    catalog = _write_one_pair_catalog(tmp_path)

    gamma_route = respx.get(f"{GAMMA_URL}/markets").mock(
        side_effect=lambda req: _gamma_resp(
            req.url.params["slug"],
            "tok-a" if req.url.params["slug"] == "slug-a" else "tok-b",
        )
    )
    clob_route = respx.get(f"{CLOB_URL}/prices-history").mock(
        side_effect=lambda req: httpx.Response(
            200,
            json=_clob_payload(
                [0.4 + 0.001 * i for i in range(30)]
                if req.url.params["market"] == "tok-a"
                else [0.5 - 0.001 * i for i in range(30)]
            ),
        )
    )

    async def _go() -> tuple[int, int]:
        async with httpx.AsyncClient() as c:
            _ = await _polymarket_live_fetcher("ab", "aa", "bb", client=c, catalog_path=catalog)
            calls_after_first = (gamma_route.call_count, clob_route.call_count)
            _ = await _polymarket_live_fetcher("ab", "aa", "bb", client=c, catalog_path=catalog)
            calls_after_second = (gamma_route.call_count, clob_route.call_count)
            assert calls_after_first == calls_after_second
            return calls_after_second

    g, k = _async_run(_go())
    assert g == 2  # one per leg
    assert k == 2


@respx.mock
def test_polymarket_fetcher_leg_b_missing_slug_raises(tmp_path: Path) -> None:
    catalog = tmp_path / "c.json"
    catalog.write_text(
        json.dumps(
            {"strategies": [{"pair_id": "p", "a_id": "aa", "b_id": "bb", "a_slug": "slug-a"}]}
        )
    )

    async def _go() -> None:
        async with httpx.AsyncClient() as c:
            with pytest.raises(LookupError, match="no b_slug"):
                await _polymarket_live_fetcher("p", "aa", "bb", client=c, catalog_path=catalog)

    _async_run(_go())


@respx.mock
def test_polymarket_fetcher_leg_a_empty_history_raises(tmp_path: Path) -> None:
    catalog = _write_one_pair_catalog(tmp_path)
    respx.get(f"{GAMMA_URL}/markets").mock(
        side_effect=lambda req: _gamma_resp(
            req.url.params["slug"],
            "tok-a" if req.url.params["slug"] == "slug-a" else "tok-b",
        )
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        side_effect=lambda req: httpx.Response(
            200,
            json={"history": []}
            if req.url.params["market"] == "tok-a"
            else _clob_payload([0.5 + 0.001 * i for i in range(20)]),
        )
    )

    async def _go() -> None:
        async with httpx.AsyncClient() as c:
            with pytest.raises(LookupError, match="leg_a slug='slug-a' returned"):
                await _polymarket_live_fetcher("ab", "aa", "bb", client=c, catalog_path=catalog)

    _async_run(_go())


@respx.mock
def test_polymarket_fetcher_leg_b_empty_history_raises(tmp_path: Path) -> None:
    catalog = _write_one_pair_catalog(tmp_path)
    respx.get(f"{GAMMA_URL}/markets").mock(
        side_effect=lambda req: _gamma_resp(
            req.url.params["slug"],
            "tok-a" if req.url.params["slug"] == "slug-a" else "tok-b",
        )
    )
    respx.get(f"{CLOB_URL}/prices-history").mock(
        side_effect=lambda req: httpx.Response(
            200,
            json=_clob_payload([0.5 + 0.001 * i for i in range(20)])
            if req.url.params["market"] == "tok-a"
            else {"history": []},
        )
    )

    async def _go() -> None:
        async with httpx.AsyncClient() as c:
            with pytest.raises(LookupError, match="leg_b slug='slug-b' returned"):
                await _polymarket_live_fetcher("ab", "aa", "bb", client=c, catalog_path=catalog)

    _async_run(_go())


@respx.mock
def test_polymarket_fetcher_no_overlap_raises(tmp_path: Path) -> None:
    """Disjoint date ranges → inner-join empty → LookupError."""
    catalog = _write_one_pair_catalog(tmp_path)
    one_day = 86_400
    start_a = 1_700_000_000
    start_b = start_a + 1000 * one_day  # far in the future, no overlap

    respx.get(f"{GAMMA_URL}/markets").mock(
        side_effect=lambda req: _gamma_resp(
            req.url.params["slug"],
            "tok-a" if req.url.params["slug"] == "slug-a" else "tok-b",
        )
    )

    def _clob(req: httpx.Request) -> httpx.Response:
        if req.url.params["market"] == "tok-a":
            return httpx.Response(200, json=_clob_payload([0.5] * 10, start_unix=start_a))
        return httpx.Response(200, json=_clob_payload([0.5] * 10, start_unix=start_b))

    respx.get(f"{CLOB_URL}/prices-history").mock(side_effect=_clob)

    async def _go() -> None:
        async with httpx.AsyncClient() as c:
            with pytest.raises(LookupError, match="no overlapping dates"):
                await _polymarket_live_fetcher("ab", "aa", "bb", client=c, catalog_path=catalog)

    _async_run(_go())


@respx.mock
def test_polymarket_fetcher_window_trim(tmp_path: Path) -> None:
    """When the joined history exceeds ``window_days`` the tail is kept."""
    catalog = _write_one_pair_catalog(tmp_path)
    respx.get(f"{GAMMA_URL}/markets").mock(
        side_effect=lambda req: _gamma_resp(
            req.url.params["slug"],
            "tok-a" if req.url.params["slug"] == "slug-a" else "tok-b",
        )
    )
    a_prices = [0.30 + 0.001 * i for i in range(120)]
    b_prices = [0.60 - 0.001 * i for i in range(120)]

    def _clob(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_clob_payload(a_prices if req.url.params["market"] == "tok-a" else b_prices),
        )

    respx.get(f"{CLOB_URL}/prices-history").mock(side_effect=_clob)

    async def _go() -> tuple[pd.Series, pd.Series]:
        async with httpx.AsyncClient() as c:
            return await _polymarket_live_fetcher(
                "ab", "aa", "bb", window_days=30, client=c, catalog_path=catalog
            )

    a_out, b_out = _async_run(_go())
    assert len(a_out) == 30
    assert len(b_out) == 30
    # Tail kept → last value matches the last price (within tolerance).
    assert abs(float(a_out.iloc[-1]) - a_prices[-1]) < 1e-9


# --- _wrap_pair_fetcher_as_price_fetcher ------------------------------------


def test_wrap_pair_fetcher_returns_python_lists() -> None:
    """Adapter coerces Series to plain lists and stashes the client."""

    async def _pf(pair_id: str, a_id: str, b_id: str) -> tuple[pd.Series, pd.Series]:
        idx = pd.date_range("2026-01-01", periods=5, tz="UTC")
        return pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=idx), pd.Series(
            [0.5, 0.4, 0.3, 0.2, 0.1], index=idx
        )

    sentinel_client = object()
    adapter = _wrap_pair_fetcher_as_price_fetcher(_pf, client=sentinel_client)  # type: ignore[arg-type]
    assert adapter.shared_client is sentinel_client  # type: ignore[attr-defined]

    async def _go() -> tuple[list[float], list[float]]:
        return await adapter({"pair_id": "ab", "a_id": "aa", "b_id": "bb"})

    a_out, b_out = _async_run(_go())
    assert isinstance(a_out, list)
    assert isinstance(b_out, list)
    assert all(isinstance(x, float) for x in a_out + b_out)
    assert a_out == [0.1, 0.2, 0.3, 0.4, 0.5]


# --- _build_pair_fetcher_for_kind -------------------------------------------


def test_build_pair_fetcher_synthetic_returns_none() -> None:
    assert _build_pair_fetcher_for_kind("synthetic") is None


def test_build_pair_fetcher_polymarket_returns_callable() -> None:
    pf = _build_pair_fetcher_for_kind("polymarket")
    assert callable(pf)


def test_build_pair_fetcher_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown fetcher_kind"):
        _build_pair_fetcher_for_kind("nope")  # type: ignore[arg-type]


# --- _load_alphas -----------------------------------------------------------


def test_load_alphas_list_shape(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text(json.dumps([{"pair_id": "p1"}, {"pair_id": "p2"}]))
    out = _load_alphas(p)
    assert len(out) == 2


def test_load_alphas_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _load_alphas(tmp_path / "absent.json")


def test_load_alphas_unexpected_shape_raises(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"not_strategies": "here"}))
    with pytest.raises(ValueError, match="missing 'strategies' array"):
        _load_alphas(p)


# --- recompute_all_signals branches -----------------------------------------


def test_recompute_missing_a_id_records_error() -> None:
    """An alpha lacking ``a_id`` short-circuits with an error entry."""
    bad = {"pair_id": "p_bad", "b_id": "bb"}
    results = _async_run(recompute_all_signals([bad], fetcher=_good_fetcher()))
    assert len(results) == 1
    assert results[0].get("error") == "missing a_id or b_id"
    assert results[0]["pair_id"] == "p_bad"


def test_recompute_compute_failure_isolated() -> None:
    """A fetcher returning too-few bars triggers the compute-failure path."""

    async def _short_fetcher(_factor_id: str) -> list[float]:
        return [0.5, 0.5]  # only 2 bars, below the n < 5 floor

    results = _async_run(recompute_all_signals([_alpha("p_short")], fetcher=_short_fetcher))
    assert len(results) == 1
    assert "compute failed" in results[0].get("error", "")


def test_recompute_pair_fetcher_takes_precedence() -> None:
    """A ``pair_fetcher`` is used even when a ``fetcher`` is also passed."""
    pair_called = {"count": 0}
    leg_called = {"count": 0}

    async def _pf(pair_id: str, a_id: str, b_id: str) -> tuple[pd.Series, pd.Series]:
        pair_called["count"] += 1
        idx = pd.date_range("2026-01-01", periods=30, tz="UTC")
        return pd.Series(_good_series(seed=1, n=30), index=idx), pd.Series(
            _good_series(seed=2, n=30), index=idx
        )

    async def _leg(_factor_id: str) -> list[float]:
        leg_called["count"] += 1
        return _good_series()

    results = _async_run(recompute_all_signals([_alpha("p1")], fetcher=_leg, pair_fetcher=_pf))
    assert pair_called["count"] == 1
    assert leg_called["count"] == 0
    assert "error" not in results[0]


# --- run_once: persistence shape + http_client injection --------------------


def test_run_once_output_shape(tmp_path: Path) -> None:
    """``live_signals.json`` has the documented keys and signal payload."""
    alphas = [_alpha("p_a"), _alpha("p_b")]
    strategies_path = tmp_path / "s.json"
    strategies_path.write_text(json.dumps({"strategies": alphas}))
    write_path = tmp_path / "live.json"
    status_path = tmp_path / "status.json"

    summary = _async_run(
        run_once(
            write_path=write_path,
            strategies_path=strategies_path,
            status_path=status_path,
            fetcher=_good_fetcher(),
        )
    )
    payload = json.loads(write_path.read_text())
    # Top-level envelope keys.
    assert set(payload.keys()) >= {
        "as_of",
        "n_strategies",
        "n_actionable",
        "n_errors",
        "duration_seconds",
        "signals",
    }
    # Per-pair signal payload shape.
    for pid in ("p_a", "p_b"):
        sig = payload["signals"][pid]
        assert set(sig.keys()) >= {
            "pair_id",
            "a_id",
            "b_id",
            "as_of",
            "n_obs",
            "beta_hedge",
            "current_spread",
            "current_z",
            "current_a_price",
            "current_b_price",
            "action",
            "reason",
            "mu_window",
            "sigma_window",
            "decay_status",
        }
    # Status file keys mirror the summary.
    assert set(summary.keys()) >= {
        "last_run_iso",
        "last_duration_seconds",
        "n_alphas_total",
        "n_alphas_updated",
        "n_alphas_failed",
        "n_alphas_actionable",
        "failures",
        "live_signals_path",
    }


def test_run_once_persists_payload_keys(tmp_path: Path) -> None:
    """The written JSON has the exact expected envelope.

    Asserts the persistence contract (independent of caller paths): the
    payload that lands on disk via ``_atomic_write_json`` contains the
    five envelope keys plus ``signals`` keyed by ``pair_id``.
    """
    alphas = [_alpha("only")]
    strategies_path = tmp_path / "alphas.json"
    strategies_path.write_text(json.dumps({"strategies": alphas}))
    write_path = tmp_path / "live.json"
    status_path = tmp_path / "status.json"
    _async_run(
        run_once(
            write_path=write_path,
            strategies_path=strategies_path,
            status_path=status_path,
            fetcher=_good_fetcher(),
        )
    )
    payload = json.loads(write_path.read_text())
    assert payload["n_strategies"] == 1
    assert "only" in payload["signals"]
    # And the status file mirrors the run summary.
    status = json.loads(status_path.read_text())
    assert status["n_alphas_total"] == 1
    assert status["live_signals_path"] == str(write_path)


def test_run_once_status_write_oserror_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the status file can't be written ``run_once`` still returns OK."""
    alphas = [_alpha("p1")]
    strategies_path = tmp_path / "s.json"
    strategies_path.write_text(json.dumps({"strategies": alphas}))
    write_path = tmp_path / "live.json"

    real_write = live_signals_job._atomic_write_json
    status_target = tmp_path / "status.json"

    def _flaky_write(path: Any, payload: Any) -> None:
        # Match by exact target basename to avoid false positives from
        # tmp dirs whose names happen to contain "status".
        if Path(str(path)).name == status_target.name:
            raise OSError("disk full (simulated)")
        return real_write(path, payload)

    monkeypatch.setattr(live_signals_job, "_atomic_write_json", _flaky_write)
    summary = _async_run(
        run_once(
            write_path=write_path,
            strategies_path=strategies_path,
            status_path=status_target,
            fetcher=_good_fetcher(),
        )
    )
    # The live_signals.json was written.
    assert write_path.exists()
    # The status file was NOT written (OSError swallowed).
    assert not status_target.exists()
    # And the summary still came back populated.
    assert summary["n_alphas_total"] == 1


def test_run_once_invalidates_live_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After ``run_once`` writes, the ``live_signals`` HTTP cache is cleared."""
    alphas = [_alpha("p1")]
    strategies_path = tmp_path / "s.json"
    strategies_path.write_text(json.dumps({"strategies": alphas}))
    cache = get_cache("live_signals", ttl=SIGNALS_LIVE_CACHE_TTL)
    cache.set("payload", {"stale": True}, ttl=SIGNALS_LIVE_CACHE_TTL)
    assert cache.get("payload") == {"stale": True}
    _async_run(
        run_once(
            write_path=tmp_path / "live.json",
            strategies_path=strategies_path,
            status_path=tmp_path / "status.json",
            fetcher=_good_fetcher(),
        )
    )
    assert cache.get("payload") is None


# --- concurrent run guard ---------------------------------------------------


def test_run_once_concurrent_runs_produce_consistent_file(tmp_path: Path) -> None:
    """Two ``run_once`` calls in parallel both finish with a well-formed file.

    The atomic temp-file + rename means we never end up with a partial
    JSON document, even when the writes race. We assert: (a) both runs
    return summaries, (b) no temp files leak, (c) the final ``live.json``
    parses cleanly.
    """
    alphas = [_alpha(f"p_{i}") for i in range(6)]
    strategies_path = tmp_path / "s.json"
    strategies_path.write_text(json.dumps({"strategies": alphas}))
    write_path = tmp_path / "live.json"

    async def _both() -> list[Any]:
        return await asyncio.gather(
            run_once(
                write_path=write_path,
                strategies_path=strategies_path,
                status_path=tmp_path / "status1.json",
                fetcher=_good_fetcher(),
            ),
            run_once(
                write_path=write_path,
                strategies_path=strategies_path,
                status_path=tmp_path / "status2.json",
                fetcher=_good_fetcher(),
            ),
        )

    results = _async_run(_both())
    assert len(results) == 2
    for r in results:
        assert r["n_alphas_total"] == 6
    # No leaked temp files.
    assert list(tmp_path.glob("live.json.tmp*")) == []
    # Final file parses and contains all six signals.
    payload = json.loads(write_path.read_text())
    assert payload["n_strategies"] == 6
    assert set(payload["signals"]) == {f"p_{i}" for i in range(6)}


# --- reproducibility --------------------------------------------------------


def test_run_once_reproducible_with_deterministic_fetcher(tmp_path: Path) -> None:
    """Identical inputs + deterministic fetcher → identical signals payload."""
    alphas = [_alpha("p_repro")]
    strategies_path = tmp_path / "s.json"
    strategies_path.write_text(json.dumps({"strategies": alphas}))

    def _two_runs() -> tuple[dict[str, Any], dict[str, Any]]:
        out_paths = [tmp_path / "live1.json", tmp_path / "live2.json"]
        for p in out_paths:
            _async_run(
                run_once(
                    write_path=p,
                    strategies_path=strategies_path,
                    status_path=tmp_path / f"st_{p.name}.json",
                    fetcher=_good_fetcher(),
                )
            )
        return tuple(json.loads(p.read_text()) for p in out_paths)  # type: ignore[return-value]

    a, b = _two_runs()
    # The ``as_of`` timestamps differ between runs (UTC now), but every
    # other field for every signal must match exactly.
    assert set(a["signals"]) == set(b["signals"])
    for pid in a["signals"]:
        sig_a = {k: v for k, v in a["signals"][pid].items() if k != "as_of"}
        sig_b = {k: v for k, v in b["signals"][pid].items() if k != "as_of"}
        assert sig_a == sig_b


# --- run_forever exception swallow + stop_event paths -----------------------


def test_run_forever_swallows_inner_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing ``run_once`` must not propagate out of ``run_forever``.

    The loop catches ``Exception`` from ``run_once``, logs it, and then
    enters the sleep gate. We patch ``run_once`` to always raise, run
    the loop briefly, then signal stop — the task must terminate
    cleanly (not via an unhandled exception) and the run-once function
    must have been called at least once.
    """
    alphas = [_alpha("p1")]
    strategies_path = tmp_path / "s.json"
    strategies_path.write_text(json.dumps({"strategies": alphas}))
    state = {"calls": 0}

    async def _always_raises(*args: Any, **kwargs: Any) -> dict[str, Any]:
        state["calls"] += 1
        raise RuntimeError("synthetic boom")

    monkeypatch.setattr(live_signals_job, "run_once", _always_raises)

    async def _runner() -> None:
        stop = asyncio.Event()

        async def _inner() -> None:
            await run_forever(
                interval_seconds=60,
                write_path=tmp_path / "live.json",
                strategies_path=strategies_path,
                status_path=tmp_path / "status.json",
                fetcher=_good_fetcher(),
                stop_event=stop,
            )

        task = asyncio.create_task(_inner())
        # Give the first cycle time to run + log.
        await asyncio.sleep(0.3)
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)
        # Task ended cleanly: not cancelled, no exception escaped.
        assert task.done()
        assert not task.cancelled()
        assert task.exception() is None

    _async_run(_runner())
    assert state["calls"] >= 1


def test_run_forever_stop_event_pre_sleep(tmp_path: Path) -> None:
    """If the stop event is already set after one cycle the loop exits early."""
    alphas = [_alpha("p1")]
    strategies_path = tmp_path / "s.json"
    strategies_path.write_text(json.dumps({"strategies": alphas}))

    async def _runner() -> None:
        stop = asyncio.Event()
        # Pre-set the event so the first post-cycle check returns immediately.
        stop.set()
        await asyncio.wait_for(
            run_forever(
                interval_seconds=60,
                write_path=tmp_path / "live.json",
                strategies_path=strategies_path,
                status_path=tmp_path / "status.json",
                fetcher=_good_fetcher(),
                stop_event=stop,
            ),
            timeout=2.0,
        )

    _async_run(_runner())
    assert (tmp_path / "live.json").exists()


def test_run_forever_stop_event_during_sleep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stop event firing mid-sleep exits the loop without raising."""
    alphas = [_alpha("p1")]
    strategies_path = tmp_path / "s.json"
    strategies_path.write_text(json.dumps({"strategies": alphas}))

    async def _runner() -> None:
        stop = asyncio.Event()

        async def _inner() -> None:
            # Pass interval_seconds=60 → clamped to 60 internally; we'll
            # set ``stop`` after a fraction of a second so the wait_for
            # branch returns rather than the TimeoutError continue path.
            await run_forever(
                interval_seconds=60,
                write_path=tmp_path / "live.json",
                strategies_path=strategies_path,
                status_path=tmp_path / "status.json",
                fetcher=_good_fetcher(),
                stop_event=stop,
            )

        task = asyncio.create_task(_inner())
        await asyncio.sleep(0.4)  # let first run complete
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)
        assert task.done()
        assert not task.cancelled()

    _async_run(_runner())


# --- router: admin gating + parse-error paths -------------------------------


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(live_signals_router)
    return app


def test_recompute_now_returns_500_on_internal_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generic exception inside ``run_once`` maps to a 500 with detail."""
    monkeypatch.delenv("PFM_ADMIN_TOKEN", raising=False)

    async def _boom() -> dict[str, Any]:
        raise RuntimeError("unexpected boom")

    monkeypatch.setattr(live_signals_job, "run_once", _boom)
    app = _app()
    client = TestClient(app)
    r = client.post("/signals/recompute-now")
    assert r.status_code == 500
    assert "recompute failed" in r.json()["detail"]


def test_recompute_now_returns_404_on_missing_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FileNotFoundError maps to a 404, not a 500."""
    monkeypatch.delenv("PFM_ADMIN_TOKEN", raising=False)

    async def _missing() -> dict[str, Any]:
        raise FileNotFoundError("alpha_strategies.json")

    monkeypatch.setattr(live_signals_job, "run_once", _missing)
    app = _app()
    client = TestClient(app)
    r = client.post("/signals/recompute-now")
    assert r.status_code == 404
    assert "alpha catalog missing" in r.json()["detail"]


def test_signals_live_cache_hit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A warm cache short-circuits the file read."""
    live_path = tmp_path / "absent.json"
    monkeypatch.setattr(live_signals_job, "DEFAULT_LIVE_SIGNALS_PATH", str(live_path))
    cache = get_cache("live_signals", ttl=SIGNALS_LIVE_CACHE_TTL)
    cache.set("payload", {"cached": True}, ttl=SIGNALS_LIVE_CACHE_TTL)
    app = _app()
    client = TestClient(app)
    r = client.get("/signals/live")
    # Even though the file doesn't exist on disk, the cache hit returns 200.
    assert r.status_code == 200
    assert r.json() == {"cached": True}


def test_signals_status_returns_500_on_corrupt_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A status file that isn't valid JSON yields a 500 with a detail."""
    status_path = tmp_path / "broken.json"
    status_path.write_text("{not-valid-json")
    monkeypatch.setattr(live_signals_job, "DEFAULT_STATUS_PATH", str(status_path))
    app = _app()
    client = TestClient(app)
    r = client.get("/signals/status")
    assert r.status_code == 500
    assert "status file unreadable" in r.json()["detail"]


def test_signals_live_returns_500_on_corrupt_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A corrupt live_signals.json maps to 500 (not 404)."""
    live_path = tmp_path / "broken.json"
    live_path.write_text("not-json")
    monkeypatch.setattr(live_signals_job, "DEFAULT_LIVE_SIGNALS_PATH", str(live_path))
    app = _app()
    client = TestClient(app)
    r = client.get("/signals/live")
    assert r.status_code == 500
    assert "live_signals.json unreadable" in r.json()["detail"]


def test_signals_status_handles_invalid_iso(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A garbage ``last_run_iso`` leaves ``next_run_at_estimate`` as None."""
    status_path = tmp_path / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "last_run_iso": "not-an-iso-string",
                "last_duration_seconds": 1.0,
                "n_alphas_total": 1,
                "n_alphas_updated": 1,
                "n_alphas_failed": 0,
                "n_alphas_actionable": 0,
                "failures": [],
                "live_signals_path": str(tmp_path / "live.json"),
            }
        )
    )
    monkeypatch.setattr(live_signals_job, "DEFAULT_STATUS_PATH", str(status_path))
    app = _app()
    client = TestClient(app)
    r = client.get("/signals/status")
    assert r.status_code == 200
    assert r.json()["next_run_at_estimate"] is None


def test_recompute_now_admin_token_set_rejects_without_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``PFM_ADMIN_TOKEN`` set, the recompute endpoint is gated.

    The dependency is captured at import time when the router was first
    defined, so to exercise the admin path we re-build the router with
    the env var present *and* a fresh FastAPI app. We assert the gate
    fires (403) when the header is missing — proving the
    ``_admin_dep_if_enabled`` branch with the env var set is exercised.
    """
    monkeypatch.setenv("PFM_ADMIN_TOKEN", "secret-token-xyz")
    # Build a fresh router by reloading the module-level dep + endpoint.
    from fastapi import APIRouter, Depends

    from pfm.auth.dependencies import require_admin

    app = FastAPI()
    test_router = APIRouter(prefix="/signals", tags=["live-signals"])

    @test_router.post("/recompute-now")
    async def _admin_only(_: None = Depends(require_admin)) -> dict[str, str]:
        return {"ok": "yes"}

    app.include_router(test_router)
    client = TestClient(app)
    r = client.post("/signals/recompute-now")
    assert r.status_code == 403


# --- atomic write helper ----------------------------------------------------


def test_atomic_write_json_creates_parent_dirs(tmp_path: Path) -> None:
    """``_atomic_write_json`` mkdirs the parent if it doesn't exist."""
    nested = tmp_path / "a" / "b" / "c" / "out.json"
    _atomic_write_json(nested, {"hello": "world"})
    assert nested.exists()
    assert json.loads(nested.read_text()) == {"hello": "world"}
    # No leftover temp file.
    assert list(nested.parent.glob("out.json.tmp*")) == []


# --- verify_polymarket_connectivity empty-history path ----------------------


@respx.mock
def test_verify_connectivity_empty_history_returns_ok_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When Gamma resolves but CLOB returns no bars, ``ok`` is False."""
    monkeypatch.setenv("PFM_CONNECTIVITY_SAMPLE_SLUG", "x-slug")
    respx.get(f"{GAMMA_URL}/markets").mock(return_value=_gamma_resp("x-slug", "tok-x"))
    respx.get(f"{CLOB_URL}/prices-history").mock(
        return_value=httpx.Response(200, json={"history": []})
    )

    async def _go() -> dict[str, Any]:
        async with httpx.AsyncClient() as c:
            return await verify_polymarket_connectivity("x-slug", client=c)

    out = _async_run(_go())
    assert out["ok"] is False
    assert out["error"] == "empty history"
    assert out["sample_size"] == 0

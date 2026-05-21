"""Tests for ``pfm.live_signals_job``.

Scope:

* ``recompute_all_signals`` returns one entry per alpha when the
  fetcher succeeds, with ``current_z`` / ``action`` / ``decay_status``
  populated.
* Failure isolation: a fetcher that raises for one alpha must NOT
  break the rest of the batch.
* ``run_once`` writes ``live_signals.json`` atomically — there's no
  partial-file window.
* ``run_forever`` exits cleanly when its task is cancelled.
* The HTTP router exposes ``/signals/recompute-now``, ``/signals/status``,
  and ``/signals/live`` and they return the expected payloads.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import live_signals_job
from pfm.cache_utils import reset_caches
from pfm.live_signals_job import (
    _compute_signal_for_alpha,
    _signal_from_z,
    recompute_all_signals,
    run_forever,
    run_once,
)
from pfm.live_signals_job import (
    router as live_signals_router,
)

# --- helpers ----------------------------------------------------------------


def _make_alpha(pair_id: str, *, beta: float = 1.0) -> dict[str, Any]:
    """Build a minimal alpha-strategy record matching the catalog shape."""
    a_id = f"{pair_id}_A"
    b_id = f"{pair_id}_B"
    return {
        "pair_id": pair_id,
        "a_id": a_id,
        "b_id": b_id,
        "beta_hedge": beta,
        "rule_window": 10,
        "rule_entry_z": 2.0,
        "rule_exit_z": 0.5,
        "rule_stop_z": 4.0,
    }


def _build_fetcher(
    series_by_id: dict[str, list[float]] | None = None,
    failures: set[str] | None = None,
) -> Callable[[str], Awaitable[list[float]]]:
    """Async fetcher that returns canned series and raises for ``failures``."""
    series_by_id = dict(series_by_id or {})
    failures = set(failures or set())

    async def _fetch(factor_id: str) -> list[float]:
        if factor_id in failures:
            raise RuntimeError(f"injected failure for {factor_id}")
        if factor_id in series_by_id:
            return list(series_by_id[factor_id])
        # Deterministic random walk fallback.
        rng = np.random.default_rng(abs(hash(factor_id)) % (2**31 - 1))
        steps = rng.normal(0.0, 0.02, size=60)
        x = np.cumsum(steps)
        prices = 1.0 / (1.0 + np.exp(-x))
        return [float(p) for p in prices]

    return _fetch


@pytest.fixture(autouse=True)
def _reset_live_signals_cache():
    """Clear the live_signals HTTP cache before each test."""
    reset_caches()
    yield
    reset_caches()


# --- pure helpers -----------------------------------------------------------


def test_signal_from_z_thresholds() -> None:
    assert _signal_from_z(2.5, 2.0, 0.5, 4.0)[0] == "OPEN_SHORT"
    assert _signal_from_z(-2.5, 2.0, 0.5, 4.0)[0] == "OPEN_LONG"
    assert _signal_from_z(0.1, 2.0, 0.5, 4.0)[0] == "CLOSE"
    assert _signal_from_z(1.0, 2.0, 0.5, 4.0)[0] == "HOLD"
    assert _signal_from_z(5.0, 2.0, 0.5, 4.0)[0] == "STOP_OUT"
    assert _signal_from_z(float("nan"), 2.0, 0.5, 4.0)[0] == "FLAT"


def test_signal_from_z_edge_trigger() -> None:
    """Crossing the entry threshold from below produces OPEN_SHORT."""
    action, _ = _signal_from_z(2.1, 2.0, 0.5, 4.0, prev_z=1.5)
    assert action == "OPEN_SHORT"


def test_compute_signal_for_alpha_basic() -> None:
    """A constructed price pair recovers a finite z-score and action."""
    rng = np.random.default_rng(0)
    n = 60
    a = (0.5 + 0.05 * np.cumsum(rng.normal(0, 0.05, n))).tolist()
    b = (0.5 + 0.04 * np.cumsum(rng.normal(0, 0.05, n))).tolist()
    alpha = _make_alpha("toy")
    out = _compute_signal_for_alpha(alpha, a, b, as_of_iso="2026-05-08T00:00:00+00:00")
    assert out["pair_id"] == "toy"
    assert out["n_obs"] == n
    assert out["current_z"] is not None
    assert out["action"] in {"OPEN_LONG", "OPEN_SHORT", "HOLD", "CLOSE", "STOP_OUT", "FLAT"}
    assert out["decay_status"] in {"ACTIVE", "QUIET", "STRESSED", "INSUFFICIENT_DATA", "UNKNOWN"}


def test_compute_signal_for_alpha_too_few_bars_raises() -> None:
    with pytest.raises(ValueError, match="too few overlapping bars"):
        _compute_signal_for_alpha(_make_alpha("x"), [0.5, 0.5], [0.5, 0.5], as_of_iso="now")


# --- recompute_all_signals --------------------------------------------------


def test_recompute_all_signals_happy_path() -> None:
    alphas = [_make_alpha(f"pair_{i}") for i in range(5)]
    results = asyncio.run(recompute_all_signals(alphas, fetcher=_build_fetcher()))
    assert len(results) == 5
    for r in results:
        assert "error" not in r, r
        assert r["current_z"] is not None
        assert "action" in r and "decay_status" in r


def test_recompute_all_signals_isolates_failures() -> None:
    """One bad fetcher must not torpedo the other four alphas."""
    alphas = [_make_alpha(f"pair_{i}") for i in range(5)]
    bad_id = alphas[2]["a_id"]
    fetcher = _build_fetcher(failures={bad_id})
    results = asyncio.run(recompute_all_signals(alphas, fetcher=fetcher))
    assert len(results) == 5
    failed = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]
    assert len(failed) == 1
    assert len(ok) == 4
    assert failed[0]["pair_id"] == "pair_2"


def test_recompute_concurrency_capped() -> None:
    """The semaphore must keep in-flight fetches at or below the cap."""
    state = {"in_flight": 0, "peak": 0}

    async def _runner() -> None:
        lock = asyncio.Lock()

        async def slow_fetcher(_factor_id: str) -> list[float]:
            async with lock:
                state["in_flight"] += 1
                state["peak"] = max(state["peak"], state["in_flight"])
            await asyncio.sleep(0.01)
            async with lock:
                state["in_flight"] -= 1
            return [0.5 + 0.001 * i for i in range(60)]

        alphas = [_make_alpha(f"p_{i}") for i in range(50)]
        await recompute_all_signals(alphas, fetcher=slow_fetcher, max_concurrency=4)

    asyncio.run(_runner())
    # The semaphore caps *alphas* in flight at 4. Each alpha fans out to
    # 2 leg fetches concurrently via asyncio.gather, so peak fetches are
    # bounded by ``2 * max_concurrency`` = 8.
    assert state["peak"] <= 8
    # And the cap must actually bite — without it 100 fetches would all
    # start concurrently and the peak would be ~100.
    assert state["peak"] < 50


# --- run_once + atomic write ------------------------------------------------


def test_run_once_writes_atomic(tmp_path: Path) -> None:
    """run_once writes a complete JSON document, never a partial blob."""
    alphas = [_make_alpha(f"pair_{i}") for i in range(3)]
    strategies_path = tmp_path / "alpha_strategies.json"
    strategies_path.write_text(json.dumps({"strategies": alphas}))
    out = tmp_path / "live_signals.json"
    status = tmp_path / "status.json"

    summary = asyncio.run(
        run_once(
            write_path=out,
            strategies_path=strategies_path,
            status_path=status,
            fetcher=_build_fetcher(),
        )
    )
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["n_strategies"] == 3
    assert set(payload["signals"].keys()) == {"pair_0", "pair_1", "pair_2"}
    # Atomicity: the temp file should already be renamed away.
    leftover = list(tmp_path.glob("live_signals.json.tmp*"))
    assert leftover == []
    # Status file is also written.
    assert status.exists()
    s = json.loads(status.read_text())
    assert s["n_alphas_total"] == 3
    assert summary["n_alphas_total"] == 3
    assert summary["n_alphas_failed"] == 0


def test_run_once_records_failures(tmp_path: Path) -> None:
    alphas = [_make_alpha(f"p_{i}") for i in range(3)]
    strategies_path = tmp_path / "s.json"
    strategies_path.write_text(json.dumps({"strategies": alphas}))
    bad = alphas[0]["a_id"]
    summary = asyncio.run(
        run_once(
            write_path=tmp_path / "live.json",
            strategies_path=strategies_path,
            status_path=tmp_path / "status.json",
            fetcher=_build_fetcher(failures={bad}),
        )
    )
    assert summary["n_alphas_failed"] == 1
    assert summary["n_alphas_updated"] == 2
    assert summary["failures"][0]["pair_id"] == "p_0"


# --- run_forever cancellation -----------------------------------------------


def test_run_forever_cancellable(tmp_path: Path) -> None:
    """The loop must shut down cleanly on task.cancel()."""
    alphas = [_make_alpha(f"p_{i}") for i in range(2)]
    strategies_path = tmp_path / "s.json"
    strategies_path.write_text(json.dumps({"strategies": alphas}))

    async def _runner() -> None:
        task = asyncio.create_task(
            run_forever(
                interval_seconds=60,  # min clamp; doesn't matter, we cancel fast
                write_path=tmp_path / "live.json",
                strategies_path=strategies_path,
                status_path=tmp_path / "status.json",
                fetcher=_build_fetcher(),
            )
        )
        # Let one full cycle run.
        await asyncio.sleep(0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_runner())
    # The first run should still have produced output before the cancel.
    assert (tmp_path / "live.json").exists()


def test_run_forever_stop_event(tmp_path: Path) -> None:
    """The optional stop_event provides a non-cancel exit path."""
    alphas = [_make_alpha("p_0")]
    strategies_path = tmp_path / "s.json"
    strategies_path.write_text(json.dumps({"strategies": alphas}))

    async def _runner() -> None:
        stop = asyncio.Event()

        async def _inner() -> None:
            await run_forever(
                interval_seconds=60,
                write_path=tmp_path / "live.json",
                strategies_path=strategies_path,
                status_path=tmp_path / "status.json",
                fetcher=_build_fetcher(),
                stop_event=stop,
            )

        task = asyncio.create_task(_inner())
        await asyncio.sleep(0.3)
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)
        assert task.done()
        assert not task.cancelled()

    asyncio.run(_runner())


# --- HTTP endpoints ---------------------------------------------------------


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(live_signals_router)
    return app


def test_endpoint_recompute_now(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """POST /signals/recompute-now triggers run_once and returns the summary."""
    monkeypatch.delenv("PFM_ADMIN_TOKEN", raising=False)
    alphas = [_make_alpha(f"p_{i}") for i in range(2)]
    strategies_path = tmp_path / "s.json"
    strategies_path.write_text(json.dumps({"strategies": alphas}))
    live_path = tmp_path / "live.json"
    status_path = tmp_path / "status.json"

    async def _fake_run_once() -> dict[str, Any]:
        return await run_once(
            write_path=live_path,
            strategies_path=strategies_path,
            status_path=status_path,
            fetcher=_build_fetcher(),
        )

    monkeypatch.setattr(live_signals_job, "run_once", _fake_run_once)
    app = _make_app()
    client = TestClient(app)
    r = client.post("/signals/recompute-now")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_alphas_total"] == 2
    assert body["n_alphas_failed"] == 0
    assert live_path.exists()


def test_endpoint_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """GET /signals/status reflects the most recent run on disk."""
    status_path = tmp_path / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "last_run_iso": "2026-05-08T12:00:00+00:00",
                "last_duration_seconds": 4.2,
                "n_alphas_total": 88,
                "n_alphas_updated": 85,
                "n_alphas_failed": 3,
                "n_alphas_actionable": 7,
                "failures": [{"pair_id": "x", "error": "boom"}],
                "live_signals_path": str(tmp_path / "live.json"),
            }
        )
    )
    monkeypatch.setattr(live_signals_job, "DEFAULT_STATUS_PATH", str(status_path))
    monkeypatch.setenv("PFM_LIVE_SIGNALS_INTERVAL_S", "900")
    app = _make_app()
    client = TestClient(app)
    r = client.get("/signals/status")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_alphas_total"] == 88
    assert body["n_alphas_failed"] == 3
    assert body["next_run_at_estimate"] is not None
    assert body["failures"] == [{"pair_id": "x", "error": "boom"}]


def test_endpoint_status_no_runs_yet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(live_signals_job, "DEFAULT_STATUS_PATH", str(tmp_path / "missing.json"))
    app = _make_app()
    client = TestClient(app)
    r = client.get("/signals/status")
    assert r.status_code == 200
    body = r.json()
    assert body["last_run_iso"] is None


def test_endpoint_live(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """GET /signals/live returns the contents of live_signals.json."""
    live_path = tmp_path / "live.json"
    payload = {
        "as_of": "2026-05-08T12:00:00+00:00",
        "n_strategies": 2,
        "n_actionable": 0,
        "n_errors": 0,
        "signals": {
            "p_0": {"pair_id": "p_0", "action": "HOLD", "current_z": 0.5},
            "p_1": {"pair_id": "p_1", "action": "CLOSE", "current_z": 0.0},
        },
    }
    live_path.write_text(json.dumps(payload))
    monkeypatch.setattr(live_signals_job, "DEFAULT_LIVE_SIGNALS_PATH", str(live_path))
    app = _make_app()
    client = TestClient(app)
    r = client.get("/signals/live")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_strategies"] == 2
    assert "p_0" in body["signals"]


def test_endpoint_live_missing_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        live_signals_job, "DEFAULT_LIVE_SIGNALS_PATH", str(tmp_path / "absent.json")
    )
    app = _make_app()
    client = TestClient(app)
    r = client.get("/signals/live")
    assert r.status_code == 404

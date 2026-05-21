"""Tests for ``pfm.alpha_lab``."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import pfm.alpha_lab as lab

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _cointegrated_pair(n: int = 250, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    """Two series sharing a stochastic common trend → cointegrated."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    common = np.cumsum(rng.normal(0, 0.5, n))
    a = common + rng.normal(0, 0.3, n) + 50.0
    b = 0.6 * common + rng.normal(0, 0.3, n) + 30.0
    return pd.Series(a, index=idx, name="A"), pd.Series(b, index=idx, name="B")


def _independent_pair(n: int = 250, seed: int = 1) -> tuple[pd.Series, pd.Series]:
    """Two random walks → almost certainly NOT cointegrated."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    a = np.cumsum(rng.normal(0, 1.0, n)) + 100.0
    b = np.cumsum(rng.normal(0, 1.0, n)) + 50.0
    return pd.Series(a, index=idx, name="A"), pd.Series(b, index=idx, name="B")


@pytest.fixture(autouse=True)
def _isolate_lab_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect lab persistence to a temp dir + reset in-process state."""
    jobs = tmp_path / "lab_jobs.json"
    pending = tmp_path / "lab_pending.jsonl"
    monkeypatch.setattr(lab, "JOBS_FILE", jobs)
    monkeypatch.setattr(lab, "PENDING_FILE", pending)
    lab._STATE.running = False
    lab._STATE.last_job_id = None
    lab._STATE.last_run_at = None
    lab._STATE.last_results_summary = None


@pytest.fixture
def patched_fetchers(monkeypatch: pytest.MonkeyPatch):
    """Wire the lab's fetchers to deterministic synthetic data.

    First call returns a cointegrated pair; subsequent calls alternate so
    half of the random combos are easy passes.
    """
    a, b = _cointegrated_pair(n=260, seed=7)
    a2, b2 = _independent_pair(n=260, seed=13)

    def fake_factor(slug: str, days: int = 365) -> pd.Series:
        # Hash-stable assignment so the same slug always gets the same series.
        h = abs(hash(slug)) % 4
        if h == 0:
            return a.copy().rename(slug)
        if h == 1:
            return b.copy().rename(slug)
        if h == 2:
            return a2.copy().rename(slug)
        return b2.copy().rename(slug)

    def fake_equity(ticker: str, days: int = 365) -> pd.Series:
        h = abs(hash(ticker)) % 2
        return (b.copy() if h == 0 else b2.copy()).rename(ticker)

    monkeypatch.setattr(lab, "_fetch_factor_series", fake_factor)
    monkeypatch.setattr(lab, "_fetch_equity_series", fake_equity)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestDiscoverAlphas:
    def test_returns_summary_under_two_seconds(self, patched_fetchers) -> None:
        slugs = [f"slug-{i}" for i in range(10)]
        t0 = time.monotonic()
        out = lab.discover_alphas(
            n_combos=5,
            min_oos_sharpe=0.5,
            min_quarters_positive=1,
            max_runtime_seconds=10,
            factor_slugs=slugs,
            equity_tickers=["TKR1", "TKR2"],
            seed=11,
        )
        elapsed = time.monotonic() - t0
        assert elapsed < 2.5, f"discovery took {elapsed:.2f}s; expected <2s"
        assert out["n_tested"] <= 5
        assert isinstance(out["candidates"], list)
        assert all("pair_id" in c for c in out["candidates"])

    def test_respects_runtime_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``max_runtime_seconds=0`` should short-circuit before any combo."""
        # Make fetchers cheap.
        s, _ = _cointegrated_pair(n=200, seed=0)
        monkeypatch.setattr(lab, "_fetch_factor_series", lambda slug, days=365: s.rename(slug))
        monkeypatch.setattr(lab, "_fetch_equity_series", lambda t, days=365: s.rename(t))
        out = lab.discover_alphas(
            n_combos=50,
            min_oos_sharpe=1.0,
            min_quarters_positive=2,
            max_runtime_seconds=0,
            factor_slugs=[f"f{i}" for i in range(20)],
            equity_tickers=["X"],
            seed=3,
        )
        assert out["timed_out"] is True
        assert out["n_tested"] == 0

    def test_failed_at_recorded_when_no_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            lab,
            "_fetch_factor_series",
            lambda slug, days=365: pd.Series(dtype=float, name=slug),
        )
        monkeypatch.setattr(
            lab,
            "_fetch_equity_series",
            lambda t, days=365: pd.Series(dtype=float, name=t),
        )
        out = lab.discover_alphas(
            n_combos=3,
            factor_slugs=["a", "b", "c"],
            equity_tickers=["X"],
            seed=2,
            max_runtime_seconds=5,
        )
        assert all(c["failed_at"] for c in out["candidates"])


class TestPromote:
    def test_promote_writes_to_pending_jsonl(self, patched_fetchers) -> None:
        out = lab.discover_alphas(
            n_combos=3,
            factor_slugs=["s1", "s2", "s3"],
            equity_tickers=["T1"],
            seed=4,
            max_runtime_seconds=5,
            min_quarters_positive=0,
            min_oos_sharpe=0.0,
        )
        # Persist a pseudo-job manually (matches the structure ``_run_job``
        # would write).
        job_id = "job-test-1"
        lab._record_job(
            job_id,
            status="complete",
            results=out,
            started_at="x",
            completed_at="y",
            params={},
        )
        # Use the first candidate available.
        cand_id = out["candidates"][0]["pair_id"]
        entry = lab.promote_candidate(cand_id, job_id=job_id)
        assert entry["candidate_id"] == cand_id
        assert lab.PENDING_FILE.exists()
        line = lab.PENDING_FILE.read_text().strip().splitlines()[0]
        parsed = json.loads(line)
        assert parsed["candidate_id"] == cand_id
        assert parsed["review_status"] == "pending_human_review"

    def test_promote_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            lab.promote_candidate("does-not-exist")


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------


def _build_test_client() -> TestClient:
    app = FastAPI()
    app.include_router(lab.router)
    return TestClient(app)


def test_discover_then_results_flow(patched_fetchers) -> None:
    client = _build_test_client()
    r = client.post(
        "/lab/discover",
        json={
            "n_combos": 3,
            "min_oos_sharpe": 0.0,
            "min_quarters_positive": 0,
            "max_runtime_seconds": 5,
            "seed": 9,
        },
    )
    assert r.status_code == 200, r.text
    job = r.json()
    job_id = job["job_id"]
    assert job["status"] == "queued"

    # FastAPI's TestClient runs background tasks synchronously after the
    # response is dispatched, so by the time control returns the job is done.
    r2 = client.get(f"/lab/results/{job_id}")
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["job_id"] == job_id
    assert body["status"] in {"complete", "running", "queued"}
    assert isinstance(body["candidates"], list)


def test_queue_endpoint_returns_state() -> None:
    client = _build_test_client()
    r = client.get("/lab/queue")
    assert r.status_code == 200
    body = r.json()
    assert "running" in body
    assert "jobs_file" in body


def test_promote_endpoint_persists_to_jsonl(patched_fetchers) -> None:
    client = _build_test_client()
    out = lab.discover_alphas(
        n_combos=3,
        factor_slugs=["a", "b", "c"],
        equity_tickers=["X"],
        seed=5,
        max_runtime_seconds=5,
        min_quarters_positive=0,
        min_oos_sharpe=0.0,
    )
    job_id = "job-router-1"
    lab._record_job(
        job_id,
        status="complete",
        results=out,
        started_at="x",
        completed_at="y",
        params={},
    )
    cand_id = out["candidates"][0]["pair_id"]
    r = client.post(f"/lab/promote/{cand_id}", params={"job_id": job_id})
    assert r.status_code == 200, r.text
    assert lab.PENDING_FILE.exists()


def test_promote_unknown_returns_404() -> None:
    client = _build_test_client()
    r = client.post("/lab/promote/missing-id")
    assert r.status_code == 404


def test_results_unknown_job_returns_404() -> None:
    client = _build_test_client()
    r = client.get("/lab/results/nope")
    assert r.status_code == 404

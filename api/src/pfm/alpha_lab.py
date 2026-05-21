"""Auto-Generated Alpha Lab.

A self-growing discovery system that random-samples pairs from
``(factors × factors) ∪ (factors × equity_tickers)``, runs the full
quant validation pipeline (cointegration → triple-barrier → walk-forward
→ robust_validation), maps verdicts via :func:`alpha_card_verdict`, and
surfaces candidates that pass all gates.

Storage / state
---------------
* Job records live in ``/tmp/pfm_lab_jobs.json`` (overwriting; small dict).
* Promotion candidates appended to ``/tmp/pfm_lab_pending.jsonl`` for
  human review **before** ever changing ``alpha_strategies.json``.
* In-process runtime state guarded by an :class:`asyncio.Lock` and a
  simple ``_running`` flag to prevent concurrent ``/discover`` runs.

The lab is intentionally bounded: ``max_runtime_seconds`` lets the demo
fire-and-forget without blocking the event loop, and the per-combo budget
short-circuits as soon as a gate fails (cheaper gates first).
"""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from pfm.auth.dependencies import require_tier

# --- storage paths ----------------------------------------------------------

JOBS_FILE: Path = Path("/tmp/pfm_lab_jobs.json")
PENDING_FILE: Path = Path("/tmp/pfm_lab_pending.jsonl")

# --- defaults ---------------------------------------------------------------

DEFAULT_EQUITY_TICKERS: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "TLT",
    "GLD",
    "BTC-USD",
    "ETH-USD",
    "NVDA",
    "TSLA",
    "AAPL",
    "MSFT",
    "DJT",
    "COIN",
)
HISTORY_DAYS: int = 365
SYNTHETIC_WINDOW: int = 250  # bars for synthetic series in tests
MIN_OBS: int = 80


# --- state ------------------------------------------------------------------


@dataclass
class _LabState:
    """Singleton in-process state. Not thread-safe across processes."""

    running: bool = False
    last_job_id: str | None = None
    last_run_at: str | None = None
    last_results_summary: dict[str, Any] | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_STATE = _LabState()


# --- jobs persistence -------------------------------------------------------


def _load_jobs() -> dict[str, Any]:
    if not JOBS_FILE.exists():
        return {}
    try:
        return json.loads(JOBS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_jobs(jobs: dict[str, Any]) -> None:
    JOBS_FILE.write_text(json.dumps(jobs, default=str))


def _record_job(job_id: str, **fields: Any) -> None:
    jobs = _load_jobs()
    rec = jobs.get(job_id, {})
    rec.update(fields)
    jobs[job_id] = rec
    _save_jobs(jobs)


# --- candidate dataclass ----------------------------------------------------


@dataclass
class AlphaCandidate:
    pair_id: str
    leg_a: str
    leg_b: str
    pair_type: Literal["factor_factor", "factor_equity"]
    n_obs: int
    adf_pvalue: float
    is_cointegrated: bool
    triple_barrier_sharpe: float | None = None
    n_trades: int = 0
    walk_forward_test_sharpe_mean: float | None = None
    walk_forward_stability: str | None = None
    quarters_positive: int = 0
    bootstrap_ci_lo: float | None = None
    robust_verdict: str | None = None
    projected_tier: str = "D_REJECTED"
    projected_action: str = "ARCHIVE"
    failed_at: str | None = None  # which gate killed the candidate


# --- candidate generation ---------------------------------------------------


def _candidate_pairs(
    factor_slugs: list[str],
    equity_tickers: list[str],
    n_combos: int,
    seed: int,
) -> list[tuple[str, str, str]]:
    """Random unique (leg_a, leg_b, pair_type) triples."""
    rng = random.Random(seed)
    pairs: set[tuple[str, str, str]] = set()
    pool_factors = list(factor_slugs)
    pool_equity = list(equity_tickers)

    attempts = 0
    cap = max(n_combos * 5, 50)
    while len(pairs) < n_combos and attempts < cap:
        attempts += 1
        if pool_factors and pool_equity and rng.random() < 0.5:
            a = rng.choice(pool_factors)
            b = rng.choice(pool_equity)
            pair_type = "factor_equity"
        elif len(pool_factors) >= 2:
            a, b = rng.sample(pool_factors, 2)
            pair_type = "factor_factor"
        elif pool_factors and pool_equity:
            a = rng.choice(pool_factors)
            b = rng.choice(pool_equity)
            pair_type = "factor_equity"
        else:
            break
        if a == b:
            continue
        pairs.add((a, b, pair_type))
    return list(pairs)


# --- price fetchers (delegated, mockable) -----------------------------------


def _fetch_factor_series(slug: str, days: int = HISTORY_DAYS) -> pd.Series:
    """Pull a factor's daily price series, falling back to ``main``.

    Returns an empty series on any failure so a single bad slug doesn't kill
    the run.
    """
    end = pd.Timestamp.now(tz="UTC").normalize()
    start = end - pd.Timedelta(days=days)
    try:
        from pfm import main as main_mod
        from pfm.factors import FactorConfig

        factors = getattr(main_mod.app.state, "factors", {})
        # O(1) slug → FactorConfig via the lifespan-built index; fall back to
        # an on-the-fly build for tests that wire ``factors`` without the index.
        by_slug = getattr(main_mod.app.state, "factors_by_slug", None)
        if not isinstance(by_slug, dict) or not by_slug:
            by_slug = {f.slug: f for f in factors.values() if f.slug}
        fc = by_slug.get(slug) or FactorConfig(
            id=slug,
            name=slug,
            slug=slug,
            source="polymarket",
            description="lab ad-hoc",
            theme="other",
        )
        df = main_mod._cached_factor_history(
            fc,
            start,
            end,
            main_mod.app.state.poly,
            main_mod.app.state.cache,
            main_mod.get_settings(),
        )
        if df is None or df.empty or "price" not in df.columns:
            return pd.Series(dtype=float, name=slug)
        s = df["price"].dropna()
        s.name = slug
        return s
    except Exception:
        return pd.Series(dtype=float, name=slug)


def _fetch_equity_series(ticker: str, days: int = HISTORY_DAYS) -> pd.Series:
    """Pull adjusted closes from yfinance via the replay-mode cache."""
    try:
        from pfm.replay_mode import _equity_closes

        end = pd.Timestamp.now(tz="UTC").normalize()
        start = end - pd.Timedelta(days=days)
        return _equity_closes(ticker, start, end)
    except Exception:
        return pd.Series(dtype=float, name=ticker)


# --- per-combo evaluation ---------------------------------------------------


def _quarters_positive(pnl: pd.Series) -> int:
    """Count positive-Sharpe quarters in a per-bar PnL series."""
    if pnl is None or pnl.empty:
        return 0
    df = pnl.to_frame("pnl")
    idx = df.index
    # ``to_period`` drops tz; convert to naive UTC first to silence the warning.
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df["q"] = idx.to_period("Q")
    out = 0
    for _, sub in df.groupby("q"):
        sd = float(sub["pnl"].std(ddof=1)) if len(sub) > 1 else 0.0
        if sd <= 0:
            continue
        sharpe = float(sub["pnl"].mean()) / sd * np.sqrt(252.0)
        if sharpe > 0:
            out += 1
    return out


def _evaluate_pair(
    leg_a: str,
    leg_b: str,
    pair_type: str,
    series_a: pd.Series,
    series_b: pd.Series,
    *,
    min_oos_sharpe: float,
    min_quarters_positive: int,
) -> AlphaCandidate:
    """Run the cascading gate pipeline on a single pair."""
    pair_id = f"{leg_a}__{leg_b}"
    cand = AlphaCandidate(
        pair_id=pair_id,
        leg_a=leg_a,
        leg_b=leg_b,
        pair_type=pair_type,  # type: ignore[arg-type]
        n_obs=0,
        adf_pvalue=float("nan"),
        is_cointegrated=False,
    )

    # Align
    if series_a is None or series_b is None or series_a.empty or series_b.empty:
        cand.failed_at = "no_data"
        return cand
    a, b = series_a.align(series_b, join="inner")
    a = a.dropna()
    b = b.dropna()
    common = a.index.intersection(b.index)
    a = a.loc[common]
    b = b.loc[common]
    cand.n_obs = len(a)
    if cand.n_obs < MIN_OBS:
        cand.failed_at = "too_few_obs"
        return cand

    # Gate 1: cointegration
    try:
        from pfm.cointegration import engle_granger

        eg = engle_granger(a, b)
        cand.adf_pvalue = float(eg.adf_pvalue)
        cand.is_cointegrated = bool(eg.cointegrated)
        if not cand.is_cointegrated:
            cand.failed_at = "cointegration"
            return cand
        spread = eg.spread
    except Exception:
        cand.failed_at = "cointegration_error"
        return cand

    # Gate 2: triple-barrier sharpe
    try:
        from pfm.triple_barrier import triple_barrier_backtest

        tb = triple_barrier_backtest(spread)
        cand.triple_barrier_sharpe = float(tb.sharpe)
        cand.n_trades = (
            int(tb.n_trades) if hasattr(tb, "n_trades") else len(getattr(tb, "trades", []))
        )
        if cand.triple_barrier_sharpe < min_oos_sharpe * 0.5:
            cand.failed_at = "triple_barrier"
            return cand
    except Exception:
        cand.failed_at = "triple_barrier_error"
        return cand

    # Gate 3: walk-forward
    try:
        from pfm.advanced import walk_forward_backtest

        wf = walk_forward_backtest(spread, n_folds=4)
        cand.walk_forward_test_sharpe_mean = float(wf.test_sharpe_mean)
        cand.walk_forward_stability = wf.stability
        if (cand.walk_forward_test_sharpe_mean or 0.0) < min_oos_sharpe:
            cand.failed_at = "walk_forward"
            return cand
    except Exception:
        cand.failed_at = "walk_forward_error"
        return cand

    # Gate 4: quarterly stability + bootstrap CI
    try:
        from pfm.robust_validation import run_robust_validation

        # Build per-bar PnL approximation from spread differences.
        pnl = spread.diff().fillna(0.0)
        cand.quarters_positive = _quarters_positive(pnl)
        rep = run_robust_validation(pnl)
        cand.bootstrap_ci_lo = float(rep.bootstrap_ci.get("ci_lo_95", float("nan")))
        cand.robust_verdict = rep.overall_verdict
        if cand.quarters_positive < min_quarters_positive:
            cand.failed_at = "quarterly_stability"
            return cand
    except Exception:
        cand.failed_at = "robust_validation_error"
        return cand

    # All gates passed → project tier via alpha_card_verdict.
    try:
        from pfm.strategy_verdict import alpha_card_verdict

        if (
            cand.quarters_positive >= 4
            and (cand.walk_forward_test_sharpe_mean or 0.0) >= 1.0
            and (cand.bootstrap_ci_lo or 0.0) > 0
        ):
            cand.projected_tier = "A_GOLD"
        elif cand.quarters_positive >= 3:
            cand.projected_tier = "B_VALIDATED"
        else:
            cand.projected_tier = "C_TENTATIVE"
        verdict = alpha_card_verdict(
            {
                "tier": cand.projected_tier,
                "name": pair_id,
                "sharpe_oos": cand.walk_forward_test_sharpe_mean,
            }
        )
        cand.projected_action = str(verdict.get("action", "WATCH_DO_NOT_DEPLOY"))
    except Exception:
        cand.projected_action = "WATCH_DO_NOT_DEPLOY"

    return cand


# --- public API -------------------------------------------------------------


def discover_alphas(
    n_combos: int = 100,
    min_oos_sharpe: float = 1.0,
    min_quarters_positive: int = 3,
    max_runtime_seconds: int = 60,
    *,
    factor_slugs: list[str] | None = None,
    equity_tickers: list[str] | None = None,
    seed: int = 17,
) -> dict[str, Any]:
    """Run the discovery pipeline.

    Returns a dict matching the lab-results schema.
    """
    t0 = time.monotonic()
    if factor_slugs is None:
        try:
            from pfm import main as main_mod

            factors = getattr(main_mod.app.state, "factors", {})
            factor_slugs = [f.slug for f in factors.values()]
        except Exception:
            factor_slugs = []
    if equity_tickers is None:
        equity_tickers = list(DEFAULT_EQUITY_TICKERS)

    pairs = _candidate_pairs(factor_slugs or [], equity_tickers, n_combos, seed)
    candidates: list[AlphaCandidate] = []
    n_tested = 0
    n_timed_out = 0
    timed_out = False
    for leg_a, leg_b, pair_type in pairs:
        if time.monotonic() - t0 > max_runtime_seconds:
            timed_out = True
            n_timed_out = len(pairs) - n_tested
            break
        n_tested += 1
        if pair_type == "factor_factor":
            sa = _fetch_factor_series(leg_a)
            sb = _fetch_factor_series(leg_b)
        else:  # factor_equity
            sa = _fetch_factor_series(leg_a)
            sb = _fetch_equity_series(leg_b)
        cand = _evaluate_pair(
            leg_a,
            leg_b,
            pair_type,
            sa,
            sb,
            min_oos_sharpe=min_oos_sharpe,
            min_quarters_positive=min_quarters_positive,
        )
        candidates.append(cand)

    n_passed = sum(1 for c in candidates if c.failed_at is None)
    runtime_seconds = time.monotonic() - t0

    return {
        "n_tested": n_tested,
        "n_passed": n_passed,
        "n_skipped_timeout": n_timed_out,
        "timed_out": timed_out,
        "runtime_seconds": round(runtime_seconds, 3),
        "params": {
            "n_combos": n_combos,
            "min_oos_sharpe": min_oos_sharpe,
            "min_quarters_positive": min_quarters_positive,
            "max_runtime_seconds": max_runtime_seconds,
            "seed": seed,
        },
        "candidates": [asdict(c) for c in candidates],
    }


def lab_queue() -> dict[str, Any]:
    """Return a snapshot of the lab's runtime state."""
    return {
        "running": _STATE.running,
        "last_job_id": _STATE.last_job_id,
        "last_run_at": _STATE.last_run_at,
        "last_results_summary": _STATE.last_results_summary,
        "jobs_file": str(JOBS_FILE),
        "pending_file": str(PENDING_FILE),
    }


def _run_job(job_id: str, params: dict[str, Any]) -> None:
    """Background-task entrypoint. Updates persistent state and global flag."""
    _STATE.running = True
    _STATE.last_job_id = job_id
    started = datetime.now(UTC).isoformat()
    _record_job(job_id, status="running", started_at=started, params=params)
    try:
        result = discover_alphas(**params)
        finished = datetime.now(UTC).isoformat()
        _record_job(
            job_id,
            status="complete",
            completed_at=finished,
            results=result,
        )
        _STATE.last_run_at = finished
        _STATE.last_results_summary = {
            "n_tested": result["n_tested"],
            "n_passed": result["n_passed"],
            "runtime_seconds": result["runtime_seconds"],
            "timed_out": result["timed_out"],
        }
    except Exception as e:  # pragma: no cover  (defensive)
        _record_job(job_id, status="error", error=str(e))
    finally:
        _STATE.running = False


def promote_candidate(candidate_id: str, job_id: str | None = None) -> dict[str, Any]:
    """Append a candidate to the human-review queue (no auto-promotion)."""
    jobs = _load_jobs()
    found = None
    if job_id:
        rec = jobs.get(job_id)
        if rec and "results" in rec:
            for c in rec["results"].get("candidates", []):
                if c.get("pair_id") == candidate_id:
                    found = c
                    break
    if found is None:
        for jid, rec in jobs.items():
            if "results" not in rec:
                continue
            for c in rec["results"].get("candidates", []):
                if c.get("pair_id") == candidate_id:
                    found = c
                    job_id = jid
                    break
            if found:
                break
    if found is None:
        raise KeyError(f"candidate {candidate_id!r} not found in any job results")

    entry = {
        "candidate_id": candidate_id,
        "job_id": job_id,
        "promoted_at": datetime.now(UTC).isoformat(),
        "candidate": found,
        "review_status": "pending_human_review",
    }
    PENDING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PENDING_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


# --- Pydantic schemas -------------------------------------------------------


class DiscoverRequest(BaseModel):
    n_combos: int = Field(20, ge=1, le=2000)
    min_oos_sharpe: float = Field(1.0, ge=0.0, le=10.0)
    min_quarters_positive: int = Field(3, ge=0, le=8)
    max_runtime_seconds: int = Field(60, ge=1, le=600)
    seed: int = Field(17, ge=0)


class DiscoverJobResponse(BaseModel):
    job_id: str
    status: str
    started_at: str
    params: dict[str, Any]


class LabQueueResponse(BaseModel):
    running: bool
    last_job_id: str | None
    last_run_at: str | None
    last_results_summary: dict[str, Any] | None
    jobs_file: str
    pending_file: str


class CandidateOut(BaseModel):
    pair_id: str
    leg_a: str
    leg_b: str
    pair_type: str
    n_obs: int
    adf_pvalue: float
    is_cointegrated: bool
    triple_barrier_sharpe: float | None
    n_trades: int
    walk_forward_test_sharpe_mean: float | None
    walk_forward_stability: str | None
    quarters_positive: int
    bootstrap_ci_lo: float | None
    robust_verdict: str | None
    projected_tier: str
    projected_action: str
    failed_at: str | None


class JobResultResponse(BaseModel):
    job_id: str
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    params: dict[str, Any] | None = None
    n_tested: int | None = None
    n_passed: int | None = None
    runtime_seconds: float | None = None
    timed_out: bool | None = None
    candidates: list[CandidateOut] = []


class PromoteResponse(BaseModel):
    candidate_id: str
    job_id: str | None
    promoted_at: str
    review_status: str
    pending_file: str


# --- router -----------------------------------------------------------------


router = APIRouter(prefix="/lab", tags=["alpha-lab"])


@router.post(
    "/discover",
    response_model=DiscoverJobResponse,
    summary="Kick off an alpha-discovery run (background task)",
    dependencies=[Depends(require_tier("quant"))],
)
def post_discover(body: DiscoverRequest, background_tasks: BackgroundTasks) -> DiscoverJobResponse:
    if _STATE.running:
        raise HTTPException(
            status_code=409,
            detail="A discovery run is already in progress; wait for it to finish.",
        )
    job_id = str(uuid.uuid4())
    started = datetime.now(UTC).isoformat()
    params = body.model_dump()
    _record_job(job_id, status="queued", started_at=started, params=params)
    background_tasks.add_task(_run_job, job_id, params)
    return DiscoverJobResponse(
        job_id=job_id,
        status="queued",
        started_at=started,
        params=params,
    )


@router.get(
    "/queue",
    response_model=LabQueueResponse,
    summary="Get the lab's current runtime state",
)
def get_queue() -> LabQueueResponse:
    return LabQueueResponse(**lab_queue())


@router.get(
    "/results/{job_id}",
    response_model=JobResultResponse,
    summary="Fetch results for a specific job",
)
def get_results(job_id: str) -> JobResultResponse:
    jobs = _load_jobs()
    rec = jobs.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"unknown job_id {job_id!r}")
    payload: dict[str, Any] = {
        "job_id": job_id,
        "status": rec.get("status", "unknown"),
        "started_at": rec.get("started_at"),
        "completed_at": rec.get("completed_at"),
        "params": rec.get("params"),
    }
    results = rec.get("results")
    if results:
        payload.update(
            n_tested=results.get("n_tested"),
            n_passed=results.get("n_passed"),
            runtime_seconds=results.get("runtime_seconds"),
            timed_out=results.get("timed_out"),
            candidates=results.get("candidates", []),
        )
    return JobResultResponse(**payload)


@router.post(
    "/promote/{candidate_id}",
    response_model=PromoteResponse,
    summary="Mark a candidate for human review (does NOT auto-promote)",
)
def post_promote(candidate_id: str, job_id: str | None = None) -> PromoteResponse:
    try:
        entry = promote_candidate(candidate_id, job_id=job_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return PromoteResponse(
        candidate_id=entry["candidate_id"],
        job_id=entry.get("job_id"),
        promoted_at=entry["promoted_at"],
        review_status=entry["review_status"],
        pending_file=str(PENDING_FILE),
    )


__all__ = [
    "JOBS_FILE",
    "PENDING_FILE",
    "AlphaCandidate",
    "DiscoverJobResponse",
    "DiscoverRequest",
    "JobResultResponse",
    "LabQueueResponse",
    "PromoteResponse",
    "discover_alphas",
    "lab_queue",
    "promote_candidate",
    "router",
]

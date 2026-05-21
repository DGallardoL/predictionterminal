"""Real walk-forward harness that regenerates ``web/data/alpha_strategies.json``.

This module is the antidote to the manually-assigned tiers in the curated
``alpha_strategies.json`` file.  For every pair already in the file we:

1. Fetch real Polymarket history (60-120 day window) for both legs.
2. Run :func:`pfm.cointegration.engle_granger` to recover the spread.
3. If cointegrated, run :func:`pfm.advanced.walk_forward_backtest` with embargo
   (4 folds, Lopez de Prado-style).
4. Run :func:`pfm.advanced.permutation_sharpe_test` (200 iterations) to obtain a
   marginal p-value for the OOS Sharpe.
5. Apply quarterly stability + alpha card verdict gates.
6. After every pair is processed, apply Benjamini-Hochberg FDR over **all**
   collected p-values to obtain q-values, then assign the final tier.

Concurrency is bounded by an :class:`asyncio.Semaphore` (default 10) so we
don't blow through Polymarket's 1000-request / 10-second budget when there are
~88 pairs to process.

All public entrypoints are pure-Python and fully unit-testable: the fetcher is
injected through the ``fetcher`` argument so tests can pass synthetic series
without ever touching the network.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from pfm.advanced import permutation_sharpe_test, walk_forward_backtest
from pfm.auth.dependencies import require_admin
from pfm.cointegration import engle_granger
from pfm.multitest import benjamini_hochberg_fdr
from pfm.strategy_verdict import alpha_card_verdict, quarterly_stability_test

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants and paths
# ---------------------------------------------------------------------------

#: Default location of the curated alpha-strategies JSON, relative to ``api/``.
DEFAULT_ALPHA_PATH: Path = (
    Path(__file__).resolve().parents[3] / "web" / "data" / "alpha_strategies.json"
)
#: Default location of the regen-report markdown directory.
DEFAULT_REPORT_DIR: Path = Path(__file__).resolve().parents[3] / "docs"
#: In-process job persistence for the admin endpoint.
JOBS_FILE: Path = Path("/tmp/pfm_alpha_tier_regen_jobs.json")
#: Default fetch window (days) — covers ~one earnings cycle and ≥ 4 quarters
#: when chained with prior data.
DEFAULT_HISTORY_DAYS: int = 120
#: Permutation iterations for the Sharpe null distribution.
DEFAULT_PERM_ITERS: int = 200
#: Walk-forward folds.
DEFAULT_N_FOLDS: int = 4
#: Polymarket fetch concurrency.
DEFAULT_FETCH_CONCURRENCY: int = 10
#: Minimum sample size to even attempt the cointegration test.
MIN_OBS: int = 50
#: v22 (2026-05-19): need 4 disjoint quarters before A/B tier
#: (see ``docs/alpha-reports/alpha-report-v22.md`` §5.3). Any pair with
#: fewer joint trading days is capped at ``C_TENTATIVE`` regardless of
#: cointegration / BH-FDR / Sharpe — the structural-confidence question is
#: settled at the data-availability layer, before statistical gates run.
JOINT_DAYS_4Q_GATE: int = 360

#: Minimum in-sample PnL observations required to emit a real ``full_sharpe``.
#: Below this we return ``None`` (not ``0.0``) so downstream consumers don't
#: mistake "sample too short" for "strategy is flat". Mirrors the constant in
#: ``scripts/backfill_ah_sweeps.py``.
_MIN_IS_OBS_FOR_SHARPE = 30
#: Same idea for OOS: an OOS Sharpe computed from <30 PnL points is noise.
_MIN_OOS_OBS_FOR_SHARPE = 30

OutputMode = Literal["update", "backup", "dry-run"]


def _guarded_wf_sharpe(
    folds: list[Any],
    side: Literal["train", "test"],
) -> float | None:
    """Aggregate walk-forward per-fold Sharpes with a sample-size + zero-variance guard.

    ``walk_forward_backtest`` coerces zero-variance folds to ``0.0`` and any
    fold whose train/test slice is too short to ``0.0`` as well. Unconditionally
    averaging those values produced impossible ``alpha_strategies.json`` rows
    like ``full_sharpe=0.00`` next to ``oos_sharpe=9.47`` (one lucky tiny-var
    fold paired with three sentinels).

    We:
    1. Drop folds whose ``n_train``/``n_test`` is below ``_MIN_*_OBS_FOR_SHARPE``.
    2. Drop folds whose per-fold Sharpe is exactly ``0.0`` — these are almost
       certainly the zero-variance sentinel from ``walk_forward_backtest``;
       a true 0.0 Sharpe on continuous PnL is measure-zero, and dropping it
       only weakens the aggregate (conservative).
    3. Return ``None`` when fewer than half the folds survive — the aggregate
       is dominated by a single fold and not trustworthy.
    """
    if side == "train":
        n_attr, s_attr, min_obs = "n_train", "train_sharpe", _MIN_IS_OBS_FOR_SHARPE
    else:
        n_attr, s_attr, min_obs = "n_test", "test_sharpe", _MIN_OOS_OBS_FOR_SHARPE
    if not folds:
        return None
    valid: list[float] = []
    for f in folds:
        n = getattr(f, n_attr, 0)
        s = float(getattr(f, s_attr, 0.0))
        if n < min_obs:
            continue
        if s == 0.0:  # zero-variance sentinel from walk_forward_backtest
            continue
        valid.append(s)
    # Require a majority of folds to be valid; otherwise the aggregate is just
    # noise from one or two outlier folds.
    if len(valid) * 2 < len(folds):
        return None
    return float(np.mean(valid))


# ---------------------------------------------------------------------------
# Per-pair result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PairRegenResult:
    """Outcome of running a single pair through the rigor pipeline."""

    pair_id: str
    a_id: str
    b_id: str
    a_slug: str | None
    b_slug: str | None
    n_obs: int = 0
    cointegrated: bool = False
    adf_pvalue: float | None = None
    half_life_days: float | None = None
    beta_hedge: float | None = None
    oos_sharpe: float | None = None
    full_sharpe: float | None = None
    perm_p: float | None = None
    bh_q_value: float | None = None
    passes_bh_q05: bool = False
    passes_bh_q10: bool = False
    quarterly_stability: dict[str, Any] = field(default_factory=dict)
    walk_forward_stability: str | None = None
    tier: str = "D_RAW"
    tier_action: str = "WATCH_DO_NOT_DEPLOY"
    regen_error: str | None = None
    regenerated_at_iso: str | None = None


# ---------------------------------------------------------------------------
# Pair-level pipeline (sync, deterministic given input series)
# ---------------------------------------------------------------------------


def _pair_id(a_id: str, b_id: str) -> str:
    return f"{a_id}__{b_id}"


def _pnl_from_position(
    spread: pd.Series,
    *,
    window: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
) -> np.ndarray:
    """Z-score state machine PnL — mirrors the formula in
    :func:`walk_forward_backtest` so OOS Sharpe and the permutation null are
    commensurable."""
    s = pd.Series(np.asarray(spread)).reset_index(drop=True)
    mu = s.rolling(window=window, min_periods=max(5, window // 2)).mean()
    sd = s.rolling(window=window, min_periods=max(5, window // 2)).std(ddof=1)
    z = (s - mu) / sd
    n = len(s)
    pos = np.zeros(n, dtype=int)
    state = 0
    for i, zi in enumerate(z.values):
        if np.isnan(zi):
            pos[i] = state
            continue
        if state == 0:
            if zi <= -entry_z:
                state = 1
            elif zi >= entry_z:
                state = -1
        elif (state == 1 and (abs(zi) < exit_z or zi <= -stop_z)) or (
            state == -1 and (abs(zi) < exit_z or zi >= stop_z)
        ):
            state = 0
        pos[i] = state
    dspread = np.diff(s.values, prepend=s.values[0])
    pnl = np.concatenate([[0.0], pos[:-1].astype(float) * dspread[1:]])
    return pnl


def _quarterly_sharpes(pnl: pd.Series, *, ann: float = 252.0) -> list[float]:
    """Per-quarter Sharpe of a per-bar PnL series. Returns [] if no DT index."""
    if pnl.empty:
        return []
    idx = pnl.index
    if not isinstance(idx, pd.DatetimeIndex):
        return []
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df = pd.DataFrame({"pnl": pnl.values}, index=idx)
    df["q"] = df.index.to_period("Q")
    out: list[float] = []
    for _, sub in df.groupby("q"):
        if len(sub) < 5:
            continue
        sd = float(sub["pnl"].std(ddof=1))
        if sd <= 0:
            out.append(0.0)
            continue
        out.append(float(sub["pnl"].mean()) / sd * math.sqrt(ann))
    return out


def evaluate_pair(
    pair: dict[str, Any],
    series_a: pd.Series,
    series_b: pd.Series,
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    perm_iters: int = DEFAULT_PERM_ITERS,
    seed: int = 42,
) -> PairRegenResult:
    """Run a single pair through cointegration → walk-forward → permutation.

    Does **not** apply BH-FDR (that's a cross-pair correction, applied in
    :func:`regenerate_alpha_tiers` after every pair has been evaluated).
    Tier defaults to ``D_RAW`` and is upgraded by the caller once q-values
    and quarterly stability are known.
    """
    a_id = str(pair.get("a_id"))
    b_id = str(pair.get("b_id"))
    pid = str(pair.get("pair_id") or _pair_id(a_id, b_id))
    res = PairRegenResult(
        pair_id=pid,
        a_id=a_id,
        b_id=b_id,
        a_slug=pair.get("a_slug"),
        b_slug=pair.get("b_slug"),
    )

    if series_a is None or series_b is None or series_a.empty or series_b.empty:
        res.regen_error = "no_data"
        return res

    a, b = series_a.align(series_b, join="inner")
    a = a.dropna()
    b = b.dropna()
    common = a.index.intersection(b.index)
    a = a.loc[common]
    b = b.loc[common]
    res.n_obs = len(a)
    if res.n_obs < MIN_OBS:
        res.regen_error = f"too_few_obs (n={res.n_obs} < {MIN_OBS})"
        return res

    # Step 1: cointegration -------------------------------------------------
    try:
        eg = engle_granger(a, b)
    except Exception as e:  # pragma: no cover  (defensive)
        res.regen_error = f"engle_granger_error: {e!s}"
        return res

    res.adf_pvalue = float(eg.adf_pvalue) if not math.isnan(eg.adf_pvalue) else None
    res.beta_hedge = float(eg.beta_hedge) if not math.isnan(eg.beta_hedge) else None
    res.half_life_days = float(eg.half_life_days) if eg.half_life_days is not None else None
    res.cointegrated = bool(eg.cointegrated)
    if not res.cointegrated:
        res.regen_error = "not_cointegrated"
        return res

    spread = eg.spread
    if len(spread) < n_folds * 25:
        res.regen_error = f"spread_too_short_for_walk_forward (n={len(spread)})"
        return res

    # Step 2: walk-forward --------------------------------------------------
    try:
        wf = walk_forward_backtest(spread, n_folds=n_folds)
    except Exception as e:
        res.regen_error = f"walk_forward_error: {e!s}"
        return res

    # Sample-size + zero-variance guard: never coerce a degenerate fold's
    # Sharpe to 0.0 — emit ``None`` so downstream consumers (alpha card,
    # sanitizer, leaderboard) can flag "insufficient data" instead of
    # rendering ``full_sharpe=0.00`` next to a spurious ``oos_sharpe=9.47``.
    # See ``_guarded_wf_sharpe`` for the exact rule.
    res.oos_sharpe = _guarded_wf_sharpe(wf.folds, side="test")
    res.full_sharpe = _guarded_wf_sharpe(wf.folds, side="train")
    res.walk_forward_stability = wf.stability

    # Step 3: permutation Sharpe test ---------------------------------------
    try:
        perm = permutation_sharpe_test(
            np.asarray(spread.values),
            pnl_strategy_fn=_pnl_from_position,
            n_iters=perm_iters,
            seed=seed,
        )
    except Exception as e:
        res.regen_error = f"permutation_error: {e!s}"
        return res
    res.perm_p = float(perm.p_value)

    # Step 4: quarterly stability (4-Q gate) --------------------------------
    pnl = spread.diff().fillna(0.0)
    qs = _quarterly_sharpes(pnl)
    qstab = quarterly_stability_test(qs)
    res.quarterly_stability = qstab

    # Default tier — final assignment happens after BH-FDR.
    res.tier = "D_RAW"
    return res


# ---------------------------------------------------------------------------
# Cross-pair tier assignment (BH-FDR + 4Q gate + alpha card verdict)
# ---------------------------------------------------------------------------


_TIER_ALLOC: dict[str, float] = {
    "A_GOLD": 0.15,
    "A_STRUCTURAL": 0.13,
    "B_VALIDATED": 0.10,
    "B_FDR_ONLY": 0.08,
    "C_TENTATIVE": 0.05,
    "D_RAW": 0.03,
}


def _final_tier(
    res: PairRegenResult,
    *,
    is_strike_family: bool = False,
) -> str:
    """Promote/demote a pair based on q-values + quarterly gate.

    Promotion rules
    ---------------
    - A_GOLD: cointegrated AND BH-q05 AND walk-forward OOS Sharpe ≥ 1
      AND quarterly stability passes 4Q gold gate.
    - B_VALIDATED: cointegrated AND BH-q05 AND OOS Sharpe ≥ 0.5.
    - B_FDR_ONLY: cointegrated AND BH-q05 (no OOS Sharpe constraint).
    - C_TENTATIVE: cointegrated AND BH-q10.
    - A_STRUCTURAL: strike-family (bounded by no-arbitrage) AND BH-q10.
    - D_RAW otherwise.
    """
    if not res.cointegrated:
        return "D_RAW"
    # Wave-7 v22 (2026-05-19) gate per docs/alpha-reports/alpha-report-v22.md §5.3:
    # require >= 360 joint trading days (4 disjoint quarters) before A/B candidacy.
    # Catches the structural issue at the data-availability layer, before
    # bootstrap-CI / BH-FDR / deflated Sharpe gates run.
    if res.n_obs is not None and res.n_obs < JOINT_DAYS_4Q_GATE:
        return "C_TENTATIVE"
    qrec = res.quarterly_stability or {}
    sharpe = res.oos_sharpe or 0.0
    if res.passes_bh_q05 and sharpe >= 1.0 and qrec.get("passes_4q_gold"):
        return "A_GOLD"
    if is_strike_family and res.passes_bh_q10:
        return "A_STRUCTURAL"
    if res.passes_bh_q05 and sharpe >= 0.5:
        return "B_VALIDATED"
    if res.passes_bh_q05:
        return "B_FDR_ONLY"
    if res.passes_bh_q10:
        return "C_TENTATIVE"
    return "D_RAW"


def _assign_tiers(
    results: list[PairRegenResult],
    pairs: list[dict[str, Any]],
    *,
    fdr_alpha_05: float = 0.05,
    fdr_alpha_10: float = 0.10,
) -> None:
    """Mutates ``results`` in place: applies BH-FDR over all valid p-values
    then resolves the final tier and tier action per pair."""

    # Collect p-values from results that produced one. Pairs with no perm_p
    # (because they failed an earlier gate) are excluded from the family.
    p_indices: list[int] = [i for i, r in enumerate(results) if r.perm_p is not None]
    p_values: list[float] = [max(0.0, min(1.0, float(results[i].perm_p))) for i in p_indices]

    if p_values:
        bh_05 = benjamini_hochberg_fdr(p_values, alpha=fdr_alpha_05)
        bh_10 = benjamini_hochberg_fdr(p_values, alpha=fdr_alpha_10)
        rejected_05 = set(bh_05["rejected_idx"])
        rejected_10 = set(bh_10["rejected_idx"])
        q_vals = bh_05["q_values"]
        for local_idx, global_idx in enumerate(p_indices):
            r = results[global_idx]
            r.bh_q_value = float(q_vals[local_idx])
            r.passes_bh_q05 = local_idx in rejected_05
            r.passes_bh_q10 = local_idx in rejected_10

    # Resolve the final tier per pair using the strike-family hint from the
    # source record (if present) so we don't lose the A_STRUCTURAL bucket.
    pair_by_id: dict[str, dict[str, Any]] = {str(p.get("pair_id")): p for p in pairs}
    now_iso = datetime.now(UTC).isoformat()
    for r in results:
        src = pair_by_id.get(r.pair_id, {})
        is_strike = bool(src.get("is_strike_family"))
        r.tier = _final_tier(r, is_strike_family=is_strike)
        verdict = alpha_card_verdict(
            {
                "tier": r.tier,
                "name": r.pair_id,
                "sharpe_oos": r.oos_sharpe,
                "allocation_pct": _TIER_ALLOC.get(r.tier, 0.03) * 100,
            }
        )
        r.tier_action = str(verdict.get("action", "WATCH_DO_NOT_DEPLOY"))
        r.regenerated_at_iso = now_iso


# ---------------------------------------------------------------------------
# Concurrent fetch driver
# ---------------------------------------------------------------------------


# A fetcher takes a slug and returns a pd.Series of daily prices indexed by
# UTC midnight Timestamps. Implementations may be sync or async — async is
# preferred so we can fan out at the connection-pool level.
PairFetcher = Callable[[str, int], Awaitable[pd.Series]]


async def _default_fetcher(slug: str, days: int = DEFAULT_HISTORY_DAYS) -> pd.Series:
    """Async wrapper around the sync :func:`fetch_factor_history` call.

    Runs in a thread so the event loop can fan out across pairs even though the
    underlying httpx client is synchronous.
    """
    from pfm.sources.polymarket import PolymarketClient, fetch_factor_history

    end = pd.Timestamp.now(tz="UTC").normalize()
    start = end - pd.Timedelta(days=days)

    def _sync() -> pd.Series:
        with PolymarketClient(
            gamma_url=os.environ.get("POLYMARKET_GAMMA_URL", "https://gamma-api.polymarket.com"),
            clob_url=os.environ.get("POLYMARKET_CLOB_URL", "https://clob.polymarket.com"),
        ) as client:
            df = fetch_factor_history(client, slug, start, end)
        if df is None or df.empty or "price" not in df.columns:
            return pd.Series(dtype=float, name=slug)
        s = df["price"].dropna()
        s.name = slug
        return s

    return await asyncio.to_thread(_sync)


async def _fetch_with_semaphore(
    sem: asyncio.Semaphore,
    slug: str,
    fetcher: PairFetcher,
    days: int,
) -> pd.Series:
    async with sem:
        try:
            return await fetcher(slug, days)
        except Exception as e:
            logger.warning("fetch failed for slug=%s: %s", slug, e)
            return pd.Series(dtype=float, name=slug)


async def _process_pair_async(
    pair: dict[str, Any],
    fetcher: PairFetcher,
    sem: asyncio.Semaphore,
    *,
    history_days: int,
    n_folds: int,
    perm_iters: int,
    seed: int,
) -> PairRegenResult:
    a_slug = pair.get("a_slug") or pair.get("a_id")
    b_slug = pair.get("b_slug") or pair.get("b_id")
    sa, sb = await asyncio.gather(
        _fetch_with_semaphore(sem, str(a_slug), fetcher, history_days),
        _fetch_with_semaphore(sem, str(b_slug), fetcher, history_days),
    )
    return await asyncio.to_thread(
        evaluate_pair,
        pair,
        sa,
        sb,
        n_folds=n_folds,
        perm_iters=perm_iters,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


async def regenerate_alpha_tiers(
    *,
    pairs: list[dict[str, Any]] | None = None,
    alpha_path: Path | None = None,
    output_mode: OutputMode = "dry-run",
    max_runtime_seconds: int = 600,
    history_days: int = DEFAULT_HISTORY_DAYS,
    n_folds: int = DEFAULT_N_FOLDS,
    perm_iters: int = DEFAULT_PERM_ITERS,
    fetch_concurrency: int = DEFAULT_FETCH_CONCURRENCY,
    seed: int = 42,
    fetcher: PairFetcher | None = None,
    report_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full regen pipeline and (optionally) persist the result.

    Parameters
    ----------
    pairs:
        Optional explicit list of pair dicts. If ``None``, loaded from
        ``alpha_path``.
    alpha_path:
        Defaults to :data:`DEFAULT_ALPHA_PATH`.
    output_mode:
        ``"update"`` overwrites in place; ``"backup"`` writes to
        ``<alpha_path>.regenerated.<ts>``; ``"dry-run"`` writes nothing.
    max_runtime_seconds:
        Best-effort wall-clock budget. ``0`` short-circuits before processing
        any pair (returns an empty result with ``timed_out=True``).
    fetcher:
        Async ``(slug, days) -> Series`` callable used to obtain price
        history. Defaults to a Polymarket-backed implementation. Tests pass a
        synthetic in-memory fetcher.
    """
    t0 = time.monotonic()
    alpha_path = alpha_path or DEFAULT_ALPHA_PATH
    report_dir = report_dir or DEFAULT_REPORT_DIR
    fetcher = fetcher or _default_fetcher

    if pairs is None:
        if not alpha_path.exists():
            raise FileNotFoundError(f"alpha_strategies.json not found at {alpha_path}")
        with alpha_path.open(encoding="utf-8") as fh:
            doc = json.load(fh)
        pairs = list(doc.get("strategies", []))
    else:
        doc = {"strategies": pairs}

    pre_tiers = Counter(str(p.get("tier", "")) for p in pairs)

    sem = asyncio.Semaphore(max(1, fetch_concurrency))
    results: list[PairRegenResult] = []
    n_processed = 0
    timed_out = False

    if max_runtime_seconds <= 0:
        # Short-circuit: nothing to do.
        timed_out = True
    else:

        async def _wrap(p: dict[str, Any]) -> PairRegenResult:
            try:
                return await _process_pair_async(
                    p,
                    fetcher,
                    sem,
                    history_days=history_days,
                    n_folds=n_folds,
                    perm_iters=perm_iters,
                    seed=seed,
                )
            except Exception as e:  # pragma: no cover  (defensive)
                pid = str(p.get("pair_id") or _pair_id(str(p.get("a_id")), str(p.get("b_id"))))
                return PairRegenResult(
                    pair_id=pid,
                    a_id=str(p.get("a_id")),
                    b_id=str(p.get("b_id")),
                    a_slug=p.get("a_slug"),
                    b_slug=p.get("b_slug"),
                    regen_error=f"unexpected: {e!s}",
                )

        tasks = [asyncio.create_task(_wrap(p)) for p in pairs]
        try:
            for fut in asyncio.as_completed(tasks, timeout=max_runtime_seconds):
                r = await fut
                results.append(r)
                n_processed += 1
        except TimeoutError:
            timed_out = True
            for t in tasks:
                if not t.done():
                    t.cancel()
            # Drain any cancellations to avoid "task pending" warnings.
            await asyncio.gather(*tasks, return_exceptions=True)

    # If we timed out before any pair completed, fall through with what we have.
    _assign_tiers(results, pairs)

    # Build new strategies list, preserving original ordering and merging.
    new_strategies = _merge_into_pairs(pairs, results)
    post_tiers = Counter(str(s.get("tier", "")) for s in new_strategies)

    runtime_s = round(time.monotonic() - t0, 3)

    summary = {
        "n_pairs": len(pairs),
        "n_processed": n_processed,
        "n_errors": sum(1 for r in results if r.regen_error),
        "runtime_seconds": runtime_s,
        "timed_out": timed_out,
        "output_mode": output_mode,
        "pre_tiers": dict(pre_tiers),
        "post_tiers": dict(post_tiers),
        "regenerated_at_iso": datetime.now(UTC).isoformat(),
    }

    written_path: str | None = None
    if output_mode != "dry-run":
        new_doc = dict(doc)
        new_doc["strategies"] = new_strategies
        new_doc["regenerated_at_iso"] = summary["regenerated_at_iso"]
        new_doc["regen_summary"] = summary
        if output_mode == "update":
            target = alpha_path
        else:  # backup
            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            target = alpha_path.with_suffix(f".regenerated.{ts}.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(new_doc, indent=2, default=str), encoding="utf-8")
        written_path = str(target)

    # Markdown report --------------------------------------------------------
    report_path: str | None = None
    try:
        report_path = _write_report(report_dir, pre_tiers, post_tiers, results, summary)
    except Exception as e:  # pragma: no cover
        logger.warning("report write failed: %s", e)

    return {
        "summary": summary,
        "results": [asdict(r) for r in results],
        "strategies": new_strategies,
        "written_path": written_path,
        "report_path": report_path,
    }


def _merge_into_pairs(
    pairs: list[dict[str, Any]],
    results: list[PairRegenResult],
) -> list[dict[str, Any]]:
    """Merge ``results`` back into the original pair dicts, preserving
    untouched fields (rationale, deploy_signal_logic, etc.) and overwriting
    just the regen-affected ones."""
    by_id: dict[str, PairRegenResult] = {r.pair_id: r for r in results}
    merged: list[dict[str, Any]] = []
    for p in pairs:
        pid = str(p.get("pair_id") or _pair_id(str(p.get("a_id")), str(p.get("b_id"))))
        out = dict(p)
        r = by_id.get(pid)
        if r is None:
            out["regen_error"] = "not_processed"
            merged.append(out)
            continue
        out["tier"] = r.tier
        out["oos_sharpe"] = r.oos_sharpe if r.oos_sharpe is not None else out.get("oos_sharpe")
        out["full_sharpe"] = r.full_sharpe if r.full_sharpe is not None else out.get("full_sharpe")
        out["perm_p"] = r.perm_p if r.perm_p is not None else out.get("perm_p")
        out["bh_q_value"] = r.bh_q_value
        out["passes_bh_q05"] = r.passes_bh_q05
        out["passes_bh_q10"] = r.passes_bh_q10
        out["quarterly_stability"] = r.quarterly_stability or {}
        out["walk_forward_stability"] = r.walk_forward_stability
        out["regenerated_at_iso"] = r.regenerated_at_iso
        out["regen_error"] = r.regen_error
        out["tier_action"] = r.tier_action
        if r.adf_pvalue is not None:
            out["adf_pvalue"] = r.adf_pvalue
        if r.beta_hedge is not None:
            out["beta_hedge"] = r.beta_hedge
        if r.half_life_days is not None:
            out["half_life_days"] = r.half_life_days
        if r.n_obs:
            out["n_obs"] = r.n_obs
        # Suggested allocation tracks the new tier.
        out["suggested_allocation"] = _TIER_ALLOC.get(r.tier, 0.03)
        merged.append(out)
    return merged


# ---------------------------------------------------------------------------
# Markdown reporting
# ---------------------------------------------------------------------------


def _write_report(
    report_dir: Path,
    pre_tiers: Counter,
    post_tiers: Counter,
    results: list[PairRegenResult],
    summary: dict[str, Any],
) -> str:
    """Write a one-page markdown report describing the regen run."""
    report_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    path = report_dir / f"alpha-tier-regen-report-{today}.md"

    by_id: dict[str, PairRegenResult] = {r.pair_id: r for r in results}
    new_gold: list[str] = []
    for pid, r in by_id.items():
        if r.tier == "A_GOLD":
            new_gold.append(pid)

    lines: list[str] = []
    lines.append(f"# Alpha-tier regeneration report — {today}")
    lines.append("")
    lines.append(
        f"Pipeline: cointegration -> walk-forward (k={DEFAULT_N_FOLDS}, embargo) -> "
        "permutation Sharpe -> BH-FDR -> 4Q stability -> alpha card verdict."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Pairs in input: **{summary['n_pairs']}**")
    lines.append(f"- Pairs processed: **{summary['n_processed']}**")
    lines.append(f"- Pairs with errors: **{summary['n_errors']}**")
    lines.append(f"- Runtime: **{summary['runtime_seconds']}s**")
    lines.append(f"- Timed out: **{summary['timed_out']}**")
    lines.append(f"- Output mode: `{summary['output_mode']}`")
    lines.append("")
    lines.append("## Tier distribution — before vs after")
    lines.append("")
    lines.append("| Tier | Before | After |")
    lines.append("|------|--------|-------|")
    all_tiers = sorted(set(pre_tiers) | set(post_tiers))
    for t in all_tiers:
        lines.append(f"| {t or 'UNKNOWN'} | {pre_tiers.get(t, 0)} | {post_tiers.get(t, 0)} |")
    lines.append("")
    lines.append("## Pairs that gained A_GOLD")
    lines.append("")
    if not new_gold:
        lines.append("_None._")
    else:
        for pid in new_gold[:50]:
            lines.append(f"- `{pid}`")
    lines.append("")
    lines.append("## Errored pairs (top 25)")
    lines.append("")
    errored = [r for r in results if r.regen_error][:25]
    if not errored:
        lines.append("_None._")
    else:
        lines.append("| pair_id | error |")
        lines.append("|---------|-------|")
        for r in errored:
            err = (r.regen_error or "").replace("|", "/")
            lines.append(f"| `{r.pair_id}` | {err} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Job persistence (admin endpoint)
# ---------------------------------------------------------------------------


@dataclass
class _RegenState:
    running: bool = False
    last_job_id: str | None = None


_STATE = _RegenState()


def _load_jobs() -> dict[str, Any]:
    if not JOBS_FILE.exists():
        return {}
    try:
        return json.loads(JOBS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_jobs(jobs: dict[str, Any]) -> None:
    JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    JOBS_FILE.write_text(json.dumps(jobs, default=str))


def _record_job(job_id: str, **fields: Any) -> None:
    jobs = _load_jobs()
    rec = jobs.get(job_id, {})
    rec.update(fields)
    jobs[job_id] = rec
    _save_jobs(jobs)


async def _run_regen_job(job_id: str, params: dict[str, Any]) -> None:
    _STATE.running = True
    _STATE.last_job_id = job_id
    started = datetime.now(UTC).isoformat()
    _record_job(job_id, status="running", started_at=started, params=params)
    try:
        out = await regenerate_alpha_tiers(**params)
        _record_job(
            job_id,
            status="complete",
            completed_at=datetime.now(UTC).isoformat(),
            summary=out["summary"],
            written_path=out.get("written_path"),
            report_path=out.get("report_path"),
        )
    except Exception as e:
        _record_job(
            job_id,
            status="error",
            completed_at=datetime.now(UTC).isoformat(),
            error=str(e),
        )
    finally:
        _STATE.running = False


# ---------------------------------------------------------------------------
# FastAPI router (admin-only)
# ---------------------------------------------------------------------------


router = APIRouter(prefix="/alpha-hub", tags=["alpha-tier-regen"])


class RegenRequest(BaseModel):
    output_mode: OutputMode = Field("backup")
    max_runtime_seconds: int = Field(600, ge=0, le=3600)
    history_days: int = Field(DEFAULT_HISTORY_DAYS, ge=30, le=365)
    n_folds: int = Field(DEFAULT_N_FOLDS, ge=2, le=10)
    perm_iters: int = Field(DEFAULT_PERM_ITERS, ge=50, le=2000)
    fetch_concurrency: int = Field(DEFAULT_FETCH_CONCURRENCY, ge=1, le=50)
    seed: int = Field(42, ge=0)


class RegenJobResponse(BaseModel):
    job_id: str
    status: str
    started_at: str
    params: dict[str, Any]


class RegenJobStatus(BaseModel):
    job_id: str
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    summary: dict[str, Any] | None = None
    written_path: str | None = None
    report_path: str | None = None
    error: str | None = None
    params: dict[str, Any] | None = None


@router.post(
    "/regenerate-tiers",
    response_model=RegenJobResponse,
    summary="Re-run the walk-forward harness over alpha_strategies.json",
    dependencies=[Depends(require_admin)],
)
def post_regenerate_tiers(
    body: RegenRequest, background_tasks: BackgroundTasks
) -> RegenJobResponse:
    if _STATE.running:
        raise HTTPException(
            status_code=409,
            detail="A regen job is already in progress; wait for it to finish.",
        )
    job_id = str(uuid.uuid4())
    started = datetime.now(UTC).isoformat()
    params = body.model_dump()
    _record_job(job_id, status="queued", started_at=started, params=params)

    def _kickoff() -> None:
        # ``asyncio.run`` to give the background task its own loop. FastAPI's
        # ``BackgroundTasks`` runs sync callables in a thread pool, so this is
        # the safe way to call our async orchestrator without colliding with
        # the request loop.
        asyncio.run(_run_regen_job(job_id, params))

    background_tasks.add_task(_kickoff)
    return RegenJobResponse(
        job_id=job_id,
        status="queued",
        started_at=started,
        params=params,
    )


@router.get(
    "/regenerate-tiers/{job_id}",
    response_model=RegenJobStatus,
    summary="Fetch status / summary of a regen job",
    dependencies=[Depends(require_admin)],
)
def get_regenerate_tiers(job_id: str) -> RegenJobStatus:
    rec = _load_jobs().get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"unknown job_id {job_id!r}")
    return RegenJobStatus(job_id=job_id, **rec)


__all__ = [
    "DEFAULT_ALPHA_PATH",
    "DEFAULT_FETCH_CONCURRENCY",
    "DEFAULT_HISTORY_DAYS",
    "DEFAULT_N_FOLDS",
    "DEFAULT_PERM_ITERS",
    "JOBS_FILE",
    "PairRegenResult",
    "RegenJobResponse",
    "RegenJobStatus",
    "RegenRequest",
    "evaluate_pair",
    "regenerate_alpha_tiers",
    "router",
]

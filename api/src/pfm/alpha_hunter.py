"""Alpha hunter — generic cross-pair alpha-discovery orchestrator.

Given a price-series dictionary keyed by factor id, this module:

1. Forms all C(N, 2) pairs.
2. Runs Engle-Granger on each pair, filters by ADF p-value.
3. For survivors, runs a z-score pairs backtest and computes OOS Sharpe.
4. For pairs that exceed an OOS-Sharpe threshold, runs a sign-flip
   permutation test on the spread to obtain a p-value against the null
   "no exploitable mean-reversion".
5. Ranks survivors and returns a structured report.

The point is to be a single entry point that callers (REST endpoint,
agent sweeps) can invoke against arbitrary factor subsets — the heavy
math primitives stay in their existing modules and we just compose them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from pfm.advanced import permutation_sharpe_test
from pfm.cointegration import engle_granger
from pfm.pairs import pairs_backtest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlphaHit:
    """One pair's full validation result."""

    a_id: str
    b_id: str
    n_obs: int
    adf_pvalue: float
    half_life_days: float | None
    beta_hedge: float
    oos_sharpe: float
    full_sharpe: float
    perm_p: float | None  # None when permutation skipped
    perm_real_sharpe: float | None
    verdict: str  # REAL_ALPHA | promising | marginal | filtered
    runtime_ms: float


@dataclass(frozen=True)
class AlphaHunterReport:
    """Full output of :func:`run_alpha_hunter`."""

    n_factors: int
    n_pairs_total: int
    n_pairs_passed_adf: int
    n_pairs_perm_tested: int
    n_real_alpha: int
    runtime_seconds: float
    hits: list[AlphaHit] = field(default_factory=list)


def _zscore_pnl_factory(
    window: int,
    entry_z: float,
    exit_z: float,
    stop_z: float,
):
    """Make a permutation-compatible PnL function for a fixed strategy."""

    def _pnl(spread_arr: np.ndarray) -> np.ndarray:
        s = pd.Series(spread_arr)
        mu = s.rolling(window=window, min_periods=max(5, window // 2)).mean()
        sd = s.rolling(window=window, min_periods=max(5, window // 2)).std(ddof=1)
        z = (s - mu) / sd
        pos = pd.Series(0, index=s.index, dtype=float)
        cur = 0.0
        for i, zi in enumerate(z.values):
            if np.isnan(zi):
                pos.iloc[i] = cur
                continue
            if cur == 0.0:
                if zi >= entry_z:
                    cur = -1.0
                elif zi <= -entry_z:
                    cur = 1.0
            elif abs(zi) >= stop_z or (cur > 0 and zi >= -exit_z) or (cur < 0 and zi <= exit_z):
                cur = 0.0
            pos.iloc[i] = cur
        dspread = s.diff().fillna(0.0)
        pnl = pos.shift(1).fillna(0.0).values * dspread.values
        return pnl

    return _pnl


def run_alpha_hunter(
    prices: dict[str, pd.Series],
    *,
    adf_threshold: float = 0.05,
    min_obs: int = 60,
    half_life_max_days: float = 30.0,
    half_life_min_days: float = 0.05,
    oos_sharpe_floor: float = 0.5,
    perm_oos_sharpe_threshold: float = 1.0,
    perm_p_threshold: float = 0.10,
    perm_n_iters: int = 200,
    backtest_window: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    max_pairs: int | None = None,
    seed: int = 42,
) -> AlphaHunterReport:
    """Run the full alpha-hunting gauntlet on every pair in ``prices``.

    Args:
        prices: factor_id → daily series (UTC index, probability or price).
        adf_threshold: ADF p-value filter for cointegration. Survivors go to backtest.
        min_obs: minimum overlapping observations to even attempt a pair.
        half_life_max_days: drop pairs whose AR(1) half-life is too slow.
        half_life_min_days: drop pairs with implausibly fast (~zero) half-life.
        oos_sharpe_floor: drop pairs whose OOS Sharpe is below this.
        perm_oos_sharpe_threshold: only run permutation test for pairs above this.
        perm_p_threshold: pairs with perm_p ≤ threshold get the REAL_ALPHA label.
        perm_n_iters: permutation iterations.
        backtest_window: rolling z-score window.
        entry_z / exit_z / stop_z: pairs strategy thresholds.
        max_pairs: optional cap on number of pairs evaluated (debug / smoke).
        seed: RNG seed for permutation reproducibility.

    Returns:
        :class:`AlphaHunterReport`. ``hits`` is sorted with REAL_ALPHA first,
        then by OOS Sharpe descending.
    """
    t0 = time.perf_counter()
    factor_ids = sorted(prices.keys())
    pairs = list(combinations(factor_ids, 2))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    hits: list[AlphaHit] = []
    n_passed_adf = 0
    n_perm_tested = 0
    n_real_alpha = 0

    pnl_fn = _zscore_pnl_factory(backtest_window, entry_z, exit_z, stop_z)

    for a_id, b_id in pairs:
        pair_t0 = time.perf_counter()
        a, b = prices[a_id].dropna(), prices[b_id].dropna()
        # Quick overlap pre-check before alignment cost.
        if len(a) < min_obs or len(b) < min_obs:
            continue

        try:
            cr = engle_granger(a, b)
        except Exception as e:
            logger.debug("EG failed on (%s, %s): %s", a_id, b_id, e)
            continue

        if cr.n_obs < min_obs:
            continue
        if not np.isfinite(cr.adf_pvalue) or cr.adf_pvalue > adf_threshold:
            continue
        if cr.half_life_days is None:
            continue
        if cr.half_life_days < half_life_min_days or cr.half_life_days > half_life_max_days:
            continue
        n_passed_adf += 1

        try:
            bt = pairs_backtest(
                cr.spread,
                window=backtest_window,
                entry_z=entry_z,
                exit_z=exit_z,
                stop_z=stop_z,
            )
        except Exception as e:
            logger.debug("backtest failed on (%s, %s): %s", a_id, b_id, e)
            continue

        oos_sh = float(bt.sharpe_oos) if bt.sharpe_oos is not None else float("nan")
        full_sh = float(bt.sharpe)

        if not np.isfinite(oos_sh) or oos_sh < oos_sharpe_floor:
            continue

        # Permutation only for pairs that look genuinely promising.
        perm_p: float | None = None
        perm_real: float | None = None
        verdict = "promising"
        if oos_sh >= perm_oos_sharpe_threshold:
            try:
                pr = permutation_sharpe_test(
                    cr.spread,
                    pnl_strategy_fn=pnl_fn,
                    n_iters=perm_n_iters,
                    seed=seed,
                )
                perm_p = float(pr.p_value)
                perm_real = float(pr.real_sharpe)
                n_perm_tested += 1
                if perm_p <= perm_p_threshold:
                    verdict = "REAL_ALPHA"
                    n_real_alpha += 1
                else:
                    verdict = "marginal"
            except Exception as e:
                logger.debug("permutation failed on (%s, %s): %s", a_id, b_id, e)

        runtime_ms = (time.perf_counter() - pair_t0) * 1000.0
        hits.append(
            AlphaHit(
                a_id=a_id,
                b_id=b_id,
                n_obs=cr.n_obs,
                adf_pvalue=float(cr.adf_pvalue),
                half_life_days=cr.half_life_days,
                beta_hedge=float(cr.beta_hedge),
                oos_sharpe=oos_sh,
                full_sharpe=full_sh,
                perm_p=perm_p,
                perm_real_sharpe=perm_real,
                verdict=verdict,
                runtime_ms=runtime_ms,
            )
        )

    def _rank_key(h: AlphaHit) -> tuple[int, float]:
        rank_class = {"REAL_ALPHA": 0, "marginal": 1, "promising": 2}.get(h.verdict, 3)
        return (rank_class, -h.oos_sharpe)

    hits.sort(key=_rank_key)
    return AlphaHunterReport(
        n_factors=len(factor_ids),
        n_pairs_total=len(pairs),
        n_pairs_passed_adf=n_passed_adf,
        n_pairs_perm_tested=n_perm_tested,
        n_real_alpha=n_real_alpha,
        runtime_seconds=time.perf_counter() - t0,
        hits=hits,
    )


__all__ = [
    "AlphaHit",
    "AlphaHunterReport",
    "run_alpha_hunter",
]

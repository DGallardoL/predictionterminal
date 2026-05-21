"""Inefficiency scanner: cartesian-pair leaderboards across the catalog.

Three independent scoring tracks, each emits a ranked leaderboard. The
scores are NOT commensurable across tracks — surface them in separate
tables, not a unified ranking.

Tracks:

1.  **Implication** (auto-discover candidate pairs from factor IDs).
    For markets organised in monotone strike families (e.g.
    ``oil_above_115`` / ``oil_above_150`` / ``oil_above_175``), the higher
    strike must have ≤ probability than the lower strike. Surface
    persistent violations (``n_violations ≥ 5``).

2.  **Conditional anomaly** (cartesian on the catalog or theme subset).
    Pairs where ``|β| ≥ β_threshold``, HAC-CI excludes 0, and
    ``R² ≥ R²_threshold``. Penalise obvious-causal pairs (high token
    overlap on the IDs) so genuine cross-theme surprises rise.

3.  **Cointegration** (Engle-Granger ADF p < 0.05 with half-life ≤ 60d).
    These are the pairs the pairs-trading backtester should score next.

Runtime: 145 factors → ~10k pairs. Daily-cached factor histories make
the per-pair work O(milliseconds for implication / Fréchet, ~50 ms for
conditional regression, ~30 ms for ADF). A theme-restricted scan
typically completes in 5–15 seconds.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd

from pfm.cointegration import engle_granger
from pfm.factors import FactorConfig
from pfm.strategies import conditional_regression, implication_test

logger = logging.getLogger(__name__)


ScanMode = Literal["implication", "conditional", "cointegration", "all"]


@dataclass(frozen=True)
class ScanHit:
    """One row in a scanner leaderboard."""

    kind: str  # "implication" | "conditional" | "cointegration"
    a_id: str
    b_id: str
    score: float  # higher = more interesting
    n_obs: int
    summary: str  # one-line human-readable explanation
    # Track-specific fields (NaN/empty when not applicable):
    n_violations: int = 0
    max_gap: float = float("nan")
    beta: float = float("nan")
    beta_ci_lo: float = float("nan")
    beta_ci_hi: float = float("nan")
    r_squared: float = float("nan")
    adf_pvalue: float = float("nan")
    half_life_days: float | None = None
    surprise: float = float("nan")


@dataclass(frozen=True)
class ScanReport:
    """Full output of :func:`run_scan`."""

    mode: ScanMode
    n_pairs_evaluated: int
    n_factors_scanned: int
    runtime_seconds: float
    implication: list[ScanHit]
    conditional: list[ScanHit]
    cointegration: list[ScanHit]


# ─────────────────────── implication-pair discovery ───────────────────


# Recognise common suffix patterns in factor IDs that encode a *strike*
# or *count* parameter. Each regex captures (a) a "family" key and (b)
# the magnitude of the threshold so we can sort within the family.
_STRIKE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # oil_above_115 / oil_above_150 / oil_below_70
    ("oil", re.compile(r"^oil_(above|below)_(\d+)(?:_.*)?$")),
    # btc_reach_100k / btc_reach_125k / btc_reach_150k
    ("btc_reach", re.compile(r"^btc_reach_(\d+)k(?:_.*)?$")),
    # btc_dip_75k_jun / btc_dip_50k_jun
    ("btc_dip", re.compile(r"^btc_dip_(\d+)k(?:_.*)?$")),
    # gold_5500_jun / gold_4000_jun
    ("gold", re.compile(r"^gold_(\d+)(?:_.*)?$")),
    # inflation_above_4_2026 / inflation_above_5_2026
    ("inflation", re.compile(r"^inflation_above_(\d+)(?:_.*)?$")),
    # fed_cuts_3_2026 / fed_cuts_4_2026 (more cuts ⇒ subset of "≥3 cuts")
    ("fed_cuts", re.compile(r"^fed_cuts_(\d+)(?:_.*)?$")),
    # k_cpi_above_4_27 (Kalshi)
    ("kalshi_cpi", re.compile(r"^k_cpi_above_(\d+)(?:_.*)?$")),
)


def _strike_family(fid: str) -> tuple[str, float] | None:
    """Return ``(family, magnitude)`` if the id matches a known strike pattern."""
    for fam, pat in _STRIKE_PATTERNS:
        m = pat.match(fid)
        if m:
            # All strike-magnitude captures live in the last group.
            mag = float(m.group(m.lastindex)) if m.lastindex else 0.0
            return (fam, mag)
    return None


def discover_implication_pairs(
    factor_ids: list[str],
) -> list[tuple[str, str]]:
    """Discover (more-specific, broader) pairs from strike families.

    For each strike family, sort by magnitude. Within an "above" family
    (e.g. ``oil_above_K``), a *higher* K is a stricter event ⇒ lower
    probability. So we emit pairs ``(higher_K, lower_K)`` and expect
    ``P(higher_K) ≤ P(lower_K)``. For "below" families the implication
    is reversed.

    Returns:
        list of ``(antecedent_id, consequent_id)`` tuples where
        ``antecedent ⇒ consequent`` is the claimed implication.
    """
    families: dict[str, list[tuple[float, str]]] = {}
    for fid in factor_ids:
        info = _strike_family(fid)
        if info is None:
            continue
        fam, mag = info
        # "below" families flip: lower K ⇒ stricter ⇒ smaller probability.
        if "below" in fid or "dip" in fid:
            fam = fam + "_below"
            # Lower magnitude = stricter event.
            mag = -mag
        families.setdefault(fam, []).append((mag, fid))

    pairs: list[tuple[str, str]] = []
    for items in families.values():
        if len(items) < 2:
            continue
        items.sort(key=lambda x: x[0])  # ascending magnitude
        # Within an ascending family, item[k] is stricter than item[k-1].
        # So the antecedent (more specific) is the higher-mag entry.
        # Emit all upper-triangle pairs.
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                anc = items[j][1]  # stricter
                con = items[i][1]  # broader
                pairs.append((anc, con))
    return pairs


# ─────────────────────── token-overlap surprise ───────────────────────


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(fid: str) -> set[str]:
    return set(_TOKEN_RE.findall(fid.lower()))


def _surprise(a_id: str, b_id: str) -> float:
    """Surprise score = 1 − jaccard(tokens(a), tokens(b)).

    Same-theme strike families share many tokens (oil, above, jun) and
    score low; cross-theme pairs (e.g. ``china_invade_taiwan_2026`` vs
    ``nvda_largest_apr``) score high.
    """
    ta, tb = _tokens(a_id), _tokens(b_id)
    if not ta and not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return 1.0 - (inter / union if union else 0.0)


# ─────────────────────── individual track scorers ─────────────────────


def _score_implication(
    a_id: str,
    b_id: str,
    p_a: pd.Series,
    p_b: pd.Series,
    *,
    tolerance: float,
    n_violations_min: int,
) -> ScanHit | None:
    """Run the implication test and return a hit if violations exceed
    ``n_violations_min``."""
    res = implication_test(p_a, p_b, tolerance=tolerance)
    n_v = len(res.violation_dates)
    if n_v < n_violations_min:
        return None
    if res.n_obs == 0 or not np.isfinite(res.max_gap):
        return None
    score = (n_v / max(res.n_obs, 1)) * min(max(res.max_gap, 0.0) / 0.10, 1.0)
    return ScanHit(
        kind="implication",
        a_id=a_id,
        b_id=b_id,
        score=float(score),
        n_obs=res.n_obs,
        n_violations=n_v,
        max_gap=float(res.max_gap),
        summary=f"P({a_id}) > P({b_id}) on {n_v}/{res.n_obs} days "
        f"(max gap {res.max_gap:.3f}, verdict={res.verdict})",
    )


def _score_conditional(
    a_id: str,
    b_id: str,
    p_a: pd.Series,
    p_b: pd.Series,
    *,
    beta_min: float,
    r2_min: float,
) -> ScanHit | None:
    """Run conditional regression and return a hit if |β| above
    threshold AND CI excludes 0 AND R² above threshold."""
    try:
        res = conditional_regression(p_a, p_b, hac_lag=5)
    except ValueError:
        return None
    if res.r_squared < r2_min:
        return None
    if abs(res.beta) < beta_min:
        return None
    ci_excludes = res.beta_ci_lo > 0.0 or res.beta_ci_hi < 0.0
    if not ci_excludes:
        return None
    surprise = _surprise(a_id, b_id)
    score = abs(res.beta) * surprise
    return ScanHit(
        kind="conditional",
        a_id=a_id,
        b_id=b_id,
        score=float(score),
        n_obs=res.n_obs,
        beta=float(res.beta),
        beta_ci_lo=float(res.beta_ci_lo),
        beta_ci_hi=float(res.beta_ci_hi),
        r_squared=float(res.r_squared),
        surprise=float(surprise),
        summary=(
            f"β={res.beta:+.3f} CI[{res.beta_ci_lo:+.2f}, {res.beta_ci_hi:+.2f}] "
            f"R²={res.r_squared:.2f} surprise={surprise:.2f}"
        ),
    )


def _score_cointegration(
    a_id: str,
    b_id: str,
    p_a: pd.Series,
    p_b: pd.Series,
    *,
    adf_max_p: float,
    half_life_max: float,
) -> ScanHit | None:
    res = engle_granger(p_a, p_b)
    if res.verdict != "cointegrated":
        return None
    if res.adf_pvalue > adf_max_p:
        return None
    if res.half_life_days is None or res.half_life_days > half_life_max:
        return None
    # Score: lower p-value AND shorter half-life are better.
    p_part = 1.0 - res.adf_pvalue / adf_max_p
    hl_part = 1.0 - res.half_life_days / half_life_max
    score = float(p_part * hl_part)
    return ScanHit(
        kind="cointegration",
        a_id=a_id,
        b_id=b_id,
        score=score,
        n_obs=res.n_obs,
        adf_pvalue=float(res.adf_pvalue),
        half_life_days=float(res.half_life_days),
        beta=float(res.beta_hedge),
        summary=(
            f"ADF p={res.adf_pvalue:.3f} half-life={res.half_life_days:.1f}d "
            f"β_hedge={res.beta_hedge:+.3f}"
        ),
    )


# ─────────────────────────── orchestrator ─────────────────────────────


def run_scan(
    factors: dict[str, FactorConfig],
    fetch_prices: Callable[[FactorConfig], pd.Series],
    *,
    mode: ScanMode = "all",
    theme: str | None = None,
    factor_ids: list[str] | None = None,
    max_pairs: int = 1000,
    n_obs_min: int = 30,
    # Implication thresholds:
    impl_tolerance: float = 0.02,
    impl_n_violations_min: int = 5,
    # Conditional thresholds:
    cond_beta_min: float = 0.30,
    cond_r2_min: float = 0.10,
    # Cointegration thresholds:
    coint_adf_max_p: float = 0.05,
    coint_half_life_max: float = 60.0,
    top_k_per_track: int = 25,
) -> ScanReport:
    """Run the inefficiency scanner.

    Args:
        factors: full id → FactorConfig map (typically ``app.state.factors``).
        fetch_prices: callable that returns a probability series for a
            single factor. Caller is responsible for caching this — the
            scanner just calls it once per factor it needs.
        mode: which tracks to run. ``"all"`` runs all three.
        theme: filter to one theme; ``None`` means all themes.
        factor_ids: explicit override of which factors to scan; if given,
            ``theme`` is ignored. Useful for "scan these 20 factors only".
        max_pairs: safety cap on (cartesian) pairs per conditional /
            cointegration track.
        n_obs_min: drop pairs with fewer aligned observations.
        ``*_threshold``: see the per-track ``_score_*`` functions.
        top_k_per_track: keep at most this many results per track.

    Returns:
        :class:`ScanReport`.
    """
    t0 = time.perf_counter()

    if factor_ids is not None:
        active = [factors[fid] for fid in factor_ids if fid in factors]
    elif theme:
        active = [fc for fc in factors.values() if fc.theme == theme]
    else:
        active = list(factors.values())
    n_factors = len(active)

    # Pull all needed series once.
    series_cache: dict[str, pd.Series] = {}
    for fc in active:
        try:
            s = fetch_prices(fc)
        except Exception as e:
            logger.warning("scanner: skipping %s — fetch failed (%s)", fc.id, e)
            continue
        if s is not None and not s.empty:
            series_cache[fc.id] = s

    impl_hits: list[ScanHit] = []
    cond_hits: list[ScanHit] = []
    coint_hits: list[ScanHit] = []

    # ── implication track ────────────────────────────────────────────
    if mode in ("implication", "all"):
        candidate_pairs = discover_implication_pairs(list(series_cache.keys()))
        for anc, con in candidate_pairs[: max_pairs * 2]:  # 2x because cheap
            sa = series_cache.get(anc)
            sb = series_cache.get(con)
            if sa is None or sb is None:
                continue
            n_obs = pd.concat({"a": sa, "b": sb}, axis=1).dropna().shape[0]
            if n_obs < n_obs_min:
                continue
            hit = _score_implication(
                anc,
                con,
                sa,
                sb,
                tolerance=impl_tolerance,
                n_violations_min=impl_n_violations_min,
            )
            if hit is not None:
                impl_hits.append(hit)

    # ── conditional + cointegration: cartesian over series_cache ─────
    # Threaded: per-pair work is statsmodels OLS / ADF — both release the
    # GIL on the heavy numpy/scipy hops, so a thread pool gets a real
    # parallel speed-up on the cartesian.
    if mode in ("conditional", "cointegration", "all"):
        ids = list(series_cache.keys())
        all_pairs = list(combinations(ids, 2))
        pairs_to_eval = all_pairs[:max_pairs] if len(all_pairs) > max_pairs else all_pairs

        def _process_pair(pair: tuple[str, str]) -> tuple[ScanHit | None, ScanHit | None]:
            a_id, b_id = pair
            sa = series_cache[a_id]
            sb = series_cache[b_id]
            joined = pd.concat({"a": sa, "b": sb}, axis=1).dropna()
            if len(joined) < n_obs_min:
                return None, None
            hit_cond: ScanHit | None = None
            hit_coint: ScanHit | None = None
            if mode in ("conditional", "all"):
                hit_cond = _score_conditional(
                    a_id,
                    b_id,
                    sa,
                    sb,
                    beta_min=cond_beta_min,
                    r2_min=cond_r2_min,
                )
            if mode in ("cointegration", "all"):
                hit_coint = _score_cointegration(
                    a_id,
                    b_id,
                    sa,
                    sb,
                    adf_max_p=coint_adf_max_p,
                    half_life_max=coint_half_life_max,
                )
            return hit_cond, hit_coint

        # Use cpu_count threads, capped at 8 (statsmodels is GIL-friendly
        # but not infinitely scalable).
        n_workers = min(8, max(2, (os.cpu_count() or 4)))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for hit_cond, hit_coint in ex.map(_process_pair, pairs_to_eval):
                if hit_cond is not None:
                    cond_hits.append(hit_cond)
                if hit_coint is not None:
                    coint_hits.append(hit_coint)

    # ── sort + truncate per track ────────────────────────────────────
    impl_hits.sort(key=lambda h: h.score, reverse=True)
    cond_hits.sort(key=lambda h: h.score, reverse=True)
    coint_hits.sort(key=lambda h: h.score, reverse=True)
    runtime = time.perf_counter() - t0

    return ScanReport(
        mode=mode,
        n_pairs_evaluated=len(impl_hits) + len(cond_hits) + len(coint_hits),
        n_factors_scanned=n_factors,
        runtime_seconds=runtime,
        implication=impl_hits[:top_k_per_track],
        conditional=cond_hits[:top_k_per_track],
        cointegration=coint_hits[:top_k_per_track],
    )


__all__ = [
    "ScanHit",
    "ScanMode",
    "ScanReport",
    "discover_implication_pairs",
    "run_scan",
]

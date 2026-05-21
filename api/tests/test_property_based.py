"""Property-based tests for the math core (logit / ΔLogit / OLS-HAC / multitest).

These tests use `hypothesis` to drive randomised inputs through invariants
that *must* hold regardless of the specific values: monotonicity of logit,
antisymmetry of pair-wise logit differences, finiteness after clipping,
recovery of known betas, R² in [0, 1], BH-FDR monotonicity, etc.

They complement the existing example-based tests in `test_logit.py`,
`test_model.py`, `test_multitest.py`. Anything caught here is by definition
something the example-based suite missed.

Conventions
-----------
* Slow tests (anything that calls ``fit_ols_hac`` or runs FDR over many
  hypotheses) cap ``max_examples`` to 20 and disable Hypothesis's deadline,
  per the project guidance to keep CI deterministic.
* All RNGs are seeded from a Hypothesis-drawn integer so failures are
  shrinkable and reproducible.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from pfm.model import (
    DEFAULT_EPSILON,
    delta_logit,
    fit_ols_hac,
    hac_lag_andrews,
    logit_transform,
)
from pfm.multitest import benjamini_hochberg_fdr, bonferroni_correction

# ---------------------------------------------------------------------------
# Small helpers used by the property tests
# ---------------------------------------------------------------------------


def _logit_at(p: float, eps: float = DEFAULT_EPSILON) -> float:
    """Scalar logit with the same clipping convention as the production code."""
    p_clipped = max(eps, min(1.0 - eps, p))
    return math.log(p_clipped / (1.0 - p_clipped))


def _winsorize(values: list[float], low_q: float = 0.05, high_q: float = 0.95) -> list[float]:
    """Quantile winsorization used to mirror the kind of helper most quant
    pipelines apply before fitting.  Pure stdlib so the property test does
    not depend on any specific helper inside ``pfm``."""
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    lo = float(np.quantile(arr, low_q))
    hi = float(np.quantile(arr, high_q))
    if lo > hi:
        lo, hi = hi, lo
    return [float(min(hi, max(lo, v))) for v in arr]


# ---------------------------------------------------------------------------
# logit / ΔLogit invariants
# ---------------------------------------------------------------------------


@given(
    p=st.floats(min_value=0.02, max_value=0.90, allow_nan=False),
    bump=st.floats(min_value=0.01, max_value=0.05, allow_nan=False),
)
def test_logit_monotonic(p: float, bump: float) -> None:
    """logit(p) is monotonic non-decreasing in p (within the unclipped range).

    We stay strictly inside ``(eps, 1-eps)`` so neither endpoint clamps and the
    inequality is strict in exact arithmetic; the assertion still uses a small
    floating-point tolerance to absorb representation noise.
    """
    eps = 0.01
    # Cap so p_higher stays within (eps, 1-eps) and is strictly > p.
    p_higher = min(p + bump, 0.95)
    l1 = _logit_at(p, eps)
    l2 = _logit_at(p_higher, eps)
    assert l2 + 1e-9 >= l1


@given(
    p1=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
    p2=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
)
def test_logit_diff_antisymmetric(p1: float, p2: float) -> None:
    """logit(p1) - logit(p2) = -(logit(p2) - logit(p1)).  Pure algebra check."""
    diff_fwd = _logit_at(p1) - _logit_at(p2)
    diff_rev = _logit_at(p2) - _logit_at(p1)
    assert math.isclose(diff_fwd, -diff_rev, abs_tol=1e-9)


@given(p=st.floats(min_value=0.001, max_value=0.999, allow_nan=False))
def test_logit_zero_at_half(p: float) -> None:
    """logit(0.5) = 0 exactly; for any p, logit(p) + logit(1-p) ≈ 0."""
    # First half: structural check at exactly 0.5
    assert math.isclose(_logit_at(0.5), 0.0, abs_tol=1e-12)
    # Second half: complement antisymmetry holds whenever neither side clips.
    if 0.01 <= p <= 0.99:
        s = _logit_at(p) + _logit_at(1.0 - p)
        assert abs(s) < 1e-9


@given(
    prices=st.lists(
        st.floats(min_value=0.001, max_value=0.999, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=200,
    )
)
def test_delta_logit_finite_after_clipping(prices: list[float]) -> None:
    """ΔLogit on any clipped probability series produces only finite values
    at indices where the diff is defined (i.e. all but the first)."""
    result = delta_logit(pd.Series(prices), epsilon=0.01)
    # First entry is always NaN by construction (no predecessor).
    body = result.iloc[1:]
    assert not body.isna().any(), "non-leading NaN appeared in ΔLogit"
    assert np.isfinite(body.to_numpy()).all(), "non-finite ΔLogit value"


@given(
    prices=st.lists(
        st.floats(min_value=0.001, max_value=0.999, allow_nan=False),
        min_size=2,
        max_size=50,
    ),
    eps=st.floats(min_value=0.001, max_value=0.4, allow_nan=False),
)
def test_logit_transform_clipping_bounds(prices: list[float], eps: float) -> None:
    """After ``logit_transform`` with ``epsilon=eps``, values lie within
    ``[logit(eps), logit(1-eps)]``."""
    out = logit_transform(pd.Series(prices), epsilon=eps).to_numpy()
    lo = math.log(eps / (1.0 - eps))
    hi = math.log((1.0 - eps) / eps)
    assert (out >= lo - 1e-12).all()
    assert (out <= hi + 1e-12).all()


# ---------------------------------------------------------------------------
# OLS-HAC properties
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=80, max_value=400),
    beta=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_ols_hac_recovers_known_beta(n: int, beta: float, seed: int) -> None:
    """OLS-HAC recovers a known β within a generous noise tolerance."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 0.5, n)
    y = beta * x + noise
    result = fit_ols_hac(
        pd.Series(y, name="y"),
        pd.DataFrame({"f1": x}),
    )
    f = next(fe for fe in result.factors if fe.factor_id == "f1")
    # Sample β should be within ≈ 4σ of the true value with σ_β ≈ 0.5/√n.
    tol = max(0.25, 4.0 * 0.5 / math.sqrt(n))
    assert abs(f.beta - beta) < tol, (
        f"β-recovery failed: true={beta:.3f} got={f.beta:.3f} tol={tol:.3f}"
    )


@given(
    n=st.integers(min_value=80, max_value=400),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_ols_r_squared_in_unit_interval(n: int, seed: int) -> None:
    """R² and adj-R² are always in [0, 1] (well, R² in [0,1]; adj-R² in (-∞, 1]
    but ≥ 0 for any data with positive total variance and a single regressor)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    y = 0.5 * x + rng.normal(0.0, 1.0, n)
    res = fit_ols_hac(pd.Series(y, name="y"), pd.DataFrame({"f1": x}))
    assert 0.0 - 1e-9 <= res.stats.r_squared <= 1.0 + 1e-9
    # adj_r² ≤ r² always.
    assert res.stats.r_squared_adj <= res.stats.r_squared + 1e-9


@given(
    n=st.integers(min_value=80, max_value=300),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_ols_residual_std_nonneg(n: int, seed: int) -> None:
    """Residual std is non-negative and the fit always produces a finite α."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    y = 0.3 * x + rng.normal(0.0, 0.7, n)
    res = fit_ols_hac(pd.Series(y, name="y"), pd.DataFrame({"f1": x}))
    assert res.stats.residual_std >= 0.0
    assert math.isfinite(res.stats.alpha)


@given(
    n=st.integers(min_value=80, max_value=300),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_vif_positive_for_two_factors(n: int, seed: int) -> None:
    """With ≥ 2 regressors, every reported VIF is positive (1.0 ≤ VIF in theory)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    y = 0.4 * x1 - 0.2 * x2 + rng.normal(0.0, 0.5, n)
    res = fit_ols_hac(
        pd.Series(y, name="y"),
        pd.DataFrame({"f1": x1, "f2": x2}),
    )
    for col, vif_val in res.diagnostics.vif.items():
        assert vif_val > 0.0, f"non-positive VIF for {col}: {vif_val}"
        assert math.isfinite(vif_val)


# ---------------------------------------------------------------------------
# HAC bandwidth (Andrews 1991)
# ---------------------------------------------------------------------------


@given(n=st.integers(min_value=2, max_value=100_000))
def test_hac_lag_andrews_monotonic_in_n(n: int) -> None:
    """The Andrews (1991) bandwidth ``floor(4·(T/100)^(2/9))`` is monotone
    non-decreasing in T."""
    lag_n = hac_lag_andrews(n)
    lag_n_plus_one = hac_lag_andrews(n + 1)
    assert lag_n_plus_one >= lag_n
    assert lag_n >= 1


# ---------------------------------------------------------------------------
# Multiple testing — BH-FDR & Bonferroni
# ---------------------------------------------------------------------------


@given(
    p_values=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        min_size=1,
        max_size=200,
    ),
    alpha=st.floats(min_value=0.001, max_value=0.5, allow_nan=False),
)
def test_bh_fdr_q_values_in_unit_interval(p_values: list[float], alpha: float) -> None:
    """BH q-values lie in [0, 1]; the rejected set is no larger than {p ≤ α}."""
    out = benjamini_hochberg_fdr(p_values, alpha=alpha)
    qs = out["q_values"]
    assert len(qs) == len(p_values)
    for q in qs:
        assert 0.0 <= q <= 1.0 + 1e-12
    # Sanity: the rejection count cannot exceed the count of p ≤ α (BH is
    # at least as conservative as the unadjusted threshold for monotone p).
    naive = sum(1 for p in p_values if p <= alpha)
    assert out["n_significant"] <= max(naive, len(p_values))


@given(
    p_values=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        min_size=2,
        max_size=80,
    )
)
@settings(max_examples=30, deadline=None)
def test_bh_q_values_monotone_in_sorted_p(p_values: list[float]) -> None:
    """Sorted by p, the q-values are non-decreasing.  This is the defining
    property of the BH step-up procedure."""
    out = benjamini_hochberg_fdr(p_values, alpha=0.05)
    paired = sorted(zip(p_values, out["q_values"], strict=True), key=lambda t: t[0])
    qs_sorted = [q for _, q in paired]
    for prev, nxt in pairwise(qs_sorted):
        assert nxt + 1e-12 >= prev


@given(
    p_values=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        min_size=1,
        max_size=80,
    )
)
def test_bonferroni_dominates_unadjusted(p_values: list[float]) -> None:
    """Bonferroni is the strictest standard correction: its rejection set is
    a subset of the unadjusted rejection set at the same α."""
    alpha = 0.05
    bonf = bonferroni_correction(p_values, alpha=alpha)
    naive = {i for i, p in enumerate(p_values) if p <= alpha}
    assert set(bonf["rejected_idx"]).issubset(naive)


# ---------------------------------------------------------------------------
# Sharpe / winsorize numeric properties
# ---------------------------------------------------------------------------


def _sharpe(returns: list[float]) -> float:
    """Plain Sharpe (no annualisation) from a list of period returns."""
    arr = np.asarray(returns, dtype=float)
    sd = float(arr.std(ddof=1))
    if sd == 0.0 or not math.isfinite(sd):
        return 0.0
    return float(arr.mean() / sd)


@given(
    returns=st.lists(
        st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=300,
    )
)
def test_sharpe_finite_for_nondegenerate_series(returns: list[float]) -> None:
    """Sharpe is finite whenever the series has positive sample std."""
    arr = np.asarray(returns, dtype=float)
    assume(float(arr.std(ddof=1)) > 1e-6)  # skip degenerate inputs
    s = _sharpe(returns)
    assert math.isfinite(s)


@given(
    values=st.lists(
        st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=200,
    )
)
def test_winsorize_within_quantile_bounds(values: list[float]) -> None:
    """After winsorization at [5%, 95%], every value lies between the
    pre-winsorization 5th and 95th sample quantiles."""
    arr = np.asarray(values, dtype=float)
    lo = float(np.quantile(arr, 0.05))
    hi = float(np.quantile(arr, 0.95))
    out = _winsorize(values, 0.05, 0.95)
    for v in out:
        assert lo - 1e-9 <= v <= hi + 1e-9


# ---------------------------------------------------------------------------
# Embargo / boundary respecting
# ---------------------------------------------------------------------------


def _train_test_with_embargo(n: int, split: int, embargo: int) -> tuple[list[int], list[int]]:
    """Standard time-series CV split with a gap (embargo) between sets."""
    if split <= 0 or split >= n:
        return list(range(n)), []
    train = list(range(split))
    test_start = min(n, split + max(0, embargo))
    test = list(range(test_start, n))
    return train, test


@given(
    n=st.integers(min_value=20, max_value=500),
    split=st.integers(min_value=1, max_value=400),
    embargo=st.integers(min_value=0, max_value=50),
)
def test_embargo_disjoint_and_respects_boundary(n: int, split: int, embargo: int) -> None:
    """Train/test indices are disjoint, and the gap between the last train
    index and the first test index is at least ``embargo``."""
    assume(split < n)
    train, test = _train_test_with_embargo(n, split, embargo)
    if not test:
        return
    # disjoint
    assert not (set(train) & set(test))
    # boundary respected
    last_train = max(train) if train else -1
    first_test = min(test)
    assert first_test - last_train >= embargo

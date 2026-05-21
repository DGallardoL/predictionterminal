"""Property-based tests for ``pfm.model.delta_logit`` and ``logit_transform``.

Hypothesis-driven. Each property runs with ``max_examples=200`` to give a
stronger guarantee than the hand-written tests in ``test_model.py``.

Properties exercised
--------------------
1. Finiteness for any clipped probability series.
2. Logit symmetry: ``logit(1 - p) == -logit(p)`` and the implied
   ``delta_logit(1 - p) == -delta_logit(p)``.
3. Clipping respected: prices below ``epsilon`` are clamped (Δlogit at the
   transition between two sub-``epsilon`` values is exactly 0).
4. ``epsilon = 0`` (and any ``epsilon`` outside ``(0, 0.5)``) raises
   ``ValueError`` — i.e. the guard in ``logit_transform`` propagates through
   ``delta_logit``.
5. Strict monotonicity of ``logit_transform`` on the open interval
   ``(epsilon, 1 - epsilon)``.
6. NaN-safety: NaN entries propagate; the remaining (finite) values still
   produce finite Δlogit at the appropriate positions.
7. Vector vs scalar inputs: scalars, length-1 arrays, plain lists, ``np.ndarray``
   and ``pd.Series`` are all accepted and return a ``pd.Series``.

The file is skipped at collection time if ``hypothesis`` is not importable,
so the rest of the suite still runs in minimal environments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

hypothesis = pytest.importorskip("hypothesis")

import itertools

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from pfm.model import DEFAULT_EPSILON, delta_logit, logit_transform

# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #

# Epsilon strictly in (0, 0.5) so logit_transform's guard accepts it.
_eps_strategy = st.floats(
    min_value=1e-6,
    max_value=0.49,
    allow_nan=False,
    allow_infinity=False,
)


def _prob_in_open_interval(eps: float) -> st.SearchStrategy[float]:
    """Probabilities strictly inside ``(eps, 1 - eps)`` so no clip binds."""
    pad = max(eps * 1e-3, 1e-9)
    return st.floats(
        min_value=eps + pad,
        max_value=1.0 - eps - pad,
        allow_nan=False,
        allow_infinity=False,
    )


# A probability series of length 2..50 in [0, 1] (clip may or may not bind).
def _prob_series(min_size: int = 2, max_size: int = 50) -> st.SearchStrategy[list[float]]:
    return st.lists(
        st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
            exclude_min=False,
            exclude_max=False,
        ),
        min_size=min_size,
        max_size=max_size,
    )


_settings = settings(
    max_examples=200,
    deadline=None,  # statsmodels / pandas are slow on first call; don't flake.
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


# --------------------------------------------------------------------------- #
# 1. Finiteness on any clipped-probability series
# --------------------------------------------------------------------------- #


@_settings
@given(prices=_prob_series(min_size=2, max_size=50), eps=_eps_strategy)
def test_delta_logit_finite_for_any_probability_input(prices: list[float], eps: float) -> None:
    """All non-leading entries are finite when input is in ``[0, 1]``.

    The first row is ``NaN`` by construction (no predecessor) so we skip it.
    Internal clipping to ``[eps, 1 - eps]`` must guarantee finite logits, hence
    finite first differences.
    """
    result = delta_logit(pd.Series(prices), epsilon=eps)
    tail = result.iloc[1:].to_numpy()
    assert np.all(np.isfinite(tail)), (
        f"non-finite Δlogit produced for prices={prices!r}, eps={eps!r}: {tail!r}"
    )


# --------------------------------------------------------------------------- #
# 2. Symmetry of logit and delta_logit
# --------------------------------------------------------------------------- #


@_settings
@given(p=st.floats(min_value=1e-6, max_value=0.49, allow_nan=False), eps=_eps_strategy)
def test_logit_symmetry_pointwise(p: float, eps: float) -> None:
    """``logit(1 - p) == -logit(p)`` after the same clip rule.

    We restrict ``p`` to the open interior ``(eps, 1 - eps)`` so neither side
    is clipped; otherwise the two clips could break exact antisymmetry. The
    test runs with floats whose magnitudes never exceed the clip bound by
    construction (``p <= 0.49`` and ``eps <= 0.49`` ⇒ ``p`` may be < eps; we
    skip those).
    """
    assume(eps < p < 1.0 - eps)
    left = float(logit_transform(pd.Series([p]), epsilon=eps).iloc[0])
    right = float(logit_transform(pd.Series([1.0 - p]), epsilon=eps).iloc[0])
    assert left == pytest.approx(-right, abs=1e-9), (
        f"logit not antisymmetric at p={p}, eps={eps}: left={left}, right={right}"
    )


@_settings
@given(eps=_eps_strategy, data=st.data())
def test_delta_logit_symmetry_on_unclipped_series(eps: float, data: st.DataObject) -> None:
    """``delta_logit(1 - p)`` should equal ``-delta_logit(p)`` element-wise.

    The first row of both is ``NaN`` so we compare the tail. We draw the
    series strictly inside ``(eps, 1 - eps)`` to avoid the clip flattening
    one side but not the other.
    """
    prices = data.draw(
        st.lists(_prob_in_open_interval(eps), min_size=2, max_size=30),
        label="prices",
    )
    p = pd.Series(prices)
    forward = delta_logit(p, epsilon=eps).iloc[1:].to_numpy()
    mirror = delta_logit(1.0 - p, epsilon=eps).iloc[1:].to_numpy()
    np.testing.assert_allclose(
        forward,
        -mirror,
        atol=1e-9,
        err_msg=f"Δlogit antisymmetry failed for prices={prices}, eps={eps}",
    )


# --------------------------------------------------------------------------- #
# 3. Clip respected — sub-epsilon values clamp to epsilon
# --------------------------------------------------------------------------- #


@_settings
@given(
    eps=st.floats(min_value=1e-3, max_value=0.4, allow_nan=False),
    low_a=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    low_b=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_clipping_below_epsilon_yields_zero_step(eps: float, low_a: float, low_b: float) -> None:
    """Two consecutive prices both ``< epsilon`` get clipped to ``epsilon``,
    so the Δlogit step between them is exactly 0.

    Mirror property holds above ``1 - epsilon``.
    """
    a = min(low_a, eps) * 0.5  # strictly below eps (handles low_a == eps edge)
    b = min(low_b, eps) * 0.5
    assume(a < eps and b < eps)  # belt-and-braces

    step = delta_logit(pd.Series([a, b]), epsilon=eps).iloc[1]
    assert step == 0.0, f"sub-eps Δlogit not 0 for a={a}, b={b}, eps={eps}: got {step}"

    # Upper-tail mirror.
    step_hi = delta_logit(pd.Series([1.0 - a, 1.0 - b]), epsilon=eps).iloc[1]
    assert step_hi == 0.0, f"super-(1-eps) Δlogit not 0 for a={a}, b={b}, eps={eps}: got {step_hi}"


# --------------------------------------------------------------------------- #
# 4. ``epsilon`` outside ``(0, 0.5)`` raises
# --------------------------------------------------------------------------- #


@_settings
@given(
    bad_eps=st.one_of(
        st.just(0.0),
        st.just(0.5),
        st.floats(min_value=-1e3, max_value=-1e-9, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.5 + 1e-9, max_value=1e3, allow_nan=False, allow_infinity=False),
    )
)
def test_epsilon_outside_valid_range_raises(bad_eps: float) -> None:
    """``epsilon <= 0`` or ``epsilon >= 0.5`` must raise ``ValueError``."""
    with pytest.raises(ValueError, match="epsilon"):
        delta_logit(pd.Series([0.3, 0.6, 0.4]), epsilon=bad_eps)


# --------------------------------------------------------------------------- #
# 5. Strict monotonicity of logit_transform
# --------------------------------------------------------------------------- #


@_settings
@given(eps=_eps_strategy, data=st.data())
def test_logit_strictly_monotonic_in_open_interval(eps: float, data: st.DataObject) -> None:
    """For strictly increasing ``p`` in ``(eps, 1 - eps)``, logit is strictly increasing.

    Draws a sorted list of distinct probabilities inside the open interval and
    checks the pairwise differences of ``logit_transform`` are positive.
    """
    raw = data.draw(
        st.lists(_prob_in_open_interval(eps), min_size=2, max_size=20, unique=True),
        label="probs",
    )
    probs = sorted(raw)
    # ``unique=True`` on floats is approximate AND ``b > a`` can hold at
    # sub-ULP distances where ``log(b/(1-b)) - log(a/(1-a))`` rounds to 0.
    # Require a real gap so logit's monotonicity is observable in float64.
    assume(all((b - a) > 1e-9 for a, b in itertools.pairwise(probs)))
    out = logit_transform(pd.Series(probs), epsilon=eps).to_numpy()
    diffs = np.diff(out)
    assert np.all(diffs > 0), (
        f"logit_transform not strictly increasing for probs={probs}, eps={eps}: diffs={diffs}"
    )


# --------------------------------------------------------------------------- #
# 6. NaN-safety
# --------------------------------------------------------------------------- #


@_settings
@given(
    prices=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=4,
        max_size=20,
    ),
    nan_mask=st.lists(st.booleans(), min_size=4, max_size=20),
    eps=_eps_strategy,
)
def test_nan_propagates_but_does_not_explode(
    prices: list[float], nan_mask: list[bool], eps: float
) -> None:
    """Injecting NaN entries must not raise and must not produce ``inf``.

    NaN at index ``i`` poisons the diff at ``i`` (input NaN) and at ``i + 1``
    (previous-row NaN); every other transition between two finite, in-range
    inputs must still be finite.
    """
    n = min(len(prices), len(nan_mask))
    arr = np.array(prices[:n], dtype=float)
    mask = np.array(nan_mask[:n], dtype=bool)
    # Guarantee at least two finite consecutive values somewhere so the test
    # has something to assert on.
    assume(n >= 4)
    assume(np.sum(~mask) >= 2)
    arr[mask] = np.nan

    result = delta_logit(pd.Series(arr), epsilon=eps).to_numpy()

    # No infinities anywhere — NaN is allowed, ±inf is not.
    assert not np.any(np.isinf(result)), (
        f"Δlogit produced inf for arr={arr.tolist()}, eps={eps}: {result.tolist()}"
    )

    # Every transition where both endpoints are finite must be finite.
    for i in range(1, n):
        if np.isfinite(arr[i]) and np.isfinite(arr[i - 1]):
            assert np.isfinite(result[i]), (
                f"finite-to-finite transition at i={i} produced "
                f"{result[i]} (arr={arr.tolist()}, eps={eps})"
            )


# --------------------------------------------------------------------------- #
# 7. Vector vs scalar input types
# --------------------------------------------------------------------------- #


@_settings
@given(p=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_scalar_input_returns_single_nan_series(p: float) -> None:
    """A scalar probability becomes a length-1 series; Δlogit is NaN (no predecessor)."""
    result = delta_logit(p)
    assert isinstance(result, pd.Series)
    assert len(result) == 1
    assert np.isnan(result.iloc[0])


@_settings
@given(
    prices=_prob_series(min_size=2, max_size=20),
    eps=_eps_strategy,
)
def test_input_types_produce_equivalent_output(prices: list[float], eps: float) -> None:
    """List, ``np.ndarray``, and ``pd.Series`` inputs yield equal Δlogit series."""
    from_list = delta_logit(prices, epsilon=eps)
    from_array = delta_logit(np.asarray(prices, dtype=float), epsilon=eps)
    from_series = delta_logit(pd.Series(prices), epsilon=eps)

    for other in (from_array, from_series):
        np.testing.assert_allclose(
            from_list.to_numpy(),
            other.to_numpy(),
            equal_nan=True,
            err_msg=f"input-type drift for prices={prices}, eps={eps}",
        )

    # All return a Series of identical length.
    assert isinstance(from_list, pd.Series)
    assert isinstance(from_array, pd.Series)
    assert isinstance(from_series, pd.Series)
    assert len(from_list) == len(prices)


# --------------------------------------------------------------------------- #
# Smoke: ensure the DEFAULT_EPSILON path still produces finite output.
# --------------------------------------------------------------------------- #


@_settings
@given(prices=_prob_series(min_size=3, max_size=30))
def test_default_epsilon_produces_finite_tail(prices: list[float]) -> None:
    """The default epsilon (0.01) path must satisfy the finiteness invariant."""
    result = delta_logit(pd.Series(prices))
    assert np.all(np.isfinite(result.iloc[1:].to_numpy())), (
        f"DEFAULT_EPSILON={DEFAULT_EPSILON} broke finiteness for {prices!r}"
    )

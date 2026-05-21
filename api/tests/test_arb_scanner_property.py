"""Hypothesis property-based tests for ``pfm.arb_scanner``.

Task T36. The user's brief sketched a hypothetical
``compute_arb(book_a, book_b, fees)`` API. The real public surface of
``api/src/pfm/arb_scanner.py`` is structured differently — there is no
"fees" parameter and prices are venue mid-prices (already in [0, 1]),
not full orderbooks. The closest analogues to the user's spec are:

  - :func:`pfm.arb_scanner._spread_record(pair, pm_price, kalshi_price,
    pm_vol, kalshi_vol, *, min_spread_pct, min_volume_usd)` —
    builds an arb record (or returns ``None``) from a paired snapshot of
    Polymarket vs Kalshi mid prices and 24h volumes. The ``min_*`` gates
    are the API's stand-in for "fees / cost-of-trade": raising them
    suppresses opportunities the same way a fee would erode an arb.

  - :func:`pfm.arb_scanner._max_pairwise_spread_pct(prices)` — returns
    ``(spread_pct, low_venue, high_venue)`` across an arbitrary set of
    per-venue prices, the multi-venue spread primitive used by
    :func:`find_4way_arb` / :func:`compute_4way_arbs`.

  - :func:`pfm.arb_scanner.compute_arb_spreads(matched_pairs, ...)` —
    high-level scanner wrapping ``_spread_record``; we drive it with
    monkey-patched fetchers (no network).

We therefore translate the user's seven properties to this real
surface:

  1. Non-negativity at zero gates: ``_spread_record`` with
     ``min_spread_pct=0`` and ``min_volume_usd=0`` either returns
     ``None`` (volume-filtered) or a record with ``spread_pct >= 0``.
  2. "Fees" (= ``min_spread_pct`` / ``min_volume_usd``) reduce the
     opportunity set: increasing either threshold cannot turn ``None``
     into a record, only the reverse. The underlying numeric
     ``spread_pct`` is independent of the gates by construction; we
     verify that as a separate property.
  3. Symmetry: swapping ``pm_price <-> kalshi_price`` (and the volume
     side) leaves ``spread_pct`` unchanged and flips ``direction``.
  4. Bounded: with PM and Kalshi prices in [0, 1], the resulting
     ``spread_pct`` is always in [0, 100]. For ``_max_pairwise_spread_pct``
     with prices in [0, 1], spread_pct is in [0, 100].
  5. Empty book: ``_max_pairwise_spread_pct({})`` returns
     ``(0.0, "", "")``; ``compute_arb_spreads([])`` returns ``[]``.
  6. One-sided / single-venue: ``_max_pairwise_spread_pct`` with one
     entry returns ``(0.0, "", "")`` (no arb).
  7. Monotonic in price gap: increasing ``|pm_price - kalshi_price|``
     never decreases the numeric ``spread_pct``.

Run with ::

    cd api && PYTHONPATH=src .venv/bin/python -m pytest \\
        tests/test_arb_scanner_property.py -q --noconftest
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `--noconftest` invocation: ensure src/ is on sys.path.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from pfm.arb_scanner import (
    _max_pairwise_spread_pct,
    _spread_record,
    compute_arb_spreads,
)

# Hypothesis strategies ------------------------------------------------------

# Prices live in (0, 1) by Polymarket/Kalshi convention.
price_strategy = st.floats(
    min_value=0.001,
    max_value=0.999,
    allow_nan=False,
    allow_infinity=False,
)

# 24h dollar volume; min_volume_usd defaults to 5_000 in production so we
# sample across that boundary.
volume_strategy = st.floats(
    min_value=0.0,
    max_value=1_000_000.0,
    allow_nan=False,
    allow_infinity=False,
)

# Pretend "fees" = min_spread_pct gate. Bounded the same way the router
# bounds the query parameter (`ge=0.0, le=100.0`).
gate_pct_strategy = st.floats(
    min_value=0.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
)

# Min-volume gate is just a positive dollar number.
gate_volume_strategy = st.floats(
    min_value=0.0,
    max_value=10_000_000.0,
    allow_nan=False,
    allow_infinity=False,
)

# A small dict of per-venue prices for the multi-venue primitive.
prices_dict_strategy = st.dictionaries(
    keys=st.sampled_from(["polymarket", "kalshi", "manifold", "predictit"]),
    values=price_strategy,
    min_size=0,
    max_size=4,
)


def _make_pair() -> dict[str, str]:
    return {"pm_slug": "fixture-pm", "kalshi_slug": "FIXTURE-KX", "label": "fixture"}


# ---------------------------------------------------------------------------
# Property 1: non-negativity at zero gates
# ---------------------------------------------------------------------------

SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@SETTINGS
@given(pm=price_strategy, kx=price_strategy, pm_vol=volume_strategy, k_vol=volume_strategy)
def test_spread_nonneg_at_zero_gates(pm: float, kx: float, pm_vol: float, k_vol: float) -> None:
    """Zero-gate spread is always >= 0 when a record is produced."""
    rec = _spread_record(
        _make_pair(),
        pm_price=pm,
        kalshi_price=kx,
        pm_vol=pm_vol,
        kalshi_vol=k_vol,
        min_spread_pct=0.0,
        min_volume_usd=0.0,
    )
    # With both gates at zero, only the volume floor (0.0) can drop the record.
    # _spread_record returns None if tradeable_size_usd (= min(vols)) < 0 → never.
    # So we always get a record back.
    assert rec is not None
    assert rec["spread_pct"] >= 0.0
    # Sanity: matches |pm-kx|*100 modulo the documented `round(_, 3)`.
    assert rec["spread_pct"] == pytest.approx(abs(pm - kx) * 100.0, abs=5e-4)


# ---------------------------------------------------------------------------
# Property 2: tighter gates ("fees") never enlarge the opportunity set
# ---------------------------------------------------------------------------


@SETTINGS
@given(
    pm=price_strategy,
    kx=price_strategy,
    pm_vol=volume_strategy,
    k_vol=volume_strategy,
    g_lo=gate_pct_strategy,
    g_hi=gate_pct_strategy,
    v_lo=gate_volume_strategy,
    v_hi=gate_volume_strategy,
)
def test_tighter_gates_never_admit_more(
    pm: float,
    kx: float,
    pm_vol: float,
    k_vol: float,
    g_lo: float,
    g_hi: float,
    v_lo: float,
    v_hi: float,
) -> None:
    """If a tighter gate (higher min_spread_pct OR higher min_volume_usd)
    accepts a snapshot, the looser gate must also accept it.

    This is the "fees reduce edge" analogue: raising the minimum-spread
    floor or the minimum-volume floor can only suppress records, never
    create new ones.
    """
    s_lo, s_hi = sorted([g_lo, g_hi])
    vol_lo, vol_hi = sorted([v_lo, v_hi])
    pair = _make_pair()

    loose = _spread_record(
        pair,
        pm,
        kx,
        pm_vol,
        k_vol,
        min_spread_pct=s_lo,
        min_volume_usd=vol_lo,
    )
    tight = _spread_record(
        pair,
        pm,
        kx,
        pm_vol,
        k_vol,
        min_spread_pct=s_hi,
        min_volume_usd=vol_hi,
    )
    # Whenever the tighter gate accepts, the looser one must also accept.
    if tight is not None:
        assert loose is not None
        # And the numeric spread is the same — it does not depend on gates.
        assert loose["spread_pct"] == pytest.approx(tight["spread_pct"], abs=1e-9)


# ---------------------------------------------------------------------------
# Property 3: symmetry (swap PM <-> Kalshi)
# ---------------------------------------------------------------------------


@SETTINGS
@given(pm=price_strategy, kx=price_strategy, pm_vol=volume_strategy, k_vol=volume_strategy)
def test_symmetry_swap_books(pm: float, kx: float, pm_vol: float, k_vol: float) -> None:
    """Swapping PM/Kalshi sides preserves |spread_pct| and flips direction
    (unless prices are equal)."""
    pair = _make_pair()
    a = _spread_record(
        pair,
        pm,
        kx,
        pm_vol,
        k_vol,
        min_spread_pct=0.0,
        min_volume_usd=0.0,
    )
    b = _spread_record(
        pair,
        kx,
        pm,
        k_vol,
        pm_vol,
        min_spread_pct=0.0,
        min_volume_usd=0.0,
    )
    assert a is not None and b is not None
    assert a["spread_pct"] == pytest.approx(b["spread_pct"], abs=1e-9)
    # tradeable_size_usd uses min() and is symmetric in pm_vol / k_vol.
    assert a["tradeable_size_usd"] == pytest.approx(b["tradeable_size_usd"], abs=1e-9)
    # Direction flips unless prices are equal.
    if pm != kx:
        assert a["direction"] != b["direction"]


# ---------------------------------------------------------------------------
# Property 4: bounded spread_pct in [0, 100]
# ---------------------------------------------------------------------------


@SETTINGS
@given(pm=price_strategy, kx=price_strategy, pm_vol=volume_strategy, k_vol=volume_strategy)
def test_spread_bounded_0_100(pm: float, kx: float, pm_vol: float, k_vol: float) -> None:
    """With prices in (0, 1), spread_pct lives in [0, 100]."""
    rec = _spread_record(
        _make_pair(),
        pm,
        kx,
        pm_vol,
        k_vol,
        min_spread_pct=0.0,
        min_volume_usd=0.0,
    )
    assert rec is not None
    assert 0.0 <= rec["spread_pct"] <= 100.0


@SETTINGS
@given(prices=prices_dict_strategy)
def test_max_pairwise_spread_bounded(prices: dict[str, float]) -> None:
    """``_max_pairwise_spread_pct`` is also in [0, 100] for prices in (0,1)."""
    spread, lo, hi = _max_pairwise_spread_pct(prices)
    assert 0.0 <= spread <= 100.0
    # Empty / single-leg short-circuit must return the documented sentinel.
    if len(prices) < 2:
        assert spread == 0.0
        assert lo == "" and hi == ""
    else:
        # The two returned venue names must be different keys in `prices`.
        assert lo in prices and hi in prices
        assert lo != hi or len(prices) == 1
        # Recompute by hand to be sure.
        manual = (max(prices.values()) - min(prices.values())) * 100.0
        assert spread == pytest.approx(manual, abs=1e-9)


# ---------------------------------------------------------------------------
# Property 5: empty book / empty match list
# ---------------------------------------------------------------------------


def test_max_pairwise_empty() -> None:
    """``_max_pairwise_spread_pct({})`` -> (0.0, '', '')."""
    s, lo, hi = _max_pairwise_spread_pct({})
    assert s == 0.0
    assert lo == ""
    assert hi == ""


def test_compute_arb_spreads_empty_input() -> None:
    """No matched pairs -> empty list (and we never touch the network)."""
    assert compute_arb_spreads([]) == []


# ---------------------------------------------------------------------------
# Property 6: one-sided / single-venue -> no arb
# ---------------------------------------------------------------------------


@SETTINGS
@given(
    venue=st.sampled_from(["polymarket", "kalshi", "manifold", "predictit"]), price=price_strategy
)
def test_single_venue_no_arb(venue: str, price: float) -> None:
    """One-leg book yields (0.0, '', '')."""
    s, lo, hi = _max_pairwise_spread_pct({venue: price})
    assert s == 0.0
    assert lo == "" and hi == ""


def test_one_sided_missing_pm_returns_none() -> None:
    """compute_arb_spreads filters out pairs whose PM mid is unavailable
    (mocked here via a monkey-patched ``_pm_mid`` -> None)."""
    from pfm import arb_scanner

    pair = {"pm_slug": "x", "kalshi_slug": "Y", "label": ""}
    saved_pm, saved_kx = arb_scanner._pm_mid, arb_scanner._kalshi_mid
    try:
        arb_scanner._pm_mid = lambda slug, http: (None, None)  # type: ignore[assignment]
        arb_scanner._kalshi_mid = lambda ticker, c: (0.5, 100_000.0)  # type: ignore[assignment]
        # The function itself constructs an httpx client; that's fine since
        # the mocks short-circuit before any network call.
        out = compute_arb_spreads([pair], min_spread_pct=0.0, min_volume_usd=0.0)
        assert out == []
    finally:
        arb_scanner._pm_mid = saved_pm
        arb_scanner._kalshi_mid = saved_kx


def test_one_sided_missing_kalshi_returns_none() -> None:
    """Symmetric: missing Kalshi side -> no record."""
    from pfm import arb_scanner

    pair = {"pm_slug": "x", "kalshi_slug": "Y", "label": ""}
    saved_pm, saved_kx = arb_scanner._pm_mid, arb_scanner._kalshi_mid
    try:
        arb_scanner._pm_mid = lambda slug, http: (0.5, 100_000.0)  # type: ignore[assignment]
        arb_scanner._kalshi_mid = lambda ticker, c: (None, None)  # type: ignore[assignment]
        out = compute_arb_spreads([pair], min_spread_pct=0.0, min_volume_usd=0.0)
        assert out == []
    finally:
        arb_scanner._pm_mid = saved_pm
        arb_scanner._kalshi_mid = saved_kx


# ---------------------------------------------------------------------------
# Property 7: monotonic in the absolute price gap
# ---------------------------------------------------------------------------


@SETTINGS
@given(
    pm=price_strategy,
    kx=price_strategy,
    bump=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
)
def test_monotonic_in_price_gap(pm: float, kx: float, bump: float) -> None:
    """If ``pm > kx`` and we bump PM further up (still <= 1), the spread
    can only grow or stay the same."""
    # Make sure pm > kx so bumping pm increases the gap monotonically.
    assume(pm > kx)
    pm_bumped = min(0.999, pm + bump)

    pair = _make_pair()
    base = _spread_record(
        pair,
        pm,
        kx,
        1_000_000.0,
        1_000_000.0,
        min_spread_pct=0.0,
        min_volume_usd=0.0,
    )
    bumped = _spread_record(
        pair,
        pm_bumped,
        kx,
        1_000_000.0,
        1_000_000.0,
        min_spread_pct=0.0,
        min_volume_usd=0.0,
    )
    assert base is not None and bumped is not None
    # Allow for `round(_, 3)` rounding noise in the stored spread_pct.
    assert bumped["spread_pct"] + 1e-3 >= base["spread_pct"]


# ---------------------------------------------------------------------------
# Extra: max pairwise spread also monotonic when we lift the highest leg
# ---------------------------------------------------------------------------


@SETTINGS
@given(
    prices=prices_dict_strategy,
    lift=st.floats(min_value=0.0, max_value=0.4, allow_nan=False, allow_infinity=False),
)
def test_max_pairwise_monotonic_lift(prices: dict[str, float], lift: float) -> None:
    """Lifting the highest leg (clamped <= 0.999) cannot shrink the spread."""
    assume(len(prices) >= 2)
    s0, _, hi = _max_pairwise_spread_pct(prices)
    lifted = dict(prices)
    lifted[hi] = min(0.999, lifted[hi] + lift)
    s1, _, _ = _max_pairwise_spread_pct(lifted)
    assert s1 + 1e-9 >= s0


# ---------------------------------------------------------------------------
# Extra: volume floor is monotonic in the wrong direction (sanity)
# ---------------------------------------------------------------------------


@SETTINGS
@given(pm=price_strategy, kx=price_strategy, vol=volume_strategy, floor=gate_volume_strategy)
def test_volume_floor_filters_when_below(
    pm: float,
    kx: float,
    vol: float,
    floor: float,
) -> None:
    """If both volumes are equal and below the floor, _spread_record drops it."""
    assume(vol < floor)
    rec = _spread_record(
        _make_pair(),
        pm,
        kx,
        vol,
        vol,
        min_spread_pct=0.0,
        min_volume_usd=floor,
    )
    assert rec is None

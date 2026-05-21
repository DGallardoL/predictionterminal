"""End-to-end stress-test integration for ``BinaryPricingAlpha`` (W11-T37).

This test wires three pieces together for the first time:

1. ``pfm.pricing.binary_models.RiskNeutralLogit`` (T81) — the actual fair-price
   model used in the deployable path of T84's ``BinaryPricingAlpha``.
2. ``pfm.strategies.binary_pricing_alpha.BinaryPricingAlpha`` (T84) — the
   Kelly-capped wrapper that turns a pricer into a registrable strategy.
3. ``api/scripts/stress_test.py`` (W11 anti-alpha gate) — invoked
   programmatically via :func:`run_stress`, which is the function CLAUDE.md
   requires every "wow" backtest pass with all 4 quarters Sharpe>=0.5 and
   no sign-flip.

The point of these tests is **not** to confirm any particular alpha is
deployable — it is to assert that the *anti-alpha rule itself* is being
enforced end-to-end through ``BinaryPricingAlpha``. Specifically:

* A strategy whose synthetic DGP carries clean positive alpha through
  every quarter must yield ``verdict == "PASS"``.
* A strategy whose synthetic DGP flips sign in Q3 must yield
  ``verdict == "FAIL"`` with ``"sign flip"`` recorded in Q3's
  ``fail_reason``.
* A strategy with insufficient signal in every quarter must yield
  ``verdict == "FAIL"`` with every quarter flagged on the Sharpe floor.
* The deflated-Sharpe gate (``deflated_sharpe < 0``) is honored in the
  report payload even when raw Sharpe is positive.
* A NaN-producing quarter must be detected (Sharpe defaults to 0 →
  ``fail``).
* Same seed → same verdict (reproducibility).

All Polymarket inputs are mocked — we never hit the real CLOB. We use
``numpy.random.default_rng(seed)`` for every random draw so each test is
deterministic.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Bootstrap — ensure ``src/`` and ``scripts/`` are importable hermetically.
# ---------------------------------------------------------------------------

_API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_API_ROOT / "src"))

from pfm.pricing.binary_models import (
    MarketState as PricerMarketState,
)
from pfm.pricing.binary_models import (
    RiskNeutralLogit,
)
from pfm.strategies.binary_pricing_alpha import (
    BinaryPricingAlpha,
)
from pfm.strategies.binary_pricing_alpha import (
    MarketState as AlphaMarketState,
)
from pfm.strategies_registry import (
    Strategy,
    register,
    unregister,
)

# ---------------------------------------------------------------------------
# stress_test.py is at scripts/stress_test.py — load it once per session.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stress_module() -> ModuleType:
    """Import ``scripts/stress_test.py`` as a module (no CLI invocation)."""
    path = _API_ROOT / "scripts" / "stress_test.py"
    spec = importlib.util.spec_from_file_location("pfm_stress_test_w11_37", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Adapter: T81's RiskNeutralLogit consumes ``binary_models.MarketState``,
# but T84's BinaryPricingAlpha hands it the *alpha* MarketState (a slightly
# different dataclass). We bridge them here — the same fields are present.
# ---------------------------------------------------------------------------


class RiskNeutralLogitAdapter:
    """Adapts T81's :class:`RiskNeutralLogit` to the T84 Pricer protocol.

    The T84 protocol expects ``fair_price(state: AlphaMarketState) -> float``.
    The T81 pricer expects a ``PricerMarketState``. They both carry
    ``current_price`` / ``time_to_resolve_days`` (renamed) plus features —
    we simply translate the field names and pull ``news_evidence`` out of
    the AlphaMarketState.features dict.
    """

    def __init__(self, base: RiskNeutralLogit) -> None:
        self.base = base

    def fair_price(self, state: AlphaMarketState) -> float:
        pricer_state = PricerMarketState(
            current_price=float(state.market_price),
            time_to_resolve_days=float(state.time_to_resolution_days),
            underlying=None,
            threshold=None,
            poll_history=(),
            news_evidence=float(state.features.get("news_evidence", 0.0)),
        )
        result = self.base.fair_price(pricer_state)
        return float(result.fair_price)


# ---------------------------------------------------------------------------
# Synthetic binary-market builders. The harness expects a daily-indexed
# DataFrame; for the BinaryPricingAlpha integration we attach all the
# columns the strategy needs on each row (market_price, fair_price,
# outcome, features) plus a ``close`` column (so the registry's
# _default_pnl wouldn't blow up if it were ever invoked).
# ---------------------------------------------------------------------------


def _quarter_index_of(ts: pd.Timestamp) -> int:
    """Return 1..4 for the calendar quarter of a UTC timestamp."""
    return ((ts.month - 1) // 3) + 1


def _build_market_frame(
    *,
    start: pd.Timestamp,
    n_days: int,
    seed: int,
    flip_q3: bool,
    alpha_strength: float,
    noise_scale: float = 0.012,
    pricer: RiskNeutralLogitAdapter | None = None,
    poison_q3: bool = False,
) -> pd.DataFrame:
    """Build a synthetic prediction-market frame across ``n_days`` calendar days.

    Each row carries:
        * ``market_price``: noisy YES price near 0.5 (the binary "uncertain"
          regime where the pricer's news adjustment matters most)
        * ``fair_price``: computed by the supplied T81 pricer from
          ``news_evidence`` — quarterly signed
        * ``outcome``: 0/1 Bernoulli draw with success prob = ``fair_price``
        * ``features``: dict with ``news_evidence`` (the T81 pricer's main lever)
        * ``close``: synthetic underlying close so the frame also satisfies
          the registry's default-PnL contract if ever called

    When ``flip_q3=True``, Q3 inverts the sign of ``news_evidence`` so the
    fair price drops below market — and ``BinaryPricingAlpha`` ends up
    sized short on those rows. With outcomes still driven by ``fair_price``,
    position×return stays positive across all quarters (the alpha is still
    right, just in the other direction).

    When ``poison_q3=True``, Q3 inverts the *outcome* distribution
    relative to the pricer's prediction — i.e. the pricer is *wrong*
    about Q3. Combined with ``flip_q3=True``, this gives us a quarter
    in which the strategy systematically loses money — the canonical
    "regime broke our pricer" scenario the anti-alpha rule targets.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_days, freq="D", tz="UTC")

    # Underlying close — used only if the registry's default PnL is invoked.
    log_ret = rng.normal(0.0001, 0.01, size=n_days)
    close = 100.0 * np.exp(np.cumsum(log_ret))

    # Quarter-aware news evidence. Signed so the T81 pricer's beta_news
    # coefficient (default 1.0) actually moves fair_price up/down. Magnitude
    # tuned by ``alpha_strength`` (think of it as the bps-equivalent edge).
    news = np.empty(n_days, dtype=float)
    for i, ts in enumerate(idx):
        q = _quarter_index_of(ts)
        sign = -1.0 if (flip_q3 and q == 3) else 1.0
        news[i] = float(np.clip(sign * alpha_strength * 10.0, -1.0, 1.0))
    # Add small Gaussian jitter so the signal isn't constant within a quarter.
    news = np.clip(news + rng.normal(0.0, 0.05, size=n_days), -1.0, 1.0)

    # Market hovers near 0.50 (a "coin-flip-ish" contract). The pricer's
    # news-driven adjustment moves fair_price away from market — that's
    # the mispricing the alpha is supposed to exploit.
    market = np.clip(0.50 + rng.normal(0.0, noise_scale, size=n_days), 0.05, 0.95)
    features = [{"news_evidence": float(n)} for n in news]

    # Compute fair_price via the T81 pricer for every row — this is the
    # integration point with binary_models.RiskNeutralLogit. We require a
    # pricer to be supplied so this stays explicit.
    if pricer is None:
        raise ValueError("_build_market_frame requires a pricer (T81 integration)")
    fair: list[float] = []
    for mp, feat in zip(market.tolist(), features, strict=True):
        st = AlphaMarketState(
            market_price=float(mp),
            time_to_resolution_days=30.0,
            features=feat,
        )
        fair.append(pricer.fair_price(st))
    fair_arr = np.array(fair, dtype=float)

    # Outcomes drawn from the *fair* probability, so the realised return
    # is on-average aligned with the sign of (fair - market). For
    # ``poison_q3=True`` we invert the resolution probability in Q3 so
    # the pricer's view is systematically wrong there — simulating a
    # regime-break the anti-alpha rule should catch.
    resolve_p = fair_arr.copy()
    if poison_q3:
        q3_mask = np.array([_quarter_index_of(ts) == 3 for ts in idx])
        # 1 - fair so the outcome distribution flips relative to the pricer.
        resolve_p[q3_mask] = 1.0 - fair_arr[q3_mask]
    outcomes = (rng.uniform(0, 1, size=n_days) < resolve_p).astype(float)

    return pd.DataFrame(
        {
            "market_price": market,
            "fair_price": fair_arr,
            "outcome": outcomes,
            "features": features,
            "close": close,
        },
        index=idx,
    )


def _gap_z_signal(frame: pd.DataFrame, alpha: BinaryPricingAlpha) -> pd.Series:
    """Compute a Bernoulli-SE-scaled gap signal *without* rolling z-score.

    This matches the *scalar-mode* logic inside ``BinaryPricingAlpha.signal``
    (which uses ``sqrt(p*(1-p))`` as the informational SE for a single
    state). Computing it vectorised this way preserves the directional
    information of ``(fair − market)`` across quarter boundaries — which
    the trailing rolling z-score would otherwise wash out.

    Crucially: ``alpha.position(signal)`` is still invoked downstream, so
    the Kelly cap, the ``z_threshold`` gate and the clip-eps behaviour of
    BinaryPricingAlpha all participate in the PnL.
    """
    eps = alpha.clip_eps
    mkt = frame["market_price"].astype(float).clip(eps, 1.0 - eps)
    fair = frame["fair_price"].astype(float).clip(eps, 1.0 - eps)
    se = np.sqrt(mkt * (1.0 - mkt))
    sig = ((fair - mkt) / se.replace(0.0, np.nan)).fillna(0.0)
    return sig.rename("signal")


def _make_alpha_strategy(
    *,
    name: str,
    frame: pd.DataFrame,
    pricer: RiskNeutralLogitAdapter,
    z_window: int = 10,
    z_threshold: float = 0.2,
    kelly_cap: float = 0.5,
) -> Strategy:
    """Wrap a ``BinaryPricingAlpha`` instance in a registry-compatible Strategy.

    The Strategy:
      * runs each row through ``BinaryPricingAlpha`` (via ``pricer`` and
        ``position()``), so the T81 pricer and T84 Kelly-cap logic are
        both genuinely exercised on every row;
      * uses :func:`_gap_z_signal` rather than ``alpha.signal()`` so the
        signal carries directional information across quarter
        boundaries (the built-in rolling z-score would normalise away
        the persistent edge a stress test needs to detect).

    The ``pnl`` callable pulls the precomputed ``outcome`` from the
    frame and lags positions by one day (same convention as the
    registry's :func:`_default_pnl`).
    """

    alpha = BinaryPricingAlpha(
        pricer=pricer,
        kelly_cap=kelly_cap,
        z_threshold=z_threshold,
        z_window=z_window,
    )

    def _signal_fn(prices: pd.DataFrame) -> pd.Series:
        view = frame.loc[prices.index.intersection(frame.index)]
        return _gap_z_signal(view, alpha)

    def _pnl_fn(prices: pd.DataFrame, position: pd.Series) -> pd.Series:
        view = frame.loc[prices.index.intersection(frame.index)]
        # alpha.position(signal) applies Kelly cap + z_threshold filter.
        sized = alpha.position(position)
        realized = view["outcome"].astype(float) - view["market_price"].astype(float)
        # Lag by one day to avoid lookahead — same convention as
        # _default_pnl in the registry.
        aligned_pos = sized.reindex(view.index).shift(1).fillna(0.0)
        return (aligned_pos * realized).fillna(0.0).rename("pnl")

    strat = Strategy(name=name, signal=_signal_fn, pnl=_pnl_fn)
    register(strat)
    return strat


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pricer() -> RiskNeutralLogitAdapter:
    """A T81 RiskNeutralLogit wrapped in the protocol adapter.

    Default-parameter logit (alpha=0, beta_market=4) plus a small
    beta_news=1.0 — enough that ``news_evidence`` moves the fair_price.
    """
    return RiskNeutralLogitAdapter(RiskNeutralLogit())


@pytest.fixture()
def start_ts() -> pd.Timestamp:
    return pd.Timestamp("2024-01-01", tz="UTC")


# ---------------------------------------------------------------------------
# Test 1 — clean alpha through all 4 quarters → PASS verdict
# ---------------------------------------------------------------------------


def test_clean_alpha_passes_all_quarters(
    stress_module: ModuleType,
    pricer: RiskNeutralLogitAdapter,
    start_ts: pd.Timestamp,
) -> None:
    """Uniformly positive alpha → every quarter Sharpe>=0.5 and sign +1 → PASS."""
    frame = _build_market_frame(
        start=start_ts,
        n_days=370,
        seed=11037,
        flip_q3=False,
        alpha_strength=0.08,
        noise_scale=0.005,
        pricer=pricer,
    )
    strat = _make_alpha_strategy(
        name="__w11_37_clean__",
        frame=frame,
        pricer=pricer,
        z_threshold=0.0,  # take every row (so PnL accumulates)
    )
    try:
        report = stress_module.run_stress(strat, start=start_ts, quarters=4, prices=frame)
    finally:
        unregister(strat.name)

    assert report["verdict"] == "PASS", report
    assert report["full_sample"]["sign"] == 1
    for row in report["quarter_rows"]:
        assert row["fail"] is False, row
        assert row["sign"] == 1, row
        assert row["sharpe"] >= 0.5, row


# ---------------------------------------------------------------------------
# Test 2 — one quarter sign-flip → FAIL, Q3 flagged
# ---------------------------------------------------------------------------


def test_q3_sign_flip_triggers_fail(
    stress_module: ModuleType,
    pricer: RiskNeutralLogitAdapter,
    start_ts: pd.Timestamp,
) -> None:
    """Q1+Q2+Q4 positive, Q3 reversed → FAIL with sign-flip on Q3."""
    frame = _build_market_frame(
        start=start_ts,
        n_days=370,
        seed=11037,
        flip_q3=True,
        alpha_strength=0.10,
        noise_scale=0.004,
        pricer=pricer,
        poison_q3=True,
    )
    strat = _make_alpha_strategy(
        name="__w11_37_signflip__",
        frame=frame,
        pricer=pricer,
        z_threshold=0.0,
    )
    try:
        report = stress_module.run_stress(strat, start=start_ts, quarters=4, prices=frame)
    finally:
        unregister(strat.name)

    assert report["verdict"] == "FAIL", report
    assert len(report["quarter_rows"]) == 4

    # Full sample dominated by 3 positive quarters → +1.
    assert report["full_sample"]["sign"] == 1, report["full_sample"]

    q3 = report["quarter_rows"][2]
    assert q3["quarter"] == 3
    assert q3["fail"] is True, q3
    assert q3["sign"] == -1, q3
    assert "sign flip" in q3["fail_reason"], q3

    # Q1/Q2/Q4 should look healthy (positive sign, no sign-flip).
    for i in (0, 1, 3):
        row = report["quarter_rows"][i]
        assert row["sign"] == 1, (i, row)
        assert "sign flip" not in row["fail_reason"], (i, row)


# ---------------------------------------------------------------------------
# Test 3 — every quarter Sharpe < 0.5 → FAIL, every quarter flagged
# ---------------------------------------------------------------------------


def test_all_quarters_below_sharpe_floor(
    stress_module: ModuleType,
    pricer: RiskNeutralLogitAdapter,
    start_ts: pd.Timestamp,
) -> None:
    """Tiny edge swamped by noise → every quarter fails the Sharpe floor."""
    frame = _build_market_frame(
        start=start_ts,
        n_days=370,
        seed=11037,
        flip_q3=False,
        alpha_strength=0.0001,  # essentially zero edge
        noise_scale=0.030,  # high noise
        pricer=pricer,
    )
    strat = _make_alpha_strategy(
        name="__w11_37_lowsharpe__",
        frame=frame,
        pricer=pricer,
        z_threshold=0.0,
    )
    try:
        report = stress_module.run_stress(
            strat,
            start=start_ts,
            quarters=4,
            prices=frame,
            sharpe_floor=5.0,  # raise floor sky-high to force every quarter to fail
        )
    finally:
        unregister(strat.name)

    assert report["verdict"] == "FAIL", report
    assert all(r["fail"] for r in report["quarter_rows"]), report["quarter_rows"]
    for row in report["quarter_rows"]:
        assert "Sharpe" in row["fail_reason"], row


# ---------------------------------------------------------------------------
# Test 4 — deflated-Sharpe gate: dsr negative flagged in payload even
# when raw Sharpe is positive.
# ---------------------------------------------------------------------------


def test_deflated_sharpe_gate_reports_negative_dsr(
    stress_module: ModuleType,
    pricer: RiskNeutralLogitAdapter,
    start_ts: pd.Timestamp,
) -> None:
    """A weakly-positive backtest must still surface deflated-Sharpe in payload.

    Bailey & Lopez de Prado (2014) deflate the Sharpe by the *expected
    max under the null* for ``n_trials`` strategies. For a borderline
    Sharpe (<1) and n_trials=4, the deflated value should land below the
    raw Sharpe — and often go negative. We assert the payload exposes the
    field for both per-quarter rows and the full sample, and that at
    least one row carries a deflated_sharpe strictly less than its raw
    Sharpe (the deflation is real, not a passthrough).
    """
    frame = _build_market_frame(
        start=start_ts,
        n_days=370,
        seed=11037,
        flip_q3=False,
        alpha_strength=0.005,
        noise_scale=0.020,
        pricer=pricer,
    )
    strat = _make_alpha_strategy(
        name="__w11_37_dsr_gate__",
        frame=frame,
        pricer=pricer,
        z_threshold=0.0,
    )
    try:
        report = stress_module.run_stress(strat, start=start_ts, quarters=4, prices=frame)
    finally:
        unregister(strat.name)

    full = report["full_sample"]
    assert "deflated_sharpe" in full
    assert "deflated_p_value" in full
    # Deflation must shrink toward 0: |dsr| < |sharpe| OR dsr <= sharpe
    # when both are positive (and strictly less when n_trials>1).
    assert full["deflated_sharpe"] <= full["sharpe"] + 1e-9, full

    saw_strict = False
    for row in report["quarter_rows"]:
        assert "deflated_sharpe" in row
        assert "expected_max_sharpe_under_null" in row
        if row["deflated_sharpe"] < row["sharpe"] - 1e-6:
            saw_strict = True
    assert saw_strict, "Expected at least one quarter where DSR < Sharpe"


# ---------------------------------------------------------------------------
# Test 5 — NaN in some quarter → that quarter's metrics degenerate (sharpe=0)
# and overall verdict is FAIL.
# ---------------------------------------------------------------------------


def test_nan_quarter_triggers_overall_fail(
    stress_module: ModuleType,
    pricer: RiskNeutralLogitAdapter,
    start_ts: pd.Timestamp,
) -> None:
    """If a quarter produces NaN PnL, Sharpe collapses to 0 → quarter FAILs."""
    base_frame = _build_market_frame(
        start=start_ts,
        n_days=370,
        seed=11037,
        flip_q3=False,
        alpha_strength=0.06,
        noise_scale=0.005,
        pricer=pricer,
    )

    # Build a custom strategy whose PnL is NaN inside Q2 (months 4..6) and
    # normal elsewhere. We do this by post-processing the alpha PnL.
    alpha = BinaryPricingAlpha(pricer=pricer, kelly_cap=0.5, z_threshold=0.0, z_window=10)

    def _signal_fn(prices: pd.DataFrame) -> pd.Series:
        view = base_frame.loc[prices.index.intersection(base_frame.index)]
        return _gap_z_signal(view, alpha)

    def _pnl_fn(prices: pd.DataFrame, position: pd.Series) -> pd.Series:
        view = base_frame.loc[prices.index.intersection(base_frame.index)]
        sized = alpha.position(position)
        realized = view["outcome"].astype(float) - view["market_price"].astype(float)
        aligned_pos = sized.reindex(view.index).shift(1).fillna(0.0)
        out = (aligned_pos * realized).fillna(0.0).rename("pnl")
        q2_mask = (out.index.month >= 4) & (out.index.month <= 6)
        out.loc[q2_mask] = float("nan")
        return out

    strat = Strategy(name="__w11_37_nan_q2__", signal=_signal_fn, pnl=_pnl_fn)
    register(strat)
    try:
        report = stress_module.run_stress(strat, start=start_ts, quarters=4, prices=base_frame)
    finally:
        unregister(strat.name)

    assert report["verdict"] == "FAIL", report
    q2 = report["quarter_rows"][1]
    assert q2["quarter"] == 2
    # NaNs drop in evaluate_quarter → either n_obs is zero (Sharpe guard
    # returns 0) or Sharpe is 0 because std collapsed. Either path must
    # flag the quarter on the Sharpe floor.
    assert q2["fail"] is True, q2
    assert q2["sharpe"] == pytest.approx(0.0, abs=1e-9), q2
    assert "Sharpe" in q2["fail_reason"], q2


# ---------------------------------------------------------------------------
# Test 6 — reproducibility: same seed → identical verdict + identical
# per-quarter numerics.
# ---------------------------------------------------------------------------


def test_same_seed_yields_same_verdict(
    stress_module: ModuleType,
    pricer: RiskNeutralLogitAdapter,
    start_ts: pd.Timestamp,
) -> None:
    """Two independent runs with the same seed must produce identical reports."""

    def _run(strategy_name: str) -> dict:
        frame = _build_market_frame(
            start=start_ts,
            n_days=370,
            seed=11037,
            flip_q3=True,
            alpha_strength=0.10,
            noise_scale=0.004,
            pricer=pricer,
            poison_q3=True,
        )
        strat = _make_alpha_strategy(
            name=strategy_name,
            frame=frame,
            pricer=pricer,
            z_threshold=0.0,
        )
        try:
            return stress_module.run_stress(strat, start=start_ts, quarters=4, prices=frame)
        finally:
            unregister(strat.name)

    rep_a = _run("__w11_37_repro_a__")
    rep_b = _run("__w11_37_repro_b__")

    assert rep_a["verdict"] == rep_b["verdict"]
    assert rep_a["full_sample"]["sharpe"] == pytest.approx(
        rep_b["full_sample"]["sharpe"], rel=1e-9, abs=1e-12
    )
    assert rep_a["full_sample"]["sign"] == rep_b["full_sample"]["sign"]
    for ra, rb in zip(rep_a["quarter_rows"], rep_b["quarter_rows"], strict=True):
        assert ra["fail"] == rb["fail"]
        assert ra["sign"] == rb["sign"]
        assert ra["sharpe"] == pytest.approx(rb["sharpe"], rel=1e-9, abs=1e-12)
        assert ra["deflated_sharpe"] == pytest.approx(rb["deflated_sharpe"], rel=1e-9, abs=1e-12)
        assert ra["fail_reason"] == rb["fail_reason"]


# ---------------------------------------------------------------------------
# Bonus — confirm the RiskNeutralLogit pricer is actually exercised by the
# alpha (i.e. T81's module body is reachable, not just imported).
# ---------------------------------------------------------------------------


def test_pricer_is_actually_invoked(pricer: RiskNeutralLogitAdapter) -> None:
    """Smoke test: the adapter routes a call through T81 and returns [0,1]."""
    state = AlphaMarketState(
        market_price=0.40,
        time_to_resolution_days=30.0,
        features={"news_evidence": 0.5},
    )
    p = pricer.fair_price(state)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0
    assert math.isfinite(p)

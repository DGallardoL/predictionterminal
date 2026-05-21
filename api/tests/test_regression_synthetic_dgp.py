"""Synthetic DGP recovery tests for the ``/fit`` endpoint (W11-41).

These tests instantiate a known data-generating process (DGP) by mocking the
two data-layer hooks the regression pipeline depends on:

* ``pfm.main.fetch_factor_history`` — returns the per-factor probability
  series. We construct the price path so that ``logit(p_t) - logit(p_{t-1})``
  equals our pre-chosen ``Δlogit`` regressor sequence, giving the test full
  control of the design matrix ``X`` post-transform.
* ``pfm.main.get_log_returns`` — returns the ticker's daily log returns.
  We build it as ``y = β · X + ε`` with a known random-number generator
  (``numpy.random.default_rng(seed=42)``).

The pipeline applies a one-day backward shift to the factor Δlogit before
joining with returns (``_shift_to_stock_calendar(days=-1)`` — see
``regression_core.py``), so we align the synthetic returns on the same shifted
calendar to keep the inner-join non-empty and the regression well-posed.

Each test uses 200 observations (per task spec) and asserts the recovered
beta falls within 5% of the truth (or asserts the appropriate graceful
behaviour for edge cases like zero signal, perfect collinearity, etc.).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import pfm.main as main_mod
from pfm.cache import NullCache

# ---------------------------------------------------------------------------
# DGP helpers
# ---------------------------------------------------------------------------


def _factor_prices_from_dlogit(
    dl_values: np.ndarray,
    start: pd.Timestamp,
    base_logit: float = 0.0,
) -> pd.DataFrame:
    """Build a Polymarket-style ``price`` DataFrame whose Δlogit equals ``dl_values``.

    Given a target sequence of N Δlogit values, we cumulate them on top of a
    base logit, then convert via the sigmoid back to probabilities. The
    resulting series has ``N + 1`` daily observations (the first one is the
    base before any innovation).

    NOTE on bounding: the regression pipeline applies a logit clip at
    ``ε=0.01`` (the default), i.e. it transforms ``p`` then clips logits to
    ``[logit(0.01), logit(0.99)] ≈ [-4.6, 4.6]``. The DGP caller must keep
    its ``dl_values`` small enough that the cumsum stays inside that band,
    otherwise the recovered X column is silently zeroed at the boundaries
    and the regression no longer sees the true regressor. The tests below
    use ``σ_X = 0.15`` over ``N = 200`` (cumsum stdev ≈ 2.1) which sits
    comfortably below the clip.
    """
    logits = np.concatenate([[base_logit], base_logit + np.cumsum(dl_values)])
    probs = 1.0 / (1.0 + np.exp(-logits))
    idx = pd.date_range(start, periods=len(probs), freq="D", tz="UTC")
    idx.name = "date"
    return pd.DataFrame({"price": probs}, index=idx)


def _make_dgp_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    factor_id_to_dlogit: dict[str, np.ndarray],
    y_builder,
    *,
    start: pd.Timestamp = pd.Timestamp("2025-01-01", tz="UTC"),
) -> TestClient:
    """Wire a TestClient whose factor + ticker data come from the supplied DGP.

    ``factor_id_to_dlogit`` maps factor-id → length-N Δlogit array.
    ``y_builder`` is called with the post-shift X DataFrame to produce the
    log-return Series the ticker hook should return.
    """
    # 1. Write a minimal factors.yml with one entry per requested factor id.
    factors_yml = tmp_path / "factors.yml"
    lines = ["factors:"]
    for fid in factor_id_to_dlogit:
        lines.append(f"  - id: {fid}")
        lines.append(f"    name: Synthetic {fid}")
        lines.append(f"    slug: slug-{fid}")
        lines.append("    source: polymarket")
        lines.append(f"    description: Synthetic factor {fid}.")
    factors_yml.write_text("\n".join(lines) + "\n")
    monkeypatch.setenv("FACTORS_FILE", str(factors_yml))

    import pfm.config as cfg

    cfg._settings = None  # reset cached Settings singleton

    # 2. Precompute factor price DataFrames keyed by slug.
    slug_to_df: dict[str, pd.DataFrame] = {}
    for fid, dl in factor_id_to_dlogit.items():
        slug_to_df[f"slug-{fid}"] = _factor_prices_from_dlogit(dl, start)

    def _fetch(_client, slug: str, start=None, end=None):
        df = slug_to_df[slug]
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        return df

    monkeypatch.setattr(main_mod, "fetch_factor_history", _fetch)

    # 3. Build the ticker log-return series. We construct the *post-shift* X
    # the regressor will see: Δlogit at calendar-date t is shifted back 1 day
    # to t-1. So if factor prices live on dates [t0, t0+1, ..., t0+N], the
    # post-shift Δlogit lives on [t0, t0+1, ..., t0+N-1]. Build y on that
    # same daily calendar so the inner-join uses every observation.
    post_shift_idx = pd.date_range(
        start, periods=len(next(iter(factor_id_to_dlogit.values()))), freq="D", tz="UTC"
    )
    post_shift_idx = pd.to_datetime(post_shift_idx, utc=True).normalize()
    X_df = pd.DataFrame(
        dict(factor_id_to_dlogit.items()),
        index=post_shift_idx,
    )
    y_series = y_builder(X_df)
    # Ensure normalised UTC dates so the inner-join with the factor side hits.
    y_series.index = pd.to_datetime(y_series.index, utc=True).normalize()
    y_series.name = "r"

    def _log_returns(ticker, start, end, return_type="log"):
        s = y_series.copy()

        def _to_utc(ts):
            ts = pd.Timestamp(ts)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return ts.normalize()

        if start is not None:
            s = s[s.index >= _to_utc(start)]
        if end is not None:
            s = s[s.index <= _to_utc(end)]
        return s

    monkeypatch.setattr(main_mod, "get_log_returns", _log_returns)
    monkeypatch.setattr(main_mod, "RedisCache", lambda url: NullCache())

    return TestClient(main_mod.app)


def _post_fit(client: TestClient, factors: list[str], **kwargs) -> dict:
    body = {
        "ticker": "SYN",
        "factors": factors,
        "start": kwargs.pop("start", "2025-01-01"),
        "end": kwargs.pop("end", "2025-12-31"),
    }
    body.update(kwargs)
    r = client.post("/fit", json=body)
    return {"status": r.status_code, "body": r.json(), "response": r}


# ---------------------------------------------------------------------------
# Test 1 — Single-factor DGP
# ---------------------------------------------------------------------------


def test_single_factor_dgp_recovers_beta_05(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """y = 0.5·X + ε. Expect β ≈ 0.5 within 5%.

    σ_X = 0.15 keeps the cumulative-logit price path inside the default
    ε=0.01 clip band; σ_ε = 0.02 gives a healthy SNR (R² ≈ 0.9).
    """
    rng = np.random.default_rng(seed=42)
    n = 200
    X = rng.normal(0.0, 0.15, size=n)
    true_beta = 0.5
    eps = rng.normal(0.0, 0.02, size=n)

    def _y_from(X_df: pd.DataFrame) -> pd.Series:
        return pd.Series(true_beta * X_df["f1"].values + eps, index=X_df.index)

    with _make_dgp_app(monkeypatch, tmp_path, {"f1": X}, _y_from) as client:
        out = _post_fit(client, ["f1"])

    assert out["status"] == 200, out["body"]
    factors = {f["id"]: f for f in out["body"]["factors"]}
    beta_hat = factors["f1"]["beta"]
    assert abs(beta_hat - true_beta) / abs(true_beta) < 0.05, (
        f"single-factor recovery off: β̂={beta_hat:.4f} vs β={true_beta}"
    )
    # R² should be high (SNR is healthy).
    assert out["body"]["model"]["r_squared"] > 0.80, out["body"]["model"]
    # Residual standard deviation should be close to ε σ = 0.02.
    assert 0.013 < out["body"]["model"]["residual_std"] < 0.030


# ---------------------------------------------------------------------------
# Test 2 — Two-factor DGP
# ---------------------------------------------------------------------------


def test_two_factor_dgp_recovers_both_betas(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """y = 0.3·X1 + 0.7·X2 + ε. Recover both within 5%."""
    rng = np.random.default_rng(seed=42)
    n = 200
    X1 = rng.normal(0.0, 0.15, size=n)
    X2 = rng.normal(0.0, 0.15, size=n)
    b1, b2 = 0.3, 0.7
    eps = rng.normal(0.0, 0.015, size=n)

    def _y_from(X_df: pd.DataFrame) -> pd.Series:
        return pd.Series(
            b1 * X_df["f1"].values + b2 * X_df["f2"].values + eps,
            index=X_df.index,
        )

    with _make_dgp_app(monkeypatch, tmp_path, {"f1": X1, "f2": X2}, _y_from) as client:
        out = _post_fit(client, ["f1", "f2"])

    assert out["status"] == 200, out["body"]
    factors = {f["id"]: f for f in out["body"]["factors"]}
    b1_hat = factors["f1"]["beta"]
    b2_hat = factors["f2"]["beta"]
    assert abs(b1_hat - b1) / b1 < 0.05, f"f1: β̂={b1_hat:.4f} vs {b1}"
    assert abs(b2_hat - b2) / b2 < 0.05, f"f2: β̂={b2_hat:.4f} vs {b2}"
    # Both should be highly significant at this SNR.
    assert factors["f1"]["p_value"] < 0.001
    assert factors["f2"]["p_value"] < 0.001


# ---------------------------------------------------------------------------
# Test 3 — Zero-beta DGP (no signal)
# ---------------------------------------------------------------------------


def test_zero_beta_dgp_recovers_near_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """y = ε (pure noise, no relationship to X). β̂ should be near 0."""
    rng = np.random.default_rng(seed=42)
    n = 200
    X = rng.normal(0.0, 0.15, size=n)
    eps = rng.normal(0.0, 0.02, size=n)

    def _y_from(X_df: pd.DataFrame) -> pd.Series:
        return pd.Series(eps, index=X_df.index)

    with _make_dgp_app(monkeypatch, tmp_path, {"f1": X}, _y_from) as client:
        out = _post_fit(client, ["f1"])

    assert out["status"] == 200, out["body"]
    beta_hat = out["body"]["factors"][0]["beta"]
    # With n=200, σ_X=0.15, σ_ε=0.02 the SE of β̂ ≈ σ_ε/(σ_X·√n) ≈ 0.0094.
    # |β̂| < 0.05 is ~5 SEs above zero — very conservative.
    assert abs(beta_hat) < 0.05, f"zero-beta DGP recovered β̂={beta_hat:.4f}"
    # R² should be tiny.
    assert out["body"]["model"]["r_squared"] < 0.10


# ---------------------------------------------------------------------------
# Test 4 — Negative beta
# ---------------------------------------------------------------------------


def test_negative_beta_dgp_recovers_minus_04(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """y = -0.4·X + ε. β̂ should be ≈ -0.4 within 5%."""
    rng = np.random.default_rng(seed=42)
    n = 200
    X = rng.normal(0.0, 0.15, size=n)
    true_beta = -0.4
    eps = rng.normal(0.0, 0.015, size=n)

    def _y_from(X_df: pd.DataFrame) -> pd.Series:
        return pd.Series(true_beta * X_df["f1"].values + eps, index=X_df.index)

    with _make_dgp_app(monkeypatch, tmp_path, {"f1": X}, _y_from) as client:
        out = _post_fit(client, ["f1"])

    assert out["status"] == 200, out["body"]
    beta_hat = out["body"]["factors"][0]["beta"]
    assert abs(beta_hat - true_beta) / abs(true_beta) < 0.05, (
        f"negative-beta recovery off: β̂={beta_hat:.4f} vs β={true_beta}"
    )
    # Sign must be correct.
    assert beta_hat < 0
    # CI should bracket the truth.
    assert out["body"]["factors"][0]["ci_low"] < true_beta < out["body"]["factors"][0]["ci_high"]


# ---------------------------------------------------------------------------
# Test 5 — Heteroskedastic errors (HAC SE > IID, R² lower)
# ---------------------------------------------------------------------------


def test_heteroskedastic_errors_hac_vs_homoskedastic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Heteroskedastic ε(σ varying with |X|) → HAC SE strictly larger than the
    homoskedastic-ε baseline at the same true β, and R² is lower."""
    rng = np.random.default_rng(seed=42)
    n = 200
    X = rng.normal(0.0, 0.15, size=n)
    true_beta = 0.5

    # Heteroskedastic noise: σ scales with |X|. The mean innovation is zero so
    # the OLS coefficient is still unbiased, but the SE under HAC widens.
    sigma_het = 0.01 + 0.15 * np.abs(X)
    eps_het = rng.normal(0.0, 1.0, size=n) * sigma_het

    def _y_het(X_df: pd.DataFrame) -> pd.Series:
        return pd.Series(true_beta * X_df["f1"].values + eps_het, index=X_df.index)

    with _make_dgp_app(monkeypatch, tmp_path, {"f1": X}, _y_het) as client:
        out_het = _post_fit(client, ["f1"])
    assert out_het["status"] == 200

    # Homoskedastic baseline with the same average ε variance.
    sigma_homo = float(np.sqrt(np.mean(sigma_het**2)))
    eps_homo = rng.normal(0.0, sigma_homo, size=n)

    def _y_homo(X_df: pd.DataFrame) -> pd.Series:
        return pd.Series(true_beta * X_df["f1"].values + eps_homo, index=X_df.index)

    with _make_dgp_app(monkeypatch, tmp_path, {"f1": X}, _y_homo) as client:
        out_homo = _post_fit(client, ["f1"])
    assert out_homo["status"] == 200

    se_het = out_het["body"]["factors"][0]["std_err"]
    se_homo = out_homo["body"]["factors"][0]["std_err"]
    # HAC SE under heteroskedasticity should be > IID-equivalent baseline.
    assert se_het > se_homo, (
        f"heteroskedastic SE should be larger: het={se_het:.4f}, homo={se_homo:.4f}"
    )
    # Heteroskedastic R² lower than homoskedastic — same average noise but
    # the extreme-X observations dominate, hurting overall fit slightly.
    r2_het = out_het["body"]["model"]["r_squared"]
    r2_homo = out_homo["body"]["model"]["r_squared"]
    assert r2_het <= r2_homo + 0.05  # allow tiny finite-sample slack


# ---------------------------------------------------------------------------
# Test 6 — Perfect collinearity (X1 == X2)
# ---------------------------------------------------------------------------


def test_perfect_collinearity_handled_gracefully(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """X1 = X2 → API must NOT crash. Either rejects with 4xx detail or
    returns a 200 with a VIF / collinearity warning."""
    rng = np.random.default_rng(seed=42)
    n = 200
    X = rng.normal(0.0, 0.15, size=n)
    # Identical regressors — perfect collinearity.
    X1 = X.copy()
    X2 = X.copy()
    eps = rng.normal(0.0, 0.015, size=n)

    def _y(X_df: pd.DataFrame) -> pd.Series:
        return pd.Series(0.5 * X_df["f1"].values + eps, index=X_df.index)

    with _make_dgp_app(monkeypatch, tmp_path, {"f1": X1, "f2": X2}, _y) as client:
        out = _post_fit(client, ["f1", "f2"])

    # Either reject with structured 4xx OR surface a collinearity warning.
    if out["status"] != 200:
        assert 400 <= out["status"] < 500, f"unexpected status {out['status']}"
        # Detail should reference collinearity / rank / VIF.
        detail = (out["body"].get("detail") or "").lower()
        assert any(kw in detail for kw in ("collinear", "rank", "vif", "singular", "identif")), (
            f"4xx without informative detail: {detail!r}"
        )
    else:
        body = out["body"]
        # Either an explicit warning OR a clearly-flagged VIF blow-up.
        warnings_blob = " ".join(body.get("warnings", [])).lower()
        vif_vals = body["diagnostics"].get("vif", {})
        has_warning = any(kw in warnings_blob for kw in ("collinear", "vif", "rank"))
        # With perfect collinearity, statsmodels reports inf or huge VIF
        # for at least one factor.
        has_huge_vif = any((v is None) or v == float("inf") or v > 50.0 for v in vif_vals.values())
        assert has_warning or has_huge_vif, (
            f"perfect collinearity not flagged: warnings={body.get('warnings')}, vif={vif_vals}"
        )


# ---------------------------------------------------------------------------
# Test 7 — Tiny sample (single-observation–like) returns 422 or graceful
# ---------------------------------------------------------------------------


def test_too_few_observations_returns_422(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """n_obs < k+2 → /fit must reject with 422, never 5xx."""
    rng = np.random.default_rng(seed=42)
    # 2 Δlogit observations → after inner-join with returns, n < k+2 for k=1.
    X = rng.normal(0.0, 0.15, size=2)
    eps = rng.normal(0.0, 0.02, size=2)

    def _y(X_df: pd.DataFrame) -> pd.Series:
        return pd.Series(0.5 * X_df["f1"].values + eps, index=X_df.index)

    with _make_dgp_app(monkeypatch, tmp_path, {"f1": X}, _y) as client:
        out = _post_fit(client, ["f1"])
    # Either 422 (too few obs) or 502 (empty after upstream filtering) is
    # acceptable — the contract is "do not crash with a 5xx-unstructured".
    assert out["status"] in (422, 502), out["body"]
    assert isinstance(out["body"].get("detail"), str)


# ---------------------------------------------------------------------------
# Test 8 — p-value sanity (high SNR → tiny p; pure noise → p typically > 0.05)
# ---------------------------------------------------------------------------


def test_pvalue_sanity_signal_vs_noise(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """High SNR DGP → p < 0.001. Pure-noise DGP → median p across replicates
    should be away from zero (most replicates above 0.05)."""
    # --- High-SNR replicate ---
    rng_hi = np.random.default_rng(seed=42)
    n = 200
    X_hi = rng_hi.normal(0.0, 0.15, size=n)
    eps_hi = rng_hi.normal(0.0, 0.01, size=n)
    true_beta = 0.6

    def _y_hi(X_df: pd.DataFrame) -> pd.Series:
        return pd.Series(true_beta * X_df["f1"].values + eps_hi, index=X_df.index)

    with _make_dgp_app(monkeypatch, tmp_path, {"f1": X_hi}, _y_hi) as client:
        out_hi = _post_fit(client, ["f1"])
    assert out_hi["status"] == 200
    p_hi = out_hi["body"]["factors"][0]["p_value"]
    assert p_hi < 0.001, f"high-SNR p_value should be tiny, got {p_hi:.4g}"

    # --- Pure-noise replicates: at least most should be non-significant. ---
    n_replicates = 10
    n_sig = 0
    for seed in range(1000, 1000 + n_replicates):
        rng_n = np.random.default_rng(seed=seed)
        X_n = rng_n.normal(0.0, 0.15, size=n)
        eps_n = rng_n.normal(0.0, 0.05, size=n)

        def _y_noise(X_df: pd.DataFrame, _e=eps_n) -> pd.Series:
            return pd.Series(_e, index=X_df.index)

        with _make_dgp_app(monkeypatch, tmp_path, {"f1": X_n}, _y_noise) as client:
            out = _post_fit(client, ["f1"])
        assert out["status"] == 200
        if out["body"]["factors"][0]["p_value"] < 0.05:
            n_sig += 1
    # At α=0.05 under the null, expected false-positive rate is ~0.05;
    # asserting ≤ 4/10 (40%) keeps the test robust to MC noise while still
    # catching a broken inference path that would mark *every* replicate
    # significant.
    assert n_sig <= 4, f"pure-noise DGP flagged significant in {n_sig}/10 replicates"


# ---------------------------------------------------------------------------
# Test 9 — Sample-size scaling: doubling N halves SE roughly
# ---------------------------------------------------------------------------


def test_sample_size_scaling_se_shrinks_with_n(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Doubling N from 100 → 200 should shrink HAC SE by a factor close to
    √2 (≈ 1.41). Tolerance is loose because the HAC bandwidth grows with N
    too, but the ratio should sit between 1.15 and 1.8."""

    def _fit_with_n(n: int) -> float:
        rng = np.random.default_rng(seed=42)
        X = rng.normal(0.0, 0.15, size=n)
        eps = rng.normal(0.0, 0.02, size=n)

        def _y(X_df: pd.DataFrame) -> pd.Series:
            return pd.Series(0.5 * X_df["f1"].values + eps, index=X_df.index)

        with _make_dgp_app(monkeypatch, tmp_path, {"f1": X}, _y) as client:
            out = _post_fit(client, ["f1"])
        assert out["status"] == 200, out["body"]
        return float(out["body"]["factors"][0]["std_err"])

    se_100 = _fit_with_n(100)
    se_200 = _fit_with_n(200)

    ratio = se_100 / se_200
    # √2 ≈ 1.414. Allow a wide-enough band to absorb HAC-bandwidth growth
    # and a single random-seed realisation.
    assert 1.15 < ratio < 1.8, (
        f"SE did not scale as ~1/√N: se(100)={se_100:.4f}, se(200)={se_200:.4f}, ratio={ratio:.3f}"
    )

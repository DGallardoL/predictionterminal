"""Risk-neutral terminal density from a listed option chain.

Recovers the density as the second derivative of the call-price curve.

We turn an SPX/SPY option chain into a smooth, arbitrage-light risk-neutral
density of the underlying at expiry, to cross-check against the Kalshi
prediction-market-implied density (:mod:`pfm.vol.implied_pdf`).

Pipeline
--------
1. **Fetch** the chain for a target expiry (``^SPX`` preferred — same index Kalshi
   ``KXINX`` tracks; ``SPY``×10 fallback) via yfinance.
2. **Forward** ``F`` from put-call parity near the money (robust to a missing/ና
   stale spot); ``T`` in years to the 16:00-ET expiry.
3. **OTM implied-vol smile** — recompute IV from the *mid* price by Black-76
   inversion (don't trust the vendor IV), using OTM puts for ``K<F`` and OTM
   calls for ``K>=F`` (tighter spreads, more information on each wing).
4. **Smooth** IV as a function of log-moneyness ``k=ln(K/F)`` with a penalised
   smoothing spline, then reprice a DENSE undiscounted call curve ``c(K)``.
5. **Call-curve second derivative** — ``f_Q(K) = e^{rT} ∂²C/∂K² = ∂²c/∂K²`` (undiscounted
   ``c`` already absorbs the discount factor), via finite differences. Clip
   negatives, renormalise to integrate to 1.

The network fetch is isolated in :func:`fetch_option_chain`; the math entry
:func:`rn_density_from_quotes` is pure and unit-tested with synthetic
Black-Scholes quotes (a known ``σ`` must recover the matching log-normal).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq
from scipy.stats import norm

logger = logging.getLogger(__name__)

_trapz = getattr(np, "trapezoid", None) or np.trapz

#: Seconds in a (calendar) year — matches the short-dated convention used by
#: the Kalshi implied-PDF engine so the two horizons line up.
_SECONDS_PER_YEAR: float = 365.25 * 86_400.0
_MIN_T_YEARS: float = 1.0 / (365.0 * 24.0)  # 1 hour floor

#: Fallback annualised risk-free rate when ^IRX is unavailable. r·T is tiny for
#: the short-dated SPX markets Kalshi lists, so the exact value barely matters.
_DEFAULT_RISK_FREE: float = 0.045


# ---------------------------------------------------------------------------
# Black-76 (undiscounted) and IV inversion
# ---------------------------------------------------------------------------


def _bs_undisc_call(forward: float, strike: float, t_years: float, sigma: float) -> float:
    """Undiscounted Black-76 call value ``c = F·N(d1) - K·N(d2)``."""
    if sigma <= 0 or t_years <= 0:
        return max(forward - strike, 0.0)
    vol = sigma * np.sqrt(t_years)
    d1 = (np.log(forward / strike) + 0.5 * vol * vol) / vol
    d2 = d1 - vol
    return float(forward * norm.cdf(d1) - strike * norm.cdf(d2))


def _bs_undisc_put(forward: float, strike: float, t_years: float, sigma: float) -> float:
    """Undiscounted Black-76 put value ``p = K·N(-d2) - F·N(-d1)``."""
    if sigma <= 0 or t_years <= 0:
        return max(strike - forward, 0.0)
    vol = sigma * np.sqrt(t_years)
    d1 = (np.log(forward / strike) + 0.5 * vol * vol) / vol
    d2 = d1 - vol
    return float(strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


def implied_vol(
    price_undisc: float, forward: float, strike: float, t_years: float, is_call: bool
) -> float | None:
    """Invert Black-76 for the implied vol of an *undiscounted* option price.

    Args:
        price_undisc: Option price already divided by the discount factor.
        forward: Forward of the underlying to expiry.
        strike: Option strike.
        t_years: Time to expiry in years.
        is_call: ``True`` for a call, ``False`` for a put.

    Returns:
        The implied vol, or ``None`` if the price is outside no-arbitrage bounds
        or the solver fails.
    """
    if t_years <= 0 or price_undisc <= 0 or forward <= 0 or strike <= 0:
        return None
    intrinsic = max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
    upper = forward if is_call else strike  # max undiscounted option value
    if price_undisc <= intrinsic + 1e-9 or price_undisc >= upper - 1e-9:
        return None
    fn = _bs_undisc_call if is_call else _bs_undisc_put

    def obj(sig: float) -> float:
        return fn(forward, strike, t_years, sig) - price_undisc

    try:
        return float(brentq(obj, 1e-4, 5.0, maxiter=100, xtol=1e-8))
    except (ValueError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Shape-restricted density (monotone + convex call curve)
# ---------------------------------------------------------------------------


def _pava_increasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators projection onto the non-decreasing cone (weighted)."""
    means: list[float] = []
    weights: list[float] = []
    counts: list[int] = []
    for yi, wi in zip(y, w, strict=True):
        means.append(float(yi))
        weights.append(float(wi))
        counts.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            w2 = weights[-1] + weights[-2]
            m2 = (means[-1] * weights[-1] + means[-2] * weights[-2]) / w2
            c2 = counts[-1] + counts[-2]
            means.pop()
            weights.pop()
            counts.pop()
            means[-1], weights[-1], counts[-1] = m2, w2, c2
    out = np.empty(len(y))
    pos = 0
    for m, c in zip(means, counts, strict=True):
        out[pos : pos + c] = m
        pos += c
    return out


def _gauss_smooth_pdf(y: np.ndarray, sigma_pts: float) -> np.ndarray:
    """Edge-aware Gaussian smooth of a density array (re-normalised weights)."""
    n = len(y)
    radius = max(1, int(np.ceil(sigma_pts * 3)))
    k = np.arange(-radius, radius + 1)
    kernel = np.exp(-(k * k) / (2.0 * sigma_pts * sigma_pts))
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        w = kernel[(lo - i + radius) : (hi - i + radius)]
        out[i] = float(np.dot(y[lo:hi], w) / w.sum())
    return out


def _bl_density_shape_restricted(grid: np.ndarray, call: np.ndarray) -> np.ndarray:
    """Risk-neutral density (second derivative of the call-price curve) under
    monotone + convex shape restrictions.

    Projects the undiscounted call curve onto the no-arbitrage cone — slope
    ``C'(K) ∈ [-1, 0]`` (monotone non-increasing) and ``C''(K) ≥ 0`` (convex) —
    then reads the density off the convexified second difference. Non-negativity
    is guaranteed by construction (the isotonic projection of the slopes), so no
    ad-hoc clipping is needed. Assumes an evenly-spaced ``grid`` (linspace).
    """
    grid = np.asarray(grid, dtype=float)
    call = np.asarray(call, dtype=float)
    n = len(grid)
    h = (grid[-1] - grid[0]) / (n - 1)
    slopes = np.diff(call) / h  # discrete C'(K), length n-1
    slopes_iso = np.clip(_pava_increasing(slopes, np.ones(n - 1)), -1.0, 0.0)
    pdf = np.zeros(n)
    pdf[1:-1] = (slopes_iso[1:] - slopes_iso[:-1]) / h  # C''(K) ≥ 0, length n-2
    return np.clip(pdf, 0.0, None)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class OptionsRNResult:
    """Options-implied risk-neutral density and the inputs that produced it."""

    asset: str
    forward: float
    t_years: float
    risk_free: float
    grid: np.ndarray
    pdf: np.ndarray
    cdf: np.ndarray
    smile_k: np.ndarray  # log-moneyness knots used for the smile fit
    smile_iv: np.ndarray  # OTM implied vols at those knots
    n_options: int
    atm_iv: float
    warnings: list[str] = field(default_factory=list)
    expiry: str | None = None  # YYYY-MM-DD expiry the chain was sourced for
    ticker: str | None = None  # underlying ticker used (^SPX / SPY / …)


# ---------------------------------------------------------------------------
# Core math — pure, unit-tested with synthetic BS quotes
# ---------------------------------------------------------------------------


def _forward_from_parity(
    strikes: np.ndarray,
    call_mid: np.ndarray,
    put_mid: np.ndarray,
    discount: float,
    spot_hint: float | None,
) -> float:
    """Estimate the forward via put-call parity ``C-P = e^{-rT}(F-K)``.

    Uses the strikes where both a call and a put mid exist and ``|C-P|`` is
    smallest (closest to ATM, where parity is tightest). Falls back to
    ``spot_hint`` then to the median strike.
    """
    both = np.isfinite(call_mid) & np.isfinite(put_mid) & (call_mid > 0) & (put_mid > 0)
    if both.sum() >= 3:
        diff = np.abs(call_mid - put_mid)
        order = np.argsort(np.where(both, diff, np.inf))[: max(3, both.sum() // 4)]
        fwds = strikes[order] + (call_mid[order] - put_mid[order]) / discount
        fwds = fwds[np.isfinite(fwds)]
        if fwds.size:
            return float(np.median(fwds))
    if spot_hint and spot_hint > 0:
        return float(spot_hint / discount)  # F = S·e^{rT} (q≈0 short-dated)
    return float(np.median(strikes))


def rn_density_from_quotes(
    strikes: np.ndarray,
    call_mid: np.ndarray,
    put_mid: np.ndarray,
    t_years: float,
    *,
    asset: str = "SPX",
    risk_free: float = _DEFAULT_RISK_FREE,
    spot_hint: float | None = None,
    grid: np.ndarray | None = None,
    grid_size: int = 400,
    smile_smoothing: float = 0.0,
    price_t_years: float | None = None,
) -> OptionsRNResult:
    """Recover the risk-neutral density from raw option mid-quotes.

    Args:
        strikes: Strikes (ascending or not — sorted internally).
        call_mid: Call mid prices aligned with ``strikes`` (``nan`` if missing).
        put_mid: Put mid prices aligned with ``strikes`` (``nan`` if missing).
        t_years: Time to the option expiry in years (used for IV inversion).
        asset: Label for the result.
        risk_free: Annualised continuously-compounded rate.
        spot_hint: Spot for the parity-forward fallback.
        grid: Optional dense strike grid to evaluate on (for alignment with the
            Kalshi density). Defaults to ±6σ around the forward.
        grid_size: Grid resolution when ``grid`` is not supplied.
        smile_smoothing: ``s`` parameter for the IV smoothing spline (0 = auto).
        price_t_years: Horizon to *evaluate the density at* (defaults to
            ``t_years``). The annualised smile is repriced at this horizon so the
            density can be aligned to a different maturity (e.g. the Kalshi
            market's), with the forward carried over by the same annual rate.

    Returns:
        :class:`OptionsRNResult`. ``t_years`` on the result is the priced horizon.

    Raises:
        ValueError: fewer than 4 usable OTM implied vols.
    """
    strikes = np.asarray(strikes, dtype=float)
    call_mid = np.asarray(call_mid, dtype=float)
    put_mid = np.asarray(put_mid, dtype=float)
    order = np.argsort(strikes)
    strikes, call_mid, put_mid = strikes[order], call_mid[order], put_mid[order]
    t_years = max(float(t_years), _MIN_T_YEARS)
    pt = max(float(price_t_years), _MIN_T_YEARS) if price_t_years else t_years
    warnings: list[str] = []

    discount = float(np.exp(-risk_free * t_years))
    forward = _forward_from_parity(strikes, call_mid, put_mid, discount, spot_hint)
    # Forward at the priced horizon: carry the implied annual (r−q) over to pt.
    if spot_hint and spot_hint > 0 and forward > 0 and abs(pt - t_years) > 1e-9:
        carry = np.log(forward / spot_hint) / t_years
        forward_price = float(spot_hint * np.exp(carry * pt))
    else:
        forward_price = forward

    # OTM implied vols: puts below the forward, calls at/above it. Undiscount the
    # quoted prices (mids are discounted) before inversion.
    ks: list[float] = []
    ivs: list[float] = []
    for k_strike, c_mid, p_mid in zip(strikes, call_mid, put_mid, strict=True):
        use_call = k_strike >= forward
        mid = c_mid if use_call else p_mid
        if not np.isfinite(mid) or mid <= 0:
            continue
        iv = implied_vol(mid / discount, forward, float(k_strike), t_years, use_call)
        if iv is None or not (0.005 < iv < 4.0):
            continue
        ks.append(float(np.log(k_strike / forward)))
        ivs.append(iv)

    if len(ivs) < 4:
        raise ValueError(f"options_rn: only {len(ivs)} usable OTM IVs (need ≥4)")

    smile_k = np.asarray(ks)
    smile_iv = np.asarray(ivs)

    # Restrict to a moneyness band ~±N·σ_ATM·√T around the forward. Deep-OTM
    # short-dated strikes carry stale prices → huge spurious IVs (e.g. 300%+ at
    # 60% OTM) that the flat-wing extrapolation turns into fat tails and wrecks
    # the variance. N=8 keeps real skew while dropping the dust.
    near = np.abs(smile_k) < 0.01
    atm0 = float(np.median(smile_iv[near])) if near.any() else float(np.median(smile_iv))
    # Band in σ-units (≈8σ) so it widens with horizon. The absolute cap must NOT
    # be a short-dated number like 0.25 — at a multi-month horizon that truncates
    # the distribution to <2σ and mis-centres the density (mean ≠ forward). The
    # bid>0/ask>0 dust filter already removes junk deep-OTM quotes.
    band = float(np.clip(8.0 * atm0 * np.sqrt(t_years), 0.04, 1.2))
    keep = np.abs(smile_k) <= band
    if keep.sum() >= 4:
        smile_k, smile_iv = smile_k[keep], smile_iv[keep]
    else:
        warnings.append("moneyness band too tight; using full smile")
    # Deduplicate identical k (spline needs strictly increasing x).
    uniq_k, idx = np.unique(smile_k, return_index=True)
    smile_k, smile_iv = uniq_k, smile_iv[idx]

    # Penalised smoothing spline of IV vs log-moneyness. s scales with n so the
    # fit tracks the smile without chasing per-quote noise.
    s = smile_smoothing if smile_smoothing > 0 else len(smile_k) * 1e-4
    deg = min(3, len(smile_k) - 1)
    try:
        spline = UnivariateSpline(smile_k, smile_iv, k=deg, s=s)
    except Exception:  # pragma: no cover — degenerate smile
        spline = UnivariateSpline(smile_k, smile_iv, k=1, s=0)

    atm_iv = float(spline(0.0))
    if not np.isfinite(atm_iv) or atm_iv <= 0:
        atm_iv = float(np.median(smile_iv))

    # Dense strike grid spanning the fitted smile band (NOT the full observed
    # strike range, which can run to 60% OTM dust). A small pad past the band
    # lets the tails decay to ~0 inside the grid.
    if grid is None:
        k_span = float(max(np.abs(smile_k).max(), atm_iv * np.sqrt(pt)))
        lo = forward_price * np.exp(-1.15 * k_span)
        hi = forward_price * np.exp(+1.15 * k_span)
        grid = np.linspace(lo, hi, grid_size)
    else:
        grid = np.asarray(grid, dtype=float)

    # IV on the grid (annualised smile, in log-moneyness vs the priced forward);
    # clamp extrapolation to the wing IVs (flat-wing assumption).
    k_grid = np.log(np.clip(grid, 1e-9, None) / forward_price)
    iv_grid = spline(np.clip(k_grid, smile_k.min(), smile_k.max()))
    iv_grid = np.clip(iv_grid, 1e-3, 5.0)

    # Undiscounted call curve at the PRICED horizon pt, then take the second derivative.
    call_curve = np.array(
        [
            _bs_undisc_call(forward_price, float(k), pt, float(s_))
            for k, s_ in zip(grid, iv_grid, strict=True)
        ]
    )
    # Shape restrictions: the undiscounted call must
    # be monotone non-increasing (slope ∈ [-1, 0]) and convex (slope
    # non-decreasing) in K. Projecting the call curve onto that cone guarantees a
    # NON-NEGATIVE, arbitrage-free density by construction — no ad-hoc clipping.
    pdf = _bl_density_shape_restricted(grid, call_curve)
    # The convexity (PAVA) projection makes the call curve piecewise-linear, so
    # its second difference (the density) comes out as stair-steps — spikes at
    # the kinks and zero-density flats between them, which wrecks bucket-level
    # probabilities. A light Gaussian smooth (≈1.5% of the grid) restores a
    # smooth density; smoothing a non-negative array stays non-negative.
    pdf = _gauss_smooth_pdf(pdf, sigma_pts=max(2.0, len(grid) / 80.0))

    area = float(_trapz(pdf, grid))
    if area <= 0:
        raise ValueError("options_rn: degenerate density (non-positive mass)")
    pdf = pdf / area
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]

    if len(ivs) < 8:
        warnings.append(f"thin chain: only {len(ivs)} OTM strikes priced")

    if abs(pt - t_years) > 1e-9:
        warnings.append(
            f"repriced to {pt * 365.25:.2f}d horizon from a {t_years * 365.25:.2f}d option expiry"
        )

    return OptionsRNResult(
        asset=asset,
        forward=forward_price,
        t_years=pt,
        risk_free=risk_free,
        grid=grid,
        pdf=pdf,
        cdf=cdf,
        smile_k=smile_k,
        smile_iv=smile_iv,
        n_options=len(ivs),
        atm_iv=atm_iv,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Network fetch (yfinance) — isolated + mockable
# ---------------------------------------------------------------------------


def _expiry_to_utc(expiry_date: str) -> datetime:
    """Map a yfinance ``YYYY-MM-DD`` expiry to its 16:00-ET (20:00 UTC) instant."""
    y, m, d = (int(x) for x in expiry_date.split("-"))
    return datetime(y, m, d, 20, 0, 0, tzinfo=UTC)  # 16:00 ET ≈ 20:00 UTC


def _nearest_expiry(expiries: list[str], target: str | None) -> tuple[str, int]:
    """Return the listed expiry closest to ``target`` and the gap in days.

    Kalshi index markets span horizons (daily, weekly, monthly, EOY), so we pick
    the option expiry nearest the Kalshi maturity rather than the soonest.
    """
    if not target:
        return expiries[0], 0
    td = datetime.strptime(target, "%Y-%m-%d").date()
    best, best_gap = expiries[0], 10**9
    for e in expiries:
        try:
            gap = abs((datetime.strptime(e, "%Y-%m-%d").date() - td).days)
        except ValueError:
            continue
        if gap < best_gap:
            best, best_gap = e, gap
    return best, best_gap


def fetch_option_chain(
    asset: str = "SPX",
    target_expiry: str | None = None,
    *,
    now_utc: datetime | None = None,
) -> dict:
    """Fetch an option chain via yfinance for the nearest matching expiry.

    Tries ``^SPX`` (the index Kalshi tracks) first; on failure or an empty chain
    falls back to ``SPY`` with strikes scaled ×10 to index points.

    Args:
        asset: ``"SPX"`` or ``"NDX"``.
        target_expiry: ``YYYY-MM-DD`` to match; defaults to the nearest listed.
        now_utc: Reference now (for ``t_years``); defaults to current UTC.

    Returns:
        A dict with ``strikes``, ``call_mid``, ``put_mid`` (np arrays),
        ``spot``, ``expiry`` (``YYYY-MM-DD``), ``t_years``, ``ticker``,
        ``scale`` (1.0 for index, 10.0 for the SPY fallback).

    Raises:
        RuntimeError: no chain available from any source.
    """
    import yfinance as yf  # lazy — keeps import-time light and test-mockable

    now = now_utc or datetime.now(UTC)
    tickers = {"SPX": ["^SPX", "SPY"], "NDX": ["^NDX", "QQQ"]}.get(asset.upper(), ["^SPX"])
    scales = {"^SPX": 1.0, "SPY": 10.0, "^NDX": 1.0, "QQQ": 1.0}

    last_err: Exception | None = None
    for tk in tickers:
        try:
            t = yf.Ticker(tk)
            exps = list(t.options or [])
            if not exps:
                continue
            # Match the listed expiry CLOSEST to the Kalshi maturity — Kalshi
            # index markets are not always 0/1DTE (weeklies, monthlies, EOY), so
            # picking the soonest expiry would mismatch horizons. The day-gap is
            # reported so the caller can flag an inexact match.
            expiry, gap_days = _nearest_expiry(exps, target_expiry)
            oc = t.option_chain(expiry)
            calls, puts = oc.calls, oc.puts
            scale = scales.get(tk, 1.0)
            spot = None
            try:
                hist = t.history(period="1d")
                if len(hist):
                    spot = float(hist["Close"].iloc[-1]) * scale
            except Exception:
                spot = None

            def _mid_map(df, scale: float = scale) -> dict[float, float]:
                # Two-sided quotes only — a zero-bid option is the classic CBOE
                # "dust" filter; its lastPrice is usually stale and yields a
                # garbage deep-OTM IV that fattens the recovered tails.
                out: dict[float, float] = {}
                for _, row in df.iterrows():
                    bid, ask = float(row.get("bid", 0) or 0), float(row.get("ask", 0) or 0)
                    if bid <= 0 or ask <= 0:
                        continue
                    out[float(row["strike"]) * scale] = (bid + ask) / 2.0 * scale
                return out

            cmap, pmap = _mid_map(calls), _mid_map(puts)
            all_k = np.array(sorted(set(cmap) | set(pmap)), dtype=float)
            if all_k.size < 6:
                continue
            t_years = max(
                (_expiry_to_utc(expiry) - now).total_seconds() / _SECONDS_PER_YEAR, _MIN_T_YEARS
            )
            return {
                "strikes": all_k,
                "call_mid": np.array([cmap.get(k, np.nan) for k in all_k]),
                "put_mid": np.array([pmap.get(k, np.nan) for k in all_k]),
                "spot": spot,
                "expiry": expiry,
                "expiry_gap_days": gap_days,
                "t_years": t_years,
                "ticker": tk,
                "scale": scale,
            }
        except Exception as exc:  # try the next ticker
            last_err = exc
            logger.warning("options_rn: %s chain fetch failed: %s", tk, exc)
            continue

    raise RuntimeError(f"options_rn: no option chain for {asset} ({last_err})")


def extract_options_rn(
    asset: str = "SPX",
    target_expiry: str | None = None,
    *,
    risk_free: float = _DEFAULT_RISK_FREE,
    grid: np.ndarray | None = None,
    now_utc: datetime | None = None,
    price_t_years: float | None = None,
) -> OptionsRNResult:
    """Fetch the chain and recover the options-implied risk-neutral density.

    Args:
        price_t_years: If given, reprice the density to this horizon (e.g. the
            Kalshi market's time-to-maturity) so the two are directly comparable.
    """
    chain = fetch_option_chain(asset, target_expiry, now_utc=now_utc)
    res = rn_density_from_quotes(
        chain["strikes"],
        chain["call_mid"],
        chain["put_mid"],
        chain["t_years"],
        asset=asset,
        risk_free=risk_free,
        spot_hint=chain.get("spot"),
        grid=grid,
        price_t_years=price_t_years,
    )
    res.expiry = chain.get("expiry")
    res.ticker = chain.get("ticker")
    gap = int(chain.get("expiry_gap_days", 0) or 0)
    if target_expiry and gap > 1:
        res.warnings.append(
            f"nearest option expiry {chain.get('expiry')} is {gap}d from the Kalshi "
            f"maturity {target_expiry} — horizon match is approximate"
        )
    if chain.get("ticker") not in ("^SPX", "^NDX"):
        res.warnings.append(f"sourced from {chain['ticker']} ×{chain['scale']:.0f} (index proxy)")
    return res

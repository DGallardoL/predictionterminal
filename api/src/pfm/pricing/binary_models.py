"""Candidate pricing models for binary prediction markets.

All four models share a common ``Pricer`` Protocol::

    class Pricer(Protocol):
        name: str
        def fair_price(self, state: MarketState) -> PricingResult: ...
        def calibrate(self, history: list[tuple[MarketState, bool]]) -> "Pricer": ...

The state is a frozen dataclass that captures the *features* a pricer can
read from. Not every model uses every field — for example, the
:class:`BlackScholesDigital` relies on ``underlying`` and ``threshold``
while the :class:`BetaBinomialBayes` relies on ``news_evidence``.

References
----------
* Wolfers & Zitzewitz (2004), "Prediction Markets", JEP 18(2).
* Hull, *Options, Futures and Other Derivatives*, §22 — digital options.
* Karatzas & Shreve (1991), §5.6 — Brownian bridge survival.

The implementations here are intentionally compact and avoid heavy
machinery; the empirical harness in T82 stresses them against resolved
Polymarket histories.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Shared data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketState:
    """Snapshot of a binary prediction market at a point in time.

    Attributes
    ----------
    current_price:
        Market's currently quoted probability in ``[0, 1]``.
    time_to_resolve_days:
        Time to resolution in calendar days (can be fractional).
    underlying:
        Spot of a numeric underlying that the contract is keyed off
        (e.g. BTC spot for ``BTC > 60k`` markets). ``None`` for purely
        poll-driven markets.
    threshold:
        Numeric resolution threshold ``K`` if applicable.
    poll_history:
        Tuple of historical probability quotes — used by
        :class:`BrownianBridge` to estimate volatility for polling-based
        markets.
    news_evidence:
        Signed evidence aggregate in ``[-1, 1]``. Positive values nudge
        the posterior toward YES.
    """

    current_price: float
    time_to_resolve_days: float
    underlying: float | None = None
    threshold: float | None = None
    poll_history: tuple[float, ...] = ()
    news_evidence: float = 0.0


@dataclass(frozen=True)
class PricingResult:
    """Output of a pricer for a single market state."""

    fair_price: float
    confidence_interval: tuple[float, float]
    model_name: str
    diagnostics: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class Pricer(Protocol):
    """Common interface for all candidate pricers."""

    name: str

    def fair_price(self, state: MarketState) -> PricingResult:
        """Return the model's fair-value estimate for ``state``."""
        ...

    def calibrate(
        self,
        history: list[tuple[MarketState, bool]],
    ) -> Pricer:
        """Return a new pricer with parameters fit to historical outcomes."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PROB_FLOOR = 1e-9
_PROB_CEIL = 1.0 - 1e-9


def _clip_prob(p: float) -> float:
    """Clip a probability to the open ``(0, 1)`` interval."""

    if not math.isfinite(p):
        return 0.5
    return max(_PROB_FLOOR, min(_PROB_CEIL, float(p)))


def _logit(p: float) -> float:
    q = _clip_prob(p)
    return math.log(q / (1.0 - q))


def _expit(z: float) -> float:
    # Numerically stable sigmoid.
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _years_from_days(days: float) -> float:
    return max(float(days), 0.0) / 365.25


def _ci_from_se(p: float, se: float, z: float = 1.96) -> tuple[float, float]:
    """Symmetric normal-CI on the *logit* scale, mapped back to [0,1]."""

    if se <= 0 or not math.isfinite(se):
        return (_clip_prob(p), _clip_prob(p))
    lz = _logit(p)
    lo = _expit(lz - z * se)
    hi = _expit(lz + z * se)
    return (_clip_prob(min(lo, hi)), _clip_prob(max(lo, hi)))


# ---------------------------------------------------------------------------
# Model 1 — Risk-neutral logit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskNeutralLogit:
    """Logit re-pricer of a binary contract.

    Model::

        p = σ(α + β₁·(market_price − 0.5)
              + β₂·log(time_to_resolve_days + 1)
              + β₃·news_evidence)

    Calibration uses ``statsmodels`` :class:`~statsmodels.api.Logit` when
    available, falling back to a small Newton solver otherwise. The
    intercept ``α`` and the three coefficients are stored on the
    instance; calling :meth:`fair_price` is then pure ``numpy``.
    """

    alpha: float = 0.0
    beta_market: float = 4.0
    beta_log_t: float = 0.0
    beta_news: float = 1.0
    coef_se: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5)
    n_obs: int = 0
    name: str = "risk_neutral_logit"

    # -- feature row -----------------------------------------------------
    @staticmethod
    def _features(state: MarketState) -> np.ndarray:
        return np.array(
            [
                1.0,
                float(state.current_price) - 0.5,
                math.log(max(state.time_to_resolve_days, 0.0) + 1.0),
                float(state.news_evidence),
            ],
            dtype=float,
        )

    # -- inference -------------------------------------------------------
    def fair_price(self, state: MarketState) -> PricingResult:
        x = self._features(state)
        beta = np.array(
            [self.alpha, self.beta_market, self.beta_log_t, self.beta_news],
            dtype=float,
        )
        z = float(x @ beta)
        p = _expit(z)
        # Linear-predictor SE via diag(SE)·|x| as a quick approximation —
        # cheap and monotone in the input magnitudes.
        se = float(np.sqrt(np.sum((np.asarray(self.coef_se) * np.abs(x)) ** 2)))
        ci = _ci_from_se(p, se)
        diag = {
            "linear_predictor": z,
            "n_obs": float(self.n_obs),
            "se_linpred": se,
        }
        return PricingResult(_clip_prob(p), ci, self.name, diag)

    # -- calibration -----------------------------------------------------
    def calibrate(
        self,
        history: list[tuple[MarketState, bool]],
    ) -> RiskNeutralLogit:
        if len(history) < 4:
            return self

        X = np.vstack([self._features(s) for s, _ in history])
        y = np.array([1.0 if out else 0.0 for _, out in history], dtype=float)

        # Degenerate target — keep priors.
        if y.std() == 0.0:
            return self

        try:
            import statsmodels.api as sm  # local import — keeps cold start cheap

            model = sm.Logit(y, X)
            res = model.fit(disp=False, method="bfgs", maxiter=200)
            params = np.asarray(res.params, dtype=float)
            bse = np.asarray(res.bse, dtype=float)
        except Exception:
            params, bse = _newton_logit(X, y)

        if params.size != 4 or not np.all(np.isfinite(params)):
            return self

        se = tuple(float(s) if math.isfinite(s) else 1.0 for s in bse)
        return replace(
            self,
            alpha=float(params[0]),
            beta_market=float(params[1]),
            beta_log_t=float(params[2]),
            beta_news=float(params[3]),
            coef_se=se,  # type: ignore[arg-type]
            n_obs=int(len(history)),
        )


def _newton_logit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 200,
    tol: float = 1e-8,
    ridge: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Tiny ridged-Newton logistic regression — fallback for the calibrator."""

    _n, k = X.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        z = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        W = p * (1.0 - p)
        grad = X.T @ (p - y) + ridge * beta
        H = X.T @ (X * W[:, None]) + ridge * np.eye(k)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        beta -= step
        if np.linalg.norm(step) < tol:
            break
    z = X @ beta
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    W = p * (1.0 - p)
    H = X.T @ (X * W[:, None]) + ridge * np.eye(k)
    try:
        cov = np.linalg.inv(H)
        bse = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        bse = np.full(k, 1.0)
    return beta, bse


# ---------------------------------------------------------------------------
# Model 2 — Black-Scholes digital
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlackScholesDigital:
    """Digital-option valuation for ``S_T ≥ K`` style binary markets.

    Closed form::

        p = Φ((ln(S/K) + (μ − σ²/2)·T) / (σ·√T))

    where ``T`` is measured in years. ``σ`` is calibrated from historical
    log-returns of the underlying when available; otherwise it falls
    back to the dispersion of the supplied ``poll_history``.
    """

    sigma: float = 0.30
    mu: float = 0.0
    n_obs: int = 0
    sigma_source: str = "default"
    name: str = "black_scholes_digital"

    # -- inference -------------------------------------------------------
    def fair_price(self, state: MarketState) -> PricingResult:
        T = _years_from_days(state.time_to_resolve_days)
        diag: dict[str, float] = {
            "sigma": float(self.sigma),
            "mu": float(self.mu),
            "T_years": T,
        }

        if state.underlying is None or state.threshold is None:
            # No threshold — degenerate to the market price.
            p = _clip_prob(state.current_price)
            diag["degenerate"] = 1.0
            return PricingResult(p, (p, p), self.name, diag)

        if state.underlying <= 0 or state.threshold <= 0:
            p = _clip_prob(state.current_price)
            diag["non_positive"] = 1.0
            return PricingResult(p, (p, p), self.name, diag)

        if T <= 0 or self.sigma <= 0:
            # Collapse to indicator.
            p = 1.0 if state.underlying >= state.threshold else 0.0
            p = _clip_prob(p)
            diag["collapsed"] = 1.0
            return PricingResult(p, (p, p), self.name, diag)

        d = (
            math.log(state.underlying / state.threshold)
            + (self.mu - 0.5 * self.sigma * self.sigma) * T
        ) / (self.sigma * math.sqrt(T))
        p = float(norm.cdf(d))
        diag["d"] = d

        # CI: bump σ by ±20% as a sensitivity band — cheap and bounded.
        ci: tuple[float, float]
        try:
            d_lo = (
                math.log(state.underlying / state.threshold)
                + (self.mu - 0.5 * (self.sigma * 1.2) ** 2) * T
            ) / (self.sigma * 1.2 * math.sqrt(T))
            d_hi = (
                math.log(state.underlying / state.threshold)
                + (self.mu - 0.5 * (self.sigma * 0.8) ** 2) * T
            ) / (self.sigma * 0.8 * math.sqrt(T))
            p_lo = float(norm.cdf(d_lo))
            p_hi = float(norm.cdf(d_hi))
            ci = (_clip_prob(min(p_lo, p_hi)), _clip_prob(max(p_lo, p_hi)))
        except (ValueError, ZeroDivisionError):
            ci = (_clip_prob(p), _clip_prob(p))

        return PricingResult(_clip_prob(p), ci, self.name, diag)

    # -- calibration -----------------------------------------------------
    def calibrate(
        self,
        history: list[tuple[MarketState, bool]],
    ) -> BlackScholesDigital:
        # σ from underlying log-returns when available; fallback to
        # poll-history dispersion. ``history`` is a sequence of states;
        # we estimate from the most-recent state's poll_history when
        # underlying is absent.
        underlyings: list[float] = []
        for state, _ in history:
            if state.underlying is not None and state.underlying > 0:
                underlyings.append(float(state.underlying))

        if len(underlyings) >= 5:
            arr = np.asarray(underlyings, dtype=float)
            logret = np.diff(np.log(arr))
            if logret.size >= 2 and logret.std() > 0:
                sigma_daily = float(logret.std(ddof=1))
                sigma_ann = sigma_daily * math.sqrt(252.0)
                return replace(
                    self,
                    sigma=max(sigma_ann, 1e-4),
                    n_obs=len(history),
                    sigma_source="underlying_returns",
                )

        polls: list[float] = []
        for state, _ in history:
            polls.extend(float(p) for p in state.poll_history)
        if len(polls) >= 5:
            arr = np.asarray(polls, dtype=float)
            sd = float(arr.std(ddof=1))
            # poll-derived σ is on a logit scale to keep things bounded.
            logits = np.log(np.clip(arr, 1e-3, 1 - 1e-3) / np.clip(1 - arr, 1e-3, 1.0))
            sd_logit = float(logits.std(ddof=1)) if logits.size >= 2 else sd
            return replace(
                self,
                sigma=max(sd_logit, 1e-4),
                n_obs=len(history),
                sigma_source="poll_dispersion",
            )

        return replace(self, n_obs=len(history))


# ---------------------------------------------------------------------------
# Model 3 — Brownian bridge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrownianBridge:
    """Brownian-bridge survival pricer for polling-style binary markets.

    Given a continuous process ``X_t`` resolving against threshold ``K``
    at fixed horizon ``T``, with current value ``x``, time-to-go
    ``τ = T − t`` and drift ``μ``::

        p = Φ((x − K + μ·τ) / (σ·√τ))

    For pure poll markets we identify ``X_t`` with the market price
    itself, so ``x = current_price`` and ``K = 0.5`` (a coin-flip
    threshold) unless an explicit threshold is provided. ``σ`` is
    estimated from ``poll_history`` increments.
    """

    sigma: float = 0.20
    drift: float = 0.0
    n_obs: int = 0
    sigma_source: str = "default"
    name: str = "brownian_bridge"

    # -- inference -------------------------------------------------------
    def fair_price(self, state: MarketState) -> PricingResult:
        tau = _years_from_days(state.time_to_resolve_days)
        diag: dict[str, float] = {
            "sigma": float(self.sigma),
            "drift": float(self.drift),
            "tau_years": tau,
        }

        x = float(state.current_price)
        # Threshold: explicit threshold if provided and bounded in [0,1],
        # otherwise default to 0.5 (a coin-flip resolution boundary).
        if state.threshold is not None and 0.0 <= float(state.threshold) <= 1.0:
            K = float(state.threshold)
        else:
            K = 0.5
        diag["K"] = K

        if tau <= 0 or self.sigma <= 0:
            p = 1.0 if x >= K else 0.0
            p = _clip_prob(p)
            diag["collapsed"] = 1.0
            return PricingResult(p, (p, p), self.name, diag)

        denom = self.sigma * math.sqrt(tau)
        z = (x - K + self.drift * tau) / denom
        p = float(norm.cdf(z))
        diag["z"] = z

        try:
            z_lo = (x - K + self.drift * tau) / (self.sigma * 1.2 * math.sqrt(tau))
            z_hi = (x - K + self.drift * tau) / (self.sigma * 0.8 * math.sqrt(tau))
            p_lo = float(norm.cdf(z_lo))
            p_hi = float(norm.cdf(z_hi))
            ci = (_clip_prob(min(p_lo, p_hi)), _clip_prob(max(p_lo, p_hi)))
        except (ValueError, ZeroDivisionError):
            ci = (_clip_prob(p), _clip_prob(p))

        return PricingResult(_clip_prob(p), ci, self.name, diag)

    # -- calibration -----------------------------------------------------
    def calibrate(
        self,
        history: list[tuple[MarketState, bool]],
    ) -> BrownianBridge:
        increments: list[float] = []
        drifts: list[float] = []
        for state, _ in history:
            polls = list(state.poll_history)
            if len(polls) >= 2:
                arr = np.asarray(polls, dtype=float)
                diffs = np.diff(arr)
                increments.extend(diffs.tolist())
                if diffs.size > 0:
                    drifts.append(float(diffs.mean()))

        if len(increments) < 3:
            return replace(self, n_obs=len(history))

        arr = np.asarray(increments, dtype=float)
        sd = float(arr.std(ddof=1))
        if not math.isfinite(sd) or sd <= 0:
            return replace(self, n_obs=len(history))

        # Annualise increments (assume daily polls) and bound away from 0.
        sigma_ann = sd * math.sqrt(252.0)
        drift = float(np.mean(drifts)) if drifts else 0.0
        return replace(
            self,
            sigma=max(sigma_ann, 1e-4),
            drift=drift,
            n_obs=len(history),
            sigma_source="poll_increments",
        )


# ---------------------------------------------------------------------------
# Model 4 — Beta-Binomial Bayes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BetaBinomialBayes:
    """Beta-Binomial posterior updated by news evidence.

    Prior is ``Beta(prior_alpha, prior_beta)``. Each call sees a signed
    ``news_evidence ∈ [-1, 1]`` and a scale ``evidence_scale`` (the
    effective sample size attributed to the news term)::

        weight = scale * |e|
        n_pos = weight if e >= 0 else 0
        n_neg = weight if e < 0  else 0
        posterior = Beta(α + n_pos, β + n_neg)
        E[p] = (α + n_pos) / (α + β + n_pos + n_neg)

    At ``e = 0`` no pseudo-counts are added, so the posterior collapses
    to the prior mean. This makes the model well-behaved when there is
    no news signal — exactly what the test suite expects.

    Calibration fits ``α`` and ``β`` to the empirical resolution rate by
    Method of Moments.
    """

    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    evidence_scale: float = 4.0
    n_obs: int = 0
    name: str = "beta_binomial_bayes"

    # -- inference -------------------------------------------------------
    def fair_price(self, state: MarketState) -> PricingResult:
        e = max(-1.0, min(1.0, float(state.news_evidence)))
        scale = max(0.0, float(self.evidence_scale))
        weight = scale * abs(e)
        n_pos = weight if e >= 0 else 0.0
        n_neg = weight if e < 0 else 0.0

        a = max(self.prior_alpha + n_pos, 1e-6)
        b = max(self.prior_beta + n_neg, 1e-6)
        mean = a / (a + b)
        var = (a * b) / (((a + b) ** 2) * (a + b + 1.0))
        sd = math.sqrt(max(var, 0.0))

        try:
            from scipy.stats import beta as beta_dist

            lo = float(beta_dist.ppf(0.025, a, b))
            hi = float(beta_dist.ppf(0.975, a, b))
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise ValueError
            ci = (_clip_prob(lo), _clip_prob(hi))
        except Exception:
            ci = (_clip_prob(mean - 1.96 * sd), _clip_prob(mean + 1.96 * sd))

        diag = {
            "alpha_post": a,
            "beta_post": b,
            "sd_post": sd,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "evidence": e,
        }
        return PricingResult(_clip_prob(mean), ci, self.name, diag)

    # -- calibration -----------------------------------------------------
    def calibrate(
        self,
        history: list[tuple[MarketState, bool]],
    ) -> BetaBinomialBayes:
        if len(history) < 2:
            return self

        outcomes = np.array([1.0 if r else 0.0 for _, r in history], dtype=float)
        m = float(outcomes.mean())
        v = float(outcomes.var(ddof=0))

        # Degenerate variance (all-0 or all-1) → use Laplace smoothing.
        if v <= 1e-12 or m <= 0.0 or m >= 1.0:
            successes = float(outcomes.sum()) + 1.0
            failures = float(outcomes.size - outcomes.sum()) + 1.0
            return replace(
                self,
                prior_alpha=successes,
                prior_beta=failures,
                n_obs=len(history),
            )

        # Method of Moments on Beta with var inflated to satisfy
        # ``v < m(1-m)``; if the observed variance is *too small* (very
        # tight Bernoulli sample), we cap κ to avoid blowing up α/β.
        max_v = m * (1.0 - m) - 1e-6
        v = min(max_v, v)
        kappa = max(m * (1.0 - m) / v - 1.0, 1.0)
        a = m * kappa
        b = (1.0 - m) * kappa
        return replace(
            self,
            prior_alpha=max(a, 1e-3),
            prior_beta=max(b, 1e-3),
            n_obs=len(history),
        )


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------


def default_pricers() -> dict[str, Pricer]:
    """Return one instance of each candidate pricer with default params."""

    return {
        "risk_neutral_logit": RiskNeutralLogit(),
        "black_scholes_digital": BlackScholesDigital(),
        "brownian_bridge": BrownianBridge(),
        "beta_binomial_bayes": BetaBinomialBayes(),
    }

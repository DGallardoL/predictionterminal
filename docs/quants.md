# Quants — math behind the model

This document derives the regression run by `POST /fit`. It is intended to be
self-contained and to be defended in a 15-minute demo.

## 1. Setup

For a stock $j$ on day $t$ and a set of prediction-market events
$\{1, \ldots, K\}$ we postulate

$$
r_{j,t} = \alpha_j + \sum_{i=1}^{K} \beta_{j,i} \cdot \Delta \text{logit}(p_{i,t}) + \varepsilon_{j,t}
$$

where

- $r_{j,t}$ is the daily **log return** of stock $j$ at $t$,
- $p_{i,t}$ is the probability implied by Polymarket market $i$ at the close
  of $t$ (mid of the YES contract, in $[0,1]$),
- $\Delta \text{logit}(p_{i,t}) = \text{logit}(p_{i,t}) - \text{logit}(p_{i,t-1})$,
- $\varepsilon_{j,t}$ is the regression residual.

## 2. Why logit?

The raw probability lives in $[0,1]$. Two problems with using $\Delta p$
directly:

1. **Heteroskedasticity at the boundary.** A move from 0.95 → 0.99 carries
   far more information than 0.50 → 0.54, even though both are $\Delta p = 0.04$.
2. **Non-linearity in information.** Posterior odds compose multiplicatively;
   probabilities don't.

The logit fixes both:

$$
\text{logit}(p) = \log\frac{p}{1-p}
$$

A $+1$ change in logit corresponds to roughly the same Bayesian information
update no matter the starting probability — that's exactly the property we
want when treating Δlogit as a regressor on the same scale across markets.

### 2.1 Clipping

To avoid $\log 0$ or $\log \infty$, probabilities are clipped to
$[\varepsilon, 1 - \varepsilon]$ before transforming, with default
$\varepsilon = 0.01$. This is exposed as a query parameter on `/fit`.

**Side effect:** if a market trades at e.g. $0.005$ and then $0.002$, both
clip to $0.01$ and Δlogit becomes 0 — that information is lost. The user
should be aware. Lowering $\varepsilon$ recovers that signal at the cost of
amplifying noise near the boundaries. ADR-0002 discusses the trade-off.

## 3. Estimator: OLS

Stack the Δlogit columns into design matrix $\mathbf{X}$ (with a leading
constant column of 1s) and the log returns into $\mathbf{y}$. The OLS
estimator is

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}.
$$

Implementation: `statsmodels.api.OLS`. We pass `cov_type='HAC'` so the
fitting routine returns the same point estimates with a different covariance
matrix.

## 4. Inference: HAC standard errors

Daily financial returns exhibit **autocorrelation and conditional
heteroskedasticity**. Plain OLS standard errors would be biased downwards.
We use the HAC (heteroskedasticity- and autocorrelation-consistent) estimator:

$$
\hat{V}_{\text{HAC}} = \hat{S}_0 + \sum_{\ell=1}^{L} w_\ell \left( \hat{S}_\ell + \hat{S}_\ell^\top \right)
$$

with Bartlett kernel weights $w_\ell = 1 - \ell / (L + 1)$.

Bandwidth $L$ is chosen using **automatic bandwidth selection**:

$$
L = \left\lfloor 4 \cdot (T/100)^{2/9} \right\rfloor
$$

floored at 1. For $T = 250$ this yields $L = 5$.

Implementation: `OLS(...).fit(cov_type='HAC', cov_kwds={'maxlags': L})`. The
chosen lag is reported in the `/fit` response as `diagnostics.hac_lag`.

## 5. Diagnostics reported

- **VIF** per factor (`statsmodels.stats.outliers_influence.variance_inflation_factor`).
  Values > 5 indicate problematic multicollinearity. We do not auto-prune in
  the POC — we report and let the user choose.
- **Durbin-Watson** on residuals. Values near 2 indicate no residual
  autocorrelation. Far from 2 ⇒ HAC is doing real work.
- **F-statistic** and joint p-value of the slope coefficients.
- **R²** and adjusted R².
- **Residual σ** (root MSE).

## 6. Attribution

For a fitted model and a target date $t^\star$:

$$
\text{contribution}_{i,t^\star} = \hat{\beta}_i \cdot \Delta\text{logit}(p_{i,t^\star})
$$

$$
\hat{r}_{t^\star} = \hat{\alpha} + \sum_i \text{contribution}_{i,t^\star}
$$

$$
e_{t^\star} = r_{t^\star} - \hat{r}_{t^\star}
$$

The endpoint returns each contribution plus the intercept, the predicted
return, and the residual. By construction
$r_{t^\star} = \hat\alpha + \sum_i \text{contribution}_{i,t^\star} + e_{t^\star}$.

## 7. Timezone alignment

Both Polymarket bars (`fidelity=1440`) and yfinance closes are normalised to
**UTC midnight** before joining. Polymarket bars are unix-second timestamps
floor-aligned to the day; yfinance closes are localised to UTC. ADR-0006
documents the choice in detail.

## 8. What this model is and is not

It **is** a clean, defensible factor regression with HAC inference.

It **is not**:

- Causal — probability changes and stock prices are simultaneously
  determined by the news that drives both.
- Forward-looking — factors are contemporaneous, so the fit cannot be used
  for prediction without a lag structure (future work).
- Identified across multiple stocks — each fit covers one ticker. Cross-
  sectional pooling is left for future work.

## References

- Newey, W. K. & West, K. D. (1987). *A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*.
  Econometrica.
- Andrews, D. W. K. (1991). *Heteroskedasticity and Autocorrelation
  Consistent Covariance Matrix Estimation*. Econometrica.
- statsmodels OLS HAC docs:
  <https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html>

## Crypto 5-minute up-down probability model

For Polymarket BTC/ETH up-down 5m and 15m markets we score the probability
that spot closes above its window-start reference using a **closed-form
GBM with a microstructure-aware drift**. Let $S_0$ be spot at window open,
$S_t$ spot now, $\Delta t$ the remaining seconds expressed in years
($\Delta t = (\text{sec remaining}) / (365 \cdot 86400)$), and $\varepsilon
= \tfrac{1}{2}\sigma_{\text{eff}}^2\,\Delta t$ the Itô correction.

$$
P(S_T \ge S_0) \;=\; \Phi(d), \qquad
d \;=\; \frac{\ln(S_t/S_0) + (\mu_{\text{eff}} - \tfrac{1}{2}\sigma_{\text{eff}}^2)\,\Delta t}{\sigma_{\text{eff}}\sqrt{\Delta t}}.
$$

### Variance blend

We blend a slow daily anchor with a responsive tick-derived estimate
using a variance-weighted mean ($\lambda = 0.4$ by default):

$$
\sigma_{\text{eff}}^2 \;=\; (1-\lambda)\,\sigma_{\text{long}}^2 + \lambda\,\sigma_{\text{short}}^2,
$$

where $\sigma_{\text{long}}$ is the Binance 30-day daily-close annualised
$\sigma$ and $\sigma_{\text{short}}$ is the tick-derived $\sigma$ from the
cryptostuff WS engine. The result is clipped to $[\,0.10,\,3.00\,]$ /yr
and floored by a window-aware adaptive minimum (e.g. $\ge 0.90$ /yr for
windows under 5 min) to stop daily-$\sigma$ from collapsing the
predictor to $0.5\%/99.5\%$ on tiny moves.

### Drift composition

The annualised drift is a sum of bounded microstructure contributions:

$$
\mu_{\text{OFI}} = \operatorname{clip}(\alpha_{\text{OFI}} \cdot \mathrm{OBI}, \pm 0.30/\text{yr}),
\qquad
\mu_{\text{whale}} = \operatorname{clip}\!\Big(\beta_{\text{whale}} \cdot \tfrac{\text{net whale notional}}{\text{total notional}}, \pm 0.15/\text{yr}\Big).
$$

When $|z_{\text{VWAP}}| > 2$ a mean-reversion overlay kicks in: the base
drift is shrunk linearly between $|z|=2$ and $|z|=4$, and a small
opposite-direction pull replaces it. At $|z| \ge 4$ the OFI drift is
fully shrunk to $0$ and the pull saturates at $\pm \tfrac{1}{2}\mu_{\text{OFI scale}}$.
The final drift is capped:

$$
\mu_{\text{eff}} \;=\; \operatorname{clip}(\mu_{\text{OFI}} + \mu_{\text{whale}} + \mu_{\text{revert}}, \;\pm 0.45/\text{yr}).
$$

### Sizing

Position sizing uses fractional Kelly capped at $f^\star \le 0.20$ to
keep a single 5-minute bet from dominating the book under model
mis-specification.

> Source of truth: module docstring and `compute_*` / `predict_up_prob`
> in `src/pfm/crypto5min/predictor.py`.

## Implied PDF from prediction-market binaries

### Motivation

A prediction-market binary contract — "will $S_T$ be above $K$ at date $T$" —
pays $1$ when the event resolves true and $0$ otherwise. Under the
risk-neutral measure its (discounted) price is, up to the numéraire,
$\mathbb{E}^{\mathbb{Q}}[\mathbf{1}\{S_T > K\}] = \mathbb{Q}(S_T > K) = S(K)$,
the **risk-neutral survival function** evaluated at strike $K$. In other
words, a binary *is* a digital option, so each quote reads off a point of the
CDF directly.

Contrast this with the **risk-neutral density from the option chain (second
derivative of the call-price curve)** on vanilla options, where
the risk-neutral density is recovered as the *second* derivative of the call
price in strike, $f(K) = e^{rT}\,\partial^2 C/\partial K^2$. Two numerical
derivatives over a sparse, noisy strike grid is a notoriously ill-posed
operation. Prediction-market binaries collapse that to **one** derivative
(or, for range buckets, *zero*), which is the central reason this feature is
cleaner than option-implied RND extraction.

### Three data shapes and their math

The engine handles three ladder geometries, each with distinct math.

**1. Terminal range buckets** (Kalshi `KXINX` "close between $a$ and $b$ on
$D$"). Each market price is *already* a probability mass,
$\pi_{[a,b]} = \mathbb{Q}(S_T \in [a,b])$. The PDF is the normalised
histogram $f \approx \pi_{[a,b]}/(b-a)$; smoothing is optional and cosmetic.
No differentiation is required.

**2. Terminal threshold ladder** (above/below at a fixed maturity). An
"above $K$" YES price gives $S(K) = \mathbb{Q}(S_T > K)$, so
$F(K) = 1 - S(K)$ and the density is the single derivative

$$
f(K) \;=\; -\frac{dS}{dK} \;=\; \frac{dF}{dK}.
$$

This is the risk-neutral density from the option chain (second derivative of
the call-price curve) specialised to digitals. "Below $K$" markets
hand back $F(K)$ directly, so the two halves of a ladder are consistency
checks on each other.

**3. Barrier / touch ladder** (Polymarket "reach $K$ by $D$"). A YES price
here is $\mathbb{Q}(M_T \ge K)$ with $M_T = \max_{t \le T} S_t$ the **running
maximum**, *not* the terminal price. Differencing the ladder recovers the law
of $M_T$, which is a different object from the terminal marginal of $S_T$.
Treating it as a terminal CDF is a category error, and §barrier below makes
the correction explicit.

### The barrier subtlety

A one-touch contract prices the survival of the running maximum,

$$
\mathrm{OT}(K) \;=\; \mathbb{Q}(M_T \ge K) \;=\; 1 - F_{M_T}(K).
$$

Differencing a touch ladder therefore yields $f_{M_T}$, the running-max
density — and this much is **model-free**. What is *not* free is the terminal
marginal: the map from touch prices to $f_{S_T}$ is **non-identifiable**
without further structure. The Skorokhod-embedding / robust-hedging
literature (Brown, Hobson & Rogers 2001; Hobson 2011) shows that infinitely
many terminal laws are consistent with a given running-max law; one needs
either additional terminal contracts or a dynamic model to pin it down.

We adopt the canonical diffusion assumption. Let the log-price be
$X_t = \sigma W_t + \nu t$ with $\nu = r - q - \tfrac12\sigma^2$ and barrier
level $a = \ln(K/S_0) \ge 0$. The **reflection principle** gives, in the
driftless-in-log case ($\nu = 0$, which is *not* the martingale case),

$$
\mathbb{Q}(M_T \ge K) \;=\; 2\,\mathbb{Q}(S_T \ge K),
$$

and in general

$$
\mathbb{Q}(M_T \ge K) \;=\; \Phi\!\Big(\tfrac{-a + \nu T}{\sigma\sqrt T}\Big)
\;+\; \Big(\tfrac{K}{S_0}\Big)^{2\nu/\sigma^2}\,
\Phi\!\Big(\tfrac{-a - \nu T}{\sigma\sqrt T}\Big).
$$

Inverting recovers the terminal survival under the same GBM,

$$
\mathbb{Q}(S_T \ge K) \;=\; \mathbb{Q}(M_T \ge K)
\;-\; \Big(\tfrac{K}{S_0}\Big)^{2\nu/\sigma^2}\,
\Phi\!\Big(\tfrac{-a - \nu T}{\sigma\sqrt T}\Big).
$$

We fit $(\nu, \sigma)$ by **nonlinear least squares** of the closed-form
running-max survival against the observed touch ladder — the exact analogue
of fitting a lognormal to a terminal threshold ladder. The fit residual is
itself a GBM diagnostic: a large residual flags that the touch prices are
incompatible with a single-regime diffusion (jumps, stochastic vol, or
microstructure).

### Smoothing and arbitrage-free construction

To turn a discrete ladder into a valid distribution we **PCHIP** (monotone
piecewise-cubic Hermite) interpolate the CDF. PCHIP preserves monotonicity,
so the interpolated $F$ is non-decreasing and the analytic derivative of the
PCHIP gives a density $f \ge 0$ everywhere — no negative-density artefacts
that plague spline-on-price RND extraction. We then renormalise so
$\int f\,dK = 1$. Clipping of input probabilities to
$[\varepsilon, 1-\varepsilon]$ is explicit and configurable (default
$\varepsilon = 0.01$, matching §2.1). Because a ladder never spans the full
support, the tails are filled by an explicit, *selectable* parametric tail
model (`lognormal`, `linear`, or `none`); a lognormal fit is overlaid as a
smooth reference curve in all cases.

### Honest-labeling policy

For barrier data the UI **headlines the running-max density** (model-free)
and shows the GBM-reflection terminal density only as a clearly-labeled,
model-dependent overlay. We never silently present a reflected-barrier curve
as "the" terminal risk-neutral density — doing so would smuggle a GBM
assumption past the user under a model-free banner.

### Implementation pointers

- `pfm/vol/implied_pdf.py` — PDF/CDF engine, PCHIP construction, reflection
  inversion, NLS fit of $(\nu, \sigma)$.
- `pfm/sources/kalshi.py::discover_index_ladder` — ladder discovery and
  bucket/threshold parsing.
- `GET /terminal/implied-pdf/{asset}` — endpoint returning the density,
  CDF, fitted parameters, and the model-dependent overlay flag.

### References

- Risk-neutral density from the option chain: recover the state-contingent
  claim prices as the second derivative of the call-price curve in strike.
- Shreve, S. E. (2004). *Stochastic Calculus for Finance II: Continuous-Time
  Models*, §3.7 (reflection principle) and §7.3 (barrier options). Springer.
- Brown, H., Hobson, D. & Rogers, L. C. G. (2001). *Robust Hedging of Barrier
  Options*. Mathematical Finance 11(3), 285–314.
- Hobson, D. (2011). *The Skorokhod Embedding Problem and Model-Independent
  Bounds for Option Prices*. In Paris-Princeton Lectures on Mathematical
  Finance 2010, Lecture Notes in Mathematics 2003, Springer.
